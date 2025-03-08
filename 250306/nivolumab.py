import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tqdm import tqdm
from torchdiffeq import odeint
from torch.nn.utils.rnn import pad_sequence

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "processed_data.parquet")
CHECKPOINT_PATH = os.path.join(BASE_DIR, "checkpoint.pth")
BATCH_SIZE = 1024
NUM_WORKERS = 16
PIN_MEMORY = True
LEARNING_RATE = 1e-3
STATIC_INPUT_DIM = 7
STATIC_HIDDEN_DIM = 16
ODE_HIDDEN_DIM = 16
MAX_FILES = 10
MAX_ROWS = 0  # 0으로 설정 시 전체 CP 데이터 사용
CP_SCALE = 1


class PKPDataset(Dataset):
    def __init__(
        self,
        parquet_path=DATA_PATH,
        transform=None,
        max_files=MAX_FILES,
        max_rows=MAX_ROWS,
    ):
        self.df = pd.read_parquet(parquet_path, engine="pyarrow")
        if max_files is not None:
            self.df = self.df.iloc[:max_files]
        if max_rows == 0:
            self.max_rows = self.df["CP_sequence"].apply(len).min()
        elif max_rows is not None:
            self.max_rows = max_rows
        else:
            self.max_rows = None

        print(f"총 {self.max_rows}시간까지의 데이터로 학습을 진행합니다.")

        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        static_features = (
            row[["BW", "EGFR", "SEX", "RAAS", "BPS", "amt", "II"]]
            .astype(np.float32)
            .values
        )
        static_features = torch.tensor(static_features)
        cp_sequence = np.array(row["CP_sequence"], dtype=np.float32) / CP_SCALE
        if self.max_rows is not None:
            cp_sequence = cp_sequence[: self.max_rows]
        t_values = np.arange(0, len(cp_sequence), dtype=np.float32)
        t = torch.tensor(t_values)
        cp = torch.tensor(cp_sequence, dtype=torch.float32).unsqueeze(1)
        amt = static_features[5].item()
        II = static_features[6].item()
        total_time = t[-1].item()
        injection_times = np.arange(0, total_time + 1e-3, II, dtype=np.float32)
        injection_times = torch.tensor(injection_times, dtype=torch.float32)
        injection_doses = torch.full(injection_times.shape, amt, dtype=torch.float32)
        sample = {
            "static_features": static_features,
            "t": t,
            "cp": cp,
            "injection_times": injection_times,
            "injection_doses": injection_doses,
        }
        if self.transform:
            sample = self.transform(sample)
        return sample


def collate_fn(batch):
    static_features = torch.stack([b["static_features"] for b in batch], dim=0)
    cp_padded = pad_sequence(
        [b["cp"] for b in batch], batch_first=True, padding_value=0.0
    )
    T_max = cp_padded.size(1)
    t = torch.arange(0, T_max, dtype=torch.float32)
    injection_times_padded = pad_sequence(
        [b["injection_times"] for b in batch], batch_first=True, padding_value=0.0
    )
    injection_doses_padded = pad_sequence(
        [b["injection_doses"] for b in batch], batch_first=True, padding_value=0.0
    )
    return {
        "static_features": static_features,
        "t": t,
        "cp": cp_padded,
        "injection_times": injection_times_padded,
        "injection_doses": injection_doses_padded,
    }


class ODEF(nn.Module):
    def __init__(self, hidden_dim, static_dim, dose_dim=1):
        super(ODEF, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + static_dim + dose_dim, 64),
            nn.Tanh(),
            nn.Linear(64, hidden_dim),
        )

    def forward(
        self, t, state, static_embed, injection_times, injection_doses, sigma=1.0
    ):
        dose = (
            injection_doses * torch.exp(-((t - injection_times) ** 2) / (2 * sigma**2))
        ).sum(dim=1, keepdim=True)
        dose = torch.clamp(dose, min=0.0, max=1e2)
        inp = torch.cat([state, static_embed, dose], dim=-1)
        return self.net(inp)


class NeuralODEModel(nn.Module):
    def __init__(
        self,
        static_input_dim=STATIC_INPUT_DIM,
        static_hidden_dim=STATIC_HIDDEN_DIM,
        ode_hidden_dim=ODE_HIDDEN_DIM,
    ):
        super(NeuralODEModel, self).__init__()
        self.static_encoder = nn.Sequential(
            nn.Linear(static_input_dim, static_hidden_dim), nn.ReLU()
        )
        self.initial_state_layer = nn.Linear(static_hidden_dim, ode_hidden_dim)
        self.ode_func = ODEF(ode_hidden_dim, static_hidden_dim, dose_dim=1)
        self.readout = nn.Sequential(nn.Linear(ode_hidden_dim, 1), nn.ReLU())

    def forward(self, t, static_features, injection_times, injection_doses):
        static_embed = self.static_encoder(static_features)
        h0 = self.initial_state_layer(static_embed)

        def func(t_val, h):
            return self.ode_func(
                t_val, h, static_embed, injection_times, injection_doses
            )

        h_t = odeint(func, h0, t, method="dopri5", rtol=1e-5, atol=1e-6)
        h_t = h_t.transpose(0, 1)
        cp_pred = self.readout(h_t)
        return cp_pred


def run_inference(
    model, static_features, injection_times, injection_doses, total_time, device
):
    T = int(total_time) + 1
    t = torch.linspace(0, total_time, steps=T).to(device)
    static_features = (
        torch.tensor(static_features, dtype=torch.float32).unsqueeze(0).to(device)
    )
    injection_times = (
        torch.tensor(injection_times, dtype=torch.float32).unsqueeze(0).to(device)
    )
    injection_doses = (
        torch.tensor(injection_doses, dtype=torch.float32).unsqueeze(0).to(device)
    )
    model.eval()
    with torch.no_grad():
        cp_pred = model(t, static_features, injection_times, injection_doses)
    cp_pred = cp_pred.squeeze().cpu().numpy()
    pred_df = pd.DataFrame({"time": t.cpu().numpy(), "CP_pred": cp_pred})
    pred_df.to_csv(os.path.join(BASE_DIR, "predicted_cp.csv"), index=False)
    plt.figure()
    plt.plot(t.cpu().numpy(), cp_pred, marker=".")
    plt.xlabel("Time")
    plt.ylabel("CP")
    plt.title("Predicted CP over Time")
    plt.savefig(os.path.join(BASE_DIR, "predicted_cp.png"))
    plt.close()
    print("Inference 완료: predicted_cp.csv, predicted_cp.png 생성됨.")
    return cp_pred


if __name__ == "__main__":
    mode_input = input("모드를 선택하세요 (t: train, i: infer): ").strip().lower()
    if mode_input == "t":
        mode_input = "train"
    elif mode_input == "i":
        mode_input = "infer"
    else:
        print("잘못된 입력입니다.")
        sys.exit(0)

    class Args:
        pass

    args = Args()
    args.mode = mode_input
    print(f"선택한 모드: {args.mode}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralODEModel(
        static_input_dim=STATIC_INPUT_DIM,
        static_hidden_dim=STATIC_HIDDEN_DIM,
        ode_hidden_dim=ODE_HIDDEN_DIM,
    ).to(device)
    checkpoint_path = CHECKPOINT_PATH
    start_epoch = 0
    loss_history = []

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        loss_history = checkpoint.get("loss_history", [])
        print(f"체크포인트 불러옴: epoch {start_epoch}부터 이어서 학습합니다.")

    if torch.cuda.is_available():
        print(f"현재 사용 중인 GPU: {torch.cuda.get_device_name(0)}")
        print(f"사용 중인 GPU 메모리: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"예약된 GPU 메모리: {torch.cuda.memory_reserved() / 1024**2:.2f} MB\n")

    if args.mode == "train":
        dataset = PKPDataset(
            parquet_path=DATA_PATH, max_files=MAX_FILES, max_rows=MAX_ROWS
        )
        dataloader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
        )
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        num_epochs = int(
            input(
                f"몇 epoch까지 학습할까요? (현재까지 학습된 epoch 수: {start_epoch}, 학습 중간에 코드가 멈추더라도, 각 epoch가 끝날 때마다 자동으로 저장됩니다.): "
            )
            or 100
        )
        criterion = nn.MSELoss()
        for epoch in range(start_epoch, num_epochs):
            model.train()
            epoch_loss = 0.0
            iterator = tqdm(dataloader, desc=f"Epoch {epoch}/{num_epochs}", ncols=80)

            for i, batch in enumerate(iterator, start=1):
                static_features = batch["static_features"].to(device)
                t = batch["t"].to(device).squeeze()
                cp_target = batch["cp"].to(device)
                injection_times = batch["injection_times"].to(device)
                injection_doses = batch["injection_doses"].to(device)
                optimizer.zero_grad()
                cp_pred = model(t, static_features, injection_times, injection_doses)
                loss = criterion(cp_pred, cp_target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                if i % 100 == 0:
                    print(
                        f"Epoch {epoch}, Batch {i}/{len(dataloader)}: Current batch loss = {loss.item():.6f}",
                        flush=True,
                    )
            epoch_loss /= len(dataloader)
            loss_history.append(epoch_loss)
            print(f"Epoch {epoch}: Average Loss = {epoch_loss:.6f}", flush=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss_history": loss_history,
                },
                checkpoint_path,
            )
            loss_df = pd.DataFrame(
                {"epoch": list(range(len(loss_history))), "loss": loss_history}
            )
            loss_df.to_csv(os.path.join(BASE_DIR, "training_loss.csv"), index=False)
            plt.figure()
            plt.plot(loss_history, marker="o")
            plt.yscale("log")
            plt.xlabel("Epoch")
            plt.ylabel("Loss (log scale)")
            plt.title("Training Loss History")
            plt.savefig(os.path.join(BASE_DIR, "training_loss.png"))
            plt.close()
        loss_df = pd.DataFrame(
            {"epoch": list(range(len(loss_history))), "loss": loss_history}
        )
        loss_df.to_csv(os.path.join(BASE_DIR, "training_loss.csv"), index=False)
        print("학습 완료 및 loss 기록 저장됨.")

    elif args.mode == "infer":
        static_input = input(
            "BW, EGFR, SEX, RAAS, BPS, amt, II 값을 순서대로 입력하세요. (공백으로 구분, 예: 68.28 66.37 1 1 1 244 482): "
        )
        static_features = [float(x.strip()) for x in static_input.split()]
        total_time = float(input("모델이 예측할 총 시간을 입력하세요. (시간): "))
        amt = static_features[5]
        II = static_features[6]
        injection_times = np.arange(0, total_time + 1e-3, II, dtype=np.float32)
        injection_doses = np.full(injection_times.shape, amt, dtype=np.float32)
        run_inference(
            model, static_features, injection_times, injection_doses, total_time, device
        )
