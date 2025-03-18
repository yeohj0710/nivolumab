import os
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
NUM_WORKERS = 8
PIN_MEMORY = True
LEARNING_RATE = 1e-3
STATIC_INPUT_DIM = 7
STATIC_HIDDEN_DIM = 32
ODE_HIDDEN_DIM = 32
MAX_DATAS = 10
MAX_CP_HOURS = 0  # 0이면 전체 CP 데이터 사용
CP_SCALE = 1


class PatientDataset(Dataset):
    def __init__(
        self,
        parquet_path=DATA_PATH,
        transform=None,
        patient_limit=MAX_DATAS,
        cp_max_hours=MAX_CP_HOURS,
    ):
        self.df = pd.read_parquet(parquet_path, engine="pyarrow")
        if patient_limit is not None:
            self.df = self.df.iloc[:patient_limit]
        self.cp_max_hours = None if cp_max_hours == 0 else cp_max_hours
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        static_feats = torch.tensor(
            row[["BW", "EGFR", "SEX", "RAAS", "BPS", "amt", "II"]]
            .astype(np.float32)
            .values
        )
        cp_seq = np.array(row["CP_sequence"], dtype=np.float32) / CP_SCALE
        if self.cp_max_hours is not None:
            cp_seq = cp_seq[: self.cp_max_hours]
        t = torch.tensor(np.arange(0, len(cp_seq), dtype=np.float32))
        cp = torch.tensor(cp_seq, dtype=torch.float32).unsqueeze(1)
        dose_amt = static_feats[5].item()
        dose_interval = static_feats[6].item()
        total_time = t[-1].item()
        inj_times = torch.tensor(
            np.arange(0, total_time + 1e-3, dose_interval, dtype=np.float32),
            dtype=torch.float32,
        )
        inj_doses = torch.full(inj_times.shape, dose_amt, dtype=torch.float32)
        sample = {
            "static_features": static_feats,
            "t": t,
            "cp": cp,
            "injection_times": inj_times,
            "injection_doses": inj_doses,
        }
        if self.transform:
            sample = self.transform(sample)
        return sample


def collate_fn(batch):
    static_feats = torch.stack([b["static_features"] for b in batch], dim=0)
    cp_padded = pad_sequence(
        [b["cp"] for b in batch], batch_first=True, padding_value=0.0
    )
    T_max = cp_padded.size(1)
    t = torch.arange(0, T_max, dtype=torch.float32)
    inj_times_padded = pad_sequence(
        [b["injection_times"] for b in batch], batch_first=True, padding_value=0.0
    )
    inj_doses_padded = pad_sequence(
        [b["injection_doses"] for b in batch], batch_first=True, padding_value=0.0
    )
    return {
        "static_features": static_feats,
        "t": t,
        "cp": cp_padded,
        "injection_times": inj_times_padded,
        "injection_doses": inj_doses_padded,
    }


class ODEF(nn.Module):
    def __init__(self, hidden_dim, static_dim, dose_dim=1):
        super(ODEF, self).__init__()
        # decay_param: 학습 가능한 감쇠 계수 (양수로 유지)
        self.decay_param = nn.Parameter(torch.tensor(0.01))
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + static_dim + dose_dim + 2, 64),
            nn.Tanh(),
            nn.Linear(64, hidden_dim),
        )

    def forward(
        self, t, state, static_embed, injection_times, injection_doses, sigma=1e-3
    ):
        # injection 효과: 매우 짧은 sigma로 sharp한 spike 생성
        dose_effect = injection_doses * torch.exp(
            -((t - injection_times) ** 2) / (2 * sigma**2)
        )
        dose_effect = dose_effect.sum(dim=1, keepdim=True)
        # 감쇠 항: state가 자연스럽게 감소하도록
        decay = -torch.abs(self.decay_param) * state
        batch_size = state.shape[0]
        t_expanded = t.expand(batch_size)
        if injection_times.size(1) >= 2:
            period = injection_times[:, 1] - injection_times[:, 0]
        else:
            period = torch.ones(batch_size, device=t.device)
        time_embed_sin = torch.sin(2 * np.pi * t_expanded / period).unsqueeze(1)
        time_embed_cos = torch.cos(2 * np.pi * t_expanded / period).unsqueeze(1)
        time_embed = torch.cat([time_embed_sin, time_embed_cos], dim=1)
        inp = torch.cat([state, static_embed, dose_effect, time_embed], dim=-1)
        out = self.net(inp) + decay
        return out


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
        # 최종 예측에 Softplus를 적용해 음수가 나오지 않도록 함
        self.readout = nn.Sequential(nn.Linear(ode_hidden_dim, 1), nn.Softplus())
        self.initial_cp = nn.Sequential(nn.Linear(static_hidden_dim, 1), nn.Softplus())

    def forward(self, t, static_features, injection_times, injection_doses):
        static_embed = self.static_encoder(static_features)
        # 초기 CP는 실제 데이터 t=0의 CP와 맞춰지도록 학습됨
        init_cp = self.initial_cp(static_embed)
        h0 = self.initial_state_layer(static_embed)

        def func(t_val, h):
            return self.ode_func(
                t_val, h, static_embed, injection_times, injection_doses
            )

        h_t = odeint(func, h0, t, method="dopri5", rtol=1e-6, atol=1e-7).transpose(0, 1)
        # 최종 CP 예측: ODE 통합 결과에 초기 CP를 더함
        cp_pred = self.readout(h_t) + init_cp.unsqueeze(1)
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
    pd.DataFrame({"time": t.cpu().numpy(), "CP_pred": cp_pred}).to_csv(
        os.path.join(BASE_DIR, "predicted_cp.csv"), index=False
    )
    plt.figure()
    plt.plot(t.cpu().numpy(), cp_pred, marker=".")
    plt.xlabel("Time")
    plt.ylabel("CP")
    plt.title("Predicted CP over Time")
    plt.savefig(os.path.join(BASE_DIR, "predicted_cp.png"))
    plt.close()
    print("Inference 완료")
    return cp_pred


if __name__ == "__main__":
    while True:
        mode_input = input("모드를 선택하세요. (학습: t, 예측: i): ").strip().lower()
        if mode_input in ["t", "train"]:
            mode_input = "train"
            break
        elif mode_input in ["i", "infer"]:
            mode_input = "infer"
            break
        else:
            print("잘못된 입력입니다. 다시 입력해 주세요.")

    class Args:
        pass

    args = Args()
    args.mode = mode_input
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralODEModel(
        static_input_dim=STATIC_INPUT_DIM,
        static_hidden_dim=STATIC_HIDDEN_DIM,
        ode_hidden_dim=ODE_HIDDEN_DIM,
    ).to(device)
    if args.mode == "infer":
        if os.path.exists(CHECKPOINT_PATH):
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
    start_epoch = 0
    loss_history = []
    if args.mode == "train":
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=3
        )
        if os.path.exists(CHECKPOINT_PATH):
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            for param_group in optimizer.param_groups:
                param_group["lr"] = LEARNING_RATE
            start_epoch = checkpoint["epoch"] + 1
            loss_history = checkpoint.get("loss_history", [])
            print(f"현재까지 학습한 epoch 수: {start_epoch}")
    if torch.cuda.is_available():
        print(f"현재 사용 중인 GPU: {torch.cuda.get_device_name(0)}")
    if args.mode == "train":
        dataset = PatientDataset(
            parquet_path=DATA_PATH, patient_limit=MAX_DATAS, cp_max_hours=MAX_CP_HOURS
        )
        dataloader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
        )
        num_epochs = int(
            input(
                f"몇 epoch까지 학습할까요? (현재까지 학습된 epoch 수: {start_epoch}): "
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
                        f"Epoch {epoch}, Batch {i}/{len(dataloader)}: Loss = {loss.item():.6f}",
                        flush=True,
                    )
            epoch_loss /= len(dataloader)
            loss_history.append(epoch_loss)
            print(f"Epoch {epoch}: Average Loss = {epoch_loss:.6f}", flush=True)
            scheduler.step(epoch_loss)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss_history": loss_history,
                },
                CHECKPOINT_PATH,
            )
            pd.DataFrame(
                {"epoch": list(range(len(loss_history))), "loss": loss_history}
            ).to_csv(os.path.join(BASE_DIR, "training_loss.csv"), index=False)
            plt.figure()
            plt.plot(loss_history, marker="o")
            plt.yscale("log")
            plt.xlabel("Epoch")
            plt.ylabel("Loss (log scale)")
            plt.title("Training Loss History")
            plt.savefig(os.path.join(BASE_DIR, "training_loss.png"))
            plt.close()
        pd.DataFrame(
            {"epoch": list(range(len(loss_history))), "loss": loss_history}
        ).to_csv(os.path.join(BASE_DIR, "training_loss.csv"), index=False)
        print("학습 완료 및 loss 기록 저장됨.")
    elif args.mode == "infer":
        static_input = input(
            "BW, EGFR, SEX, RAAS, BPS, amt, II 값을 순서대로 입력하세요: "
        )
        static_features = [float(x.strip()) for x in static_input.split()]
        total_time = float(input("모델이 예측할 총 시간을 입력하세요. (hours): "))
        dose_amt = static_features[5]
        dose_interval = static_features[6]
        inj_times = np.arange(0, total_time + 1e-3, dose_interval, dtype=np.float32)
        inj_doses = np.full(inj_times.shape, dose_amt, dtype=np.float32)
        run_inference(model, static_features, inj_times, inj_doses, total_time, device)
