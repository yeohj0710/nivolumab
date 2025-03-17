import os
import sys
import random
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
import torch.multiprocessing as mp
import torch.distributed as dist

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "processed_data.parquet")
CHECKPOINT_PATH = os.path.join(BASE_DIR, "checkpoint.pth")

BATCH_SIZE = 1024
NUM_WORKERS = 8

PIN_MEMORY = True
LEARNING_RATE = 3e-3
STATIC_INPUT_DIM = 7
STATIC_HIDDEN_DIM = 16
ODE_HIDDEN_DIM = 16

NUM_TRAINING_SAMPLES = 10000
CP_SEQUENCE_LENGTH_LIMIT = 0
MIN_CP_LENGTH = 2000

CONCENTRATION_SCALE = 1.0
SCALING_FACTOR = 1.0

RTOL = 1e-5
ATOL = 1e-6


def get_valid_input(
    prompt, valid_fn=lambda s: s.strip() != "", err_msg="값을 입력해주세요."
):
    while True:
        try:
            s = input(prompt)
        except EOFError:
            s = ""
        if valid_fn(s):
            return s.strip()
        else:
            print(err_msg)


class PKPDataset(Dataset):
    def __init__(
        self,
        parquet_path=DATA_PATH,
        transform=None,
        num_samples=NUM_TRAINING_SAMPLES,
        cp_length_limit=CP_SEQUENCE_LENGTH_LIMIT,
    ):
        self.df = pd.read_parquet(parquet_path, engine="pyarrow")
        if num_samples is not None:
            self.df = self.df.iloc[:num_samples]
        if cp_length_limit == 0:
            self.df = self.df[self.df["CP_sequence"].apply(len) >= MIN_CP_LENGTH]
            self.cp_length = MIN_CP_LENGTH
            if (not dist.is_initialized()) or (dist.get_rank() == 0):
                print(f"최소 CP 시퀀스 길이를 만족하는 데이터 개수: {len(self.df)}")
        elif cp_length_limit is not None:
            self.cp_length = cp_length_limit
        else:
            self.cp_length = None
        if (not dist.is_initialized()) or (dist.get_rank() == 0):
            print(f"총 {self.cp_length} 시간까지의 데이터로 학습을 진행합니다.")
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
        concentration_sequence = (
            np.array(row["CP_sequence"], dtype=np.float32) / CONCENTRATION_SCALE
        )
        if self.cp_length is not None:
            concentration_sequence = concentration_sequence[: self.cp_length]
        t_values = np.arange(0, len(concentration_sequence), dtype=np.float32)
        t = torch.tensor(t_values)
        concentration = torch.tensor(
            concentration_sequence, dtype=torch.float32
        ).unsqueeze(1)
        sample = {
            "static_features": static_features,
            "t": t,
            "concentration": concentration,
        }
        if self.transform:
            sample = self.transform(sample)
        return sample


def collate_fn(batch):
    from torch.nn.utils.rnn import pad_sequence

    static_features = torch.stack([b["static_features"] for b in batch], dim=0)
    concentration_padded = pad_sequence(
        [b["concentration"] for b in batch], batch_first=True, padding_value=0.0
    )
    T_max = concentration_padded.size(1)
    t = torch.arange(0, T_max, dtype=torch.float32)
    return {
        "static_features": static_features,
        "t": t,
        "concentration": concentration_padded,
    }


class LSTMModel(nn.Module):
    def __init__(
        self,
        static_input_dim=STATIC_INPUT_DIM,
        static_hidden_dim=STATIC_HIDDEN_DIM,
        lstm_hidden_dim=ODE_HIDDEN_DIM,
        num_layers=1,
    ):
        super(LSTMModel, self).__init__()
        self.static_encoder = nn.Sequential(
            nn.Linear(static_input_dim, static_hidden_dim), nn.ReLU()
        )
        self.initial_state_layer = nn.Linear(static_hidden_dim, lstm_hidden_dim)
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=lstm_hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.readout = nn.Sequential(nn.Linear(lstm_hidden_dim, 1), nn.Softplus())

    def forward(
        self,
        t,
        static_features,
        cp_seq=None,
        teacher_forcing_ratio=0.5,
        target_length=None,
    ):
        batch = static_features.size(0)
        if cp_seq is not None:
            T_seq = cp_seq.size(1)
        elif target_length is not None:
            T_seq = target_length
        else:
            raise ValueError("cp_seq 또는 target_length를 제공해야 합니다.")
        static_embed = self.static_encoder(static_features)
        h0 = self.initial_state_layer(static_embed).unsqueeze(0)
        c0 = torch.zeros_like(h0)
        outputs = []
        input_token = torch.zeros(batch, 1, 1, device=static_features.device)
        for t_idx in range(T_seq):
            out, (h0, c0) = self.lstm(input_token, (h0, c0))
            pred = self.readout(out)
            outputs.append(pred)
            if cp_seq is not None and t_idx < T_seq - 1:
                if random.random() < teacher_forcing_ratio:

                    input_token = cp_seq[:, t_idx].unsqueeze(1)
                else:
                    input_token = pred
            else:
                input_token = pred
        outputs = torch.cat(outputs, dim=1)
        return outputs


def run_inference(model, static_features, total_time, device):
    T = int(total_time) + 1
    t = torch.linspace(0, total_time, steps=T).to(device)
    static_features = (
        torch.tensor(static_features, dtype=torch.float32).unsqueeze(0).to(device)
    )
    model.eval()
    with torch.no_grad():
        predicted_concentration = model(
            t, static_features, cp_seq=None, teacher_forcing_ratio=0.0, target_length=T
        )
    predicted_concentration = predicted_concentration.squeeze().cpu().numpy()
    pred_df = pd.DataFrame(
        {"time": t.cpu().numpy(), "predicted_concentration": predicted_concentration}
    )
    pred_df.to_csv(os.path.join(BASE_DIR, "predicted_concentration.csv"), index=False)
    plt.figure()
    plt.plot(t.cpu().numpy(), predicted_concentration, marker=".")
    plt.xlabel("Time")
    plt.ylabel("Concentration")
    plt.title("Predicted Concentration over Time")
    plt.savefig(os.path.join(BASE_DIR, "predicted_concentration.png"))
    plt.close()
    print(
        "Inference 완료: predicted_concentration.csv, predicted_concentration.png 생성됨."
    )
    return predicted_concentration


def main_worker(gpu: int, world_size: int, args: object):
    if world_size > 1:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"
        dist.init_process_group("nccl", rank=gpu, world_size=world_size)
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    if gpu == 0:
        print(f"선택한 모드: {args.mode}")

    model = LSTMModel(
        static_input_dim=STATIC_INPUT_DIM,
        static_hidden_dim=STATIC_HIDDEN_DIM,
        lstm_hidden_dim=ODE_HIDDEN_DIM,
        num_layers=1,
    ).to(device)

    checkpoint = None
    start_epoch = 0
    loss_history = []
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        loss_history = checkpoint.get("loss_history", [])
        if gpu == 0:
            print(f"체크포인트 불러옴: epoch {start_epoch}부터 학습 재개합니다.")

    if world_size > 1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    if args.mode == "train":
        from torch.utils.data.distributed import DistributedSampler

        dataset = PKPDataset(
            parquet_path=DATA_PATH,
            num_samples=NUM_TRAINING_SAMPLES,
            cp_length_limit=CP_SEQUENCE_LENGTH_LIMIT,
        )
        sampler = (
            DistributedSampler(dataset, num_replicas=world_size, rank=gpu, shuffle=True)
            if world_size > 1
            else None
        )
        dataloader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=(sampler is None),
            sampler=sampler,
            collate_fn=collate_fn,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
        )
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        if checkpoint is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        criterion = nn.MSELoss(reduction="none")

        for epoch in range(start_epoch, args.num_epochs):
            if world_size > 1 and sampler is not None:
                sampler.set_epoch(epoch)
            model.train()
            epoch_loss = 0.0
            iterator = tqdm(
                dataloader,
                desc=f"Epoch {epoch}/{args.num_epochs}",
                ncols=80,
                disable=(gpu != 0),
            )
            for i, batch in enumerate(iterator, start=1):
                static_features = batch["static_features"].to(device)
                t_seq = batch["t"].to(device).squeeze()
                concentration_target = batch["concentration"].to(device)
                optimizer.zero_grad()
                predicted_concentration = model(
                    t_seq,
                    static_features,
                    cp_seq=concentration_target,
                    teacher_forcing_ratio=0.5,
                )
                loss_all = criterion(predicted_concentration, concentration_target)
                mask = (concentration_target != 0).float()
                loss = (loss_all * mask).sum() / mask.sum()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                if i % 100 == 0 and gpu == 0:
                    print(
                        f"Epoch {epoch}, Batch {i}/{len(dataloader)}: Current batch loss = {loss.item():.6f}",
                        flush=True,
                    )
            epoch_loss /= len(dataloader)
            if gpu == 0:
                loss_history.append(epoch_loss)
                print(f"Epoch {epoch}: Average Loss = {epoch_loss:.6f}", flush=True)
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": (
                            model.module.state_dict()
                            if world_size > 1
                            else model.state_dict()
                        ),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss_history": loss_history,
                    },
                    CHECKPOINT_PATH,
                )
                loss_df = pd.DataFrame(
                    {
                        "epoch": list(range(1, len(loss_history) + 1)),
                        "loss": loss_history,
                    }
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
        if gpu == 0:
            loss_df = pd.DataFrame(
                {"epoch": list(range(1, len(loss_history) + 1)), "loss": loss_history}
            )
            loss_df.to_csv(os.path.join(BASE_DIR, "training_loss.csv"), index=False)
            print("학습 완료 및 loss 기록 저장됨.")
    elif args.mode == "infer":
        if gpu == 0:
            static_input = get_valid_input(
                "BW, EGFR, SEX, RAAS, BPS, amt, II 값을 순서대로 입력하세요. (공백으로 구분, 예: 68.28 66.37 1 1 1 244 482): ",
                valid_fn=lambda s: s.strip() != "",
                err_msg="값을 입력해주세요.",
            )
            try:
                static_features = [float(x) for x in static_input.split()]
            except Exception as e:
                print("입력 오류:", e)
                return
            total_time_str = get_valid_input(
                "모델이 예측할 총 시간을 정수로 입력하세요: ",
                valid_fn=lambda s: s.strip() != "",
                err_msg="시간 길이를 입력해주세요.",
            )
            try:
                total_time = float(total_time_str)
            except Exception as e:
                print("시간 입력 오류:", e)
                return
            run_inference(
                model.module if world_size > 1 else model,
                static_features,
                total_time,
                device,
            )
    if world_size > 1:
        dist.destroy_process_group()


class Args:
    pass


if __name__ == "__main__":
    mode_input = get_valid_input(
        "모드를 선택하세요 (t: train, i: infer): ",
        valid_fn=lambda s: s.strip() in ["t", "i"],
        err_msg="t 또는 i를 입력해주세요.",
    ).lower()
    if mode_input == "t":
        mode_input = "train"
    elif mode_input == "i":
        mode_input = "infer"
    else:
        print("잘못된 입력입니다.")
        sys.exit(0)
    args = Args()
    args.mode = mode_input
    if args.mode == "train":
        start_epoch = 0
        if os.path.exists(CHECKPOINT_PATH):
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=torch.device("cpu"))
            start_epoch = checkpoint["epoch"] + 1
        num_epochs_str = get_valid_input(
            f"몇 epoch까지 학습할까요? (현재까지 학습된 epoch 수: {start_epoch}, 각 epoch마다 자동 저장됩니다.): ",
            valid_fn=lambda s: s.strip() != "",
            err_msg="epoch 수를 입력해주세요.",
        )
        try:
            args.num_epochs = int(num_epochs_str)
        except Exception as e:
            print("정수를 입력해주세요. 기본값 10000으로 설정합니다.", e)
            args.num_epochs = 10000
        if torch.cuda.device_count() > 1:
            world_size = torch.cuda.device_count()
            mp.spawn(main_worker, args=(world_size, args), nprocs=world_size, join=True)
        else:
            main_worker(0, 1, args)
    elif args.mode == "infer":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = LSTMModel(
            static_input_dim=STATIC_INPUT_DIM,
            static_hidden_dim=STATIC_HIDDEN_DIM,
            lstm_hidden_dim=ODE_HIDDEN_DIM,
            num_layers=1,
        ).to(device)
        if os.path.exists(CHECKPOINT_PATH):
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
        static_input = get_valid_input(
            "BW, EGFR, SEX, RAAS, BPS, amt, II 값을 순서대로 입력하세요. (공백으로 구분, 예: 68.28 66.37 1 1 1 244 482): ",
            valid_fn=lambda s: s.strip() != "",
            err_msg="값을 입력해주세요.",
        )
        try:
            static_features = [float(x) for x in static_input.split()]
        except Exception as e:
            print("입력 오류:", e)
            sys.exit(1)
        total_time_str = get_valid_input(
            "모델이 예측할 총 시간을 정수로 입력하세요: ",
            valid_fn=lambda s: s.strip() != "",
            err_msg="시간 길이를 입력해주세요.",
        )
        try:
            total_time = float(total_time_str)
        except Exception as e:
            print("시간 입력 오류:", e)
            sys.exit(1)
        run_inference(model, static_features, total_time, device)
