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
import torch.multiprocessing as mp
import torch.distributed as dist

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "processed_data.parquet")
CHECKPOINT_PATH = os.path.join(BASE_DIR, "checkpoint.pth")

BATCH_SIZE = 128
NUM_WORKERS = 8

PIN_MEMORY = True
LEARNING_RATE = 1e-5
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


class PKPDataset(Dataset):
    """데이터셋을 초기화하고 parquet 파일에서 데이터를 로드 및 필터링"""

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
        """데이터셋의 전체 샘플 개수를 반환한다."""
        return len(self.df)

    def __getitem__(self, idx):
        """주어진 인덱스의 데이터를 로드하여 필요한 텐서들로 구성된 샘플을 반환"""
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
        amt = static_features[5].item()
        II = static_features[6].item()
        total_time = t[-1].item()
        injection_times = np.arange(0, total_time + 1e-3, II, dtype=np.float32)
        injection_times = torch.tensor(injection_times, dtype=torch.float32)
        injection_doses = torch.full(injection_times.shape, amt, dtype=torch.float32)
        sample = {
            "static_features": static_features,
            "t": t,
            "concentration": concentration,
            "injection_times": injection_times,
            "injection_doses": injection_doses,
        }
        if self.transform:
            sample = self.transform(sample)
        return sample


def collate_fn(batch):
    """배치 내 개별 샘플들을 패딩 및 스택하여 하나의 배치로 결합"""
    static_features = torch.stack([b["static_features"] for b in batch], dim=0)
    concentration_padded = pad_sequence(
        [b["concentration"] for b in batch], batch_first=True, padding_value=0.0
    )
    T_max = concentration_padded.size(1)
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
        "concentration": concentration_padded,
        "injection_times": injection_times_padded,
        "injection_doses": injection_doses_padded,
    }


class ODEFunction(nn.Module):
    """ODE 함수 네트워크를 초기화"""

    def __init__(self, hidden_dim, static_dim, dose_dim=1):
        super(ODEFunction, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + static_dim + dose_dim, 64),
            nn.Tanh(),
            nn.Linear(64, hidden_dim),
        )

    def forward(
        self, t, state, static_embed, injection_times, injection_doses, sigma=0.1
    ):
        """현재 상태, 정적 임베딩, 주입 정보를 바탕으로 ODE의 미분값을 계산"""
        dose_effect = SCALING_FACTOR * (
            injection_doses * torch.exp(-((t - injection_times) ** 2) / (2 * sigma**2))
        ).sum(dim=1, keepdim=True)
        dose_effect = torch.clamp(dose_effect, min=0.0, max=1e2)
        inp = torch.cat([state, static_embed, dose_effect], dim=-1)
        return self.net(inp)


class NeuralODEModel(nn.Module):
    """정적 입력을 인코딩하고 Neural ODE를 구성하는 모델을 초기화"""

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
        self.ode_func = ODEFunction(ode_hidden_dim, static_hidden_dim, dose_dim=1)
        self.readout = nn.Sequential(nn.Linear(ode_hidden_dim, 1), nn.Softplus())

    def forward(self, t, static_features, injection_times, injection_doses):
        """주어진 시간 및 입력 데이터를 바탕으로 약물 농도를 예측"""
        static_embed = self.static_encoder(static_features)
        h0 = self.initial_state_layer(static_embed)

        def func(t_val, h):
            return self.ode_func(
                t_val, h, static_embed, injection_times, injection_doses
            )

        h_t = odeint(func, h0, t, method="dopri5", rtol=RTOL, atol=ATOL)
        h_t = h_t.transpose(0, 1)
        predicted_concentration = self.readout(h_t)
        return predicted_concentration


def run_inference(
    model, static_features, injection_times, injection_doses, total_time, device
):
    """입력 데이터를 이용하여 모델의 추론을 실행하고 결과를 파일과 플롯으로 저장"""
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
        predicted_concentration = model(
            t, static_features, injection_times, injection_doses
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
    """주어진 GPU와 분산 설정을 기반으로 학습 또는 추론 작업을 실행"""
    if world_size > 1:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"
        dist.init_process_group("nccl", rank=gpu, world_size=world_size)
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    if gpu == 0:
        print(f"선택한 모드: {args.mode}")

    model = NeuralODEModel(
        static_input_dim=STATIC_INPUT_DIM,
        static_hidden_dim=STATIC_HIDDEN_DIM,
        ode_hidden_dim=ODE_HIDDEN_DIM,
    ).to(device)

    checkpoint = None
    start_epoch = 0
    loss_history = []
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        loss_history = checkpoint.get("loss_history", [])
        if gpu == 0:
            print(f"체크포인트 불러옴: epoch {start_epoch}부터 이어서 학습")

    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    if args.mode == "train":
        dataset = PKPDataset(
            parquet_path=DATA_PATH,
            num_samples=NUM_TRAINING_SAMPLES,
            cp_length_limit=CP_SEQUENCE_LENGTH_LIMIT,
        )
        sampler = (
            torch.utils.data.distributed.DistributedSampler(
                dataset, num_replicas=world_size, rank=gpu
            )
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
                t = batch["t"].to(device).squeeze()
                concentration_target = batch["concentration"].to(device)
                injection_times = batch["injection_times"].to(device)
                injection_doses = batch["injection_doses"].to(device)
                optimizer.zero_grad()
                predicted_concentration = model(
                    t, static_features, injection_times, injection_doses
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
        if gpu == 0:
            loss_df = pd.DataFrame(
                {"epoch": list(range(len(loss_history))), "loss": loss_history}
            )
            loss_df.to_csv(os.path.join(BASE_DIR, "training_loss.csv"), index=False)
            print("학습 완료 및 loss 기록 저장됨.")
    elif args.mode == "infer":
        if gpu == 0:
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
                model.module if world_size > 1 else model,
                static_features,
                injection_times,
                injection_doses,
                total_time,
                device,
            )
    if world_size > 1:
        dist.destroy_process_group()


class Args:
    """명령행 인자들을 저장하기 위한 빈 클래스"""

    pass


if __name__ == "__main__":
    mode_input = input("모드를 선택하세요 (t: train, i: infer): ").strip().lower()
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
        args.num_epochs = int(
            input(
                f"몇 epoch까지 학습할까요? (현재까지 학습된 epoch 수: {start_epoch}, 학습 중간에 코드가 멈추더라도, 각 epoch가 끝날 때마다 자동으로 저장됩니다.): "
            )
            or 10000
        )
        if torch.cuda.device_count() > 1:
            world_size = torch.cuda.device_count()
            mp.spawn(main_worker, args=(world_size, args), nprocs=world_size)
        else:
            main_worker(0, 1, args)
    elif args.mode == "infer":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = NeuralODEModel(
            static_input_dim=STATIC_INPUT_DIM,
            static_hidden_dim=STATIC_HIDDEN_DIM,
            ode_hidden_dim=ODE_HIDDEN_DIM,
        ).to(device)
        if os.path.exists(CHECKPOINT_PATH):
            checkpoint = torch.load(
                CHECKPOINT_PATH, map_location=device, weights_only=True
            )
            model.load_state_dict(checkpoint["model_state_dict"])
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
