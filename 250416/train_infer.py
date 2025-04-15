import os
import sys
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

##############################################
# 기본 설정 및 경로
##############################################
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARQUET_PATH = os.path.join(BASE_DIR, "../processed_data.parquet")
BATCH_SIZE = 128
NUM_EPOCHS = 100000
LEARNING_RATE = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# loss를 텍스트 파일에 저장할 경로
loss_log_path = os.path.join(BASE_DIR, "loss_log.txt")


##############################################
# 데이터 전처리
##############################################
def expand_row_to_timeseries(row):
    """
    각 환자별로 CP 시퀀스를 받아오고,
    정적 특성( BW, EGFR, SEX, RAAS, BPS, amt, II )은 각 시점마다 동일하게 반복.
    """
    cp_seq = np.array(row["CP_sequence"])
    cp_seq = cp_seq - cp_seq[0]  # baseline 0으로 맞춤
    length = len(cp_seq)
    data = {}
    for col in ["BW", "EGFR", "SEX", "RAAS", "BPS", "amt", "II"]:
        data[col] = [row[col]] * length
    data["CP"] = cp_seq
    data["time"] = np.arange(length)
    return pd.DataFrame(data)


def create_dataset(df):
    timeseries_list = [expand_row_to_timeseries(row) for _, row in df.iterrows()]
    updated_list = []
    for ts in timeseries_list:
        df0 = ts.copy()
        # amt 수정: 주입 시점에 맞게 적용 (원래 코드 유지)
        df0["amt"] = df0["amt"] * (df0["time"] % df0["II"] == 0)
        updated_list.append(df0)
        non_zero = df0[df0["amt"] != 0]
        if len(non_zero) != 6:
            print("error in amt count:", len(non_zero))
    return updated_list


# 정규화에 사용할 통계값 (예시)
std_sr = pd.Series(
    {
        "BW": 20.0,
        "EGFR": 20.0,
        "SEX": 1.0,
        "RAAS": 1.0,
        "BPS": 1.0,
        "amt": 400,
        "CP": 1.0,
        "II": 1.0,
    }
)
mean_sr = pd.Series(
    {
        "BW": 80.0,
        "EGFR": 80,
        "SEX": 0,
        "RAAS": 0,
        "BPS": 0,
        "amt": 0,
        "CP": 0,
        "II": 0.0,
    }
)


##############################################
# Dataset 및 collate_fn
##############################################
class TimeSeriesDataset(Dataset):
    """
    각 환자에 대해,
      - 정적 특성: ["BW", "EGFR", "SEX", "RAAS", "BPS", "amt", "II"] (7차원) → 입력 (한 번만 사용)
      - 출력(label): CP 시퀀스 (시간축에 따른 CP 값)
    """

    def __init__(self, dataframes, mean_sr, std_sr):
        self.dataframes = dataframes
        self.mean_sr = mean_sr
        self.std_sr = std_sr
        self.static_cols = ["BW", "EGFR", "SEX", "RAAS", "BPS", "amt", "II"]

    def __len__(self):
        return len(self.dataframes)

    def __getitem__(self, idx):
        df = self.dataframes[idx]
        # 정적 특성: 첫 행의 값 사용
        static_vec = (
            df[self.static_cols].iloc[0] - self.mean_sr[self.static_cols]
        ) / self.std_sr[self.static_cols]
        static_vec = static_vec.values.astype(np.float32)
        # CP 시퀀스: 정규화 적용
        cp_seq = (df["CP"].values - self.mean_sr["CP"]) / self.std_sr["CP"]
        cp_seq = cp_seq.astype(np.float32)
        return static_vec, cp_seq


def collate_fn(batch):
    """
    각 샘플은 (static_vec, cp_seq)
      - static_vec: (7,)
      - cp_seq: (L,)
    → static_vec는 그대로 유지하고, cp_seq는 최대 길이에 맞춰 패딩.
    """
    batch_static, batch_cp = zip(*batch)
    lengths = [len(cp) for cp in batch_cp]
    max_len = max(lengths)
    bs = len(batch)

    static_inputs = np.stack(batch_static, axis=0)  # (bs, 7)
    cp_labels = np.zeros((bs, max_len), dtype=np.float32)
    mask = np.zeros((bs, max_len), dtype=np.float32)
    for i, cp_seq in enumerate(batch_cp):
        L = len(cp_seq)
        cp_labels[i, :L] = cp_seq
        mask[i, :L] = 1.0

    return (
        torch.tensor(static_inputs, dtype=torch.float32),
        torch.tensor(cp_labels, dtype=torch.float32),
        torch.tensor(mask, dtype=torch.float32),
        torch.tensor(lengths, dtype=torch.long),
    )


##############################################
# Causal Convolution 1D Layer
##############################################
class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size - 1  # 왼쪽에만 padding
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=0, **kwargs
        )

    def forward(self, x):
        # x: (batch, channels, length)
        x = nn.functional.pad(x, (self.padding, 0))  # 왼쪽만 0 패딩
        return self.conv(x)


##############################################
# 모델: 정적 입력으로 CP 시퀀스 예측 (Static-to-Sequence 모델)
##############################################
class StaticToSequenceModel(nn.Module):
    def __init__(
        self, input_dim=7, latent_dim=256, max_injection_interval=100, kernel_size=3
    ):
        """
        input_dim: 정적 입력 차원 (7)
        latent_dim: latent 공간 차원 (예: 256)
        max_injection_interval: 주입 후 시간 경과를 표현하는 최대 길이 (learnable positional embedding의 행 수)
        kernel_size: 컨볼루션의 커널 사이즈 (예: 3, causal convolution 적용)
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.max_injection_interval = (
            max_injection_interval  # 기존 max_seq_len 대신 주입 간격의 최대 길이로 사용
        )
        self.kernel_size = kernel_size

        # 정적 입력 conditioning network: 정적 벡터를 latent 공간으로 매핑
        self.cond_mlp = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        # 학습 가능한 positional embedding lookup table (행: injection interval 길이, 열: latent_dim)
        # 예를 들어, 주입 후 최대 100시간까지 있다면 0~99시간에 해당하는 embedding을 배정
        self.pos_emb = nn.Parameter(torch.randn(max_injection_interval, latent_dim))
        # Decoder: Causal Conv1d 블록을 사용하여 시퀀스 출력 (출력: CP 값)
        self.decoder = nn.Sequential(
            CausalConv1d(latent_dim, latent_dim, kernel_size=kernel_size),
            nn.ReLU(),
            CausalConv1d(latent_dim, latent_dim, kernel_size=kernel_size),
            nn.ReLU(),
            CausalConv1d(latent_dim, 1, kernel_size=kernel_size),
        )

    def forward(self, static_input, seq_len):
        """
        static_input: (batch, 7) → 정적 특성, 여기서 7번째 요소(인덱스 -1)는 injection interval (II)
        seq_len: int, 배치 내 최대 시퀀스 길이
        """
        batch = static_input.size(0)
        # 정적 정보를 latent vector로 변환
        cond = self.cond_mlp(static_input)  # (batch, latent_dim)
        # 정적 벡터를 시퀀스 길이만큼 반복하여, 각 시간 스텝마다 조건 정보로 사용
        cond_rep = cond.unsqueeze(1).repeat(
            1, seq_len, 1
        )  # (batch, seq_len, latent_dim)

        # injection interval은 static_input의 마지막 요소라고 가정 (형태: (batch,))
        # 정수형으로 변환
        ii = static_input[:, -1].long()
        # 모든 배치에 대해, 0부터 seq_len-1까지의 위치 인덱스 (모든 샘플에 공통으로 0,1,2,... seq_len-1)
        pos_indices = (
            torch.arange(seq_len, device=static_input.device)
            .unsqueeze(0)
            .expand(batch, seq_len)
        )
        # 각 샘플마다 injection interval로 나눈 나머지를 사용해 "상대적 시간" 계산
        # 즉, injection이 일어날 때마다 0부터 다시 시작하는 방식
        relative_pos = pos_indices % ii.unsqueeze(
            1
        )  # (batch, seq_len), 각 원소는 0 이상 ii-1 이하의 값

        # 이제, learnable한 pos_emb (크기: (max_injection_interval, latent_dim))에서
        # 각 sample, 각 시점에 해당하는 row를 lookup합니다.
        # pos_emb_expanded: (batch, max_injection_interval, latent_dim)
        pos_emb_expanded = self.pos_emb.unsqueeze(0).expand(batch, -1, -1)
        # relative_pos: (batch, seq_len) → expand to (batch, seq_len, latent_dim)
        # torch.gather: 각 sample별로, 시점에 해당하는 embedding을 가져옵니다.
        pos = torch.gather(
            pos_emb_expanded,
            1,
            relative_pos.unsqueeze(2).expand(batch, seq_len, self.latent_dim),
        )
        # 이제 pos: (batch, seq_len, latent_dim)

        # 두 요소를 합칩니다.
        x = cond_rep + pos  # (batch, seq_len, latent_dim)
        x = x.transpose(1, 2)  # (batch, latent_dim, seq_len)
        out = self.decoder(x)  # (batch, 1, seq_len)
        out = out.transpose(1, 2).squeeze(2)  # (batch, seq_len)
        return out


##############################################
# 학습 및 추론 루틴
##############################################
mode = input("모드를 입력하세요. (훈련: t, 모델 실행: i): ").strip().lower()

if mode in ["t", "train"]:
    df_all = pd.read_parquet(PARQUET_PATH, engine="pyarrow")
    updated_list = create_dataset(df_all)
    boundary_idx = int(len(updated_list) * 0.8)
    train_list = updated_list[:boundary_idx]
    test_list = updated_list[boundary_idx:]

    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU Name: {torch.cuda.get_device_name(device)}")

    train_dataset = TimeSeriesDataset(train_list, mean_sr, std_sr)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
    )

    model = StaticToSequenceModel(
        input_dim=7, latent_dim=256, max_injection_interval=100, kernel_size=3
    ).to(device)
    criterion = nn.MSELoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    checkpoint_path = os.path.join(BASE_DIR, "model_static2seq.pth")
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resuming training from epoch {start_epoch}")

    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        epoch_start = time.time()
        total_loss = 0.0

        for static_inputs, cp_labels, mask, lengths in tqdm(
            train_loader, desc=f"Epoch {epoch}", ncols=80
        ):
            static_inputs = static_inputs.to(device)
            cp_labels = cp_labels.to(device)
            mask = mask.to(device)
            current_seq_len = cp_labels.size(1)

            optimizer.zero_grad()
            outputs = model(static_inputs, current_seq_len)
            loss_all = criterion(outputs, cp_labels)
            masked_loss = (loss_all * mask).sum() / mask.sum()
            masked_loss.backward()
            optimizer.step()
            total_loss += masked_loss.item()

        epoch_end = time.time()
        epoch_loss = total_loss / len(train_loader)
        print(
            f"Epoch [{epoch}/{NUM_EPOCHS}] | Loss: {epoch_loss:.4f} | Time: {epoch_end - epoch_start:.2f}s"
        )

        # 매 100 epoch마다 loss 값을 텍스트 파일에 기록 (append)
        if epoch % 100 == 0:
            with open(loss_log_path, "a") as f:
                f.write(f"{epoch},{epoch_loss}\n")
            # 주기적으로 checkpoint 저장 (loss는 텍스트 파일에 따로 저장됨)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                checkpoint_path,
            )
            print(f"Model saved and loss logged at epoch {epoch}")

elif mode in ["i", "infer"]:
    model_path = os.path.join(BASE_DIR, "model_static2seq.pth")
    if not os.path.exists(model_path):
        print("모델 체크포인트가 없습니다. 먼저 학습을 진행하세요.")
        sys.exit(0)

    df_all = pd.read_parquet(PARQUET_PATH, engine="pyarrow")
    updated_list = create_dataset(df_all)
    boundary_idx = int(len(updated_list) * 0.8)
    test_list = updated_list[boundary_idx:]
    test_dataset = TimeSeriesDataset(test_list, mean_sr, std_sr)
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )

    ckpt = torch.load(model_path)
    print(f"Checkpoint loaded from epoch {ckpt['epoch']}")
    model = StaticToSequenceModel(
        input_dim=7, latent_dim=256, max_injection_interval=100, kernel_size=3
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    criterion = nn.MSELoss(reduction="none")
    total_test_loss = 0.0
    total_count = 0.0
    predictions_per_sample = []
    targets_per_sample = []

    with torch.no_grad():
        for static_inputs, cp_labels, mask, lengths in test_loader:
            static_inputs = static_inputs.to(device)
            cp_labels = cp_labels.to(device)
            mask = mask.to(device)
            current_seq_len = cp_labels.size(1)
            outputs = model(static_inputs, current_seq_len)
            loss_all = criterion(outputs, cp_labels)
            total_test_loss += (loss_all * mask).sum().item()
            total_count += mask.sum().item()

            outputs_np = outputs.cpu().numpy()
            labels_np = cp_labels.cpu().numpy()
            lengths_np = lengths.cpu().numpy()
            for i in range(outputs_np.shape[0]):
                L = lengths_np[i]
                predictions_per_sample.append(outputs_np[i, :L])
                targets_per_sample.append(labels_np[i, :L])

    test_loss = total_test_loss / total_count
    print(f"Test Loss: {test_loss:.4f}")

    sample_idx = int(input("그래프를 비교할 데이터의 번호를 입력하세요 (1~2000): ")) - 1
    if sample_idx < 0 or sample_idx >= len(predictions_per_sample):
        sample_idx = 0

    plt.figure(figsize=(10, 6))
    plt.plot(
        targets_per_sample[sample_idx],
        label="True CP(t)",
        color="blue",
        linestyle="--",
        linewidth=2,
    )
    plt.plot(
        predictions_per_sample[sample_idx],
        label="Predicted CP(t)",
        color="red",
        linestyle="-",
        linewidth=2,
        alpha=0.8,
    )
    plt.title("CP(t) Prediction vs True", fontsize=14)
    plt.xlabel("Time index", fontsize=12)
    plt.ylabel("CP value (normalized)", fontsize=12)
    plt.legend(fontsize=12)
    plt.savefig(os.path.join(BASE_DIR, "inference_plot.png"))
    plt.close()

else:
    print("잘못된 모드입니다. t 또는 i를 입력하세요.")
