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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARQUET_PATH = os.path.join(BASE_DIR, "../processed_data.parquet")
BATCH_SIZE = 128
NUM_EPOCHS = 10000
LEARNING_RATE = 1e-4
MODEL_DIM = 256  # Transformer 모델 차원
NUM_LAYERS = 2  # Transformer Encoder 레이어 수
NUM_HEADS = 4  # multi-head attention 헤드 수

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def expand_row_to_timeseries(row):
    cp_seq = np.array(row["CP_sequence"])
    cp_seq = cp_seq - cp_seq[0]
    length = len(cp_seq)
    data = {}
    for col in ["BW", "EGFR", "SEX", "RAAS", "BPS", "amt", "II"]:
        data[col] = [row[col]] * length
    data["CP"] = cp_seq
    data["time"] = list(range(length))
    return pd.DataFrame(data)


def create_dataset(df):
    timeseries_list = [expand_row_to_timeseries(row) for _, row in df.iterrows()]
    updated_list = []
    for ts in timeseries_list:
        df0 = ts.copy()
        df0["amt"] = df0["amt"] * (df0["time"] % df0["II"] == 0)
        updated_list.append(df0)
        non_zero = df0[df0["amt"] != 0]
        if len(non_zero) != 6:
            print("error", len(non_zero))
    return updated_list


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


class TimeSeriesDataset(Dataset):
    def __init__(self, dataframes, mean_sr, std_sr):
        self.dataframes = dataframes
        self.mean_sr = mean_sr
        self.std_sr = std_sr
        self.data_column = mean_sr.index  # [BW, EGFR, SEX, RAAS, BPS, amt, CP, II]

    def __len__(self):
        return len(self.dataframes)

    def __getitem__(self, idx):
        # 정규화된 DataFrame, shape: (seq_len, 8)
        return (self.dataframes[idx][self.data_column] - self.mean_sr) / self.std_sr


def collate_fn(batch):
    lengths = [len(seq) for seq in batch]
    max_len = max(lengths)

    # 여기서는 time 제외 7개 특성 [CP, BW, EGFR, SEX, RAAS, BPS, amt] 사용
    padded_batch_input = np.zeros((len(batch), max_len, 7), dtype=np.float32)
    # 라벨 (batch, max_len, 1) : 정답 CP 값
    padded_batch_label = np.zeros((len(batch), max_len, 1), dtype=np.float32)
    mask = np.zeros((len(batch), max_len, 1), dtype=np.float32)

    for idx, seq in enumerate(batch):
        cur_len = lengths[idx]
        # seq.columns: [BW, EGFR, SEX, RAAS, BPS, amt, CP, II]
        cp = seq["CP"].values
        bw = seq["BW"].values
        egfr = seq["EGFR"].values
        sex = seq["SEX"].values
        raas = seq["RAAS"].values
        bps = seq["BPS"].values
        amt = seq["amt"].values

        # 입력의 첫 컬럼은 CP, 이후 6개 특성: BW, EGFR, SEX, RAAS, BPS, amt
        padded_batch_input[idx, :cur_len, 0] = cp
        padded_batch_input[idx, :cur_len, 1] = bw
        padded_batch_input[idx, :cur_len, 2] = egfr
        padded_batch_input[idx, :cur_len, 3] = sex
        padded_batch_input[idx, :cur_len, 4] = raas
        padded_batch_input[idx, :cur_len, 5] = bps
        padded_batch_input[idx, :cur_len, 6] = amt

        # 라벨은 CP 값 (전체 시퀀스)
        padded_batch_label[idx, :cur_len, 0] = cp

        mask[idx, :cur_len, 0] = 1

    return (
        torch.tensor(padded_batch_input, dtype=torch.float32),
        torch.tensor(padded_batch_label, dtype=torch.float32),
        torch.tensor(mask, dtype=torch.float32),
        torch.tensor(lengths, dtype=torch.long),
    )


##############################################
# Sinusoidal positional encoding (파라미터 없는 위치 임베딩)
##############################################
def get_sinusoidal_pos_encoding(seq_len: int, model_dim: int, device: torch.device):
    pos = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(
        1
    )  # (seq_len, 1)
    i = torch.arange(model_dim, dtype=torch.float, device=device).unsqueeze(
        0
    )  # (1, model_dim)
    angle_rates = 1 / torch.pow(10000, (2 * (i // 2)) / model_dim)
    angle_rads = pos * angle_rates
    # sin for even indices; cos for odd indices
    pos_encoding = torch.zeros((seq_len, model_dim), device=device)
    pos_encoding[:, 0::2] = torch.sin(angle_rads[:, 0::2])
    pos_encoding[:, 1::2] = torch.cos(angle_rads[:, 1::2])
    return pos_encoding  # (seq_len, model_dim)


##############################################
# Transformer 기반 CP(t) 예측 모델 (Non-Autoregressive)
##############################################
class TransformerPredictor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        model_dim: int,
        num_layers: int,
        nhead: int = 4,
        dropout: float = 0.1,
    ):
        """
        input_dim: 입력 특성 차원 (여기서는 7)
        model_dim: Transformer 모델 차원 (예: 256)
        num_layers: Transformer Encoder 레이어 수
        nhead: multi-head attention 헤드 수
        dropout: dropout 확률
        """
        super().__init__()
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.input_embed = nn.Linear(input_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.fc_out = nn.Linear(model_dim, 1)

    def forward(self, x):
        """
        x: (batch, seq_len, input_dim)
        """
        batch_size, seq_len, _ = x.size()
        # 임베딩
        x = self.input_embed(x)  # (batch, seq_len, model_dim)
        # Sinusoidal positional encoding (seq_len, model_dim)
        pos_enc = get_sinusoidal_pos_encoding(seq_len, self.model_dim, x.device)
        # pos_enc 확장 후 합산
        pos_enc = pos_enc.unsqueeze(0).expand(batch_size, -1, -1)
        x = x + pos_enc
        encoded = self.transformer_encoder(x)  # (batch, seq_len, model_dim)
        out = self.fc_out(encoded)  # (batch, seq_len, 1)
        return out


##############################################
# 학습 및 추론 (Train / Infer) 로직
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
    dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=8,
    )

    # Transformer 기반 모델 생성
    # 입력 특성 차원: 7, 모델 차원: MODEL_DIM, 레이어 수: NUM_LAYERS, nhead: NUM_HEADS
    model = TransformerPredictor(
        input_dim=7,
        model_dim=MODEL_DIM,
        num_layers=NUM_LAYERS,
        nhead=NUM_HEADS,
        dropout=0.1,
    )
    model.to(device)

    criterion = nn.MSELoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    checkpoint_path = os.path.join(BASE_DIR, "model_transformer.pth")
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resuming training from epoch {start_epoch}")

    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        epoch_start = time.time()
        total_loss = 0.0

        for batch_data in tqdm(dataloader, desc=f"Epoch {epoch}", ncols=80):
            padded_batch_input, padded_batch_label, mask, lengths = batch_data
            padded_batch_input = padded_batch_input.to(device)
            padded_batch_label = padded_batch_label.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()
            outputs = model(padded_batch_input)  # (batch, seq_len, 1)
            loss = criterion(outputs, padded_batch_label)
            masked_loss = (loss * mask).sum() / mask.sum()
            masked_loss.backward()
            optimizer.step()
            total_loss += masked_loss.item()

        epoch_end = time.time()
        print(
            f"Epoch [{epoch}/{NUM_EPOCHS}] | Loss: {total_loss/len(dataloader):.4f} | Time: {epoch_end - epoch_start:.2f}s"
        )

        if epoch % 10 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                checkpoint_path,
            )
            print(f"Model saved at epoch {epoch}")

elif mode in ["i", "infer"]:
    model_path = os.path.join(BASE_DIR, "model_transformer.pth")
    if not os.path.exists(model_path):
        print("모델 체크포인트가 없습니다. 먼저 학습을 진행하세요.")
        sys.exit(0)

    df_all = pd.read_parquet(PARQUET_PATH, engine="pyarrow")
    updated_list = create_dataset(df_all)
    boundary_idx = int(len(updated_list) * 0.8)
    test_list = updated_list[boundary_idx:]
    test_dataset = TimeSeriesDataset(test_list, mean_sr, std_sr)
    dataloader_test = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )

    checkpoint = torch.load(model_path)
    print(f"Checkpoint loaded from epoch {checkpoint['epoch']}")
    model = TransformerPredictor(
        input_dim=7,
        model_dim=MODEL_DIM,
        num_layers=NUM_LAYERS,
        nhead=NUM_HEADS,
        dropout=0.1,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    criterion = nn.MSELoss(reduction="none")
    total_test_loss = 0.0
    total_count = 0.0
    predictions_per_sample = []
    targets_per_sample = []

    with torch.no_grad():
        for batch_data in dataloader_test:
            padded_batch_input, padded_batch_label, mask, lengths = batch_data
            padded_batch_input = padded_batch_input.to(device)
            padded_batch_label = padded_batch_label.to(device)
            mask = mask.to(device)

            outputs = model(padded_batch_input)  # (batch, seq_len, 1)
            loss = criterion(outputs, padded_batch_label)
            total_test_loss += (loss * mask).sum().item()
            total_count += mask.sum().item()

            outputs_np = outputs.cpu().numpy()
            labels_np = padded_batch_label.cpu().numpy()
            lengths_np = lengths.cpu().numpy()
            for i in range(outputs_np.shape[0]):
                L = lengths_np[i]
                predictions_per_sample.append(outputs_np[i, :L, 0])
                targets_per_sample.append(labels_np[i, :L, 0])

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
    plt.ylabel("CP value", fontsize=12)
    plt.legend(fontsize=12)
    plt.savefig(os.path.join(BASE_DIR, "inference_plot.png"))
    plt.close()

else:
    print("잘못된 모드입니다. t 또는 i를 입력하세요.")
