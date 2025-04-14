import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARQUET_PATH = os.path.join(BASE_DIR, "../processed_data.parquet")
BATCH_SIZE = 128
NUM_EPOCHS = 10000
LEARNING_RATE = 1e-4
HIDDEN_SIZE = 256
INPUT_SIZE = 7
NUM_LAYERS = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def expand_row_to_timeseries(row):
    cp_seq = row["CP_sequence"]
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
        self.data_column = mean_sr.index

    def __len__(self):
        return len(self.dataframes)

    def __getitem__(self, idx):
        return (self.dataframes[idx][self.data_column] - self.mean_sr) / self.std_sr


def collate_fn(batch):
    lengths = [len(seq) - 1 for seq in batch]
    max_len = max(lengths)
    padded_batch_input = np.zeros((len(batch), max_len, 7), dtype=np.float32)
    padded_batch_label = np.zeros((len(batch), max_len, 1), dtype=np.float32)
    pad_mask = np.zeros((len(batch), max_len, 1), np.float32)
    inject_batch_input = np.zeros((len(batch), 5, 7), dtype=np.float32)
    inject_batch_label = np.zeros((len(batch), 5, 1), dtype=np.float32)
    for idx, seq in enumerate(batch):
        cp_diff_df = seq.diff()[["CP"]].fillna(0)
        main_cp_diff = cp_diff_df.iloc[1:].values
        padded_batch_input[idx, : lengths[idx], 1:] = (
            seq.iloc[:-1].loc[:, ["BW", "EGFR", "SEX", "RAAS", "BPS", "amt"]].values
        )
        padded_batch_input[idx, : lengths[idx], 0] = main_cp_diff.squeeze()
        padded_batch_label[idx, : lengths[idx], :] = main_cp_diff
        pad_mask[idx, : lengths[idx], :] = 1
        loc_1 = ((np.arange(5) * seq["II"].iloc[0]).astype(int)) + 1
        loc_2 = loc_1 + 1
        in_df = seq.iloc[:-1].loc[:, ["BW", "EGFR", "SEX", "RAAS", "BPS", "amt"]]
        inject_batch_input[idx, :, 1:] = in_df.values[:5]
        inject_batch_input[idx, :, 0:1] = cp_diff_df.iloc[loc_1, :].values
        inject_batch_label[idx, :] = cp_diff_df.iloc[loc_2, :].values
    return (
        torch.tensor(padded_batch_input),
        torch.tensor(padded_batch_label),
        torch.tensor(pad_mask),
        torch.tensor(lengths),
        torch.tensor(inject_batch_input),
        torch.tensor(inject_batch_label),
    )


class MLPEstimate(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(MLPEstimate, self).__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.SiLU())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(hidden_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


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
    model = MLPEstimate(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS)
    model.to(device)
    criterion = nn.MSELoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    checkpoint_path = os.path.join(BASE_DIR, "model.pth")
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resuming training from epoch {start_epoch}")
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        total_loss2 = 0
        for batch in dataloader:
            inputs, labels, masks, lengths, inject_in, inject_label = batch
            inputs, labels, masks = (
                inputs.to(device),
                labels.to(device),
                masks.to(device),
            )
            inject_in, inject_label = inject_in.to(device), inject_label.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            masked_loss = (loss * masks).mean()
            masked_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            outputs2 = model(inject_in)
            loss2 = criterion(outputs2, inject_label)
            loss3 = loss2.mean()
            loss3.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += masked_loss.item()
            total_loss2 += loss3.item()
        if epoch % 10 == 0:
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}], Normal Loss: {total_loss/len(dataloader):.2f}"
            )
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}], Injection Loss: {total_loss2/len(dataloader):.2f}"
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                checkpoint_path,
            )
elif mode in ["i", "infer"]:
    model_path = os.path.join(BASE_DIR, "model.pth")
    print(f"{model_path}을 불러왔습니다.")
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
    model = MLPEstimate(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    predictions_per_sample = []
    targets_per_sample = []
    for batch in dataloader_test:
        inputs, labels, masks, lengths, _, _ = batch
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        outputs_np = outputs.detach().cpu().numpy()
        labels_np = labels.cpu().numpy()
        for i in range(outputs_np.shape[0]):
            L = int(lengths[i].item())
            predictions_per_sample.append(outputs_np[i, :L, 0])
            targets_per_sample.append(labels_np[i, :L, 0])
    raw_CP_predictions = []
    raw_CP_true = []
    for idx, df in enumerate(test_list):
        CP_raw = df["CP"].values
        CP0 = CP_raw[0]
        pred_diff = predictions_per_sample[idx]
        CP_pred = CP0 + np.cumsum(pred_diff)
        raw_CP_predictions.append(CP_pred)
        raw_CP_true.append(CP_raw[1:])
    plt.figure(figsize=(10, 6))
    plt.plot(raw_CP_predictions[0], label="Predicted CP(t)")
    plt.plot(raw_CP_true[0], label="True CP(t)")
    plt.title("CP(t) Prediction vs True", fontsize=14)
    plt.xlabel("Time index", fontsize=12)
    plt.ylabel("CP value", fontsize=12)
    plt.legend(fontsize=12)
    plt.savefig(os.path.join(BASE_DIR, "inference_plot.png"))
    plt.close()
else:
    print("잘못된 모드입니다. t 또는 i를 입력하세요.")
