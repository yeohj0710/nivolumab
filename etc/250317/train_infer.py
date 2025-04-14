import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARQUET_PATH = os.path.join(BASE_DIR, "../processed_data.parquet")
BATCH_SIZE = 512
NUM_EPOCHS = 1000
LEARNING_RATE = 1e-4
HIDDEN_SIZE = 256
INPUT_SIZE = 7
NUM_LAYERS = 10

df_all = pd.read_parquet(PARQUET_PATH, engine="pyarrow")


def expand_row_to_timeseries(row):
    cp_seq = row["CP_sequence"]
    length = len(cp_seq)
    data = {}
    for col in ["BW", "EGFR", "SEX", "RAAS", "BPS", "amt", "II"]:
        data[col] = [row[col]] * length
    data["CP"] = cp_seq
    return pd.DataFrame(data)


timeseries_list = [expand_row_to_timeseries(row) for _, row in df_all.iterrows()]

boundary_idx = int(len(timeseries_list) * 0.8)
train_list = timeseries_list[:boundary_idx]
test_list = timeseries_list[boundary_idx:]

train_all = pd.concat(train_list, axis=0)
test_all = pd.concat(test_list, axis=0)

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
    pad_mask = np.zeros((len(batch), max_len, 1), dtype=np.float32)
    inject_batch_input = np.zeros((len(batch), 5, 7), dtype=np.float32)
    inject_batch_label = np.zeros((len(batch), 5, 1), dtype=np.float32)
    for idx, seq in enumerate(batch):
        padded_batch_input[idx, : lengths[idx], 1:] = (
            seq.iloc[:-1].loc[:, ["BW", "EGFR", "SEX", "RAAS", "BPS", "amt"]].values
        )
        cp_diff = (
            seq.iloc[1:].loc[:, ["CP"]].values - seq.iloc[:-1].loc[:, ["CP"]].values
        )
        padded_batch_input[idx, 1 : lengths[idx], 0] = cp_diff[:-1].squeeze()
        padded_batch_label[idx, : lengths[idx], :] = cp_diff
        pad_mask[idx, : lengths[idx], :] = 1
        cp_diff_df = seq.diff()[["CP"]].fillna(0)
        if len(seq) >= 6:
            loc_1 = np.arange(5)
            loc_2 = loc_1 + 1
            in_df = seq.iloc[:-1].loc[:, ["BW", "EGFR", "SEX", "RAAS", "BPS", "amt"]]
            inject_batch_input[idx, :, 1:] = in_df.values[:5]
            inject_batch_input[idx, :, 0:1] = cp_diff_df.iloc[loc_1, :].values
            inject_batch_label[idx, :] = cp_diff_df.iloc[loc_2, :].values
        else:
            inject_batch_input[idx] = 0
            inject_batch_label[idx] = 0
    return (
        torch.tensor(padded_batch_input),
        torch.tensor(padded_batch_label),
        torch.tensor(pad_mask),
        torch.tensor(lengths),
        torch.tensor(inject_batch_input),
        torch.tensor(inject_batch_label),
    )


train_dataset = TimeSeriesDataset(train_list, mean_sr, std_sr)
test_dataset = TimeSeriesDataset(test_list, mean_sr, std_sr)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        out = self.network(x)
        return out


model = MLPEstimate(
    input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS
)
model.to(device)

dataloader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    collate_fn=collate_fn,
    shuffle=True,
    num_workers=8,
)
criterion = nn.MSELoss(reduction="none")
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

os.makedirs("snapshot6", exist_ok=True)

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    total_loss2 = 0
    for idx, batch in enumerate(dataloader):
        inputs, labels, masks, lengths, inject_in, inject_label = batch
        inputs = inputs.to(device)
        labels = labels.to(device)
        masks = masks.to(device)
        inject_in = inject_in.to(device)
        inject_label = inject_label.to(device)
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
    if epoch % 100 == 0:
        torch.save(
            model.state_dict(), os.path.join("snapshot6", f"model_e_{epoch}.pkl")
        )
    if epoch % 10 == 0:
        print(f"Epoch [{epoch}/{NUM_EPOCHS}], Loss: {total_loss/len(dataloader):.4f}")
        print(f"Epoch [{epoch}/{NUM_EPOCHS}], Loss2: {total_loss2/len(dataloader):.4f}")

model.load_state_dict(torch.load(os.path.join(BASE_DIR, "snapshot1/model_e_400.pkl")))
model.eval()
dataloader_test = DataLoader(test_dataset, batch_size=10, collate_fn=collate_fn)
all_predictions = []
all_targets = []
with torch.no_grad():
    for idx, batch in enumerate(dataloader_test):
        inputs, labels, masks, lengths, _, _ = batch
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        all_predictions.extend(outputs.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())

plt.plot(all_predictions[100])
plt.plot(all_targets[100])
plt.title("Prediction vs Target")
plt.xlabel("Index")
plt.ylabel("Value")
plt.legend(["Prediction", "Target"])
plt.show()
