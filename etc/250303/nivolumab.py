import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchdiffeq import odeint

# matplotlib 한글 폰트 설정 (Windows의 경우 "Malgun Gothic")
plt.rcParams["font.family"] = "Malgun Gothic"

# ======================================
# 상수 설정
# ======================================
CP_TIME_LIMIT = 30  # CP 시퀀스 시간 제한 (0이면 최대 길이 사용)
MAX_ROWS = 5  # 학습에 사용할 최대 row 수
HIDDEN_DIM = 32  # ODE 내부 은닉층 차원
EPOCHS = 10  # 기본 epoch 수 (추가 학습 시 변경 가능)
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(BASE_DIR, "..", "processed_data.parquet")
CHECKPOINT_PATH = os.path.join(BASE_DIR, "checkpoint.pth")
LOSS_HISTORY_FILE = os.path.join(BASE_DIR, "loss_history.txt")
LOSS_PLOT_FILE = os.path.join(BASE_DIR, "loss_plot.png")
INFER_PLOT_FILE = os.path.join(BASE_DIR, "infer_plot.png")


# ======================================
# 1. 데이터셋 정의 (../processed_data.parquet)
# ======================================
class CPDataset(Dataset):
    def __init__(self, file_path, cp_time_limit=CP_TIME_LIMIT, max_rows=MAX_ROWS):
        df = pd.read_parquet(file_path)
        if max_rows > 0:
            df = df.iloc[:max_rows]
        self.df = df.reset_index(drop=True)
        # 각 row의 CP 시퀀스 길이가 다를 수 있으므로, global max 길이를 구함
        self.global_max_length = max(
            len(row["CP_sequence"]) for _, row in self.df.iterrows()
        )
        # cp_time_limit이 0이면 모든 row의 cp 시퀀스를 최대 길이로 사용
        self.cp_time_limit = (
            cp_time_limit if cp_time_limit != 0 else self.global_max_length
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # 환자 특성: BW, EGFR, SEX, RAAS, BPS
        features = torch.tensor(
            [row["BW"], row["EGFR"], row["SEX"], row["RAAS"], row["BPS"]],
            dtype=torch.float,
        )
        # 투여 정보: amt, II (각 환자마다 고정)
        amt = torch.tensor(row["amt"], dtype=torch.float)
        II = torch.tensor(row["II"], dtype=torch.float)
        # CP 시퀀스: cp_time_limit까지 사용. 길이가 모자라면 마지막 값을 반복하여 패딩.
        cp_seq_raw = row["CP_sequence"]
        if not isinstance(cp_seq_raw, list):
            cp_seq_raw = cp_seq_raw.tolist()
        if len(cp_seq_raw) >= self.cp_time_limit:
            cp_seq = cp_seq_raw[: self.cp_time_limit]
        else:
            pad_length = self.cp_time_limit - len(cp_seq_raw)
            cp_seq = cp_seq_raw + [cp_seq_raw[-1]] * pad_length
        cp_seq = torch.tensor(cp_seq, dtype=torch.float)
        # 초기 CP값 (첫 번째 값)
        cp0 = cp_seq[0].unsqueeze(0)
        return features, cp0, amt, II, cp_seq


def collate_fn(batch):
    static_features = torch.stack([b[0] for b in batch], dim=0)  # (B, 5)
    t = torch.arange(0, batch[0][4].shape[0], dtype=torch.float)  # (T,)
    cp = torch.stack([b[4] for b in batch], dim=0)  # (B, T)
    # 각 샘플의 cp 시퀀스의 첫 번째 값(초기 CP값)을 cp0로 사용 (B, 1)
    cp0 = cp[:, 0].unsqueeze(1)
    amt = torch.stack([b[2] for b in batch], dim=0)  # (B,)
    II = torch.stack([b[3] for b in batch], dim=0)  # (B,)
    return static_features, cp0, amt, II, cp


# ======================================
# 2. 개선된 ODE 함수 (baseline dynamics)
# ======================================
class ODEFunc(nn.Module):
    def __init__(self, feature_dim, hidden_dim):
        super(ODEFunc, self).__init__()
        # 입력: t, cp, features → (1+1+feature_dim)
        self.input_layer = nn.Linear(1 + 1 + feature_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.hidden_layer = nn.Sequential(
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, t, cp, features):
        # t: 스칼라, cp: [batch, 1], features: [batch, feature_dim]
        t_tensor = t * torch.ones(cp.shape[0], 1, device=cp.device)
        inp = torch.cat([t_tensor, cp, features], dim=1)
        x = self.input_layer(inp)
        x = self.layer_norm(x)
        h = self.hidden_layer(x)
        x = x + h  # Residual 연결
        return self.output_layer(x)


# ======================================
# 3. 개선된 투여 효과 (DosingEffect) 모듈
# ======================================
class DosingEffect(nn.Module):
    def __init__(self):
        super(DosingEffect, self).__init__()
        self.a = nn.Parameter(torch.tensor(1.0))
        self.b = nn.Parameter(torch.tensor(0.1))

    def forward(self, t, amt, II):
        device = t.device
        T_len = t.shape[0]
        max_doses = int(torch.ceil(t[-1] / II.max()).item()) + 1
        # dose index 벡터 생성: [max_doses]
        n = torch.arange(max_doses, device=device).float()
        # 각 배치별 투여 시각: [B, max_doses]
        dose_times = n.unsqueeze(0) * II.unsqueeze(1)
        # 시간 그리드: [T_len, 1, 1]
        t_grid = t.unsqueeze(1).unsqueeze(2)
        # dose_times 확장: [1, B, max_doses]
        dose_times = dose_times.unsqueeze(0)
        # 각 시간과 투여 시각 간 차이: [T_len, B, max_doses]
        delta_t = t_grid - dose_times
        mask = (delta_t >= 0).float()
        # amt 확장: [1, B, 1]
        amt_expand = amt.unsqueeze(0).unsqueeze(2)
        # 각 투여의 기여도: [T_len, B, max_doses]
        contrib = mask * (amt_expand * self.a * torch.exp(-self.b * delta_t))
        # dose 차원으로 합산 후: [T_len, B] → [B, T_len]
        effects = contrib.sum(dim=2).transpose(0, 1)
        return effects


# ======================================
# 4. 전체 CP 예측 모델 (Neural ODE + DosingEffect)
# ======================================
class CPModel(nn.Module):
    def __init__(self, feature_dim, hidden_dim, T):
        """
        feature_dim: 환자 특성 차원 (BW, EGFR, SEX, RAAS, BPS → 5)
        hidden_dim: ODE 함수 내부 은닉층 차원
        T: 예측할 최종 시간 (CP_sequence의 길이 - 1)
        """
        super(CPModel, self).__init__()
        self.odefunc = ODEFunc(feature_dim, hidden_dim)
        self.dosing = DosingEffect()
        self.T = T

    def forward(self, features, cp0, amt, II):
        device = features.device
        # 시간 그리드: 0부터 T까지 (총 T+1 포인트)
        t = torch.linspace(0, self.T, self.T + 1, device=device)
        # dopri5 solver와 rtol/atol 지정으로 ODE 풀기
        baseline = odeint(
            lambda t, cp: self.odefunc(t, cp, features),
            cp0,
            t,
            method="dopri5",
            rtol=1e-3,
            atol=1e-4,
        )
        # baseline: [T+1, B, 1] → [B, T+1]
        baseline = baseline.squeeze(-1).transpose(0, 1)
        dosing_effect = self.dosing(t, amt, II)
        cp_pred = baseline + dosing_effect
        return cp_pred


# ======================================
# 5. 학습 루프 (tqdm 진행바, checkpoint 저장, loss 기록 및 그래프 업데이트)
# ======================================
def train_model(
    model, dataset, start_epoch, target_epoch, batch_size, lr, checkpoint_path
):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
    )
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    criterion = nn.MSELoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    loss_history = []
    for epoch in range(target_epoch - start_epoch):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {start_epoch + epoch + 1}/{target_epoch}")
        for features, cp0, amt, II, cp_seq in pbar:
            features = features.to(device)
            cp0 = cp0.to(device)
            amt = amt.to(device)
            II = II.to(device)
            cp_seq = cp_seq.to(device)

            optimizer.zero_grad()
            cp_pred = model(features, cp0, amt, II)
            loss = criterion(cp_pred, cp_seq)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / len(dataloader)
        current_epoch = start_epoch + epoch + 1
        print(f"Epoch {current_epoch} 평균 Loss: {avg_loss:.4f}")
        torch.save(
            {
                "epoch": current_epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
            },
            checkpoint_path,
        )

        loss_history.append(avg_loss)
        with open(LOSS_HISTORY_FILE, "w") as f:
            for e, l in enumerate(loss_history, start=start_epoch + 1):
                f.write(f"{e},{l}\n")
        plt.figure()
        plt.plot(range(start_epoch + 1, current_epoch + 1), loss_history, marker="o")
        plt.yscale("log")
        plt.xlabel("Epoch")
        plt.ylabel("Loss (log scale)")
        plt.title("Epoch별 Loss")
        plt.grid(True)
        plt.savefig(LOSS_PLOT_FILE)
        plt.close()

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {current_epoch} learning rate: {current_lr:.6f}")


# ======================================
# 6. 예측 및 결과 플로팅 (체크포인트 불러오기, 예측 및 실제 CP 데이터 비교, 결과 그래프 저장)
# ======================================
def predict_and_plot(model, features, cp0, amt, II, T, actual_cp=None, save_path=None):
    model.eval()
    device = next(model.parameters()).device
    features = features.to(device)
    cp0 = cp0.to(device)
    amt = amt.to(device)
    II = II.to(device)
    with torch.no_grad():
        cp_pred = model(features, cp0, amt, II)
    cp_pred = cp_pred.cpu().numpy()[0]
    t_axis = range(len(cp_pred))

    plt.figure(figsize=(10, 5))
    plt.plot(t_axis, cp_pred, label="예측 CP")
    if actual_cp is not None:
        t_actual = min(len(actual_cp), len(cp_pred))
        plt.plot(
            range(t_actual),
            actual_cp[:t_actual],
            label="0번 환자의 실제 CP",
            linestyle="--",
        )
    plt.xlabel("시간")
    plt.ylabel("CP 값")
    plt.title("시간에 따른 CP")
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()


# ======================================
# 7. Main: 학습 및 예측 실행 (모드: 학습 모드(t/T), 예측 모드(i/I))
# ======================================
if __name__ == "__main__":
    mode = (
        input("학습 모드(t/T) 또는 예측 모드(i/I)를 선택하여 입력해 주세요: ")
        .strip()
        .upper()
    )

    if mode == "T":
        dataset = CPDataset(FILE_PATH, cp_time_limit=CP_TIME_LIMIT, max_rows=MAX_ROWS)
        T_value = dataset.cp_time_limit - 1
        feature_dim = 5  # BW, EGFR, SEX, RAAS, BPS
        model = CPModel(feature_dim, HIDDEN_DIM, T_value)

        if os.path.exists(CHECKPOINT_PATH):
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=torch.device("cpu"))
            model.load_state_dict(checkpoint["model_state_dict"])
            current_epoch = checkpoint.get("epoch", 0)
            print(f"현재까지 학습된 epoch 수: {current_epoch}")
        else:
            current_epoch = 0
            print("현재 학습된 모델이 없습니다. (epoch = 0)")

        target_epoch = int(input("몇 epoch까지 학습할까요? (예: 10): "))
        if target_epoch <= current_epoch:
            print("목표 epoch는 현재 epoch보다 작거나 같습니다. 학습을 종료합니다.")
        else:
            train_model(
                model,
                dataset,
                current_epoch,
                target_epoch,
                BATCH_SIZE,
                LEARNING_RATE,
                CHECKPOINT_PATH,
            )
            print("학습이 완료되었습니다.")

    elif mode == "I":
        try:
            inputs = input(
                "BW, EGFR, SEX, RAAS, BPS, amt, II, t_max 값을 공백으로 구분하여 입력하세요(예: 68.28 66.37 1 1 1 244 482 30): "
            ).split()
            BW, EGFR, SEX, RAAS, BPS = map(float, inputs[:5])
            amt = float(inputs[5])
            II = float(inputs[6])
            t_max = int(inputs[7])
        except (ValueError, IndexError):
            print("입력 값이 올바르지 않습니다. 프로그램을 종료합니다.")
            exit()

        feature_dim = 5
        model = CPModel(feature_dim, HIDDEN_DIM, t_max)
        if os.path.exists(CHECKPOINT_PATH):
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=torch.device("cpu"))
            model.load_state_dict(checkpoint["model_state_dict"])
            print(
                f"체크포인트에서 불러왔습니다 (epoch: {checkpoint.get('epoch', 'N/A')})."
            )
        else:
            print("저장된 체크포인트가 없습니다. 추론을 수행할 수 없습니다.")
            exit()

        cp0 = torch.tensor([[0.0]])
        features = torch.tensor([[BW, EGFR, SEX, RAAS, BPS]], dtype=torch.float)
        amt_tensor = torch.tensor([amt], dtype=torch.float)
        II_tensor = torch.tensor([II], dtype=torch.float)
        dataset = CPDataset(FILE_PATH, cp_time_limit=CP_TIME_LIMIT, max_rows=MAX_ROWS)
        _, _, _, _, cp_seq = dataset[0]
        actual_cp = cp_seq.numpy()
        actual_t_max = len(actual_cp) - 1
        effective_t_max = min(t_max, actual_t_max)
        predict_and_plot(
            model,
            features,
            cp0,
            amt_tensor,
            II_tensor,
            effective_t_max,
            actual_cp=actual_cp,
            save_path=INFER_PLOT_FILE,
        )

    else:
        print("올바른 모드를 선택하지 않았습니다. 프로그램을 종료합니다.")
