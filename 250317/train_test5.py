import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
import torch.distributed as dist
import matplotlib.pyplot as plt
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../processed_data.parquet")
CHECKPOINT_PATH = os.path.join(BASE_DIR, "checkpoint.pth")
BATCH_SIZE = 32
NUM_WORKERS = 8
NUM_EPOCHS = 50000
LEARNING_RATE = 1e-4
HIDDEN_DIM = 256
INPUT_DIM = 7
NUM_LAYERS = 10

MEAN_VALUES = pd.Series(
    {
        "BW": 80.0,
        "EGFR": 80.0,
        "SEX": 0.0,
        "RAAS": 0.0,
        "BPS": 0.0,
        "amt": 0.0,
        "CP": 0.0,
        "II": 0.0,
    }
)
STD_VALUES = pd.Series(
    {
        "BW": 20.0,
        "EGFR": 20.0,
        "SEX": 1.0,
        "RAAS": 1.0,
        "BPS": 1.0,
        "amt": 400.0,
        "CP": 1.0,
        "II": 1.0,
    }
)


df = pd.read_parquet(DATA_PATH, engine="pyarrow")
patientSeriesList = []
for _, row in df.iterrows():
    seqLen = len(row.CP_sequence)
    patientSeriesList.append(
        pd.DataFrame(
            {
                "BW": [row.BW] * seqLen,
                "EGFR": [row.EGFR] * seqLen,
                "SEX": [row.SEX] * seqLen,
                "RAAS": [row.RAAS] * seqLen,
                "BPS": [row.BPS] * seqLen,
                "amt": [row.amt] * seqLen,
                "II": [row.II] * seqLen,
                "CP": row.CP_sequence,
            }
        )
    )
splitIdx = int(len(patientSeriesList) * 0.8)
trainSeries = patientSeriesList[:splitIdx]
testSeries = patientSeriesList[splitIdx:]


class PatientTimeSeriesDataset(Dataset):
    def __init__(self, seriesList, meanSeries, stdSeries):
        self.seriesList = seriesList
        self.meanSeries = meanSeries
        self.stdSeries = stdSeries
        self.cols = meanSeries.index

    def __len__(self):
        return len(self.seriesList)

    def __getitem__(self, idx):
        ts = self.seriesList[idx][self.cols]
        return (ts - self.meanSeries) / self.stdSeries


def collateFn(batch):
    seqLens = [len(x) - 1 for x in batch]
    maxLen = max(seqLens)
    bs = len(batch)
    inputsPadded = np.zeros((bs, maxLen, 7), dtype=np.float32)
    labelsPadded = np.zeros((bs, maxLen, 1), dtype=np.float32)
    masks = np.zeros((bs, maxLen, 1), dtype=np.float32)
    injInputs = np.zeros((bs, 5, 7), dtype=np.float32)
    injLabels = np.zeros((bs, 5, 1), dtype=np.float32)
    for i, seq in enumerate(batch):
        curLen = seqLens[i]
        inputsPadded[i, :curLen, 1:] = seq.iloc[:-1][
            ["BW", "EGFR", "SEX", "RAAS", "BPS", "amt"]
        ].values
        cpDiff = seq.iloc[1:][["CP"]].values - seq.iloc[:-1][["CP"]].values
        inputsPadded[i, 1:curLen, 0] = cpDiff[:-1].squeeze()
        labelsPadded[i, :curLen, :] = cpDiff
        masks[i, :curLen, :] = 1
        cpDiffDF = seq.diff()[["CP"]].fillna(0)
        injIdx = (np.arange(5) * seq["II"].iloc[0]).astype(int)
        injNext = injIdx + 1
        injInputs[i, :, 1:] = seq.iloc[:-1][
            ["BW", "EGFR", "SEX", "RAAS", "BPS", "amt"]
        ][seq.iloc[:-1]["amt"] > 0].values
        injInputs[i, :, 0:1] = cpDiffDF.iloc[injIdx, :].values
        injLabels[i, :] = cpDiffDF.iloc[injNext, :].values
    return (
        torch.tensor(inputsPadded),
        torch.tensor(labelsPadded),
        torch.tensor(masks),
        torch.tensor(seqLens),
        torch.tensor(injInputs),
        torch.tensor(injLabels),
    )


trainDataset = PatientTimeSeriesDataset(trainSeries, MEAN_VALUES, STD_VALUES)
testDataset = PatientTimeSeriesDataset(testSeries, MEAN_VALUES, STD_VALUES)


class CPDiffMLP(nn.Module):
    def __init__(self, inDim, hidDim, nLayers):
        super(CPDiffMLP, self).__init__()
        layers = [nn.Linear(inDim, hidDim), nn.SiLU()]
        for _ in range(nLayers - 1):
            layers += [nn.Linear(hidDim, hidDim), nn.SiLU()]
        layers.append(nn.Linear(hidDim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def mainWorker(gpu, worldSize, args):
    if worldSize > 1:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"
        dist.init_process_group("nccl", rank=gpu, world_size=worldSize)
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    trainSampler = None
    if worldSize > 1:
        trainSampler = torch.utils.data.distributed.DistributedSampler(
            trainDataset, num_replicas=worldSize, rank=gpu
        )
    trainLoader = DataLoader(
        trainDataset,
        batch_size=BATCH_SIZE,
        shuffle=(trainSampler is None),
        sampler=trainSampler,
        collate_fn=collateFn,
        num_workers=NUM_WORKERS,
    )
    model = CPDiffMLP(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS).to(device)
    if worldSize > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    criterion = nn.MSELoss(reduction="none")
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    lossHistory = []
    for epoch in range(args.startEpoch, args.numEpochs):
        if worldSize > 1 and trainSampler is not None:
            trainSampler.set_epoch(epoch)
        model.train()
        totalLossMain = 0
        totalLossInj = 0
        progressBar = tqdm(
            trainLoader, desc=f"Epoch {epoch}/{args.numEpochs}", ncols=80
        )
        for batch in progressBar:
            (
                batchInputs,
                batchLabels,
                batchMask,
                batchSeqLens,
                batchInjInputs,
                batchInjLabels,
            ) = batch
            batchInputs = batchInputs.cuda()
            batchLabels = batchLabels.cuda()
            batchMask = batchMask.cuda()
            batchInjInputs = batchInjInputs.cuda()
            batchInjLabels = batchInjLabels.cuda()
            outputsMain = model(batchInputs)
            lossMain = criterion(outputsMain, batchLabels)
            maskedLoss = (lossMain * batchMask).mean()
            optimizer.zero_grad()
            maskedLoss.backward()
            optimizer.step()
            outputsInj = model(batchInjInputs)
            lossInj = criterion(outputsInj, batchInjLabels).mean()
            optimizer.zero_grad()
            lossInj.backward()
            optimizer.step()
            totalLossMain += maskedLoss.item()
            totalLossInj += lossInj.item()
            progressBar.set_postfix(
                {
                    "MainLoss": f"{maskedLoss.item():.4f}",
                    "InjLoss": f"{lossInj.item():.4f}",
                }
            )
        avgLossMain = totalLossMain / len(trainLoader)
        avgLossInj = totalLossInj / len(trainLoader)
        if gpu == 0:
            print(
                f"Epoch [{epoch}/{args.numEpochs}] Avg Loss: {avgLossMain:.4f}, Avg Inj Loss: {avgLossInj:.4f}"
            )
            lossHistory.append(avgLossMain)
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": (
                    model.module.state_dict() if worldSize > 1 else model.state_dict()
                ),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss_history": lossHistory,
            }
            torch.save(checkpoint, CHECKPOINT_PATH)
    if worldSize > 1:
        dist.destroy_process_group()


def runInference(model, staticFeatures, t_max, device):
    model.eval()
    staticFeatures = [float(x) for x in staticFeatures]
    staticInput = staticFeatures[:6]
    seqLen = int(t_max)
    inputSeq = np.zeros((1, seqLen, 7), dtype=np.float32)
    inputSeq[:, :, 1:] = np.array(staticInput).reshape(1, 1, 6).repeat(seqLen, axis=1)
    inputTensor = torch.tensor(inputSeq).to(device)
    with torch.no_grad():
        outputDiff = model(inputTensor)
    diffArray = outputDiff.cpu().numpy().squeeze()
    cpArray = np.cumsum(diffArray)
    timeAxis = np.arange(seqLen)
    plt.figure()
    plt.plot(timeAxis, cpArray, marker=".")
    plt.xlabel("Time")
    plt.ylabel("CP")
    plt.title("Predicted CP(t)")
    plt.savefig(os.path.join(BASE_DIR, "predicted_cp.png"))
    plt.close()
    print("Inference 완료: predicted_cp.png 생성됨.")


class Args:
    pass


if __name__ == "__main__":
    modeInput = input("모드를 선택하세요 (t: train, i: infer): ").strip().lower()
    if modeInput == "t":
        modeInput = "train"
    elif modeInput == "i":
        modeInput = "infer"
    else:
        print("잘못된 입력")
        sys.exit(0)
    args = Args()
    args.mode = modeInput
    if args.mode == "train":
        startEpoch = 0
        if os.path.exists(CHECKPOINT_PATH):
            ckpt = torch.load(CHECKPOINT_PATH, map_location=torch.device("cpu"))
            startEpoch = ckpt.get("epoch", 0) + 1
        args.startEpoch = startEpoch
        args.numEpochs = int(
            input(f"몇 epoch까지 학습할까요? (현재 epoch: {startEpoch}): ") or 10000
        )
        gpuCount = torch.cuda.device_count()
        if gpuCount > 1:
            mp.spawn(mainWorker, args=(gpuCount, args), nprocs=gpuCount)
        else:
            mainWorker(0, 1, args)
    elif args.mode == "infer":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = CPDiffMLP(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS).to(device)
        if os.path.exists(CHECKPOINT_PATH):
            ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        staticInputStr = input(
            "BW, EGFR, SEX, RAAS, BPS, amt, II 값을 공백으로 구분하여 입력하세요 (예: 68.28 66.37 1 1 1 244 482): "
        )
        t_max = float(input("CP(t) 그래프 생성을 위한 t_max 값을 입력하세요: "))
        runInference(model, staticInputStr.split(), t_max, device)
