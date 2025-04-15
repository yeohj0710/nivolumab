import os
import matplotlib.pyplot as plt

# loss_log.txt 파일 경로 (학습 코드와 동일한 경로여야 함)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
loss_log_path = os.path.join(BASE_DIR, "loss_log.txt")
plot_png_path = os.path.join(BASE_DIR, "training_loss.png")

# loss 로그 파일이 존재하는지 확인
if not os.path.exists(loss_log_path):
    print("loss_log.txt 파일이 존재하지 않습니다. 학습 코드를 먼저 실행하세요.")
    exit(1)

# 파일에서 epoch와 loss 값을 읽어오기
epochs = []
losses = []
with open(loss_log_path, "r") as f:
    lines = f.readlines()
    for line in lines:
        # 파일 형식: epoch,loss
        parts = line.strip().split(",")
        if len(parts) != 2:
            continue
        try:
            epoch_val = int(parts[0])
            loss_val = float(parts[1])
            epochs.append(epoch_val)
            losses.append(loss_val)
        except:
            continue

# 플롯 생성: log scale (y 축)
plt.figure()
plt.plot(epochs, losses, marker="o")
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("Loss (log scale)")
plt.title("Training Loss History")
plt.savefig(plot_png_path)
plt.close()
print(f"Loss 그래프가 {plot_png_path}에 저장되었습니다.")
