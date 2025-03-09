import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():

    if len(sys.argv) < 2:
        i = int(input("CP 그래프를 그릴 데이터의 인덱스(i)를 입력하세요: "))
    else:
        i = int(sys.argv[1])

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, "..", "processed_data.parquet")

    df = pd.read_parquet(DATA_PATH, engine="pyarrow")
    if i < 0 or i >= len(df):
        print(f"인덱스 {i}는 범위를 벗어났습니다. 전체 데이터 수: {len(df)}")
        sys.exit(1)

    row = df.iloc[i]

    cp_sequence = np.array(row["CP_sequence"], dtype=np.float32) / 1
    t = np.arange(0, len(cp_sequence), dtype=np.float32)

    plt.figure()
    plt.plot(t, cp_sequence, marker=".")
    plt.xlabel("Time")
    plt.ylabel("Concentration")
    plt.title("Real Concentration over Time")

    output_file = os.path.join(BASE_DIR, f"real_concentration_{i}.png")
    plt.savefig(output_file)
    plt.close()
    print(f"CP 그래프가 저장되었습니다: {output_file}")


if __name__ == "__main__":
    main()
