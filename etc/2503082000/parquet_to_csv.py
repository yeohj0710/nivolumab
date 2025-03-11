import os
import pandas as pd
import numpy as np
import csv
from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))
parquet_path = os.path.join(script_dir, "..", "..", "processed_data.parquet")
csv_path = os.path.join(script_dir, "..", "..", "processed_data.csv")
np.set_printoptions(threshold=np.inf)
df = pd.read_parquet(parquet_path)
tqdm.pandas(ncols=50)
if "CP_sequence" in df.columns:
    cp_expanded = pd.DataFrame(df["CP_sequence"].tolist())
    cp_expanded.columns = [f"CP_{i}" for i in range(cp_expanded.shape[1])]
    df = df.drop("CP_sequence", axis=1).join(cp_expanded)
df.to_csv(csv_path, index=False, quoting=csv.QUOTE_NONE, escapechar="\\")
