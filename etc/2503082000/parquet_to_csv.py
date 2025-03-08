import os
import pandas as pd
import numpy as np
import csv
from tqdm import tqdm


script_dir = os.path.dirname(os.path.abspath(__file__))


parquet_path = os.path.join(script_dir, "..", "processed_data.parquet")
csv_path = os.path.join(script_dir, "..", "processed_data.csv")


np.set_printoptions(threshold=np.inf)


df = pd.read_parquet(parquet_path)


tqdm.pandas(ncols=50)


if "CP_sequence" in df.columns:
    df["CP_sequence"] = df["CP_sequence"].progress_apply(
        lambda x: np.array2string(
            x, threshold=np.inf, separator=" ", max_line_width=100000
        )
    )


df.to_csv(csv_path, index=False, quoting=csv.QUOTE_NONE, escapechar="\\")
