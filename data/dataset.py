import os

import kagglehub
import pandas as pd

import pandas as pd
import numpy as np

# local_orig, local_resp have only "-"" values
SKIPPED_COLUMNS = [
  'ts', 'uid', 'id.orig_h', 'id.resp_h', 'tunnel_parents', 'detailed-label', 'id.orig_p', 'id.resp_p', 'local_orig', 'local_resp']

ONE_HOT_COLUMNS = ['proto', 'service', 'conn_state']# 'history']
NUMERIC_COLUMNS = [
   'id.orig_p', 'id.resp_p', 'duration', 'orig_bytes', 'resp_bytes', 'missed_bytes', 'orig_pkts', 'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes'
]
LABEL_COLUMN = 'label'

class MalwareDatasetLoader:

    def __init__(self):
        print("Downloading and loading kaggle dataset...")

        path = kagglehub.dataset_download(
            "agungpambudi/network-malware-detection-connection-analysis"
        )

        dataframes = []
        for dirname, _, filenames in os.walk(path):
            for _, filename in enumerate(filenames):
                full_path = os.path.join(dirname, filename)
                print(f"Using file: {full_path}")
                dataframes.append(pd.read_csv(full_path, sep="|"))

        df = pd.concat(dataframes, ignore_index=True)

        # convert to numeric
        # [duration, orig_bytes, resp_bytes] are objects we need to convert to numeric
        for col in NUMERIC_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(-1)

        df = df[NUMERIC_COLUMNS + ONE_HOT_COLUMNS + [LABEL_COLUMN]]
        self.df = df

    def make_data_splits(
        self,
        train_size: float = 0.70,
        val_size: float = 0.15,
        seed: int = 123,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Default 70-15-15 train-val-test split
        """

        n = len(self.df)
        rng = np.random.default_rng(seed)
        idx = np.arange(n)
        rng.shuffle(idx)

        n_train = int(train_size * n)
        n_val = int(val_size * n)

        train_indices = idx[:n_train]
        val_indices = idx[n_train : n_train + n_val]
        test_indices = idx[n_train + n_val :]

        df_train = self.df.iloc[train_indices]
        df_val = self.df.iloc[val_indices]
        df_test = self.df.iloc[test_indices]

        print(f"Train: {len(df_train)}")
        print(f"Val: {len(df_val)}")
        print(f"Test: {len(df_test)}")

        return df_train, df_val, df_test