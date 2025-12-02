import os
from typing import Optional

import kagglehub
import pandas as pd
import torch
from torch.utils.data import Dataset


import pandas as pd
import numpy as np


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
        df = df.replace("-", np.nan)

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


class MalwareDataset(Dataset):
    def __init__(self, X, y):
        super().__init__()

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        return x, y
