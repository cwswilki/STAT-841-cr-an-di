import os

import kagglehub
import pandas as pd
import torch
from torch.utils.data import Dataset


import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


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

# local_orig, local_resp have only "-"" values
SKIPPED_COLUMNS = [
  'ts', 'uid', 'id.orig_h', 'id.resp_h', 'tunnel_parents', 'detailed-label', 'id.orig_p', 'id.resp_p', 'local_orig', 'local_resp', 'history']

ONE_HOT_COLUMNS = ['proto', 'service', 'conn_state']
NUMERIC_COLUMNS = [
   'duration', 'orig_bytes', 'resp_bytes', 'missed_bytes', 'orig_pkts', 'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes'
]
LABEL_COLUMN = 'label'

num_transformer = Pipeline(
  [
    ("imputer", SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=-1)),
    ("scalar", StandardScaler())
  ]
)
cat_transformer = Pipeline(
  [
    ("imputer", SimpleImputer(missing_values=np.nan, strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown='ignore', sparse_output=False))
  ]
)

preprocessor = ColumnTransformer(
    [("numeric", num_transformer, NUMERIC_COLUMNS),
    ("categorical", cat_transformer, ONE_HOT_COLUMNS)],
    remainder='passthrough'
)

def process_data(df, using_train_data):
    tmp_df = df[ONE_HOT_COLUMNS + NUMERIC_COLUMNS]
    # fit only on training data
    # only transforming for val and test data
    if using_train_data:
        X = preprocessor.fit_transform(tmp_df)
    else:
        X = preprocessor.transform(tmp_df)

    y = np.where(df[LABEL_COLUMN] == 'Benign', 0, 1)
    y = y.reshape(-1, 1)
    return X, y
