import os
from typing import Optional

import kagglehub
import pandas as pd
import torch
from torch.utils.data import Dataset


class MalwareDataset(Dataset):

    _df_cache: Optional[pd.DataFrame] = None

    @classmethod
    def load_df(cls, reload: bool = False) -> pd.DataFrame:
        if cls._df_cache is not None and not reload:
            print("Returning cached dataframe")
            return cls._df_cache

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

        cls._df_cache = df
        return df

    def __init__(self):
        if self._df_cache is not None:
            df = self.load_df()
        pass

    def __len__(self):
        if self._df_cache is None:
            raise ValueError("dataframe does not exist but should")

        return len(self._df_cache)

    def __getitem__(self, idx: int):
        pass
