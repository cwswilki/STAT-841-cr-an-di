import pandas as pd
import numpy as np


def make_data_splits(
    df: pd.DataFrame,
    train_size: float = 0.70,
    val_size: float = 0.15,
    seed: int = 123,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Default 70-15-15 train-val-test split
    """

    n = len(df)
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)

    n_train = int(train_size * n)
    n_val = int(val_size * n)

    train_indices = idx[:n_train]
    val_indices = idx[n_train : n_train + n_val]
    test_indices = idx[n_train + n_val :]

    df_train = df.iloc[train_indices]
    df_val = df.iloc[val_indices]
    df_test = df.iloc[test_indices]

    print(f"Train: {len(df_train)}")
    print(f"Val: {len(df_val)}")
    print(f"Test: {len(df_test)}")

    return df_train, df_val, df_test
