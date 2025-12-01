import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def make_data_splits(
    df: pd.DataFrame,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 123,
    reset_index: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    shuffled_df = df.sample(frac=1, random_state=seed).reset_index(drop=reset_index)

    # For a 70-15-15 train-val-test split
    # 70% train - 30% val/test split
    df_train, df_test = train_test_split(
        shuffled_df, test_size=1 - train_size, random_state=seed
    )
    # (0.15 / 0.30 = 0.5) -> 50% split between val/test to give us
    # 70-15-15 split
    df_val, df_test = train_test_split(
        df_test, test_size=test_size / (test_size + val_size), random_state=seed
    )

    print(f"Train: {len(df_train)}")
    print(f"Val: {len(df_val)}")
    print(f"Test: {len(df_test)}")

    return df_train, df_val, df_test
