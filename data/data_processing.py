from typing import Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import make_column_transformer
from data import dataset
from sklearn.preprocessing import PowerTransformer
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin
from sklearn.pipeline import make_pipeline

def split_out_targets(df):
  features = df.drop(columns=[dataset.LABEL_COLUMN])
  target = pd.DataFrame(index=df.index)
  
  #y_train['label'] = (1 if df_train[LABEL_COLUMN[0]] == 'Benign' else 0)
  target['label'] = np.where(df[dataset.LABEL_COLUMN] == 'Benign', 1, 0)

  return (
      features.reset_index(drop=True), 
      target.reset_index(drop=True), 
  )


def force_dense(matrix):
    if hasattr(matrix, "toarray"):
        dense_matrix = matrix.toarray()
    else:
        dense_matrix = matrix
    return dense_matrix

class QuantileClipper(OneToOneFeatureMixin, BaseEstimator, TransformerMixin):
    """
    Clips each feature to [lower_q, upper_q] based on train-set quantiles.
    """
    def __init__(self, lower_q: float = 0.0, upper_q: float = 0.999):
        self.lower_q = lower_q
        self.upper_q = upper_q
        self.lower_: Optional[np.ndarray] = None
        self.upper_: Optional[np.ndarray] = None

    def fit(self, X, y=None):
        X = np.asarray(X)
        # compute per-feature quantiles (axis=0)
        self.lower_ = np.quantile(X, self.lower_q, axis=0)
        self.upper_ = np.quantile(X, self.upper_q, axis=0)
        return self

    def transform(self, X):
        upper_replacement = (2 * self.upper_) + 1
        X = np.asarray(X)
        X_transformed = np.where(X > self.upper_, upper_replacement, X)
        X_transformed = np.where(X_transformed < self.lower_, self.lower_, X_transformed)
        return X_transformed

def preprocess_fit(df_features, include_onehot=True,
                   quantile_clipping=False, power_transform=True,
                   feature_scaling=True, sequential_numeric=True):
    usable_numeric_columns = dataset.NUMERIC_COLUMNS.copy()
    # Any numeric column with only one value, will result in division by zero
    # during normalization, and should just be removed.
    for col in dataset.ONE_HOT_COLUMNS:
        df_features[col] = df_features[col].fillna("unknown").astype(str)

    for col in dataset.NUMERIC_COLUMNS:
        df_features[col] = df_features[col].fillna(-1)
        df_features[col] = pd.to_numeric(df_features[col], errors="coerce")
        if df_features[col].max(axis=0) == df_features[col].min(axis=0):
            df_features = df_features.drop(columns=[col])
            usable_numeric_columns.remove(col)
    transforms_list = []
    if include_onehot:
        transforms_list.append((OneHotEncoder(handle_unknown='ignore'), dataset.ONE_HOT_COLUMNS))
    
    if sequential_numeric:
        numeric_steps = []
        if quantile_clipping:
            # Clip extreme outliers
            numeric_steps.append(QuantileClipper(lower_q=0.0, upper_q=0.999))
        if power_transform:
            # Reshape the distribution
            numeric_steps.append(PowerTransformer(method='yeo-johnson'))
        if feature_scaling:
            # Normalize
            numeric_steps.append(MinMaxScaler())
        if numeric_steps:
            transforms_list.append((make_pipeline(*numeric_steps), usable_numeric_columns))
    else:
        if quantile_clipping:
            # Clip extreme outliers
            transforms_list.append((QuantileClipper(lower_q=0.0, upper_q=0.999), usable_numeric_columns))
        if power_transform:
            # Reshape the distribution
            transforms_list.append((PowerTransformer(method='yeo-johnson'), usable_numeric_columns))
        if feature_scaling:
            # Normalize
            transforms_list.append((MinMaxScaler(), usable_numeric_columns))

    preprocessor = make_column_transformer(
        *transforms_list,
        remainder='passthrough'
    )
    preprocessor.fit(df_features)
    X_transformed = preprocessor.transform(df_features)
    print(f"Global feature space dimensionality: {X_transformed.shape[1]}")
    return X_transformed, preprocessor

def preprocess_existing(df_features, preprocessor):
    X_transformed = preprocessor.transform(df_features)
    return X_transformed
