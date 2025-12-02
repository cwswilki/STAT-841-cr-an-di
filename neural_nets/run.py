from dataset import MalwareDatasetLoader, MalwareDataset
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from torch.utils.data import DataLoader

import torch
import torch.optim as optim

from model import DNN
from trainer import Trainer

loader = MalwareDatasetLoader()
df_train, df_val, df_test = loader.make_data_splits()

# local_orig, local_resp have only "-"" values
SKIPPED_COLUMNS = [
  'ts', 'uid', 'id.orig_h', 'id.resp_h', 'tunnel_parents', 'detailed-label', 'id.orig_p', 'id.resp_p', 'local_orig', 'local_resp']

ONE_HOT_COLUMNS = ['proto', 'service', 'conn_state', 'history']
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

    y = np.where(df[LABEL_COLUMN] == 'Benign', 1, 0)
    y = y.reshape(-1, 1)
    return X, y

def get_device():
    """Get available device"""

    if torch.cuda.is_available():
        print("Using CUDA...")
        return torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        print("Using MPS...")
        return torch.device("mps")
    else:
        print("Using CPU...")
        return torch.device("cpu")

X_train, y_train = process_data(df_train, using_train_data=True)
X_val, y_val = process_data(df_val, using_train_data=False)
X_test, y_test = process_data(df_test, using_train_data=False)

print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)
print(X_test.shape, y_test.shape)


train_ds = MalwareDataset(X_train, y_train)
val_ds = MalwareDataset(X_val, y_val)
test_ds = MalwareDataset(X_test, y_test)

BATCH_SIZE = 4096

train_dataloader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
val_dataloader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_dataloader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

device = get_device()

# 2 class output -> malware or benign

# no dropout
model_2l= DNN(input_dim=X_train.shape[1], hidden_dims=[512], output_dim=1)
model_3l= DNN(input_dim=X_train.shape[1], hidden_dims=[512, 256], output_dim=1)

# with dropout
model_2l_dropout= DNN(input_dim=X_train.shape[1], hidden_dims=[512], output_dim=1, dropout=0.5)
model_3l_dropout = DNN(input_dim=X_train.shape[1], hidden_dims=[512, 256], output_dim=1, dropout=0.5)

models = [model_2l, model_3l, model_2l_dropout, model_3l_dropout]

for m in models:
    print("Training with model...")
    print(m)
    print("---\n")

    learning_rate=1e-2
    optimizer = optim.SGD(m.parameters(), lr=learning_rate) # optimizer

    trainer = Trainer(
        m,
        optimizer,
        batch_size=BATCH_SIZE,
        learning_rate=learning_rate,
        num_epochs=4,
        check_val_every_n_epoch=2,
        device=device
    )

    trainer.train(train_dataloader, val_dataloader)
    trainer.test(test_dataloader)
    trainer.plot_metrics()

    del m
    del optimizer
    del trainer
