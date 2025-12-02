from data.dataset import MalwareDatasetLoader, MalwareDataset
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from torch.utils.data import DataLoader

import torch
import torch.optim as optim
import multiprocessing
import concurrent

from neuralnets.model import DNN
from neuralnets.trainer import Trainer

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

X_train, y_train = process_data(df_train, using_train_data=True)
X_val, y_val = process_data(df_val, using_train_data=False)
X_test, y_test = process_data(df_test, using_train_data=False)

print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)
print(X_test.shape, y_test.shape)

IN_FEATURES = X_train.shape[1]
train_ds = MalwareDataset(X_train, y_train)
val_ds = MalwareDataset(X_val, y_val)
test_ds = MalwareDataset(X_test, y_test)

del loader 
del df_test, df_val, df_train
del X_train, y_train, X_val, y_val, X_test, y_test
import gc
gc.collect()

BATCH_SIZE = 1024*64

train_dataloader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
val_dataloader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_dataloader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

if torch.cuda.is_available():
   multiprocessing.set_start_method('spawn', force=True)
   print("Multiprocessing start method set to 'spawn' for CUDA compatibility.")

PARALLEL=True
NUM_WORKERS = 6
accuracies = []
futures = []
with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
    for initial_lr in [1, 1e-1, 1e-3, 1e-5]:
        for weight_decay in [0, 0.0001, 0.001, 0.01, 0.1]:
            for dropout in [0, 0.1, 0.5]:
                for optim_type in ["adamw", "sgd"]:
                    for activation_type in ["sigmoid", "relu"]:
                        for layers in [[512 for _ in range(12)], [512], [512, 256], ]:
                            if optim_type == "sgd" and weight_decay != 0:
                                continue
                            desc = (f"Optimizer: {optim_type} - Weight Decay: {weight_decay} - Dropout: {dropout}"
                                 f" - initial LR: {initial_lr} - act: {activation_type} - layers: {layers}")
                            print(f"SCHEDULING: {desc}")
                            if PARALLEL:
                                futures.append(executor.submit(training_run,
                                                               IN_FEATURES,
                                                               BATCH_SIZE,
                                                               train_dataloader,
                                                               val_dataloader,
                                                               test_dataloader,
                                                               weight_decay, dropout, initial_lr,
                                                optim_type, activation_type, layers, log_progress=False,
                                                ))
                            else:
                                result = training_run(IN_FEATURES,
                                                      BATCH_SIZE,
                                                            train_dataloader,
                                                            val_dataloader,
                                                             test_dataloader,
                                                            weight_decay, dropout, initial_lr, optim_type,
                                    activation_type, layers)
                                accuracies.append(result)

CATCH_EXCEPTIONS=False
if CATCH_EXCEPTIONS:
    for future in concurrent.futures.as_completed(futures):
        try:
            result = future.result()
            print(f"FINISHED RESULT: {result}")
            if result is not None:
                accuracies.append(result)
        except Exception as e:
            print(e)
else:
    for future in concurrent.futures.as_completed(futures):
        result = future.result()
        if result is not None:
            accuracies.append(result)