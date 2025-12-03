import argparse
import os

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from model import DNN
from trainer import Trainer
from helpers import get_device
from dataset import MalwareDatasetLoader, MalwareDataset, process_data


CACHE_PATH = "neuralnets/cache/"
CACHE_DATA_PATH = os.path.join(CACHE_PATH, "data.npz")

def parse_args():
    parser = argparse.ArgumentParser(description="Train DNN for malware detection")

    parser.add_argument(
        "--run",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--learning-rates",
        type=float,
        nargs="+",
        required=True,
        help="List of learning rates, e.g. --learning-rates 0.01 0.001",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["sgd", "adam"],
        required=True,
        help="Optimizer to use: sgd or adam",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        required=True,
        help="Hidden layer sizes, e.g. --layers 512 256",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        required=True,
        help="Batch size for training, e.g. --batch-size 512",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=30,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--check-val-every-n-epoch",
        type=int,
        default=5,
        help="Run validation every N epochs",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.5,
        help="Dropout rate (default=0.5)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # === Data loading / preprocessing (with cache) ===
    if os.path.exists(CACHE_DATA_PATH):
        print("Loading cached transformed data...")
        data = np.load(CACHE_DATA_PATH)
        X_train, y_train = data["X_train"], data["y_train"]
        X_val, y_val = data["X_val"], data["y_val"]
        X_test, y_test = data["X_test"], data["y_test"]
    else:
        os.makedirs(CACHE_PATH, exist_ok=True)
        loader = MalwareDatasetLoader()
        df_train, df_val, df_test = loader.make_data_splits()

        X_train, y_train = process_data(df_train, using_train_data=True)
        X_val, y_val = process_data(df_val, using_train_data=False)
        X_test, y_test = process_data(df_test, using_train_data=False)

        np.savez(
            CACHE_DATA_PATH,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
        )

    # Data sizes
    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)
    print(X_test.shape, y_test.shape)

    # Class distribution
    print(
        f"\nClass distribution - Train: Benign={np.sum(y_train == 0) / len(y_train):.2%}, "
        f"Malware={np.sum(y_train == 1) / len(y_train):.2%}"
    )
    print(
        f"Class distribution - Val: Benign={np.sum(y_val == 0) / len(y_val):.2%}, "
        f"Malware={np.sum(y_val == 1) / len(y_val):.2%}"
    )
    print(
        f"Class distribution - Test: Benign={np.sum(y_test == 0) / len(y_test):.2%}, "
        f"Malware={np.sum(y_test == 1) / len(y_test):.2%}\n"
    )

    BATCH_SIZE = args.batch_size

    # Datasets / Dataloaders
    train_ds = MalwareDataset(X_train, y_train)
    val_ds = MalwareDataset(X_val, y_val)
    test_ds = MalwareDataset(X_test, y_test)

    train_dataloader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_dataloader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    device = get_device()

    # Positive weight for class imbalance
    num_neg = np.sum(y_train == 0)
    num_pos = np.sum(y_train == 1)
    pos_weight = torch.tensor([num_neg / num_pos]).to(device)

    model_results = []

    for lr in args.learning_rates:
        model = DNN(
            input_dim=X_train.shape[1],
            hidden_dims=args.layers,
            output_dim=1,
            dropout=args.dropout,
        ).to(device)

        if args.optimizer.lower() == "sgd":
            optimizer = optim.SGD(model.parameters(), lr=lr)
        else:  # adam
            optimizer = optim.Adam(model.parameters(), lr=lr)

        trainer = Trainer(
            model=model,
            model_name=args.run,
            optimizer=optimizer,
            batch_size=BATCH_SIZE,
            learning_rate=lr,
            num_epochs=args.num_epochs,
            check_val_every_n_epoch=args.check_val_every_n_epoch,
            device=device,
            pos_weight=pos_weight
        )

        trainer.logger.info(f"\n##### Model ({args.run}) #####")
        trainer.logger.info(f"Learning Rate: {lr}")
        trainer.logger.info(f"Optimizer: {args.optimizer}")
        trainer.logger.info(f"Hidden Layers: {args.layers}")
        trainer.logger.info(f"Batch Size: {args.batch_size}")
        trainer.logger.info(f"Num Epochs: {args.num_epochs}")
        trainer.logger.info(f"Dropout: {args.dropout}")

        trainer.train(train_dataloader, val_dataloader, val_ds)
        test_acc = trainer.test(test_dataloader)
        trainer.plot_metrics()

        model_results.append((lr, test_acc))

        del model, optimizer, trainer
        if device.type == "cuda":
            torch.cuda.empty_cache()

    best_lr, best_acc = max(model_results, key=lambda x: x[1])
    print(f"\nBest lr={best_lr} (test acc = {best_acc})")


if __name__ == "__main__":
    main()
