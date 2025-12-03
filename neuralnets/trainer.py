import sys, os
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import copy
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from typing import Optional
import numpy as np


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        optimizer: optim.Optimizer,
        batch_size: int = 256,
        learning_rate: float = 0.01,
        num_epochs: int = 30,
        check_val_every_n_epoch: int = 1,
        device: torch.device = torch.device("cpu"),
        threshold: float = 0.5,
        pos_weight: Optional[np.ndarray] = None,
    ) -> None:
        """Trainer object to facilitate training and evaluation"""

        self.model = model

        # training configurations
        self.batch_size = batch_size  # does nothing; mainly for viz
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.check_val_every_n_epoch = check_val_every_n_epoch
        self.device = device
        self.threshold = threshold
        self.model.to(self.device)

        # set loss function and optimizer
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        self.optimizer = optimizer
        self.optimizer_name = self.optimizer.__class__.__name__

        self.model_name = model_name

        # model metrics
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []

        out_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "results",
            model_name
        )
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        log_file = out_path / "train.log"

        # plots/ subfolder
        self.plots_dir = out_path / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(levelname)s | %(message)s",
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(log_file, mode="w"),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader, val_dataset) -> None:

        # early stopping configs 
        early_stopping_patience = 3
        best_val_acc = 0
        early_stopping_counter = 0
        best_model_state = None

        for epoch in range(self.num_epochs):
            self.model.train()  # set model to train

            # loss tracking metrics
            running_loss = 0.0
            running_vloss = 0.0
            batch_loss = 0.0
            running_acc = 0.0

            pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))

            for i, (inputs, labels) in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # zero gradients for every batch
                self.optimizer.zero_grad()

                # compute predictions + loss
                outputs = self.model(inputs)  # predicted class
                loss = self.criterion(outputs, labels)

                # compute training accuracy
                running_acc += self.__accuracy(outputs, labels)

                # perform backpropagation
                loss.backward()  # compute gradients

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()  # update model parameters

                # gather data and report
                running_loss += loss.item()
                batch_loss += loss.item()
                if i % 10 == 0:
                    batch_loss = batch_loss / 10  # loss per batch
                    pbar.set_postfix({"loss": round(batch_loss, 5)})
                    batch_loss = 0.0

            train_accuracy = running_acc / len(train_dataloader)
            self.train_accuracies.append((epoch, train_accuracy))

            avg_loss = running_loss / len(train_dataloader)
            self.train_losses.append((epoch, avg_loss))

            if epoch % self.check_val_every_n_epoch == 0:
                self.model.eval()  # set model to evaluation
                with torch.no_grad():
                    running_val_acc = 0
                    for inputs, labels in val_dataloader:
                        inputs, labels = inputs.to(self.device), labels.to(self.device)

                        outputs = self.model(inputs)
                        outputs = outputs.clamp(min=-50, max=50)
                        loss = self.criterion(outputs, labels)

                        running_vloss += loss.item()
                        # compute validtion accuracy
                        running_val_acc += self.__accuracy(outputs, labels)

                val_accuracy = running_val_acc / len(val_dataloader)
                self.val_accuracies.append((epoch, val_accuracy))

                avg_vloss = running_vloss / len(val_dataloader)
                self.val_losses.append((epoch, avg_vloss))

                self.logger.info(
                    f"[EPOCH {epoch + 1}] LOSS : train={avg_loss} val={avg_vloss} | ACCURACY : train={train_accuracy} val={val_accuracy}"
                )

                # crit_debug = torch.nn.BCEWithLogitsLoss(reduction="none")

                # self.model.eval()
                # with torch.no_grad():
                #     all_losses = []
                #     all_logits = []
                #     all_labels = []

                #     for inputs, labels in val_dataloader:
                #         inputs, labels = inputs.to(self.device), labels.to(self.device).float()
                #         logits = self.model(inputs).view(-1)
                #         labels = labels.view(-1)
                #         per_item = crit_debug(logits, labels)

                #         all_losses.append(per_item.cpu())
                #         all_logits.append(logits.cpu())
                #         all_labels.append(labels.cpu())

                # all_losses = torch.cat(all_losses)
                # all_logits = torch.cat(all_logits)
                # all_labels = torch.cat(all_labels)

                # topk_vals, topk_idx = torch.topk(all_losses, k=20)
                # topk_idx = topk_idx.tolist()

                # for i in topk_idx:
                #     x_i, y_i = val_dataset[i]  # adjust for your dataset type
                #     print("Index:", i)
                #     print("Label:", y_i)
                #     print("Features:", x_i)
                #     print("Feature max abs:", x_i.abs().max())
                #     print("-" * 40)

                # Early stopping check
                if val_accuracy > best_val_acc:
                    best_val_acc = val_accuracy
                    early_stopping_counter = 0
                    # Save best model state
                    best_model_state = copy.deepcopy(self.model.state_dict())
                else:
                    early_stopping_counter += 1
                    if early_stopping_counter >= early_stopping_patience:
                        self.logger.info(
                            f"Early stopping triggered! No improvement in val_acc "
                            f"for {early_stopping_patience} epochs. Best val_acc: {best_val_acc:.6f}"
                        )
                        # Restore best model weights
                        if best_model_state is not None:
                            self.model.load_state_dict(best_model_state)
                        break
        
        # Restore best model weights at the end of training
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            self.logger.info(f"Restored best model weights with val_acc={best_val_acc:.6f}")

    def test(self, test_dataloader: DataLoader) -> float:
        correct = 0
        self.model.eval()
        with torch.no_grad():
            for inputs, labels in test_dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                correct += self.__accuracy(outputs, labels)

        self.logger.info(f"Test accuracy: {(correct / len(test_dataloader)) * 100} %")
        return correct / len(test_dataloader)

    def plot_metrics(self) -> None:
        t_iters, t_loss = list(zip(*self.train_losses))
        v_iters, v_loss = list(zip(*self.val_losses))
        _, acc = list(zip(*self.train_accuracies))
        _, v_acc = list(zip(*self.val_accuracies))

        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"Model: [{self.model_name}]")

        ax[0].set_title(f"Loss Curve (batch_size={self.batch_size}, lr={self.learning_rate})")
        ax[0].plot(t_iters, t_loss)
        ax[0].plot(v_iters, v_loss)
        ax[0].set_xlabel("Epochs")
        ax[0].set_ylabel("Loss")
        ax[0].legend(["Train", "Validation"])
        ax[0].set_xticks(t_iters)

        ax[1].set_title(f"Accuracy Curve (batch_size={self.batch_size}, lr={self.learning_rate})")
        ax[1].plot(t_iters, acc)
        ax[1].plot(v_iters, v_acc)
        ax[1].set_xlabel("Epochs")
        ax[1].set_ylabel("Accuracy")
        ax[1].legend(["Train", "Validation"])
        ax[1].set_xticks(t_iters)

        plot_path = self.plots_dir / f"{self.model_name}_metrics.png"
        fig.savefig(plot_path)
        plt.show()

    def __accuracy(self, outputs: torch.Tensor, labels: torch.Tensor) -> float:
        probs = torch.sigmoid(outputs)
        preds = (probs >= self.threshold).int()
        return (torch.sum(preds == labels) / len(preds)).item()