from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

class SimpleMLP(nn.Module):
    def __init__(self, 
                input_dim: int, 
                hidden_dims: list[int] = [512],
                output_dim: int = 10,
                dropout: float = 0.0,
                activation_type: str = "relu",
                use_batch_norm: bool = True,
        ):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            if activation_type == "relu":
                layers.append(nn.ReLU())
            elif activation_type == "sigmoid":
                layers.append(nn.Sigmoid())
            else:
                assert(False)
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(num_features=h))

            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))

            prev_dim = h

        layers.append(nn.Linear(prev_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def get_device():
    """Get available device"""

    if torch.cuda.is_available():
        print("Using CUDA...")
        return torch.device("cuda:0")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        print("Using MPS...")
        return torch.device("mps")
    else:
        print("Using CPU...")
        return torch.device("cpu")

def accuracy(outputs: torch.Tensor, labels: torch.Tensor, threshold=0.5):
    probs = torch.sigmoid(outputs)
    preds = (probs >= threshold).int()
    return torch.sum(preds == labels) / len(preds)

def train(model, 
          epochs,
          optimizer,
          scheduler,
          criterion,
          train_dataloader,
          val_dataloader,
          check_val_every_n_epoch,
          log_progress,
          desc,
    ):
    train_accuracies = []
    train_losses = []
    val_accuracies = []
    val_losses = []
    device = get_device()

    for epoch in range(epochs):
        model.train()
        if log_progress:
            pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        else:
            pbar = enumerate(train_dataloader)
        batch_loss = 0.0
        running_loss = 0.0
        running_acc = 0.0
        running_vloss = 0.0

        for i, (inputs, labels) in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.reshape(-1, 1)

            # zero gradients for every batch
            optimizer.zero_grad()

            # compute predictions + loss
            outputs = model(inputs)  # predicted class
            loss = criterion(outputs, labels)

            # compute training accuracy
            running_acc += accuracy(outputs, labels)

            # perform backpropagation
            loss.backward()  # compute gradients
            optimizer.step()  # update model parameters

            # gather data and report
            running_loss += loss.item()
            batch_loss += loss.item()
            if i % 10 == 0:
                batch_loss = batch_loss / 10  # loss per batch
                if log_progress:
                    pbar.set_postfix({"loss": round(batch_loss, 5)})
                batch_loss = 0.0

        train_accuracy = running_acc / len(train_dataloader)
        train_accuracies.append((epoch, train_accuracy))

        avg_loss = running_loss / len(train_dataloader)
        train_losses.append((epoch, avg_loss))
        scheduler.step()

        if epoch % check_val_every_n_epoch == 0:
            model.eval()  # set model to evaluation
            with torch.no_grad():
                running_val_acc = 0
                if log_progress:
                    pbar = tqdm(val_dataloader)
                else:
                    pbar = val_dataloader
                for inputs, labels in pbar:
                    inputs, labels = inputs.to(device), labels.to(device)
                    labels = labels.reshape(-1, 1)

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    running_vloss += loss.item()
                    # compute validtion accuracy
                    running_val_acc += accuracy(outputs, labels)

            val_accuracy = running_val_acc / len(val_dataloader)
            val_accuracies.append((epoch, val_accuracy))

            avg_vloss = running_vloss / len(val_dataloader)
            val_losses.append((epoch, avg_vloss))

            current_lr = scheduler.get_last_lr()[0]
            print(
                f"{desc} \n[EPOCH {epoch + 1}] LOSS : train={avg_loss} val={avg_vloss} | ACCURACY : train={train_accuracy} val={val_accuracy}\n"
                f"LR: {current_lr}"
            )
    return val_accuracies


def training_run(input_dim,
                 train_dataloader,
                 val_dataloader,
                 weight_decay: float,
                 dropout: float,
                 initial_lr: float,
                 optim_type: str,
                 activation_type: str,
                 layer_count: int,
                 log_progress:bool =False):
    desc = f"Optimizer: {optim_type} - Weight Decay: {weight_decay} - Dropout: {dropout} - initial LR: {initial_lr}"
    print(f"Starting: {desc}")
    criterion = nn.BCEWithLogitsLoss() # Loss function

    model = SimpleMLP(input_dim=input_dim,
                        hidden_dims=[512 for _ in range(layer_count)],
                        output_dim=1,
                        dropout=dropout,
                        activation_type=activation_type)
    device = get_device()
    model = model.to(device)

    if optim_type == "sgd":
        optimizer = optim.AdamW(model.parameters(), lr=initial_lr, betas=(0.9, 0.999), eps=1e-9, weight_decay=weight_decay) # optimizer
    else:
        optimizer = optim.SGD(model.parameters(), lr=initial_lr) # optimizer
        
    epochs = 10
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    check_val_every_n_epoch = 1

    val_accuracies = train(model, epochs, optimizer, scheduler, criterion, train_dataloader, val_dataloader, check_val_every_n_epoch,
                           log_progress=log_progress, desc=desc)
    print(f"MAX ACC: {max(val_accuracies)}")
    return (desc, val_accuracies)
