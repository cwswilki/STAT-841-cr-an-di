from neuralnets.trainer import Trainer
import torch
from torch import optim
from neuralnets.model import DNN


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

def training_run(input_dim,
                 batch_size,
                 train_dataloader,
                 val_dataloader,
                 test_dataloader,
                 weight_decay: float,
                 dropout: float,
                 initial_lr: float,
                 optim_type: str,
                 activation_type: str,
                 layers: list[int],
                 log_progress=True):
    desc = (f"Optimizer: {optim_type} - Weight Decay: {weight_decay} - Dropout: {dropout}"
        f" - initial LR: {initial_lr} - act: {activation_type} - layers: {layers}")
    print(f"Starting: {desc}")

    model = DNN(input_dim=input_dim, hidden_dims=layers, output_dim=1, dropout=dropout, activation_type=activation_type)

    device = get_device()
    model = model.to(device)

    if optim_type == "sgd":
        optimizer = optim.AdamW(model.parameters(), lr=initial_lr, betas=(0.9, 0.999), eps=1e-9, weight_decay=weight_decay) # optimizer
    else:
        optimizer = optim.SGD(model.parameters(), lr=initial_lr) # optimizer
        
    epochs = 10

    trainer = Trainer(
        model,
        optimizer,
        batch_size=batch_size,
        learning_rate=initial_lr,
        num_epochs=epochs,
        check_val_every_n_epoch=1,
        device=device,
    )

    trainer.train(train_dataloader, val_dataloader)
    val_acc = trainer.test(val_dataloader)
    trainer.plot_metrics()

    del optimizer, trainer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"Final ACC: {val_acc}")
    return (desc, val_acc)


