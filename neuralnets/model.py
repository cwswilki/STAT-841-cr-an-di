import torch
import torch.nn as nn

class DNN(nn.Module):
    def __init__(self, 
                input_dim: int, 
                hidden_dims: list[int] = [512],
                output_dim: int = 10,
                dropout: float = 0.0
        ):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())

            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))

            prev_dim = h

        layers.append(nn.Linear(prev_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)