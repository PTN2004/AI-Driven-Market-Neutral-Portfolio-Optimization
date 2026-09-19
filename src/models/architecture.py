import torch
import torch.nn as nn
from typing import List, Optional

from src.utils.config import Config

class AlphaMLP(nn.Module):
    """
    Deep MLP Architecture for Alpha Prediction with BatchNorm and Dropout to prevent overfitting.
    """
    def __init__(self, input_dim: int, hidden_dims: Optional[List[int]] = None, dropout_rate: float = 0.25):
        super(AlphaMLP, self).__init__()
        if hidden_dims is None:
            hidden_dims = Config.HIDDEN_DIMS

        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            # If input is a sequence (batch, seq_len, features), use the last time step
            x = x[:, -1, :]
        return self.network(x).squeeze(-1)

class AlphaTFT(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout_rate: float = 0.25):
        super(AlphaTFT, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out, _ = self.lstm(x)
        last_out = out[:, -1, :]
        return self.fc(last_out).squeeze(-1)
