"""LSTM / BiLSTM sequence classifiers: [B, T, F] -> logits [B, num_classes] (no softmax)."""
from __future__ import annotations

import torch
from torch import nn


class LSTMClassifier(nn.Module):
    """Unidirectional LSTM; representation = final hidden state of the top layer."""

    bidirectional = False

    def __init__(self, input_dim: int, num_classes: int, hidden_size: int = 128, num_layers: int = 2,
                 dropout: float = 0.30) -> None:
        super().__init__()
        self.input_dim, self.num_classes, self.hidden_size = input_dim, num_classes, hidden_size
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0, bidirectional=self.bidirectional)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * (2 if self.bidirectional else 1), num_classes)

    def _check(self, x: torch.Tensor) -> None:
        if x.ndim != 3 or x.shape[-1] != self.input_dim:
            raise ValueError(f"expected input [B, T, {self.input_dim}], got {list(x.shape)}")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        self._check(x)
        out, _ = self.lstm(x)
        return out[:, -1, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.dropout(self.encode(x)))


class BiLSTMClassifier(LSTMClassifier):
    """Bidirectional LSTM; representation = [forward state after the last step |
    backward state after the first step] of the top layer (2 x hidden)."""

    bidirectional = True

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        self._check(x)
        out, _ = self.lstm(x)
        h = self.hidden_size
        return torch.cat([out[:, -1, :h], out[:, 0, h:]], dim=1)


MODELS = {"lstm": LSTMClassifier, "bilstm": BiLSTMClassifier}


def build_model(name: str, input_dim: int, num_classes: int, neural_cfg: dict) -> LSTMClassifier:
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; expected one of {sorted(MODELS)}")
    return MODELS[name](input_dim, num_classes, hidden_size=neural_cfg["hidden_size"],
                        num_layers=neural_cfg["num_layers"], dropout=neural_cfg["dropout"])


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
