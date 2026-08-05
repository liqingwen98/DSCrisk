
from __future__ import annotations

import torch
from torch import nn


class PairMLP(nn.Module):

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        proj_dim: int = 128,
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_dim * 4, hidden_dim)
        self.act1 = nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.act2 = nn.ReLU(inplace=True)
        self.drop2 = nn.Dropout(dropout)
        self.fc_out = nn.Linear(hidden_dim, num_classes)

        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, h_a: torch.Tensor, h_b: torch.Tensor):
        feat = torch.cat([h_a, h_b, (h_a - h_b).abs(), h_a * h_b], dim=-1)
        h1 = self.drop1(self.act1(self.fc1(feat)))
        z = self.proj(h1)
        h2 = self.drop2(self.act2(self.fc2(h1)))
        logits = self.fc_out(h2)
        return logits, z
