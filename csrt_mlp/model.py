"""MLP regression head: frozen YOLOX feature -> normalised CSRT parameters."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

__all__ = ["CSRTParamMLP", "build_loss"]


class CSRTParamMLP(nn.Module):
    """Predicts every tunable CSRT parameter in the unit interval.

    The final ``sigmoid`` guarantees the decoded parameters always fall inside
    the search ranges used during tuning, so the prediction can be fed straight
    into ``cv2.TrackerCSRT``.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: Sequence[int] = (1024, 512, 256),
        dropout: float = 0.1,
        input_norm: bool = True,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)

        self.input_norm = nn.LayerNorm(in_dim) if input_norm else nn.Identity()

        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(prev, out_dim)

        # start near the middle of every range (~ the OpenCV defaults)
        nn.init.zeros_(self.head.bias)
        nn.init.normal_(self.head.weight, std=1e-3)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(feats.float())
        return torch.sigmoid(self.head(self.trunk(x)))


def build_loss(name: str = "smooth_l1", beta: float = 0.1):
    """Per-element loss (reduction='none' so samples can be weighted)."""
    name = name.lower()
    if name in ("l1", "mae"):
        return nn.L1Loss(reduction="none")
    if name in ("l2", "mse"):
        return nn.MSELoss(reduction="none")
    if name in ("smooth_l1", "smoothl1", "huber"):
        return nn.SmoothL1Loss(reduction="none", beta=beta)
    raise ValueError(f"unknown loss '{name}'")
