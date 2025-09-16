# models/dynamo_field.py
# Lightweight wrapper around a Dynamo-learned vector field.
# You can either (A) call Dynamo directly via an adapter, or (B) distill it to a small MLP.

from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional

class DynamoAdapter:
    """
    Plug your trained Dynamo vector field here.
    Implement `predict(x)` to return f(x) with the same normalization you trained Dynamo on.
    """
    def __init__(self, callable_field=None, device="cpu"):
        self.device = device
        self._call = callable_field  # e.g., a Python callable or a pybind to Dynamo
        if self._call is None:
            # Fallback: zero field (safe no-op)
            self._call = lambda x: torch.zeros_like(x)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, D]
        return self._call(x)

class DynamoSurrogate(nn.Module):
    """
    Optional: small MLP surrogate trained to mimic Dynamo f(x).
    Train offline with pairs (x, f_dyn(x)).
    """
    def __init__(self, x_dim: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, x_dim)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def make_lambda_schedule(kind: str = "cosine", start: float = 1.0, end: float = 0.1):
    """
    Guidance strength over time t∈[0,1]. Use high guidance early, anneal near data manifold.
    """
    if kind == "linear":
        return lambda t: start + (end - start) * t
    if kind == "cosine":
        import math
        return lambda t: end + (start - end) * 0.5 * (1 + torch.cos(torch.tensor(t) * math.pi))
    return lambda t: torch.tensor(start)
