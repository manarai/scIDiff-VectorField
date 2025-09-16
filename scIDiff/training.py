import torch
from torch.utils.data import DataLoader
from typing import Iterable
from .models import dsm_loss, reverse_sample, ScIDiffModel

class ScIDiffTrainer:
    def __init__(self, model: ScIDiffModel, train_loader: Iterable, lr: float = 2e-4, ot_weight: float = 0.0):
        self.model = model
        self.train_loader = train_loader
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.ot_weight = ot_weight  # placeholder for future OT integration

    def train(self, num_epochs: int = 10):
        self.model.train()
        for ep in range(num_epochs):
            for batch in self.train_loader:
                x0 = batch[0] if isinstance(batch, (tuple, list)) else batch
                x0 = x0.to(next(self.model.parameters()).device)
                loss = dsm_loss(self.model.score, x0, c=None)
                self.opt.zero_grad(); loss.backward(); self.opt.step()
            print(f"[epoch {ep+1}] loss={loss.item():.4f}")

