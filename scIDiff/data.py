import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Optional

class SingleCellDataset(Dataset):
    """Very small wrapper around a numpy array or torch tensor of shape [N,D]."""
    def __init__(self, X, device: Optional[str] = None):
        if isinstance(X, np.ndarray):
            self.X = torch.tensor(X, dtype=torch.float32)
        elif torch.is_tensor(X):
            self.X = X.float()
        else:
            raise TypeError("X must be numpy array or torch tensor")
        if device:
            self.X = self.X.to(device)
    def __len__(self):
        return self.X.size(0)
    def __getitem__(self, idx):
        return self.X[idx]

