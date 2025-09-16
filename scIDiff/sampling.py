import torch
from dataclasses import dataclass
from typing import Dict, List, Optional
from .models import ScIDiffModel

@dataclass
class PhenotypeTarget:
    gene_targets: Dict[str, float]
    marker_genes: Optional[List[str]] = None
    suppressed_genes: Optional[List[str]] = None

class GeneExpressionObjective:
    """Simple L2 objective on selected genes by index mapping."""
    def __init__(self, var_names: List[str]):
        self.var_names = var_names
        self.g2i = {g:i for i,g in enumerate(var_names)}
    def score(self, x: torch.Tensor, targets: Dict[str, float]) -> torch.Tensor:
        idx, vals = [], []
        for g,v in targets.items():
            if g in self.g2i:
                idx.append(self.g2i[g]); vals.append(v)
        if not idx:
            return torch.zeros(x.size(0), device=x.device)
        idx = torch.tensor(idx, dtype=torch.long, device=x.device)
        tgt = torch.tensor(vals, dtype=torch.float32, device=x.device).view(1, -1)
        return (x[:, idx] - tgt).pow(2).mean(dim=1)

class InverseDesigner:
    def __init__(self, model: ScIDiffModel, var_names: Optional[List[str]] = None):
        self.model = model
        self.var_names = var_names or [f"g{i}" for i in range(model.gene_dim)]
        self.obj = GeneExpressionObjective(self.var_names)
    @torch.no_grad()
    def design(self, target: PhenotypeTarget, batch_size: int = 512):
        samples = self.model.sample(batch_size=batch_size)
        scores = self.obj.score(samples, target.gene_targets)
        best = torch.argmin(scores)
        return samples[best:best+1]
