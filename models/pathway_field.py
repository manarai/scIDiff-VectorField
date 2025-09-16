# models/pathway_field.py
import torch

class PathwayField:
    """
    Build a simple drift from pathway gene sets and expected sign per drug.
    - gene_sets: dict{name: idx_tensor} mapping to column indices in x
    - effects: dict{drug: dict{name: +1/-1 or real weight}}
    - scale: global magnitude
    """
    def __init__(self, gene_sets, effects, scale=0.05, device="cpu"):
        self.gene_sets = gene_sets
        self.effects = effects
        self.scale = scale
        self.device = device

    @torch.no_grad()
    def g(self, x, drug_name: str):
        B, D = x.shape
        drift = torch.zeros_like(x)
        if drug_name not in self.effects: 
            return drift
        for pname, sign in self.effects[drug_name].items():
            if pname not in self.gene_sets: 
                continue
            idx = self.gene_sets[pname].to(x.device)   # LongTensor indices
            # push pathway genes along expected direction
            drift[:, idx] += self.scale * float(sign)
        return drift
