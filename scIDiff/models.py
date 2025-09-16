import math, torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# ---- basic VP-SDE helpers ----
def beta_t(t, beta_min=0.1, beta_max=20.0):
    return beta_min + t * (beta_max - beta_min)

def marginal_std(t):
    return torch.sqrt(1.0 - torch.exp(-(0.5 * (0.1 + 19.9 * t) * t)))

# ---- tiny Score/Control nets ----
class _TimeMLP(nn.Module):
    def __init__(self, dim=128, out=256):
        super().__init__()
        self.dim = dim
        self.fc = nn.Sequential(nn.Linear(dim, out), nn.SiLU())
    def forward(self, t):
        # Fourier time features
        half = self.dim // 2
        freqs = torch.exp(torch.linspace(0, 6, half, device=t.device))
        ang = t[:, None] * freqs[None, :] * 2 * math.pi
        te = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
        return self.fc(te)

class _ScoreNet(nn.Module):
    def __init__(self, x_dim, hidden=512):
        super().__init__()
        self.tmlp = _TimeMLP()
        self.net = nn.Sequential(
            nn.Linear(x_dim + 256, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, x_dim)
        )
    def forward(self, x, t, c: Optional[torch.Tensor] = None):
        te = self.tmlp(t)
        return self.net(torch.cat([x, te], dim=-1))

class _ControlNet(nn.Module):
    def __init__(self, x_dim, hidden=512, cond_dim=0):
        super().__init__()
        self.cond_dim = cond_dim
        self.net = nn.Sequential(
            nn.Linear(x_dim + cond_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, x_dim)
        )
    def forward(self, x, c: Optional[torch.Tensor] = None):
        h = torch.cat([x, c], dim=-1) if (c is not None and self.cond_dim > 0) else x
        return self.net(h)

# ---- core losses / sampler ----
def dsm_loss(score_net: nn.Module, x0: torch.Tensor, c: Optional[torch.Tensor] = None):
    B = x0.size(0)
    t = torch.rand(B, device=x0.device)
    sigma = marginal_std(t).unsqueeze(-1)
    eps = torch.randn_like(x0)
    xt = torch.sqrt(1 - sigma**2) * x0 + sigma * eps
    s_pred = score_net(xt, t, c)
    target = -eps / (sigma + 1e-7)
    return F.mse_loss(s_pred, target)

@torch.no_grad()
def reverse_sample(score_net: nn.Module, u_net: nn.Module, x_init: torch.Tensor,
                   steps: int = 1000, c: Optional[torch.Tensor] = None):
    x = x_init.clone()
    ts = torch.linspace(1., 0., steps, device=x.device)
    for i in range(steps - 1):
        t = ts[i].expand(x.size(0))
        bt = beta_t(t)
        score = score_net(x, t, c)
        guide = u_net(x, c) if u_net is not None else 0.0
        drift = -(bt/2).unsqueeze(-1) * x - bt.unsqueeze(-1) * score + guide
        dt = (ts[i+1] - ts[i]).item()
        noise = torch.sqrt(torch.clamp(bt, min=1e-8)).unsqueeze(-1) * torch.randn_like(x)
        x = x + drift * dt + noise * (abs(dt) ** 0.5)
    return x

# ---- public model API ----
class ScIDiffModel(nn.Module):
    """
    Minimal wrapper exposing .sample() like you expected.
    (For full OT/Dynamo features, use the extended helpers we added earlier.)
    """
    def __init__(self, gene_dim: int, hidden_dim: int = 512, num_layers: int = 6, num_timesteps: int = 1000):
        super().__init__()
        self.gene_dim = gene_dim
        self.num_timesteps = num_timesteps
        self.score = _ScoreNet(gene_dim, hidden=hidden_dim)
        self.u_net = _ControlNet(gene_dim, hidden=hidden_dim)
    def forward(self, x, t, c=None):
        return self.score(x, t, c)
    @torch.no_grad()
    def sample(self, batch_size: int = 16, x_init: Optional[torch.Tensor] = None, condition=None):
        device = next(self.parameters()).device
        if x_init is None:
            x_init = torch.randn(batch_size, self.gene_dim, device=device)
        return reverse_sample(self.score, self.u_net, x_init, steps=self.num_timesteps, c=condition)



# ---- Optimal Transport utilities ----
def pairwise_cost(x: torch.Tensor, y: torch.Tensor, p: int = 2):
    """Compute pairwise cost matrix between two sets of points"""
    diff = x[:, None, :] - y[None, :, :]
    if p == 1:
        return diff.abs().sum(-1)
    return (diff.pow(2).sum(-1))

def sinkhorn(a, b, C, eps=0.01, n_iter=100):
    """Sinkhorn algorithm for entropic optimal transport"""
    K = torch.exp(-C / eps)
    u = torch.ones_like(a)
    v = torch.ones_like(b)
    for _ in range(n_iter):
        u = a / (K @ v + 1e-12)
        v = b / (K.transpose(0,1) @ u + 1e-12)
    P = torch.diag(u) @ K @ torch.diag(v)
    return P

