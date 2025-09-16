# models/control_sde.py
# Reverse SDE with vector-field guidance (Dynamo) + optional SB/OT regularization.

from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F
from typing import Optional, Callable
from .dynamo_field import DynamoAdapter, make_lambda_schedule

def beta_t(t, beta_min=0.1, beta_max=20.0):
    return beta_min + t * (beta_max - beta_min)

def marginal_std(t):
    # coarse VP-SDE variance schedule approximation (works fine in practice)
    return torch.sqrt(1.0 - torch.exp(-(0.5 * (0.1 + 19.9 * t) * t)))

class ControlNet(nn.Module):
    """
    Learnable control uθ(x,c) that we align to Dynamo f(x).
    If you prefer fixed guidance = λ(t)*f(x), you can skip this and add the field directly.
    """
    def __init__(self, x_dim: int, cond_dim: int = 0, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim + cond_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, x_dim)
        )
        self.cond_dim = cond_dim

    def forward(self, x: torch.Tensor, c: Optional[torch.Tensor] = None):
        if c is not None and self.cond_dim > 0:
            h = torch.cat([x, c], dim=-1)
        else:
            h = x
        return self.net(h)

def dsm_loss(score_net, x0, c=None):
    B = x0.size(0)
    t = torch.rand(B, device=x0.device)  # U[0,1]
    sigma = marginal_std(t).unsqueeze(-1)
    eps = torch.randn_like(x0)
    xt = torch.sqrt(1 - sigma**2) * x0 + sigma * eps
    s_pred = score_net(xt, t, c)
    target = -eps / (sigma + 1e-7)
    return F.mse_loss(s_pred, target), {"t": t, "xt": xt}

def vf_align_loss(u_net, x, c, f_dyn, lam_t: Callable[[torch.Tensor], torch.Tensor]):
    with torch.no_grad():
        f = f_dyn(x)
    u = u_net(x, c)
    lam = lam_t.view(-1, 1) if isinstance(lam_t, torch.Tensor) else lam_t
    if callable(lam):
        # lam takes scalar or tensor t → scalar
        # here we broadcast a single scalar lam for the batch
        lam_val = lam( torch.tensor(0.5, device=x.device) )  # default mid-t if none provided
        lam = lam_val.view(1,1)
    return F.mse_loss(u, lam * f)

def control_cost(u_net, x, c, w=1e-3):
    return w * (u_net(x, c).pow(2).mean())

@torch.no_grad()
def reverse_sample(score_net, u_net, x_init, c_drug, n_steps=1000, lam_sched=None, f_dyn=None):
    x = x_init.clone()
    ts = torch.linspace(1., 0., n_steps, device=x.device)
    for i in range(n_steps-1):
        t = ts[i].expand(x.size(0))
        bt = beta_t(t)
        score = score_net(x, t, c_drug)
        guide = u_net(x, c_drug) if u_net is not None else 0.0
        if f_dyn is not None and lam_sched is not None:
            lam = lam_sched(float(t[0]))
            guide = guide + lam * f_dyn(x)
        drift = -(bt/2).unsqueeze(-1)*x - bt.unsqueeze(-1)*score + guide
        dt = (ts[i+1] - ts[i]).item()
        noise = torch.sqrt(torch.clamp(bt, min=1e-8)).unsqueeze(-1) * torch.randn_like(x)
        x = x + drift * dt + noise * (abs(dt) ** 0.5)
    return x
