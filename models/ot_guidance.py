# models/ot_guidance.py
# Entropic OT / Schrödinger-bridge style regularization utilities (minibatch Sinkhorn).

from __future__ import annotations
import torch
import torch.nn.functional as F

def pairwise_cost(x: torch.Tensor, y: torch.Tensor, p: int = 2):
    # x: [B, D], y: [B, D] (or [B2,D]) → cost matrix [B,B2]
    # L_p cost; default squared Euclidean (p=2)
    diff = x[:, None, :] - y[None, :, :]
    if p == 1:
        return diff.abs().sum(-1)
    return (diff.pow(2).sum(-1))  # p=2

def sinkhorn(a, b, C, eps=0.01, n_iter=100):
    # a: [B], b: [B2], histograms sum to 1
    K = torch.exp(-C / eps)  # [B,B2]
    u = torch.ones_like(a)
    v = torch.ones_like(b)
    for _ in range(n_iter):
        u = a / (K @ v + 1e-12)
        v = b / (K.transpose(0,1) @ u + 1e-12)
    P = torch.diag(u) @ K @ torch.diag(v)
    return P  # transport plan

@torch.no_grad()
def minibatch_ot_loss(x_gen: torch.Tensor, x_target: torch.Tensor, eps=0.05, p=2, iters=50):
    """
    OT distance between generated samples and target minibatch (drug state).
    Returns entropic OT objective ~ ⟨P, C⟩.
    """
    B = x_gen.size(0)
    B2 = x_target.size(0)
    a = torch.full((B,), 1.0/B, device=x_gen.device)
    b = torch.full((B2,), 1.0/B2, device=x_gen.device)
    C = pairwise_cost(x_gen, x_target, p=p)
    P = sinkhorn(a, b, C, eps=eps, n_iter=iters)
    return (P * C).sum()
