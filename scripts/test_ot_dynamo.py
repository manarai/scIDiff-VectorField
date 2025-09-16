import torch, torch.nn as nn
from models.control_sde import ControlNet, dsm_loss, reverse_sample
from models.ot_guidance import minibatch_ot_loss
from models.dynamo_field import DynamoAdapter, make_lambda_schedule

class DummyScore(nn.Module):
    def forward(self, x, t, c=None):
        return torch.zeros_like(x)

def test_ot_loss_scalar():
    xg = torch.randn(16, 64)
    xt = torch.randn(12, 64)
    L = minibatch_ot_loss(xg, xt)
    assert L.ndim == 0

def test_reverse_sample_shapes():
    score = DummyScore()
    u = ControlNet(64, 0)
    x0 = torch.randn(8, 64)
    c = None
    lam = make_lambda_schedule()
    dyn = DynamoAdapter(lambda x: torch.zeros_like(x)).predict
    x1 = reverse_sample(score, u, x0, c, n_steps=8, lam_sched=lam, f_dyn=dyn)
    assert x1.shape == x0.shape
PY
