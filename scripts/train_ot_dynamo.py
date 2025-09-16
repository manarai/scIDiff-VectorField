# scripts/train_ot_dynamo.py
# scIDiff+ trainer: DSM (score), Dynamo guidance, OT regularization, optional pathway priors
# -------------------------------------------------------------------------------
# Usage (demo with dummy data):
#   python scripts/train_ot_dynamo.py \
#       --x-dim 128 --batch-size 128 --epochs 3 \
#       --ot-weight 0.1 --use-dynamo
#
# Replace `dummy_loader(...)` with your real dataloader returning
# (x_control, cond_vector_or_empty, x_drug).
# -------------------------------------------------------------------------------

import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from models.control_sde import (
    ControlNet,
    dsm_loss,
    control_cost,
    reverse_sample,   # for quick OT-guided sampling
)
from models.dynamo_field import (
    DynamoAdapter,
    make_lambda_schedule,
)
from models.ot_guidance import minibatch_ot_loss

# Optional pathway priors (create models/pathway_field.py from earlier snippet)
try:
    from models.pathway_field import PathwayField
    HAS_PATH = True
except Exception:
    HAS_PATH = False
    PathwayField = None


# -------- Minimal score net stub (replace with your project model/import) --------
class ScoreNet(nn.Module):
    def __init__(self, x_dim, cond_dim=0, hidden=1024):
        super().__init__()
        self.cond_dim = cond_dim
        self.fc_t = nn.Linear(128, 256)
        self.fc_c = nn.Linear(cond_dim, 256) if cond_dim > 0 else None
        self.net = nn.Sequential(
            nn.Linear(x_dim + 256 + (256 if cond_dim > 0 else 0), hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, x_dim),
        )

    def time_embed(self, t, dim=128):
        device, half = t.device, dim // 2
        # log-spaced Fourier features
        freqs = torch.exp(torch.linspace(0, 6, half, device=device))
        ang = t[:, None] * freqs[None, :] * 2 * math.pi
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)

    def forward(self, x, t, c=None):
        te = self.fc_t(self.time_embed(t))
        if self.cond_dim > 0 and c is not None:
            ce = self.fc_c(c)
            h = torch.cat([x, te, ce], dim=-1)
        else:
            h = torch.cat([x, te], dim=-1)
        return self.net(h)


# -------- Dummy loader (swap with real dataset) ---------------------------------
def dummy_loader(x_dim=128, cond_dim=0, n=4096, batch=128):
    """
    Returns batches of (x_control, cond, x_drug).
    - x_drug is a shifted/noisy version of x_control (toy target for OT).
    """
    x0 = torch.randn(n, x_dim)
    x1 = x0 + 0.5 + 0.1 * torch.randn_like(x0)  # pretend "drug" endpoint
    if cond_dim > 0:
        c = torch.zeros(n, cond_dim)
        c[:, 0] = 1.0
    else:
        c = torch.zeros(n, 0)  # keep shape logic simple
    ds = TensorDataset(x0, c, x1)
    return DataLoader(ds, batch_size=batch, shuffle=True, drop_last=True)


# -------- Pathway directional loss (soft prior) ---------------------------------
def pathway_direction_loss(x_now, x_prev, gene_idx, sign=+1.0, weight=0.1):
    """
    Encourage movement of pathway genes along expected direction (sign ∈ {+1, -1} or real).
    Uses a simple delta surrogate between two tensors in the same space.
    """
    if gene_idx.numel() == 0:
        return torch.tensor(0.0, device=x_now.device)
    delta = (x_now[:, gene_idx] - x_prev[:, gene_idx]).mean(dim=0)  # [|G|]
    # Penalize movement opposite to expected sign
    return weight * torch.relu(-sign * delta.mean())


# -------- Main ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Train scIDiff+ with Dynamo/OT/Pathway priors")
    # Data / model dims
    ap.add_argument("--x-dim", type=int, default=128, help="Gene/features dimension")
    ap.add_argument("--cond-dim", type=int, default=0, help="Condition vector dim (0 for none)")
    # Training
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    # Regularization weights
    ap.add_argument("--ot-weight", type=float, default=0.1, help="Weight for OT minibatch loss")
    ap.add_argument("--u-cost", type=float, default=1e-3, help="Control effort (SB) weight")
    ap.add_argument("--path-weight", type=float, default=0.0, help="Extra pathway directional loss weight")
    # Dynamo & pathway toggles
    ap.add_argument("--use-dynamo", action="store_true", help="Enable Dynamo vector-field alignment in guidance")
    ap.add_argument("--use-pathway", action="store_true", help="Enable pathway priors (requires models/pathway_field.py)")
    # Lambda schedules (start high, anneal near t->0)
    ap.add_argument("--dyn-start", type=float, default=1.0)
    ap.add_argument("--dyn-end", type=float, default=0.1)
    ap.add_argument("--path-start", type=float, default=0.8)
    ap.add_argument("--path-end", type=float, default=0.0)
    # Misc
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = args.device

    # TODO: Replace with your real dataloader returning (x_control, cond, x_drug)
    loader = dummy_loader(args.x_dim, args.cond_dim, batch=args.batch_size)

    # Score / control networks
    score = ScoreNet(args.x_dim, args.cond_dim).to(device)
    u_net = ControlNet(args.x_dim, args.cond_dim).to(device)
    opt = torch.optim.AdamW(list(score.parameters()) + list(u_net.parameters()), lr=args.lr)

    # Dynamo setup
    dyn = DynamoAdapter(callable_field=None, device=device)  # replace callable_field with your Dynamo function
    lam_dyn = make_lambda_schedule("cosine", start=args.dyn-start if hasattr(args, "dyn-start") else args.dyn_start,
                                   end=args.dyn-end if hasattr(args, "dyn-end") else args.dyn_end)
    # NOTE: argparse attributes can't contain '-', but we guard it just in case.

    # Optional pathway field
    if args.use_pathway and not HAS_PATH:
        raise RuntimeError("use_pathway=True but models/pathway_field.py not found. Create it first.")
    pf = None
    lam_path = None
    if args.use_pathway and HAS_PATH:
        # You must construct these for your dataset:
        # gene_sets: dict{name -> LongTensor(indices)}
        # effects:  dict{drug_name -> dict{name -> sign/weight}}
        # For the dummy run, we create empty sets (no-op); replace with real ones.
        gene_sets = {}
        effects = {}
        pf = PathwayField(gene_sets, effects, scale=0.05, device=device)
        lam_path = make_lambda_schedule("cosine", start=args.path_start, end=args.path_end)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        for x0, c, x1 in loader:
            x0, x1 = x0.to(device), x1.to(device)
            c = c.to(device) if (c is not None and c.numel() > 0) else None

            # 1) Score loss (DSM)
            loss_score, aux = dsm_loss(score, x0, c)
            xt, t = aux["xt"], aux["t"]

            # 2) Vector-field alignment (Dynamo → control net)
            if args.use_dynamo:
                lam_t = lam_dyn(float(t.mean().item()))
                vf = dyn.predict(xt)  # f(x_t)
                vf_loss = F.mse_loss(u_net(xt, c), lam_t * vf)
            else:
                vf_loss = torch.tensor(0.0, device=device)

            # 3) Control effort (Schrödinger-bridge flavor)
            u_cost = control_cost(u_net, xt, c, w=args.u_cost)

            # 4) Quick reverse samples for OT loss (and for pathway delta surrogate if used)
            with torch.no_grad():
                x_gen = reverse_sample(
                    score_net=score,
                    u_net=u_net,
                    x_init=x0,
                    c_drug=c,
                    n_steps=64,  # short path for OT signal (keep training fast)
                    lam_sched=lam_dyn if args.use_dynamo else None,
                    f_dyn=dyn.predict if args.use_dynamo else None,
                )

            # 5) OT endpoint matching
            ot_loss = minibatch_ot_loss(x_gen, x1, eps=0.05, p=2, iters=50)

            # 6) (Optional) Pathway directional loss on noised states (cheap surrogate)
            if args.use_pathway and pf is not None and len(pf.gene_sets) > 0:
                path_loss_total = torch.tensor(0.0, device=device)
                # Example: iterate declared gene sets and drug effect signs
                # Replace "drug_name" with your conditioning metadata if available
                drug_name = "drug"  # placeholder key into pf.effects
                for pname, idx in pf.gene_sets.items():
                    sign = 0.0
                    if drug_name in pf.effects and pname in pf.effects[drug_name]:
                        sign = float(pf.effects[drug_name][pname])
                    if sign == 0.0:
                        continue
                    # Encourage xt to move relative to x0 in expected direction for this pathway
                    path_loss_total = path_loss_total + pathway_direction_loss(
                        x_now=xt, x_prev=x0, gene_idx=idx.to(device), sign=sign, weight=args.path_weight
                    )
            else:
                path_loss_total = torch.tensor(0.0, device=device)

            # 7) Total loss
            loss = loss_score + vf_loss + u_cost + args.ot_weight * ot_loss + path_loss_total
            opt.zero_grad()
            loss.backward()
            opt.step()

        print(
            f"[epoch {epoch}] "
            f"score={loss_score.item():.4f}  "
            f"vf={vf_loss.item():.4f}  "
            f"u_cost={u_cost.item():.4f}  "
            f"ot={ot_loss.item():.4f}  "
            f"path={path_loss_total.item():.4f}"
        )


if __name__ == "__main__":
    main()
