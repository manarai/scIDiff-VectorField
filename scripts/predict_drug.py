import argparse, numpy as np, torch
from models.control_sde import reverse_sample, ControlNet
from models.dynamo_field import make_lambda_schedule, DynamoAdapter

# Replace with your real score net import/checkpoint restore.
class ScoreNet(torch.nn.Module):
    def __init__(self, x_dim, cond_dim=0, hidden=1024):
        super().__init__()
        self.cond_dim = cond_dim
        self.fc_t = torch.nn.Linear(128, 256)
        self.fc_c = torch.nn.Linear(cond_dim, 256) if cond_dim>0 else None
        self.net = torch.nn.Sequential(
            torch.nn.Linear(x_dim + 256 + (256 if cond_dim>0 else 0), hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, x_dim)
        )
    def time_embed(self, t, dim=128):
        device, half = t.device, dim//2
        freqs = torch.exp(torch.linspace(0, 6, half, device=device))
        ang = t[:, None] * freqs[None, :] * 2 * torch.pi
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
    def forward(self, x, t, c=None):
        te = self.fc_t(self.time_embed(t))
        if self.cond_dim>0 and c is not None:
            ce = self.fc_c(c)
            h = torch.cat([x, te, ce], dim=-1)
        else:
            h = torch.cat([x, te], dim=-1)
        return self.net(h)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x-path", type=str, required=True, help="Numpy .npy with CONTROL cells [N,D]")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--cond-dim", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = args.device
    X = torch.tensor(np.load(args.x_path), dtype=torch.float32, device=device)
    N, D = X.shape
    C = torch.zeros(N, args.cond_dim, device=device) if args.cond_dim>0 else None

    score = ScoreNet(D, args.cond_dim).to(device)
    u_net = ControlNet(D, args.cond_dim).to(device)
    # TODO: load real checkpoints here

    lam_sched = make_lambda_schedule("cosine", 1.0, 0.1)
    dyn = DynamoAdapter(callable_field=None, device=device)  # wire your Dynamo call here

    X_drug = reverse_sample(score, u_net, X, C, n_steps=args.steps, lam_sched=lam_sched, f_dyn=dyn.predict)
    np.save("predicted_drug.npy", X_drug.detach().cpu().numpy())
    print("Saved: predicted_drug.npy")

if __name__ == "__main__":
    main()
PY
