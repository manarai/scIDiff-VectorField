# scIDiff+: OT & Dynamo Guidance Extension

This add-on extends [scIDiff](https://github.com/manarai/scIDiff) with:

- **Dynamo vector-field guidance** (RNA velocity–based global dynamics)  
- **Learnable control network** \(u_\theta\)  
- **Entropic optimal transport (OT)** regularization for endpoint matching  

The goal: make scIDiff a **trajectory-aware simulator** for perturbation prediction in single-cell omics.

---

## Mathematical Model

We model expression states \(x \in \mathbb{R}^d\) with **score-based diffusion** regularized by biological priors.

### 1. Forward (noising) SDE
$$
dx_t = -\tfrac{1}{2}\beta(t)x_t\,dt + \sqrt{\beta(t)}\,dW_t, 
\quad x_0 \sim p_0(x \mid c)
$$

### 2. Reverse (denoising) SDE with guidance
$$
dx_t = \Big(-\tfrac{1}{2}\beta(t)x_t - \beta(t)\,s_\theta(x,t,c) 
+ u_\theta(x,t,c) + \lambda(t)\,f(x)\Big)dt 
+ \sqrt{\beta(t)}\,d\bar W_t
$$

- \(s_\theta(x,t,c)\): score network  
- \(u_\theta(x,t,c)\): learnable control, aligned with Dynamo  
- \(f(x)\): Dynamo vector field (RNA velocity–derived drift)  
- \(\lambda(t)\): guidance schedule (strong early, weak near manifold)  

### 3. Training Objective
$$
\begin{aligned}
\mathcal{L} &= 
\underbrace{\mathcal{L}_{\text{score}}}_{\text{DSM}}
+ \gamma \,\|u_\theta - \lambda(t) f(x)\|^2 \\
&\quad + \eta \,\|u_\theta\|^2
+ \tau \,\mathcal{L}_{\text{OT}}(p_0 \!\to\! p_1)
\end{aligned}
$$

- **Score loss**: denoising score matching  
- **Vector-field alignment**: match \(u_\theta\) to Dynamo prior  
- **Control cost**: Schrödinger bridge penalty  
- **OT regularization**: entropic Sinkhorn distance between generated & target distributions  

---

## Installation
```bash
git clone https://github.com/manarai/scIDiff.git
cd scIDiff
pip install -e .
```

Dependencies:  
- `torch`, `numpy`, `scipy`  
- `dynamo-release` (for vector-field fitting)  

---

## Training (demo with dummy data)
```bash
python scripts/train_ot_dynamo.py   --x-dim 128   --batch-size 128   --epochs 3   --ot-weight 0.1   --use-dynamo
```

- `--ot-weight`: weight for OT regularization (default `0.1`)  
- `--use-dynamo`: enables Dynamo alignment  

Replace the dummy dataloader in `scripts/train_ot_dynamo.py` with your real `(x_control, cond, x_drug)` loader.

---

## Sampling (control → drug)
```bash
python scripts/predict_drug.py   --x-path CONTROL.npy   --steps 1000
```

Produces `predicted_drug.npy`.

---

## Dynamo Integration

Wrap Dynamo’s vector field:
```python
from models.dynamo_field import DynamoAdapter
dynamo = DynamoAdapter(callable_field=lambda x: torch.tensor(
    dyn.vf(x.cpu().numpy()), device=x.device, dtype=x.dtype
))
```

Use in sampler:
```python
from models.control_sde import reverse_sample
from models.dynamo_field import make_lambda_schedule

x_pred = reverse_sample(
    score_net, u_net, x_control, c_drug,
    n_steps=1000,
    lam_sched=make_lambda_schedule("cosine", 1.0, 0.1),
    f_dyn=dynamo.predict
)
```

⚠️ **Normalization must match** between Dynamo and scIDiff (log1p CPM, z-score).

---

## Testing
```bash
pytest tests/test_ot_dynamo.py -q
```

---

## Roadmap
- ✅ OT + Dynamo integration  
- 🔜 Multi-condition conditioning (dose, time)  
- 🔜 Cytoscape/GraphML export  

---

## Citation
> Terooatea, T.W. *scIDiff+: Integrating Vector-Field Priors and Optimal Transport for Perturbation Prediction in Single-Cell Diffusion Models.* (2025)
