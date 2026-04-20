"""
train_resac_offline.py
======================
Offline RL variant of RE-SAC: ensemble-Q SAC with two uncertainty regularizers,
trained purely on offline data (no environment rollouts).

RE-SAC's two regularization terms, adapted for offline RL:

1. **Aleatoric uncertainty regularization** (network-complexity)
   - Sum of L1 norms of each critic's weights and biases → per-ensemble-member
     regularizer `reg_norm ∈ R^E`.
   - Added to target Q bootstrap: `Q_target ← Q_target + weight_reg * reg_norm`.
   - Penalizes overly-complex critics that overfit to offline data noise.
   - Origin: RNAC (Robust Natural Actor-Critic) network-complexity term.

2. **Epistemic uncertainty regularization** (ensemble Q-std pessimism)
   - Policy loss: `-(Q_mean + β * Q_std)` with β < 0  → LCB pessimism.
     Policy prefers actions where ensemble agrees (low std) and Q is high.
     In offline RL, this is exactly what's needed: penalize OOD actions where
     ensemble disagreement is naturally high.
   - Inspired by RORL / EDAC / PBRL.

Plus two offline-specific helpers:
- **Behavior cloning regularizer** (β_bc * MSE between policy and dataset action):
  stabilizes early training, standard in TD3+BC.
- **OOD Q-std regularizer in critic loss** (β_ood * std(Q_pred)):
  prevents ensemble std from collapsing on in-distribution samples.

Reference papers:
- RE-SAC:  /home/erzhu419/mine_code/RE-SAC/sac_ensemble_original_logging.py
- RNAC:    /home/erzhu419/mine_code/RNAC/
- RORL:    /home/erzhu419/mine_code/RORL/lifelong_rl/trainers/q_learning/sac.py

Run:
    cd /home/erzhu419/mine_code/offline-sumo
    conda run -n LSTM-RL python train_resac_offline.py --n_steps 100000 --device cuda
"""

import os, sys, csv, json, time, argparse, datetime, math
import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy

# ── Path setup ──────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
for p in [_HERE, os.path.join(_HERE, "env"), os.path.join(_HERE, "agents"),
          os.path.join(_HERE, "buffers"), os.path.join(_HERE, "utils")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from model import (
    EmbeddingLayer, BusEmbeddingPolicy, BusEnsembleCritic,
    BusSamplerPolicy, Scalar, soft_target_update,
)
from bus_replay_buffer import BusMixedReplayBuffer
from bus_sampler import BusEvalSampler
from envs.sumo_gym_env import SumoGymEnv
from common.data_utils import set_route_length, build_edge_linear_map

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Args ────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--seed",          type=int,   default=42)
parser.add_argument("--device",        type=str,   default="cpu")
parser.add_argument("--n_steps",       type=int,   default=100000)
parser.add_argument("--batch_size",    type=int,   default=1024)
parser.add_argument("--ensemble_size", type=int,   default=10)
parser.add_argument("--beta",          type=float, default=-2.0,
                    help="epistemic (Q-std) weight in policy loss — negative = pessimistic")
parser.add_argument("--beta_ood",      type=float, default=0.01,
                    help="OOD Q-std penalty in critic loss")
parser.add_argument("--beta_bc",       type=float, default=0.005,
                    help="behavior cloning weight in policy loss")
parser.add_argument("--weight_reg",    type=float, default=1e-5,
                    help="aleatoric (L1 weight norm) term weight")
parser.add_argument("--lr",            type=float, default=3e-4)
parser.add_argument("--eval_every",    type=int,   default=2500)
parser.add_argument("--ckpt_every",    type=int,   default=10000)
parser.add_argument("--n_eval",        type=int,   default=10)
parser.add_argument("--dataset_file",  type=str,
                    default=os.path.join(_HERE, "data", "datasets_v2", "merged_all_v2.h5"))
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

_ts = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
out_dir = os.path.join(_HERE, "experiment_output", f"resac_offline_seed{args.seed}_{_ts}")
os.makedirs(out_dir, exist_ok=True)
print(f"Output: {out_dir}")

# ── Route ───────────────────────────────────────────────────────────────────
edge_xml = os.path.join(_HERE, "env", "network_data", "a_sorted_busline_edge.xml")
if os.path.exists(edge_xml):
    edge_map = build_edge_linear_map(edge_xml, "7X")
    route_length = max(edge_map.values()) if edge_map else 13119.0
else:
    route_length = 13119.0
set_route_length(route_length)

# ── Networks ────────────────────────────────────────────────────────────────
cat_cols = ["line_id", "bus_id", "station_id", "time_period", "direction"]
cat_code_dict = {
    "line_id":     {i: i for i in range(12)},
    "bus_id":      {i: i for i in range(389)},
    "station_id":  {i: i for i in range(1)},
    "time_period": {i: i for i in range(1)},
    "direction":   {0: 0, 1: 1},
}
obs_dim, action_dim, hidden_sz = 17, 2, 48
E = args.ensemble_size

emb_tmpl  = EmbeddingLayer(cat_code_dict, cat_cols, layer_norm=True, dropout=0.05)
state_dim = emb_tmpl.output_dim + (obs_dim - len(cat_cols))

policy    = BusEmbeddingPolicy(state_dim, action_dim, hidden_sz, emb_tmpl.clone(), action_range=1.0)
qfs       = BusEnsembleCritic(state_dim, action_dim, hidden_sz, E, emb_tmpl.clone())
target_qfs= deepcopy(qfs)
for p in target_qfs.parameters():
    p.requires_grad_(False)
print(f"RE-SAC (offline): ensemble_size={E}, β={args.beta}, β_ood={args.beta_ood}, "
      f"β_bc={args.beta_bc}, weight_reg={args.weight_reg}")

dev = torch.device(args.device)
for m in [policy, qfs, target_qfs]:
    m.to(dev)

# ── Temperature α ───────────────────────────────────────────────────────────
INIT_LOG_ALPHA = -2.3
TARGET_ENTROPY = -0.21
MAX_ALPHA      = 0.6
log_alpha = Scalar(INIT_LOG_ALPHA).to(dev)

# ── Optimizers ──────────────────────────────────────────────────────────────
policy_optim = torch.optim.Adam(policy.parameters(), lr=args.lr)
qf_optim     = torch.optim.Adam(qfs.parameters(), lr=args.lr)
alpha_optim  = torch.optim.Adam(log_alpha.parameters(), lr=args.lr)

# ── Buffer (offline only) ───────────────────────────────────────────────────
print(f"Loading offline data: {args.dataset_file}")
buffer = BusMixedReplayBuffer(
    state_dim=obs_dim, action_dim=action_dim, context_dim=30,
    dataset_file=args.dataset_file, device=args.device,
    buffer_ratio=0.01,
    reward_scale=1.0, reward_bias=0.0,
)
print(f"Offline data: {buffer.fixed_dataset_size:,} transitions")
reward_mean, reward_std = buffer.get_reward_stats()
print(f"Offline reward stats: mean={reward_mean:.2f}, std={reward_std:.2f}")

# ── Eval env (SUMO) ─────────────────────────────────────────────────────────
_SUMO_DIR = os.path.join(
    os.path.dirname(_HERE), "sumo-rl",
    "_standalone_f543609", "SUMO_ruiguang", "online_control",
)
eval_env = SumoGymEnv(sumo_dir=_SUMO_DIR, edge_xml=edge_xml, max_steps=18000, line_id="7X")
eval_sampler = BusEvalSampler(eval_env)
sampler_policy = BusSamplerPolicy(policy, args.device)


REWARD_SCALE = 10.0
def normalize_rewards(r):
    return (r - reward_mean) / (reward_std + 1e-6) * REWARD_SCALE


# ── RE-SAC's aleatoric (L1 weight norm) per ensemble member ────────────────
def compute_reg_norm():
    """Sum of L1 norms of critic weights and biases per ensemble member, shape [E].

    Only VectorizedLinear params (.w and .b) have shape starting with E;
    LayerNorm params are shared across ensemble and excluded.
    """
    parts = []
    for name, param in qfs.named_parameters():
        last = name.split('.')[-1]
        if last in ('w', 'b') and param.dim() >= 2 and param.shape[0] == E:
            parts.append(torch.norm(param.reshape(E, -1), p=1, dim=1))
    return torch.stack(parts, dim=0).sum(dim=0)  # [E]


# ── Training step ───────────────────────────────────────────────────────────
DISCOUNT = 0.80
SOFT_TAU = 5e-3

def train_step():
    batch = buffer.sample(args.batch_size, scope="real")
    obs      = batch["observations"]
    actions  = batch["actions"]
    rewards  = normalize_rewards(batch["rewards"].squeeze())  # [B]
    next_obs = batch["next_observations"]
    dones    = batch["dones"].squeeze()  # [B]

    alpha = log_alpha().exp().detach().clamp(max=MAX_ALPHA)
    # Aleatoric reg: WITH gradient for L1 penalty on critic weights
    reg = compute_reg_norm()  # [E]

    # ── Q target (uses reg as constant shift, detached) ──
    with torch.no_grad():
        new_next_a, next_lp = policy(next_obs)
        next_lp = next_lp.squeeze(-1)  # [B]
        target_q = target_qfs(next_obs, new_next_a)  # [E, B]
        target_q = target_q - alpha * next_lp.unsqueeze(0) + args.weight_reg * reg.detach().unsqueeze(1)
        td_target = rewards.unsqueeze(0) + (1 - dones).unsqueeze(0) * DISCOUNT * target_q  # [E, B]

    # ── Q loss: TD + OOD std + explicit L1 (aleatoric) ──
    q_pred = qfs(obs, actions)  # [E, B]
    ood_loss = q_pred.std(dim=0).mean()
    l1_reg_loss = args.weight_reg * reg.sum()  # L1 penalty on critic weights
    q_loss = F.mse_loss(q_pred, td_target.detach()) + args.beta_ood * ood_loss + l1_reg_loss

    qf_optim.zero_grad()
    q_loss.backward()
    torch.nn.utils.clip_grad_norm_(qfs.parameters(), 1.0)
    qf_optim.step()

    # ── Policy loss (LCB pessimism + BC) ──
    # reg is detached here — it's only a constant shift on Q, no policy gradient
    with torch.no_grad():
        reg_p = compute_reg_norm()
    new_a, lp = policy(obs)
    lp = lp.squeeze(-1)
    q_new = qfs(obs, new_a) + args.weight_reg * reg_p.unsqueeze(1) - alpha * lp.unsqueeze(0)
    q_mean = q_new.mean(dim=0)
    q_std  = q_new.std(dim=0)
    lcb_loss = -(q_mean + args.beta * q_std).mean()
    bc_loss  = F.mse_loss(new_a, actions)
    policy_loss = args.beta_bc * bc_loss + lcb_loss

    policy_optim.zero_grad()
    policy_loss.backward()
    policy_optim.step()

    # ── Alpha ──
    with torch.no_grad():
        _, lp2 = policy(obs)
    alpha_loss = -(log_alpha() * (lp2.detach().squeeze(-1) + TARGET_ENTROPY)).mean()
    alpha_optim.zero_grad()
    alpha_loss.backward()
    alpha_optim.step()

    # ── Target net ──
    soft_target_update(qfs, target_qfs, SOFT_TAU)

    return dict(
        q_loss=q_loss.item(), lcb_loss=lcb_loss.item(), bc_loss=bc_loss.item(),
        ood_loss=ood_loss.item(), q_mean=q_mean.mean().item(), q_std=q_std.mean().item(),
        alpha=log_alpha().exp().item(), reg_norm=reg.mean().item(),
    )


# ── CSV logger ──────────────────────────────────────────────────────────────
csv_path = os.path.join(out_dir, "train_log.csv")
csv_f = open(csv_path, "w", newline="")
csv_w = csv.writer(csv_f)
csv_w.writerow(["step", "q_loss", "lcb_loss", "bc_loss", "ood_loss",
                "q_mean", "q_std", "alpha", "reg_norm", "eval_return", "wall_sec"])

# ── Training loop ───────────────────────────────────────────────────────────
print(f"\nRE-SAC offline training: {args.n_steps} steps, batch={args.batch_size}")
t0 = time.time()
eval_return = 0.0
history = []

for step in range(1, args.n_steps + 1):
    losses = train_step()

    if step % args.eval_every == 0:
        trajs = eval_sampler.sample(sampler_policy, n_trajs=2, deterministic=True)
        eval_return = float(np.mean([sum(t["rewards"]) for t in trajs])) if trajs else 0.0
        print(f"  [Step {step:6d}/{args.n_steps}]  "
              f"q_loss={losses['q_loss']:.2f}  "
              f"lcb={losses['lcb_loss']:.2f}  "
              f"bc={losses['bc_loss']:.3f}  "
              f"ood={losses['ood_loss']:.3f}  "
              f"q_std={losses['q_std']:.2f}  "
              f"α={losses['alpha']:.3f}  "
              f"eval={eval_return:.0f}  "
              f"t={time.time()-t0:.0f}s")
        history.append(dict(step=step, eval_return=eval_return, **losses))

    csv_w.writerow([step, losses['q_loss'], losses['lcb_loss'], losses['bc_loss'],
                    losses['ood_loss'], losses['q_mean'], losses['q_std'],
                    losses['alpha'], losses['reg_norm'], eval_return, time.time() - t0])
    if step % 100 == 0:
        csv_f.flush()

    if step % args.ckpt_every == 0:
        torch.save(dict(policy=policy.state_dict(), qfs=qfs.state_dict(), step=step),
                   os.path.join(out_dir, f"checkpoint_step{step}.pt"))

csv_f.close()

# ── Final model save ────────────────────────────────────────────────────────
torch.save(dict(policy=policy.state_dict(), qfs=qfs.state_dict(), step=args.n_steps),
           os.path.join(out_dir, "model_final.pt"))
print(f"\nFinal model saved → {out_dir}/model_final.pt")

# ── Final eval (statistics) ────────────────────────────────────────────────
print(f"\nFinal eval: {args.n_eval} episodes...")
returns = []
for i in range(args.n_eval):
    trajs = eval_sampler.sample(sampler_policy, n_trajs=1, deterministic=True)
    ret = sum(trajs[0]["rewards"])
    returns.append(ret)
    print(f"  Episode {i+1}/{args.n_eval}: return={ret:.1f}")
returns = np.array(returns)
result = {
    "method": "RE-SAC (offline)",
    "seed": args.seed, "n_eval": args.n_eval,
    "mean_return": float(returns.mean()),
    "std_return":  float(returns.std()),
    "min_return":  float(returns.min()),
    "max_return":  float(returns.max()),
    "all_returns": returns.tolist(),
}
with open(os.path.join(out_dir, "eval_result.json"), "w") as f:
    json.dump(result, f, indent=2)

print(f"\n{'='*50}")
print(f"RE-SAC offline eval: {result['mean_return']:.1f} ± {result['std_return']:.1f}")
print(f"  min={result['min_return']:.1f}, max={result['max_return']:.1f}")
print(f"{'='*50}")

# ── Plot ────────────────────────────────────────────────────────────────────
if history:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    steps = [h["step"] for h in history]
    axes[0].plot(steps, [h["eval_return"] for h in history], "b-o", ms=4)
    axes[0].set_title("Eval Return"); axes[0].set_xlabel("Step")
    axes[1].plot(steps, [h["q_std"] for h in history], "r-", label="Q-std")
    axes[1].set_title("Ensemble Q-std"); axes[1].set_xlabel("Step")
    axes[2].plot(steps, [h["bc_loss"] for h in history], "g-", label="BC")
    axes[2].plot(steps, [h["ood_loss"] for h in history], "m-", label="OOD")
    axes[2].set_title("Regularizers"); axes[2].set_xlabel("Step"); axes[2].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_curve.png"), dpi=150)
    plt.close()

print(f"Done in {time.time()-t0:.0f}s.  Log: {csv_path}")
