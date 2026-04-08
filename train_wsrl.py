"""
train_wsrl.py
=============
WSRL: Policy-dominant offline-to-online RL for SUMO bus holding.

- Warm-start from pretrained/offline_final.pt  (100K-step AWR offline pretraining)
- Online SAC fine-tuning with BusSimEnv (no offline data retained)
- Fresh replay buffer: only online sim transitions

Key difference from RLPD: NO offline data mixed in during online phase.
The offline contribution is entirely through the warm-started policy weights.

Reference: "Efficient Online RL Fine-Tuning Need Not Retain Offline Data" (wsrl/)
Algorithm: Online SAC with V-function (same architecture as pretrained model)

Run:
    cd /home/erzhu419/mine_code/offline-sumo
    conda run -n LSTM-RL python train_wsrl.py --n_epochs 500 --device cpu
"""

import os, sys, math, csv, time, argparse, datetime
import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy

# ── Path setup ──────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ENV  = os.path.join(_HERE, "env")
_AGENTS = os.path.join(_HERE, "agents")
_BUFFERS = os.path.join(_HERE, "buffers")
_UTILS = os.path.join(_HERE, "utils")
for p in [_HERE, _ENV, _AGENTS, _BUFFERS, _UTILS]:
    if p not in sys.path:
        sys.path.insert(0, p)

from model import (
    EmbeddingLayer, BusEmbeddingPolicy,
    BusEmbeddingQFunction, BusEmbeddingVFunction,
    BusSamplerPolicy, Scalar, soft_target_update,
)
from bus_replay_buffer import BusMixedReplayBuffer
from bus_sampler import BusStepSampler, BusEvalSampler
from envs.bus_sim_env import BusSimEnv
from common.data_utils import set_route_length, build_edge_linear_map

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Args ────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--seed",        type=int,   default=42)
parser.add_argument("--device",      type=str,   default="cpu")
parser.add_argument("--n_epochs",    type=int,   default=500)
parser.add_argument("--batch_size",  type=int,   default=512)
parser.add_argument("--n_rollout",   type=int,   default=100,  help="decision events per epoch")
parser.add_argument("--n_train",     type=int,   default=100,  help="SAC updates per epoch")
parser.add_argument("--min_online",  type=int,   default=500,  help="minimum online transitions before training")
parser.add_argument("--utd",         type=int,   default=1,    help="update-to-data ratio")
parser.add_argument("--eval_every",  type=int,   default=10)
parser.add_argument("--ckpt_every",  type=int,   default=50)
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# ── Output dir ───────────────────────────────────────────────────────────────
_ts = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
out_dir = os.path.join(_HERE, "experiment_output", f"wsrl_seed{args.seed}_{_ts}")
os.makedirs(out_dir, exist_ok=True)
print(f"Output: {out_dir}")

# ── Route setup ──────────────────────────────────────────────────────────────
edge_xml = os.path.join(_ENV, "network_data", "a_sorted_busline_edge.xml")
if os.path.exists(edge_xml):
    edge_map = build_edge_linear_map(edge_xml, "7X")
    route_length = max(edge_map.values()) if edge_map else 13119.0
else:
    route_length = 13119.0
set_route_length(route_length)

# ── Network architecture (must match pretrained model) ───────────────────────
cat_cols = ["line_id", "bus_id", "station_id", "time_period", "direction"]
cat_code_dict = {
    "line_id":     {i: i for i in range(12)},
    "bus_id":      {i: i for i in range(389)},
    "station_id":  {i: i for i in range(1)},
    "time_period": {i: i for i in range(1)},
    "direction":   {0: 0, 1: 1},
}
obs_dim    = 17
action_dim = 2
hidden_sz  = 48

emb_tmpl  = EmbeddingLayer(cat_code_dict, cat_cols, layer_norm=True, dropout=0.05)
state_dim = emb_tmpl.output_dim + (obs_dim - len(cat_cols))

policy     = BusEmbeddingPolicy(state_dim, action_dim, hidden_sz, emb_tmpl.clone(), action_range=1.0)
qf1        = BusEmbeddingQFunction(state_dim, action_dim, hidden_sz, emb_tmpl.clone())
qf2        = BusEmbeddingQFunction(state_dim, action_dim, hidden_sz, emb_tmpl.clone())
target_qf1 = deepcopy(qf1)
target_qf2 = deepcopy(qf2)
vf         = BusEmbeddingVFunction(state_dim, hidden_sz, emb_tmpl.clone())

# ── Load pretrained weights (warm start) ─────────────────────────────────────
pretrained_path = os.path.join(_HERE, "pretrained", "offline_final.pt")
if os.path.exists(pretrained_path):
    ckpt = torch.load(pretrained_path, map_location="cpu")
    policy.load_state_dict(ckpt["policy"])
    qf1.load_state_dict(ckpt["qf1"])
    qf2.load_state_dict(ckpt["qf2"])
    target_qf1.load_state_dict(ckpt["qf1"])
    target_qf2.load_state_dict(ckpt["qf2"])
    vf.load_state_dict(ckpt["vf"])
    print(f"Loaded pretrained weights from {pretrained_path}")
else:
    print(f"WARNING: pretrained model not found at {pretrained_path}, using random init")

dev = torch.device(args.device)
for m in [policy, qf1, qf2, target_qf1, target_qf2, vf]:
    m.to(dev)

# ── Temperature (α) ──────────────────────────────────────────────────────────
INIT_LOG_ALPHA  = -2.3   # α ≈ 0.10, match offline pretraining
TARGET_ENTROPY  = -0.21
MAX_ALPHA       = 0.6
log_alpha = Scalar(INIT_LOG_ALPHA).to(dev)

# ── Optimizers ───────────────────────────────────────────────────────────────
LR = 3e-4
policy_optim = torch.optim.Adam(policy.parameters(), lr=LR)
qf_optim     = torch.optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=LR)
vf_optim     = torch.optim.Adam(vf.parameters(), lr=LR)
alpha_optim  = torch.optim.Adam(log_alpha.parameters(), lr=LR)

# ── Online-only replay buffer (no offline data) ───────────────────────────────
buffer = BusMixedReplayBuffer(
    state_dim=obs_dim, action_dim=action_dim, context_dim=30,
    dataset_file=None,   # WSRL: no offline data
    device=args.device,
    buffer_ratio=500,    # online buffer ~ 500K transitions
)
print(f"Online-only buffer capacity: {buffer.max_size:,}")

# ── Sim environment ───────────────────────────────────────────────────────────
sim_env_path = os.path.join(_ENV, "calibrated_env")
env = BusSimEnv(path=sim_env_path)

sampler_policy = BusSamplerPolicy(policy, args.device)
sampler = BusStepSampler(
    env=env,
    replay_buffer=buffer,
    max_traj_events=100,
    p_reset=0.0,          # WSRL: no snapshot reset (no offline snapshots)
    h_rollout=30,
    w_threshold=0.0,
    warmup_episodes=0,
    action_dim=action_dim,
)

eval_env = BusSimEnv(path=sim_env_path)
eval_sampler = BusEvalSampler(eval_env)


# ── SAC step (online, no offline data) ───────────────────────────────────────
DISCOUNT   = 0.80
SOFT_TAU   = 1e-2
QUANTILE   = 0.7   # upper quantile → slightly optimistic online V

def sac_step(batch):
    obs      = batch["observations"]
    actions  = batch["actions"]
    rewards  = batch["rewards"].squeeze()
    next_obs = batch["next_observations"]
    dones    = batch["dones"].squeeze()
    alpha    = log_alpha.exp().detach().clamp(max=MAX_ALPHA)

    # V update: V(s) ← quantile_reg(E_π[Q(s,·) - α·log_π])
    with torch.no_grad():
        pi_a, pi_lp = policy(obs)
        q_pi = torch.min(qf1(obs, pi_a), qf2(obs, pi_a)).squeeze()
        v_target = q_pi - alpha * pi_lp.squeeze()
    v_pred = vf(obs).squeeze()
    diff = v_target - v_pred
    w = torch.where(diff >= 0,
                    torch.full_like(diff, QUANTILE),
                    torch.full_like(diff, 1.0 - QUANTILE))
    vf_loss = (w * diff.abs()).mean()
    vf_optim.zero_grad(); vf_loss.backward(); vf_optim.step()

    # Q update: Q(s,a) ← r + γ·V(s')
    with torch.no_grad():
        td_target = rewards + (1.0 - dones) * DISCOUNT * vf(next_obs).squeeze()
    q1 = qf1(obs, actions).squeeze()
    q2 = qf2(obs, actions).squeeze()
    qf1_loss = F.mse_loss(q1, td_target)
    qf2_loss = F.mse_loss(q2, td_target)
    qf_optim.zero_grad()
    (qf1_loss + qf2_loss).backward()
    torch.nn.utils.clip_grad_norm_(list(qf1.parameters()) + list(qf2.parameters()), 1.0)
    qf_optim.step()
    soft_target_update(qf1, target_qf1, SOFT_TAU)
    soft_target_update(qf2, target_qf2, SOFT_TAU)

    # Policy update: maximize E_π[Q - α·log_π]
    pi_a2, pi_lp2 = policy(obs)
    q_pi2 = torch.min(qf1(obs, pi_a2), qf2(obs, pi_a2)).squeeze()
    policy_loss = (alpha * pi_lp2.squeeze() - q_pi2).mean()
    policy_optim.zero_grad(); policy_loss.backward(); policy_optim.step()

    # Alpha update
    with torch.no_grad():
        _, pi_lp3 = policy(obs)
    alpha_loss = -(log_alpha * (pi_lp3.detach().squeeze() + TARGET_ENTROPY)).mean()
    alpha_optim.zero_grad(); alpha_loss.backward(); alpha_optim.step()
    with torch.no_grad():
        log_alpha.data.clamp_(max=math.log(MAX_ALPHA))

    return dict(
        policy_loss=policy_loss.item(),
        qf1_loss=qf1_loss.item(),
        qf2_loss=qf2_loss.item(),
        vf_loss=vf_loss.item(),
        alpha=log_alpha.exp().item(),
        mean_q=q1.mean().item(),
        mean_v=v_pred.mean().item(),
    )


# ── CSV logger ────────────────────────────────────────────────────────────────
csv_path = os.path.join(out_dir, "train_log.csv")
csv_f = open(csv_path, "w", newline="")
csv_w = csv.writer(csv_f)
csv_w.writerow(["epoch", "online_buf", "policy_loss", "qf1_loss", "qf2_loss",
                "vf_loss", "alpha", "mean_q", "eval_return", "wall_sec"])

# ── Training loop ─────────────────────────────────────────────────────────────
print(f"\nWSRL training: {args.n_epochs} epochs, batch={args.batch_size}, "
      f"UTD={args.utd}, min_online={args.min_online}")
t0 = time.time()
history = []
eval_return = 0.0

for epoch in range(1, args.n_epochs + 1):
    # ── Rollout ──────────────────────────────────────────────────────
    roll_info = sampler.sample(sampler_policy, n_steps=args.n_rollout)
    n_online = buffer.size - buffer.fixed_dataset_size  # fixed=0 for WSRL

    # ── Train ────────────────────────────────────────────────────────
    losses = {}
    if n_online >= args.min_online:
        for _ in range(args.n_train * args.utd):
            if not buffer.has_online_data():
                break
            batch = buffer.sample(args.batch_size, scope="sim")
            losses = sac_step(batch)

    # ── Eval ─────────────────────────────────────────────────────────
    if epoch % args.eval_every == 0:
        trajs = eval_sampler.sample(sampler_policy, n_trajs=2)
        eval_return = float(np.mean([sum(t["rewards"]) for t in trajs])) if trajs else 0.0
        print(f"  [Epoch {epoch:4d}/{args.n_epochs}] "
              f"online={n_online:6d}  "
              f"pi={losses.get('policy_loss', 0):.4f}  "
              f"q={losses.get('qf1_loss', 0):.2f}  "
              f"α={losses.get('alpha', 0):.3f}  "
              f"eval={eval_return:.1f}  "
              f"t={time.time()-t0:.0f}s")

    csv_w.writerow([
        epoch, n_online,
        losses.get("policy_loss", 0), losses.get("qf1_loss", 0),
        losses.get("qf2_loss", 0), losses.get("vf_loss", 0),
        losses.get("alpha", 0), losses.get("mean_q", 0),
        eval_return, time.time() - t0,
    ])
    csv_f.flush()

    if epoch % args.eval_every == 0:
        history.append(dict(epoch=epoch, eval_return=eval_return, **losses))

    # ── Checkpoint ───────────────────────────────────────────────────
    if epoch % args.ckpt_every == 0:
        ckpt_p = os.path.join(out_dir, f"checkpoint_epoch{epoch}.pt")
        torch.save(dict(policy=policy.state_dict(), qf1=qf1.state_dict(),
                        qf2=qf2.state_dict(), vf=vf.state_dict(), epoch=epoch), ckpt_p)

csv_f.close()

# ── Final save ───────────────────────────────────────────────────────────────
torch.save(dict(policy=policy.state_dict(), qf1=qf1.state_dict(),
                qf2=qf2.state_dict(), vf=vf.state_dict(), epoch=args.n_epochs),
           os.path.join(out_dir, "model_final.pt"))
print(f"\nFinal model saved → {out_dir}/model_final.pt")

# ── Plot ──────────────────────────────────────────────────────────────────────
if history:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    epochs = [h["epoch"] for h in history]
    axes[0].plot(epochs, [h.get("eval_return", 0) for h in history], "b-o", ms=4)
    axes[0].set_title("Eval Return"); axes[0].set_xlabel("Epoch")
    axes[1].plot(epochs, [h.get("policy_loss", 0) for h in history], "r-")
    axes[1].set_title("Policy Loss"); axes[1].set_xlabel("Epoch")
    axes[2].plot(epochs, [h.get("alpha", 0) for h in history], "g-")
    axes[2].set_title("Alpha"); axes[2].set_xlabel("Epoch")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_curve.png"), dpi=150)
    plt.close()

print(f"Done in {time.time()-t0:.0f}s.  Log: {csv_path}")
