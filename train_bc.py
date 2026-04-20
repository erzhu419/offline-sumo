"""
train_bc.py
===========
Behavior Cloning baseline: supervised learning on offline (state, action) pairs.

Simplest offline RL baseline — no Q-learning, no value function, just MLE on
the action distribution given the state.

After training, evaluates on SUMO and saves results (same format as
eval_offline.py, for apples-to-apples comparison).

Run:
    cd /home/erzhu419/mine_code/offline-sumo
    conda run -n LSTM-RL python train_bc.py --n_steps 100000 --n_eval 10
"""

import os, sys, json, time, csv, argparse, datetime
import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
for p in [_HERE, os.path.join(_HERE, "env"), os.path.join(_HERE, "agents"),
          os.path.join(_HERE, "buffers"), os.path.join(_HERE, "utils")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from model import EmbeddingLayer, BusEmbeddingPolicy, BusSamplerPolicy
from bus_replay_buffer import BusMixedReplayBuffer
from bus_sampler import BusEvalSampler
from envs.sumo_gym_env import SumoGymEnv
from common.data_utils import set_route_length, build_edge_linear_map

parser = argparse.ArgumentParser()
parser.add_argument("--seed",       type=int, default=42)
parser.add_argument("--device",     type=str, default="cpu")
parser.add_argument("--n_steps",    type=int, default=100000)
parser.add_argument("--batch_size", type=int, default=2048)
parser.add_argument("--lr",         type=float, default=3e-4)
parser.add_argument("--n_eval",     type=int, default=10)
parser.add_argument("--eval_every", type=int, default=5000)
parser.add_argument("--dataset_file", type=str,
                    default=os.path.join(_HERE, "data", "datasets_v2", "merged_all_v2.h5"))
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

_ts = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
out_dir = os.path.join(_HERE, "experiment_output", f"bc_seed{args.seed}_{_ts}")
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

# ── Policy ──────────────────────────────────────────────────────────────────
cat_cols = ["line_id", "bus_id", "station_id", "time_period", "direction"]
cat_code_dict = {
    "line_id":     {i: i for i in range(12)},
    "bus_id":      {i: i for i in range(389)},
    "station_id":  {i: i for i in range(1)},
    "time_period": {i: i for i in range(1)},
    "direction":   {0: 0, 1: 1},
}
obs_dim, action_dim, hidden_sz = 17, 2, 48

emb_tmpl = EmbeddingLayer(cat_code_dict, cat_cols, layer_norm=True, dropout=0.05)
state_dim = emb_tmpl.output_dim + (obs_dim - len(cat_cols))
policy = BusEmbeddingPolicy(state_dim, action_dim, hidden_sz, emb_tmpl.clone(), action_range=1.0)
policy.to(args.device)
optim_policy = torch.optim.Adam(policy.parameters(), lr=args.lr)

# ── Load offline data ───────────────────────────────────────────────────────
print(f"Loading offline data: {args.dataset_file}")
buffer = BusMixedReplayBuffer(
    state_dim=obs_dim, action_dim=action_dim, context_dim=30,
    dataset_file=args.dataset_file, device=args.device,
    buffer_ratio=0.01,   # no online expansion
    reward_scale=1.0, reward_bias=0.0,
)
print(f"Offline data: {buffer.fixed_dataset_size:,} transitions")

# ── SUMO env for periodic eval ──────────────────────────────────────────────
_SUMO_DIR = os.path.join(
    os.path.dirname(_HERE), "sumo-rl",
    "_standalone_f543609", "SUMO_ruiguang", "online_control",
)
eval_env = SumoGymEnv(sumo_dir=_SUMO_DIR, edge_xml=edge_xml, max_steps=18000, line_id="7X")
eval_sampler = BusEvalSampler(eval_env)
sampler_policy = BusSamplerPolicy(policy, args.device)


# ── Training ────────────────────────────────────────────────────────────────
csv_path = os.path.join(out_dir, "train_log.csv")
csv_f = open(csv_path, "w", newline="")
csv_w = csv.writer(csv_f)
csv_w.writerow(["step", "bc_loss", "eval_return", "wall_sec"])

print(f"\nBC training: {args.n_steps} steps, batch={args.batch_size}")
t0 = time.time()
eval_return = 0.0

for step in range(1, args.n_steps + 1):
    batch = buffer.sample(args.batch_size, scope="real")
    obs = batch["observations"]
    actions = batch["actions"]  # already in [-1, 1] (raw tanh)

    # Predict mean + log_std, maximize log-likelihood of actions
    pi_a, pi_lp = policy(obs)
    # Use the policy's log_prob directly on actions (atanh adjustment handled in policy)
    try:
        log_prob = policy.log_prob(obs, actions)
    except AttributeError:
        # Fallback: MSE on the mean head if log_prob not exposed
        with torch.no_grad():
            pi_mean = policy(obs, deterministic=True)[0]
        log_prob = -F.mse_loss(pi_mean, actions, reduction="none").sum(-1)
    bc_loss = -log_prob.mean()

    optim_policy.zero_grad()
    bc_loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    optim_policy.step()

    if step % args.eval_every == 0:
        trajs = eval_sampler.sample(sampler_policy, n_trajs=2, deterministic=True)
        eval_return = float(np.mean([sum(t["rewards"]) for t in trajs])) if trajs else 0.0
        print(f"  [Step {step:6d}/{args.n_steps}]  bc_loss={bc_loss.item():.4f}  "
              f"eval={eval_return:.1f}  t={time.time()-t0:.0f}s")

    csv_w.writerow([step, bc_loss.item(), eval_return, time.time() - t0])
    if step % 100 == 0:
        csv_f.flush()

csv_f.close()

# ── Final save ──────────────────────────────────────────────────────────────
torch.save(dict(policy=policy.state_dict(), step=args.n_steps),
           os.path.join(out_dir, "model_final.pt"))
print(f"\nModel saved → {out_dir}/model_final.pt")

# ── Final eval (more episodes for statistics) ──────────────────────────────
print(f"\nFinal eval: {args.n_eval} episodes...")
returns = []
for i in range(args.n_eval):
    trajs = eval_sampler.sample(sampler_policy, n_trajs=1, deterministic=True)
    ret = sum(trajs[0]["rewards"])
    returns.append(ret)
    print(f"  Episode {i+1}/{args.n_eval}: return={ret:.1f}")

returns = np.array(returns)
result = {
    "method": "BC",
    "seed": args.seed,
    "n_eval": args.n_eval,
    "mean_return": float(returns.mean()),
    "std_return": float(returns.std()),
    "min_return": float(returns.min()),
    "max_return": float(returns.max()),
    "all_returns": returns.tolist(),
}
with open(os.path.join(out_dir, "eval_result.json"), "w") as f:
    json.dump(result, f, indent=2)

print(f"\n{'='*50}")
print(f"BC eval: {result['mean_return']:.1f} ± {result['std_return']:.1f}")
print(f"  min={result['min_return']:.1f}, max={result['max_return']:.1f}")
print(f"{'='*50}")
print(f"Done in {time.time()-t0:.0f}s")
