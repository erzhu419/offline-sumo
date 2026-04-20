"""
eval_offline.py
===============
Evaluate the offline-only pretrained model on SUMO without fine-tuning.

Loads pretrained/offline_final.pt and runs N evaluation episodes.
Reports mean ± std of episode returns.

Run:
    cd /home/erzhu419/mine_code/offline-sumo
    conda run -n LSTM-RL python eval_offline.py --n_eval 10
"""

import os, sys, json, argparse
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
for p in [_HERE, os.path.join(_HERE, "env"), os.path.join(_HERE, "agents"),
          os.path.join(_HERE, "buffers"), os.path.join(_HERE, "utils")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from model import EmbeddingLayer, BusEmbeddingPolicy, BusSamplerPolicy
from bus_sampler import BusEvalSampler
from envs.sumo_gym_env import SumoGymEnv
from common.data_utils import set_route_length, build_edge_linear_map

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, default=os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "pretrained", "offline_final.pt"))
parser.add_argument("--n_eval", type=int, default=10, help="number of eval episodes")
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# ── Route setup ──────────────────────────────────────────────────────────────
edge_xml = os.path.join(_HERE, "env", "network_data", "a_sorted_busline_edge.xml")
if os.path.exists(edge_xml):
    edge_map = build_edge_linear_map(edge_xml, "7X")
    route_length = max(edge_map.values()) if edge_map else 13119.0
else:
    route_length = 13119.0
set_route_length(route_length)

# ── Load model ──────────────────────────────────────────────────────────────
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

policy = BusEmbeddingPolicy(state_dim, action_dim, hidden_sz, emb_tmpl.clone(), action_range=1.0)
ckpt = torch.load(args.ckpt, map_location=args.device)
policy.load_state_dict(ckpt["policy"])
policy.to(args.device)
policy.eval()
print(f"Loaded model from {args.ckpt}")

sampler_policy = BusSamplerPolicy(policy, args.device)

# ── SUMO environment ────────────────────────────────────────────────────────
_SUMO_DIR = os.path.join(
    os.path.dirname(_HERE), "sumo-rl",
    "_standalone_f543609", "SUMO_ruiguang", "online_control",
)
_EDGE_XML = os.path.join(_HERE, "env", "network_data", "a_sorted_busline_edge.xml")
eval_env = SumoGymEnv(sumo_dir=_SUMO_DIR, edge_xml=_EDGE_XML, max_steps=18000, line_id="7X")
eval_sampler = BusEvalSampler(eval_env)

# ── Evaluation ──────────────────────────────────────────────────────────────
print(f"Evaluating {args.n_eval} episodes (deterministic)...")
returns = []
for i in range(args.n_eval):
    trajs = eval_sampler.sample(sampler_policy, n_trajs=1, deterministic=True)
    ep_return = sum(trajs[0]["rewards"])
    returns.append(ep_return)
    print(f"  Episode {i+1}/{args.n_eval}: return={ep_return:.1f}")

returns = np.array(returns)
result = {
    "checkpoint": args.ckpt,
    "n_eval": args.n_eval,
    "seed": args.seed,
    "mean_return": float(returns.mean()),
    "std_return": float(returns.std()),
    "min_return": float(returns.min()),
    "max_return": float(returns.max()),
    "all_returns": returns.tolist(),
}

print(f"\n{'='*50}")
print(f"Offline-only eval: {result['mean_return']:.1f} ± {result['std_return']:.1f}")
print(f"  min={result['min_return']:.1f}, max={result['max_return']:.1f}")
print(f"{'='*50}")

# Save results
out_path = os.path.join(_HERE, "experiment_output", "eval_offline_results.json")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)
print(f"Results saved to {out_path}")
