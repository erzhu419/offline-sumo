"""Round-4 review additions: passenger-level metrics + no-holding baseline.

For each selected (method, seed, best-ckpt) tuple plus a synthetic NoHoldPolicy that
always issues hold=0 (action [-1, 0]), evaluate 10 SUMO episodes and collect:
  - mean episode return (the original metric)
  - mean / p50 / p90 passenger trip-time (depart-to-arrive seconds)
  - count of completed vs pending passengers per episode
  - existing operational metrics (forward/backward headway dev, hold time, etc.)

Output: experiment_output/eval_passenger.csv with one row per (method, seed) — each row aggregates over the 10 episodes for that ckpt.

Methods covered:
  - NoHold (synthetic, 3 seeds — same SUMO reset sequence as RL methods)
  - BC (best ckpt per seed, 3 seeds)
  - CQL
  - Online SAC (5 seeds)
  - WSRL (5 seeds)
  - RLPD-0.50 (5 seeds)
  - RE-SAC offline (5 seeds)
"""
import os, csv, time, json, argparse
import multiprocessing as mp
import pandas as pd, numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
import sys
for p in [_HERE, os.path.join(_HERE, "env"), os.path.join(_HERE, "agents"),
          os.path.join(_HERE, "buffers"), os.path.join(_HERE, "utils")]:
    if p not in sys.path:
        sys.path.insert(0, p)

import eval_operational as eo

parser = argparse.ArgumentParser()
parser.add_argument("--n_workers", type=int, default=14)
parser.add_argument("--n_eval",    type=int, default=10)
parser.add_argument("--out_csv",   type=str,
                    default=os.path.join(_HERE, "experiment_output", "eval_passenger.csv"))
args = parser.parse_args()
eo.args.n_eval = args.n_eval

METHODS_FROM_RESULTS = {
    "BC": [42, 123, 456],
    "CQL": [42, 123, 456],
    "Online SAC": [42, 123, 456, 789, 1024],
    "WSRL": [42, 123, 456, 789, 1024],
    "RLPD (0.50)": [42, 123, 456, 789, 1024],
    "RE-SAC offline": [42, 123, 456, 789, 1024],
}


def best_ckpt_tasks():
    """For each (method, seed) in METHODS_FROM_RESULTS, find the best-return ckpt
    from eval_results.csv and queue an eval task."""
    df = pd.read_csv(os.path.join(_HERE, "experiment_output", "eval_results.csv"))
    tasks = []
    for method, seeds in METHODS_FROM_RESULTS.items():
        sub = df[(df["method"] == method) & (df["kind"].isin(["epoch", "step"]))]
        for s in seeds:
            g = sub[sub["seed"] == s]
            if len(g) == 0:
                continue
            row = g.loc[g["mean_return"].idxmax()]
            tasks.append({"method": method, "seed": int(s), "kind": "best",
                          "step": int(row["step"]), "ckpt": row["ckpt"]})
    # Add the NoHold synthetic policy: 3 seeds (we just need 3 distinct SUMO realisation runs)
    for s in [42, 123, 456]:
        tasks.append({"method": "NoHold", "seed": s, "kind": "synthetic",
                      "step": 0, "ckpt": "<NoHold>"})
    return tasks


class NoHoldPolicy:
    """Stateless policy that always returns action [-1, 0] -> hold=0s, neutral speed."""
    def __init__(self, action_dim=2):
        self.action_dim = action_dim
    def __call__(self, obs_tensor, deterministic=True):
        bsz = obs_tensor.shape[0]
        out = np.zeros((bsz, self.action_dim), dtype=np.float32)
        out[:, 0] = -1.0
        return out


def worker_pax(task):
    """Run 10 SUMO episodes for the given task, recording return + passenger stats."""
    try:
        import torch
        torch.set_num_threads(1)
        from envs.sumo_gym_env import SumoGymEnv
        from bus_sampler import BusEvalSampler
        from common.data_utils import set_route_length, build_edge_linear_map

        edge_xml = os.path.join(_HERE, "env", "network_data", "a_sorted_busline_edge.xml")
        edge_map = build_edge_linear_map(edge_xml, "7X") if os.path.exists(edge_xml) else {}
        route_length = max(edge_map.values()) if edge_map else 13119.0
        set_route_length(route_length)

        # Build policy (RL or NoHold)
        if task["method"] == "NoHold":
            sp = NoHoldPolicy()
        else:
            from model import EmbeddingLayer, BusEmbeddingPolicy, BusSamplerPolicy
            cat_cols = ["line_id", "bus_id", "station_id", "time_period", "direction"]
            cat_code_dict = {
                "line_id":     {i: i for i in range(12)},
                "bus_id":      {i: i for i in range(389)},
                "station_id":  {i: i for i in range(1)},
                "time_period": {i: i for i in range(1)},
                "direction":   {0: 0, 1: 1},
            }
            emb = EmbeddingLayer(cat_code_dict, cat_cols, layer_norm=True, dropout=0.05)
            state_dim = emb.output_dim + (17 - len(cat_cols))
            policy = BusEmbeddingPolicy(state_dim, 2, 48, emb.clone(), action_range=1.0)
            ckpt = torch.load(task["ckpt"], map_location="cpu", weights_only=False)
            pol_sd = ckpt.get("policy") or ckpt.get("policy_state_dict")
            if pol_sd is None:
                return {**task, "error": "no policy"}
            policy.load_state_dict(pol_sd); policy.eval()
            sp = BusSamplerPolicy(policy, device="cpu")

        sumo_dir = os.path.join(os.path.dirname(_HERE), "sumo-rl",
                                 "_standalone_f543609", "SUMO_ruiguang", "online_control")
        env = SumoGymEnv(sumo_dir=sumo_dir, edge_xml=edge_xml, max_steps=18000, line_id="7X")
        sampler = BusEvalSampler(env)

        returns = []
        pax_completed, pax_pending = [], []
        pax_mean_total, pax_p50, pax_p90, pax_max = [], [], [], []
        pax_pending_load = []
        for _ in range(args.n_eval):
            trajs = sampler.sample(sp, n_trajs=1, deterministic=True)
            if not trajs:
                continue
            t = trajs[0]
            returns.append(float(sum(t["rewards"])))
            ps = t.get("pax_stats", {}) or {}
            pax_completed.append(int(ps.get("n_completed", 0)))
            pax_pending.append(int(ps.get("n_pending", 0)))
            pax_mean_total.append(float(ps.get("mean_total_s", 0.0)))
            pax_p50.append(float(ps.get("p50_total_s", 0.0)))
            pax_p90.append(float(ps.get("p90_total_s", 0.0)))
            pax_max.append(float(ps.get("max_total_s", 0.0)))
            pax_pending_load.append(float(ps.get("sum_pending_so_far_s", 0.0)))

        agg = lambda v: float(np.mean(v)) if v else 0.0
        return {
            **task,
            "n_episodes":           len(returns),
            "mean_return":          agg(returns),
            "pax_completed_mean":   agg(pax_completed),
            "pax_pending_mean":     agg(pax_pending),
            "pax_mean_total_s":     agg(pax_mean_total),
            "pax_p50_total_s":      agg(pax_p50),
            "pax_p90_total_s":      agg(pax_p90),
            "pax_max_total_s":      agg(pax_max),
            "pax_pending_load_s":   agg(pax_pending_load),
            "all_returns":          json.dumps(returns),
            "all_pax_means":        json.dumps(pax_mean_total),
        }
    except Exception as e:
        import traceback
        return {**task, "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()[:300]}"}


def main():
    tasks = best_ckpt_tasks()
    print(f"Selected {len(tasks)} (method, seed) cells (incl. 3 NoHold).")
    f = open(args.out_csv, "w", newline="")
    w = csv.writer(f)
    w.writerow(["method", "seed", "kind", "step", "ckpt", "n_episodes", "mean_return",
                "pax_completed_mean", "pax_pending_mean", "pax_mean_total_s",
                "pax_p50_total_s", "pax_p90_total_s", "pax_max_total_s",
                "pax_pending_load_s", "all_returns", "all_pax_means", "error"])
    f.flush()
    t0 = time.time(); done = 0
    with mp.Pool(processes=args.n_workers, maxtasksperchild=1) as pool:
        for r in pool.imap_unordered(worker_pax, tasks):
            done += 1
            err = r.get("error", "")
            w.writerow([r["method"], r["seed"], r["kind"], r["step"], r["ckpt"],
                        r.get("n_episodes", 0), r.get("mean_return", 0.0),
                        r.get("pax_completed_mean", 0.0), r.get("pax_pending_mean", 0.0),
                        r.get("pax_mean_total_s", 0.0), r.get("pax_p50_total_s", 0.0),
                        r.get("pax_p90_total_s", 0.0), r.get("pax_max_total_s", 0.0),
                        r.get("pax_pending_load_s", 0.0),
                        r.get("all_returns", "[]"), r.get("all_pax_means", "[]"), err])
            f.flush()
            tag = "OK" if not err else "ERR"
            print(f"[{done}/{len(tasks)}] {tag} {r['method']:14s} s{r['seed']:5d}  "
                  f"return={r.get('mean_return',0):8.0f}  "
                  f"pax_compl={r.get('pax_completed_mean',0):7.1f}  "
                  f"pax_mean_s={r.get('pax_mean_total_s',0):7.0f}  "
                  f"({(time.time()-t0)/60:.1f}m)")
    f.close()
    print(f"\nDone in {(time.time()-t0)/60:.1f} min. Saved: {args.out_csv}")


if __name__ == "__main__":
    main()
