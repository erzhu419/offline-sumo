"""
eval_operational.py
===================
Compute operational transit metrics (headway deviation, bunching rate, holding
time) for a subset of representative checkpoints.

Selects: for each method+seed, the FINAL and BEST checkpoint (per existing
eval_results.csv mean_return). Runs N=10 SUMO episodes per ckpt and reports
per-decision operational stats.

Output: experiment_output/eval_operational.csv with columns method, seed,
kind, step, ckpt, mean_return, mean_hw_dev, mean_hw_dev_back, bunching_rate,
mean_hold_s, n_decisions.

Run:
    cd /home/erzhu419/mine_code/offline-sumo
    conda run -n LSTM-RL python eval_operational.py --n_workers 14 --n_eval 10
"""

import os, re, json, csv, time, argparse
import multiprocessing as mp
import pandas as pd
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for p in [_HERE, os.path.join(_HERE, "env"), os.path.join(_HERE, "agents"),
          os.path.join(_HERE, "buffers"), os.path.join(_HERE, "utils")]:
    if p not in __import__("sys").path:
        __import__("sys").path.insert(0, p)

parser = argparse.ArgumentParser()
parser.add_argument("--n_workers", type=int, default=14)
parser.add_argument("--n_eval",    type=int, default=10)
parser.add_argument("--in_csv",    type=str, default=os.path.join(_HERE, "experiment_output", "eval_results.csv"))
parser.add_argument("--out_csv",   type=str, default=os.path.join(_HERE, "experiment_output", "eval_operational.csv"))
parser.add_argument("--methods",   type=str, default="",
                    help="comma-separated method filter; empty = all methods in in_csv")
args = parser.parse_args()


def select_representative_ckpts():
    """Pick (final, best) checkpoint per (method, seed) from eval_results.csv."""
    df = pd.read_csv(args.in_csv)
    df["_norm_step"] = df.apply(lambda r: r["step"] if r["kind"] in ("epoch", "step") else 0, axis=1)
    if args.methods:
        keep = set(args.methods.split(","))
        df = df[df["method"].isin(keep)]
    tasks = []
    for (method, seed), g in df.groupby(["method", "seed"]):
        g_step = g[g["_norm_step"] > 0].sort_values("_norm_step")
        if g_step.empty:
            sub = g
        else:
            sub = g_step
        # final
        final_row = sub.iloc[-1]
        tasks.append({"method": method, "seed": int(seed), "kind": "final",
                      "step": int(final_row["_norm_step"]), "ckpt": final_row["ckpt"]})
        # best
        best_row = sub.loc[sub["mean_return"].idxmax()]
        if best_row["ckpt"] != final_row["ckpt"]:
            tasks.append({"method": method, "seed": int(seed), "kind": "best",
                          "step": int(best_row["_norm_step"]), "ckpt": best_row["ckpt"]})
    return tasks


def worker(task):
    try:
        import torch
        torch.set_num_threads(1)
        from model import EmbeddingLayer, BusEmbeddingPolicy, BusSamplerPolicy
        from bus_sampler import BusEvalSampler
        from envs.sumo_gym_env import SumoGymEnv
        from common.data_utils import set_route_length, build_edge_linear_map

        edge_xml = os.path.join(_HERE, "env", "network_data", "a_sorted_busline_edge.xml")
        if os.path.exists(edge_xml):
            edge_map = build_edge_linear_map(edge_xml, "7X")
            route_length = max(edge_map.values()) if edge_map else 13119.0
        else:
            route_length = 13119.0
        set_route_length(route_length)

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
        policy.load_state_dict(pol_sd)
        policy.eval()

        sumo_dir = os.path.join(os.path.dirname(_HERE), "sumo-rl",
                                 "_standalone_f543609", "SUMO_ruiguang", "online_control")
        env = SumoGymEnv(sumo_dir=sumo_dir, edge_xml=edge_xml, max_steps=18000, line_id="7X")
        sampler = BusEvalSampler(env)
        sp = BusSamplerPolicy(policy, device="cpu")

        returns = []
        all_hw_dev_fwd, all_hw_dev_bwd, all_holds = [], [], []
        n_bunch, n_dec = 0, 0
        for _ in range(args.n_eval):
            trajs = sampler.sample(sp, n_trajs=1, deterministic=True)
            if not trajs: continue
            t = trajs[0]
            returns.append(sum(t["rewards"]))
            for o, a in zip(t.get("observations", []), t.get("actions", [])):
                if len(o) < 9: continue
                h_fwd, h_bwd, tgt = o[5], o[6], o[8]
                if tgt <= 0: continue
                all_hw_dev_fwd.append(abs(h_fwd - tgt))
                all_hw_dev_bwd.append(abs(h_bwd - tgt))
                if h_fwd < 0.5 * tgt:
                    n_bunch += 1
                if len(a) >= 1:
                    all_holds.append(max(0.0, min(60.0, 30.0 * a[0] + 30.0)))
                n_dec += 1

        return {
            **task,
            "mean_return":      float(np.mean(returns)) if returns else 0.0,
            "n_episodes":       len(returns),
            "n_decisions":      n_dec,
            "mean_hw_dev_fwd":  float(np.mean(all_hw_dev_fwd)) if all_hw_dev_fwd else 0.0,
            "mean_hw_dev_bwd":  float(np.mean(all_hw_dev_bwd)) if all_hw_dev_bwd else 0.0,
            "bunching_rate":    float(n_bunch / n_dec) if n_dec else 0.0,
            "mean_hold_s":      float(np.mean(all_holds)) if all_holds else 0.0,
        }
    except Exception as e:
        import traceback
        return {**task, "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()[:300]}"}


def main():
    tasks = select_representative_ckpts()
    print(f"Selected {len(tasks)} representative checkpoints (final+best per method×seed).")

    write_header = not os.path.exists(args.out_csv)
    f = open(args.out_csv, "a", newline="")
    w = csv.writer(f)
    if write_header:
        w.writerow(["method", "seed", "kind", "step", "ckpt", "mean_return",
                    "n_episodes", "n_decisions", "mean_hw_dev_fwd", "mean_hw_dev_bwd",
                    "bunching_rate", "mean_hold_s", "error"])
        f.flush()

    t0 = time.time()
    done = 0
    with mp.Pool(processes=args.n_workers, maxtasksperchild=1) as pool:
        for r in pool.imap_unordered(worker, tasks):
            done += 1
            err = r.get("error", "")
            w.writerow([
                r["method"], r["seed"], r["kind"], r["step"], r["ckpt"],
                r.get("mean_return", 0.0), r.get("n_episodes", 0),
                r.get("n_decisions", 0), r.get("mean_hw_dev_fwd", 0.0),
                r.get("mean_hw_dev_bwd", 0.0), r.get("bunching_rate", 0.0),
                r.get("mean_hold_s", 0.0), err,
            ])
            f.flush()
            elapsed_min = (time.time() - t0) / 60
            rate = done / max(elapsed_min, 0.01)
            eta = (len(tasks) - done) / max(rate, 0.01)
            tag = "OK" if not err else "ERR"
            print(f"[{done}/{len(tasks)}] {tag} {r['method']:18s} s{r['seed']} {r['kind']:5s}  "
                  f"return={r.get('mean_return',0):8.0f}  hw_fwd={r.get('mean_hw_dev_fwd',0):6.1f}  "
                  f"bunch={100*r.get('bunching_rate',0):4.1f}%  hold={r.get('mean_hold_s',0):4.1f}s  "
                  f"({elapsed_min:.1f}m, ETA {eta:.1f}m)")

    f.close()
    print(f"\nDone in {(time.time()-t0)/60:.1f} min. Saved: {args.out_csv}")


if __name__ == "__main__":
    main()
