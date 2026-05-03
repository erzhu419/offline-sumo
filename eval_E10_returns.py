"""Quick re-eval of the BEST checkpoint per (E10/noBC method, seed) recording
per-episode returns so we can run hierarchical bootstrap. Output: experiment_output/eval_E10_returns.csv
with columns method, seed, ckpt, mean_return, all_returns (JSON list of 10 floats)."""
import os, csv, json, time, argparse
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
                    default=os.path.join(_HERE, "experiment_output", "eval_E10_returns.csv"))
args = parser.parse_args()


def best_ckpts():
    """Pick the best checkpoint per seed for each E10 / noBC method from existing eval CSVs."""
    tasks = []
    for csv_name, methods in [
        ('eval_E10.csv', ['RLPD-E10', 'WSRL-E10']),
        ('eval_noBC.csv', ['RE-SAC noBC']),
    ]:
        df = pd.read_csv(os.path.join(_HERE, 'experiment_output', csv_name))
        for m in methods:
            sub = df[df['method'] == m]
            for seed, g in sub.groupby('seed'):
                idx = g['mean_return'].idxmax()
                row = g.loc[idx]
                tasks.append({"method": m, "seed": int(seed),
                              "kind": "best", "step": int(row['step']),
                              "ckpt": row['ckpt']})
    return tasks


def worker_with_returns(task):
    """Wrapper around eo.worker that ALSO returns per-episode returns."""
    try:
        import torch
        torch.set_num_threads(1)
        from model import EmbeddingLayer, BusEmbeddingPolicy, BusSamplerPolicy
        from bus_sampler import BusEvalSampler
        from envs.sumo_gym_env import SumoGymEnv
        from common.data_utils import set_route_length, build_edge_linear_map

        edge_xml = os.path.join(_HERE, "env", "network_data", "a_sorted_busline_edge.xml")
        edge_map = build_edge_linear_map(edge_xml, "7X") if os.path.exists(edge_xml) else {}
        route_length = max(edge_map.values()) if edge_map else 13119.0
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
        policy.load_state_dict(pol_sd); policy.eval()

        sumo_dir = os.path.join(os.path.dirname(_HERE), "sumo-rl",
                                 "_standalone_f543609", "SUMO_ruiguang", "online_control")
        env = SumoGymEnv(sumo_dir=sumo_dir, edge_xml=edge_xml, max_steps=18000, line_id="7X")
        sampler = BusEvalSampler(env)
        sp = BusSamplerPolicy(policy, device="cpu")

        returns = []
        for _ in range(args.n_eval):
            trajs = sampler.sample(sp, n_trajs=1, deterministic=True)
            if trajs:
                returns.append(float(sum(trajs[0]["rewards"])))
        return {**task, "all_returns": returns,
                "mean_return": float(np.mean(returns)) if returns else 0.0}
    except Exception as e:
        import traceback
        return {**task, "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()[:300]}"}


def main():
    tasks = best_ckpts()
    print(f"Selected {len(tasks)} best checkpoints (3 per method x 3 methods = 9 expected).")
    f = open(args.out_csv, "w", newline="")
    w = csv.writer(f)
    w.writerow(["method","seed","kind","step","ckpt","mean_return","all_returns","error"])
    f.flush()
    t0 = time.time(); done = 0
    with mp.Pool(processes=args.n_workers, maxtasksperchild=1) as pool:
        for r in pool.imap_unordered(worker_with_returns, tasks):
            done += 1
            err = r.get("error", "")
            w.writerow([r["method"], r["seed"], r["kind"], r["step"], r["ckpt"],
                        r.get("mean_return", 0.0),
                        json.dumps(r.get("all_returns", [])), err])
            f.flush()
            print(f"[{done}/{len(tasks)}] {'OK' if not err else 'ERR'} {r['method']:14s} s{r['seed']} step{r['step']:6d}  return={r.get('mean_return', 0):8.0f}  ({(time.time()-t0)/60:.1f}m)")
    f.close()
    print(f"\nDone in {(time.time()-t0)/60:.1f} min. Saved: {args.out_csv}")


if __name__ == "__main__":
    main()
