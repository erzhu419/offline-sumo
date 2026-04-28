"""
eval_checkpoints_parallel.py
============================
Evaluate ALL checkpoints (every save point + final) across all experiment runs,
using n_eval SUMO episodes per checkpoint. Parallelises over checkpoints using
multiprocessing.

Output: writes `eval_results.csv` with (method, seed, step_or_epoch, mean_return,
std_return, n_eval, ckpt_path).

Run:
    cd /home/erzhu419/mine_code/offline-sumo
    conda run -n LSTM-RL python eval_checkpoints_parallel.py --n_workers 14 --n_eval 10
"""

import os, sys, re, json, csv, time, argparse, traceback
import multiprocessing as mp
from copy import deepcopy

_HERE = os.path.dirname(os.path.abspath(__file__))
for p in [_HERE, os.path.join(_HERE, "env"), os.path.join(_HERE, "agents"),
          os.path.join(_HERE, "buffers"), os.path.join(_HERE, "utils")]:
    if p not in sys.path:
        sys.path.insert(0, p)

parser = argparse.ArgumentParser()
parser.add_argument("--n_workers", type=int, default=14)
parser.add_argument("--n_eval",    type=int, default=10, help="episodes per checkpoint")
parser.add_argument("--ckpt_dir",  type=str, default=os.path.join(_HERE, "experiment_output"))
parser.add_argument("--out_csv",   type=str, default=os.path.join(_HERE, "experiment_output", "eval_results.csv"))
parser.add_argument("--only",      type=str, default="", help="substring filter on run dir")
parser.add_argument("--skip_existing", action="store_true", help="skip checkpoints already in out_csv")
args = parser.parse_args()


# ── Patterns identifying which runs to evaluate ───────────────────────────────
RUN_PATTERNS = [
    ("BC",              re.compile(r"^bc_seed(\d+)_26-04-(20|21)")),
    ("CQL",             re.compile(r"^cql_seed(\d+)_26-04-27")),
    ("RE-SAC offline",  re.compile(r"^resac_offline_seed(\d+)_26-04-(20|21)")),
    ("RE-SAC no LCB",   re.compile(r"^resac_offline_noLCB_seed(\d+)_26-04-(20|21)")),
    ("RE-SAC no L1",    re.compile(r"^resac_offline_noL1_seed(\d+)_26-04-(20|21)")),
    ("RE-SAC no reg",   re.compile(r"^resac_offline_noBoth_seed(\d+)_26-04-(20|21)")),
    ("H2O+ offline",    re.compile(r"^offline_only_seed(\d+)_26-04-(20|21)")),
    ("Online SAC",      re.compile(r"^online_seed(\d+)_26-04-(20|21)")),
    ("WSRL",            re.compile(r"^wsrl_seed(\d+)_26-04-(20|21)")),
    ("RLPD (0.50)",     re.compile(r"^rlpd_seed(\d+)_26-04-(20|21)")),
    ("RLPD (0.25)",     re.compile(r"^rlpd_ratio0\.25_seed(\d+)_26-04-(20|21)")),
    ("RLPD (0.75)",     re.compile(r"^rlpd_ratio0\.75_seed(\d+)_26-04-(20|21)")),
]

# Checkpoint filename -> step/epoch parser
def parse_step(fname):
    m = re.match(r"checkpoint_epoch(\d+)\.pt$", fname)
    if m: return ("epoch", int(m.group(1)))
    m = re.match(r"checkpoint_step(\d+)\.pt$", fname)
    if m: return ("step", int(m.group(1)))
    if fname == "model_final.pt": return ("final", 0)
    return None


def build_task_list(ckpt_dir):
    tasks = []
    for dirname in sorted(os.listdir(ckpt_dir)):
        if args.only and args.only not in dirname:
            continue
        for method, pat in RUN_PATTERNS:
            m = pat.match(dirname)
            if m:
                seed = int(m.group(1))
                run_dir = os.path.join(ckpt_dir, dirname)
                for fname in sorted(os.listdir(run_dir)):
                    if not fname.endswith(".pt"): continue
                    sp = parse_step(fname)
                    if sp is None: continue
                    tasks.append({
                        "method": method,
                        "seed": seed,
                        "kind": sp[0],
                        "step": sp[1],
                        "run_dir": dirname,
                        "ckpt": os.path.join(run_dir, fname),
                    })
                break
    return tasks


# ── Worker ────────────────────────────────────────────────────────────────────
def worker_eval(task):
    """Load checkpoint → n_eval SUMO episodes → return (task, mean, std, returns)."""
    try:
        import numpy as np
        import torch
        torch.set_num_threads(1)  # each worker = 1 thread
        os.environ.setdefault("SUMO_HOME", "/usr/share/sumo")
        # Let the env auto-detect if unset
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
        obs_dim, action_dim, hidden_sz = 17, 2, 48
        emb = EmbeddingLayer(cat_code_dict, cat_cols, layer_norm=True, dropout=0.05)
        state_dim = emb.output_dim + (obs_dim - len(cat_cols))
        policy = BusEmbeddingPolicy(state_dim, action_dim, hidden_sz, emb.clone(), action_range=1.0)

        ckpt = torch.load(task["ckpt"], map_location="cpu", weights_only=False)
        # Find policy state dict in the ckpt under known keys
        pol_sd = ckpt.get("policy") or ckpt.get("policy_state_dict")
        if pol_sd is None:
            return {**task, "error": f"no policy key in ckpt: {list(ckpt.keys())}"}
        policy.load_state_dict(pol_sd)
        policy.eval()

        # Unique SUMO worker id via PID to avoid port conflicts
        sumo_dir = os.path.join(
            os.path.dirname(_HERE), "sumo-rl",
            "_standalone_f543609", "SUMO_ruiguang", "online_control",
        )
        env = SumoGymEnv(sumo_dir=sumo_dir, edge_xml=edge_xml, max_steps=18000, line_id="7X")
        sampler = BusEvalSampler(env)
        sp = BusSamplerPolicy(policy, device="cpu")

        returns = []
        for i in range(args.n_eval):
            trajs = sampler.sample(sp, n_trajs=1, deterministic=True)
            if trajs:
                returns.append(sum(trajs[0]["rewards"]))
        env.close() if hasattr(env, "close") else None

        mean_r = float(np.mean(returns)) if returns else 0.0
        std_r  = float(np.std(returns))  if returns else 0.0
        return {**task, "mean_return": mean_r, "std_return": std_r,
                "n_eval": len(returns), "all_returns": returns}
    except Exception as e:
        return {**task, "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()[:500]}"}


def main():
    tasks = build_task_list(args.ckpt_dir)
    print(f"Found {len(tasks)} checkpoints to evaluate")

    # Skip already-evaluated if resuming
    existing = set()
    if args.skip_existing and os.path.exists(args.out_csv):
        with open(args.out_csv) as f:
            r = csv.DictReader(f)
            for row in r:
                existing.add(row["ckpt"])
        print(f"Skipping {len(existing)} already evaluated checkpoints")
    tasks = [t for t in tasks if t["ckpt"] not in existing]
    print(f"Running {len(tasks)} checkpoints on {args.n_workers} workers")
    if not tasks:
        return

    # Open CSV with header only if new
    write_header = not os.path.exists(args.out_csv)
    csv_f = open(args.out_csv, "a", newline="")
    csv_w = csv.writer(csv_f)
    if write_header:
        csv_w.writerow(["method", "seed", "kind", "step", "run_dir", "ckpt",
                        "mean_return", "std_return", "n_eval", "all_returns", "error"])
        csv_f.flush()

    t0 = time.time()
    done = 0
    with mp.Pool(processes=args.n_workers, maxtasksperchild=1) as pool:
        for result in pool.imap_unordered(worker_eval, tasks):
            done += 1
            status = "OK" if "error" not in result else "ERR"
            mean_r = result.get("mean_return", 0.0)
            err    = result.get("error", "")
            csv_w.writerow([
                result["method"], result["seed"], result["kind"], result["step"],
                result["run_dir"], result["ckpt"],
                mean_r, result.get("std_return", 0.0), result.get("n_eval", 0),
                json.dumps(result.get("all_returns", [])), err,
            ])
            csv_f.flush()
            elapsed = time.time() - t0
            rate    = done / max(elapsed / 60, 0.001)  # tasks per minute
            eta_min = (len(tasks) - done) / max(rate, 0.001)
            print(f"[{done:3d}/{len(tasks)}] {status} {result['method']:15s} seed={result['seed']} "
                  f"{result['kind']}={result['step']:6d} mean={mean_r:8.0f} "
                  f"elapsed={elapsed/60:.1f}m ETA={eta_min:.1f}m")

    csv_f.close()
    print(f"\nDone in {(time.time()-t0)/60:.1f} min. Results saved: {args.out_csv}")


if __name__ == "__main__":
    main()
