"""Focused eval for the 3 RE-SAC noBC seeds. Evaluates all 10 ckpts per seed
(step10000-step90000 + model_final = step100000), 10 SUMO episodes each.
Output: experiment_output/eval_noBC.csv. 14 workers parallel."""
import os, glob, csv, time, argparse
import multiprocessing as mp
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
import sys
for p in [_HERE, os.path.join(_HERE, "env"), os.path.join(_HERE, "agents"),
          os.path.join(_HERE, "buffers"), os.path.join(_HERE, "utils")]:
    if p not in sys.path:
        sys.path.insert(0, p)

# reuse worker from eval_operational
import eval_operational as eo

parser = argparse.ArgumentParser()
parser.add_argument("--n_workers", type=int, default=14)
parser.add_argument("--n_eval",    type=int, default=10)
parser.add_argument("--out_csv",   type=str,
                    default=os.path.join(_HERE, "experiment_output", "eval_noBC.csv"))
args = parser.parse_args()
eo.args.n_eval = args.n_eval

OUT_DIR = os.path.join(_HERE, "experiment_output")
SEEDS = [42, 123, 456]
DIRS = {
    42:  os.path.join(OUT_DIR, "resac_offline_noBC_seed42_26-05-03-07-43-56"),
    123: os.path.join(OUT_DIR, "resac_offline_noBC_seed123_26-05-03-07-43-57"),
    456: os.path.join(OUT_DIR, "resac_offline_noBC_seed456_26-05-03-08-03-31"),
}

def build_tasks():
    tasks = []
    for seed, d in DIRS.items():
        # step10000 .. step90000
        for s in range(10_000, 100_000, 10_000):
            ck = os.path.join(d, f"checkpoint_step{s}.pt")
            if os.path.exists(ck):
                tasks.append({"method": "RE-SAC noBC", "seed": seed,
                              "kind": "step", "step": s, "ckpt": ck})
        # final ckpt = model_final.pt (= step 100000)
        ck = os.path.join(d, "model_final.pt")
        if os.path.exists(ck):
            tasks.append({"method": "RE-SAC noBC", "seed": seed,
                          "kind": "step", "step": 100_000, "ckpt": ck})
    return tasks


def main():
    tasks = build_tasks()
    print(f"Selected {len(tasks)} noBC checkpoints (10/seed × 3 seeds expected = 30).")
    write_header = not os.path.exists(args.out_csv)
    f = open(args.out_csv, "a", newline="")
    w = csv.writer(f)
    if write_header:
        w.writerow(["method", "seed", "kind", "step", "ckpt", "mean_return",
                    "n_episodes", "n_decisions", "avg_decisions_per_ep",
                    "per_decision_reward", "mean_hw_dev_fwd", "mean_hw_dev_bwd",
                    "bunching_rate", "largegap_rate", "mean_per_line_cv",
                    "mean_hold_s", "hold_p50_s", "hold_p90_s", "error"])
        f.flush()
    t0 = time.time()
    done = 0
    with mp.Pool(processes=args.n_workers, maxtasksperchild=1) as pool:
        for r in pool.imap_unordered(eo.worker, tasks):
            done += 1
            err = r.get("error", "")
            w.writerow([
                r["method"], r["seed"], r["kind"], r["step"], r["ckpt"],
                r.get("mean_return", 0.0), r.get("n_episodes", 0),
                r.get("n_decisions", 0), r.get("avg_decisions_per_ep", 0.0),
                r.get("per_decision_reward", 0.0),
                r.get("mean_hw_dev_fwd", 0.0), r.get("mean_hw_dev_bwd", 0.0),
                r.get("bunching_rate", 0.0), r.get("largegap_rate", 0.0),
                r.get("mean_per_line_cv", 0.0), r.get("mean_hold_s", 0.0),
                r.get("hold_p50_s", 0.0), r.get("hold_p90_s", 0.0), err,
            ])
            f.flush()
            tag = "OK" if not err else "ERR"
            print(f"[{done}/{len(tasks)}] {tag} s{r['seed']} step{r['step']:6d}  "
                  f"return={r.get('mean_return',0):8.0f}  hold={r.get('mean_hold_s',0):4.1f}s  "
                  f"({(time.time()-t0)/60:.1f}m)")
    f.close()
    print(f"\nDone in {(time.time()-t0)/60:.1f} min. Saved: {args.out_csv}")


if __name__ == "__main__":
    main()
