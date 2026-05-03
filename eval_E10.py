"""Focused eval for the 6 RLPD-E10 / WSRL-E10 jobs (3 seeds each).
Evaluates ckpts at epoch {50, 100, 150, 200, 250} + model_final.pt (=epoch 300),
10 SUMO episodes each, 14 workers parallel.
Output: experiment_output/eval_E10.csv."""
import os, csv, time, argparse
import multiprocessing as mp

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
                    default=os.path.join(_HERE, "experiment_output", "eval_E10.csv"))
args = parser.parse_args()
eo.args.n_eval = args.n_eval

OUT_DIR = os.path.join(_HERE, "experiment_output")
DIRS = {
    ("RLPD-E10", 42):  "rlpd_E10_seed42_26-05-03-08-04-32",
    ("RLPD-E10", 123): "rlpd_E10_seed123_26-05-03-08-13-06",
    ("RLPD-E10", 456): "rlpd_E10_seed456_26-05-03-08-13-35",
    ("WSRL-E10", 42):  "wsrl_E10_seed42_26-05-03-08-04-34",
    ("WSRL-E10", 123): "wsrl_E10_seed123_26-05-03-08-13-10",
    ("WSRL-E10", 456): "wsrl_E10_seed456_26-05-03-08-13-08",
}


def build_tasks():
    tasks = []
    for (method, seed), d in DIRS.items():
        full = os.path.join(OUT_DIR, d)
        for ep in (50, 100, 150, 200, 250):
            ck = os.path.join(full, f"checkpoint_epoch{ep}.pt")
            if os.path.exists(ck):
                tasks.append({"method": method, "seed": seed,
                              "kind": "epoch", "step": ep, "ckpt": ck})
        ck = os.path.join(full, "model_final.pt")
        if os.path.exists(ck):
            tasks.append({"method": method, "seed": seed,
                          "kind": "epoch", "step": 300, "ckpt": ck})
    return tasks


def main():
    tasks = build_tasks()
    print(f"Selected {len(tasks)} E10 checkpoints (expect 36 = 6 ckpts/seed × 6 jobs).")
    if len(tasks) < 36:
        print("WARNING: missing checkpoints — proceeding with what we have")
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
            print(f"[{done}/{len(tasks)}] {tag} {r['method']:9s} s{r['seed']} ep{r['step']:3d}  "
                  f"return={r.get('mean_return',0):8.0f}  "
                  f"({(time.time()-t0)/60:.1f}m)")
    f.close()
    print(f"\nDone in {(time.time()-t0)/60:.1f} min. Saved: {args.out_csv}")


if __name__ == "__main__":
    main()
