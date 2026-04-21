"""
analyze_results.py
==================
Aggregate all experiment results and produce summary table + learning curves.

Reads train_log.csv from each experiment directory, groups by method and seed,
computes mean±std across seeds at each eval point.

Run after all experiments complete:
    cd /home/erzhu419/mine_code/offline-sumo
    python analyze_results.py
"""

import os, re, json, csv
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.join(_HERE, "experiment_output")

# ── Parse experiment directories ────────────────────────────────────────────
PATTERNS = {
    "BC":             re.compile(r"bc_seed(\d+)_"),
    "RE-SAC offline": re.compile(r"resac_offline_seed(\d+)_"),
    "Online SAC":     re.compile(r"online_seed(\d+)_"),
    "WSRL":           re.compile(r"wsrl_seed(\d+)_"),
    "RLPD (0.50)":    re.compile(r"rlpd_seed(\d+)_(?!.*ratio)"),
    "RLPD (0.25)":    re.compile(r"rlpd_ratio0\.25_seed(\d+)_"),
    "RLPD (0.75)":    re.compile(r"rlpd_ratio0\.75_seed(\d+)_"),
}

# Also check for ratio in rlpd dirs via config or directory name
def detect_method(dirname):
    for method, pat in PATTERNS.items():
        m = pat.match(dirname)
        if m:
            return method, int(m.group(1))
    return None, None

def load_eval_returns(csv_path):
    """Load (step_or_epoch, eval_return) pairs from train_log.csv.

    Offline scripts (bc, resac) use 'step' + 'eval_return' logged sparsely.
    Online scripts (online, wsrl, rlpd) use 'epoch' + 'eval_return' each epoch.
    """
    results = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "eval_return" not in reader.fieldnames:
            return results
        x_key = "epoch" if "epoch" in reader.fieldnames else "step"
        prev_ret = None
        for row in reader:
            try:
                ret = float(row["eval_return"])
            except (KeyError, ValueError, TypeError):
                continue
            try:
                ep = int(float(row[x_key]))
            except (KeyError, ValueError, TypeError):
                continue
            if ret != 0.0 or (prev_ret is not None and prev_ret != ret):
                results.append((ep, ret))
            prev_ret = ret
    return results

# ── Collect all results ─────────────────────────────────────────────────────
method_data = {}  # {method: {seed: [(epoch, return), ...]}}

for dirname in sorted(os.listdir(EXP_DIR)):
    csv_path = os.path.join(EXP_DIR, dirname, "train_log.csv")
    if not os.path.exists(csv_path):
        continue
    method, seed = detect_method(dirname)
    if method is None:
        continue
    returns = load_eval_returns(csv_path)
    if not returns:
        continue
    if method not in method_data:
        method_data[method] = {}
    method_data[method][seed] = returns

# ── Load offline eval ───────────────────────────────────────────────────────
offline_eval_path = os.path.join(EXP_DIR, "eval_offline_results.json")
offline_mean = None
if os.path.exists(offline_eval_path):
    with open(offline_eval_path) as f:
        offline_result = json.load(f)
    offline_mean = offline_result["mean_return"]
    offline_std = offline_result["std_return"]

# ── Summary table ───────────────────────────────────────────────────────────
print("=" * 80)
print("OFFLINE-TO-ONLINE RL FOR SUMO BUS HOLDING — RESULTS SUMMARY")
print("=" * 80)

if offline_mean is not None:
    print(f"\nOffline-only (no fine-tune): {offline_mean:.0f} ± {offline_std:.0f}")

print(f"\n{'Method':<20} {'Seeds':>5} {'Best (mean±std)':>20} {'Final (mean±std)':>20} {'Best Epoch':>10}")
print("-" * 80)

for method in ["BC", "RE-SAC offline", "Online SAC", "WSRL", "RLPD (0.25)", "RLPD (0.50)", "RLPD (0.75)"]:
    if method not in method_data:
        continue
    seeds = method_data[method]
    n_seeds = len(seeds)

    # For each seed, find best and final return
    bests = []
    finals = []
    best_epochs = []
    for seed, data in seeds.items():
        returns = [r for _, r in data]
        epochs = [e for e, _ in data]
        best_idx = np.argmax(returns)
        bests.append(returns[best_idx])
        best_epochs.append(epochs[best_idx])
        finals.append(returns[-1])

    bests = np.array(bests)
    finals = np.array(finals)

    print(f"{method:<20} {n_seeds:>5} "
          f"{bests.mean():>8.0f} ± {bests.std():>6.0f}   "
          f"{finals.mean():>8.0f} ± {finals.std():>6.0f}   "
          f"{np.mean(best_epochs):>8.0f}")

print("-" * 80)

# ── Per-seed details ────────────────────────────────────────────────────────
print("\n\nPer-seed details:")
for method in sorted(method_data.keys()):
    print(f"\n  {method}:")
    for seed in sorted(method_data[method].keys()):
        data = method_data[method][seed]
        returns = [r for _, r in data]
        print(f"    seed={seed}: n_eval={len(data)}, "
              f"best={max(returns):.0f}, final={returns[-1]:.0f}, "
              f"mean={np.mean(returns):.0f}")

# ── Save CSV summary ────────────────────────────────────────────────────────
summary_path = os.path.join(EXP_DIR, "summary_table.csv")
with open(summary_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["method", "n_seeds", "best_mean", "best_std", "final_mean", "final_std"])
    if offline_mean is not None:
        w.writerow(["Offline-only", 1, offline_mean, offline_std, offline_mean, offline_std])
    for method in ["BC", "RE-SAC offline", "Online SAC", "WSRL", "RLPD (0.25)", "RLPD (0.50)", "RLPD (0.75)"]:
        if method not in method_data:
            continue
        seeds = method_data[method]
        bests = np.array([max(r for _, r in data) for data in seeds.values()])
        finals = np.array([data[-1][1] for data in seeds.values()])
        w.writerow([method, len(seeds), bests.mean(), bests.std(), finals.mean(), finals.std()])

print(f"\nSummary CSV saved to {summary_path}")

# ── Learning curve data (for plotting) ──────────────────────────────────────
curve_path = os.path.join(EXP_DIR, "learning_curves.csv")
with open(curve_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["method", "epoch", "mean_return", "std_return", "n_seeds"])
    for method in sorted(method_data.keys()):
        seeds = method_data[method]
        # Align by epoch
        all_epochs = sorted(set(e for data in seeds.values() for e, _ in data))
        for ep in all_epochs:
            vals = []
            for data in seeds.values():
                for e, r in data:
                    if e == ep:
                        vals.append(r)
                        break
            if vals:
                w.writerow([method, ep, np.mean(vals), np.std(vals), len(vals)])

print(f"Learning curves CSV saved to {curve_path}")
print("\nDone!")
