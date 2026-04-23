"""
plot_results.py
===============
Plot learning curves and bar chart for offline-to-online RL experiments.

Reads per-seed train_log.csv (and train_log_eval.csv for step-based methods),
produces:
  1. learning_curves.pdf  — smoothed mean±std shading per method
  2. bar_chart.pdf        — best and final return comparison

Run:
    cd /home/erzhu419/mine_code/offline-sumo
    python plot_results.py
"""

import os, re, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

_HERE = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.join(_HERE, "experiment_output")

# ── Color palette ────────────────────────────────────────────────────────────
COLORS = {
    "BC":              "#9467bd",
    "RE-SAC offline":  "#8c564b",
    "Online SAC":      "#7f7f7f",
    "WSRL":            "#2ca02c",
    "RLPD (0.25)":     "#ff9896",
    "RLPD (0.50)":     "#d62728",
    "RLPD (0.75)":     "#aec7e8",
}
LINESTYLES = {
    "BC":              "--",
    "RE-SAC offline":  "-.",
    "Online SAC":      ":",
    "WSRL":            "-",
    "RLPD (0.25)":     "-",
    "RLPD (0.50)":     "-",
    "RLPD (0.75)":     "-",
}

# ── Directory → method mapping ────────────────────────────────────────────────
PATTERNS = [
    ("BC",              re.compile(r"bc_seed(\d+)_"),             "step",  "train_log_eval.csv"),
    ("RE-SAC offline",  re.compile(r"resac_offline_seed(\d+)_"),  "step",  "train_log_eval.csv"),
    ("Online SAC",      re.compile(r"online_seed(\d+)_"),         "epoch", "train_log.csv"),
    ("WSRL",            re.compile(r"wsrl_seed(\d+)_"),           "epoch", "train_log.csv"),
    ("RLPD (0.50)",     re.compile(r"rlpd_seed(\d+)_"),           "epoch", "train_log.csv"),
    ("RLPD (0.25)",     re.compile(r"rlpd_ratio0\.25_seed(\d+)_"),"epoch", "train_log.csv"),
    ("RLPD (0.75)",     re.compile(r"rlpd_ratio0\.75_seed(\d+)_"),"epoch", "train_log.csv"),
]

def detect(dirname):
    for method, pat, x_key, fname in PATTERNS:
        m = pat.match(dirname)
        if m:
            return method, int(m.group(1)), x_key, fname
    return None, None, None, None

def load_eval(csv_path, x_key):
    rows = []
    with open(csv_path) as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                x = int(float(row.get(x_key, 0)))
                ret = float(row["eval_return"])
                rows.append((x, ret))
            except (ValueError, TypeError, KeyError):
                continue
    # keep only rows where eval_return is non-trivially zero (skip warmup zeros)
    if rows:
        non_zero = [(x, r) for x, r in rows if r != 0.0]
        if non_zero:
            rows = non_zero
    return rows


# ── Collect data ──────────────────────────────────────────────────────────────
data = {}  # {method: {seed: [(x, return)]}}

for dirname in sorted(os.listdir(EXP_DIR)):
    method, seed, x_key, fname = detect(dirname)
    if method is None:
        continue
    csv_path = os.path.join(EXP_DIR, dirname, fname)
    if not os.path.exists(csv_path):
        continue
    rows = load_eval(csv_path, x_key)
    if not rows:
        continue
    data.setdefault(method, {})[seed] = rows

# ── Offline-only baseline ─────────────────────────────────────────────────────
offline_json = os.path.join(EXP_DIR, "eval_offline_results.json")
offline_mean = offline_std = None
if os.path.exists(offline_json):
    import json
    with open(offline_json) as f:
        oj = json.load(f)
    offline_mean = oj["mean_return"]
    offline_std  = oj["std_return"]

# ── Helpers ───────────────────────────────────────────────────────────────────
def smooth(y, w=5):
    if len(y) < w:
        return np.array(y, dtype=float)
    out = np.convolve(y, np.ones(w)/w, mode='valid')
    pad_l = (len(y) - len(out)) // 2
    pad_r = len(y) - len(out) - pad_l
    return np.concatenate([y[:pad_l], out, y[-pad_r:]]) if pad_r else np.concatenate([y[:pad_l], out])

def align(seeds_dict, n_points=None):
    """Return (common_x, matrix [n_seeds × n_points]) aligned by index."""
    all_xy = list(seeds_dict.values())
    min_len = min(len(s) for s in all_xy)
    if n_points:
        min_len = min(min_len, n_points)
    xs = np.array([s[min_len - 1][0] for s in all_xy])
    # Use the x-axis from the longest seed at evenly-spaced indices
    ref = all_xy[0]
    idx = np.round(np.linspace(0, len(ref) - 1, min_len)).astype(int)
    x_axis = np.array([ref[i][0] for i in idx])
    mat = []
    for s in all_xy:
        sidx = np.round(np.linspace(0, len(s) - 1, min_len)).astype(int)
        mat.append([s[i][1] for i in sidx])
    return x_axis, np.array(mat)


METHOD_ORDER = ["BC", "RE-SAC offline", "Online SAC", "WSRL", "RLPD (0.25)", "RLPD (0.50)", "RLPD (0.75)"]

# ── Figure 1: Learning curves ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for method in METHOD_ORDER:
    if method not in data:
        continue
    seeds = data[method]
    x_axis, mat = align(seeds)
    mean_r = np.mean(mat, axis=0)
    std_r  = np.std(mat, axis=0)
    sm = smooth(mean_r, w=3)
    color = COLORS[method]
    ls    = LINESTYLES[method]

    # Left: x = epoch/step index (normalised 0-100%)
    pct = np.linspace(0, 100, len(x_axis))
    axes[0].plot(pct, sm, label=method, color=color, ls=ls, lw=1.8)
    axes[0].fill_between(pct, sm - std_r, sm + std_r, alpha=0.15, color=color)

    # Right: raw x axis
    axes[1].plot(x_axis, sm, label=method, color=color, ls=ls, lw=1.8)
    axes[1].fill_between(x_axis, sm - std_r, sm + std_r, alpha=0.15, color=color)

for ax in axes:
    if offline_mean is not None:
        ax.axhline(offline_mean, color="black", ls="--", lw=1.2, label="Offline policy (no fine-tune)")
    ax.set_ylabel("Episode Return")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)

axes[0].set_xlabel("Training Progress (%)")
axes[0].set_title("Learning Curves (normalised axis)")
axes[1].set_xlabel("Epoch / Step")
axes[1].set_title("Learning Curves (raw axis)")

fig.tight_layout()
out1 = os.path.join(EXP_DIR, "learning_curves.pdf")
fig.savefig(out1, bbox_inches="tight")
print(f"Saved: {out1}")
fig.savefig(out1.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# ── Figure 2: Bar chart (best return) ─────────────────────────────────────────
fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))

def method_stats(method):
    if method not in data:
        return None, None, None, None
    seeds = data[method]
    bests  = [max(r for _, r in v) for v in seeds.values()]
    finals = [v[-1][1]             for v in seeds.values()]
    return np.mean(bests), np.std(bests), np.mean(finals), np.std(finals)

shown = [m for m in METHOD_ORDER if m in data]
x_pos = np.arange(len(shown))
width = 0.6

for ax2, stat_idx, title in zip(axes2, [0, 2], ["Best Return (mean ± std)", "Final Return (mean ± std)"]):
    vals, errs = [], []
    for method in shown:
        s = method_stats(method)
        vals.append(s[stat_idx])
        errs.append(s[stat_idx + 1])
    bars = ax2.bar(x_pos, vals, width, yerr=errs, capsize=4,
                   color=[COLORS[m] for m in shown], alpha=0.8, zorder=3)
    if offline_mean is not None:
        ax2.axhline(offline_mean, color="black", ls="--", lw=1.5, label="Offline policy")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(shown, rotation=30, ha="right", fontsize=9)
    ax2.set_ylabel("Episode Return")
    ax2.set_title(title)
    ax2.grid(axis="y", alpha=0.3, zorder=0)
    if offline_mean is not None:
        ax2.legend(fontsize=9)

fig2.tight_layout()
out2 = os.path.join(EXP_DIR, "bar_chart.pdf")
fig2.savefig(out2, bbox_inches="tight")
print(f"Saved: {out2}")
fig2.savefig(out2.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
plt.close(fig2)

# ── Figure 3: RLPD ablation (offline ratio) ───────────────────────────────────
rlpd_methods = ["RLPD (0.25)", "RLPD (0.50)", "RLPD (0.75)"]
rlpd_available = [m for m in rlpd_methods if m in data]
if len(rlpd_available) >= 2:
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    for method in rlpd_available:
        x_axis, mat = align(data[method])
        mean_r = np.mean(mat, axis=0)
        std_r  = np.std(mat, axis=0)
        sm = smooth(mean_r, w=3)
        pct = np.linspace(0, 100, len(x_axis))
        ratio = method.split("(")[1].rstrip(")")
        color = COLORS[method]
        ax3.plot(pct, sm, label=f"offline_ratio={ratio}", color=color, lw=1.8)
        ax3.fill_between(pct, sm - std_r, sm + std_r, alpha=0.15, color=color)
    if offline_mean is not None:
        ax3.axhline(offline_mean, color="black", ls="--", lw=1.2, label="Offline only")
    ax3.set_xlabel("Training Progress (%)")
    ax3.set_ylabel("Episode Return")
    ax3.set_title("RLPD Ablation: Offline Data Ratio")
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10)
    fig3.tight_layout()
    out3 = os.path.join(EXP_DIR, "rlpd_ablation.pdf")
    fig3.savefig(out3, bbox_inches="tight")
    print(f"Saved: {out3}")
    fig3.savefig(out3.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig3)

print("\nAll figures done.")
