"""
plot_eval_results.py
====================
Plot learning curves and bar charts using the post-hoc n=10 SUMO evaluation
(eval_results.csv) produced by eval_checkpoints_parallel.py.

This is the proper paper-grade evaluation: every saved checkpoint is evaluated
on 10 SUMO episodes, giving honest mean ± std across seeds.
"""

import os, json, csv
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.join(_HERE, "experiment_output")

COLORS = {
    "BC":              "#9467bd",
    "H2O+ offline":    "#17becf",
    "RE-SAC offline":  "#8c564b",
    "Online SAC":      "#7f7f7f",
    "WSRL":            "#2ca02c",
    "RLPD (0.25)":     "#ff9896",
    "RLPD (0.50)":     "#d62728",
    "RLPD (0.75)":     "#aec7e8",
}
LINESTYLES = {
    "BC":              "--",
    "H2O+ offline":    ":",
    "RE-SAC offline":  "-.",
    "Online SAC":      ":",
    "WSRL":            "-",
    "RLPD (0.25)":     "-",
    "RLPD (0.50)":     "-",
    "RLPD (0.75)":     "-",
}

# ── Load eval_results.csv ─────────────────────────────────────────────────────
df = pd.read_csv(os.path.join(EXP_DIR, "eval_results.csv"))
# Filter out model_final duplicate rows where the same seed has a ckpt at the same step
# (model_final.pt for online methods == checkpoint_epoch300.pt; we keep checkpoint_epochN)
# Drop model_final rows when a non-final checkpoint exists for the same method/seed
def filter_final_dupes(df):
    out = []
    for (method, seed), sub in df.groupby(['method', 'seed']):
        has_non_final = (sub['kind'] != 'final').any()
        if has_non_final:
            out.append(sub[sub['kind'] != 'final'])
        else:
            out.append(sub)  # BC: final only
    return pd.concat(out, ignore_index=True)

df = filter_final_dupes(df)
# For Online SAC seed 42 we may have two runs (300 and 600 epoch); dedupe by (method,seed,step) taking the best
df['_norm_step'] = df.apply(lambda r: r['step'] if r['kind'] in ('epoch','step') else 0, axis=1)
df = df.sort_values(['method', 'seed', '_norm_step']).drop_duplicates(['method', 'seed', '_norm_step'], keep='first')

# Cap epoch-based methods at max 300 so all seeds contribute (Online SAC seed-42 has a 600-ep extension)
EPOCH_MAX = 300
_is_epoch_method = df['method'].isin(['Online SAC', 'WSRL', 'RLPD (0.25)', 'RLPD (0.50)', 'RLPD (0.75)'])
df = df[~_is_epoch_method | (df['_norm_step'] <= EPOCH_MAX)]

# ── Offline-only baseline ─────────────────────────────────────────────────────
offline_json = os.path.join(EXP_DIR, "eval_offline_results.json")
offline_mean = offline_std = None
if os.path.exists(offline_json):
    with open(offline_json) as f:
        oj = json.load(f)
    offline_mean = oj["mean_return"]
    offline_std  = oj["std_return"]

# ── Summary table (best / final per method, averaged across seeds) ─────────────
print("=" * 80)
print("N=10 SUMO EVAL — POST-HOC CHECKPOINT EVALUATION")
print("=" * 80)
if offline_mean is not None:
    print(f"\nOffline-only (H2O+ AWR, no fine-tune): {offline_mean:.0f} ± {offline_std:.0f}")

METHODS = ["BC", "H2O+ offline", "RE-SAC offline", "Online SAC", "WSRL", "RLPD (0.25)", "RLPD (0.50)", "RLPD (0.75)"]
print(f"\n{'Method':<18} {'Seeds':>5} {'Best (mean±std)':>24} {'Final (mean±std)':>24}")
print("-" * 80)

summary_rows = []
for method in METHODS:
    sub = df[df['method'] == method]
    if sub.empty: continue
    # per seed: best ckpt mean_return, final ckpt mean_return
    bests, finals = [], []
    for seed, g in sub.groupby('seed'):
        bests.append(g['mean_return'].max())
        finals.append(g.sort_values('_norm_step').iloc[-1]['mean_return'])
    bests = np.array(bests); finals = np.array(finals)
    print(f"{method:<18} {len(bests):>5} "
          f"{bests.mean():>12.0f} ± {bests.std():>6.0f}   "
          f"{finals.mean():>12.0f} ± {finals.std():>6.0f}")
    summary_rows.append({'method': method, 'n_seeds': len(bests),
                         'best_mean': bests.mean(), 'best_std': bests.std(),
                         'final_mean': finals.mean(), 'final_std': finals.std()})
print("-" * 80)

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(EXP_DIR, "eval_summary_n10.csv"), index=False)
print(f"\nSaved: eval_summary_n10.csv")

# ── Figure 1: Learning curves ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# Separate step-based (BC, RE-SAC) from epoch-based (online methods)
step_methods = ["BC", "H2O+ offline", "RE-SAC offline"]
epoch_methods = ["Online SAC", "WSRL", "RLPD (0.25)", "RLPD (0.50)", "RLPD (0.75)"]

def plot_curves(ax, methods, x_col):
    for method in methods:
        sub = df[df['method'] == method]
        if sub.empty: continue
        # Aggregate across seeds at each step (skip sentinel step=0 from 'final' kind)
        agg = sub.groupby('_norm_step')['mean_return'].agg(['mean', 'std', 'count']).reset_index()
        agg = agg[agg['_norm_step'] > 0]
        if agg.empty: continue
        xs = agg['_norm_step'].values
        ys = agg['mean'].values
        stds = agg['std'].values
        ax.plot(xs, ys, label=method, color=COLORS[method], ls=LINESTYLES[method],
                lw=1.8, marker='o', ms=4)
        ax.fill_between(xs, ys - stds, ys + stds, alpha=0.15, color=COLORS[method])

plot_curves(axes[0], step_methods, 'step')
axes[0].set_xlabel("Offline training step")
axes[0].set_title("Offline methods (BC, H2O+ offline, RE-SAC offline)")

plot_curves(axes[1], epoch_methods, 'epoch')
axes[1].set_xlabel("Online training epoch")
axes[1].set_title("Online & offline-to-online methods")

for ax in axes:
    if offline_mean is not None:
        ax.axhline(offline_mean, color="black", ls="--", lw=1.2, label="H2O+ offline (no fine-tune)")
    ax.set_ylabel("Episode Return (SUMO, n=10)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='lower right')

fig.tight_layout()
out1 = os.path.join(EXP_DIR, "learning_curves_n10.pdf")
fig.savefig(out1, bbox_inches="tight")
fig.savefig(out1.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
print(f"Saved: {out1}")
plt.close(fig)

# ── Figure 2: Bar chart ────────────────────────────────────────────────────────
fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))
shown = [m for m in METHODS if m in summary_df['method'].values]
x_pos = np.arange(len(shown))
width = 0.6

for ax2, col_mean, col_std, title in zip(
    axes2, ['best_mean', 'final_mean'], ['best_std', 'final_std'],
    ['Best Return over training (mean ± std across seeds)',
     'Final Return (mean ± std across seeds)']):
    vals = [summary_df[summary_df['method']==m][col_mean].iloc[0] for m in shown]
    errs = [summary_df[summary_df['method']==m][col_std].iloc[0]  for m in shown]
    bars = ax2.bar(x_pos, vals, width, yerr=errs, capsize=4,
                   color=[COLORS[m] for m in shown], alpha=0.85, zorder=3)
    if offline_mean is not None:
        ax2.axhline(offline_mean, color="black", ls="--", lw=1.5, label="H2O+ offline")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(shown, rotation=30, ha="right", fontsize=9)
    ax2.set_ylabel("Episode Return (n=10)")
    ax2.set_title(title)
    ax2.grid(axis="y", alpha=0.3, zorder=0)
    if offline_mean is not None:
        ax2.legend(fontsize=9)

fig2.tight_layout()
out2 = os.path.join(EXP_DIR, "bar_chart_n10.pdf")
fig2.savefig(out2, bbox_inches="tight")
fig2.savefig(out2.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
print(f"Saved: {out2}")
plt.close(fig2)

# ── Figure 3: RLPD ablation ───────────────────────────────────────────────────
rlpd_methods = ["RLPD (0.25)", "RLPD (0.50)", "RLPD (0.75)"]
fig3, ax3 = plt.subplots(figsize=(8, 5))
for method in rlpd_methods:
    sub = df[df['method'] == method]
    if sub.empty: continue
    agg = sub.groupby('_norm_step')['mean_return'].agg(['mean', 'std']).reset_index()
    agg = agg[agg['_norm_step'] > 0]
    xs = agg['_norm_step'].values
    ys = agg['mean'].values
    stds = agg['std'].values
    ratio = method.split("(")[1].rstrip(")")
    ax3.plot(xs, ys, label=f"offline_ratio={ratio}", color=COLORS[method],
             lw=1.8, marker='o', ms=4)
    ax3.fill_between(xs, ys - stds, ys + stds, alpha=0.15, color=COLORS[method])
if offline_mean is not None:
    ax3.axhline(offline_mean, color="black", ls="--", lw=1.2, label="H2O+ offline (no fine-tune)")
ax3.set_xlabel("Online training epoch")
ax3.set_ylabel("Episode Return (SUMO, n=10)")
ax3.set_title("RLPD Ablation: Offline Data Ratio")
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=10)
fig3.tight_layout()
out3 = os.path.join(EXP_DIR, "rlpd_ablation_n10.pdf")
fig3.savefig(out3, bbox_inches="tight")
fig3.savefig(out3.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
print(f"Saved: {out3}")
plt.close(fig3)

print("\nAll done.")
