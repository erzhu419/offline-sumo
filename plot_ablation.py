"""
plot_ablation.py
================
Learning curves for the four RE-SAC ablation variants, so you can see at a glance
why LCB is the dominant ingredient.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.join(_HERE, "experiment_output")
OUT_PDF = os.path.join(_HERE, "paper", "figures", "ablation_curves.pdf")
OUT_PNG = OUT_PDF.replace(".pdf", ".png")

df = pd.read_csv(os.path.join(EXP_DIR, "eval_results.csv"))
df['_norm_step'] = df.apply(lambda r: r['step'] if r['kind'] in ('epoch','step') else 0, axis=1)
df = df[df['_norm_step'] > 0]

VARIANTS = [
    ("RE-SAC no reg",  "Ensemble only (no LCB, no $L_1$)", "#9467bd", ":"),
    ("RE-SAC no LCB",  "$L_1$ only ($\\beta=0$)",            "#ff7f0e", "--"),
    ("RE-SAC no L1",   "LCB only ($\\lambda=0$)",            "#2ca02c", "-"),
    ("RE-SAC offline", "Full (LCB $+$ $L_1$)",               "#d62728", "-."),
]

fig, ax = plt.subplots(figsize=(7.5, 4.5))
for method, label, color, ls in VARIANTS:
    sub = df[df['method'] == method]
    if sub.empty: continue
    agg = sub.groupby('_norm_step')['mean_return'].agg(['mean', 'std']).reset_index()
    xs   = agg['_norm_step'].values
    ys   = agg['mean'].values
    stds = agg['std'].values
    ax.plot(xs, ys, label=label, color=color, ls=ls, lw=1.8, marker='o', ms=4)
    ax.fill_between(xs, ys - stds, ys + stds, alpha=0.15, color=color)

ax.set_xlabel("Offline training step")
ax.set_ylabel("Episode return (SUMO, $N=10$)")
ax.set_title("RE-SAC regularizer ablation (3 seeds, post-hoc $N=10$ eval)")
ax.grid(True, alpha=0.3)
ax.legend(loc='lower right', fontsize=9)
plt.tight_layout()
os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True)
plt.savefig(OUT_PDF, bbox_inches="tight")
plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
print(f"Saved: {OUT_PDF}")
