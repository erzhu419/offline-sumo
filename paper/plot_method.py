"""
plot_method.py
==============
RE-SAC offline method block diagram for the paper.
Simplified linear layout to avoid arrow crossing.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_PDF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures", "method_diagram.pdf")

fig, ax = plt.subplots(figsize=(11, 3.0))
ax.set_xlim(0, 11)
ax.set_ylim(0, 3)
ax.axis("off")

BOX_Y = 1.0
BOX_H = 1.2

def box(x, w, text, color="#e8eef7", edge="#2c3e50", fontsize=9, weight="normal"):
    ax.add_patch(plt.Rectangle((x, BOX_Y), w, BOX_H, linewidth=1.3,
                                edgecolor=edge, facecolor=color))
    ax.text(x + w/2, BOX_Y + BOX_H/2, text, ha="center", va="center",
            fontsize=fontsize, weight=weight)

def harrow(x1, x2, y=None, color="#444"):
    if y is None: y = BOX_Y + BOX_H/2
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.3))

# Linear flow: Dataset → Encoder → Ensemble → Stats → LCB loss → Policy update
BOXES = [
    (0.2, 1.5, "Offline dataset\n$\\mathcal{D}$\n3.1M transitions",  "#f4ecd8", "#8c6d31"),
    (2.0, 1.8, "State encoder\n(categorical +\ncontinuous, 30-d)", "#e8eef7", "#2c3e50"),
    (4.1, 1.6, "Ensemble of 10\n$Q$-networks\n(REDQ-style)",        "#e5f1e4", "#2a6f2a"),
    (6.0, 1.2, "$\\mu_Q$ and $\\sigma_Q$\n(ensemble\nstatistics)",  "#d1e9fc", "#2b6fa3"),
    (7.5, 1.6, "LCB pessimism\n$-\\!(\\mu_Q + \\beta\\sigma_Q)$\n$\\beta = -2$",
                                                                     "#fff1cc", "#8c6d31"),
    (9.4, 1.4, "Policy\n$\\pi_\\phi(a|s)$\n(tanh-Gaussian)",         "#f6e1e7", "#8c2c4b"),
]
weights = ["normal", "normal", "bold", "normal", "bold", "bold"]

xs = []
for (x, w, text, color, edge), wt in zip(BOXES, weights):
    box(x, w, text, color=color, edge=edge, weight=wt)
    xs.append((x, x+w))

# Arrows between consecutive boxes
for i in range(len(xs)-1):
    x1 = xs[i][1]
    x2 = xs[i+1][0]
    harrow(x1, x2)

# Annotation: LCB → Policy relation
ax.text((xs[-2][1]+xs[-1][0])/2, BOX_Y + BOX_H + 0.12,
        "update $\\phi$", ha="center", fontsize=8, style="italic", color="#444")

# Caption
ax.text(5.5, 0.35,
        "Actor update: offline transition $\\to$ state encoder $\\to$ $Q$-ensemble $\\to$ "
        "$(\\mu_Q, \\sigma_Q)$ $\\to$ LCB pessimism loss $\\to$ policy.",
        ha="center", va="center", fontsize=7.5, style="italic", color="#555")
ax.text(5.5, 0.15,
        "Critic update uses the standard TD loss with the same ensemble; an optional "
        "$L_1$ weight penalty (ablated in Table 3) was initially included but is dispensable.",
        ha="center", va="center", fontsize=7.5, style="italic", color="#555")

plt.tight_layout()
os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True)
plt.savefig(OUT_PDF, bbox_inches="tight")
plt.savefig(OUT_PDF.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
print(f"Saved: {OUT_PDF}")
