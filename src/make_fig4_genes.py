r"""
make_fig4_genes.py
==================
Figure 4 - recovered genes in four neuronal contexts.

For four neuronal cell types, the top-15 scRED genes are listed with direction
of change (up/down) and SFARI status (filled marker = curated). Reads the ranked
lists from ablation_rankings.csv so the figure always matches the run.

Run:  python make_fig4_genes.py
Output: fig4_genes.png  (300 dpi)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

CONTEXTS = ["L5/6", "IN-SST", "IN-VIP", "Neu-mat"]
TOPN = 15

df = pd.read_csv("ablation_results/ablation_rankings.csv")
df = df.rename(columns={c: c.strip() for c in df.columns})
df = df[df.Method == "CELLS_AS_NODES"]

fig, axes = plt.subplots(1, 4, figsize=(11, 4.6))

for ax, ct in zip(axes, CONTEXTS):
    sub = df[df.CellType == ct].sort_values("Rank").head(TOPN)
    genes = list(sub["Gene"])[::-1]
    dirs = list(sub["Direction"])[::-1]
    sfari = [str(s).strip().lower() == "true" for s in sub["SFARI"]][::-1]
    y = range(len(genes))

    for i, (g, d, sf) in enumerate(zip(genes, dirs, sfari)):
        up = str(d).strip().lower() == "up"
        col = "#b0521f" if up else "#1f4e9c"
        ax.scatter(0, i, marker="^" if up else "v", s=60,
                   facecolor=col if sf else "white",
                   edgecolor=col, linewidth=1.3, zorder=3)
        ax.text(0.12, i, g, va="center", ha="left", fontsize=7.5,
                fontweight="bold" if sf else "normal",
                color="#111111" if sf else "#444444")

    ax.set_xlim(-0.15, 1.2)
    ax.set_ylim(-0.8, len(genes) - 0.2)
    ax.set_title(ct, fontsize=10, fontweight="bold")
    ax.axis("off")

# legend
from matplotlib.lines import Line2D
handles = [
    Line2D([0],[0],marker="^",color="w",markerfacecolor="#b0521f",markeredgecolor="#b0521f",markersize=8,label="Up, SFARI"),
    Line2D([0],[0],marker="^",color="w",markerfacecolor="white",markeredgecolor="#b0521f",markersize=8,label="Up, not SFARI"),
    Line2D([0],[0],marker="v",color="w",markerfacecolor="#1f4e9c",markeredgecolor="#1f4e9c",markersize=8,label="Down, SFARI"),
    Line2D([0],[0],marker="v",color="w",markerfacecolor="white",markeredgecolor="#1f4e9c",markersize=8,label="Down, not SFARI"),
]
fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8,
           frameon=False, bbox_to_anchor=(0.5, -0.02))

fig.suptitle("Top-15 scRED genes per neuronal context (filled = SFARI-curated)",
             fontsize=10, y=1.02)
fig.tight_layout(rect=[0, 0.04, 1, 1])
fig.savefig("fig4_genes.png", dpi=300, bbox_inches="tight")
print("wrote fig4_genes.png")
