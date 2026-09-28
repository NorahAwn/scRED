r"""
make_fig3_methods.py
====================
Figure 3 - method comparison, two panels.

(A) Mean SFARI overlap versus ranking depth, one line per method, averaged
    across the 17 cell types, with the mean measured baseline shown.
(B) Per-cell-type paired difference at k = 250 between scRED and the gene-graph
    variant, and between scRED and the MLP, sorted, to show where the graph and
    the node choice matter.

Reads ablation_summary.csv from the working directory.
Run:  python make_fig3_methods.py
Output: fig3_methods.png  (300 dpi)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv("ablation_results/ablation_summary.csv")
df = df.rename(columns={c: c.strip() for c in df.columns})

DEPTHS = [50, 100, 150, 200, 250]
METHODS = ["CELLS_AS_NODES", "FULL", "MEAN_POOL", "NO_FILTER", "NO_GRAPH", "WILCOXON_FAIR"]
LABEL = {"CELLS_AS_NODES": "scRED (cell graph)", "FULL": "gene graph",
         "MEAN_POOL": "mean/std pool", "NO_FILTER": "no filter",
         "NO_GRAPH": "MLP (no graph)", "WILCOXON_FAIR": "Wilcoxon"}
COLOR = {"CELLS_AS_NODES": "#1f4e9c", "FULL": "#3a8fb7", "MEAN_POOL": "#8bbcd4",
         "NO_FILTER": "#c0a35e", "NO_GRAPH": "#b0521f", "WILCOXON_FAIR": "#4a4a4a"}

# per-context baseline (unweighted, k=250) for the mean baseline line
BASE = {"ALL_CELLS":8.07,"L5/6-CC":9.03,"L5/6":9.23,"IN-SV2C":9.27,"L2/3":9.27,
        "IN-PV":9.53,"IN-VIP":9.60,"L4":9.97,"IN-SST":10.57,"AST-PP":11.77,
        "Endothelial":11.90,"OPC":12.40,"Neu-mat":13.07,"Microglia":13.53,
        "Neu-NRGN-I":14.20,"Oligodendrocytes":14.90,"AST-FB":15.13,"Neu-NRGN-II":15.60}

fig, (axA, axB) = plt.subplots(1, 2, figsize=(10, 4.4),
                               gridspec_kw={"width_ratios": [1, 1.15]})

# ---- Panel A: depth curves ----
for m in METHODS:
    ys = [df[(df.Ablation == m) & (df.Depth == d)]["mean"].mean() for d in DEPTHS]
    axA.plot(DEPTHS, ys, marker="o", ms=4, lw=1.8 if m == "CELLS_AS_NODES" else 1.2,
             color=COLOR[m], label=LABEL[m],
             zorder=3 if m == "CELLS_AS_NODES" else 2)
axA.axhline(np.mean(list(BASE.values())), ls="--", c="#999999", lw=1,
            label="mean baseline")
axA.set_xlabel("Ranking depth (top-k)", fontsize=9)
axA.set_ylabel("Mean SFARI overlap (%)", fontsize=9)
axA.set_xticks(DEPTHS)
axA.tick_params(labelsize=8)
axA.legend(fontsize=7, frameon=False, loc="upper right")
axA.set_title("A  Overlap vs depth", fontsize=10, loc="left", fontweight="bold")
for sp in ("top", "right"):
    axA.spines[sp].set_visible(False)

# ---- Panel B: per-context paired differences at k=250 ----
d250 = df[df.Depth == 250].pivot_table(index="CellType", columns="Ablation", values="mean")
diff_gene = (d250["CELLS_AS_NODES"] - d250["FULL"]).sort_values()
diff_mlp = (d250["CELLS_AS_NODES"] - d250["NO_GRAPH"])
order = diff_gene.index
y = np.arange(len(order))

axB.barh(y - 0.2, diff_gene.loc[order], height=0.38, color="#3a8fb7",
         label="scRED \u2212 gene graph")
axB.barh(y + 0.2, diff_mlp.loc[order], height=0.38, color="#b0521f",
         label="scRED \u2212 MLP")
axB.axvline(0, c="#333333", lw=0.8)
axB.set_yticks(y)
axB.set_yticklabels(order, fontsize=6.5)
axB.set_xlabel("Difference in SFARI overlap at k = 250 (%)", fontsize=9)
axB.tick_params(labelsize=8)
axB.legend(fontsize=7, frameon=False, loc="lower right")
axB.set_title("B  Paired differences per cell type", fontsize=10, loc="left", fontweight="bold")
for sp in ("top", "right"):
    axB.spines[sp].set_visible(False)

fig.tight_layout()
fig.savefig("fig3_methods.png", dpi=300, bbox_inches="tight")
print("wrote fig3_methods.png")
