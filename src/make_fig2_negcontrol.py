r"""
make_fig2_negcontrol.py
=======================
Figure 2 - negative control (donor-level).

Real disease contrast (fold over per-context baseline) plotted against the
matched, donor-disjoint control-versus-control negative control, one point per
cell type, with per-partition SD whiskers. Points above the diagonal are
contexts where disease adds signal beyond the scoring asymmetry.

REPRODUCIBLE: reads negctrl_donorlevel.csv (produced by
check_negative_control_donor.py) instead of hardcoded values, so the figure
always matches the run that generated the table.

Run:  python make_fig2_negcontrol.py
Output: fig2_negcontrol.png  (300 dpi)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

CSV = "negctrl_donorlevel.csv"          # columns: context, neg, real, diff,
                                        # n_donors, n_seeds, neg_sd, real_sd
OUT = "fig2_negcontrol.png"

NEURONAL = {"L2/3", "L4", "L5/6", "L5/6-CC",
            "IN-PV", "IN-SST", "IN-SV2C", "IN-VIP", "Neu-mat"}
NRGN = {"Neu-NRGN-I", "Neu-NRGN-II"}


def group_of(ct):
    if ct in NEURONAL:
        return "neuronal"
    if ct in NRGN:
        return "NRGN"
    return "glial"


STYLE = {
    "neuronal": dict(color="#1f77b4", marker="o", label="Neuronal"),
    "glial":    dict(color="#e0a020", marker="s", label="Glial / endothelial"),
    "NRGN":     dict(color="#2ca089", marker="^", label="NRGN"),
}


def main():
    df = pd.read_csv(CSV).dropna(subset=["neg", "real"])
    # tolerate missing SD columns (older runs)
    for c in ("neg_sd", "real_sd"):
        if c not in df.columns:
            df[c] = 0.0
    df[["neg_sd", "real_sd"]] = df[["neg_sd", "real_sd"]].fillna(0.0)

    fig, ax = plt.subplots(figsize=(7.2, 7.0))

    lo = float(min(df["neg"].min(), df["real"].min())) - 0.1
    hi = float(max(df["neg"].max(), df["real"].max())) + 0.1
    ax.plot([lo, hi], [lo, hi], ls="--", color="0.5", lw=1.2, zorder=1)
    ax.fill_between([lo, hi], [lo, hi], hi, color="#1f77b4", alpha=0.05, zorder=0)
    ax.text(hi, hi, "  real > neg", ha="right", va="top",
            style="italic", color="0.5", fontsize=10)

    seen = set()
    for _, r in df.iterrows():
        g = group_of(r["context"])
        st = STYLE[g]
        lbl = st["label"] if g not in seen else None
        seen.add(g)
        ax.errorbar(r["neg"], r["real"],
                    xerr=r["neg_sd"], yerr=r["real_sd"],
                    fmt=st["marker"], color=st["color"], ms=9,
                    ecolor=st["color"], elinewidth=1, capsize=0,
                    alpha=0.9, label=lbl, zorder=3)
        ax.annotate(str(r["context"]), (r["neg"], r["real"]),
                    xytext=(6, 2), textcoords="offset points", fontsize=9)

    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Negative-control enrichment (fold over baseline)")
    ax.set_ylabel("Real disease contrast (fold over baseline)")
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(OUT, dpi=300)
    print(f"wrote {OUT} from {CSV} ({len(df)} contexts)")


if __name__ == "__main__":
    main()
