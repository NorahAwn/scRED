r"""
check_detection.py
==================check_baseline
STEP 2 VERIFICATION: are the reported top genes real signal or sparsity artifacts?

For each cell-type context, reports for a named set of genes:
  det_ctrl / det_pat  = fraction of nuclei with non-zero counts
  mean_ctrl / mean_pat = mean log1p(CPM)
  disp_rank            = rank by dispersion within the 3,000-gene universe
                         (1 = most dispersed = most strongly selected by HVG)

and compares them against the universe-wide detection distribution.

WHY: HVG selection here is raw Fano factor (var/mean) on log1p data, which
rises as the mean falls. Genes detected in very few nuclei can therefore enter
the universe and score highly on a deviation-based statistic without carrying
biological signal. A gene detected in <5% of nuclei should not be interpreted
as a cell-type-specific finding.

Run from F:\Norah\GeneGCN with the venv active:
    python check_detection.py
"""
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

from model_per_celltype import (
    data_normalize, attach_celltype_from_meta, load_meta,
    select_highly_variable_genes, _apply_gene_filter_adata,
    CONTROL_PATH, PATIENT_PATH, META_PATH, CELLTYPE_COL,
    TOP_N_GENES, HVG_METHOD, MIN_CELLS,
)

# Genes named in section 4.4 of the manuscript, by context.
GENES_OF_INTEREST = {
    "ALL_CELLS": [
        # myelination signature
        "MAG", "MOG", "MYRF", "RNASE1", "ERMN", "NKX6-2", "PPP1R14A",
        # other reported top-20 members
        "NPAS4", "FGFR3", "APOE", "MYO1E",
    ],
    "Oligodendrocytes": [
        # reported inflammatory axis - the genes under suspicion
        "CSF3", "IL25", "CCL4", "IL1B", "CCL2", "CXCL1", "CXCL2",
        "TNF", "IL17A", "OAS1", "STAT1", "EEF1A2",
        # canonical oligodendrocyte markers, as a positive control
        "PLP1", "MBP", "MOG", "MAG", "CNP",
    ],
    "IN-SST": [
        "GAD1", "GAD2", "SST", "CALB1", "PVALB", "NPY",
        "EGR3", "HTR2C", "ITGA8", "SNTG2", "NR1D1", "CHD7",
    ],
    "L5/6": [
        "IL6", "IL1B", "TNF",
        "FOSB", "NPAS4", "GADD45G", "FOS", "EGR1", "ARC", "JUN",
    ],
}

# Genes detected in fewer than this fraction of nuclei are flagged.
DETECTION_FLOOR = 0.05


def _dense_col(X, j):
    col = X[:, j]
    if sp.issparse(col):
        col = col.toarray()
    return np.asarray(col).ravel()


def profile_genes(ctrl, pat, universe, genes, disp_rank):
    rows = []
    index = {g: i for i, g in enumerate(universe)}
    for g in genes:
        gu = g.upper()
        if gu not in index:
            rows.append({"gene": g, "in_universe": False})
            continue
        j = index[gu]
        c = _dense_col(ctrl.X, j)
        p = _dense_col(pat.X, j)
        rows.append({
            "gene": g,
            "in_universe": True,
            "det_ctrl": float((c > 0).mean()),
            "det_pat": float((p > 0).mean()),
            "mean_ctrl": float(c.mean()),
            "mean_pat": float(p.mean()),
            "disp_rank": int(disp_rank[j]),
        })
    return pd.DataFrame(rows)


def main():
    ctrl_full = data_normalize(sc.read_h5ad(CONTROL_PATH))
    pat_full = data_normalize(sc.read_h5ad(PATIENT_PATH))
    meta = load_meta(META_PATH)
    ctrl_full = attach_celltype_from_meta(ctrl_full, meta, "control")
    pat_full = attach_celltype_from_meta(pat_full, meta, "patient")

    available = set(ctrl_full.obs[CELLTYPE_COL].dropna().unique())
    all_rows = []

    for ct, genes in GENES_OF_INTEREST.items():
        if ct == "ALL_CELLS":
            ctrl, pat = ctrl_full.copy(), pat_full.copy()
        else:
            if ct not in available:
                print(f"{ct}: NOT FOUND in metadata\n")
                continue
            ctrl = ctrl_full[ctrl_full.obs[CELLTYPE_COL] == ct].copy()
            pat = pat_full[pat_full.obs[CELLTYPE_COL] == ct].copy()
            if ctrl.n_obs < MIN_CELLS or pat.n_obs < MIN_CELLS:
                print(f"{ct}: SKIP (too few cells)\n")
                continue

        # Rebuild the exact ranking universe
        ctrl_f = _apply_gene_filter_adata(ctrl)
        pat_f = pat[:, list(ctrl_f.var_names)].copy()
        ctrl_hvg, pat_hvg, _ = select_highly_variable_genes(
            TOP_N_GENES, ctrl_f, pat_f, method=HVG_METHOD)
        universe = [str(g).upper() for g in ctrl_hvg.var_names]

        # Dispersion rank within the universe (1 = most dispersed)
        Xc = ctrl_hvg.X
        Xp = pat_hvg.X
        comb = np.vstack([
            Xc.toarray() if sp.issparse(Xc) else np.asarray(Xc),
            Xp.toarray() if sp.issparse(Xp) else np.asarray(Xp),
        ])
        means = comb.mean(axis=0)
        variances = comb.var(axis=0)
        disp = variances / (means + 1e-8)
        order = np.argsort(-disp)
        disp_rank = np.empty(len(universe), dtype=int)
        disp_rank[order] = np.arange(1, len(universe) + 1)

        # Universe-wide detection distribution, for reference
        det_all = (comb > 0).mean(axis=0)

        print(f"=== {ct} ===")
        print(f"  cells (ctrl/pat): {ctrl.n_obs} / {pat.n_obs}")
        print(f"  universe detection rate percentiles:")
        for q in (5, 25, 50, 75, 95):
            print(f"      p{q:<3d}: {100*np.percentile(det_all, q):6.2f}%")
        print(f"  genes in universe detected in <{100*DETECTION_FLOOR:.0f}% "
              f"of nuclei: {100*(det_all < DETECTION_FLOOR).mean():.1f}%")
        print()

        df = profile_genes(ctrl_hvg, pat_hvg, universe, genes, disp_rank)
        df.insert(0, "context", ct)

        with pd.option_context("display.width", 200,
                               "display.max_columns", 20):
            show = df.copy()
            for c in ("det_ctrl", "det_pat"):
                if c in show:
                    show[c] = (100 * show[c]).round(2)
            for c in ("mean_ctrl", "mean_pat"):
                if c in show:
                    show[c] = show[c].round(4)
            print(show.to_string(index=False))

        if "det_pat" in df and "det_ctrl" in df:
            sus = df[(df.in_universe == True) &
                     (df.det_ctrl < DETECTION_FLOOR) &
                     (df.det_pat < DETECTION_FLOOR)]
            if len(sus):
                print(f"\n  FLAGGED (detected in <{100*DETECTION_FLOOR:.0f}% "
                      f"of nuclei in BOTH groups):")
                print("   ", ", ".join(sus.gene.tolist()))
        missing = df[df.in_universe == False]
        if len(missing):
            print(f"\n  NOT IN UNIVERSE: {', '.join(missing.gene.tolist())}")
        print()

        all_rows.append(df)

    if all_rows:
        out = pd.concat(all_rows, ignore_index=True)
        out.to_csv("detection_check.csv", index=False)
        print("Written: detection_check.csv")


if __name__ == "__main__":
    main()
