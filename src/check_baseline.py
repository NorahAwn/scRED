r"""
check_baseline.py
=================
STEP 1 VERIFICATION: what is the true random baseline for SFARI overlap?

Measures, per cell-type context:
  A  = size of the filtered HVG universe actually used for ranking
  B  = how many of those A genes are SFARI genes
  baseline% = B / A * 100          <- the correct "random" expectation
  plus an empirical permutation null for a top-250 draw.

Run from F:\Norah\GeneGCN:
    python check_baseline.py
"""
import numpy as np
import pandas as pd
import scanpy as sc

from model_per_celltype import (
    data_normalize, attach_celltype_from_meta, load_meta,
    select_highly_variable_genes, _apply_gene_filter_adata,
    CONTROL_PATH, PATIENT_PATH, META_PATH, CELLTYPE_COL,
    TOP_N_GENES, HVG_METHOD, MIN_CELLS,
)

SFARI_PATH = "SFARI-Gene_genes.csv"
SFARI_COL = "gene-symbol"
DEPTH = 250
N_PERM = 1000

# Start with the two that matter most; set to None to run every cell type.
CONTEXTS = ["Oligodendrocytes", "IN-SST", "L5/6"]


def main():
    sfari = set(pd.read_csv(SFARI_PATH)[SFARI_COL].astype(str).str.upper())
    print(f"SFARI symbols loaded: {len(sfari)}\n")

    ctrl_full = data_normalize(sc.read_h5ad(CONTROL_PATH))
    pat_full = data_normalize(sc.read_h5ad(PATIENT_PATH))
    meta = load_meta(META_PATH)
    ctrl_full = attach_celltype_from_meta(ctrl_full, meta, "control")
    pat_full = attach_celltype_from_meta(pat_full, meta, "patient")

    available = sorted(set(ctrl_full.obs[CELLTYPE_COL].dropna().unique()))
    print(f"Cell types found in metadata ({len(available)}):")
    print(" ", available, "\n")

    cts = CONTEXTS if CONTEXTS is not None else ["ALL_CELLS"] + available

    rows = []
    for ct in cts:
        if ct == "ALL_CELLS":
            ctrl, pat = ctrl_full.copy(), pat_full.copy()
        else:
            if ct not in available:
                print(f"{ct}: NOT FOUND in metadata - check the label and "
                      f"edit CONTEXTS at the top of this file\n")
                continue
            ctrl = ctrl_full[ctrl_full.obs[CELLTYPE_COL] == ct].copy()
            pat = pat_full[pat_full.obs[CELLTYPE_COL] == ct].copy()
            if ctrl.n_obs < MIN_CELLS or pat.n_obs < MIN_CELLS:
                print(f"{ct}: SKIP (too few cells)\n")
                continue

        # Reproduce the EXACT universe used for ranking
        ctrl_f = _apply_gene_filter_adata(ctrl)
        pat_f = pat[:, list(ctrl_f.var_names)].copy()
        ctrl_hvg, _, _ = select_highly_variable_genes(
            TOP_N_GENES, ctrl_f, pat_f, method=HVG_METHOD)
        universe = [str(g).upper() for g in ctrl_hvg.var_names]

        A = len(universe)
        hits = np.array([g in sfari for g in universe], dtype=bool)
        B = int(hits.sum())
        baseline = 100.0 * B / A if A else float("nan")

        # Empirical null: draw DEPTH genes at random from the same universe
        rng = np.random.RandomState(0)
        idx = np.arange(A)
        d = min(DEPTH, A)
        null = np.array([
            100.0 * hits[rng.choice(idx, d, replace=False)].sum() / d
            for _ in range(N_PERM)
        ])

        print(f"=== {ct} ===")
        print(f"  cells (ctrl/pat)          : {ctrl.n_obs} / {pat.n_obs}")
        print(f"  A  universe size          : {A}")
        print(f"  B  SFARI genes in universe: {B}")
        print(f"  baseline                  : {baseline:.2f}%")
        print(f"  null mean (top-{DEPTH})      : {null.mean():.2f}%")
        print(f"  null 95% interval         : "
              f"[{np.percentile(null, 2.5):.2f}%, "
              f"{np.percentile(null, 97.5):.2f}%]")
        print(f"  null max observed         : {null.max():.2f}%\n")

        rows.append({
            "context": ct, "n_ctrl": ctrl.n_obs, "n_pat": pat.n_obs,
            "A": A, "B": B, "baseline_pct": baseline,
            "null_mean": null.mean(),
            "null_lo": np.percentile(null, 2.5),
            "null_hi": np.percentile(null, 97.5),
            "null_max": null.max(),
        })

    if rows:
        pd.DataFrame(rows).to_csv("baseline_check.csv", index=False)
        print("Written: baseline_check.csv")
    else:
        print("No contexts processed.")


if __name__ == "__main__":
    main()