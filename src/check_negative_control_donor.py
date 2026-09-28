r"""
check_negative_control_donor.py
===============================
DONOR-LEVEL negative control.

WHY THIS VERSION
----------------
The earlier negative control split CONTROL CELLS at random, so the same donors
sat on both sides. The real contrast, by contrast, is patient donors vs control
donors. That asymmetry means donor and batch effects could be counted as
"disease". This version removes it: the control DONORS are split into two
halves, so every comparison is between disjoint sets of people.

DESIGN (matched training set, donor-disjoint evaluation)
--------------------------------------------------------
Per cell-type context, per seed:
  * Shuffle the control donors, split into group A and group B.
  * Train the model on the cells of group-A donors only.
  * NEGATIVE control : score group-B donors' cells as if they were patients.
                       No disease -> enrichment here is train/test asymmetry
                       plus any residual donor effect that survives a
                       donor-disjoint split.
  * REAL contrast    : score the real patient cells with the SAME group-A model.
Both evaluated sets are donor-disjoint from the training donors, and both use
the same training set, so the two folds are directly comparable.

Read the two MEAN lines per context exactly as before.

Run from F:\Norah\GeneGCN with the venv active:
    python check_negative_control_donor.py > negctrl_donorlevel.log 2>&1
"""

import numpy as np
import pandas as pd
import scanpy as sc

import cells_as_nodes as can
import model_per_celltype as mpc
from model_per_celltype import (
    data_normalize, attach_celltype_from_meta, load_meta,
    select_highly_variable_genes, _apply_gene_filter_adata,
    CONTROL_PATH, PATIENT_PATH, META_PATH, CELLTYPE_COL,
    TOP_N_GENES, HVG_METHOD, MIN_CELLS,
)
from sfari_scoring import load_sfari, universe_null, score_ranked

# All 17 cell-type contexts (ALL_CELLS is excluded, as in the paper's n = 17).
CONTEXTS = ["L2/3", "L4", "L5/6", "L5/6-CC", "IN-PV", "IN-SST", "IN-SV2C",
            "IN-VIP", "Neu-mat", "Neu-NRGN-I", "Neu-NRGN-II", "AST-FB",
            "AST-PP", "Microglia", "Oligodendrocytes", "OPC", "Endothelial"]
SEEDS = [0, 1, 2, 3, 4]
DEPTH = 250

DONOR_COL = "individual"        # the donor column in meta.txt
MIN_DONORS = 4                  # need at least this many control donors to split
MIN_SIDE_CELLS = 20             # min cells on each donor side

# Legacy scoring configuration — the one the paper reports.
can.SCORE_MODE = "raw"
can.HOLDOUT_FRAC = 0.0
can.MASK_RATE = 0.0

# Make attach_celltype_from_meta carry the donor label onto every cell.
mpc.DONOR_COL = DONOR_COL


def _dense(a):
    return a.toarray() if hasattr(a, "toarray") else np.asarray(a)


def run_context(ct, ctrl_full, pat_full, sfari, sfari_w, rows):
    ctrl = ctrl_full[ctrl_full.obs[CELLTYPE_COL] == ct].copy()
    pat = pat_full[pat_full.obs[CELLTYPE_COL] == ct].copy()
    if ctrl.n_obs < MIN_CELLS or pat.n_obs < MIN_CELLS:
        print(f"\n=== {ct} ===\n  SKIP (too few cells)")
        return

    if DONOR_COL not in ctrl.obs:
        print(f"\n=== {ct} ===\n  SKIP (donor column '{DONOR_COL}' not attached)")
        return

    ctrl_f = _apply_gene_filter_adata(ctrl)
    pat_f = pat[:, list(ctrl_f.var_names)].copy()
    ctrl_hvg, pat_hvg, _ = select_highly_variable_genes(
        TOP_N_GENES, ctrl_f, pat_f, method=HVG_METHOD)

    genes = [str(g) for g in ctrl_hvg.var_names]
    Xc = _dense(ctrl_hvg.X).astype(np.float32)
    Xp = _dense(pat_hvg.X).astype(np.float32)
    donors_c = np.asarray(ctrl_hvg.obs[DONOR_COL].values)
    uniq = np.unique(donors_c)

    null = universe_null(genes, sfari, sfari_w, [DEPTH], seed=0)
    base = null[DEPTH]["pct_baseline"]

    print(f"\n=== {ct} ===")
    print(f"  control cells {Xc.shape[0]} ({len(uniq)} donors)   "
          f"patient cells {Xp.shape[0]}")
    print(f"  SFARI baseline {base:.2f}%")

    if len(uniq) < MIN_DONORS:
        print(f"  SKIP (only {len(uniq)} control donors, need >= {MIN_DONORS})")
        rows.append(dict(context=ct, neg=np.nan, real=np.nan,
                         n_donors=len(uniq), note="too few donors"))
        return

    neg, real = [], []
    for seed in SEEDS:
        rs = np.random.RandomState(1000 + seed)
        shuffled = rs.permutation(uniq)
        half = len(shuffled) // 2
        gA = set(shuffled[:half])          # training donors
        gB = set(shuffled[half:])          # held-out control donors (fake patient)

        inA = np.array([d in gA for d in donors_c])
        inB = ~inA
        if inA.sum() < MIN_SIDE_CELLS or inB.sum() < MIN_SIDE_CELLS:
            print(f"  seed={seed}  SKIP (uneven donor split: "
                  f"{inA.sum()} / {inB.sum()} cells)")
            continue

        A = Xc[inA]        # train
        B = Xc[inB]        # negative-control "patient" (donor-disjoint from A)

        # NEGATIVE control: train on donor-group A, score donor-group B
        _, absred_n, _, _ = can.cells_as_nodes_score(A, B, genes, seed=seed)
        ranked_n = [genes[i] for i in np.argsort(-absred_n)]
        s_n = score_ranked(ranked_n, sfari, sfari_w, null, [DEPTH])[DEPTH]

        # REAL contrast: train on donor-group A, score real patients
        _, absred_r, _, _ = can.cells_as_nodes_score(A, Xp, genes, seed=seed)
        ranked_r = [genes[i] for i in np.argsort(-absred_r)]
        s_r = score_ranked(ranked_r, sfari, sfari_w, null, [DEPTH])[DEPTH]

        neg.append(s_n["SFARI_pct_fold"])
        real.append(s_r["SFARI_pct_fold"])
        print(f"  seed={seed}  donorsA={len(gA)} cellsA={int(inA.sum())}  "
              f"NEG fold={s_n['SFARI_pct_fold']:.2f}  |  "
              f"REAL fold={s_r['SFARI_pct_fold']:.2f}")

    if not neg:
        print("  (no valid splits)")
        rows.append(dict(context=ct, neg=np.nan, real=np.nan,
                         n_donors=len(uniq), note="no valid split"))
        return

    mn, mr = float(np.mean(neg)), float(np.mean(real))
    print(f"  ---------------------------------------------------")
    print(f"  MEAN   negative control : {mn:.2f}")
    print(f"  MEAN   real contrast    : {mr:.2f}")
    rows.append(dict(context=ct, neg=round(mn, 3), real=round(mr, 3),
                     diff=round(mr - mn, 3), n_donors=len(uniq),
                     n_seeds=len(neg),
                     neg_sd=round(float(np.std(neg, ddof=1)), 3) if len(neg) > 1 else 0.0,
                     real_sd=round(float(np.std(real, ddof=1)), 3) if len(real) > 1 else 0.0))


def main():
    sfari, sfari_w = load_sfari()
    print(f"SFARI genes loaded: {len(sfari)}")

    ctrl_full = data_normalize(sc.read_h5ad(CONTROL_PATH))
    pat_full = data_normalize(sc.read_h5ad(PATIENT_PATH))
    meta = load_meta(META_PATH)
    ctrl_full = attach_celltype_from_meta(ctrl_full, meta, "control")
    pat_full = attach_celltype_from_meta(pat_full, meta, "patient")

    if DONOR_COL not in ctrl_full.obs:
        raise SystemExit(
            f"Donor column '{DONOR_COL}' was not attached to the control cells. "
            f"Check that meta.txt has an '{DONOR_COL}' column and that "
            f"model_per_celltype.DONOR_COL can be set.")

    print(f"[config] score={can.SCORE_MODE} holdout={can.HOLDOUT_FRAC} "
          f"mask={can.MASK_RATE}  |  donor-level split on '{DONOR_COL}'")
    print(f"[config] control donors total: "
          f"{ctrl_full.obs[DONOR_COL].nunique()}")

    rows = []
    for ct in CONTEXTS:
        run_context(ct, ctrl_full, pat_full, sfari, sfari_w, rows)

    df = pd.DataFrame(rows)
    df.to_csv("negctrl_donorlevel.csv", index=False)
    print("\n================ SUMMARY (donor-level) ================")
    print(df.to_string(index=False))
    print("\nWritten: negctrl_donorlevel.csv")

    # neuronal vs the rest, quick paired test
    try:
        from scipy.stats import wilcoxon
        neuronal = {"L2/3", "L4", "L5/6", "L5/6-CC", "IN-PV", "IN-SST",
                    "IN-SV2C", "IN-VIP", "Neu-mat"}
        sub = df.dropna(subset=["neg", "real"])
        for name, keep in [("neuronal", sub["context"].isin(neuronal)),
                           ("all 17", sub["context"].notna())]:
            s = sub[keep]
            if len(s) >= 6:
                w, p = wilcoxon(s["real"], s["neg"])
                wins = int((s["real"] > s["neg"]).sum())
                print(f"{name:9s} n={len(s)}  NEG {s['neg'].mean():.2f}  "
                      f"REAL {s['real'].mean():.2f}  wins {wins}/{len(s)}  "
                      f"Wilcoxon p={p:.4f}")
    except Exception as e:
        print(f"(paired test skipped: {e})")


if __name__ == "__main__":
    main()
