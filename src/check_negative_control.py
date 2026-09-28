r"""
check_negative_control.py
=========================
NEGATIVE CONTROL: is the SFARI enrichment disease signal, or an artifact of
scoring in-sample control cells against out-of-sample patient cells?

THE QUESTION
------------
cells_as_nodes_score() trains on ALL control cells, then computes

    ctrl_err  on those same control cells   (IN-sample)
    pat_err   on patient cells              (OUT-of-sample)
    RED = pat_err - ctrl_err

Any difference therefore contains a per-gene generalisation gap on top of any
disease effect. Gene fit difficulty tracks expression level and variance, and
those track how well studied a gene is - so the gap could by itself produce
SFARI enrichment with no biology involved.

THE TEST
--------
Split the CONTROL cells in half. Train on half A. Score half B exactly as if
it were the patient group. There is no disease contrast, so:

    fold ~ 1.0  ->  the asymmetry is harmless. Enrichment seen against real
                    patients is disease signal. The method stands.
    fold >> 1.0 ->  the enrichment is an artifact of the train/test split and
                    is unrelated to ASD.

Real patients are scored alongside, with the same split sizes, so the two
numbers are directly comparable.

Run from F:\Norah\GeneGCN with the venv active:
    python check_negative_control.py
"""

import numpy as np
import scanpy as sc

import cells_as_nodes as can
from model_per_celltype import (
    data_normalize, attach_celltype_from_meta, load_meta,
    select_highly_variable_genes, _apply_gene_filter_adata,
    CONTROL_PATH, PATIENT_PATH, META_PATH, CELLTYPE_COL,
    TOP_N_GENES, HVG_METHOD, MIN_CELLS,
)
from sfari_scoring import load_sfari, universe_null, score_ranked

CONTEXTS = ["L2/3", "L4", "L5/6", "L5/6-CC", "IN-PV", "IN-SST", "IN-SV2C",
            "IN-VIP", "Neu-mat", "Neu-NRGN-I", "Neu-NRGN-II", "AST-FB",
            "AST-PP", "Microglia", "Oligodendrocytes", "OPC", "Endothelial"]
SEEDS = [0, 1, 2, 3, 4]
DEPTH = 250

# Force the LEGACY configuration - the one that produced fold 1.98 on L5/6.
can.SCORE_MODE = "raw"
can.HOLDOUT_FRAC = 0.0
can.MASK_RATE = 0.0


def _dense(a):
    return a.toarray() if hasattr(a, "toarray") else np.asarray(a)


def run_context(ct, ctrl_full, pat_full, sfari, sfari_w):
    ctrl = ctrl_full[ctrl_full.obs[CELLTYPE_COL] == ct].copy()
    pat = pat_full[pat_full.obs[CELLTYPE_COL] == ct].copy()
    if ctrl.n_obs < MIN_CELLS or pat.n_obs < MIN_CELLS:
        print(f"{ct}: SKIP (too few cells)")
        return

    ctrl_f = _apply_gene_filter_adata(ctrl)
    pat_f = pat[:, list(ctrl_f.var_names)].copy()
    ctrl_hvg, pat_hvg, _ = select_highly_variable_genes(
        TOP_N_GENES, ctrl_f, pat_f, method=HVG_METHOD)

    genes = [str(g) for g in ctrl_hvg.var_names]
    Xc = _dense(ctrl_hvg.X).astype(np.float32)
    Xp = _dense(pat_hvg.X).astype(np.float32)

    null = universe_null(genes, sfari, sfari_w, [DEPTH], seed=0)
    base = null[DEPTH]["pct_baseline"]

    print(f"\n=== {ct} ===")
    print(f"  control cells {Xc.shape[0]}   patient cells {Xp.shape[0]}")
    print(f"  SFARI baseline {base:.2f}%  "
          f"(null 95% [{null[DEPTH]['pct_lo']:.2f}, {null[DEPTH]['pct_hi']:.2f}])")

    neg, real = [], []
    for seed in SEEDS:
        rs = np.random.RandomState(1000 + seed)
        perm = rs.permutation(Xc.shape[0])
        half = Xc.shape[0] // 2
        A, B = Xc[perm[:half]], Xc[perm[half:]]

        # --- NEGATIVE CONTROL: control-A trained, control-B scored as "patient"
        _, absred_n, _, _ = can.cells_as_nodes_score(A, B, genes, seed=seed)
        ranked_n = [genes[i] for i in np.argsort(-absred_n)]
        s_n = score_ranked(ranked_n, sfari, sfari_w, null, [DEPTH])[DEPTH]

        # --- REAL: control-A trained, patients scored (same train size)
        _, absred_r, _, _ = can.cells_as_nodes_score(A, Xp, genes, seed=seed)
        ranked_r = [genes[i] for i in np.argsort(-absred_r)]
        s_r = score_ranked(ranked_r, sfari, sfari_w, null, [DEPTH])[DEPTH]

        neg.append(s_n["SFARI_pct_fold"])
        real.append(s_r["SFARI_pct_fold"])
        print(f"  seed={seed}   NEG ctrl-vs-ctrl fold={s_n['SFARI_pct_fold']:.2f}"
              f"   |   REAL ctrl-vs-patient fold={s_r['SFARI_pct_fold']:.2f}")

    mn, mr = float(np.mean(neg)), float(np.mean(real))
    print(f"  ---------------------------------------------------")
    print(f"  MEAN   negative control : {mn:.2f}")
    print(f"  MEAN   real contrast    : {mr:.2f}")
    if mn >= 1.15:
        print(f"  >> The split alone produces {mn:.2f}x enrichment with NO disease.")
    elif mr - mn >= 0.25:
        print(f"  >> Disease contrast adds {mr - mn:+.2f} over the null split. "
              f"Signal looks real.")
    else:
        print(f"  >> Negative control near 1.0 and little separation; "
              f"inconclusive at this n.")


def main():
    sfari, sfari_w = load_sfari()
    print(f"SFARI genes loaded: {len(sfari)}")

    ctrl_full = data_normalize(sc.read_h5ad(CONTROL_PATH))
    pat_full = data_normalize(sc.read_h5ad(PATIENT_PATH))
    meta = load_meta(META_PATH)
    ctrl_full = attach_celltype_from_meta(ctrl_full, meta, "control")
    pat_full = attach_celltype_from_meta(pat_full, meta, "patient")

    print(f"\n[config] score={can.SCORE_MODE} holdout={can.HOLDOUT_FRAC} "
          f"mask={can.MASK_RATE}  (legacy settings)")

    for ct in CONTEXTS:
        run_context(ct, ctrl_full, pat_full, sfari, sfari_w)


if __name__ == "__main__":
    main()
