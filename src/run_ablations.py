"""
run_ablations.py
================

Benchmark driver for the 5 ablation methods. Optimized to run gene filtering
and HVG selection ONCE per cell type instead of inside the ablation/seed loops.
Updated to support modern AnnData (>=0.10) concatenation syntax.

STATISTICAL CORRECTIONS (v2)
----------------------------
1. Paired tests exclude ALL_CELLS because it is a superset of the other
   cell-type groups and therefore not independent.
2. Cell type is treated as the experimental unit; seeds are technical
   replicates (affecting model init + training only).
3. Pairwise Wilcoxon p-values are Holm-Bonferroni corrected per depth.
4. A Friedman omnibus test is reported before pairwise post-hoc tests.
5. SFARI top-K overlap is tested against a hypergeometric null
   (universe = the method's own HVG gene set).
6. Confidence intervals (t-based, over seeds) are reported for every
   (cell type, method, depth) cell.
7. Direction ("Up"/"Down") is computed as a proper log2 fold-change on
   the linear CPM scale, not as a difference of log1p means.

Methods
-------
FULL           : gene-node GCN + 64-quantile pool + filter   ("GeneGCN gene-graph")
NO_GRAPH       : MLP autoencoder, same shape, no graph
MEAN_POOL      : gene-node GCN + mean/std pool
NO_FILTER      : gene-node GCN, no gene-symbol filter
WILCOXON_FAIR  : Scanpy rank_genes_groups on the same filtered HVG universe
CELLS_AS_NODES : cell-node GCN autoencoder  ("the cell-graph method")

Usage
-----
python run_ablations.py --celltypes all --seeds 5
"""

from __future__ import annotations
import argparse, os, time, gc

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

from scipy.stats import (
    wilcoxon as wilcoxon_test,
    hypergeom,
    friedmanchisquare,
    rankdata,
    t as t_dist,
)

from gene_filter import filter_artifact_genes

# Reuse data loading from your working pipeline
import cells_as_nodes as can
from cells_as_nodes import (
    data_normalize, load_meta, attach_celltype_from_meta,
    select_highly_variable_genes,
    CONTROL_PATH, PATIENT_PATH, META_PATH, CELLTYPE_COL,
    SFARI_PATH, SFARI_SYMBOL_COL, DEPTHS, MIN_CELLS,
    TOP_N_GENES, EPOCHS, LR, DROPOUT, DEVICE,
    sfari_overlap, load_sfari,
)

# [SINGLE-PIPELINE] The cell-graph method is run here as a sixth ablation so
# that every method shares ONE gene universe, ONE preprocessing path and ONE
# seed scheme. Force the legacy scoring configuration, which is the one the
# paper reports.
can.SCORE_MODE   = "raw"
can.HOLDOUT_FRAC = 0.0
can.MASK_RATE    = 0.0

# --------------------------------------------------------------------------- #
# DEVICE RESOLUTION (GPU Auto-Detection)
# --------------------------------------------------------------------------- #
device = torch.device("cuda" if torch.cuda.is_available() else DEVICE)
print(f"--> Active execution device: {device}")

# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #
OUT_DIR        = "ablation_results"
CORR_THRESHOLD = 0.3       # gene-gene co-expression threshold
N_POOL         = 64        # quantile bins for gene-node pooling

# ---- Statistical config ---------------------------------------------------- #
ALPHA          = 0.05      # family-wise / FDR level
MIN_CELLTYPES_FOR_PAIRED = 5   # need enough cell types for a meaningful paired test


# --------------------------------------------------------------------------- #
# STATISTICAL HELPERS
# --------------------------------------------------------------------------- #
def mean_ci(x, alpha: float = 0.05):
    """Mean and t-based 95% CI. Returns (mean, lo, hi)."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = x.size
    if n == 0:
        return np.nan, np.nan, np.nan
    m = float(x.mean())
    if n < 2:
        return m, m, m
    s = float(x.std(ddof=1))
    se = s / np.sqrt(n)
    tcrit = float(t_dist.ppf(1 - alpha / 2, df=n - 1))
    return m, m - tcrit * se, m + tcrit * se


def holm_bonferroni(pvals):
    """Holm-Bonferroni step-down adjustment (returns adjusted p-values)."""
    p = np.asarray(pvals, dtype=float)
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p)
    adj = np.empty(n, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (n - rank) * p[idx]
        running = max(running, val)
        adj[idx] = min(running, 1.0)
    return adj


def wilcoxon_rank_biserial(x, y):
    """Matched-pairs rank-biserial correlation effect size for a Wilcoxon test."""
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    d = d[~np.isnan(d)]
    d = d[d != 0]
    if d.size == 0:
        return 0.0
    ranks = rankdata(np.abs(d))
    r_plus = float(ranks[d > 0].sum())
    r_minus = float(ranks[d < 0].sum())
    denom = r_plus + r_minus
    return 0.0 if denom == 0 else (r_plus - r_minus) / denom


def sfari_enrichment_test(ranked, sfari_set_upper, k):
    """
    One-sided hypergeometric test for SFARI enrichment in the top-k of `ranked`.

    Universe = the method's own gene list (`ranked`), so methods with different
    HVG universes (e.g. NO_FILTER vs FULL) get an honest null.

    Returns (obs, expected, p_value).
    """
    universe = {g.upper() for g in ranked}
    N = len(universe)
    k = min(int(k), N)
    if N == 0 or k == 0:
        return 0, 0.0, 1.0
    K = len(universe & sfari_set_upper)
    if K == 0:
        return 0, 0.0, 1.0
    top_k = ranked[:k]
    obs = sum(1 for g in top_k if g.upper() in sfari_set_upper)
    exp = k * K / N
    # P(X >= obs) under Hypergeom(N, K, k)
    p = float(hypergeom.sf(obs - 1, N, K, k))
    return obs, float(exp), p


# --------------------------------------------------------------------------- #
# MODELS
# --------------------------------------------------------------------------- #
class GeneGCN(nn.Module):
    def __init__(self, in_channels, dropout):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 64)
        self.conv3 = GCNConv(64, in_channels)
        self.dropout = dropout

    def forward(self, data):
        h = self.conv1(data.x, data.edge_index, data.edge_attr)
        h = torch.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.conv3(h, data.edge_index, data.edge_attr)


class GeneMLP(nn.Module):
    def __init__(self, in_channels, dropout):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, 64)
        self.fc2 = nn.Linear(64, in_channels)
        self.dropout = dropout

    def forward(self, data):
        h = torch.relu(self.fc1(data.x))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.fc2(h)


# --------------------------------------------------------------------------- #
# GRAPH + POOLING
# --------------------------------------------------------------------------- #
def build_gene_graph(expr_df, thresh=CORR_THRESHOLD):
    X = expr_df.values
    n_genes = X.shape[1]
    corr = np.corrcoef(X.T)
    np.fill_diagonal(corr, 0.0)
    iu = np.triu_indices(n_genes, k=1)
    keep = np.abs(corr[iu]) >= thresh
    src = iu[0][keep]; dst = iu[1][keep]
    w = np.abs(corr[src, dst]).astype(np.float32)
    edge_index = torch.tensor(
        np.stack([np.concatenate([src, dst]), np.concatenate([dst, src])]), dtype=torch.long
    )
    edge_weight = torch.tensor(np.concatenate([w, w]), dtype=torch.float)
    return edge_index, edge_weight


def pool_quantiles(expr_df, n_pool=N_POOL):
    qs = np.linspace(0, 1, n_pool)
    X = expr_df.values
    return np.quantile(X, qs, axis=0).T.astype(np.float32)


def pool_mean_std(expr_df):
    X = expr_df.values
    return np.stack([X.mean(0), X.std(0)], axis=1).astype(np.float32)


# --------------------------------------------------------------------------- #
# SCORERS
# --------------------------------------------------------------------------- #
def train_one(model, data_train, epochs=EPOCHS, lr=LR):
    opt = optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(data_train)
        loss = F.mse_loss(out, data_train.x)
        loss.backward()
        opt.step()
    return model


def score_gene_gcn(ctrl_df, pat_df, n_pool, seed, use_graph=True):
    torch.manual_seed(seed); np.random.seed(seed)

    if n_pool == 2:
        ctrl_prof = pool_mean_std(ctrl_df)
        pat_prof  = pool_mean_std(pat_df)
    else:
        ctrl_prof = pool_quantiles(ctrl_df, n_pool)
        pat_prof  = pool_quantiles(pat_df,  n_pool)

    in_dim = ctrl_prof.shape[1]
    ctrl_x = torch.tensor(ctrl_prof, dtype=torch.float).to(device)
    pat_x  = torch.tensor(pat_prof,  dtype=torch.float).to(device)

    if use_graph:
        ei, ew = build_gene_graph(ctrl_df)
        data_ctrl = Data(x=ctrl_x, edge_index=ei.to(device),
                         edge_attr=ew.to(device)).to(device)
        data_pat  = Data(x=pat_x,  edge_index=ei.to(device),
                         edge_attr=ew.to(device)).to(device)
        model = GeneGCN(in_dim, DROPOUT).to(device)
    else:
        data_ctrl = Data(x=ctrl_x).to(device)
        data_pat  = Data(x=pat_x).to(device)
        model = GeneMLP(in_dim, DROPOUT).to(device)

    model = train_one(model, data_ctrl)
    model.eval()
    with torch.no_grad():
        ctrl_recon = model(data_ctrl).cpu().numpy()
        pat_recon  = model(data_pat).cpu().numpy()

    del model, data_ctrl, data_pat
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    ctrl_err = ((ctrl_recon - ctrl_prof) ** 2).mean(axis=1)
    pat_err  = ((pat_recon  - pat_prof)  ** 2).mean(axis=1)
    red = pat_err - ctrl_err
    return np.abs(red)


def score_wilcoxon(ctrl_hvg, pat_hvg, seed):
    np.random.seed(seed)
    ctrl = ctrl_hvg.copy()
    pat  = pat_hvg.copy()
    ctrl.obs["group"] = "control"
    pat.obs["group"]  = "patient"

    combined = ad.concat([ctrl, pat], label="batch")
    combined.obs["group"] = combined.obs["group"].astype("category")

    sc.tl.rank_genes_groups(
        combined, groupby="group", reference="control",
        method="wilcoxon", n_genes=combined.n_vars
    )
    names  = combined.uns["rank_genes_groups"]["names"]["patient"]
    scores = combined.uns["rank_genes_groups"]["scores"]["patient"]
    score_map = dict(zip(names, np.abs(scores)))
    return np.array([score_map.get(g, 0.0) for g in ctrl_hvg.var_names])


def score_cells_as_nodes(ctrl_hvg, pat_hvg, seed):
    """[SINGLE-PIPELINE] Cell-graph autoencoder scorer, called on the SAME
    ctrl_hvg / pat_hvg matrices every other ablation receives."""
    Xc = ctrl_hvg.X
    Xp = pat_hvg.X
    Xc = (Xc.toarray() if hasattr(Xc, "toarray") else np.asarray(Xc)).astype(np.float32)
    Xp = (Xp.toarray() if hasattr(Xp, "toarray") else np.asarray(Xp)).astype(np.float32)
    genes = [str(g) for g in ctrl_hvg.var_names]
    _, abs_red, _, _ = can.cells_as_nodes_score(Xc, Xp, genes, seed=seed)
    return np.asarray(abs_red, dtype=float)


# --------------------------------------------------------------------------- #
# DIRECTION (proper log2FC on linear CPM)
# --------------------------------------------------------------------------- #
def compute_direction(ctrl_df, pat_df, eps: float = 1e-8):
    """
    log2FC computed on the *linear* (CPM) scale. The input data are log1p(CPM),
    so expm1 recovers CPM before taking the ratio.
    """
    ctrl_lin = np.expm1(ctrl_df.values.astype(np.float64))
    pat_lin  = np.expm1(pat_df.values.astype(np.float64))
    log2fc = np.log2((pat_lin.mean(0) + eps) / (ctrl_lin.mean(0) + eps))
    genes = list(ctrl_df.columns)
    return {genes[i]: ("Up" if log2fc[i] > 0 else "Down") for i in range(len(genes))}


# --------------------------------------------------------------------------- #
# CORE ABLATION EXECUTION
# --------------------------------------------------------------------------- #
def run_one_ablation(ablation, ctrl_hvg, pat_hvg, seed):
    ctrl_df = ctrl_hvg.to_df()
    pat_df  = pat_hvg.to_df()[ctrl_df.columns]
    genes = list(ctrl_df.columns)

    direction = compute_direction(ctrl_df, pat_df)

    if ablation == "FULL":
        scores = score_gene_gcn(ctrl_df, pat_df, n_pool=N_POOL, seed=seed, use_graph=True)
    elif ablation == "NO_GRAPH":
        scores = score_gene_gcn(ctrl_df, pat_df, n_pool=N_POOL, seed=seed, use_graph=False)
    elif ablation == "MEAN_POOL":
        scores = score_gene_gcn(ctrl_df, pat_df, n_pool=2, seed=seed, use_graph=True)
    elif ablation == "NO_FILTER":
        scores = score_gene_gcn(ctrl_df, pat_df, n_pool=N_POOL, seed=seed, use_graph=True)
    elif ablation == "WILCOXON_FAIR":
        scores = score_wilcoxon(ctrl_hvg, pat_hvg, seed=seed)
    elif ablation == "CELLS_AS_NODES":
        scores = score_cells_as_nodes(ctrl_hvg, pat_hvg, seed=seed)
    else:
        raise ValueError(f"Unknown ablation: {ablation}")

    del ctrl_df, pat_df
    gc.collect()

    order = np.argsort(-scores)
    ranked = [genes[i] for i in order]
    return ranked, len(genes), direction


# --------------------------------------------------------------------------- #
# MAIN PIPELINE
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--celltypes", default="ALL_CELLS,IN-SST,Oligodendrocytes,L2_3")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--ablations",
                    default="CELLS_AS_NODES,FULL,NO_GRAPH,MEAN_POOL,"
                            "NO_FILTER,WILCOXON_FAIR")
    ap.add_argument("--out_dir", default=OUT_DIR)
    ap.add_argument("--max_cells", type=int, default=20000,
                    help="Max cells per condition (0 to disable).")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading raw datasets...")
    ctrl_full = data_normalize(sc.read(CONTROL_PATH))
    pat_full  = data_normalize(sc.read(PATIENT_PATH))
    meta = load_meta(META_PATH)
    ctrl_full = attach_celltype_from_meta(ctrl_full, meta, "control")
    pat_full  = attach_celltype_from_meta(pat_full,  meta, "patient")

    sfari = load_sfari()
    sfari_upper = {s.upper() for s in sfari}
    print(f"SFARI genes loaded: {len(sfari)}")

    if args.celltypes == "all":
        shared = sorted(set(ctrl_full.obs[CELLTYPE_COL].unique()) &
                        set(pat_full.obs[CELLTYPE_COL].unique()))
        cts = ["ALL_CELLS"] + shared
    else:
        cts = args.celltypes.split(",")

    ablations = args.ablations.split(",")
    rows = []
    ranking_rows = []

    # Make sure we test every depth in DEPTHS
    depths = list(DEPTHS)

    for ct in cts:
        print(f"\n=== CELL TYPE: {ct} ===")
        if ct == "ALL_CELLS":
            ctrl = ctrl_full.copy(); pat = pat_full.copy()
        else:
            ctrl = ctrl_full[ctrl_full.obs[CELLTYPE_COL] == ct].copy()
            pat  = pat_full[pat_full.obs[CELLTYPE_COL]  == ct].copy()
            if ctrl.n_obs < MIN_CELLS or pat.n_obs < MIN_CELLS:
                print(f"  SKIP (cells < {MIN_CELLS})"); continue

        if args.max_cells > 0:
            if ctrl.n_obs > args.max_cells:
                print(f"  Subsampling control from {ctrl.n_obs} down to {args.max_cells} cells...")
                sc.pp.subsample(ctrl, n_obs=args.max_cells, random_state=42)
            if pat.n_obs > args.max_cells:
                print(f"  Subsampling patient from {pat.n_obs} down to {args.max_cells} cells...")
                sc.pp.subsample(pat, n_obs=args.max_cells, random_state=42)

        # -------------------- Pre-compute HVGs once per cell type ---------- #
        precomputed_hvgs = {}

        has_filtered_ablations = any(ab != "NO_FILTER" for ab in ablations)
        if has_filtered_ablations:
            print(f"  [Pre-processing] Applying artifact filter & selecting top-{TOP_N_GENES} HVGs...")
            kept = filter_artifact_genes(list(ctrl.var_names), verbose=True)
            ctrl_f = ctrl[:, kept].copy()
            pat_f  = pat[:, kept].copy()
            ctrl_hvg_f, pat_hvg_f, _ = select_highly_variable_genes(
                TOP_N_GENES, ctrl_f, pat_f, method="dispersion"
            )
            precomputed_hvgs["FILTERED"] = (ctrl_hvg_f, pat_hvg_f)
            del ctrl_f, pat_f
            gc.collect()

        if "NO_FILTER" in ablations:
            print(f"  [Pre-processing] Selecting top-{TOP_N_GENES} HVGs without artifact filter...")
            ctrl_hvg_nf, pat_hvg_nf, _ = select_highly_variable_genes(
                TOP_N_GENES, ctrl, pat, method="dispersion"
            )
            precomputed_hvgs["UNFILTERED"] = (ctrl_hvg_nf, pat_hvg_nf)

        del ctrl, pat
        gc.collect()

        # -------------------- Run ablations & seeds ------------------------ #
        for ab in ablations:
            group_key = "UNFILTERED" if ab == "NO_FILTER" else "FILTERED"
            ctrl_hvg, pat_hvg = precomputed_hvgs[group_key]

            for seed in range(args.seeds):
                t0 = time.time()
                try:
                    ranked, n_genes, direction = run_one_ablation(ab, ctrl_hvg, pat_hvg, seed)

                    # Hypergeometric SFARI enrichment at each depth (proper null)
                    per_depth = []
                    for k in depths:
                        obs, exp, p = sfari_enrichment_test(ranked, sfari_upper, k)
                        pct = 100.0 * obs / k if k > 0 else np.nan
                        rows.append({
                            "CellType": ct, "Ablation": ab, "Seed": seed,
                            "Depth": k, "N_Genes": n_genes,
                            "SFARI_obs": obs, "SFARI_expected": exp,
                            "SFARI_pct": pct, "Hypergeom_p": p,
                        })
                        per_depth.append(f"top{k}={pct:.1f}%(p={p:.2g})")

                    # Keep per-seed rankings for seed==0 (representative)
                    if seed == 0:
                        for rank_i, g in enumerate(ranked[:250], 1):
                            ranking_rows.append({
                                "CellType": ct, "Method": ab, "Rank": rank_i,
                                "Gene": g, "Direction": direction.get(g, ""),
                                "SFARI": g.upper() in sfari_upper,
                            })

                    print(f"  {ab:14s} seed={seed}  " + "  ".join(per_depth)
                          + f"  ({time.time()-t0:.1f}s)")

                except Exception as e:
                    print(f"  [skip] {ab} seed={seed}: {e}")

                finally:
                    for name in ("ranked", "direction", "per_depth"):
                        if name in locals():
                            del locals()[name]
                    gc.collect()

        del precomputed_hvgs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ----------------------------------------------------------------------- #
    # SAVE PER-SEED RESULTS
    # ----------------------------------------------------------------------- #
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_dir, "ablation_results.csv"), index=False)

    # ----------------------------------------------------------------------- #
    # SUMMARY: mean ± 95% CI over seeds + median hypergeom p + frac. sig.
    # ----------------------------------------------------------------------- #
    def _summarise(g):
        m, lo, hi = mean_ci(g["SFARI_pct"].values, alpha=ALPHA)
        p = g["Hypergeom_p"].values
        return pd.Series({
            "mean": m, "ci_lo": lo, "ci_hi": hi,
            "std": float(np.nanstd(g["SFARI_pct"].values, ddof=1))
                   if len(g) > 1 else np.nan,
            "n_seeds": int(len(g)),
            "median_hypergeom_p": float(np.nanmedian(p)),
            "frac_seeds_sig": float(np.nanmean(p < ALPHA)),
        })

    summ = (df.groupby(["CellType", "Ablation", "Depth"])
              .apply(_summarise)
              .reset_index())
    summ.to_csv(os.path.join(args.out_dir, "ablation_summary.csv"), index=False)

    if ranking_rows:
        rdf = pd.DataFrame(ranking_rows)
        rdf.to_csv(os.path.join(args.out_dir, "ablation_rankings.csv"), index=False)
        print(f"\nRankings : {args.out_dir}/ablation_rankings.csv ({len(rdf)} rows)")

    print(f"Results  : {args.out_dir}/ablation_results.csv")
    print(f"Summary  : {args.out_dir}/ablation_summary.csv")

    # ----------------------------------------------------------------------- #
    # PAIRED ANALYSIS ACROSS CELL TYPES (cell type = experimental unit)
    #   - ALL_CELLS is excluded: it is a superset of the other groups.
    #   - Friedman omnibus test, then pairwise Wilcoxon + Holm correction.
    #   - Seeds are averaged first (technical replicates).
    # ----------------------------------------------------------------------- #
    pair_df = summ[summ.CellType != "ALL_CELLS"].copy()

    for depth in depths:
        sub = pair_df[pair_df.Depth == depth]
        if sub.empty:
            continue
        pv = sub.pivot_table(index="CellType", columns="Ablation", values="mean")
        pv = pv.dropna()
        methods_present = list(pv.columns)
        n_ct = len(pv)

        print(f"\n=== Paired analysis at top-{depth} (cell type = unit, n={n_ct}) ===")
        if n_ct < MIN_CELLTYPES_FOR_PAIRED or len(methods_present) < 2:
            print(f"  Not enough cell types/methods for paired tests; skipping.")
            continue

        # Omnibus: Friedman test across methods
        try:
            stat, p_fried = friedmanchisquare(
                *[pv[m].values for m in methods_present]
            )
            print(f"  Friedman omnibus: chi2={stat:.3f}, p={p_fried:.4g}")
        except Exception as e:
            print(f"  Friedman test failed: {e}")

        # Pairwise Wilcoxon + Holm correction + rank-biserial effect size
        pairs, pvals = [], []
        for i, a in enumerate(methods_present):
            for b in methods_present[i + 1:]:
                try:
                    W, p = wilcoxon_test(pv[a].values, pv[b].values)
                    es = wilcoxon_rank_biserial(pv[a].values, pv[b].values)
                    pairs.append((a, b, W, p, es))
                    pvals.append(p)
                except Exception as e:
                    pairs.append((a, b, np.nan, np.nan, np.nan))
                    pvals.append(np.nan)

        pvals = np.asarray(pvals, dtype=float)
        valid = ~np.isnan(pvals)
        padj = np.full_like(pvals, np.nan)
        if valid.any():
            padj[valid] = holm_bonferroni(pvals[valid])

        for (a, b, W, p, es), pq in zip(pairs, padj):
            ma, mb = pv[a].mean(), pv[b].mean()
            sign = "+" if ma > mb else "-"
            print(f"  {a:14s} vs {b:14s}: n={n_ct:2d}  "
                  f"W={W:6.2f}  p={p:.4g}  p_holm={pq:.4g}  "
                  f"r_rb={es:+.2f}  meanD={sign}{abs(ma-mb):.2f}%")


if __name__ == "__main__":
    main()