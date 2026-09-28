"""
CellGCN: cells-as-nodes alternative to GeneGCN. 2GB GPU-friendly.

Nodes = cells, edges = kNN (k=15) in PCA space of controls; the GCN learns to
reconstruct each cell's gene-expression vector. Per-gene RED is the mean
squared error across patient cells minus mean across control cells. Patient
cells are projected through the control-fitted PCA so the cell-cell graph
carries patient-specific signal that the gene-graph variant lacked.

------------------------------------------------------------------------------
STATISTICAL CORRECTIONS (added in response to major review comments)
------------------------------------------------------------------------------
[STAT-FIX 1] Gene universe held constant across seeds.
    HVGs are now selected ONCE on the full (unsampled) data per cell type,
    and the same gene set is reused for every seed. Previously HVGs were
    re-selected after per-seed subsampling, so every seed ranked a *different*
    gene universe; SFARI overlaps across seeds were therefore not comparable
    and the reported std/count were not a valid estimate of seed variance.

[STAT-FIX 2] Direction labels now reflect expression difference, not RED sign.
    `signed_RED = pat_err - ctrl_err` measures reconstruction difficulty, not
    regulation direction; calling positive RED "Up" is unjustified. Direction
    is now computed from the actual mean-expression difference
    (mean(pat) - mean(ctrl)) and reported per gene.

[STAT-FIX 3] Multiple-testing correction.
    Benjamini-Hochberg FDR is applied to the paired Wilcoxon p-values across
    methods and depths. Raw and adjusted p-values are both saved.

[STAT-FIX 4] Paired-test guardrails.
    * Minimum n enforced: with n < 6 the two-sided Wilcoxon signed-rank test
      cannot reach p < 0.05 at all, so the test is skipped and a warning is
      emitted instead of reporting a meaningless p-value.
    * Ties are handled by the zero_method="wilcox" default; ties reported.
    * Effect size (median paired difference + matched-pairs rank-biserial r)
      is reported alongside the p-value, per review guidance.

[STAT-FIX 5] Uncertainty reported.
    Summary now reports mean, median, std, SEM, and a percentile bootstrap
    95% CI across seeds (and across cell types in the cross-cell-type table).

[STAT-FIX 6] Independent randomness between subsampling and model init.
    Subsampling uses `seed`, model init / graph construction use
    `seed + SEED_OFFSET` so the two sources of randomness are not shared.

[STAT-FIX 7] Depth-wise dependence acknowledged.
    Depth curves are nested (top-k subset of top-(k+Delta)) and therefore not
    independent; per-depth p-values are labelled descriptive and the primary
    inference is drawn from a single pre-specified depth (PRIMARY_DEPTH = 250).
------------------------------------------------------------------------------
OPTION-2 PATCHES (for the paper framing built around cells-as-nodes)
------------------------------------------------------------------------------
[OPT2-1] ALL_CELLS is excluded from the paired tests. It is a superset of the
    other cell-type groups and therefore not an independent experimental unit.
    ALL_CELLS still appears in the per-seed and per-cell-type summary tables.

[OPT2-2] Top-250 rankings are saved for EVERY seed, not just seed 0. This lets
    make_paper_outputs.py measure per-seed stability (fraction of seeds a gene
    appears in the top-K) and build a consensus shortlist per cell type.

Drop in next to model_per_celltype.py and gene_filter.py.

Run on a 2GB GPU
----------------
python cells_as_nodes.py --celltypes all --seeds 5 --max_cells 3000 --hidden 64
"""

import argparse, os, time, gc
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.stats import wilcoxon as wilcoxon_test
from statsmodels.stats.multitest import multipletests   # [STAT-FIX 3]

from model_per_celltype import (
    data_normalize, attach_celltype_from_meta, load_meta,
    select_highly_variable_genes, _apply_gene_filter_adata,
    CONTROL_PATH, PATIENT_PATH, META_PATH, CELLTYPE_COL,
    TOP_N_GENES, HVG_METHOD, EPOCHS, LR, DROPOUT, MIN_CELLS, DEVICE,
)

# --------------------------------------------------------------------------- #
# CONFIG (overridable from CLI)
# --------------------------------------------------------------------------- #
SFARI_PATH        = "SFARI-Gene_genes.csv"
SFARI_SYMBOL_COL  = "gene-symbol"
DEPTHS            = [50, 100, 150, 200, 250]
PRIMARY_DEPTH     = 250            # [STAT-FIX 7] pre-specified primary endpoint
OUT_DIR           = "cellgcn_results"

N_PCS             = 50
K_NEIGHBORS       = 15
HIDDEN_DIM        = 64
MAX_CELLS         = 3000
UNIVERSE_MAX_CELLS = 8000          # cap for HVG universe selection only (RAM)

# --- [IMPROVE] method enhancements ---------------------------------------- #
HOLDOUT_FRAC      = 0.2            # control cells held out of training, used
                                   # for ctrl_err. 0.0 = legacy in-sample score.
MASK_RATE         = 0.2            # denoising: fraction of inputs zeroed per
                                   # epoch. 0.0 = plain self-reconstruction.
SCORE_MODE        = "relative"     # "raw"      : pat_err - ctrl_err  (legacy)
                                   # "relative" : (pat_err-ctrl_err)/ctrl_err
                                   # "logratio" : log(pat_err / ctrl_err)
SCORE_EPS         = 1e-8

SEED_OFFSET       = 10_000         # [STAT-FIX 6] decouple subsample vs model seeds
MIN_PAIRED_N      = 6              # [STAT-FIX 4] Wilcoxon floor
N_BOOTSTRAP       = 2000           # [STAT-FIX 5] bootstrap resamples


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class CellGCN(nn.Module):
    """Two-layer GCN autoencoder over a cell-cell graph."""
    def __init__(self, n_genes, hidden, dropout):
        super().__init__()
        self.conv1 = GCNConv(n_genes, hidden)
        self.conv2 = GCNConv(hidden, n_genes)
        self.dropout = dropout

    def forward(self, data):
        h = self.conv1(data.x, data.edge_index, data.edge_attr)
        h = torch.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.conv2(h, data.edge_index, data.edge_attr)


# --------------------------------------------------------------------------- #
# Cell graph
# --------------------------------------------------------------------------- #
def build_cell_graph(expr_cells_by_genes, seed=0, pca_model=None):
    """Symmetric kNN in PCA space. Returns (edge_index, edge_weight, pca)."""
    X = expr_cells_by_genes
    n_cells = X.shape[0]
    k_eff = min(K_NEIGHBORS, n_cells - 1)
    n_pcs_eff = min(N_PCS, n_cells - 1, X.shape[1] - 1)

    if pca_model is None:
        pca_model = PCA(n_components=n_pcs_eff, random_state=seed)
        Z = pca_model.fit_transform(X)
    else:
        Z = pca_model.transform(X)

    nbrs = NearestNeighbors(n_neighbors=k_eff + 1).fit(Z)
    dists, idxs = nbrs.kneighbors(Z)
    dists, idxs = dists[:, 1:], idxs[:, 1:]

    src = np.repeat(np.arange(n_cells), k_eff)
    dst = idxs.flatten()
    d = dists.flatten()
    src_sym = np.concatenate([src, dst])
    dst_sym = np.concatenate([dst, src])
    d_sym = np.concatenate([d, d])
    med = np.median(d_sym) if np.median(d_sym) > 0 else 1.0
    w = np.exp(-d_sym / med)

    edge_index = torch.tensor(np.stack([src_sym, dst_sym]), dtype=torch.long)
    edge_weight = torch.tensor(w, dtype=torch.float)
    return edge_index, edge_weight, pca_model


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train_cell_gcn(data_train, n_genes, device, hidden, seed=0, verbose=False,
                   train_rows=None):
    """Train the cell-graph autoencoder.

    [IMPROVE] Two changes over the original:
      * train_rows: loss is computed only on these node rows, so control cells
        held out for scoring never enter training. Without this, ctrl_err is an
        in-sample error while pat_err is out-of-sample, and their difference
        carries a per-gene generalisation gap on top of any disease signal.
      * MASK_RATE: a denoising objective. Inputs are randomly zeroed each epoch
        and the model reconstructs the clean profile, which prevents it from
        settling into a near-identity map.
    """
    torch.manual_seed(seed); np.random.seed(seed)
    model = CellGCN(n_genes, hidden=hidden, dropout=DROPOUT).to(device)
    opt = optim.Adam(model.parameters(), lr=LR)
    x_clean = data_train.x
    for ep in range(EPOCHS):
        model.train()
        opt.zero_grad()
        if MASK_RATE > 0:
            keep = (torch.rand_like(x_clean) >= MASK_RATE).float()
            noisy = x_clean * keep / max(1.0 - MASK_RATE, 1e-6)
            data_in = Data(x=noisy, edge_index=data_train.edge_index,
                           edge_attr=data_train.edge_attr)
        else:
            data_in = data_train
        out = model(data_in)
        if train_rows is None:
            loss = F.mse_loss(out, x_clean)
        else:
            loss = F.mse_loss(out[train_rows], x_clean[train_rows])
        loss.backward()
        opt.step()
        if verbose and (ep + 1) % 50 == 0:
            print(f"    epoch {ep+1:4d}  loss={loss.item():.4f}")
    return model


def _to_device(ctrl_X, pat_X, ei_ctrl, ew_ctrl, ei_pat, ew_pat, device):
    x_ctrl = torch.tensor(ctrl_X, dtype=torch.float).to(device)
    x_pat  = torch.tensor(pat_X,  dtype=torch.float).to(device)
    return (x_ctrl, x_pat,
            ei_ctrl.to(device), ew_ctrl.to(device),
            ei_pat.to(device),  ew_pat.to(device))


# --------------------------------------------------------------------------- #
# Scoring with GPU->CPU fallback
# --------------------------------------------------------------------------- #
def cells_as_nodes_score(ctrl_X, pat_X, gene_names, seed=0, verbose=False):
    """Returns (gene_names, |RED|, signed_RED, expr_diff).

    [STAT-FIX 2] The returned `expr_diff` is the real expression contrast
    (mean patient expression - mean control expression) so downstream
    "Up"/"Down" labels are meaningful. `signed_RED` is still returned for
    continuity but is no longer used to assign direction.
    """
    # [STAT-FIX 6] use an offset seed for model init so it is not correlated
    # with whatever seed the caller used for subsampling.
    model_seed = seed + SEED_OFFSET
    torch.manual_seed(model_seed); np.random.seed(model_seed)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if ctrl_X.shape[0] > MAX_CELLS:
        idx = np.random.RandomState(seed).choice(ctrl_X.shape[0], MAX_CELLS, replace=False)
        ctrl_X = ctrl_X[idx]
    if pat_X.shape[0] > MAX_CELLS:
        idx = np.random.RandomState(seed + 1).choice(pat_X.shape[0], MAX_CELLS, replace=False)
        pat_X = pat_X[idx]

    n_genes = ctrl_X.shape[1]
    if verbose:
        print(f"    ctrl: {ctrl_X.shape[0]} cells, pat: {pat_X.shape[0]} cells, "
              f"genes: {n_genes}, hidden: {HIDDEN_DIM}")

    ei_ctrl, ew_ctrl, pca = build_cell_graph(ctrl_X, seed=model_seed)
    ei_pat,  ew_pat,  _   = build_cell_graph(pat_X, seed=model_seed, pca_model=pca)

    # [IMPROVE] hold out control cells so ctrl_err is out-of-sample, matching
    # pat_err. The graph still spans every control cell; only the LOSS is
    # restricted, so held-out cells keep their neighbourhood structure.
    n_ctrl_cells = ctrl_X.shape[0]
    if HOLDOUT_FRAC > 0 and n_ctrl_cells >= 20:
        rs = np.random.RandomState(model_seed)
        perm = rs.permutation(n_ctrl_cells)
        n_hold = max(1, int(round(HOLDOUT_FRAC * n_ctrl_cells)))
        hold_np, train_np = perm[:n_hold], perm[n_hold:]
    else:
        hold_np = np.arange(n_ctrl_cells)
        train_np = np.arange(n_ctrl_cells)

    used_device = DEVICE
    try:
        x_ctrl, x_pat, ei_ctrl_d, ew_ctrl_d, ei_pat_d, ew_pat_d = _to_device(
            ctrl_X, pat_X, ei_ctrl, ew_ctrl, ei_pat, ew_pat, used_device)
        data_ctrl = Data(x=x_ctrl, edge_index=ei_ctrl_d, edge_attr=ew_ctrl_d)
        data_pat  = Data(x=x_pat,  edge_index=ei_pat_d,  edge_attr=ew_pat_d)
        train_rows = torch.tensor(train_np, dtype=torch.long, device=used_device)
        model = train_cell_gcn(data_ctrl, n_genes, used_device, HIDDEN_DIM,
                               seed=model_seed, verbose=verbose,
                               train_rows=train_rows)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" not in str(e).lower(): raise
        print(f"    [OOM on GPU] falling back to CPU")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        used_device = torch.device("cpu")
        x_ctrl, x_pat, ei_ctrl_d, ew_ctrl_d, ei_pat_d, ew_pat_d = _to_device(
            ctrl_X, pat_X, ei_ctrl, ew_ctrl, ei_pat, ew_pat, used_device)
        data_ctrl = Data(x=x_ctrl, edge_index=ei_ctrl_d, edge_attr=ew_ctrl_d)
        data_pat  = Data(x=x_pat,  edge_index=ei_pat_d,  edge_attr=ew_pat_d)
        train_rows = torch.tensor(train_np, dtype=torch.long, device=used_device)
        model = train_cell_gcn(data_ctrl, n_genes, used_device, HIDDEN_DIM,
                               seed=model_seed, verbose=verbose,
                               train_rows=train_rows)

    model.eval()
    with torch.no_grad():
        ctrl_recon = model(data_ctrl).cpu().numpy()
        pat_recon  = model(data_pat).cpu().numpy()

    # [IMPROVE] ctrl_err on HELD-OUT control cells only
    ctrl_err = ((ctrl_recon[hold_np] - ctrl_X[hold_np]) ** 2).mean(axis=0)
    pat_err  = ((pat_recon  - pat_X)  ** 2).mean(axis=0)

    # [IMPROVE] scoring statistic. "raw" reproduces the original behaviour.
    # A raw difference is dominated by genes that are simply hard to
    # reconstruct in both groups; normalising by the control error asks
    # instead how much WORSE a gene reconstructs in patients, relative to how
    # well the model knows it in controls.
    if SCORE_MODE == "raw":
        red = pat_err - ctrl_err
    elif SCORE_MODE == "relative":
        red = (pat_err - ctrl_err) / (ctrl_err + SCORE_EPS)
    elif SCORE_MODE == "logratio":
        red = np.log((pat_err + SCORE_EPS) / (ctrl_err + SCORE_EPS))
    else:
        raise ValueError(f"Unknown SCORE_MODE: {SCORE_MODE}")

    # [STAT-FIX 2] real expression contrast, not reconstruction-error contrast
    expr_diff = pat_X.mean(axis=0) - ctrl_X.mean(axis=0)

    del model, data_ctrl, data_pat, x_ctrl, x_pat
    del ei_ctrl_d, ew_ctrl_d, ei_pat_d, ew_pat_d
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return gene_names, np.abs(red), red, expr_diff


# --------------------------------------------------------------------------- #
# Wrapper
# --------------------------------------------------------------------------- #
def run_cells_as_nodes(ctrl_ad, pat_ad, seed=0, apply_filter=True,
                       gene_subset=None, verbose=False):
    """If `gene_subset` is provided (list of gene names), the Adatas are
    restricted to that gene set BEFORE any further processing.

    [STAT-FIX 1] This lets the caller fix the gene universe once and reuse it
    across all seeds, so per-seed SFARI overlaps are directly comparable.
    """
    if apply_filter:
        ctrl_ad = _apply_gene_filter_adata(ctrl_ad)
        pat_ad  = pat_ad[:, list(ctrl_ad.var_names)].copy()

    if gene_subset is None:
        # Fallback: pick HVGs on whatever we were handed (legacy behaviour).
        ctrl_ad, pat_ad, _ = select_highly_variable_genes(
            TOP_N_GENES, ctrl_ad, pat_ad, method=HVG_METHOD)
    else:
        # [STAT-FIX 1] restrict to the pre-selected universe, preserving order
        keep = [g for g in gene_subset if g in ctrl_ad.var_names and g in pat_ad.var_names]
        ctrl_ad = ctrl_ad[:, keep].copy()
        pat_ad  = pat_ad[:,  keep].copy()

    genes = list(ctrl_ad.var_names)
    ctrl_X = ctrl_ad.X.toarray().astype(np.float32) if hasattr(ctrl_ad.X, "toarray") else ctrl_ad.X.astype(np.float32)
    pat_X  = pat_ad.X.toarray().astype(np.float32)  if hasattr(pat_ad.X,  "toarray") else pat_ad.X.astype(np.float32)

    gene_names, abs_red, signed_red, expr_diff = cells_as_nodes_score(
        ctrl_X, pat_X, genes, seed=seed, verbose=verbose)
    order = np.argsort(-abs_red)
    ranked = [gene_names[i] for i in order]
    # [STAT-FIX 2] direction from expression contrast, NOT from RED sign
    direction_map = {gene_names[i]: ("Up" if expr_diff[i] > 0 else "Down")
                     for i in range(len(gene_names))}
    return ranked, len(genes), direction_map


# --------------------------------------------------------------------------- #
# SFARI
# --------------------------------------------------------------------------- #
from sfari_scoring import (
    load_sfari as _load_sfari_weighted,
    sfari_overlap, sfari_weighted, universe_null, score_ranked,
)


def load_sfari():
    """Symbol set only. Kept for run_ablations.py, which imports this."""
    s, _ = _load_sfari_weighted()
    return s


def load_sfari_weighted():
    """(symbol_set, weight_map) - weight by SFARI confidence score."""
    return _load_sfari_weighted()


# --------------------------------------------------------------------------- #
# [STAT-FIX 4/5] Statistical helpers
# --------------------------------------------------------------------------- #
def _paired_wilcoxon_with_effects(a, b, method_name):
    """Paired Wilcoxon with guardrails, effect sizes, and an unambiguous
    return dict so the caller can assemble a results table.

    Returns None if the test is underpowered (n < MIN_PAIRED_N) or degenerate.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    n = len(a)
    if n < MIN_PAIRED_N:
        return {"Method": method_name, "n_pairs": n, "skipped": True,
                "reason": f"n<{MIN_PAIRED_N}: Wilcoxon two-sided cannot reach p<0.05"}
    diff = a - b
    if np.allclose(diff, 0):
        return {"Method": method_name, "n_pairs": n, "skipped": True,
                "reason": "all paired differences are zero"}
    try:
        stat, pval = wilcoxon_test(a, b, zero_method="wilcox", alternative="two-sided")
    except Exception as e:
        return {"Method": method_name, "n_pairs": n, "skipped": True,
                "reason": f"wilcoxon failed: {e}"}

    # Matched-pairs rank-biserial correlation (effect size for Wilcoxon)
    # r_rb = (W+ - W-) / (W+ + W-); recomputed from signed ranks.
    ranks = pd.Series(np.abs(diff)).rank().values
    w_pos = ranks[diff > 0].sum()
    w_neg = ranks[diff < 0].sum()
    denom = w_pos + w_neg
    r_rb = (w_pos - w_neg) / denom if denom > 0 else 0.0

    n_ties = int((diff == 0).sum())
    return {
        "Method": method_name,
        "n_pairs": n,
        "skipped": False,
        "median_diff": float(np.median(diff)),
        "mean_diff": float(np.mean(diff)),
        "r_rb": float(r_rb),
        "W": float(stat),
        "p_raw": float(pval),
        "n_ties": n_ties,
        "wins_A": int((a > b).sum()),
        "wins_B": int((b > a).sum()),
    }


def _bootstrap_ci(values, n_boot=N_BOOTSTRAP, alpha=0.05, seed=0):
    """Percentile bootstrap CI of the mean. Returns (lo, hi)."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return (np.nan, np.nan)
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def summarize_long(df, group_cols):
    """[STAT-FIX 5] Mean, median, std, SEM, and bootstrap 95% CI per group."""
    rows = []
    for keys, sub in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        vals = sub["SFARI_pct"].values.astype(float)
        lo, hi = _bootstrap_ci(vals, seed=0)
        rows.append({
            **dict(zip(group_cols, keys)),
            "mean":  float(np.mean(vals)),
            "median": float(np.median(vals)),
            "std":   float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "sem":   float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0,
            "ci_lo": lo,
            "ci_hi": hi,
            "count": int(len(vals)),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--celltypes", default="ALL_CELLS,IN-SST,Oligodendrocytes,L2_3")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--max_cells", type=int, default=3000,
                    help="cap cells per group (lower = less VRAM); default 3000")
    ap.add_argument("--hidden", type=int, default=64,
                    help="GCN hidden dim (lower = less VRAM); default 64")
    ap.add_argument("--score", type=str, default="relative",
                    choices=["raw", "relative", "logratio"],
                    help="[IMPROVE] ranking statistic; 'raw' = legacy")
    ap.add_argument("--holdout", type=float, default=0.2,
                    help="[IMPROVE] control fraction held out of training "
                         "for ctrl_err; 0 = legacy in-sample")
    ap.add_argument("--mask", type=float, default=0.2,
                    help="[IMPROVE] denoising mask rate; 0 = legacy")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    global MAX_CELLS, HIDDEN_DIM, SCORE_MODE, HOLDOUT_FRAC, MASK_RATE
    MAX_CELLS  = args.max_cells
    HIDDEN_DIM = args.hidden
    SCORE_MODE = args.score
    HOLDOUT_FRAC = args.holdout
    MASK_RATE = args.mask
    print(f"Settings: max_cells={MAX_CELLS}, hidden={HIDDEN_DIM}, device={DEVICE}")
    print(f"[IMPROVE] score={SCORE_MODE}  holdout={HOLDOUT_FRAC}  mask={MASK_RATE}")
    print(f"[STAT] primary depth = {PRIMARY_DEPTH}; "
          f"min paired n for Wilcoxon = {MIN_PAIRED_N}")

    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading data...")
    ctrl_full = data_normalize(sc.read(CONTROL_PATH))
    pat_full  = data_normalize(sc.read(PATIENT_PATH))
    meta = load_meta(META_PATH)
    ctrl_full = attach_celltype_from_meta(ctrl_full, meta, "control")
    pat_full  = attach_celltype_from_meta(pat_full,  meta, "patient")

    sfari, sfari_w = load_sfari_weighted()
    n_hi = sum(1 for v in sfari_w.values() if v >= 1.0)
    print(f"SFARI genes loaded: {len(sfari)} ({n_hi} high-confidence)")

    if args.celltypes == "all":
        shared = sorted(set(ctrl_full.obs[CELLTYPE_COL].unique()) &
                        set(pat_full.obs[CELLTYPE_COL].unique()))
        cts = ["ALL_CELLS"] + shared
    else:
        cts = args.celltypes.split(",")

    rows = []
    ranking_rows = []

    for ct in cts:
        print(f"\n=== {ct} ===")
        if ct == "ALL_CELLS":
            ctrl = ctrl_full.copy(); pat = pat_full.copy()
        else:
            ctrl = ctrl_full[ctrl_full.obs[CELLTYPE_COL] == ct].copy()
            pat  = pat_full[pat_full.obs[CELLTYPE_COL]  == ct].copy()
            if ctrl.n_obs < MIN_CELLS or pat.n_obs < MIN_CELLS:
                print(f"  SKIP (cells < {MIN_CELLS})"); continue

        # ----------------------------------------------------------------- #
        # [STAT-FIX 1] Fix the gene universe ONCE per cell type, on the full
        # (unsampled) data, then reuse it for every seed. This makes SFARI
        # overlaps across seeds directly comparable.
        # ----------------------------------------------------------------- #
        ctrl_filtered = _apply_gene_filter_adata(ctrl)
        pat_filtered  = pat[:, list(ctrl_filtered.var_names)].copy()

        # [MEM] Cap cells used for UNIVERSE SELECTION ONLY. The HVG step
        # densifies the matrix; ALL_CELLS (49k x 22k float32) exceeds RAM.
        # random_state=0 keeps the universe identical across seeds.
        if ctrl_filtered.n_obs > UNIVERSE_MAX_CELLS:
            sc.pp.subsample(ctrl_filtered, n_obs=UNIVERSE_MAX_CELLS, random_state=0)
        if pat_filtered.n_obs > UNIVERSE_MAX_CELLS:
            sc.pp.subsample(pat_filtered, n_obs=UNIVERSE_MAX_CELLS, random_state=0)

        ctrl_hvg, pat_hvg, _ = select_highly_variable_genes(
            TOP_N_GENES, ctrl_filtered, pat_filtered, method=HVG_METHOD)
        fixed_genes = list(ctrl_hvg.var_names)
        print(f"  [STAT-FIX 1] fixed gene universe: {len(fixed_genes)} genes")

        # Measured null for THIS context's universe
        null = universe_null(fixed_genes, sfari, sfari_w, DEPTHS, seed=0)
        print(f"  [NULL] SFARI baseline {null['pct_baseline_universe']:.2f}% | "
              f"weighted {null['wt_baseline_universe']:.2f}% "
              f"({null['n_sfari_in_universe']}/{null['universe_size']} curated)")
        del ctrl_filtered, pat_filtered, ctrl_hvg, pat_hvg
        gc.collect()

        for seed in range(args.seeds):
            t0 = time.time()
            try:
                ctrl_seed = ctrl.copy()
                pat_seed  = pat.copy()

                # [STAT-FIX 6] subsample uses `seed`; model init uses `seed+OFFSET`
                # (inside cells_as_nodes_score) so the two randomness sources
                # are independent.
                if ctrl_seed.n_obs > MAX_CELLS:
                    sc.pp.subsample(ctrl_seed, n_obs=MAX_CELLS, random_state=seed)
                if pat_seed.n_obs > MAX_CELLS:
                    sc.pp.subsample(pat_seed, n_obs=MAX_CELLS, random_state=seed)

                ranked, n_genes, dir_map = run_cells_as_nodes(
                    ctrl_seed, pat_seed, seed=seed,
                    apply_filter=True,
                    gene_subset=fixed_genes,
                    verbose=args.verbose)
            except Exception as e:
                print(f"  [skip] seed={seed}: {e}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                continue
            finally:
                if 'ctrl_seed' in locals(): del ctrl_seed
                if 'pat_seed' in locals(): del pat_seed
                gc.collect()

            scored = score_ranked(ranked, sfari, sfari_w, null, DEPTHS)
            for k, m in scored.items():
                rows.append({"CellType": ct, "Method": "CELLS_AS_NODES",
                             "Seed": seed, "Depth": k,
                             "N_Genes": n_genes, **m})

            # [OPT2-2] Save top-250 for EVERY seed so per-seed stability can
            # be measured and a consensus shortlist built per cell type.
            for rank, g in enumerate(ranked[:250], 1):
                ranking_rows.append({
                    "CellType": ct, "Seed": seed, "Rank": rank, "Gene": g,
                    "Direction": dir_map.get(g, ""),
                    "SFARI": g.upper() in sfari,
                    "SFARI_weight": sfari_w.get(g.upper(), 0.0),
                })

            s250 = scored[250]
            print(f"  seed={seed} top250 pct={s250['SFARI_pct']:.1f}% "
                  f"(base {s250['SFARI_pct_baseline']:.1f}%, "
                  f"fold {s250['SFARI_pct_fold']:.2f}) "
                  f"wt={s250['SFARI_wt']:.1f}% "
                  f"(fold {s250['SFARI_wt_fold']:.2f}) "
                  f"({time.time()-t0:.1f}s)")

    # ------------------------------------------------------------------- #
    # Save per-seed results and [STAT-FIX 5] richer summary
    # ------------------------------------------------------------------- #
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "cells_as_nodes_results.csv"), index=False)

    summ = summarize_long(df, ["CellType", "Method", "Depth"])
    summ.to_csv(os.path.join(OUT_DIR, "cells_as_nodes_summary.csv"), index=False)

    rdf = pd.DataFrame(ranking_rows)
    rdf.to_csv(os.path.join(OUT_DIR, "cells_as_nodes_rankings.csv"), index=False)

    print(f"\nResults   : {OUT_DIR}/cells_as_nodes_results.csv ({len(df)} rows)")
    print(f"Summary   : {OUT_DIR}/cells_as_nodes_summary.csv "
          f"(mean/median/std/sem/bootstrap CI)")
    print(f"Rankings  : {OUT_DIR}/cells_as_nodes_rankings.csv "
          f"({len(rdf)} rows; top-250 per seed)")

    # ------------------------------------------------------------------- #
    # Head-to-head comparison with prior ablation results
    # ------------------------------------------------------------------- #
    prior_path = "ablation_results/ablation_summary.csv"
    if not os.path.exists(prior_path):
        print(f"\n[info] {prior_path} not found; skipping head-to-head comparison.")
        return

    print(f"\nComparing against {prior_path}")
    prior = pd.read_csv(prior_path).rename(columns={"Ablation": "Method"})
    # Prior summary may only have mean/std/count; harmonize columns.
    for c in ["median", "sem", "ci_lo", "ci_hi"]:
        if c not in prior.columns:
            prior[c] = np.nan
    full = pd.concat([prior, summ], ignore_index=True)

    # ------------------------------------------------------------------ #
    # Descriptive depth curve (nested top-k; NOT independent) [STAT-FIX 7]
    # ------------------------------------------------------------------ #
    print("\nDescriptive mean SFARI overlap (%) across cell types, "
          "per method per depth (top-k sets are nested; treat as descriptive):")
    across = full.groupby(["Method", "Depth"])["mean"].mean().reset_index()
    print(across.pivot(index="Method", columns="Depth", values="mean").round(2).to_string())

    # ------------------------------------------------------------------ #
    # Primary inference at pre-specified PRIMARY_DEPTH [STAT-FIX 7]
    # [OPT2-1] ALL_CELLS is excluded from paired tests: it is a superset of
    # the other cell-type groups and therefore not an independent unit.
    # ------------------------------------------------------------------ #
    print(f"\nPrimary inference at pre-specified depth = {PRIMARY_DEPTH}")
    full_for_test = full[full.CellType != "ALL_CELLS"].copy()
    pv = full_for_test[full_for_test.Depth == PRIMARY_DEPTH].pivot_table(
        index="CellType", columns="Method", values="mean")
    if "CELLS_AS_NODES" not in pv.columns:
        print("  CELLS_AS_NODES missing -- skipping paired test"); return

    # [STAT-FIX 3] collect all raw p-values first, then BH-adjust
    stats_rows = []
    for m in pv.columns:
        if m == "CELLS_AS_NODES": continue
        paired = pv[["CELLS_AS_NODES", m]].dropna()
        res = _paired_wilcoxon_with_effects(
            paired["CELLS_AS_NODES"].values, paired[m].values, m)
        if res is None:
            continue
        res["Comparison"] = f"CELLS_AS_NODES vs {m}"
        res["Depth"] = PRIMARY_DEPTH
        stats_rows.append(res)

    if not stats_rows:
        print("  No valid paired comparisons to test."); return

    stats_df = pd.DataFrame(stats_rows)

    # [STAT-FIX 3] BH-FDR across all comparisons at the primary depth.
    tested = stats_df[stats_df["skipped"] == False].copy()
    if len(tested):
        reject, p_adj, _, _ = multipletests(
            tested["p_raw"].values, alpha=0.05, method="fdr_bh")
        tested["p_adj_bh"] = p_adj
        tested["reject_bh_0.05"] = reject
        stats_df.loc[tested.index, "p_adj_bh"] = p_adj
        stats_df.loc[tested.index, "reject_bh_0.05"] = reject

    stats_df.to_csv(os.path.join(OUT_DIR, "cells_as_nodes_paired_tests.csv"),
                    index=False)
    print(f"  Paired-test table written to "
          f"{OUT_DIR}/cells_as_nodes_paired_tests.csv")

    for _, r in stats_df.iterrows():
        if r.get("skipped", False):
            print(f"  {r['Comparison']:34s}: SKIPPED ({r.get('reason','')})")
            continue
        sig = "*" if bool(r.get("reject_bh_0.05", False)) else " "
        print(f"  {r['Comparison']:34s}: n={int(r['n_pairs'])}  "
              f"median_diff={r['median_diff']:+.2f}  "
              f"r_rb={r['r_rb']:+.2f}  "
              f"W={r['W']:.1f}  p={r['p_raw']:.4g}  "
              f"p_BH={r.get('p_adj_bh', float('nan')):.4g} {sig}"
              f"  wins={int(r['wins_A'])}/{int(r['n_pairs'])}"
              f"  ties={int(r['n_ties'])}")


if __name__ == "__main__":
    main()