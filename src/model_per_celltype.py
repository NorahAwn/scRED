"""
GeneGCN - per-cell-type analysis (major-revision version).

Runs the full pipeline (HVG selection -> per-cell-type co-expression graph ->
GCN self-reconstruction with K-fold CV -> gene perturbation scoring) separately
for each cell type, looping over them one by one.

Scoring: per-gene reconstruction-error deviation
    RED = patient_recon_error - control_recon_error   (per gene)
Genes are RANKED by |RED| (magnitude of dysregulation). The DIRECTION of change
(Up/Down) is taken separately from raw log1p-CPM expression as the sign of
mean(patient) - mean(control), reported as log2FC_raw. Genes with |log2FC| below
LOG2FC_FLOOR are labelled "NC" and, by default, EXCLUDED from the ranked output
so that no top-ranked gene is directionless.

Major-revision fixes integrated:
  * R^2 evaluation compares matched tensors (no control-full vs val-split mix-up).
  * Gene labels come from the ORDERED common-gene list (never scrambled HVG order).
  * Per-fold models are not leaked into scoring: a single model is retrained on
    all control cells of the cell type before scoring.
  * CV holds out CELLS (or DONORS via GroupKFold when donor IDs are known),
    not quantile positions of the pooled profile. Optionally rebuilds the
    co-expression graph per fold (STRICT_CV=True) to avoid graph leakage.
  * Permutation null (cell-level or donor-level) with BH-FDR per gene.
    By default the null reuses the control-fitted model (fast). Set
    PERM_RETRAIN=True for a fully refit null (slow but assumption-free).
  * Non-graph PCA baseline reconstruction error reported alongside the GCN.
  * Barcode-collision sanity check between control & patient.
  * Robust overall aggregation (max + mean-of-top-k) across cell types.
  * Donor info is now passed to the GLOBAL run too, so donor-aware CV /
    permutation are used consistently across global and per-cell-type runs.
  * EXCLUDE_NC_FROM_RANKED defaults to True (matches docstring and prevents
    directionless top-ranked genes).
  * Loud warning when DONOR_COL is None (cell-level analysis).
  * PERM_RETRAIN flag for a properly refit permutation null.

HVG-SELECTION FIX (this version):
  * A DETECTION FLOOR (MIN_DETECTION_RATE) is applied before HVG selection.
  * HVG_METHOD="dispersion" now uses MEAN-BINNED NORMALIZED DISPERSION
    (Scanpy/Seurat flavour) instead of raw var/mean.
  The previous raw Fano factor (var/mean) rises as the mean falls, so it
  selected the sparsest genes in the matrix: in per-cell-type universes,
  essentially every selected gene was detected in <5% of nuclei and the
  canonical markers of each cell type (PLP1/MBP/MAG/MOG in oligodendrocytes,
  GAD1/GAD2/SST in IN-SST) were excluded. Set HVG_METHOD="dispersion_raw"
  to reproduce the old behaviour for comparison.

Import direction (do NOT import from cells_as_nodes here -- that creates a
circular import because cells_as_nodes imports from this module):
    gene_filter.py         (no internal deps)
    model_per_celltype.py  -> gene_filter
    cells_as_nodes.py      -> model_per_celltype, gene_filter
    run_ablations.py       -> cells_as_nodes, gene_filter

Usage:
    python model_per_celltype.py
Edit the CONFIG block below.
"""

import os
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, KFold
from scipy.stats import spearmanr
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

from gene_filter import filter_artifact_genes, filter_report


# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #
CONTROL_PATH = "ASD dataset/control_adata.h5ad"
PATIENT_PATH = "ASD dataset/patient_adata.h5ad"

META_PATH = "ASD dataset/rawMatrix/meta.txt"
META_BARCODE_COL = "cell"
CELLTYPE_COL = "cluster"
DONOR_COL = "individual"             # Set to a real column name for donor-aware
                               # analysis. None => cell-level CV/permutation and
                               # a loud warning is emitted.

CELL_TYPES = None

TOP_N_GENES = 3000
MIN_DETECTION_RATE = 0.05      # drop genes detected in <5% of nuclei BEFORE HVG
HVG_METHOD = "dispersion"      # "dispersion"     -> mean-binned normalized
                               #                     dispersion (seurat flavour)
                               # "dispersion_raw" -> legacy raw var/mean
CORR_THRESHOLD = 0.3
EPOCHS = 300
K_FOLDS = 10
USE_CV = True
STRICT_CV = True
LR = 0.001
DROPOUT = 0.0
RANDOM_STATE = 25
MIN_CELLS = 20
N_POOL = 64

# --- Permutation null + FDR ---
USE_PERMUTATION = True
N_PERMUTATIONS = 200
FDR_ALPHA = 0.05
PERM_RETRAIN = False           # True: refit a fresh model inside every
                               # permutation on the pseudo-control profile.
                               # Slow (~N_PERMUTATIONS x EPOCHS) but the null is
                               # not conditional on the control-fitted model.
                               # False: reuse the control-trained model (fast,
                               # conditional null).

# --- Baseline comparison ---
USE_PCA_BASELINE = True
PCA_N_COMPONENTS = 10

# --- Direction / ranking consistency ---
LOG2FC_FLOOR = 0.1
EXCLUDE_NC_FROM_RANKED = True  # Drop "NC" genes from the ranked output so top
                               # hits always carry a direction.

# --- Artifact gene filtering ---
FILTER_ARTIFACTS = True
FILTER_BEFORE_HVG = True
DROP_SEX = True
DROP_LINCRNA = False
DROP_ANTISENSE = False

OUTPUT_DIR = "results_per_celltype"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# Data helpers
# --------------------------------------------------------------------------- #
def data_normalize(adata):
    sc.pp.normalize_total(adata, target_sum=1e6)
    sc.pp.log1p(adata)
    return adata


def load_meta(path):
    """Load the metadata TSV, indexed by cell barcode."""
    meta = pd.read_csv(path, sep="\t")
    if META_BARCODE_COL is None:
        meta = meta.set_index(meta.columns[0])
    else:
        if META_BARCODE_COL not in meta.columns:
            raise KeyError(
                f"'{META_BARCODE_COL}' not in meta.tsv. Columns: {list(meta.columns)}")
        meta = meta.set_index(META_BARCODE_COL)
    if CELLTYPE_COL not in meta.columns:
        raise KeyError(
            f"'{CELLTYPE_COL}' not in meta.tsv. Columns: {list(meta.columns)}")
    if DONOR_COL is not None and DONOR_COL not in meta.columns:
        raise KeyError(
            f"DONOR_COL='{DONOR_COL}' not in meta.tsv. Columns: {list(meta.columns)}")
    if DONOR_COL is None:
        warnings.warn(
            "DONOR_COL is None: CV and the permutation null will operate at the "
            "CELL level, not the DONOR level. This treats cells from the same "
            "individual as independent and inflates effective sample size. "
            "Set DONOR_COL to a column in meta.tsv for donor-aware analysis.",
            stacklevel=2,
        )
    return meta


def attach_celltype_from_meta(adata, meta, name):
    cols = [CELLTYPE_COL]
    if DONOR_COL is not None and DONOR_COL in meta.columns:
        cols.append(DONOR_COL)
    labels = meta[cols].reindex(adata.obs_names)
    matched = labels[CELLTYPE_COL].notna().sum()
    total = adata.n_obs
    if matched == 0:
        raise ValueError(
            f"[{name}] No barcodes in the .h5ad matched meta.tsv. "
            f"Example adata barcode: {adata.obs_names[0]!r}; "
            f"example meta barcode: {meta.index[0]!r}.")
    if matched < total:
        print(f"  [{name}] {total - matched}/{total} cells had no meta entry "
              f"and will be dropped.")
    adata = adata[labels[CELLTYPE_COL].notna()].copy()
    keep = labels[CELLTYPE_COL].notna()
    for c in cols:
        adata.obs[c] = labels.loc[keep, c].astype(str).values
    print(f"  [{name}] {adata.n_obs} cells labeled across "
          f"{adata.obs[CELLTYPE_COL].nunique()} cell types.")
    if DONOR_COL is not None and DONOR_COL in adata.obs:
        print(f"  [{name}] donors: {adata.obs[DONOR_COL].nunique()}")
    return adata


def _to_dense(x):
    return x.toarray() if hasattr(x, "toarray") else np.asarray(x)


def select_highly_variable_genes(top_n_genes, adata_control, adata_patient, method):
    """Select HVGs on the combined control+patient matrix (order-preserving).

    Two changes from the original raw-Fano implementation:

    (1) DETECTION FLOOR. Genes detected in fewer than MIN_DETECTION_RATE of
        nuclei are removed before selection. Raw var/mean rises as the mean
        falls, so without a floor the universe fills with genes seen in a
        handful of nuclei, and a deviation-based score on those genes measures
        sampling noise rather than expression change.

    (2) MEAN-BINNED NORMALIZED DISPERSION ("seurat" flavour). Genes are binned
        by mean expression and dispersion is z-scored within each bin, so a
        gene competes against others at a similar expression level instead of
        against the whole matrix. This is the field-standard correction.

    Set HVG_METHOD = "dispersion_raw" to reproduce the original behaviour.
    """
    # Control-only universe: patient cells never inform gene selection.
    # Detection floor and dispersion are computed on controls; the selected
    # genes are applied to both groups at the end of this function.
    combined_X = _to_dense(adata_control.X)
    gene_names = np.asarray(adata_control.var_names)

    # ---- (1) detection floor --------------------------------------------- #
    det_rate = (combined_X > 0).mean(axis=0)
    keep_mask = det_rate >= MIN_DETECTION_RATE
    n_dropped = int((~keep_mask).sum())
    if int(keep_mask.sum()) == 0:
        raise ValueError(
            f"Detection floor {MIN_DETECTION_RATE:.1%} removed every gene. "
            f"Lower MIN_DETECTION_RATE.")
    if int(keep_mask.sum()) < top_n_genes:
        warnings.warn(
            f"Detection floor {MIN_DETECTION_RATE:.1%} leaves only "
            f"{int(keep_mask.sum())} genes (< top_n_genes={top_n_genes}); "
            f"using all of them.")
    combined_X = combined_X[:, keep_mask]
    gene_names = gene_names[keep_mask]
    print(f"  [HVG] detection floor {MIN_DETECTION_RATE:.1%}: "
          f"dropped {n_dropped}, {len(gene_names)} genes remain")

    n_select = int(min(top_n_genes, len(gene_names)))

    # ---- (2) selection ---------------------------------------------------- #
    if method == "dispersion_raw":
        gene_means = np.mean(combined_X, axis=0)
        gene_vars = np.var(combined_X, axis=0)
        dispersion = gene_vars / (gene_means + 1e-8)
        top_indices = np.argsort(dispersion)[-n_select:]
        hvg_genes = [gene_names[i] for i in np.sort(top_indices)]

    elif method == "dispersion":
        tmp = ad.AnnData(X=np.ascontiguousarray(combined_X, dtype=np.float32))
        tmp.var_names = list(gene_names)
        sc.pp.highly_variable_genes(tmp, n_top_genes=n_select, flavor="seurat")
        sel = np.asarray(tmp.var["highly_variable"].values, dtype=bool)
        hvg_genes = [g for g, s in zip(gene_names, sel) if s]

    else:
        raise ValueError(f"Unknown HVG selection method: {method}")

    patient_gene_set = set(adata_patient.var_names)
    common_genes = [g for g in hvg_genes if g in patient_gene_set]

    adata_control_hvg = adata_control[:, common_genes]
    adata_patient_hvg = adata_patient[:, common_genes]

    assert list(adata_control_hvg.var_names) == list(adata_patient_hvg.var_names), \
        "Gene names do not match between control and patient datasets."
    return adata_control_hvg, adata_patient_hvg, common_genes


def build_graph_edges(expr_df, threshold):
    """Undirected gene co-expression graph (upper triangle only)."""
    corr = expr_df.corr().values
    n = corr.shape[0]
    iu, ju = np.triu_indices(n, k=1)
    w = corr[iu, ju]
    mask = w > threshold
    src = iu[mask]
    dst = ju[mask]
    weights = w[mask].astype(np.float32)

    if src.size == 0:
        warnings.warn("No edges passed the correlation threshold; graph has no edges.")
        return (torch.empty((2, 0), dtype=torch.long),
                torch.empty((0,), dtype=torch.float))
    edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long).contiguous()
    edge_weight = torch.tensor(weights, dtype=torch.float)
    return edge_index, edge_weight


def pool_gene_profiles(expr_df, n_pool):
    """[cells x genes] -> [genes x n_pool] via per-gene quantile pooling."""
    X = expr_df.values
    qs = np.linspace(0.0, 1.0, n_pool)
    profiles = np.quantile(X, qs, axis=0).T.astype(np.float32)
    return profiles


def bh_fdr(p_vals):
    """Benjamini-Hochberg FDR."""
    p_vals = np.asarray(p_vals, dtype=float)
    n = len(p_vals)
    order = np.argsort(p_vals)
    ranked = p_vals[order]
    q = ranked * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = q
    return np.clip(out, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class GeneGCN(nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 64)
        self.conv3 = GCNConv(64, out_channels)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index, edge_attr)
        x = torch.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index, edge_attr)
        return x


def train_one_model(data_train, data_val, in_dim, epochs, lr, dropout,
                    verbose=False):
    """Train a fresh model on data_train, optionally evaluating on data_val."""
    model = GeneGCN(in_dim, in_dim, dropout).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses, train_rmse, val_rmse = [], [], [], []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out_train = model(data_train)
        loss_train = F.mse_loss(out_train, data_train.x)
        loss_train.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            if data_val is not None:
                loss_val = F.mse_loss(model(data_val), data_val.x)
            else:
                loss_val = loss_train

        train_losses.append(loss_train.item())
        val_losses.append(loss_val.item())
        train_rmse.append(float(np.sqrt(loss_train.item())))
        val_rmse.append(float(np.sqrt(loss_val.item())))

        if verbose and epoch % 50 == 0:
            print(f"    epoch {epoch:>3}  train_mse {loss_train.item():.5f}  "
                  f"val_mse {loss_val.item():.5f}")

    curves = dict(train_losses=train_losses, val_losses=val_losses,
                  train_rmse=train_rmse, val_rmse=val_rmse)
    return model, curves


# --------------------------------------------------------------------------- #
# CV / permutation utilities
# --------------------------------------------------------------------------- #
def _make_folds(n_cells, donors, k_folds, random_state):
    """Return an iterable of (train_idx, val_idx). Uses GroupKFold on donors
    when donors is not None and enough distinct donors are present."""
    if donors is not None:
        donors = np.asarray(donors)
        n_groups = len(np.unique(donors))
        n_splits = min(k_folds, n_groups)
        if n_splits >= 2:
            gkf = GroupKFold(n_splits=n_splits)
            return list(gkf.split(np.arange(n_cells), groups=donors))
    n_splits = min(k_folds, n_cells)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(kf.split(np.arange(n_cells)))


def _permute_donors(combined_df, combined_donors, ctrl_donor_set, rng):
    """Shuffle donor -> group assignment while preserving per-donor cells."""
    donor_to_idx = {}
    for i, d in enumerate(combined_donors):
        donor_to_idx.setdefault(d, []).append(i)
    all_donors = list(donor_to_idx.keys())
    n_ctrl_donors = sum(1 for d in all_donors if d in ctrl_donor_set)
    perm = rng.permutation(len(all_donors))
    ctrl_donor_ids = [all_donors[i] for i in perm[:n_ctrl_donors]]
    ctrl_idx = np.concatenate([donor_to_idx[d] for d in ctrl_donor_ids])
    pat_idx = np.setdiff1d(np.arange(len(combined_donors)), ctrl_idx)
    return (combined_df.iloc[ctrl_idx].reset_index(drop=True),
            combined_df.iloc[pat_idx].reset_index(drop=True))


def _permute_cells(combined_df, n_ctrl, rng):
    perm = rng.permutation(combined_df.shape[0])
    return (combined_df.iloc[perm[:n_ctrl]].reset_index(drop=True),
            combined_df.iloc[perm[n_ctrl:]].reset_index(drop=True))


def _pca_red(control_profiles, patient_profiles, n_components):
    """PCA-baseline reconstruction-error deviation, mirroring the GCN score."""
    n_comp = min(n_components, control_profiles.shape[0] - 1,
                 control_profiles.shape[1] - 1)
    if n_comp < 1:
        return np.zeros(control_profiles.shape[0], dtype=np.float32)
    pca = PCA(n_components=n_comp, random_state=RANDOM_STATE)
    pca.fit(control_profiles)
    ctrl_recon = pca.inverse_transform(pca.transform(control_profiles))
    pat_recon = pca.inverse_transform(pca.transform(patient_profiles))
    ctrl_err = ((ctrl_recon - control_profiles) ** 2).mean(axis=1)
    pat_err = ((pat_recon - patient_profiles) ** 2).mean(axis=1)
    return (pat_err - ctrl_err).astype(np.float32)


# --------------------------------------------------------------------------- #
# Per-cell-type pipeline
# --------------------------------------------------------------------------- #
def _apply_gene_filter_adata(adata):
    kept = filter_artifact_genes(
        list(adata.var_names), drop_sex=DROP_SEX,
        drop_lincrna=DROP_LINCRNA, drop_antisense=DROP_ANTISENSE, verbose=False)
    return adata[:, kept].copy()


def run_cell_type(ct, control_full, patient_full, out_dir):
    print(f"\n=== Cell type: {ct} ===")
    ctrl = control_full[control_full.obs[CELLTYPE_COL] == ct].copy()
    pat = patient_full[patient_full.obs[CELLTYPE_COL] == ct].copy()
    print(f"  control cells: {ctrl.n_obs}   patient cells: {pat.n_obs}")
    if ctrl.n_obs < MIN_CELLS or pat.n_obs < MIN_CELLS:
        print(f"  SKIP (fewer than {MIN_CELLS} cells in one group)")
        return None
    ctrl_donors = ctrl.obs[DONOR_COL].values if (
        DONOR_COL is not None and DONOR_COL in ctrl.obs) else None
    pat_donors = pat.obs[DONOR_COL].values if (
        DONOR_COL is not None and DONOR_COL in pat.obs) else None
    return run_analysis(ctrl, pat, tag=ct, out_dir=out_dir,
                        ctrl_donors=ctrl_donors, pat_donors=pat_donors)


def run_analysis(ctrl, pat, tag, out_dir, ctrl_donors=None, pat_donors=None):
    """Core pipeline for one (control, patient) pair."""
    # ---- Pre-HVG gene filtering ----
    if FILTER_ARTIFACTS and FILTER_BEFORE_HVG:
        n_before = ctrl.n_vars
        ctrl = _apply_gene_filter_adata(ctrl)
        pat = pat[:, list(ctrl.var_names)].copy()
        print(f"  [{tag}] gene filter (pre-HVG): {ctrl.n_vars}/{n_before} genes kept")

    # ---- HVG ----
    ctrl, pat, common_genes = select_highly_variable_genes(
        TOP_N_GENES, ctrl, pat, method=HVG_METHOD)

    if FILTER_ARTIFACTS and not FILTER_BEFORE_HVG:
        kept_genes = filter_artifact_genes(
            common_genes, drop_sex=DROP_SEX,
            drop_lincrna=DROP_LINCRNA, drop_antisense=DROP_ANTISENSE)
        ctrl = ctrl[:, kept_genes].copy()
        pat = pat[:, kept_genes].copy()
        common_genes = kept_genes

    control_df = ctrl.to_df()
    patient_df = pat.to_df()[control_df.columns]
    genes = list(control_df.columns)
    n_genes = len(genes)

    # ---- Graph from control cells ----
    edge_index_full, edge_weight_full = build_graph_edges(control_df, CORR_THRESHOLD)
    edge_index_full = edge_index_full.to(DEVICE)
    edge_weight_full = edge_weight_full.to(DEVICE)
    print(f"  genes (nodes): {n_genes}   edges: {edge_index_full.shape[1]}")

    # ---- Pooled profiles ----
    control_profiles = pool_gene_profiles(control_df, N_POOL)
    patient_profiles = pool_gene_profiles(patient_df, N_POOL)
    control_x = torch.tensor(control_profiles, dtype=torch.float)
    patient_x = torch.tensor(patient_profiles, dtype=torch.float)

    # ---- CV: hold out CELLS (or DONORS) ----
    fold_val_rmse = []
    if USE_CV:
        folds = _make_folds(control_df.shape[0], ctrl_donors, K_FOLDS, RANDOM_STATE)
        for fold_num, (tr_idx, va_idx) in enumerate(folds, start=1):
            tr_df = control_df.iloc[tr_idx]
            va_df = control_df.iloc[va_idx]
            tr_prof = pool_gene_profiles(tr_df, N_POOL)
            va_prof = pool_gene_profiles(va_df, N_POOL)

            if STRICT_CV:
                ei, ew = build_graph_edges(tr_df, CORR_THRESHOLD)
                ei, ew = ei.to(DEVICE), ew.to(DEVICE)
            else:
                ei, ew = edge_index_full, edge_weight_full

            data_tr = Data(x=torch.tensor(tr_prof, dtype=torch.float).to(DEVICE),
                           edge_index=ei, edge_attr=ew)
            data_va = Data(x=torch.tensor(va_prof, dtype=torch.float).to(DEVICE),
                           edge_index=ei, edge_attr=ew)
            m, _ = train_one_model(data_tr, data_va, N_POOL, EPOCHS, LR, DROPOUT)
            m.eval()
            with torch.no_grad():
                recon_va = m(data_va).cpu().numpy()
            fold_val_rmse.append(
                float(np.sqrt(((recon_va - va_prof) ** 2).mean())))
        print(f"  CV val RMSE: {np.mean(fold_val_rmse):.5f} +/- "
              f"{np.std(fold_val_rmse):.5f}")

    # ---- Final model trained on all control cells ----
    data_control = Data(x=control_x.to(DEVICE),
                        edge_index=edge_index_full, edge_attr=edge_weight_full)
    data_patient = Data(x=patient_x.to(DEVICE),
                        edge_index=edge_index_full, edge_attr=edge_weight_full)
    model, _ = train_one_model(data_control, None, N_POOL, EPOCHS, LR, DROPOUT)

    model.eval()
    with torch.no_grad():
        control_recon = model(data_control).cpu().numpy()
        patient_recon = model(data_patient).cpu().numpy()

    r2_per_gene = [r2_score(control_profiles[i], control_recon[i])
                   for i in range(n_genes)]
    print(f"  control reconstruction mean R^2: {np.mean(r2_per_gene):.4f}")

    # ---- Perturbation score ----
    ctrl_err = ((control_recon - control_profiles) ** 2).mean(axis=1)
    pat_err = ((patient_recon - patient_profiles) ** 2).mean(axis=1)
    recon_dev = pat_err - ctrl_err

    # ---- Direction from raw log1p-CPM ----
    ln2 = np.log(2.0)
    mean_ctrl = control_df.values.mean(axis=0)
    mean_pat = patient_df.values.mean(axis=0)
    log2fc = (mean_pat - mean_ctrl) / ln2
    direction = np.where(log2fc > 0, "Up", np.where(log2fc < 0, "Down", "NC"))
    direction = np.where(np.abs(log2fc) < LOG2FC_FLOOR, "NC", direction)

    # ---- Detection rates (audit trail for the ranked output) ----
    det_ctrl = (control_df.values > 0).mean(axis=0)
    det_pat = (patient_df.values > 0).mean(axis=0)

    # ---- Permutation null + FDR ----
    p_vals = np.ones(n_genes, dtype=np.float32)
    q_vals = np.ones(n_genes, dtype=np.float32)
    if USE_PERMUTATION:
        combined_df = pd.concat([control_df, patient_df], axis=0).reset_index(drop=True)
        n_ctrl_cells = control_df.shape[0]
        combined_donors = None
        ctrl_donor_set = None
        if ctrl_donors is not None and pat_donors is not None:
            combined_donors = np.concatenate([np.asarray(ctrl_donors),
                                              np.asarray(pat_donors)])
            ctrl_donor_set = set(ctrl_donors)
        rng = np.random.default_rng(RANDOM_STATE)
        perm_absRED = np.zeros((N_PERMUTATIONS, n_genes), dtype=np.float32)

        for p in range(N_PERMUTATIONS):
            if combined_donors is not None:
                pctrl_df, ppat_df = _permute_donors(
                    combined_df, combined_donors, ctrl_donor_set, rng)
            else:
                pctrl_df, ppat_df = _permute_cells(combined_df, n_ctrl_cells, rng)

            pctrl_prof = pool_gene_profiles(pctrl_df, N_POOL)
            ppat_prof = pool_gene_profiles(ppat_df, N_POOL)
            pctrl_x = torch.tensor(pctrl_prof, dtype=torch.float).to(DEVICE)
            ppat_x = torch.tensor(ppat_prof, dtype=torch.float).to(DEVICE)
            d_pctrl = Data(x=pctrl_x, edge_index=edge_index_full,
                           edge_attr=edge_weight_full)
            d_ppat = Data(x=ppat_x, edge_index=edge_index_full,
                          edge_attr=edge_weight_full)

            if PERM_RETRAIN:
                perm_model, _ = train_one_model(
                    d_pctrl, None, N_POOL, EPOCHS, LR, DROPOUT)
                perm_model.eval()
                with torch.no_grad():
                    pctrl_rec = perm_model(d_pctrl).cpu().numpy()
                    ppat_rec = perm_model(d_ppat).cpu().numpy()
                del perm_model
            else:
                with torch.no_grad():
                    pctrl_rec = model(d_pctrl).cpu().numpy()
                    ppat_rec = model(d_ppat).cpu().numpy()

            pctrl_err = ((pctrl_rec - pctrl_prof) ** 2).mean(axis=1)
            ppat_err = ((ppat_rec - ppat_prof) ** 2).mean(axis=1)
            perm_absRED[p] = np.abs(ppat_err - pctrl_err)

        obs_absRED = np.abs(recon_dev)
        p_vals = (perm_absRED >= obs_absRED[None, :]).mean(axis=0).astype(np.float32)
        p_vals = np.maximum(p_vals, 1.0 / (N_PERMUTATIONS + 1))
        q_vals = bh_fdr(p_vals)

    # ---- PCA baseline ----
    pca_red = np.zeros(n_genes, dtype=np.float32)
    spearman_gcn_vs_pca = np.nan
    if USE_PCA_BASELINE:
        pca_red = _pca_red(control_profiles, patient_profiles, PCA_N_COMPONENTS)
        try:
            spearman_gcn_vs_pca = float(spearmanr(np.abs(recon_dev),
                                                  np.abs(pca_red)).correlation)
        except Exception:
            spearman_gcn_vs_pca = np.nan

    # ---- Signed rank score (direction x magnitude) ----
    signed_rank = np.sign(log2fc) * np.abs(recon_dev)

    gene_df = pd.DataFrame({
        "ReconErrorDeviation": recon_dev,
        "absRED": np.abs(recon_dev),
        "signed_rank_score": signed_rank,
        "log2FC_raw": log2fc,
        "Direction": direction,
        "MeanPatient": mean_pat,
        "MeanControl": mean_ctrl,
        "DetectionRatePatient": det_pat,
        "DetectionRateControl": det_ctrl,
        "PatientError": pat_err,
        "ControlError": ctrl_err,
        "perm_p": p_vals,
        "perm_q": q_vals,
        "PCA_RED": pca_red,
        "PCA_absRED": np.abs(pca_red),
    }, index=genes).sort_values(by="absRED", ascending=False)

    if EXCLUDE_NC_FROM_RANKED:
        gene_df = gene_df[gene_df["Direction"] != "NC"]

    tag_safe = _safe_name(tag)
    gene_df.to_csv(os.path.join(out_dir, f"perturbation_scores_{tag_safe}.csv"))
    torch.save(model.state_dict(),
               os.path.join(out_dir, f"gene_gcn_model_{tag_safe}.pth"))

    n_sig = int((gene_df["perm_q"] < FDR_ALPHA).sum()) if USE_PERMUTATION else 0

    return dict(
        group=tag,
        n_control=int(ctrl.n_obs),
        n_patient=int(pat.n_obs),
        n_genes=n_genes,
        n_edges=int(edge_index_full.shape[1]),
        hvg_method=HVG_METHOD,
        min_detection_rate=MIN_DETECTION_RATE,
        median_detection_universe=float(np.median((det_ctrl + det_pat) / 2.0)),
        cv_val_rmse_mean=(float(np.mean(fold_val_rmse)) if fold_val_rmse else None),
        cv_val_rmse_std=(float(np.std(fold_val_rmse)) if fold_val_rmse else None),
        recon_mean_r2=float(np.mean(r2_per_gene)),
        n_significant_fdr05=n_sig,
        spearman_gcn_vs_pca=spearman_gcn_vs_pca,
        used_donor_grouping=bool(ctrl_donors is not None),
        perm_retrain=bool(PERM_RETRAIN),
    )


# --------------------------------------------------------------------------- #
# small utilities
# --------------------------------------------------------------------------- #
def _safe_name(s):
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(s))


def merge_overall(out_dir, cell_types, topk=3):
    """Merge per-cell-type perturbation scores into ONE ranked list.

    Overall scores per gene:
      * OverallScore_max     : max |RED| across cell types
      * OverallScore_top3mean: mean of the top-3 |RED| across cell types
      * SourceCellType       : cell type giving the max
      * NumCellTypesScored   : how many cell types ranked that gene
    """
    score_cols = {}
    for ct in cell_types:
        path = os.path.join(out_dir, f"perturbation_scores_{_safe_name(ct)}.csv")
        if not os.path.exists(path):
            continue
        s = pd.read_csv(path, index_col=0)["absRED"]
        score_cols[ct] = s

    if not score_cols:
        print("No per-cell-type score files found to merge.")
        return

    mat = pd.DataFrame(score_cols)

    def _topk_mean(row):
        vals = row.dropna().sort_values(ascending=False)[:topk]
        return vals.mean() if len(vals) else np.nan

    overall = pd.DataFrame({
        "OverallScore_max": mat.max(axis=1, skipna=True),
        "OverallScore_top3mean": mat.apply(_topk_mean, axis=1),
        "SourceCellType": mat.idxmax(axis=1),
        "NumCellTypesScored": mat.notna().sum(axis=1),
    }).sort_values(by="OverallScore_max", ascending=False)

    overall.to_csv(os.path.join(out_dir, "overall_perturbation_scores.csv"))
    mat.to_csv(os.path.join(out_dir, "score_matrix_gene_by_celltype.csv"))
    print(f"\nMerged overall list written to "
          f"{out_dir}/overall_perturbation_scores.csv "
          f"({overall.shape[0]} genes across {mat.shape[1]} cell types).")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading data...")
    control_full = data_normalize(sc.read(CONTROL_PATH))
    patient_full = data_normalize(sc.read(PATIENT_PATH))

    shared = set(control_full.obs_names) & set(patient_full.obs_names)
    if shared:
        raise ValueError(
            f"{len(shared)} barcodes are shared between control and patient "
            f"(.h5ad). Example: {next(iter(shared))!r}. Prefix/suffix barcodes "
            f"before running, or the metadata will mislabel both datasets.")

    print("Loading cell-type metadata...")
    meta = load_meta(META_PATH)
    control_full = attach_celltype_from_meta(control_full, meta, "control")
    patient_full = attach_celltype_from_meta(patient_full, meta, "patient")

    if FILTER_ARTIFACTS:
        rep = filter_report(list(control_full.var_names), drop_sex=DROP_SEX,
                            drop_lincrna=DROP_LINCRNA,
                            drop_antisense=DROP_ANTISENSE)
        rows = [(cls, g) for cls, genes in rep.items() for g in genes]
        pd.DataFrame(rows, columns=["class", "gene"]).to_csv(
            os.path.join(OUTPUT_DIR, "filtered_genes_audit.csv"), index=False)
        print(f"  Artifact filter audit: {len(rows)} genes flagged "
              f"-> {OUTPUT_DIR}/filtered_genes_audit.csv")

    if CELL_TYPES is None:
        ctrl_types = set(control_full.obs[CELLTYPE_COL].unique())
        pat_types = set(patient_full.obs[CELLTYPE_COL].unique())
        cell_types = sorted(ctrl_types & pat_types)
        print(f"Auto-detected {len(cell_types)} shared cell types: {cell_types}")
    else:
        cell_types = CELL_TYPES

    # ---- GLOBAL model ----
    # Donor info is passed here too, so donor-aware CV / permutation are used
    # consistently across global and per-cell-type runs.
    print("\n##### GLOBAL run (all cells) #####")
    g_ctrl_donors = (control_full.obs[DONOR_COL].values
                     if DONOR_COL is not None and DONOR_COL in control_full.obs
                     else None)
    g_pat_donors = (patient_full.obs[DONOR_COL].values
                    if DONOR_COL is not None and DONOR_COL in patient_full.obs
                    else None)
    global_summary = run_analysis(control_full.copy(), patient_full.copy(),
                                  tag="ALL_CELLS", out_dir=OUTPUT_DIR,
                                  ctrl_donors=g_ctrl_donors,
                                  pat_donors=g_pat_donors)
    if global_summary:
        pd.DataFrame([global_summary]).to_csv(
            os.path.join(OUTPUT_DIR, "summary_global.csv"), index=False)
        print(f"Global ranked list: "
              f"{OUTPUT_DIR}/perturbation_scores_ALL_CELLS.csv")

    # ---- Per-cell-type loop ----
    print("\n##### PER-CELL-TYPE runs #####")
    summaries = []
    for ct in cell_types:
        try:
            res = run_cell_type(ct, control_full, patient_full, OUTPUT_DIR)
            if res is not None:
                summaries.append(res)
        except Exception as e:
            print(f"  ERROR on cell type {ct}: {e}")

    if summaries:
        pd.DataFrame(summaries).to_csv(
            os.path.join(OUTPUT_DIR, "summary_per_celltype.csv"), index=False)
        print(f"\nDone. Summary written to "
              f"{OUTPUT_DIR}/summary_per_celltype.csv")

    merge_overall(OUTPUT_DIR, cell_types)


if __name__ == "__main__":
    main()