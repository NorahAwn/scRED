r"""
sfari_scoring.py
================
SFARI validation metrics for the GeneGCN benchmark.

Replaces the bare "percentage of top-k that are SFARI genes" with three things
a reviewer can check:

  1. sfari_pct       - plain overlap, unchanged, for comparability.
  2. sfari_weighted  - overlap weighted by SFARI confidence score, so a
                       high-confidence (score-1) hit counts for more than a
                       score-3 hit. Same 0-100 scale as sfari_pct.
  3. A MEASURED NULL - the baseline is computed per cell-type context by
                       drawing top-k gene sets at random from that context's
                       own ranking universe. This replaces the quoted constant
                       the manuscript previously used.

WHY THIS EXISTS
---------------
The ranking universe is 3,000 genes chosen by HVG selection from the filtered
gene pool. SFARI-curated genes are well-expressed and well-annotated, so they
are strongly over-represented in any universe of real genes: measured
baselines are 9-15% depending on cell type, not the ~0.25% the manuscript
assumed. Enrichment must therefore be reported relative to the universe the
ranking was actually drawn from, not relative to the genome.

WEIGHTING
---------
    SFARI score 1 (high confidence) -> 1.00
    SFARI score 2                   -> 0.50
    SFARI score 3                   -> 0.25
    no score but syndromic          -> 0.25
    not curated                     -> 0.00

The scheme is monotone in confidence and declared a priori. Report it in
Methods; do not tune it after seeing results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SFARI_PATH = "SFARI-Gene_genes.csv"
SFARI_SYMBOL_COL = "gene-symbol"
SFARI_SCORE_COL = "gene-score"
SFARI_SYNDROMIC_COL = "syndromic"

SCORE_WEIGHTS = {1: 1.00, 2: 0.50, 3: 0.25}
SYNDROMIC_ONLY_WEIGHT = 0.25

DEFAULT_DEPTHS = [50, 100, 150, 200, 250]
N_NULL_DRAWS = 2000


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_sfari(path: str = SFARI_PATH):
    """Return (sfari_set, weight_map).

    sfari_set  : set of upper-case symbols (membership, as before)
    weight_map : dict symbol -> confidence weight in [0, 1]
    """
    df = pd.read_csv(path)
    symbols = df[SFARI_SYMBOL_COL].astype(str).str.upper()

    scores = pd.to_numeric(df.get(SFARI_SCORE_COL), errors="coerce")
    syndromic = pd.to_numeric(df.get(SFARI_SYNDROMIC_COL), errors="coerce").fillna(0)

    weight_map = {}
    for sym, sc, syn in zip(symbols, scores, syndromic):
        if pd.notna(sc) and int(sc) in SCORE_WEIGHTS:
            w = SCORE_WEIGHTS[int(sc)]
        elif syn == 1:
            w = SYNDROMIC_ONLY_WEIGHT
        else:
            w = 0.0
        # keep the highest weight if a symbol appears more than once
        weight_map[sym] = max(weight_map.get(sym, 0.0), w)

    return set(symbols), weight_map


# --------------------------------------------------------------------------- #
# Observed scores
# --------------------------------------------------------------------------- #
def sfari_overlap(ranked, sfari_set, depths=None):
    """Plain % of the top-k ranked genes that are SFARI-curated."""
    depths = DEFAULT_DEPTHS if depths is None else depths
    return {k: 100.0 * sum(g.upper() in sfari_set for g in ranked[:k]) / max(k, 1)
            for k in depths}


def sfari_weighted(ranked, weight_map, depths=None):
    """Confidence-weighted SFARI score of the top-k, on the same 0-100 scale."""
    depths = DEFAULT_DEPTHS if depths is None else depths
    return {k: 100.0 * sum(weight_map.get(g.upper(), 0.0) for g in ranked[:k])
            / max(k, 1)
            for k in depths}


# --------------------------------------------------------------------------- #
# Measured null
# --------------------------------------------------------------------------- #
def universe_null(universe, sfari_set, weight_map, depths=None,
                  n_draws: int = N_NULL_DRAWS, seed: int = 0):
    """Empirical null for both metrics, drawn from the ranking universe itself.

    Returns a dict keyed by depth:
        {k: {"pct_null": ndarray, "wt_null": ndarray,
             "pct_baseline": float, "wt_baseline": float}}
    plus "universe_size" and "n_sfari_in_universe" at the top level.
    """
    depths = DEFAULT_DEPTHS if depths is None else depths
    uni = [str(g).upper() for g in universe]
    A = len(uni)
    is_sfari = np.array([g in sfari_set for g in uni], dtype=bool)
    weights = np.array([weight_map.get(g, 0.0) for g in uni], dtype=float)

    out = {
        "universe_size": A,
        "n_sfari_in_universe": int(is_sfari.sum()),
        "pct_baseline_universe": 100.0 * is_sfari.mean() if A else float("nan"),
        "wt_baseline_universe": 100.0 * weights.mean() if A else float("nan"),
    }

    rng = np.random.RandomState(seed)
    idx = np.arange(A)
    for k in depths:
        d = min(k, A)
        pct = np.empty(n_draws, dtype=float)
        wt = np.empty(n_draws, dtype=float)
        for i in range(n_draws):
            pick = rng.choice(idx, d, replace=False)
            pct[i] = 100.0 * is_sfari[pick].sum() / d
            wt[i] = 100.0 * weights[pick].sum() / d
        out[k] = {
            "pct_null": pct,
            "wt_null": wt,
            "pct_baseline": float(pct.mean()),
            "wt_baseline": float(wt.mean()),
            "pct_lo": float(np.percentile(pct, 2.5)),
            "pct_hi": float(np.percentile(pct, 97.5)),
            "wt_lo": float(np.percentile(wt, 2.5)),
            "wt_hi": float(np.percentile(wt, 97.5)),
        }
    return out


def enrichment(observed: float, null_vals: np.ndarray):
    """Fold-change over the null mean and a two-sided empirical p-value."""
    mu = float(np.mean(null_vals))
    fold = observed / mu if mu > 0 else float("nan")
    n = len(null_vals)
    # two-sided: how often is the null at least as extreme as observed?
    more_extreme = np.sum(np.abs(null_vals - mu) >= abs(observed - mu))
    p = (more_extreme + 1) / (n + 1)
    return fold, float(p)


# --------------------------------------------------------------------------- #
# Convenience: one row of metrics for a ranked list
# --------------------------------------------------------------------------- #
def score_ranked(ranked, sfari_set, weight_map, null, depths=None):
    """Return {depth: {metric: value}} for one ranked gene list."""
    depths = DEFAULT_DEPTHS if depths is None else depths
    pct = sfari_overlap(ranked, sfari_set, depths)
    wt = sfari_weighted(ranked, weight_map, depths)
    res = {}
    for k in depths:
        pct_fold, pct_p = enrichment(pct[k], null[k]["pct_null"])
        wt_fold, wt_p = enrichment(wt[k], null[k]["wt_null"])
        res[k] = {
            "SFARI_pct": pct[k],
            "SFARI_pct_baseline": null[k]["pct_baseline"],
            "SFARI_pct_fold": pct_fold,
            "SFARI_pct_p": pct_p,
            "SFARI_wt": wt[k],
            "SFARI_wt_baseline": null[k]["wt_baseline"],
            "SFARI_wt_fold": wt_fold,
            "SFARI_wt_p": wt_p,
        }
    return res


if __name__ == "__main__":
    s, w = load_sfari()
    print(f"SFARI symbols: {len(s)}")
    dist = pd.Series(list(w.values())).value_counts().sort_index(ascending=False)
    print("weight distribution:")
    for val, n in dist.items():
        print(f"  {val:.2f}: {n}")