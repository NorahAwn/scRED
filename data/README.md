
# Data

The datasets are **not** included in this repository. Download them from the
original sources and place them as described below. The expected paths match the
defaults in `src/model_per_celltype.py`:

```python
CONTROL_PATH = "ASD dataset/control_adata.h5ad"
PATIENT_PATH = "ASD dataset/patient_adata.h5ad"
```

## 1. Velmeshev et al. 2019 — ASD cortex snRNA-seq (primary cohort)

- Source: UCSC Cell Browser — https://autism.cells.ucsc.edu/
- Reference: Velmeshev et al., *Science* 2019, 364(6441):685–689.
- 104,559 nuclei, 15 ASD + 16 control donors, prefrontal and anterior
  cingulate cortex, 17 cell types.
- Split the cohort into control and patient objects and save as:
  - `data/ASD dataset/control_adata.h5ad`
  - `data/ASD dataset/patient_adata.h5ad`
- The cell-type label is read from the `cluster` column and the donor label
  from the `individual` column of the cell metadata (`CELLTYPE_COL` /
  `DONOR_COL` in `model_per_celltype.py`).

## 2. SFARI Gene — ASD risk-gene reference

- Source: https://gene.sfari.org (human gene module; export the CSV).
- Save as: `data/SFARI-Gene_genes.csv`
- Columns used by `src/sfari_scoring.py`: `gene-symbol`, `gene-score`,
  `syndromic`. Confidence weighting: score 1 → 1.00, score 2 → 0.50,
  score 3 or syndromic-only → 0.25.

## Notes

- Expression is normalised to CPM and log1p-transformed at load time; provide
  raw or CPM counts, not pre-scaled data.
- XIST and Y-linked transcripts are removed automatically (the cohort is
  sex-imbalanced); see `src/gene_filter.py`.
- A second, independent ASD cortex cohort (Wamsley/Geschwind, PsychENCODE
  study syn51032009) exists but is under NIMH controlled access and is **not**
  required to reproduce the results in this repository.
