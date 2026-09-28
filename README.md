# scRED — single-cell Reconstruction-Error Deviation

Control-trained, graph-aware prioritisation of disease risk genes from single-nucleus RNA-seq.

scRED is a graph-convolutional autoencoder built on a **cell–cell** similarity graph (not a gene–gene graph). The model is trained on control cortical cells only; genes are ranked by the per-gene deviation in reconstruction error between patient and control cells (RED). Applied to the Velmeshev et al. ASD cortex cohort, scRED recovers SFARI-curated risk genes specifically in cortical neurons and not in glia.

> **Repository note:** this repository is named `GeneGCN`, reflecting an earlier name of the method now described as **scRED**.

📄 Associated manuscript: *"Cell-graph reconstruction error prioritises autism risk genes in cortical neurons but not glia"* (submitted, BMC Bioinformatics).

---

## Key idea

- A 50-component PCA is fitted **on control cells only**; patient cells are projected through that same basis, so their displacement from the healthy manifold becomes the disease signal.
- A weighted k-NN cell–cell graph (k = 15) is built in PC space.
- A two-layer GCN autoencoder is trained to reconstruct per-cell expression from neighbour-aggregated messages, on control cells only.
- For each gene *g*: `RED_g = ε_g(patient) − ε_g(control)`; genes are ranked by `|RED_g|`.
- Enrichment for SFARI risk genes is measured against a **per-context empirical null** and a **control-versus-control negative control** that separates true disease signal from the train/test asymmetry of the score.

---

## Installation

```bash
git clone https://github.com/NorahAwn/GeneGCN.git
cd GeneGCN
python -m venv .venv
# Windows: .venv\Scripts\activate   |   Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

PyTorch Geometric may need wheels matched to your PyTorch/CUDA build — see
https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

---

## Data

Neither dataset is redistributed here; both are freely available from the original sources.

| Resource | Source | Place in |
|---|---|---|
| Velmeshev et al. 2019 snRNA-seq (ASD cortex) | UCSC Cell Browser — https://autism.cells.ucsc.edu/ | `data/velmeshev/` |
| SFARI Gene reference | https://gene.sfari.org | `data/sfari/` |

See `data/README.md` for the exact files expected.

---

## Usage / reproducing the paper

All paper numbers use the **raw** RED score with holdout 0 and mask 0; the two
driver scripts force that config explicitly.

```bash
# Table 2 — donor-disjoint control-versus-control negative control
python src/check_negative_control_donor.py

# Tables 3–4 + Fig 3 + Fig 4 — the 6-method comparison on one shared universe
python src/run_ablations.py --celltypes all --seeds 5

# Figures
python src/make_fig2_negcontrol.py     # reads negctrl_donorlevel.csv
python src/make_fig3_methods.py        # reads ablation_summary.csv
python src/make_fig4_genes.py          # reads ablation_rankings.csv
```

| Output | Script |
|---|---|
| Table 1 baselines | `sfari_scoring.py` (measured null, called by the drivers) |
| Table 2 | `check_negative_control_donor.py` |
| Tables 3–4, Fig 3, Fig 4 | `run_ablations.py` |
| Fig 2 | `make_fig2_negcontrol.py` |

Hyperparameters are fixed a priori (`model_per_celltype.py`): hidden = 64, k = 15,
50 PCs, lr = 1e-3, epochs = 300, dropout = 0, 3000 HVGs, 5 seeds per context.
HVGs are selected on **control cells only**; the negative-control splits are
**donor-disjoint** (8 + 8 control donors; Microglia 7 + 7).

---

## Repository layout

See `REPO_STRUCTURE.md` for the full tree. Scripts live in `src/`; the core
pipeline is `model_per_celltype.py` (config + HVG), `cells_as_nodes.py` (the
scRED cell-graph autoencoder and RED scoring), `gene_filter.py`,
`sfari_scoring.py`, `run_ablations.py`, and `check_negative_control_donor.py`.

---

## Citation

If you use this code, please cite the manuscript (see `CITATION.cff`) and archive:

> Zenodo: https://doi.org/10.5281/zenodo.21132225

When using the Velmeshev data, cite Velmeshev et al., *Science* 2019, and the SFARI Gene database.

## License

MIT — see [LICENSE](LICENSE).
