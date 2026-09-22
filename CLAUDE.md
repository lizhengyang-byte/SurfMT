# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SurfMT-GNN — a PyTorch implementation of a multi-task graph neural network for surfactant property prediction, based on the 2026 Digital Discovery paper. A single model predicts 6 surfactant interface properties simultaneously from molecular graph + temperature + RDKit descriptors, with missing labels handled via masked MSE loss.

## Architecture (Big Picture)

**Three-branch encoder → fusion → shared layer → 6 task heads**

1. **Graph encoder** (`models/attentive_fp.py`): Custom multi-head AttentiveFP (GATEConv + GATConv with 4 heads, GRU state updates, super-node readout). 3 layers, 256-dim output.
2. **Temperature encoder** (`models/surfmt_gnn.py`): MLP 1→32→64 (GELU). Output is zeroed out via `temp_mask` when temperature is missing.
3. **Descriptor encoder** (`models/surfmt_gnn.py`): MLP 12→32→64 (ReLU). 12 RDKit descriptors (MolLogP, TPSA, MolWt, etc.) Z-score normalized.

**Fusion**: concat(256+64+64=384) → MLP 384→256 (dropout=0.1) → Linear+LayerNorm+ReLU → 128-dim shared → 6 parallel MLP heads (128→64→32→1).

### Key non-obvious design decisions

- **`y` and `mask` stored as `[1, 6]` per graph** (not `[6]`). PyG concatenates 1D vectors across graphs in a batch, so `[6]` would flatten to `[B*6]`. Storing as `[1, 6]` ensures batching stacks to `[B, 6]`.
- **Target Z-score normalization is essential**. Property magnitudes differ wildly (Γ_max ~1e-6, π_CMC ~35), so raw MSE is meaningless. Model predicts normalized values; denormalize before computing metrics.
- **Scalers saved separately** (`scaler_{split}.pt`). `InMemoryDataset.load()` doesn't restore custom Python attributes, so desc_mean/std, target_mean/std, temp_mean/std, type_map are persisted in a sidecar file and reloaded after `self.load()`.
- **GATEConv import path**: In PyG 2.8.0, it's in `torch_geometric.nn.models.attentive_fp`, not the main `nn` namespace.
- **Early stopping uses `avg_r2`** (higher-is-better), not validation loss. Task-weighted normalized loss doesn't directly correspond to real-world prediction quality.
- **Scheduler types**: `reduce_on_plateau` (default, most stable for small data), `cosine`, `cosine_restart`. ReduceLROnPlateau steps with `val_avg_r2` as the metric.

### Feature dimensions

- Atom: 39-dim (atomic number 12 + degree 6 + formal charge 5 + hybridization 5 + aromatic 1 + H count 5 + chirality 3 + ring 1 + radical 1)
- Bond: 10-dim (bond type 4 + conjugated 1 + ring 1 + stereo 4)
- Descriptors: 12-dim

### Task index mapping

| idx | name      | CSV column  | approx data completeness |
|-----|-----------|-------------|--------------------------|
| 0   | pCMC      | pCMC        | ~90%                     |
| 1   | gamma_CMC | AW_ST_CMC   | ~58%                     |
| 2   | Gamma_max | Gamma_max   | ~42% (lowest)            |
| 3   | A_min     | Area_min    | ~48%                     |
| 4   | pi_CMC    | Pi_CMC      | ~57%                     |
| 5   | pC20      | pC20        | ~90%                     |

## Commands

```bash
# Install deps
pip install -r requirements.txt

# Single model train + test evaluation (quickest feedback)
python surfmt_gnn/scripts/train_single.py --seed 42 --output_dir outputs/single_seed42

# 10-fold cross-validation (uses pre-defined `fold` column in CSV)
python surfmt_gnn/scripts/train_cv.py --output_dir outputs/cv_seed42

# Full deep ensemble (6 seeds × 10 folds = 60 models)
python surfmt_gnn/scripts/train_ensemble.py --output_dir outputs/ensemble

# Evaluate ensemble with uncertainty (coverage, Spearman, error ratio)
python surfmt_gnn/scripts/evaluate_ensemble.py --ensemble_dir outputs/ensemble --output_dir outputs/eval

# LightGBM baseline (per-task regressors)
python surfmt_lgb/main.py --seed 42 --output_dir outputs/lgb_seed42
```

All scripts are run from the project root. There is no build step, no lint config, and no test suite — this is a research codebase.

## Where Things Live

- **Hyperparameters**: `surfmt_gnn/config.py` — single `Config` dataclass, change defaults there.
- **Data pipeline**: `surfmt_gnn/data/` — featurizer → descriptors → dataset.
- **Model forward pass**: `surfmt_gnn/models/surfmt_gnn.py` — the 3-branch architecture.
- **Training loop**: `surfmt_gnn/training/trainer.py` — Trainer class with early stopping on `avg_r2`.
- **Loss**: `surfmt_gnn/training/loss.py` — `masked_mse_loss(pred, target, mask, task_weights)`.
- **Scripts**: `surfmt_gnn/scripts/` — GNN entry points that wire config, data, model, and trainer together.
- **LightGBM baseline**: `surfmt_lgb/` — self-contained per-task LightGBM regressors (entry point: `surfmt_lgb/main.py`).

Each model lives in its own top-level folder containing all of its code and its own scripts, so a new model drops in as a new self-contained folder under the project root. `outputs/` is the log/result archive (kept unchanged).

## Data Caching

`SurfProDataset` caches processed graphs under `data/surfpro/processed/data_{split}.pt`. If you change featurization code, delete this directory to force reprocessing. Scalers are cached alongside as `scaler_{split}.pt`.
