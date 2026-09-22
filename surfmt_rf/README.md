# SurfMT-RandomForest

Scikit-Learn Random Forest multi-task version of SurfMT-GNN for surfactant
property prediction. A second tree-based baseline alongside
[`surfmt_lgb`](../surfmt_lgb/) — the same feature and masking strategy, but the
regressor is a RandomForestRegressor instead of LightGBM.

## Approach

SurfMT-GNN uses a GNN to predict 6 surfactant interface properties. Tree models
cannot directly consume the heavily masked multi-task structure, so one
`RandomForestRegressor` is trained per task, each using only the labeled
samples for that task.

**Features (per molecule, 4329-dim tabular):**
- ECFP4 Morgan fingerprint (2048-bit)
- ECFP6 Morgan fingerprint (2048-bit)
- MACCS structural keys (167-bit)
- Expanded RDKit 2D descriptors (65)
- Temperature (scalar)

Input requirements: only SMILES + temperature (no molecular graph), so features
are cheap and fast to compute.

**Tree-count selection:** unlike gradient boosting, Random Forest has no
sequential boosting iterations to early-stop. The number of trees is chosen per
task by cross-validating a small `n_estimators` sweep over the predefined
10-fold column, then the best-count model refits on ALL labeled samples.

**Data split:** consistent with the GNN/LightGBM versions — fold 9 is the
held-out validation fold, the rest is training; the fixed test set is evaluated.

## Usage

```bash
# From the project root
python -m surfmt_rf.main --seed 42 --output_dir outputs/rf_seed42
```

Or run directly in the folder:

```bash
python surfmt_rf/main.py --seed 42 --output_dir outputs/rf_seed42
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--seed` | 42 | Random seed |
| `--seeds` | — | Comma-separated seeds for a multi-seed average ensemble |
| `--output_dir` | outputs/rf_seed42 | Results output dir |
| `--hetero` | False | Heterogeneous ensemble over `DIVERSE_CONFIGS` |

## Files

- `features.py` — feature extraction (ECFP4/6 + MACCS + descriptors + temperature)
- `data.py` — CSV loading, missing-label masks, fold split
- `train.py` — per-task Random Forest training with CV-selected tree count
- `metrics.py` — masked R2 / RMSE / MAE
- `main.py` — end-to-end entry point

## Configuration

Training hyperparameters live in `train.py` as `RF_PARAMS` (scikit-learn
`RandomForestRegressor` settings tuned for small molecular datasets); the
`n_estimators` sweep is in `N_ESTIMATORS_CANDIDATES`.

## Notes

- **No target normalization**: tree models are scale-invariant, so raw targets
  are used directly (metrics computed on raw scale).
- **Missing labels**: each task trains on its own labeled subset; no imputation.
- **Temperature imputation**: missing temperature filled with 25.0 (the data's
  approximate mean; the feature is near-constant across the dataset anyway).