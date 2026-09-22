# SurfMT-LightGBM

LightGBM multi-task version of SurfMT-GNN for surfactant property prediction.

## Approach

SurfMT-GNN uses a GNN to predict 6 surfactant interface properties. LightGBM
cannot natively handle the heavily masked multi-task structure, so this version
trains **one LightGBM regressor per task**, each using only the labeled samples
for that task.

**Features (per molecule, 2061-dim tabular):**
- ECFP4 Morgan fingerprint (2048-bit)
- 12 RDKit 2D descriptors
- Temperature (scalar)

Input requirements: only SMILES + temperature (no molecular graph), so features
are cheap and fast to compute.

**Data split:** consistent with the GNN version — fold 9 is the held-out
validation fold (early stopping), the rest is training; the fixed test set is
evaluated.

## Usage

```bash
# From the project root
python -m surfmt_lgb.main --seed 42 --output_dir outputs/lgb_seed42
```

Or run directly in the folder:

```bash
python surfmt_lgb/main.py --seed 42 --output_dir outputs/lgb_seed42
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--seed` | 42 | Random seed |
| `--output_dir` | outputs/lgb_seed42 | Results output dir |
| `--max_rounds` | 2000 | Max boosting rounds per task |
| `--early_stopping` | 80 | Early stopping rounds per task |
| `--val_fold` | 9 | Validation fold column value |

## Files

- `features.py` — feature extraction (ECFP4 + descriptors + temperature)
- `data.py` — CSV loading, missing-label masks, fold split
- `train.py` — per-task LightGBM training with early stopping
- `metrics.py` — masked R2 / RMSE / MAE
- `main.py` — end-to-end entry point

## Configuration

Training hyperparameters live in `train.py` as `LGB_PARAMS` (standard LightGBM
GDBT settings tuned for small molecular datasets).

## Notes

- **No target normalization**: tree models are scale-invariant, so raw targets
  are used directly (metrics computed on raw scale).
- **Missing labels**: each task trains on its own labeled subset; no imputation.
- **Temperature imputation**: missing temperature filled with 25.0 (the data's
  approximate mean; the feature is near-constant across the dataset anyway).