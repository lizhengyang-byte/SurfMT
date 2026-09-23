# SurfMT-LightGBM

LightGBM multi-task version of SurfMT-GNN for surfactant property prediction.

## Approach

SurfMT-GNN uses a GNN to predict 6 surfactant interface properties. LightGBM
cannot natively handle the heavily masked multi-task structure, so this version
trains **one LightGBM regressor per task**, each using only the labeled samples
for that task.

**Features (per molecule, 4329-dim tabular):**
- ECFP4 Morgan fingerprint (2048-bit)
- ECFP6 Morgan fingerprint (2048-bit)
- MACCS structural keys (167-bit)
- 65 expanded RDKit 2D descriptors
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
| `--max_rounds` | 5000 | Max boosting rounds per task |
| `--early_stopping` | 100 | Early stopping rounds per task |

## Analysis: tree-model interpretability

Run the professional analysis bundle (feature importance, SHAP, fingerprint
substructure mapping, markdown report):

```bash
python surfmt_lgb/analyze.py --seed 42 --output_dir outputs/lgb_analysis
```

`analyze.py` retrains the per-task models, then writes to
`outputs/lgb_analysis/`:

- `feature_importance/` — native gain/split, block aggregation, permutation
  importance, descriptor gain heatmap.
- `shap/` — per-task beeswarm summary, dependence, waterfall, global block
  |SHAP|, raw SHAP values (`shap_values.npz`).
- `substructures/top_fingerprint_bits.json` — reverse-mapped Morgan substructures
  for the most important fingerprint bits.
- `report.md` — structured Chinese analysis report.
- `analysis_summary.json` — machine-readable summary.

Options: `--task <idx>` (single task), `--no_shap` (skip SHAP), `--shap_samples`
(rows sub-sampled per task for SHAP), `--top_bits` (fingerprint bits to map).
Add `shap` to your env via `pip install -r requirements.txt` (now includes it).

## Files

- `features.py` — feature extraction (ECFP4/ECFP6 + MACCS + 65 descriptors + temperature)
- `data.py` — CSV loading, missing-label masks, fold split
- `train.py` — per-task LightGBM training with early stopping
- `metrics.py` — masked R2 / RMSE / MAE
- `interpret.py` — interpretability helpers (importance, SHAP, substructure mapping)
- `analyze.py` — end-to-end tree-model analysis entry point
- `main.py` — end-to-end training entry point

## Configuration

Training hyperparameters live in `train.py` as `LGB_PARAMS` (standard LightGBM
GDBT settings tuned for small molecular datasets).

## Notes

- **Target transform, not feature normalization**: the *target* is optionally
  log-transformed (based on skew) then Z-standardized so the RMSE early-stop
  metric is comparable across tasks; predictions are denormalized back to the
  raw scale. LightGBM is scale-invariant on *features*, so features are used raw.
- **Missing labels**: each task trains on its own labeled subset; no imputation.
- **Temperature imputation**: missing temperature filled with 25.0 (the data's
  approximate mean; the feature is near-constant across the dataset anyway).