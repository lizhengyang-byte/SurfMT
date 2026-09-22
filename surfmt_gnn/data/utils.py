"""Utility functions for dataset rescaling and scaler computation.

Used primarily for cross-validation to avoid scaler data leakage:
each fold should compute its own scaler from the training fold only,
then apply it to both train and validation folds.
"""
import numpy as np
import torch


def compute_scaler_from_subset(dataset, indices: list) -> dict:
    """Compute scaler statistics from a subset of the dataset.

    Uses raw (unnormalized) values stored in each Data object:
      - desc_raw: raw descriptor values
      - desc_valid: which descriptors were successfully computed
      - y_raw: raw target values
      - mask: which targets are present
      - temp_raw: raw temperature values
      - temp_mask: which temperatures are present

    Args:
        dataset: SurfProDataset (or any PyG dataset with the above attributes).
        indices: List of sample indices to use for scaler computation.

    Returns:
        Dict with keys: desc_mean, desc_std, target_mean, target_std,
        temp_mean, temp_std (all numpy arrays or floats, float32).
    """
    # Collect raw values from the subset
    all_desc_raw = []
    all_desc_valid = []
    all_targets = [[] for _ in range(6)]
    all_temp = []

    for idx in indices:
        data = dataset[idx]

        # Descriptors
        all_desc_raw.append(data.desc_raw.numpy())
        all_desc_valid.append(data.desc_valid.numpy().astype(bool))

        # Targets
        y = data.y_raw.numpy().squeeze(0)  # (6,)
        m = data.mask.numpy().squeeze(0).astype(bool)  # (6,)
        for t in range(6):
            if m[t]:
                all_targets[t].append(y[t])

        # Temperature
        if data.temp_mask.item() > 0.5:
            all_temp.append(data.temp_raw.item())

    # ---- Descriptor scaler (per-dim, only valid values, ddof=1) ----
    all_desc_raw = np.stack(all_desc_raw, axis=0)  # (N, 12)
    all_desc_valid = np.stack(all_desc_valid, axis=0)  # (N, 12)

    desc_mean = np.zeros(12, dtype=np.float32)
    desc_std = np.zeros(12, dtype=np.float32)
    for i in range(12):
        vals = all_desc_raw[all_desc_valid[:, i], i]
        if len(vals) < 2:
            desc_mean[i] = 0.0
            desc_std[i] = 1.0
        else:
            desc_mean[i] = np.mean(vals)
            desc_std[i] = np.std(vals, ddof=1).clip(min=1e-8)

    # ---- Target scaler (per-task, only valid values, ddof=1) ----
    target_mean = np.zeros(6, dtype=np.float32)
    target_std = np.zeros(6, dtype=np.float32)
    for t in range(6):
        vals = np.array(all_targets[t], dtype=np.float32)
        if len(vals) < 2:
            target_mean[t] = 0.0
            target_std[t] = 1.0
        else:
            target_mean[t] = np.mean(vals)
            target_std[t] = np.std(vals, ddof=1).clip(min=1e-8)

    # ---- Temperature scaler ----
    if len(all_temp) < 2:
        temp_mean = 25.0
        temp_std = 35.0
    else:
        temp_mean = float(np.mean(all_temp))
        temp_std = float(np.std(all_temp, ddof=1).clip(min=1e-8))

    return {
        "desc_mean": desc_mean,
        "desc_std": desc_std,
        "target_mean": target_mean,
        "target_std": target_std,
        "temp_mean": temp_mean,
        "temp_std": temp_std,
    }


def rescale_dataset(dataset, scaler: dict) -> None:
    """Re-normalize all samples in a dataset using the given scaler.

    Modifies the dataset in-place by updating:
      - data.descriptors (normalized descriptors)
      - data.temp_norm (normalized temperature)
      - data.y (normalized targets)

    Uses raw values (desc_raw, temp_raw, y_raw) and masks to recompute.

    Operates directly on PyG InMemoryDataset's internal collated storage
    (dataset._data) for efficiency.

    Args:
        dataset: SurfProDataset (or PyG InMemoryDataset with raw values).
        scaler: Dict with desc_mean, desc_std, target_mean, target_std,
                temp_mean, temp_std (numpy arrays / floats).
    """
    desc_mean = torch.tensor(np.asarray(scaler["desc_mean"], dtype=np.float32))
    desc_std = torch.tensor(np.asarray(scaler["desc_std"], dtype=np.float32))
    target_mean = torch.tensor(np.asarray(scaler["target_mean"], dtype=np.float32))
    target_std = torch.tensor(np.asarray(scaler["target_std"], dtype=np.float32))
    temp_mean = float(scaler["temp_mean"])
    temp_std = float(scaler["temp_std"])

    data = dataset._data
    num_graphs = data.y.shape[0]

    # ---- Targets (y): [num_graphs, 6] ----
    # y_norm = (y_raw - mean) / std, then zero out missing entries
    y_raw = data.y_raw  # [N, 6]
    mask = data.mask    # [N, 6]
    y_norm = (y_raw - target_mean) / target_std
    dataset._data.y = y_norm * mask

    # ---- Temperature: [num_graphs] ----
    # Only normalize where temp_mask == 1; set to 0 elsewhere
    temp_raw = data.temp_raw      # [N]
    temp_mask = data.temp_mask    # [N]
    temp_norm = (temp_raw - temp_mean) / temp_std
    dataset._data.temp_norm = temp_norm * temp_mask

    # ---- Descriptors: flat [num_graphs * 12] ----
    # Reshape to [N, 12], fill invalid with mean, normalize, flatten back
    desc_raw_flat = data.desc_raw      # [N*12]
    desc_valid_flat = data.desc_valid  # [N*12]

    desc_raw = desc_raw_flat.view(num_graphs, 12)   # [N, 12]
    desc_valid = desc_valid_flat.view(num_graphs, 12)  # [N, 12]

    # Fill failed descriptors with training set mean
    desc_filled = desc_raw.clone()
    desc_filled[desc_valid < 0.5] = desc_mean.expand(num_graphs, 12)[desc_valid < 0.5]

    # Z-score normalize
    desc_norm = (desc_filled - desc_mean) / desc_std  # [N, 12]
    dataset._data.descriptors = desc_norm.flatten()  # back to [N*12]

    # Clear PyG's per-sample cache so that future __getitem__ calls
    # pick up the updated values from the collated storage.
    # PyG InMemoryDataset caches separated Data objects in _data_list,
    # so modifying _data alone is not reflected in subsequent indexing.
    if hasattr(dataset, '_data_list'):
        dataset._data_list = None

    # Also update dataset-level scaler attributes for consistency
    dataset.desc_mean = scaler["desc_mean"].astype(np.float32)
    dataset.desc_std = scaler["desc_std"].astype(np.float32)
    dataset.target_mean = scaler["target_mean"].astype(np.float32)
    dataset.target_std = scaler["target_std"].astype(np.float32)
    dataset.temp_mean = temp_mean
    dataset.temp_std = temp_std
