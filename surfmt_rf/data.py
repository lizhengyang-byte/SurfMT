"""Data loading and feature building for LightGBM."""
from pathlib import Path

import numpy as np
import pandas as pd

from .features import (
    FEATURE_DIM,
    compute_features,
)


class SurfProData:
    """Loads train/test CSVs and builds tabular features + labels.

    Attributes:
        X: Feature matrix [N, FEATURE_DIM]
        y: Raw targets [N, 6] (NaN-free, zeros filled where missing)
        mask: Binary mask [N, 6]
        fold: Fold label per row [N]
        smiles: List of SMILES
    """

    def __init__(self, csv_path, task_cols=None, temp_mean=None):
        self.csv_path = Path(csv_path)
        df = pd.read_csv(csv_path)
        if task_cols is None:
            from .metrics import TASK_COLS
            task_cols = TASK_COLS

        smiles = df["SMILES"].tolist()

        # Temperature: fill missing with mean (default 25 for test)
        if "temp" in df.columns:
            temps_raw = df["temp"].astype(float).values
            if temp_mean is not None:
                temps = np.where(np.isnan(temps_raw), temp_mean, temps_raw)
            else:
                temps = np.where(np.isnan(temps_raw), 25.0, temps_raw)
        else:
            temps = np.full(len(df), 25.0, dtype=float)

        # Build features (N, FEATURE_DIM)
        X = compute_features(smiles, temps)

        # Targets + mask
        y = np.zeros((len(df), len(task_cols)), dtype=np.float32)
        mask = np.zeros((len(df), len(task_cols)), dtype=np.float32)
        for t, col in enumerate(task_cols):
            col_vals = df[col].astype(float).values
            present = ~np.isnan(col_vals)
            mask[:, t] = present
            y[:, t] = np.where(present, col_vals, 0.0)

        fold = np.full(len(df), -1, dtype=int)
        if "fold" in df.columns:
            fold = df["fold"].fillna(-1).astype(int).values

        self.X = X
        self.y = y
        self.mask = mask
        self.fold = fold
        self.smiles = smiles
        self.num_tasks = len(task_cols)
        self.task_cols = task_cols


def load_data(data_dir="data/surfpro"):
    """Load train and test data. Returns (train, test)."""
    data_dir = Path(data_dir)
    if not data_dir.is_absolute():
        data_dir = Path(__file__).resolve().parent.parent / data_dir

    train = SurfProData(data_dir / "surfpro_train.csv")

    # Test uses train temp mean for missing temp imputation
    test = SurfProData(data_dir / "surfpro_test.csv", temp_mean=25.0)

    print(f"Train: {train.X.shape[0]} samples, {train.X.shape[1]} features")
    print(f"Test:  {test.X.shape[0]} samples")
    return train, test