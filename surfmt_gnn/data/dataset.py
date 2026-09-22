"""SurfPro dataset for PyTorch Geometric."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data import InMemoryDataset, Data

from .featurizer import smiles_to_graph
from .descriptors import compute_descriptors, fit_descriptor_scaler
from .fingerprints import compute_morgan_fp


# Property column name in CSV -> task index
CSV_PROP_COLS = [
    "pCMC",         # idx 0
    "AW_ST_CMC",    # idx 1  (gamma_CMC)
    "Gamma_max",    # idx 2
    "Area_min",     # idx 3  (A_min)
    "Pi_CMC",       # idx 4  (pi_CMC)
    "pC20",         # idx 5
]


class SurfProDataset(InMemoryDataset):
    """SurfPro dataset for multi-task GNN.

    Each sample contains:
      - x, edge_index, edge_attr: molecular graph
      - temp_norm: Z-score normalized temperature
      - temp_raw: raw temperature value (0.0 if missing)
      - temp_mask: 1 if temperature is present, 0 if missing
      - descriptors: 12-dim Z-score normalized RDKit descriptors
      - desc_raw: 12-dim raw (unnormalized) RDKit descriptors
      - desc_valid: 12-dim binary mask (1=computed ok, 0=failed/mean-filled)
      - y: 6-dim normalized target vector (missing entries are 0, mask handles them)
      - y_raw: 6-dim original (unnormalized) target vector
      - mask: 6-dim binary mask (1=present, 0=missing)
      - smiles: SMILES string (stored as list[str] in PyG)
      - mol_type: surfactant type index
    """

    def __init__(
        self,
        root: str,
        csv_path: str = None,
        split: str = "train",
        desc_mean: np.ndarray = None,
        desc_std: np.ndarray = None,
        target_mean: np.ndarray = None,
        target_std: np.ndarray = None,
        temp_mean: float = None,
        temp_std: float = None,
        transform=None,
        pre_transform=None,
    ):
        """
        Args:
            root: Root directory for processed data cache.
            csv_path: Path to the CSV file. If None, inferred from split.
            split: 'train' or 'test'.
            desc_mean: Descriptor mean (12,), required for test split.
            desc_std: Descriptor std (12,), required for test split.
            target_mean: Target mean (6,), required for test split.
            target_std: Target std (6,), required for test split.
            temp_mean: Temperature mean for Z-score normalization.
            temp_std: Temperature std for Z-score normalization.
            transform: PyG transform.
            pre_transform: PyG pre_transform.
        """
        self.split = split
        self.csv_path = Path(csv_path) if csv_path else None
        self.desc_mean = desc_mean
        self.desc_std = desc_std
        self.target_mean = target_mean
        self.target_std = target_std
        self.temp_mean = temp_mean
        self.temp_std = temp_std

        # For train split, we compute desc_mean/std and target_mean/std inside process()
        # We'll store them as attributes after processing
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

        # After loading (from cache or fresh process), ensure scaler attributes are set
        # If process() was not called (loaded from cache), load scaler from saved file
        if self.desc_mean is None or self.target_mean is None or not hasattr(self, 'temp_mean') or self.temp_mean is None:
            scaler_path = Path(self.processed_dir) / f"scaler_{self.split}.pt"
            if scaler_path.exists():
                scaler = torch.load(scaler_path, weights_only=False)
                if self.desc_mean is None and "desc_mean" in scaler:
                    self.desc_mean = scaler["desc_mean"].numpy()
                if self.desc_std is None and "desc_std" in scaler:
                    self.desc_std = scaler["desc_std"].numpy()
                if self.target_mean is None and "target_mean" in scaler:
                    self.target_mean = scaler["target_mean"].numpy()
                if self.target_std is None and "target_std" in scaler:
                    self.target_std = scaler["target_std"].numpy()
                if "temp_mean" in scaler:
                    self.temp_mean = float(scaler["temp_mean"])
                if "temp_std" in scaler:
                    self.temp_std = float(scaler["temp_std"])
                if "type_map" in scaler:
                    self.type_map = scaler["type_map"]

    @property
    def raw_file_names(self):
        if self.csv_path is not None and self.csv_path.exists():
            # Return the CSV name; raw_dir is root/raw by default
            return [self.csv_path.name]
        return []

    @property
    def processed_file_names(self):
        return [f"data_{self.split}.pt"]

    def download(self):
        # Data is provided locally; copy to raw_dir if needed
        if self.csv_path is not None and self.csv_path.exists():
            import shutil
            dest = Path(self.raw_dir) / self.csv_path.name
            if not dest.exists():
                shutil.copy(self.csv_path, dest)

    def process(self):
        # Read CSV
        if self.csv_path is not None:
            df = pd.read_csv(self.csv_path)
        else:
            raw_csv = Path(self.raw_dir) / self.raw_file_names[0]
            df = pd.read_csv(raw_csv)

        # ---- Fit descriptor scaler on train split ----
        smiles_list = df["SMILES"].tolist()
        if self.split == "train" and (self.desc_mean is None or self.desc_std is None):
            self.desc_mean, self.desc_std = fit_descriptor_scaler(smiles_list)
        # Ensure we have float32 arrays
        self.desc_mean = np.asarray(self.desc_mean, dtype=np.float32)
        self.desc_std = np.asarray(self.desc_std, dtype=np.float32)

        # ---- Fit temperature scaler on train split (Z-score from data) ----
        # Use actual training data statistics for proper signal scaling.
        if self.split == "train":
            temp_vals = df["temp"].dropna().values if "temp" in df.columns else np.array([25.0])
            if len(temp_vals) < 2:
                self.temp_mean = 25.0
                self.temp_std = 35.0
            else:
                self.temp_mean = float(np.mean(temp_vals))
                # Use sample std (ddof=1)
                self.temp_std = float(np.std(temp_vals, ddof=1).clip(min=1e-8))
        else:
            if not hasattr(self, 'temp_mean') or self.temp_mean is None:
                self.temp_mean = 25.0
                self.temp_std = 1.0
            else:
                self.temp_mean = float(self.temp_mean)
                self.temp_std = float(self.temp_std)

        # ---- Fit target scaler on train split (Z-score normalization) ----
        # This is essential for multi-task learning with different magnitude targets.
        # Uses sample standard deviation (ddof=1).
        if self.split == "train":
            target_vals = []
            for col in CSV_PROP_COLS:
                vals = df[col].dropna().values
                target_vals.append(vals)
            self.target_mean = np.array(
                [np.mean(v) for v in target_vals], dtype=np.float32
            )
            self.target_std = np.array(
                [np.std(v, ddof=1).clip(min=1e-8) for v in target_vals], dtype=np.float32
            )
        else:
            # For test split, target_mean/std should be passed in
            if not hasattr(self, 'target_mean') or self.target_mean is None:
                self.target_mean = np.zeros(6, dtype=np.float32)
                self.target_std = np.ones(6, dtype=np.float32)
            else:
                self.target_mean = np.asarray(self.target_mean, dtype=np.float32)
                self.target_std = np.asarray(self.target_std, dtype=np.float32)

        # Build surfactant type mapping
        if "type" in df.columns:
            types = sorted(df["type"].unique().tolist())
            self.type_map = {t: i for i, t in enumerate(types)}
        else:
            self.type_map = {}

        data_list = []
        for idx, row in df.iterrows():
            smi = row["SMILES"]
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue  # skip invalid SMILES

            # Graph features
            x, edge_index, edge_attr = smiles_to_graph(smi)

            # Temperature - Z-score normalized
            temp_raw_val = row.get("temp", np.nan)
            if pd.isna(temp_raw_val):
                temp_raw = 0.0
                temp_norm = 0.0
                temp_mask = 0
            else:
                temp_raw = float(temp_raw_val)
                temp_norm = (temp_raw - self.temp_mean) / self.temp_std
                temp_mask = 1

            # Descriptors (raw + normalized)
            # compute_descriptors returns (values, valid_mask)
            desc_raw_arr, desc_valid_arr = compute_descriptors(mol)
            desc_raw_arr = desc_raw_arr.astype(np.float32)
            desc_valid_arr = desc_valid_arr.astype(np.float32)

            # Morgan fingerprint (ECFP4, 2048-bit)
            fp_arr = compute_morgan_fp(mol, radius=2, n_bits=2048)

            # Fill failed descriptors with training set mean (so normalized value = 0)
            # This is more reasonable than filling with 0, which could be far from the distribution.
            desc_filled = desc_raw_arr.copy()
            desc_filled[desc_valid_arr == 0] = self.desc_mean[desc_valid_arr == 0]

            # Z-score normalize
            desc_norm = (desc_filled - self.desc_mean) / self.desc_std

            # Targets and mask (6 tasks) - store as [1, 6] so PyG batch stacks to [B, 6]
            y_raw = np.zeros((1, 6), dtype=np.float32)
            mask = np.zeros((1, 6), dtype=np.float32)
            for task_idx, col in enumerate(CSV_PROP_COLS):
                val = row.get(col, np.nan)
                if pd.notna(val):
                    y_raw[0, task_idx] = float(val)
                    mask[0, task_idx] = 1.0
                else:
                    y_raw[0, task_idx] = 0.0
                    mask[0, task_idx] = 0.0

            # Normalize targets (Z-score) for stable multi-task training
            y_norm = (y_raw - self.target_mean) / self.target_std
            y = y_norm * mask  # zero out missing entries

            # Surfactant type
            mol_type = 0
            if "type" in row and pd.notna(row["type"]) and row["type"] in self.type_map:
                mol_type = self.type_map[row["type"]]

            # Fold (if available)
            fold = int(row["fold"]) if "fold" in df.columns and pd.notna(row.get("fold", np.nan)) else -1

            data = Data(
                x=torch.from_numpy(x),
                edge_index=torch.from_numpy(edge_index),
                edge_attr=torch.from_numpy(edge_attr),
                temp_norm=torch.tensor([temp_norm], dtype=torch.float32),
                temp_raw=torch.tensor([temp_raw], dtype=torch.float32),
                temp_mask=torch.tensor([temp_mask], dtype=torch.float32),
                descriptors=torch.from_numpy(desc_norm.astype(np.float32)),
                desc_raw=torch.from_numpy(desc_raw_arr),
                desc_valid=torch.from_numpy(desc_valid_arr),
                fingerprint=torch.from_numpy(fp_arr),
                y=torch.from_numpy(y),  # normalized targets
                y_raw=torch.from_numpy(y_raw),  # original targets
                mask=torch.from_numpy(mask),
                smiles=smi,
                mol_type=torch.tensor([mol_type], dtype=torch.long),
                fold=torch.tensor([fold], dtype=torch.long),
            )

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        # Save
        self.save(data_list, self.processed_paths[0])

        # Also save scaler info for convenience
        scaler_path = Path(self.processed_dir) / f"scaler_{self.split}.pt"
        torch.save(
            {
                "desc_mean": torch.from_numpy(self.desc_mean),
                "desc_std": torch.from_numpy(self.desc_std),
                "target_mean": torch.from_numpy(self.target_mean),
                "target_std": torch.from_numpy(self.target_std),
                "temp_mean": self.temp_mean,
                "temp_std": self.temp_std,
                "type_map": self.type_map,
            },
            scaler_path,
        )
