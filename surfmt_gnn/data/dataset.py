"""SurfPro dataset for PyTorch Geometric."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data import InMemoryDataset, Data

from .featurizer import smiles_to_graph
from .descriptors import compute_descriptors, fit_descriptor_scaler


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
      - temp_norm: normalized temperature (T - 25) / 35
      - temp_mask: 1 if temperature is present, 0 if missing
      - descriptors: 12-dim Z-score normalized RDKit descriptors
      - y: 6-dim target vector (NaN entries are set to 0, mask handles them)
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
            transform: PyG transform.
            pre_transform: PyG pre_transform.
        """
        self.split = split
        self.csv_path = Path(csv_path) if csv_path else None
        self.desc_mean = desc_mean
        self.desc_std = desc_std

        # For train split, we compute desc_mean/std inside process()
        # We'll store them as attributes after processing
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

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

            # Temperature
            temp_raw = row.get("temp", np.nan)
            if pd.isna(temp_raw):
                temp_norm = 0.0
                temp_mask = 0
            else:
                temp_norm = (float(temp_raw) - 25.0) / 35.0
                temp_mask = 1

            # Descriptors (Z-score normalized)
            desc = compute_descriptors(mol).astype(np.float32)
            desc_norm = (desc - self.desc_mean) / self.desc_std

            # Targets and mask (6 tasks) - store as [1, 6] so PyG batch stacks to [B, 6]
            y = np.zeros((1, 6), dtype=np.float32)
            mask = np.zeros((1, 6), dtype=np.float32)
            for task_idx, col in enumerate(CSV_PROP_COLS):
                val = row.get(col, np.nan)
                if pd.notna(val):
                    y[0, task_idx] = float(val)
                    mask[0, task_idx] = 1.0
                else:
                    y[0, task_idx] = 0.0
                    mask[0, task_idx] = 0.0

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
                temp_mask=torch.tensor([temp_mask], dtype=torch.float32),
                descriptors=torch.from_numpy(desc_norm),
                y=torch.from_numpy(y),
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
                "type_map": self.type_map,
            },
            scaler_path,
        )
