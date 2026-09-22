"""RDKit descriptor computation and scaling for SurfMT-GNN.

12 expert descriptors used in the paper:
  MolLogP, TPSA, MolWt, NumRotatableBonds,
  NumHAcceptors, NumHDonors, NumAromaticRings,
  FractionCSP3, NumHeteroatoms, LabuteASA,
  BalabanJ, BertzCT
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors


DESCRIPTOR_NAMES = [
    "MolLogP",
    "TPSA",
    "MolWt",
    "NumRotatableBonds",
    "NumHAcceptors",
    "NumHDonors",
    "NumAromaticRings",
    "FractionCSP3",
    "NumHeteroatoms",
    "LabuteASA",
    "BalabanJ",
    "BertzCT",
]


def compute_descriptors(mol: Chem.Mol) -> tuple:
    """Compute 12 RDKit descriptors for a molecule.

    Args:
        mol: RDKit Mol object.

    Returns:
        Tuple of (desc_array, valid_mask):
          - desc_array: np.ndarray of shape (12,), dtype float64.
            Failed descriptors are filled with 0.0 (use valid_mask to identify).
          - valid_mask: np.ndarray of shape (12,), dtype bool.
            True = computed successfully, False = failed / NaN.
    """
    vals = []
    valid = []
    # MolLogP
    try:
        v = Descriptors.MolLogP(mol)
        vals.append(v)
        valid.append(not np.isnan(v))
    except Exception:
        vals.append(0.0)
        valid.append(False)
    # TPSA
    try:
        v = Descriptors.TPSA(mol)
        vals.append(v)
        valid.append(not np.isnan(v))
    except Exception:
        vals.append(0.0)
        valid.append(False)
    # MolWt
    try:
        v = Descriptors.MolWt(mol)
        vals.append(v)
        valid.append(not np.isnan(v))
    except Exception:
        vals.append(0.0)
        valid.append(False)
    # NumRotatableBonds
    try:
        v = Descriptors.NumRotatableBonds(mol)
        vals.append(v)
        valid.append(not np.isnan(v))
    except Exception:
        vals.append(0.0)
        valid.append(False)
    # NumHAcceptors
    try:
        v = Lipinski.NumHAcceptors(mol)
        vals.append(v)
        valid.append(not np.isnan(v))
    except Exception:
        vals.append(0.0)
        valid.append(False)
    # NumHDonors
    try:
        v = Lipinski.NumHDonors(mol)
        vals.append(v)
        valid.append(not np.isnan(v))
    except Exception:
        vals.append(0.0)
        valid.append(False)
    # NumAromaticRings
    try:
        v = rdMolDescriptors.CalcNumAromaticRings(mol)
        vals.append(v)
        valid.append(not np.isnan(v))
    except Exception:
        vals.append(0.0)
        valid.append(False)
    # FractionCSP3
    try:
        v = rdMolDescriptors.CalcFractionCSP3(mol)
        vals.append(v)
        valid.append(not np.isnan(v))
    except Exception:
        vals.append(0.0)
        valid.append(False)
    # NumHeteroatoms
    try:
        v = Lipinski.NumHeteroatoms(mol)
        vals.append(v)
        valid.append(not np.isnan(v))
    except Exception:
        vals.append(0.0)
        valid.append(False)
    # LabuteASA
    try:
        v = rdMolDescriptors.CalcLabuteASA(mol)
        vals.append(v)
        valid.append(not np.isnan(v))
    except Exception:
        vals.append(0.0)
        valid.append(False)
    # BalabanJ
    try:
        v = Descriptors.BalabanJ(mol)
        vals.append(v)
        valid.append(not np.isnan(v))
    except Exception:
        vals.append(0.0)
        valid.append(False)
    # BertzCT
    try:
        v = Descriptors.BertzCT(mol)
        vals.append(v)
        valid.append(not np.isnan(v))
    except Exception:
        vals.append(0.0)
        valid.append(False)

    arr = np.array(vals, dtype=np.float64)
    mask = np.array(valid, dtype=bool)
    # Replace any NaN with 0 and mark as invalid
    nan_mask = np.isnan(arr)
    if nan_mask.any():
        arr = np.where(nan_mask, 0.0, arr)
        mask = mask & ~nan_mask
    return arr, mask


def fit_descriptor_scaler(smiles_list: list) -> tuple:
    """Fit Z-score scaler on a list of SMILES.

    Only successfully computed descriptor values are used for mean/std.

    Args:
        smiles_list: List of SMILES strings.

    Returns:
        Tuple of (mean, std), each np.ndarray of shape (12,).
        std is clamped to minimum of 1e-8 to avoid division by zero.
        Uses sample standard deviation (ddof=1).
    """
    all_desc = []
    all_valid = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        desc, valid = compute_descriptors(mol)
        all_desc.append(desc)
        all_valid.append(valid)
    all_desc = np.stack(all_desc, axis=0)  # (N, 12)
    all_valid = np.stack(all_valid, axis=0)  # (N, 12)

    mean = np.zeros(12, dtype=np.float64)
    std = np.zeros(12, dtype=np.float64)
    for i in range(12):
        vals = all_desc[all_valid[:, i], i]
        if len(vals) < 2:
            mean[i] = 0.0
            std[i] = 1.0
        else:
            mean[i] = np.mean(vals)
            std[i] = np.std(vals, ddof=1)
    std = np.clip(std, a_min=1e-8, a_max=None)
    return mean.astype(np.float32), std.astype(np.float32)