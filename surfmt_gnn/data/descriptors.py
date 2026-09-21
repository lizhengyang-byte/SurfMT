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


def compute_descriptors(mol: Chem.Mol) -> np.ndarray:
    """Compute 12 RDKit descriptors for a molecule.

    Args:
        mol: RDKit Mol object.

    Returns:
        np.ndarray of shape (12,), dtype float64.
        NaN values (e.g. BalabanJ failure) are replaced with 0.
    """
    vals = []
    # MolLogP
    try:
        vals.append(Descriptors.MolLogP(mol))
    except Exception:
        vals.append(0.0)
    # TPSA
    try:
        vals.append(Descriptors.TPSA(mol))
    except Exception:
        vals.append(0.0)
    # MolWt
    try:
        vals.append(Descriptors.MolWt(mol))
    except Exception:
        vals.append(0.0)
    # NumRotatableBonds
    try:
        vals.append(Descriptors.NumRotatableBonds(mol))
    except Exception:
        vals.append(0.0)
    # NumHAcceptors
    try:
        vals.append(Lipinski.NumHAcceptors(mol))
    except Exception:
        vals.append(0.0)
    # NumHDonors
    try:
        vals.append(Lipinski.NumHDonors(mol))
    except Exception:
        vals.append(0.0)
    # NumAromaticRings
    try:
        vals.append(rdMolDescriptors.CalcNumAromaticRings(mol))
    except Exception:
        vals.append(0.0)
    # FractionCSP3
    try:
        vals.append(rdMolDescriptors.CalcFractionCSP3(mol))
    except Exception:
        vals.append(0.0)
    # NumHeteroatoms
    try:
        vals.append(Lipinski.NumHeteroatoms(mol))
    except Exception:
        vals.append(0.0)
    # LabuteASA
    try:
        vals.append(rdMolDescriptors.CalcLabuteASA(mol))
    except Exception:
        vals.append(0.0)
    # BalabanJ
    try:
        vals.append(Descriptors.BalabanJ(mol))
    except Exception:
        vals.append(0.0)
    # BertzCT
    try:
        vals.append(Descriptors.BertzCT(mol))
    except Exception:
        vals.append(0.0)

    arr = np.array(vals, dtype=np.float64)
    # Replace any NaN with 0
    arr = np.where(np.isnan(arr), 0.0, arr)
    return arr


def fit_descriptor_scaler(smiles_list: list) -> tuple:
    """Fit Z-score scaler on a list of SMILES.

    Args:
        smiles_list: List of SMILES strings.

    Returns:
        Tuple of (mean, std), each np.ndarray of shape (12,).
        std is clamped to minimum of 1e-8 to avoid division by zero.
    """
    all_desc = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        all_desc.append(compute_descriptors(mol))
    all_desc = np.stack(all_desc, axis=0)  # (N, 12)
    mean = np.mean(all_desc, axis=0)
    std = np.std(all_desc, axis=0)
    std = np.clip(std, a_min=1e-8, a_max=None)
    return mean.astype(np.float32), std.astype(np.float32)
