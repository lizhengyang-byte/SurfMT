"""Molecular fingerprint computation for SurfMT-GNN.

Morgan fingerprints (ECFP) are circular fingerprints that capture
substructural features of molecules. They provide complementary
information to GNN-based graph representations.

Uses the modern rdFingerprintGenerator.MorganGenerator API (the
legacy GetMorganFingerprintAsBitVect / GetHashedMorganFingerprint
calls emit deprecation warnings in newer RDKit).
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator


def compute_morgan_fp(
    mol: Chem.Mol,
    radius: int = 2,
    n_bits: int = 2048,
    use_counts: bool = False,
) -> np.ndarray:
    """Compute Morgan fingerprint for a molecule.

    Args:
        mol: RDKit Mol object.
        radius: Fingerprint radius (2 = ECFP4).
        n_bits: Number of bits in the fingerprint.
        use_counts: If True, return count-based fingerprint;
            if False, return binary fingerprint.

    Returns:
        np.ndarray of shape (n_bits,), dtype float32.
        Returns zeros if computation fails.
    """
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)

    try:
        if use_counts:
            gener = rdFingerprintGenerator.GetMorganGenerator(
                radius=radius, fpSize=n_bits,
                countSimulation=True,
                countBounds=[(1, 2), (2, 3), (3, 5), (5, 9)],
            )
            fp = gener.GetCountFingerprint(mol)
            arr = np.zeros(n_bits, dtype=np.float32)
            for idx, count in fp.GetNonzeroElements().items():
                arr[idx] = float(count)
        else:
            gener = rdFingerprintGenerator.GetMorganGenerator(
                radius=radius, fpSize=n_bits
            )
            fp = gener.GetFingerprint(mol)
            arr = np.zeros(n_bits, dtype=np.float32)
            for idx in fp.GetOnBits():
                arr[idx] = 1.0
        return arr
    except Exception:
        return np.zeros(n_bits, dtype=np.float32)


def compute_morgan_fp_batch(
    smiles_list: list,
    radius: int = 2,
    n_bits: int = 2048,
    use_counts: bool = False,
) -> np.ndarray:
    """Compute Morgan fingerprints for a batch of molecules.

    Args:
        smiles_list: List of SMILES strings.
        radius: Fingerprint radius.
        n_bits: Number of bits.
        use_counts: Count-based or binary.

    Returns:
        np.ndarray of shape (N, n_bits).
    """
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        fps.append(compute_morgan_fp(mol, radius, n_bits, use_counts))
    return np.stack(fps, axis=0)