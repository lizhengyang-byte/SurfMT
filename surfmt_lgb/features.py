"""Feature extraction for LightGBM version.

Tabular features (tree-model friendly, no normalization needed):
  - ECFP4 Morgan fingerprint (2048-bit, radius 2)
  - ECFP6 Morgan fingerprint (2048-bit, radius 3)
  - MACCS structural keys (167-bit)
  - Expanded RDKit 2D descriptors (65)
  - Temperature (scalar)
Total per-sample dimension: 2048 + 2048 + 167 + 65 + 1 = 4329.

Uses the modern rdFingerprintGenerator API to avoid RDKit deprecation
warnings (the legacy GetMorganFingerprintAsBitVect emits them).
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem import rdFingerprintGenerator, MACCSkeys

# ---- Feature block dimensions ----
ECFP4_BITS = 2048
ECFP6_BITS = 2048
MACCS_BITS = 167
DESC_N = 65
TEMP_N = 1
FEATURE_DIM = ECFP4_BITS + ECFP6_BITS + MACCS_BITS + DESC_N + TEMP_N  # 4329

# Block slice offsets
_OFF_ECFP4 = 0
_OFF_ECFP6 = ECFP4_BITS                              # 2048
_OFF_MACCS = _OFF_ECFP6 + ECFP6_BITS                 # 4096
_OFF_DESC = _OFF_MACCS + MACCS_BITS                  # 4263
_OFF_TEMP = _OFF_DESC + DESC_N                       # 4328

_SLICES = {
    "ecfp4": (_OFF_ECFP4, _OFF_ECFP6),
    "ecfp6": (_OFF_ECFP6, _OFF_MACCS),
    "maccs": (_OFF_MACCS, _OFF_DESC),
    "desc": (_OFF_DESC, _OFF_TEMP),
    "temp": (_OFF_TEMP, FEATURE_DIM),
}


# Expanded 2D descriptor list (curated, all resolvable in RDKit)
DESCRIPTOR_NAMES = [
    # Core
    "MolLogP", "TPSA", "MolWt", "NumRotatableBonds",
    "NumHAcceptors", "NumHDonors", "NumAromaticRings",
    "FractionCSP3", "NumHeteroatoms", "LabuteASA",
    "BalabanJ", "BertzCT",
    # Size / topology
    "HeavyAtomCount", "RingCount", "NumSaturatedRings", "NumAliphaticRings",
    "NumAromaticCarbocycles", "NumAromaticHeterocycles",
    "NumSaturatedCarbocycles", "NumSaturatedHeterocycles",
    "NumAliphaticCarbocycles", "NumAliphaticHeterocycles",
    # Chi / kappa
    "Chi0", "Chi1", "Chi0n", "Chi0v", "Chi1n", "Chi1v",
    "Chi2n", "Chi2v", "Chi3n", "Chi3v", "Chi4n", "Chi4v",
    "HallKierAlpha", "Kappa1", "Kappa2", "Kappa3",
    # Charge / electronic
    "NumRadicalElectrons", "NumValenceElectrons",
    "MaxAbsPartialCharge", "MinAbsPartialCharge",
    "MaxPartialCharge", "MinPartialCharge",
    "MaxAbsEStateIndex", "MinAbsEStateIndex",
    "MaxEStateIndex", "MinEStateIndex",
    # VSA
    "SlogP_VSA1", "SlogP_VSA2", "SlogP_VSA3", "SlogP_VSA4",
    "SlogP_VSA5", "SlogP_VSA6", "SlogP_VSA7", "SlogP_VSA8",
    "SlogP_VSA9", "SlogP_VSA10", "SlogP_VSA11", "SlogP_VSA12",
    # Other
    "NHOHCount", "NOCount", "MolMR", "NumSpiroAtoms", "NumBridgeheadAtoms",
]


def _desc_functions():
    funcs = []
    for name in DESCRIPTOR_NAMES:
        fn = None
        if hasattr(Descriptors, name):
            fn = getattr(Descriptors, name)
        elif hasattr(rdMolDescriptors, name):
            fn = getattr(rdMolDescriptors, name)
        elif hasattr(Lipinski, name):
            fn = getattr(Lipinski, name)
        funcs.append((name, fn))
    return funcs


_DESC_FUNCS = _desc_functions()


def _fingerprint_bits(fn_gen, mol):
    """Return list of on-bit indices for a generated fingerprint."""
    fp = fn_gen.GetFingerprint(mol)
    return list(fp.GetOnBits())


def _maccs_bits(mol):
    """Return binary array of MACCS keys (167-bit)."""
    arr = np.zeros(MACCS_BITS, dtype=np.float32)
    maccs = MACCSkeys.GenMACCSKeys(mol)
    bits = list(maccs.ToBitString())
    n = min(len(bits), MACCS_BITS)
    for i in range(n):
        if bits[i] == '1':
            arr[i] = 1.0
    return arr


def _descriptor_vals(mol):
    """Return 65-dim descriptor array (failed -> 0)."""
    arr = np.zeros(DESC_N, dtype=np.float32)
    for i, (name, fn) in enumerate(_DESC_FUNCS):
        if fn is None:
            continue
        try:
            v = fn(mol)
            v = float(v)
            if np.isfinite(v):
                arr[i] = v
        except Exception:
            pass
    return arr


# Lazily build the fingerprint generators once
_ECFP4_GEN = None
_ECFP6_GEN = None


def _get_generators():
    global _ECFP4_GEN, _ECFP6_GEN
    if _ECFP4_GEN is None:
        _ECFP4_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=ECFP4_BITS)
        _ECFP6_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=ECFP6_BITS)
    return _ECFP4_GEN, _ECFP6_GEN


def compute_features(smiles_list, temps) -> np.ndarray:
    """Compute the full feature matrix for many samples.

    Args:
        smiles_list: list of SMILES strings.
        temps: array/list of temperature values (missing already imputed).

    Returns:
        float32 array of shape (N, FEATURE_DIM).
    """
    ecfp4_gen, ecfp6_gen = _get_generators()
    n = len(smiles_list)
    X = np.zeros((n, FEATURE_DIM), dtype=np.float32)

    for i in range(n):
        mol = Chem.MolFromSmiles(smiles_list[i])
        if mol is None:
            continue
        try:
            # ECFP4 / ECFP6
            for idx in _fingerprint_bits(ecfp4_gen, mol):
                X[i, _OFF_ECFP4 + idx] = 1.0
            for idx in _fingerprint_bits(ecfp6_gen, mol):
                X[i, _OFF_ECFP6 + idx] = 1.0
        except Exception:
            pass
        # MACCS
        try:
            X[i, _OFF_MACCS:_OFF_DESC] = _maccs_bits(mol)
        except Exception:
            pass
        # Descriptors
        X[i, _OFF_DESC:_OFF_TEMP] = _descriptor_vals(mol)
        # Temperature
        X[i, _OFF_TEMP] = temps[i]

    return X