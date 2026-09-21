"""Molecular graph featurizer for SurfMT-GNN.

Atom features (39-dim):
  - Atomic number one-hot: 12 dims (C, N, O, F, Na, P, S, Cl, K, Br, I, other)
  - Degree one-hot: 6 dims (0, 1, 2, 3, 4, 5+)
  - Formal charge one-hot: 5 dims (-2, -1, 0, +1, +2)
  - Hybridization one-hot: 5 dims (S, SP, SP2, SP3, SP3D)
  - Aromatic: 1 dim
  - Number of Hs one-hot: 5 dims (0, 1, 2, 3, 4+)
  - Chirality one-hot: 3 dims (none, CW, CCW)
  - In ring: 1 dim
  - Radical electrons: 1 dim
  Total = 12 + 6 + 5 + 5 + 1 + 5 + 3 + 1 + 1 = 39

Bond features (10-dim):
  - Bond type one-hot: 4 dims (SINGLE, DOUBLE, TRIPLE, AROMATIC)
  - Conjugated: 1 dim
  - In ring: 1 dim
  - Stereochemistry one-hot: 4 dims (STEREONONE, ANY, Z, E)
  Total = 4 + 1 + 1 + 4 = 10
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdchem


# ---------- Atom feature constants ----------

ATOMIC_NUMBERS = [6, 7, 8, 9, 11, 15, 16, 17, 19, 35, 53]  # 11 common + other = 12
DEGREES = [0, 1, 2, 3, 4, 5]  # 6 levels (5 means 5+)
FORMAL_CHARGES = [-2, -1, 0, 1, 2]  # 5 levels
HYBRIDIZATIONS = [
    rdchem.HybridizationType.S,
    rdchem.HybridizationType.SP,
    rdchem.HybridizationType.SP2,
    rdchem.HybridizationType.SP3,
    rdchem.HybridizationType.SP3D,
]  # 5 types


def _one_hot(value, allowable_set, include_other: bool = False) -> list:
    """Return one-hot list.

    If include_other is True, the last dimension represents "other" and
    has length len(allowable_set) + 1. Otherwise length is len(allowable_set)
    and out-of-range values fall into the last bucket.
    """
    if include_other:
        length = len(allowable_set) + 1
    else:
        length = len(allowable_set)
    encoding = [0] * length
    try:
        idx = allowable_set.index(value)
    except ValueError:
        idx = length - 1  # fallback to last (other / max bucket)
    encoding[idx] = 1
    return encoding


def atom_to_feature_vector(atom: Chem.Atom) -> np.ndarray:
    """Convert an RDKit Atom to a 39-dim feature vector.

    Args:
        atom: RDKit Atom object.

    Returns:
        np.ndarray of shape (39,), dtype float32.
    """
    features = []

    # 1. Atomic number one-hot (12 dims: 11 common + other)
    features += _one_hot(atom.GetAtomicNum(), ATOMIC_NUMBERS, include_other=True)

    # 2. Degree one-hot (6 dims)
    degree = min(atom.GetDegree(), DEGREES[-1])
    features += _one_hot(degree, DEGREES)

    # 3. Formal charge one-hot (5 dims)
    fc = atom.GetFormalCharge()
    if fc < FORMAL_CHARGES[0]:
        fc = FORMAL_CHARGES[0]
    elif fc > FORMAL_CHARGES[-1]:
        fc = FORMAL_CHARGES[-1]
    features += _one_hot(fc, FORMAL_CHARGES)

    # 4. Hybridization one-hot (5 dims)
    hyb = atom.GetHybridization()
    if hyb not in HYBRIDIZATIONS:
        hyb = HYBRIDIZATIONS[-1]  # fallback to last
    features += _one_hot(hyb, HYBRIDIZATIONS)

    # 5. Aromatic (1 dim)
    features += [1 if atom.GetIsAromatic() else 0]

    # 6. Number of Hs one-hot (5 dims)
    num_h = min(atom.GetTotalNumHs(), 4)  # 4 means 4+
    features += _one_hot(num_h, [0, 1, 2, 3, 4])

    # 7. Chirality (3 dims): unspec, CW, CCW
    chi = atom.GetChiralTag()
    if chi == rdchem.ChiralType.CHI_UNSPECIFIED:
        features += [1, 0, 0]
    elif chi == rdchem.ChiralType.CHI_TETRAHEDRAL_CW:
        features += [0, 1, 0]
    elif chi == rdchem.ChiralType.CHI_TETRAHEDRAL_CCW:
        features += [0, 0, 1]
    else:
        features += [1, 0, 0]  # default to unspecified

    # 8. In ring (1 dim)
    features += [1 if atom.IsInRing() else 0]

    # 9. Radical electrons (1 dim)
    features += [1 if atom.GetNumRadicalElectrons() > 0 else 0]

    # Total: 12 + 6 + 5 + 5 + 1 + 5 + 3 + 1 + 1 = 39
    return np.array(features, dtype=np.float32)


# ---------- Bond feature constants ----------

BOND_TYPES = [
    rdchem.BondType.SINGLE,
    rdchem.BondType.DOUBLE,
    rdchem.BondType.TRIPLE,
    rdchem.BondType.AROMATIC,
]  # 4 types

STEREO_TYPES = [
    rdchem.BondStereo.STEREONONE,
    rdchem.BondStereo.STEREOANY,
    rdchem.BondStereo.STEREOZ,
    rdchem.BondStereo.STEREOE,
]  # 4 types


def bond_to_feature_vector(bond: Chem.Bond) -> np.ndarray:
    """Convert an RDKit Bond to a 10-dim feature vector.

    Args:
        bond: RDKit Bond object.

    Returns:
        np.ndarray of shape (10,), dtype float32.
    """
    features = []

    # 1. Bond type one-hot (4 dims)
    bt = bond.GetBondType()
    features += _one_hot(bt, BOND_TYPES)

    # 2. Conjugated (1 dim)
    features += [1 if bond.GetIsConjugated() else 0]

    # 3. In ring (1 dim)
    features += [1 if bond.IsInRing() else 0]

    # 4. Stereochemistry one-hot (4 dims)
    stereo = bond.GetStereo()
    if stereo not in STEREO_TYPES:
        stereo = STEREO_TYPES[0]  # default to STEREONONE
    features += _one_hot(stereo, STEREO_TYPES)

    # Total: 4 + 1 + 1 + 4 = 10
    return np.array(features, dtype=np.float32)


def smiles_to_graph(smiles: str):
    """Convert a SMILES string to graph arrays.

    Args:
        smiles: SMILES string.

    Returns:
        Tuple of (x, edge_index, edge_attr):
            x: np.ndarray (num_atoms, 39)
            edge_index: np.ndarray (2, num_edges * 2), int64
            edge_attr: np.ndarray (num_edges * 2, 10)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    # Atom features
    num_atoms = mol.GetNumAtoms()
    x = np.zeros((num_atoms, 39), dtype=np.float32)
    for i, atom in enumerate(mol.GetAtoms()):
        x[i] = atom_to_feature_vector(atom)

    # Edge index and edge attributes (undirected: add both directions)
    edges_src = []
    edges_dst = []
    edge_feats = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        feat = bond_to_feature_vector(bond)
        # i -> j
        edges_src.append(i)
        edges_dst.append(j)
        edge_feats.append(feat)
        # j -> i
        edges_src.append(j)
        edges_dst.append(i)
        edge_feats.append(feat)

    edge_index = np.array([edges_src, edges_dst], dtype=np.int64)
    edge_attr = np.stack(edge_feats, axis=0) if edge_feats else np.zeros((0, 10), dtype=np.float32)

    return x, edge_index, edge_attr
