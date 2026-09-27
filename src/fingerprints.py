"""
fingerprints.py
---------------
Generation of molecular fingerprints (Morgan/ECFP, MACCS keys, RDKit topological)
for machine learning representation in OralAbsPredict.
"""

from typing import Union, Dict, List, Optional, Tuple, Any
import numpy as np
from rdkit import Chem
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator
from rdkit.Chem import rdMolDescriptors
from src.preprocessing import validate_smiles


def calculate_morgan_fingerprint(
    mol_or_smiles: Union[str, Chem.Mol],
    radius: int = 2,
    n_bits: int = 512,
    use_features: bool = False
) -> np.ndarray:
    """
    Computes Morgan circular fingerprint (ECFP/FCFP equivalent) as a deterministic NumPy array.

    Parameters
    ----------
    mol_or_smiles : Union[str, Chem.Mol]
        SMILES string or RDKit Mol object.
    radius : int, default 2
        Fingerprint radius (radius=2 corresponds to ECFP4 diameter 4).
    n_bits : int, default 512
        Length of bit vector.
    use_features : bool, default False
        If True, computes FCFP (pharmacophoric features) instead of ECFP.

    Returns
    -------
    np.ndarray
        1D binary NumPy array of shape (n_bits,).
    """
    if isinstance(mol_or_smiles, str):
        is_valid, _, mol = validate_smiles(mol_or_smiles)
        if not is_valid or mol is None:
            raise ValueError(f"Invalid SMILES string: {mol_or_smiles}")
    else:
        mol = mol_or_smiles

    if mol is None:
        raise ValueError("Cannot calculate fingerprints for None molecule")

    try:
        gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius,
            fpSize=n_bits,
            atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen() if use_features else None
        )
        fp_bits = gen.GetFingerprint(mol)
        arr = np.zeros((n_bits,), dtype=np.int8)
        for bit in fp_bits.GetOnBits():
            arr[bit] = 1
        return arr
    except Exception:
        # Fallback to rdMolDescriptors if generator unavailable
        bit_vect = rdMolDescriptors.GetMorganFingerprintAsBitVect(
            mol,
            radius=radius,
            nBits=n_bits,
            useFeatures=use_features
        )
        arr = np.zeros((n_bits,), dtype=np.int8)
        for on_bit in bit_vect.GetOnBits():
            arr[on_bit] = 1
        return arr


def calculate_maccs_keys(mol_or_smiles: Union[str, Chem.Mol]) -> np.ndarray:
    """
    Computes 166-bit MACCS structural keys.

    Parameters
    ----------
    mol_or_smiles : Union[str, Chem.Mol]
        SMILES string or RDKit Mol object.

    Returns
    -------
    np.ndarray
        1D binary NumPy array of shape (166,).
    """
    if isinstance(mol_or_smiles, str):
        is_valid, _, mol = validate_smiles(mol_or_smiles)
        if not is_valid or mol is None:
            raise ValueError(f"Invalid SMILES string: {mol_or_smiles}")
    else:
        mol = mol_or_smiles

    if mol is None:
        raise ValueError("Cannot calculate MACCS keys for None molecule")

    maccs_vect = MACCSkeys.GenMACCSKeys(mol)
    arr = np.zeros((166,), dtype=np.int8)
    for bit in maccs_vect.GetOnBits():
        if bit < 166:
            arr[bit] = 1
    return arr


def get_morgan_bit_info(
    mol_or_smiles: Union[str, Chem.Mol],
    radius: int = 2,
    n_bits: int = 512
) -> Tuple[np.ndarray, Dict[int, List[Tuple[int, int]]]]:
    """
    Computes Morgan fingerprint and extracts atom environment bit information
    mapping each 'on' bit to (atom_index, radius).
    Useful for local SHAP substructure interpretation.

    Parameters
    ----------
    mol_or_smiles : Union[str, Chem.Mol]
        SMILES string or RDKit Mol.
    radius : int
        Radius of circular environment.
    n_bits : int
        Number of bits.

    Returns
    -------
    Tuple[np.ndarray, Dict[int, List[Tuple[int, int]]]]
        (fingerprint_array, bit_info_dict)
    """
    if isinstance(mol_or_smiles, str):
        is_valid, _, mol = validate_smiles(mol_or_smiles)
        if not is_valid or mol is None:
            raise ValueError(f"Invalid SMILES: {mol_or_smiles}")
    else:
        mol = mol_or_smiles

    bit_info: Dict[int, List[Tuple[int, int]]] = {}
    bit_vect = rdMolDescriptors.GetMorganFingerprintAsBitVect(
        mol,
        radius=radius,
        nBits=n_bits,
        bitInfo=bit_info
    )
    arr = np.zeros((n_bits,), dtype=np.int8)
    for on_bit in bit_vect.GetOnBits():
        arr[on_bit] = 1
    return arr, bit_info
