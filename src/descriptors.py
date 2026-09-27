"""
descriptors.py
--------------
Calculation of interpretable 2D molecular descriptors, physicochemical properties,
and drug-likeness rules using RDKit and Mordred for OralAbsPredict.
"""

from typing import Dict, Any, Optional, List, Union, Tuple
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Crippen, MolSurf, rdMolDescriptors
from src.preprocessing import validate_smiles

# Core physicochemical descriptors with human-readable names and descriptions
CORE_DESCRIPTORS_META = {
    "MolWt": {"name": "Molecular Weight", "unit": "g/mol", "description": "Sum of atomic weights (Rule of 5 < 500)"},
    "LogP": {"name": "Wildman-Crippen LogP", "unit": "", "description": "Octanol-water partition coefficient (Rule of 5 < 5)"},
    "TPSA": {"name": "Topological Polar Surface Area", "unit": "Å²", "description": "Polar surface area (Veber rule <= 140 Å²)"},
    "NumHDonors": {"name": "H-Bond Donors", "unit": "", "description": "Sum of OH and NH groups (Rule of 5 <= 5)"},
    "NumHAcceptors": {"name": "H-Bond Acceptors", "unit": "", "description": "Sum of O and N atoms (Rule of 5 <= 10)"},
    "NumRotatableBonds": {"name": "Rotatable Bonds", "unit": "", "description": "Single non-ring bonds (Veber rule <= 10)"},
    "NumRings": {"name": "Total Rings", "unit": "", "description": "Total number of rings"},
    "NumAromaticRings": {"name": "Aromatic Rings", "unit": "", "description": "Number of aromatic ring systems"},
    "NumAliphaticRings": {"name": "Aliphatic Rings", "unit": "", "description": "Number of aliphatic ring systems"},
    "FractionCSP3": {"name": "Fraction Csp3", "unit": "", "description": "Fraction of sp3 hybridized carbons to total carbons"},
    "HeavyAtomCount": {"name": "Heavy Atom Count", "unit": "", "description": "Number of non-hydrogen atoms"},
    "NumHeteroatoms": {"name": "Heteroatom Count", "unit": "", "description": "Number of non-C and non-H atoms"},
    "FormalCharge": {"name": "Formal Charge", "unit": "e", "description": "Net formal charge on molecule"},
    "MolMR": {"name": "Molar Refractivity", "unit": "cm³/mol", "description": "Measure of polarizability and molecular volume"},
    "LabuteASA": {"name": "Labute ASA", "unit": "Å²", "description": "Approximate surface area calculation"},
    "BertzCT": {"name": "Bertz Complexity", "unit": "", "description": "Topological molecular complexity index"},
    "Ro5_Violations": {"name": "Lipinski Rule Violations", "unit": "", "description": "Violations of Lipinski Rule of 5 (<= 1 allowed)"},
    "Veber_Compliant": {"name": "Veber Rule Compliant", "unit": "", "description": "1 if TPSA <= 140 Å² and RotBonds <= 10, else 0"}
}


def calculate_rdkit_descriptors(mol_or_smiles: Union[str, Chem.Mol]) -> Dict[str, float]:
    """
    Computes curated RDKit physicochemical descriptors and ADME-relevant drug-likeness metrics.

    Parameters
    ----------
    mol_or_smiles : Union[str, Chem.Mol]
        Input SMILES string or RDKit Mol object.

    Returns
    -------
    Dict[str, float]
        Dictionary of descriptor values.
    """
    if isinstance(mol_or_smiles, str):
        is_valid, _, mol = validate_smiles(mol_or_smiles)
        if not is_valid or mol is None:
            raise ValueError(f"Invalid SMILES string: {mol_or_smiles}")
    else:
        mol = mol_or_smiles

    if mol is None:
        raise ValueError("Cannot calculate descriptors for None molecule")

    # Basic physicochemical descriptors
    mw = float(Descriptors.MolWt(mol))
    logp = float(Crippen.MolLogP(mol))
    tpsa = float(rdMolDescriptors.CalcTPSA(mol))
    hbd = int(Lipinski.NumHDonors(mol))
    hba = int(Lipinski.NumHAcceptors(mol))
    rot_bonds = int(Lipinski.NumRotatableBonds(mol))
    num_rings = int(rdMolDescriptors.CalcNumRings(mol))
    num_aromatic_rings = int(rdMolDescriptors.CalcNumAromaticRings(mol))
    num_aliphatic_rings = int(rdMolDescriptors.CalcNumAliphaticRings(mol))
    fraction_csp3 = float(rdMolDescriptors.CalcFractionCSP3(mol))
    heavy_atoms = int(mol.GetNumHeavyAtoms())
    num_heteroatoms = int(Lipinski.NumHeteroatoms(mol))
    formal_charge = int(Chem.GetFormalCharge(mol))
    mr = float(Crippen.MolMR(mol))
    labute_asa = float(rdMolDescriptors.CalcLabuteASA(mol))
    bertz_ct = float(Descriptors.BertzCT(mol))

    # Lipinski Rule of 5 Violations
    ro5_violations = 0
    if mw > 500:
        ro5_violations += 1
    if logp > 5.0:
        ro5_violations += 1
    if hbd > 5:
        ro5_violations += 1
    if hba > 10:
        ro5_violations += 1

    # Veber Rule compliance: TPSA <= 140 and rot_bonds <= 10
    veber_compliant = 1.0 if (tpsa <= 140.0 and rot_bonds <= 10) else 0.0

    return {
        "MolWt": mw,
        "LogP": logp,
        "TPSA": tpsa,
        "NumHDonors": float(hbd),
        "NumHAcceptors": float(hba),
        "NumRotatableBonds": float(rot_bonds),
        "NumRings": float(num_rings),
        "NumAromaticRings": float(num_aromatic_rings),
        "NumAliphaticRings": float(num_aliphatic_rings),
        "FractionCSP3": fraction_csp3,
        "HeavyAtomCount": float(heavy_atoms),
        "NumHeteroatoms": float(num_heteroatoms),
        "FormalCharge": float(formal_charge),
        "MolMR": mr,
        "LabuteASA": labute_asa,
        "BertzCT": bertz_ct,
        "Ro5_Violations": float(ro5_violations),
        "Veber_Compliant": veber_compliant,
    }


def calculate_mordred_subset(mol_or_smiles: Union[str, Chem.Mol]) -> Dict[str, float]:
    """
    Computes a curated set of 2D Mordred descriptors with robust fallback
    against non-finite values or calculation errors.
    """
    if isinstance(mol_or_smiles, str):
        is_valid, _, mol = validate_smiles(mol_or_smiles)
        if not is_valid or mol is None:
            return {}
    else:
        mol = mol_or_smiles

    mordred_features: Dict[str, float] = {}
    try:
        from mordred import Calculator, descriptors as mordred_descriptors
        # Use a selected, fast 2D subset of Mordred descriptors
        calc = Calculator([
            mordred_descriptors.AtomCount,
            mordred_descriptors.RingCount,
            mordred_descriptors.CarbonTypes,
            mordred_descriptors.HydrogenBond,
            mordred_descriptors.Polarizability,
            mordred_descriptors.TopologicalIndex,
            mordred_descriptors.Aromatic
        ], ignore_3D=True)
        res = calc(mol)
        for desc, val in res.items():
            name = f"Mordred_{str(desc)}"
            try:
                val_float = float(val)
                if np.isfinite(val_float):
                    mordred_features[name] = val_float
            except Exception:
                continue
    except Exception:
        # Graceful fallback if Mordred fails
        pass

    return mordred_features


def extract_all_features(
    mol_or_smiles: Union[str, Chem.Mol],
    n_bits: int = 512
) -> Tuple[Dict[str, float], List[str]]:
    """
    Extracts all molecular features: RDKit physicochemical descriptors,
    Mordred 2D subset, and Morgan circular fingerprint bits.

    Parameters
    ----------
    mol_or_smiles : Union[str, Chem.Mol]
        SMILES string or RDKit Mol.
    n_bits : int, default 512
        Number of Morgan fingerprint bits.

    Returns
    -------
    Tuple[Dict[str, float], List[str]]
        (feature_dict, sorted_feature_names)
    """
    if isinstance(mol_or_smiles, str):
        is_valid, _, mol = validate_smiles(mol_or_smiles)
        if not is_valid or mol is None:
            raise ValueError(f"Invalid SMILES string: {mol_or_smiles}")
    else:
        mol = mol_or_smiles

    if mol is None:
        raise ValueError("Cannot extract features for None molecule")

    features: Dict[str, float] = {}

    # 1. RDKit Physicochemical Descriptors
    rdkit_desc = calculate_rdkit_descriptors(mol)
    features.update(rdkit_desc)

    # 2. Mordred 2D Descriptors
    mordred_desc = calculate_mordred_subset(mol)
    features.update(mordred_desc)

    # 3. Morgan Fingerprint Bits
    from src.fingerprints import calculate_morgan_fingerprint
    fp = calculate_morgan_fingerprint(mol, radius=2, n_bits=n_bits)
    for bit_idx, bit_val in enumerate(fp):
        features[f"Morgan_Bit_{bit_idx}"] = float(bit_val)

    feature_names = list(features.keys())
    return features, feature_names


def generate_features(
    smiles: str,
    feature_names: Optional[List[str]] = None,
    n_bits: int = 512
) -> np.ndarray:
    """
    Generates the exact numerical feature vector expected by trained models.
    Guarantees deterministic feature ordering based on `feature_names`.

    Parameters
    ----------
    smiles : str
        SMILES string of target molecule.
    feature_names : Optional[List[str]]
        Exact list of expected feature names in order. If None, uses all extracted features.
    n_bits : int, default 512
        Length of Morgan fingerprint bits if extracting fresh.

    Returns
    -------
    np.ndarray
        1D NumPy array of shape (num_features,) with np.float32 values.

    Raises
    ------
    ValueError
        If SMILES is invalid or cannot be parsed.
    """
    features_dict, extracted_names = extract_all_features(smiles, n_bits=n_bits)

    target_names = feature_names if feature_names is not None else extracted_names
    vector = np.zeros(len(target_names), dtype=np.float32)

    for i, name in enumerate(target_names):
        vector[i] = features_dict.get(name, 0.0)

    return vector


def get_descriptor_table_df(descriptors: Dict[str, float]) -> pd.DataFrame:
    """
    Converts a descriptor dictionary into a formatted presentation DataFrame
    for UI rendering.
    """
    rows = []
    for k, meta in CORE_DESCRIPTORS_META.items():
        if k in descriptors:
            val = descriptors[k]
            if isinstance(val, (int, float)) and float(val).is_integer() and k not in ["LogP"]:
                formatted_val = f"{int(val)}"
            elif isinstance(val, (int, float)):
                formatted_val = f"{float(val):.2f}"
            else:
                formatted_val = str(val)

            rows.append({
                "Property": meta["name"],
                "Value": formatted_val,
                "Unit": meta["unit"] if meta["unit"] else "—",
                "Description": meta["description"]
            })
    return pd.DataFrame(rows)

