"""
preprocessing.py
----------------
Molecular validation, SMILES canonicalization, duplicate handling, and
dataset cleaning for OralAbsPredict.
"""

from typing import Tuple, Optional, Dict, Any, List
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import SaltRemover


def validate_smiles(smiles: str, remove_salts: bool = False) -> Tuple[bool, Optional[str], Optional[Chem.Mol]]:
    """
    Validates a SMILES string using RDKit and returns canonical SMILES and Mol object.

    Parameters
    ----------
    smiles : str
        Input SMILES string.
    remove_salts : bool, default False
        Whether to strip common salts and counterions.

    Returns
    -------
    Tuple[bool, Optional[str], Optional[Chem.Mol]]
        (is_valid, canonical_smiles, mol_object)
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return False, None, None

    clean_smiles = smiles.strip()
    try:
        mol = Chem.MolFromSmiles(clean_smiles)
        if mol is None:
            return False, None, None

        if remove_salts:
            remover = SaltRemover.SaltRemover()
            mol = remover.StripMol(mol, dontRemoveEverything=True)

        canonical_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        return True, canonical_smiles, mol
    except Exception:
        return False, None, None


def canonicalize_smiles(smiles: str) -> Optional[str]:
    """
    Converts a SMILES string to its canonical isomeric representation.

    Parameters
    ----------
    smiles : str
        Input SMILES string.

    Returns
    -------
    Optional[str]
        Canonical SMILES or None if parsing fails.
    """
    is_valid, can_smiles, _ = validate_smiles(smiles)
    return can_smiles if is_valid else None


def clean_dataset(
    df: pd.DataFrame,
    smiles_col: str = "SMILES",
    target_col: Optional[str] = "Y",
    remove_conflicting_duplicates: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Cleans a cheminformatics dataset by:
    1. Validating and canonicalizing SMILES structures with RDKit
    2. Dropping invalid/unparseable SMILES
    3. Dropping rows with missing target values
    4. Identifying and handling duplicate molecules (canonical SMILES)
    5. Resolving conflicting duplicate targets

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset.
    smiles_col : str, default "SMILES"
        Name of the column containing SMILES strings.
    target_col : Optional[str], default "Y"
        Name of the target column (if available).
    remove_conflicting_duplicates : bool, default True
        If True, drops duplicates where the target labels conflict.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any]]
        Cleaned DataFrame and a summary dictionary of cleaning actions taken.
    """
    stats: Dict[str, Any] = {
        "initial_rows": len(df),
        "missing_smiles_dropped": 0,
        "invalid_smiles_dropped": 0,
        "missing_target_dropped": 0,
        "duplicates_dropped": 0,
        "conflicting_duplicates_dropped": 0,
        "final_rows": 0,
    }

    cleaned = df.copy()

    # Drop missing SMILES
    missing_smiles = cleaned[smiles_col].isna() | (cleaned[smiles_col].astype(str).str.strip() == "")
    stats["missing_smiles_dropped"] = int(missing_smiles.sum())
    cleaned = cleaned[~missing_smiles].reset_index(drop=True)

    # Validate SMILES and generate canonical representations
    valid_flags: List[bool] = []
    canonical_list: List[Optional[str]] = []

    for smi in cleaned[smiles_col]:
        valid, can_smi, _ = validate_smiles(str(smi))
        valid_flags.append(valid)
        canonical_list.append(can_smi)

    cleaned["is_valid_smiles"] = valid_flags
    cleaned["canonical_smiles"] = canonical_list

    invalid_count = int((~cleaned["is_valid_smiles"]).sum())
    stats["invalid_smiles_dropped"] = invalid_count
    cleaned = cleaned[cleaned["is_valid_smiles"]].reset_index(drop=True)

    # Drop invalid flag column, use canonical_smiles
    cleaned = cleaned.drop(columns=["is_valid_smiles"])
    cleaned[smiles_col] = cleaned["canonical_smiles"]
    cleaned = cleaned.drop(columns=["canonical_smiles"])

    # Drop missing targets if target column specified
    if target_col and target_col in cleaned.columns:
        missing_target = cleaned[target_col].isna()
        stats["missing_target_dropped"] = int(missing_target.sum())
        cleaned = cleaned[~missing_target].reset_index(drop=True)

        # Ensure target is integer for classification
        try:
            if set(cleaned[target_col].dropna().unique()).issubset({0, 1, 0.0, 1.0}):
                cleaned[target_col] = cleaned[target_col].astype(int)
        except Exception:
            pass

        # Handle duplicates
        duplicate_mask = cleaned.duplicated(subset=[smiles_col], keep=False)
        if duplicate_mask.any():
            if remove_conflicting_duplicates:
                # Group by smiles and check target variance
                grouped = cleaned.groupby(smiles_col)[target_col].nunique()
                conflicting_smiles = grouped[grouped > 1].index
                conflict_mask = cleaned[smiles_col].isin(conflicting_smiles)
                stats["conflicting_duplicates_dropped"] = int(conflict_mask.sum())
                cleaned = cleaned[~conflict_mask].reset_index(drop=True)

            # Drop remaining identical duplicates
            before_dedup = len(cleaned)
            cleaned = cleaned.drop_duplicates(subset=[smiles_col]).reset_index(drop=True)
            stats["duplicates_dropped"] = before_dedup - len(cleaned)
    else:
        # No target column, just deduplicate SMILES
        before_dedup = len(cleaned)
        cleaned = cleaned.drop_duplicates(subset=[smiles_col]).reset_index(drop=True)
        stats["duplicates_dropped"] = before_dedup - len(cleaned)

    stats["final_rows"] = len(cleaned)
    return cleaned, stats
