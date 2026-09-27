"""
data_loader.py
--------------
Dataset loading, column discovery, validation, and stratified splitting
for OralAbsPredict.
"""

from typing import Tuple, Optional, Dict, Any, List
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from src.preprocessing import clean_dataset

# Common column candidates in public ADMET datasets
SMILES_COL_CANDIDATES = [
    "SMILES", "smiles", "Smiles", "canonical_smiles",
    "structure", "mol", "compound_smiles", "SMILE"
]

TARGET_COL_CANDIDATES = [
    "Y", "target", "label", "HIA", "HOB", "Class", "activity",
    "Activity", "bioavailability", "absorption", "value"
]


def detect_columns(
    df: pd.DataFrame,
    preferred_smiles_col: Optional[str] = None,
    preferred_target_col: Optional[str] = None
) -> Tuple[str, Optional[str]]:
    """
    Intelligently identifies the SMILES column and Target column from DataFrame columns.

    Parameters
    ----------
    df : pd.DataFrame
        Loaded dataset.
    preferred_smiles_col : Optional[str]
        Explicitly preferred SMILES column name.
    preferred_target_col : Optional[str]
        Explicitly preferred Target column name.

    Returns
    -------
    Tuple[str, Optional[str]]
        (smiles_col_name, target_col_name)

    Raises
    ------
    ValueError
        If no valid SMILES column can be identified.
    """
    cols = list(df.columns)

    # 1. SMILES column resolution
    smiles_col = None
    if preferred_smiles_col and preferred_smiles_col in cols:
        smiles_col = preferred_smiles_col
    else:
        for candidate in SMILES_COL_CANDIDATES:
            if candidate in cols:
                smiles_col = candidate
                break
        if smiles_col is None:
            # Case-insensitive search
            cols_lower = {c.lower(): c for c in cols}
            for candidate in SMILES_COL_CANDIDATES:
                if candidate.lower() in cols_lower:
                    smiles_col = cols_lower[candidate.lower()]
                    break

    if smiles_col is None:
        raise ValueError(
            f"Could not identify a SMILES column in dataset. Available columns: {cols}. "
            f"Please rename the column or specify preferred_smiles_col."
        )

    # 2. Target column resolution
    target_col = None
    if preferred_target_col and preferred_target_col in cols:
        target_col = preferred_target_col
    else:
        for candidate in TARGET_COL_CANDIDATES:
            if candidate in cols:
                target_col = candidate
                break
        if target_col is None:
            cols_lower = {c.lower(): c for c in cols}
            for candidate in TARGET_COL_CANDIDATES:
                if candidate.lower() in cols_lower:
                    target_col = cols_lower[candidate.lower()]
                    break

    return smiles_col, target_col


def load_dataset(
    file_path: str,
    preferred_smiles_col: Optional[str] = None,
    preferred_target_col: Optional[str] = None,
    clean: bool = True
) -> Tuple[pd.DataFrame, str, Optional[str], Dict[str, Any]]:
    """
    Loads a dataset from CSV, detects columns, validates chemical structures,
    and returns the cleaned DataFrame with metadata.

    Parameters
    ----------
    file_path : str
        Path to CSV file.
    preferred_smiles_col : Optional[str]
        Explicit SMILES column name.
    preferred_target_col : Optional[str]
        Explicit Target column name.
    clean : bool, default True
        Whether to clean SMILES and drop duplicates.

    Returns
    -------
    Tuple[pd.DataFrame, str, Optional[str], Dict[str, Any]]
        (DataFrame, smiles_col, target_col, summary_info)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Dataset not found at '{file_path}'. "
            f"Please verify that the dataset exists or place it into data/raw/."
        )

    df = pd.read_csv(file_path)
    smiles_col, target_col = detect_columns(df, preferred_smiles_col, preferred_target_col)

    cleaning_stats = {}
    if clean:
        df, cleaning_stats = clean_dataset(df, smiles_col=smiles_col, target_col=target_col)

    metadata: Dict[str, Any] = {
        "file_path": file_path,
        "smiles_column": smiles_col,
        "target_column": target_col,
        "num_samples": len(df),
        "cleaning_stats": cleaning_stats,
    }

    if target_col and target_col in df.columns:
        value_counts = df[target_col].value_counts().to_dict()
        metadata["class_distribution"] = value_counts
        metadata["unique_classes"] = list(value_counts.keys())

    return df, smiles_col, target_col, metadata


def split_data(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    val_size: Optional[float] = None,
    random_state: int = 42
) -> Dict[str, pd.DataFrame]:
    """
    Splits dataset into stratified train, (optional) validation, and test subsets.

    Parameters
    ----------
    df : pd.DataFrame
        Input cleaned dataset.
    target_col : str
        Target column name for stratification.
    test_size : float, default 0.2
        Proportion for test set.
    val_size : Optional[float], default None
        Proportion for validation set (e.g. 0.15). If None, splits into train/test only.
    random_state : int, default 42
        Reproducibility seed.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary with keys 'train', 'test' and optionally 'val'.
    """
    stratify = df[target_col] if target_col in df.columns else None

    if val_size is not None and val_size > 0:
        # First split into train_val and test
        train_val, test = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify
        )
        # Then split train_val into train and val
        val_relative = val_size / (1.0 - test_size)
        stratify_val = train_val[target_col] if target_col in train_val.columns else None
        train, val = train_test_split(
            train_val,
            test_size=val_relative,
            random_state=random_state,
            stratify=stratify_val
        )
        return {
            "train": train.reset_index(drop=True),
            "val": val.reset_index(drop=True),
            "test": test.reset_index(drop=True),
        }
    else:
        train, test = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify
        )
        return {
            "train": train.reset_index(drop=True),
            "test": test.reset_index(drop=True),
        }
