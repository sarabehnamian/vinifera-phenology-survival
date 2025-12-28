"""
Utility functions for data processing and normalization.
"""

import pandas as pd
import numpy as np


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case and snake_case column names."""
    out = df.copy()
    out.columns = (
        out.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )
    return out


from typing import Union

def parse_status(x) -> Union[float, bool]:
    """
    Normalize phenophase_status to booleans (True for 'yes/present', False for 'no/absent').
    Accepts numeric encodings (1, 0, -1) and common string variants.
    """
    if pd.isna(x):
        return np.nan

    # Numeric-like
    try:
        xi = int(float(x))
        if xi == 1:
            return True
        if xi == 0:
            return False
        if xi == -1:
            return np.nan
    except Exception:
        pass

    # String-like
    s = str(x).strip().lower()
    if s in {"yes", "y", "true", "present"}:
        return True
    if s in {"no", "n", "false", "absent"}:
        return False
    if s in {"-1", "uncertain"}:
        return np.nan

    return np.nan


def zscore(series: pd.Series) -> pd.Series:
    """Standardize a series to z-scores."""
    mu = series.mean()
    sd = series.std()
    if sd == 0 or pd.isna(sd):
        return pd.Series(0.0, index=series.index)
    return (series - mu) / sd

