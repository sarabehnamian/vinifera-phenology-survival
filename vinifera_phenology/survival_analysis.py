"""
Survival analysis functions for phenology data.

Converts USA-NPN observation data into interval-censored survival format.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from .utils import normalize_columns, parse_status


def first_yes_interval(group: pd.DataFrame) -> pd.Series:
    """
    For a group = (individual_id, year, phenophase), sorted by date,
    compute interval (L, R] where:
      - R = DOY of the first YES (present) observation
      - L = DOY of the last NO (absent) observation before R
    If the first observation is YES, set L = max(0, first_observed_DOY - 1).
    If no YES occurs in the year, right-censor: event=0, L=last observed DOY, R=NaN.
    """
    g = group.sort_values("observation_date")

    yes_rows = g[g["phenophase_status_bool"] == True]
    if not yes_rows.empty:
        first_yes = yes_rows.iloc[0]
        R = int(first_yes["doy"])

        before_yes = g[g["observation_date"] < first_yes["observation_date"]]
        no_before = before_yes[before_yes["phenophase_status_bool"] == False]
        if not no_before.empty:
            L = int(no_before["doy"].iloc[-1])
        else:
            first_obs = int(g["doy"].iloc[0])
            L = max(0, first_obs - 1)

        event = 1
    else:
        R = np.nan
        L = int(g["doy"].iloc[-1])
        event = 0

    return pd.Series(
        {
            "L": int(L),
            "R": (int(R) if pd.notna(R) else np.nan),
            "event": int(event),
            "n_visits": int(len(g)),
            "first_obs_doy": int(g["doy"].iloc[0]),
            "last_obs_doy": int(g["doy"].iloc[-1]),
        }
    )


def build_intervals(df: pd.DataFrame, phenophase_name_filter: str) -> pd.DataFrame:
    """
    Build an interval-censored table for a given phenophase_name (case-insensitive exact match).
    Returns one row per (individual_id, year).
    
    Parameters
    ----------
    df : pd.DataFrame
        Observation data with columns: observation_date, phenophase_status, phenophase_name,
        individual_id, site_id, year, doy
    phenophase_name_filter : str
        Name of phenophase to filter (e.g., "Open flowers", "Ripe fruits")
    
    Returns
    -------
    pd.DataFrame
        Interval-censored data with columns: individual_id, site_id, year, L, R, event,
        n_visits, first_obs_doy, last_obs_doy
    """
    mask = df["phenophase_name"].str.lower() == phenophase_name_filter.lower()
    sub = df[mask].copy()
    if sub.empty:
        available_names = sorted(df["phenophase_name"].dropna().unique())
        raise ValueError(
            f"No rows found for phenophase_name == '{phenophase_name_filter}'. "
            f"Available names (first 10): {available_names[:10]}"
        )

    grouped = sub.groupby(["individual_id", "site_id", "year"], sort=False, as_index=False)
    out = grouped.apply(first_yes_interval, include_groups=False).reset_index()

    cols = ["individual_id", "site_id", "year", "L", "R", "event", "n_visits", "first_obs_doy", "last_obs_doy"]
    return out[cols].sort_values(["site_id", "individual_id", "year"]).reset_index(drop=True)


def prepare_observation_data(
    csv_path: str | Path,
    filter_species: str | None = None
) -> pd.DataFrame:
    """
    Load and prepare observation data from CSV.
    
    Parameters
    ----------
    csv_path : str | Path
        Path to USA-NPN observation CSV file
    filter_species : str | None
        Optional species filter (case-insensitive)
    
    Returns
    -------
    pd.DataFrame
        Prepared observation data
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df = normalize_columns(df)

    # Required columns check
    required_cols = {
        "observation_date",
        "phenophase_status",
        "phenophase_name",
        "individual_id",
        "site_id",
        "species",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Parse dates
    df["observation_date"] = pd.to_datetime(df["observation_date"], errors="coerce")
    if df["observation_date"].isna().all():
        raise ValueError("All observation_date values failed to parse.")

    df["year"] = df["observation_date"].dt.year.astype(int)
    df["doy"] = df["observation_date"].dt.dayofyear.astype(int)

    # Normalize status
    df["phenophase_status_bool"] = df["phenophase_status"].map(parse_status)

    # Optional species filter
    if filter_species:
        df = df[df["species"].str.contains(filter_species, case=False, na=False)].copy()

    # Keep only rows with defined boolean status
    df = df[df["phenophase_status_bool"].isin([True, False])].copy()

    return df


def process_phenology_data(
    csv_path: str | Path,
    phenophases: list[str] | None = None,
    filter_species: str | None = None,
    output_path: str | Path | None = None
) -> dict[str, pd.DataFrame]:
    """
    Process phenology observation data into interval-censored format.
    
    Parameters
    ----------
    csv_path : str | Path
        Path to observation CSV file
    phenophases : list[str] | None
        List of phenophase names to process. Default: ["Open flowers", "Ripe fruits"]
    filter_species : str | None
        Optional species filter
    output_path : str | Path | None
        Optional path to save Excel output
    
    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary mapping phenophase names to interval DataFrames
    """
    if phenophases is None:
        phenophases = ["Open flowers", "Ripe fruits"]
    
    # Load and prepare data
    df = prepare_observation_data(csv_path, filter_species=filter_species)
    
    # Build intervals for each phenophase
    results = {}
    for pheno in phenophases:
        try:
            intervals = build_intervals(df, pheno)
            results[pheno] = intervals
        except ValueError as e:
            print(f"Warning: {e}")
            results[pheno] = pd.DataFrame()
    
    # Save to Excel if requested
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            for pheno, intervals_df in results.items():
                if not intervals_df.empty:
                    sheet_name = pheno.lower().replace(" ", "_")
                    intervals_df.to_excel(writer, index=False, sheet_name=sheet_name)
    
    return results

