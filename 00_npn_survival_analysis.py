#!/usr/bin/env python3
"""
Survival-ready prep for USA-NPN "Status and Intensity" data.

Functionality
-------------
- Reads a USA-NPN Status & Intensity CSV.
- Builds interval-censored tables (L, R] for time-to-first "Open flowers" and "Ripe fruits".
- One row per (individual_id, year).
- Exports a single Excel workbook (.xlsx) with multiple sheets to a subfolder.

Usage
------
    python 00_npn_survival_analysis.py

Assumptions
-----------
- Input CSV path: ./data/status_intensity_observation_data.csv (relative to this script)
- Output directory: ./00_npn_survival_analysis/ (relative to this script)

Dependencies
------------
    pip install pandas openpyxl
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# ---------- PATHS (relative to this script) ----------

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPT_STEM = Path(__file__).stem  # "00_npn_survival_analysis"

# Input CSV under ./data/
INPUT_CSV = SCRIPT_DIR / "data" / "status_intensity_observation_data.csv"

# Output directory: ./00_npn_survival_analysis/
OUTPUT_DIR = SCRIPT_DIR / SCRIPT_STEM

# Optional: filter to a single species by scientific name (case-insensitive). Set to None to skip.
FILTER_SPECIES: str | None = None  # e.g., "Vitis vinifera" or "Malus domestica"

# Output Excel workbook name
OUTPUT_XLSX = "survival_intervals_ready.xlsx"


# -------------------- Helper functions --------------------

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


def parse_status(x) -> float | bool | np.nan:
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

    yes_rows = g[g["phenophase_status_bool"] is True] if False else g[g["phenophase_status_bool"] == True]
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


# -------------------- Main --------------------

def main() -> str:
    # Prepare output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / OUTPUT_XLSX

    print(f"Reading CSV file: {INPUT_CSV}")
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} rows from CSV")

    # Normalize columns and types
    df = normalize_columns(df)

    # Basic required columns check
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

    # Parse dates and year/DOY
    print("Parsing dates...")
    df["observation_date"] = pd.to_datetime(df["observation_date"], errors="coerce")
    if df["observation_date"].isna().all():
        raise ValueError("All observation_date values failed to parse.")

    df["year"] = df["observation_date"].dt.year.astype(int)
    df["doy"] = df["observation_date"].dt.dayofyear.astype(int)

    # Normalize status to booleans
    print("Normalizing phenophase status...")
    df["phenophase_status_bool"] = df["phenophase_status"].map(parse_status)

    # Optional species filter
    if FILTER_SPECIES:
        print(f"Species filter: {FILTER_SPECIES}")
        df = df[df["species"].str.contains(FILTER_SPECIES, case=False, na=False)].copy()
        print(f"After species filter: {len(df)} rows")

    # Keep only rows with a defined boolean status
    before_filter = len(df)
    df = df[df["phenophase_status_bool"].isin([True, False])].copy()
    print(f"After removing undefined status: {len(df)} rows (removed {before_filter - len(df)})")

    # Show available phenophases (first 10)
    available_phenophases = sorted(df["phenophase_name"].dropna().unique())
    print(f"Available phenophases (sample): {available_phenophases[:10]}")

    # -------------------- Build survival-ready tables --------------------
    PHENO_OPEN = "Open flowers"
    PHENO_RIPE = "Ripe fruits"

    print(f"Building intervals for '{PHENO_OPEN}'...")
    try:
        open_intervals = build_intervals(df, PHENO_OPEN)
        print(f"Created {len(open_intervals)} intervals for open flowers")
    except ValueError as e:
        print(f"Warning: {e}")
        open_intervals = pd.DataFrame()

    print(f"Building intervals for '{PHENO_RIPE}'...")
    try:
        ripe_intervals = build_intervals(df, PHENO_RIPE)
        print(f"Created {len(ripe_intervals)} intervals for ripe fruits")
    except ValueError as e:
        print(f"Warning: {e}")
        ripe_intervals = pd.DataFrame()

    # Quick summary
    summary_data = []
    if not open_intervals.empty:
        summary_data.append(
            [PHENO_OPEN, len(open_intervals), int(open_intervals["event"].sum()), float(open_intervals["event"].mean())]
        )
    if not ripe_intervals.empty:
        summary_data.append(
            [PHENO_RIPE, len(ripe_intervals), int(ripe_intervals["event"].sum()), float(ripe_intervals["event"].mean())]
        )
    summary = pd.DataFrame(summary_data, columns=["phenophase", "n_rows", "n_events", "event_rate"])
    if not summary.empty:
        summary["event_rate"] = summary["event_rate"].round(3)

    # README sheet text (as a one-column table)
    readme_lines = [
        "USA-NPN Status & Intensity → survival-ready intervals (.xlsx)",
        "",
        "Generated on: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        f"Input file: {INPUT_CSV}",
        f"Species filter: {FILTER_SPECIES or 'None (all species)'}",
        "",
        "One row per (individual_id, year).",
        "Columns:",
        "  - L, R: interval bounds in Day-of-Year where event time T ∈ (L, R].",
        "    - If no event observed: event=0 and R is blank (right-censored at L).",
        "  - event: 1 if first 'Yes' observed for the phenophase in that year, else 0.",
        "  - n_visits: number of observations for that individual/year/phenophase.",
        "  - first_obs_doy / last_obs_doy: first and last observed day-of-year.",
        "",
        f"Phenophases included: '{PHENO_OPEN}', '{PHENO_RIPE}'.",
        "Compatible with interval-censored models:",
        "  - R: Surv(L, R, type='interval2')",
        "  - Python (lifelines): IntervalRegressionFitter(lower=L, upper=R)",
    ]
    readme_df = pd.DataFrame({"README": readme_lines})

    # -------------------- Write Excel --------------------
    print(f"Writing Excel file to: {output_path}")
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        if not open_intervals.empty:
            open_intervals.to_excel(writer, index=False, sheet_name="open_flowers")
        if not ripe_intervals.empty:
            ripe_intervals.to_excel(writer, index=False, sheet_name="ripe_fruits")
        summary.to_excel(writer, index=False, sheet_name="summary")
        readme_df.to_excel(writer, index=False, sheet_name="README")

    print("Completed.")
    return str(output_path)


if __name__ == "__main__":
    out = main()
    print(f"Results saved to: {out}")
