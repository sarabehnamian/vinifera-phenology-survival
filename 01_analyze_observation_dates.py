# 01_analyze_observation_dates.py
# Creates date/time summaries for USA-NPN obs, saving ONLY in survival_analysis_results/
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# --- console encoding (Windows) ---
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# --- paths ---
IN_OBS   = Path("data/status_intensity_observation_data.csv")
IN_SITES = Path("data/ancillary_site_data.csv")
OUT_DIR  = Path("01_analyze_observation_dates")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_XLSX = OUT_DIR / "date_time_summary.xlsx"

def norm(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (df.columns.str.strip().str.lower()
                  .str.replace(" ", "_").str.replace("-", "_"))
    return df

def pick(df: pd.DataFrame, cols):
    """Return only existing columns (in order)."""
    return df[[c for c in cols if c in df.columns]]

print("Loading observation and site data...")
obs = norm(pd.read_csv(IN_OBS))
sites = norm(pd.read_csv(IN_SITES))

# --- required obs fields ---
if "observation_date" not in obs.columns:
    raise ValueError("Column 'observation_date' not found in observation data.")
if "site_id" not in obs.columns:
    raise ValueError("Column 'site_id' not found in observation data.")

# --- parse dates & derive fields ---
obs["observation_date"] = pd.to_datetime(obs["observation_date"], errors="coerce")
obs = obs.dropna(subset=["observation_date", "site_id"]).copy()
obs["year"] = obs["observation_date"].dt.year.astype(int)
obs["month"] = obs["observation_date"].dt.month.astype(int)
obs["day"] = obs["observation_date"].dt.day.astype(int)
obs["day_of_year"] = obs["observation_date"].dt.dayofyear.astype(int)

print(f"Total observations: {len(obs):,}")
print(f"Total sites in obs: {obs['site_id'].nunique():,}")
print(f"Total sites in ancillary: {len(sites):,}")

# --- site coords/meta (only columns that exist) ---
site_coord_cols = ["site_id", "latitude", "longitude", "state", "site_name"]
site_coords = pick(sites, site_coord_cols).drop_duplicates("site_id")

# --- site-year summary ---
print("Creating site-year summary...")
g = obs.groupby(["site_id", "year"], as_index=False)
site_year_summary = g.agg(
    first_obs_date=("observation_date", "min"),
    last_obs_date =("observation_date", "max"),
    total_observations=("observation_date", "count"),
    first_doy=("day_of_year", "min"),
    last_doy =("day_of_year", "max"),
    unique_individuals=("individual_id", "nunique") if "individual_id" in obs.columns else ("year", "size"),
)
# Merge site coords
site_year_summary = site_year_summary.merge(site_coords, on="site_id", how="left")

# --- overall site summary ---
print("Creating overall site summary...")
agg_dict = {
    "observation_date": ["min", "max", "count"],
    "year": ["min", "max", "nunique"],
}
if "individual_id" in obs.columns:
    agg_dict["individual_id"] = ["nunique"]
species_col = "species_id" if "species_id" in obs.columns else ("species" if "species" in obs.columns else None)
if species_col:
    agg_dict[species_col] = ["nunique"]

site_summary = obs.groupby("site_id").agg(agg_dict).round(2)
# flatten columns
site_summary.columns = ["_".join([c for c in col if c]).strip("_") for col in site_summary.columns.to_flat_index()]
# rename to readable
rename_map = {
    "observation_date_min": "first_ever_date",
    "observation_date_max": "last_ever_date",
    "observation_date_count": "total_all_obs",
    "year_min": "first_year",
    "year_max": "last_year",
    "year_nunique": "years_active",
    "individual_id_nunique": "total_individuals",
    "species_id_nunique": "species_count",
    "species_nunique": "species_count",
}
site_summary = site_summary.rename(columns=rename_map).reset_index()
site_summary = site_summary.merge(site_coords, on="site_id", how="left")

# --- weather needs per site (min/max obs dates + year span) ---
print("Creating weather data requirements...")
w = site_year_summary.groupby("site_id", as_index=False).agg(
    weather_start_year=("year", "min"),
    weather_end_year=("year", "max"),
    weather_start_date=("first_obs_date", "min"),
    weather_end_date=("last_obs_date", "max"),
)
w = w.merge(site_coords, on="site_id", how="left")
w["date_range_days"] = (w["weather_end_date"] - w["weather_start_date"]).dt.days
w["year_range"] = (w["weather_end_year"] - w["weather_start_year"] + 1)

# --- monthly observation patterns ---
print("Creating monthly observation patterns...")
monthly = obs.groupby(["site_id", "month"], as_index=False).agg(
    obs_count=("observation_date", "count"),
    individual_count=("individual_id", "nunique") if "individual_id" in obs.columns else ("year", "size"),
)
# pivot months 1..12
pivot = monthly.pivot(index="site_id", columns="month", values="obs_count").reindex(columns=range(1,13)).fillna(0)
pivot.columns = [f"month_{m}_obs" for m in range(1,13)]
monthly_pivot = pivot.reset_index()

# --- write Excel ---
print(f"Saving summaries to {OUT_XLSX} ...")
with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as xw:
    site_year_summary.to_excel(xw, sheet_name="Site_Year_Summary", index=False)
    site_summary.to_excel(xw, sheet_name="Site_Summary", index=False)
    w.to_excel(xw, sheet_name="Weather_Requirements", index=False)
    monthly_pivot.to_excel(xw, sheet_name="Monthly_Patterns", index=False)

    # Weather coordinates sheet for fetchers
    wx_cols = ["site_id", "latitude", "longitude", "weather_start_date", "weather_end_date",
               "weather_start_year", "weather_end_year"]
    weather_coords = pick(w, wx_cols)
    weather_coords.to_excel(xw, sheet_name="Weather_Coordinates", index=False)

    readme = pd.DataFrame({
        "Sheet_Name": ["Site_Year_Summary","Site_Summary","Weather_Requirements","Monthly_Patterns","Weather_Coordinates"],
        "Description": [
            "Detailed observation coverage per site-year",
            "Overall site coverage and activity",
            "Per-site observed date span and year span for weather fetch",
            "Monthly observation counts per site (columns month_1..month_12)",
            "Per-site coords + date ranges for weather fetching"
        ],
        "Usage": [
            "Assess seasonal coverage & effort",
            "Report site-level coverage",
            "Drive external weather downloads (Daymet / POWER)",
            "Visualize seasonal patterns",
            "Direct input for weather scripts; join on site_id"
        ]
    })
    readme.to_excel(xw, sheet_name="README", index=False)

print("Created comprehensive date/time summary.")
print(f"Site-Year rows: {len(site_year_summary):,}")
print(f"Unique sites: {site_summary['site_id'].nunique():,}")
print(f"Date range: {obs['observation_date'].min()} → {obs['observation_date'].max()}")
print(f"Years covered: {int(obs['year'].min())} → {int(obs['year'].max())}")

# Brief display
print("\n--- Weather Data Requirements (head) ---")
show_cols = [c for c in ["site_id", "state", "weather_start_year", "weather_end_year", "date_range_days"] if c in w.columns]
print(w[show_cols].head(12).to_string(index=False))
