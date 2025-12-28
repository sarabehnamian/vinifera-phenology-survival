# 03_merge_survival_with_weather.py
# Merge survival intervals with real daily weather (POWER) as per-year covariates.
# Inputs:
#   survival_analysis_results/survival_intervals_ready.xlsx (sheets: open_flowers, ripe_fruits)
#   survival_analysis_results/site_daily_weather.xlsx (sheet: daily_weather)
# Output:
#   survival_analysis_results/survival_with_weather.xlsx (sheets: open_flowers, ripe_fruits, README)

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# --- console encoding (Windows) ---
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

IN_SURV  = Path("00_npn_survival_analysis/survival_intervals_ready.xlsx")
IN_WX    = Path("02_fetch_nasa_power_weather/site_daily_weather.xlsx")
OUT_XLSX = Path("03_merge_survival_with_weather/survival_with_weather.xlsx")
OUT_XLSX.parent.mkdir(parents=True, exist_ok=True)

def norm(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (df.columns.str.strip().str.lower()
                  .str.replace(" ", "_").str.replace("-", "_"))
    return df

print("Loading survival intervals...")
xl = pd.ExcelFile(IN_SURV)
sheets = [s for s in xl.sheet_names if s.lower() in ("open_flowers","ripe_fruits")]
if not sheets:
    raise SystemExit("No sheets named 'open_flowers' or 'ripe_fruits' in survival_intervals_ready.xlsx")

print("Loading daily weather...")
wx = pd.read_excel(IN_WX, sheet_name="daily_weather")
wx = norm(wx)
for col in ["site_id","date","tmin_c","tmax_c","prcp_mm","gdd_base10"]:
    if col not in wx.columns:
        raise SystemExit(f"Weather file missing column: {col}")

# Ensure consistent types / fields
wx["site_id"] = wx["site_id"].astype(str)
wx["date"] = pd.to_datetime(wx["date"])
wx["year"] = wx["date"].dt.year.astype(int)
wx["doy"]  = wx["date"].dt.dayofyear.astype(int)

# Precompute per-site-year cumulative metrics for efficient lookups
wx_sorted = wx.sort_values(["site_id","year","doy"]).copy()
wx_sorted["day_count"] = 1
wx_sorted["frost_day"] = (wx_sorted["tmin_c"] <= 0).astype(int)
wx_sorted["heat_day"]  = (wx_sorted["tmax_c"] >= 35).astype(int)

group_cols = ["site_id","year"]
for newc, src in {
    "gdd_cum": "gdd_base10",
    "prcp_cum": "prcp_mm",
    "tmin_sum_cum": "tmin_c",
    "tmax_sum_cum": "tmax_c",
    "days_cum": "day_count",
    "frost_days_cum": "frost_day",
    "heat_days_cum": "heat_day",
}.items():
    wx_sorted[newc] = wx_sorted.groupby(group_cols, sort=False)[src].cumsum()

def take_up_to_cutoff(sub: pd.DataFrame, cutoff_doy: int):
    """Return cumulative metrics up to (and including) cutoff_doy."""
    sub2 = sub[sub["doy"] <= int(cutoff_doy)]
    if sub2.empty:
        return None
    last = sub2.iloc[-1]
    days = float(last["days_cum"])
    return {
        "gdd_sum_to_cutoff": float(last["gdd_cum"]),
        "prcp_sum_to_cutoff": float(last["prcp_cum"]),
        "tmin_mean_to_cutoff": float(last["tmin_sum_cum"] / days),
        "tmax_mean_to_cutoff": float(last["tmax_sum_cum"] / days),
        "frost_days_to_cutoff": int(last["frost_days_cum"]),
        "heat_days_to_cutoff": int(last["heat_days_cum"]),
        "wx_days_covered": int(days),
        "wx_max_doy_used": int(last["doy"]),
    }

def enrich_sheet(sheet_name: str) -> pd.DataFrame:
    print(f"Processing sheet: {sheet_name}")
    df = norm(pd.read_excel(IN_SURV, sheet_name=sheet_name))
    df["site_id"] = df["site_id"].astype(str)

    need = {"site_id","year","l","r","event","first_obs_doy","last_obs_doy"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"[{sheet_name}] missing columns: {missing}")

    # cutoff_doy = R if event==1 and R present, else last_obs_doy (defensive clamp to <=366)
    df["cutoff_doy"] = np.where((df["event"] == 1) & df["r"].notna(), df["r"], df["last_obs_doy"])
    df["cutoff_doy"] = df["cutoff_doy"].clip(lower=1, upper=366).astype(int)

    covar_cols = ["gdd_sum_to_cutoff","prcp_sum_to_cutoff",
                  "tmin_mean_to_cutoff","tmax_mean_to_cutoff",
                  "frost_days_to_cutoff","heat_days_to_cutoff",
                  "wx_days_covered","wx_max_doy_used"]
    for c in covar_cols:
        df[c] = np.nan

    wx_groups = {k: v for k, v in wx_sorted.groupby(group_cols, sort=False)}
    misses = 0
    for idx, row in df.iterrows():
        key = (row["site_id"], int(row["year"]))
        sub = wx_groups.get(key)
        if sub is None or sub.empty:
            misses += 1
            continue
        vals = take_up_to_cutoff(sub, int(row["cutoff_doy"]))
        if vals is None:
            misses += 1
            continue
        for k, v in vals.items():
            df.at[idx, k] = v

    # Cast ints cleanly where present
    for c in ["frost_days_to_cutoff","heat_days_to_cutoff","wx_days_covered","wx_max_doy_used"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").dropna().astype(int, errors="ignore")

    print(f"  filled rows: {(df['gdd_sum_to_cutoff'].notna()).sum()} / {len(df)}"
          f"  (missing site-year weather matches: {misses})")
    return df

enriched = {s: enrich_sheet(s) for s in sheets}

with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as xw:
    for s, frame in enriched.items():
        frame.to_excel(xw, index=False, sheet_name=s)
    readme = pd.DataFrame({
        "README":[
            "Survival intervals joined with real NASA POWER weather (daily).",
            "Aggregation per (site_id, year) is through cutoff_doy:",
            "  cutoff_doy = R (first YES) if event=1, else last_obs_doy.",
            "Features: gdd_sum_to_cutoff, tmin_mean_to_cutoff, tmax_mean_to_cutoff, prcp_sum_to_cutoff,",
            "          frost_days_to_cutoff (tmin≤0°C), heat_days_to_cutoff (tmax≥35°C),",
            "          wx_days_covered, wx_max_doy_used.",
            "Outputs are saved only under survival_analysis_results/."
        ]
    })
    readme.to_excel(xw, index=False, sheet_name="README")

print(f"Saved: {OUT_XLSX}")
