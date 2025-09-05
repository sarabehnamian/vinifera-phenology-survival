# 10_refit_simple_models.py
# Refit simple (univariate) interval-censored AFT models using PRE-SEASON GDD only.
# Each script saves to its own folder; we also copy the main results into
# survival_analysis_results/ for downstream compatibility.
#
# Inputs:
#   02_fetch_nasa_power_weather/site_daily_weather.xlsx   (sheet: daily_weather)
#   07_validate_sensitivity/survival_with_weather_clean.xlsx   [preferred]
#   03_merge_survival_with_weather/survival_with_weather.xlsx  [fallback]
#
# Outputs (in 10_refit_simple_models/):
#   fixed_window_features.xlsx
#   interval_model_results_simple.xlsx
#   debug_simplefit_open_flowers.xlsx
#   debug_simplefit_ripe_fruits.xlsx
# Plus a copy of interval_model_results_simple.xlsx in survival_analysis_results/

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Console UTF-8 (Windows-friendly)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ---- paths following your folder organization ----
ROOT = Path.cwd()

OUT_DIR = ROOT / "10_refit_simple_models"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# optional convenience copy for step-11
RESULTS_COPY_DIR = ROOT / "survival_analysis_results"
RESULTS_COPY_DIR.mkdir(parents=True, exist_ok=True)

# Inputs
WX_XLSX   = ROOT / "02_fetch_nasa_power_weather" / "site_daily_weather.xlsx"
SURV_CLEAN = ROOT / "07_validate_sensitivity" / "survival_with_weather_clean.xlsx"
SURV_RAW   = ROOT / "03_merge_survival_with_weather" / "survival_with_weather.xlsx"

# Outputs
FIXED_FEATS     = OUT_DIR / "fixed_window_features.xlsx"
RESULTS_SIMPLE  = OUT_DIR / "interval_model_results_simple.xlsx"
RESULTS_SIMPLE_COPY = RESULTS_COPY_DIR / "interval_model_results_simple.xlsx"

def norm(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to lowercase with underscores."""
    df = df.copy()
    df.columns = (df.columns.str.strip()
                  .str.lower().str.replace(" ", "_").str.replace("-", "_"))
    return df

def zscore(series: pd.Series) -> pd.Series:
    """Compute z-scores for a series; return 0 if constant/missing."""
    x = pd.to_numeric(series, errors="coerce")
    mu = np.nanmean(x)
    sd = np.nanstd(x, ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(0.0, index=series.index)
    return (x - mu) / sd

def load_daily_weather() -> pd.DataFrame:
    """Load and process daily weather (step 02)."""
    if not WX_XLSX.exists():
        raise FileNotFoundError(f"Weather file not found: {WX_XLSX}")
    wx = pd.read_excel(WX_XLSX, sheet_name="daily_weather")
    wx = norm(wx)

    need = {"site_id", "date", "gdd_base10", "tmin_c", "tmax_c", "prcp_mm"}
    missing = need - set(wx.columns)
    if missing:
        raise SystemExit(f"[weather] Missing columns: {missing}")

    wx["site_id"] = wx["site_id"].astype(str)
    wx["date"] = pd.to_datetime(wx["date"], errors="coerce")
    wx = wx.dropna(subset=["site_id", "date"]).copy()
    wx["year"] = wx["date"].dt.year.astype(int)
    wx["doy"]  = wx["date"].dt.dayofyear.astype(int)
    return wx

def build_fixed_window_features(wx: pd.DataFrame,
                                flowers_end=120,
                                fruits_end=180,
                                coverage_thresh=0.70):
    """Aggregate pre-season windows: DOY 1..flowers_end and 1..fruits_end."""
    rows = []

    def agg_window(df, end_doy, label):
        exp_days = end_doy
        sub = df[df["doy"].between(1, end_doy)].copy()
        if sub.empty:
            return None
        present = sub.shape[0]
        coverage = present / float(exp_days)
        if coverage < coverage_thresh:
            return {"_coverage_ok": False, "coverage": coverage}
        out = {
            "gdd_pre": float(pd.to_numeric(sub["gdd_base10"], errors="coerce").sum()),
            "prcp_pre": float(pd.to_numeric(sub["prcp_mm"],   errors="coerce").sum()),
            "tmin_pre_mean": float(pd.to_numeric(sub["tmin_c"], errors="coerce").mean()),
            "tmax_pre_mean": float(pd.to_numeric(sub["tmax_c"], errors="coerce").mean()),
            "frost_pre": int((pd.to_numeric(sub["tmin_c"], errors="coerce") <= 0).sum()),
            "heat_pre":  int((pd.to_numeric(sub["tmax_c"], errors="coerce") >= 35).sum()),
            "days_present": int(present),
            "coverage": float(coverage),
            "_coverage_ok": True,
            "window_label": label,
            "window_end_doy": int(end_doy),
        }
        return out

    for (sid, yr), grp in wx.groupby(["site_id", "year"], sort=False):
        f = agg_window(grp, flowers_end, "flowers_pre")
        if f is not None and f.get("_coverage_ok", False):
            rec = {"site_id": str(sid), "year": int(yr)}
            rec.update({f"flowers_{k}": v for k, v in f.items() if not k.startswith("_")})
            rows.append(rec)

        r = agg_window(grp, fruits_end, "fruits_pre")
        if r is not None and r.get("_coverage_ok", False):
            rec = {"site_id": str(sid), "year": int(yr)}
            rec.update({f"fruits_{k}": v for k, v in r.items() if not k.startswith("_")})
            rows.append(rec)

    fx = norm(pd.DataFrame(rows))
    if fx.empty:
        # still write an empty workbook with the expected sheets
        with pd.ExcelWriter(FIXED_FEATS, engine="openpyxl") as xw:
            pd.DataFrame().to_excel(xw, index=False, sheet_name="long_by_window")
            pd.DataFrame().to_excel(xw, index=False, sheet_name="flowers_window")
            pd.DataFrame().to_excel(xw, index=False, sheet_name="fruits_window")
        print(f"  Saved (empty): {FIXED_FEATS}")
        return pd.DataFrame(), pd.DataFrame()

    # Wide tables by window
    keep_cols = ["site_id", "year"]
    value_cols = [c for c in fx.columns if c not in keep_cols]
    fx_wide = (
        fx.melt(id_vars=keep_cols, value_vars=value_cols, var_name="var", value_name="val")
          .assign(prefix=lambda d: d["var"].str.split("_", n=1, expand=True)[0],
                  name=lambda d: d["var"].str.split("_", n=1, expand=True)[1])
          .pivot_table(index=["site_id", "year", "prefix"], columns="name", values="val", aggfunc="first")
          .reset_index()
    )

    flowers = fx_wide[fx_wide["prefix"] == "flowers"].drop(columns=["prefix"]).copy()
    fruits  = fx_wide[fx_wide["prefix"] == "fruits"].drop(columns=["prefix"]).copy()

    with pd.ExcelWriter(FIXED_FEATS, engine="openpyxl") as xw:
        fx.to_excel(xw, index=False, sheet_name="long_by_window")
        flowers.to_excel(xw, index=False, sheet_name="flowers_window")
        fruits.to_excel(xw, index=False, sheet_name="fruits_window")
    print(f"  Saved: {FIXED_FEATS}")
    return flowers, fruits

def load_survival_sheet(name: str) -> pd.DataFrame:
    """Load a specific sheet from the survival workbook (clean preferred)."""
    if SURV_CLEAN.exists():
        src = SURV_CLEAN
        data_type = "cleaned"
        print(f"  Using cleaned data from: {src.name}")
    elif SURV_RAW.exists():
        src = SURV_RAW
        data_type = "raw"
        print(f"  Using raw data (fallback) from: {src.name}")
    else:
        raise FileNotFoundError(f"No survival workbook found at {SURV_CLEAN} or {SURV_RAW}")

    df = norm(pd.read_excel(src, sheet_name=name))

    # Required columns
    for need in ["site_id", "year", "l", "r", "event", "first_obs_doy", "last_obs_doy"]:
        if need not in df.columns:
            raise SystemExit(f"[{name}] missing column '{need}' in {src.name}")

    # Coerce basics
    df["site_id"] = df["site_id"].astype(str)
    df["year"] = df["year"].astype(int)
    df["l"] = pd.to_numeric(df["l"], errors="coerce").clip(lower=1, upper=366)
    df["r"] = pd.to_numeric(df["r"], errors="coerce").clip(lower=1, upper=366)

    # Right-censor: R = +inf if missing; fix degenerate intervals
    df["r_filled"] = df["r"].where(df["r"].notna(), np.inf)
    eq = np.isfinite(df["r_filled"]) & (df["r_filled"] <= df["l"])
    df.loc[eq, "r_filled"] = df.loc[eq, "l"] + 1e-6

    # If using raw data, drop implausibly early fruit events (carry-over)
    if data_type == "raw" and name == "ripe_fruits":
        early_fruit_mask = (df["event"] == 1) & df["r"].notna() & (df["r"] < 120)
        if early_fruit_mask.any():
            print(f"    Dropping {int(early_fruit_mask.sum())} early fruit events (R < 120 DOY)")
            df = df[~early_fruit_mask].copy()

    return df

def col_present(cols, *cands):
    """Return the first present candidate column name (or None)."""
    for c in cands:
        if c in cols:
            return c
    return None

def prep_and_fit(name: str,
                 surv_df: pd.DataFrame,
                 window_df: pd.DataFrame,
                 window_prefix: str,
                 min_coverage=0.70):
    """Prepare data and fit Log-Logistic & Weibull AFT models (univariate)."""
    from lifelines import LogLogisticAFTFitter, WeibullAFTFitter

    # Join on site_id, year
    df = surv_df.merge(window_df, on=["site_id", "year"], how="left")

    # Detect feature column names (wide tables use bare names)
    cov_name = col_present(df.columns, f"{window_prefix}_gdd_pre", "gdd_pre")
    cov_cov  = col_present(df.columns, f"{window_prefix}_coverage", "coverage")

    if cov_name is None:
        raise SystemExit(f"[{name}] Could not find pre-season GDD column after merge.")

    # Coverage filter if available
    if cov_cov is not None and cov_cov in df.columns:
        before = len(df)
        df = df[pd.to_numeric(df[cov_cov], errors="coerce") >= float(min_coverage)].copy()
        print(f"  {name}: kept {len(df)}/{before} rows with coverage >= {int(min_coverage*100)}%")

    # Keep needed columns & clean
    keep = ["site_id", "year", "l", "r_filled", "event", cov_name]
    df = df[keep].copy()
    df = df.dropna(subset=["l", cov_name]).copy()

    # Z-score the covariate
    zcol = cov_name + "_z"
    df[zcol] = zscore(df[cov_name])

    # Save diagnostics
    dbg_path = OUT_DIR / f"debug_simplefit_{name}.xlsx"
    with pd.ExcelWriter(dbg_path, engine="openpyxl") as xw:
        df.to_excel(xw, index=False, sheet_name="fit_data")
        pd.DataFrame({
            "col": ["l", "r_filled", cov_name, zcol],
            "n_na": [df["l"].isna().sum(),
                     df["r_filled"].isna().sum(),
                     df[cov_name].isna().sum(),
                     df[zcol].isna().sum()]
        }).to_excel(xw, index=False, sheet_name="na_check")

    # Design matrix for lifelines (allow +inf in r_filled; drop NaNs only)
    design = df[["l", "r_filled", zcol]].copy()
    design_no_na = design[~design.isna().any(axis=1)].copy()
    dropped = len(design) - len(design_no_na)
    if dropped:
        print(f"  {name}: dropped {dropped} rows with NaNs in design matrix")

    print(f"  Rows after join/clean: {len(design_no_na)} (events present: {int((surv_df['event']==1).sum())})")
    print(f"  [diag] wrote {dbg_path.name}")

    results = []
    fits = {}

    for model_name, Fitter in [("LogLogisticAFT", LogLogisticAFTFitter),
                               ("WeibullAFT",    WeibullAFTFitter)]:
        try:
            print(f"  Fitting {model_name} ({zcol})...")
            m = Fitter()
            m.fit_interval_censoring(design_no_na,
                                     lower_bound_col="l",
                                     upper_bound_col="r_filled")

            # Predicted medians at z=0 and z=+1 SD
            med0 = float(m.predict_median(pd.DataFrame([{zcol: 0.0}])).iloc[0])
            med1 = float(m.predict_median(pd.DataFrame([{zcol: 1.0}])).iloc[0])
            tr = med1 / med0

            # --- MultiIndex-safe extraction of coef & se(coef) for zcol ---
            summ = m.summary
            if isinstance(summ.index, pd.MultiIndex):
                # last level is typically the covariate name
                mask = (summ.index.get_level_values(-1) == zcol)
            else:
                mask = summ.index.astype(str).str.contains(zcol, regex=False)

            if mask.any():
                coef = float(summ.loc[mask, "coef"].iloc[0])
                se   = float(summ.loc[mask, "se(coef)"].iloc[0])
                ci_lower = float(np.exp(coef - 1.96 * se))
                ci_upper = float(np.exp(coef + 1.96 * se))
            else:
                ci_lower = np.nan
                ci_upper = np.nan
            # --------------------------------------------------------------

            results.append({
                "sheet": name,
                "model": model_name,
                "covariate": cov_name,
                "median_DOY_at_z0": med0,
                "median_DOY_at_z+1": med1,
                "time_ratio_(+1SD)": tr,
                "CI_lower_95%": ci_lower,
                "CI_upper_95%": ci_upper,
                "n_fit_rows": len(design_no_na),
                "n_rows": len(design_no_na)  # alias for downstream code
            })
            fits[model_name] = m

        except Exception as e:
            print(f"  {model_name} failed: {e}")
            results.append({
                "sheet": name,
                "model": model_name,
                "covariate": cov_name,
                "median_DOY_at_z0": np.nan,
                "median_DOY_at_z+1": np.nan,
                "time_ratio_(+1SD)": np.nan,
                "CI_lower_95%": np.nan,
                "CI_upper_95%": np.nan,
                "n_fit_rows": len(design_no_na),
                "n_rows": len(design_no_na),
                "error": str(e)
            })

    return pd.DataFrame(results), fits

def main():
    print("=" * 60)
    print("SCRIPT 10: REFIT SIMPLE MODELS")
    print("=" * 60)

    print("\nStep 1: Loading daily weather data...")
    wx = load_daily_weather()
    print(f"  Loaded {len(wx)} weather records")

    print("\nStep 2: Building fixed-window features...")
    print("  Computing windows: flowers(1-120), fruits(1-180)...")
    flowers_fx, fruits_fx = build_fixed_window_features(wx,
                                                        flowers_end=120,
                                                        fruits_end=180,
                                                        coverage_thresh=0.70)
    print(f"  Created {len(flowers_fx)} flower windows, {len(fruits_fx)} fruit windows")

    print("\nStep 3: Loading survival data...")
    surv_open = load_survival_sheet("open_flowers")
    surv_ripe = load_survival_sheet("ripe_fruits")

    all_res = []

    print("\n" + "=" * 40)
    print("FITTING MODELS: open_flowers")
    print("=" * 40)
    res_open, _ = prep_and_fit("open_flowers", surv_open, flowers_fx, window_prefix="flowers")
    all_res.append(res_open)

    print("\n" + "=" * 40)
    print("FITTING MODELS: ripe_fruits")
    print("=" * 40)
    res_ripe, _ = prep_and_fit("ripe_fruits", surv_ripe, fruits_fx, window_prefix="fruits")
    all_res.append(res_ripe)

    if not all_res:
        print("\nNo model results to save.")
        return

    # Combine results
    res = pd.concat(all_res, ignore_index=True)

    # Step 4: Save results
    print("\nStep 4: Saving results...")
    with pd.ExcelWriter(RESULTS_SIMPLE, engine="openpyxl") as xw:
        res.to_excel(xw, index=False, sheet_name="univariate_results")

        # Summary by phenophase
        summary = (res.groupby("sheet", as_index=True)
                     .agg(time_ratio_mean=("time_ratio_(+1SD)", "mean"),
                          time_ratio_sd  =("time_ratio_(+1SD)", "std"),
                          n_rows         =("n_rows", "first"))
                     .round(3))
        summary.to_excel(xw, sheet_name="summary_by_phenophase")

        # README sheet
        pd.DataFrame({
            "README": [
                "Univariate AFT (Log-Logistic, Weibull) with pre-season GDD only.",
                "Right-censoring handled by setting upper bound (R) = +inf.",
                "Time ratio reported as median_DOY(z=+1) / median_DOY(z=0), with z = standardized GDD_pre.",
                "Columns used from features: gdd_pre, coverage, days_present (if present).",
                "",
                "Data source priority:",
                "1) Cleaned data from 07_validate_sensitivity/ (if available)",
                "2) Raw data from 03_merge_survival_with_weather/ (fallback)",
                "",
                "Pre-season windows:",
                "- Flowers: DOY 1–120 (before typical flowering ~150)",
                "- Fruits: DOY 1–180 (before typical fruiting ~205)"
            ]
        }).to_excel(xw, index=False, sheet_name="README")

    # optional: drop a copy for survival_analysis_results/
    try:
        res.to_excel(RESULTS_SIMPLE_COPY, index=False, sheet_name="univariate_results")
        print(f"  Saved copy: {RESULTS_SIMPLE_COPY}")
    except Exception as e:
        print(f"  (Could not save copy in survival_analysis_results/): {e}")

    print(f"  Saved: {RESULTS_SIMPLE}")

    print("\n" + "=" * 60)
    print("OUTPUTS IN 10_refit_simple_models/:")
    print("=" * 60)
    for file in sorted(OUT_DIR.glob("*.xlsx")):
        print(f"  - {file.name}")

    if SURV_CLEAN.exists():
        print("\nUsed CLEANED survival data from step 07")
    else:
        print("\nUsed RAW survival data from step 03 (cleaned data not available)")

    print("\nScript 10 completed successfully!")

if __name__ == "__main__":
    main()
