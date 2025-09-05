# 07_validate_sensitivity.py
# Validate & clean survival_with_weather, then refit AFT models on the cleaned data.
# I/O (run from project root D:/American_Vitis):
#   Input : 03_merge_survival_with_weather/survival_with_weather.xlsx
#   Output: 07_validate_sensitivity/ (Excel + TXT)

from pathlib import Path
import sys
import numpy as np
import pandas as pd
from lifelines import LogLogisticAFTFitter, WeibullAFTFitter

# ----- Console UTF-8 (Windows-friendly) -----
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ----- Paths -----
ROOT = Path.cwd()
IN_XLSX = ROOT / "03_merge_survival_with_weather" / "survival_with_weather.xlsx"

OUT_DIR = ROOT / "07_validate_sensitivity"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CLEAN = OUT_DIR / "survival_with_weather_clean.xlsx"
OUT_MODELS = OUT_DIR / "interval_model_results_sensitivity.xlsx"
OUT_SUMMARY = OUT_DIR / "sensitivity_summary.xlsx"
OUT_NOTES = OUT_DIR / "07_validate_sensitivity_notes.txt"

# ----- Config -----
EARLY_FRUIT_DOY = 120
BASE_COVARS = ["gdd_sum_to_cutoff", "prcp_sum_to_cutoff", "frost_days_to_cutoff", "heat_days_to_cutoff"]

# ----- Helpers -----
def norm(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.str.strip().str.lower()
        .str.replace(" ", "_").str.replace("-", "_")
    )
    return df

def add_cutoff(df: pd.DataFrame) -> pd.DataFrame:
    """
    cutoff_doy = R if event & R present; otherwise the last observation DOY.
    """
    df = df.copy()
    if "last_obs_doy" not in df.columns:
        # Fallback: if last_obs_doy is missing, use R where present else leave NaN
        df["last_obs_doy"] = np.nan
    df["cutoff_doy"] = np.where((df.get("event", 0) == 1) & df["r"].notna(), df["r"], df["last_obs_doy"])
    df["cutoff_doy"] = pd.to_numeric(df["cutoff_doy"], errors="coerce").clip(lower=1, upper=366)
    return df

def apply_filters(df: pd.DataFrame, sheet: str):
    """
    Filters:
      1) Require wx coverage to at least the cutoff: wx_max_doy_used >= cutoff_doy (if both present)
      2) For ripe_fruits: drop event rows with R < EARLY_FRUIT_DOY (carry-over fruit)
    """
    df = df.copy()
    before = len(df)

    # Filter 1: require weather through cutoff
    if {"wx_max_doy_used", "cutoff_doy"} <= set(df.columns):
        df = df[pd.to_numeric(df["wx_max_doy_used"], errors="coerce") >= pd.to_numeric(df["cutoff_doy"], errors="coerce")]

    # Filter 2: drop implausibly early ripe fruit events
    dropped_early = 0
    if sheet == "ripe_fruits":
        is_event = df.get("event", 0) == 1
        r_num = pd.to_numeric(df.get("r", np.nan), errors="coerce")
        mask_early = is_event & r_num.notna() & (r_num < EARLY_FRUIT_DOY)
        dropped_early = int(mask_early.sum())
        df = df.loc[~mask_early].copy()

    after = len(df)
    return df, before, after, dropped_early

def prep_for_fit(df: pd.DataFrame):
    """
    Prepare fitting matrix:
      - keep L, R, and base covariates
      - coerce numerics; drop rows missing L or any covariate (R can be right-censored)
      - clip L to [1, 366]
      - set R=+inf for right-censored; clip finite R to [1, 366]; fix degenerate intervals (R<=L)
      - z-score covariates to *_z
    """
    keep = ["l", "r"] + BASE_COVARS
    cols_present = [c for c in keep if c in df.columns]
    X = df[cols_present].copy()

    # numerics
    for c in cols_present:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    # require L and all covariates
    X = X.dropna(subset=["l"] + [c for c in BASE_COVARS if c in X.columns]).copy()

    # L bounds
    X["l"] = X["l"].clip(1, 366)

    # Right-censoring: set R=+inf when missing
    if "r" in X.columns:
        X["r"] = X["r"].where(X["r"].notna(), np.inf)
        # clip finite R only
        finite_r = np.isfinite(X["r"])
        X.loc[finite_r, "r"] = X.loc[finite_r, "r"].clip(1, 366)
        # fix degenerate finite intervals
        eq = finite_r & (X["r"] <= X["l"])
        X.loc[eq, "r"] = X.loc[eq, "l"] + 1e-6
    else:
        # If no R column, create one as +inf (all right-censored)
        X["r"] = np.inf

    # z-score covariates
    zcovars = []
    for c in BASE_COVARS:
        if c in X.columns:
            mu = X[c].mean()
            sd = X[c].std(ddof=0)
            if not np.isfinite(sd) or sd == 0:
                # drop constant columns from modeling
                continue
            zname = f"{c}_z"
            X[zname] = (X[c] - mu) / sd
            zcovars.append(zname)

    # keep only L, R, and z-covariates for fitting
    fit_df = X[["l", "r"] + zcovars].copy()

    # drop rows with NaN/Inf in L or z-covars (R may be +inf)
    check_cols = ["l"] + zcovars
    ok = np.isfinite(fit_df[check_cols]).all(axis=1) & ~fit_df[check_cols].isna().any(axis=1)
    fit_df = fit_df.loc[ok].copy()

    return fit_df, zcovars

def fit_models(fit_df: pd.DataFrame, zcovars: list[str]):
    """
    Fit LogLogistic and Weibull AFT on interval-censored data.
    Return concatenated coefficients and quickview medians at mean covariates.
    """
    if len(fit_df) < 10 or not zcovars:
        return pd.DataFrame(), pd.DataFrame()

    results = []
    quick = {}

    def do_fit(model, name):
        model.fit_interval_censoring(
            fit_df, lower_bound_col="l", upper_bound_col="r", show_progress=False
        )
        s = model.summary.reset_index().rename(columns={"index": "covariate"})
        s.insert(0, "model", name)
        # median at mean covariates (z=0)
        med = model.predict_median(pd.DataFrame([dict.fromkeys(zcovars, 0.0)])).iloc[0]
        quick[f"{name}_median_DOY_at_mean_covs"] = float(med)
        return s

    # Try both models, collect what succeeds
    try:
        results.append(do_fit(LogLogisticAFTFitter(), "LogLogisticAFT"))
    except Exception as e:
        results.append(pd.DataFrame({"model": ["LogLogisticAFT"], "note": [str(e)]}))

    try:
        results.append(do_fit(WeibullAFTFitter(), "WeibullAFT"))
    except Exception as e:
        results.append(pd.DataFrame({"model": ["WeibullAFT"], "note": [str(e)]}))

    coef = pd.concat(results, ignore_index=True, sort=False)
    quick_df = pd.DataFrame([quick]) if quick else pd.DataFrame()
    return coef, quick_df

# ----- Load input -----
print("Loading merged dataset…")
xls = pd.ExcelFile(IN_XLSX)
flowers = norm(pd.read_excel(xls, sheet_name="open_flowers"))
fruits = norm(pd.read_excel(xls, sheet_name="ripe_fruits"))

# ----- Apply validation filters -----
notes_lines = []
summary_rows = []
cleaned = {}

for sheet_name, df in (("open_flowers", flowers), ("ripe_fruits", fruits)):
    df = add_cutoff(df)
    df_clean, n_before, n_after, dropped_early = apply_filters(df, sheet_name)
    cleaned[sheet_name] = df_clean

    # event count post-cleaning
    events_after = int(df_clean["event"].sum()) if "event" in df_clean.columns else np.nan

    # sanity correlation among events
    pearson_r = np.nan
    ev = df_clean[df_clean.get("event", 0) == 1].copy()
    if not ev.empty and {"r", "gdd_sum_to_cutoff"} <= set(ev.columns):
        r_ser = pd.to_numeric(ev["r"], errors="coerce")
        gdd_ser = pd.to_numeric(ev["gdd_sum_to_cutoff"], errors="coerce")
        mask = r_ser.notna() & gdd_ser.notna()
        if mask.sum() >= 2:
            pearson_r = float(np.corrcoef(r_ser[mask], gdd_ser[mask])[0, 1])

    summary_rows.append({
        "sheet": sheet_name,
        "n_before": n_before,
        "n_after": n_after,
        "events_after": events_after,
        "dropped_ripe_r_lt_120": dropped_early if sheet_name == "ripe_fruits" else 0,
        "pearson_r_R_vs_GDD_events": pearson_r
    })

notes_lines.append(
    "Applied filters:\n"
    f"  • Require weather coverage to cutoff where available (wx_max_doy_used ≥ cutoff_doy).\n"
    f"  • Drop ripe_fruits event rows with R < {EARLY_FRUIT_DOY} DOY (likely carry-over fruit).\n"
)

summary_tbl = pd.DataFrame(summary_rows)

# ----- Save cleaned workbook -----
with pd.ExcelWriter(OUT_CLEAN, engine="openpyxl") as xw:
    cleaned["open_flowers"].to_excel(xw, index=False, sheet_name="open_flowers")
    cleaned["ripe_fruits"].to_excel(xw, index=False, sheet_name="ripe_fruits")
    pd.DataFrame({"README": [
        "This workbook contains validated/cleaned phenology + weather data.",
        f"Filters: wx_max_doy_used ≥ cutoff_doy; ripe_fruits events with R < {EARLY_FRUIT_DOY} dropped."
    ]}).to_excel(xw, index=False, sheet_name="README")

# ----- Refit AFT models on cleaned data -----
all_coef = []
quickviews = []

for sheet_name in ("open_flowers", "ripe_fruits"):
    print(f"Refitting AFT on cleaned data: {sheet_name}")
    fit_df, zcovars = prep_for_fit(cleaned[sheet_name])
    if fit_df.empty or not zcovars:
        quickviews.append(pd.DataFrame([{
            "sheet": sheet_name, "note": "insufficient rows or covariate variation after cleaning"
        }]))
        continue
    coef_tbl, quick_tbl = fit_models(fit_df, zcovars)
    if not coef_tbl.empty:
        coef_tbl.insert(0, "sheet", sheet_name)
        all_coef.append(coef_tbl)
    if not quick_tbl.empty:
        quick_tbl.insert(0, "sheet", sheet_name)
        quickviews.append(quick_tbl)

coef_out = pd.concat(all_coef, ignore_index=True, sort=False) if all_coef else pd.DataFrame()
quick_out = pd.concat(quickviews, ignore_index=True, sort=False) if quickviews else pd.DataFrame()

# ----- Save model results workbook -----
with pd.ExcelWriter(OUT_MODELS, engine="openpyxl") as xw:
    if not quick_out.empty:
        quick_out.to_excel(xw, index=False, sheet_name="model_quickview")
    if not coef_out.empty:
        coef_out.to_excel(xw, index=False, sheet_name="coefficients")
    summary_tbl.to_excel(xw, index=False, sheet_name="cleaning_summary")
    pd.DataFrame({"README": [
        "AFT interval-censored models refit on cleaned data.",
        "lifelines reports exp(coef) as a Time Ratio (TR): TR>1 → later timing; TR<1 → earlier timing."
    ]}).to_excel(xw, index=False, sheet_name="README")

# ----- Save compact summary workbook -----
with pd.ExcelWriter(OUT_SUMMARY, engine="openpyxl") as xw:
    summary_tbl.to_excel(xw, index=False, sheet_name="summary")

# ----- Save notes -----
notes_lines.append("\nCleaning summary table:\n")
notes_lines.append(summary_tbl.to_string(index=False))
OUT_NOTES.write_text("\n".join(notes_lines), encoding="utf-8")

print("\nSaved:")
print(f"  Cleaned data   → {OUT_CLEAN}")
print(f"  Model results  → {OUT_MODELS}")
print(f"  Summary        → {OUT_SUMMARY}")
print(f"  Notes          → {OUT_NOTES}")
