# 06_format_model_outputs.py
# Tidy & format AFT model outputs from 05_fit_interval_models.py (Excel only)
# Input : 05_fit_interval_models/interval_model_results.xlsx
# Output: 06_format_model_outputs/interval_model_formatted.xlsx

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# --- console UTF-8 (Windows) ---
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Run this from the project root: D:/American_Vitis
ROOT = Path.cwd()

IN_XLSX = ROOT / "05_fit_interval_models" / "interval_model_results.xlsx"
OUT_DIR = ROOT / "06_format_model_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_XLSX = OUT_DIR / "interval_model_formatted.xlsx"

def normcols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (df.columns.str.strip().str.lower()
                  .str.replace(" ", "_")
                  .str.replace("-", "_"))
    return df

print("Reading model results…")
xls = pd.ExcelFile(IN_XLSX)
sheets = set(xls.sheet_names)

# --- load quickview if present ---
quick = pd.DataFrame()
if "model_quickview" in sheets:
    quick = pd.read_excel(IN_XLSX, sheet_name="model_quickview")
    quick = normcols(quick)

# --- load coefficients (required) ---
if "coefficients" not in sheets:
    raise SystemExit("Sheet 'coefficients' not found in interval_model_results.xlsx")

coef_raw = pd.read_excel(IN_XLSX, sheet_name="coefficients")
coef = normcols(coef_raw)

# Harmonize column names possibly produced by step 05
# lifelines summary after reset_index typically has 'param' and 'covariate'
# but step 05 might have 'parameter' instead of 'param'
if "parameter" in coef.columns and "param" not in coef.columns:
    coef = coef.rename(columns={"parameter": "param"})

# Defensive: ensure required columns exist
need = {
    "sheet","model","param","covariate",
    "coef","exp(coef)","se(coef)","z","p",
    "coef_lower_95%","coef_upper_95%",
    "exp(coef)_lower_95%","exp(coef)_upper_95%",
    "log_likelihood"
}
missing = need - set(coef.columns)
if missing:
    raise SystemExit(f"'coefficients' is missing columns: {missing}")

# Keep a clean subset and friendly order
keep_cols = [
    "sheet","model","param","covariate",
    "coef","exp(coef)","se(coef)","z","p",
    "coef_lower_95%","coef_upper_95%",
    "exp(coef)_lower_95%","exp(coef)_upper_95%",
    "log_likelihood"
]
coef_clean = coef[keep_cols].copy()

# Helper flags & sorting
coef_clean["is_intercept"] = (coef_clean["covariate"].str.lower() == "intercept")
coef_clean = coef_clean.sort_values(
    ["sheet","model","param","is_intercept","covariate"]
).reset_index(drop=True)

# Split by parameter type
alpha_like = {"alpha_","lambda_","mu_"}   # main time component
shape_like = {"sigma_","rho_","beta_"}    # ancillary/shape component

coef_time  = coef_clean[coef_clean["param"].isin(alpha_like)].copy()
coef_shape = coef_clean[coef_clean["param"].isin(shape_like)].copy()

# Effects tables (exclude intercepts by default)
effects_time = (
    coef_time[~coef_time["is_intercept"]]
      .rename(columns={
          "covariate":"term",
          "exp(coef)":"time_ratio",
          "coef_lower_95%":"coef_lo95",
          "coef_upper_95%":"coef_hi95",
          "exp(coef)_lower_95%":"tr_lo95",
          "exp(coef)_upper_95%":"tr_hi95"
      })
      .sort_values(["sheet","model","param","p","term"])
      .reset_index(drop=True)
)

# Quick interpretation
effects_time["interpretation"] = np.where(
    effects_time["time_ratio"] > 1.0,
    "Later timing (TR>1)",
    "Earlier timing (TR<1)"
)

# Top effects (p<=0.05)
top_effects = effects_time[effects_time["p"] <= 0.05].copy()

# README sheet
readme = pd.DataFrame({
    "README":[
        "This workbook formats outputs from the interval-censored AFT models (step 05).",
        "For AFT models, lifelines reports exp(coef) which is a Time Ratio (TR).",
        "Interpretation: TR>1 → later timing; TR<1 → earlier timing (per 1 SD increase in the z-scored covariate).",
        "Sheets:",
        "  - coefficients_clean: all parameters exactly as returned by lifelines (cleaned).",
        "  - effects_time: covariate effects on the main time component (alpha_/lambda_/mu_), intercepts removed.",
        "  - top_effects: subset of effects_time with p ≤ 0.05.",
        "  - effects_shape: ancillary/shape parameters (sigma_/rho_/beta_), usually reported but less interpretable.",
        "  - model_quickview: model-based medians if provided by step 05."
    ]
})

# Write the single output workbook
with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as xw:
    coef_clean.to_excel(xw, index=False, sheet_name="coefficients_clean")
    effects_time.to_excel(xw, index=False, sheet_name="effects_time")
    top_effects.to_excel(xw, index=False, sheet_name="top_effects")
    if not coef_shape.empty:
        coef_shape.to_excel(xw, index=False, sheet_name="effects_shape")
    if not quick.empty:
        quick.to_excel(xw, index=False, sheet_name="model_quickview")
    readme.to_excel(xw, index=False, sheet_name="README")

print(f"Saved: {OUT_XLSX}")
