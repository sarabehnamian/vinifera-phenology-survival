# 05_fit_interval_models.py  — interval-censored AFT fits (single output folder, no duplicates)

from pathlib import Path
import sys
import numpy as np
import pandas as pd
from lifelines import LogLogisticAFTFitter, WeibullAFTFitter

# --- console encoding (Windows) ---
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# --- paths (run from project root that contains the 03_merge_survival_with_weather folder) ---
ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

IN_XLSX = ROOT / "03_merge_survival_with_weather" / "survival_with_weather.xlsx"

OUT_DIR = ROOT / "05_fit_interval_models"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# main results file (single location)
OUT_XLSX = OUT_DIR / "interval_model_results.xlsx"

# --- settings ---
EARLY_FRUIT_DOY = 120
BASE_COVARS = ["gdd_sum_to_cutoff", "prcp_sum_to_cutoff", "frost_days_to_cutoff", "heat_days_to_cutoff"]

# --- helpers ---
def norm(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )
    return df

def zscore_cols(df: pd.DataFrame, cols):
    zcols, kept = [], []
    for c in cols:
        if c not in df.columns:
            continue
        x = pd.to_numeric(df[c], errors="coerce")
        m, s = x.mean(), x.std(ddof=0)
        if not np.isfinite(s) or s == 0:
            continue
        df[f"{c}_z"] = (x - m) / s
        zcols.append(f"{c}_z")
        kept.append(c)
    return df, zcols, kept

def _write_debug_xlsx(df: pd.DataFrame, path: Path):
    """Write the exact fit matrix to XLSX (human-friendly)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out["right_censored"] = np.isinf(out["r"]).astype(int)
    out["r_display"] = out["r"].replace({np.inf: pd.NA})
    cols = ["l", "r_display", "right_censored", "event"] + [c for c in out.columns if c.endswith("_z")]
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        out[cols].to_excel(xw, index=False, sheet_name="fit_data")
        pd.DataFrame(
            {
                "README": [
                    "This is the exact matrix used for AFT fitting.",
                    "r_display is blank for right-censored rows (R=+inf in the model).",
                    "right_censored = 1 means upper bound was +inf.",
                    "Columns *_z are z-scored covariates.",
                ]
            }
        ).to_excel(xw, index=False, sheet_name="README")

def prep_sheet(sheet_name: str, raw: pd.DataFrame):
    notes = {"sheet": sheet_name}
    df = norm(raw)

    # Remove carry-over fruit (R < 120 DOY)
    if sheet_name.lower() == "ripe_fruits":
        early = df[(df["event"] == 1) & df["r"].notna() & (df["r"] < EARLY_FRUIT_DOY)]
        notes["dropped_early_ripe_n"] = int(len(early))
        df = df.drop(early.index)

    need = {"l", "r", "event"} | set(BASE_COVARS)
    miss = need - set(df.columns)
    if miss:
        raise SystemExit(f"[{sheet_name}] missing columns: {miss}")

    for c in ["l", "r"] + BASE_COVARS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["l"] = df["l"].clip(1, 366)

    # Ensure R > L for finite R
    same = df["r"].notna() & (df["r"] <= df["l"])
    df.loc[same, "r"] = df.loc[same, "l"] + 1e-6

    before = len(df)
    df = df.dropna(subset=["l"] + BASE_COVARS).copy()
    notes["dropped_missing_basecovars_or_L"] = int(before - len(df))

    # Right-censor: set R = +inf where missing
    df["r"] = df["r"].where(df["r"].notna(), np.inf)

    df, zcovars, kept = zscore_cols(df, BASE_COVARS)
    notes["covars_used"] = ", ".join(kept)
    notes["zcovars"] = ", ".join(zcovars)
    notes["n_rows_after_prep"] = int(len(df))
    notes["n_events"] = int(pd.to_numeric(df["event"], errors="coerce").fillna(0).sum())

    fit_df = pd.concat([df[["l", "r"]], df[zcovars]], axis=1)

    # Drop any row with NaN/Inf in L or z-covars (R may be +inf)
    check_cols = ["l"] + zcovars
    mask_bad = ~np.isfinite(fit_df[check_cols]).all(axis=1) | fit_df[check_cols].isna().any(axis=1)
    dropped = int(mask_bad.sum())
    if dropped:
        print(f"  [diag {sheet_name}] dropping {dropped} row(s) with NaN/Inf in {check_cols}")
    fit_df = fit_df.loc[~mask_bad].copy()

    # Write ONE debug XLSX (no duplicates, same folder as OUT_XLSX)
    dbg = pd.concat([df[["l", "r", "event"]], df[zcovars]], axis=1).loc[fit_df.index]
    _write_debug_xlsx(dbg, OUT_DIR / f"debug_fitdata_{sheet_name.lower()}.xlsx")
    print(f"  [diag] wrote debug_fitdata_{sheet_name.lower()}.xlsx")

    return fit_df, zcovars, notes

def fit_model(name: str, fit_df: pd.DataFrame):
    fitter = LogLogisticAFTFitter() if name == "LogLogisticAFT" else WeibullAFTFitter()
    return fitter.fit_interval_censoring(
        fit_df, lower_bound_col="l", upper_bound_col="r", show_progress=False
    )

def tidy_summary(sheet: str, model_name: str, fit):
    s = fit.summary.reset_index().rename(columns={"index": "parameter"})
    s.insert(0, "sheet", sheet)
    s.insert(1, "model", model_name)
    s["log_likelihood"] = fit.log_likelihood_
    return s

# --- run ---
print("Loading merged dataset…")
if not IN_XLSX.exists():
    raise SystemExit(f"Input file not found: {IN_XLSX}")

xls = pd.ExcelFile(IN_XLSX)
sheets = [s for s in xls.sheet_names if s.lower() in ("open_flowers", "ripe_fruits")]
if not sheets:
    raise SystemExit("No sheets named 'open_flowers' or 'ripe_fruits' in survival_with_weather.xlsx")

coef_tables, quickview, notes_rows = [], [], []

for sheet in sheets:
    print(f"\nPreparing data: {sheet}")
    raw = pd.read_excel(IN_XLSX, sheet_name=sheet)
    fit_df, zcovars, notes = prep_sheet(sheet, raw)
    notes_rows.append(notes)

    if len(fit_df) < 10 or len(zcovars) == 0:
        print(f"  Skipping {sheet}: not enough rows or covariates after cleaning.")
        continue

    for model_name in ("LogLogisticAFT", "WeibullAFT"):
        try:
            print(f"  Fitting {model_name} with covariates: {', '.join(zcovars)}")
            m = fit_model(model_name, fit_df)
            coef_tables.append(tidy_summary(sheet, model_name, m))
            # median at mean covariates (z=0) — FutureWarning-safe:
            med = m.predict_median(pd.DataFrame([dict.fromkeys(zcovars, 0.0)])).iloc[0]
            quickview.append({"sheet": sheet, f"{model_name}_median_DOY_at_meanX": float(med)})
        except Exception as e:
            print(f"  {model_name} failed: {e}")

def write_results(path: Path):
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        if quickview:
            pd.DataFrame(quickview).to_excel(xw, index=False, sheet_name="model_quickview")
        if coef_tables:
            pd.concat(coef_tables, ignore_index=True).to_excel(xw, index=False, sheet_name="coefficients")
        pd.DataFrame(notes_rows).to_excel(xw, index=False, sheet_name="data_notes")
        pd.DataFrame(
            {
                "README": [
                    "Interval-censored AFT (lifelines).",
                    f"Dropped ripe-fruit events with R<{EARLY_FRUIT_DOY} DOY (carry-over fruit).",
                    "Right-censoring handled by setting upper bound R=+inf (debug sheets show blank R instead, with right_censored=1).",
                    "Z-scored covariates: *_z. Positive coef = later timing (slower), negative = earlier (faster).",
                    "Debug fit matrix is saved once in this folder (no duplicate subfolder).",
                ]
            }
        ).to_excel(xw, index=False, sheet_name="README")

print("\nWriting results…")
write_results(OUT_XLSX)
print(f"Saved: {OUT_XLSX}")
