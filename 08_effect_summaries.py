# 08_effect_summaries.py — collate AFT effects and endpoint summaries (TIFF 300 dpi)
# Inputs (from earlier steps):
#   07_validate_sensitivity/survival_with_weather_clean.xlsx
#   07_validate_sensitivity/interval_model_results_sensitivity.xlsx   [preferred]
#   05_fit_interval_models/interval_model_results.xlsx                [fallback]
# Outputs (into 08_effect_summaries/):
#   summary.xlsx                  (sheets: sample_sizes, coefficients_used [if available])
#   time_ratio_open_flowers.tif   (forest-style TR plot)
#   time_ratio_ripe_fruits.tif
#   time_ratio_all.tif            (only if per-sheet TR plots unavailable)

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- console UTF-8 (Windows-friendly) ----
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ---- paths (run from D:/American_Vitis) ----
ROOT = Path.cwd()
IN_CLEAN = ROOT / "07_validate_sensitivity" / "survival_with_weather_clean.xlsx"
IN_MODELS_PRIMARY = ROOT / "07_validate_sensitivity" / "interval_model_results_sensitivity.xlsx"
IN_MODELS_FALLBACK = ROOT / "05_fit_interval_models" / "interval_model_results.xlsx"

# Fixed output folder name (logic-based)
OUT_DIR = ROOT / "08_effect_summaries"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_XLSX  = OUT_DIR / "summary.xlsx"
PLOT_OPEN  = OUT_DIR / "time_ratio_open_flowers.tif"
PLOT_FRUIT = OUT_DIR / "time_ratio_ripe_fruits.tif"
PLOT_ALL   = OUT_DIR / "time_ratio_all.tif"

# ---- helpers ----
def norm(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (df.columns.str.strip().str.lower()
                  .str.replace(" ", "_").str.replace("-", "_"))
    return df

def load_clean():
    if not IN_CLEAN.exists():
        raise FileNotFoundError(f"Missing cleaned workbook: {IN_CLEAN}")
    flowers = norm(pd.read_excel(IN_CLEAN, sheet_name="open_flowers"))
    fruits  = norm(pd.read_excel(IN_CLEAN, sheet_name="ripe_fruits"))
    return flowers, fruits

def pick_column(df, candidates, allow_contains=True):
    cols = list(df.columns)
    for c in candidates:
        if c in cols:
            return c
    if allow_contains:
        for c in candidates:
            for col in cols:
                if c in col:
                    return col
    return None

def choose_models_path():
    if IN_MODELS_PRIMARY.exists():
        print(f"Loaded coefficients from: {IN_MODELS_PRIMARY}")
        return IN_MODELS_PRIMARY
    print(f"Using fallback coefficients from: {IN_MODELS_FALLBACK}")
    return IN_MODELS_FALLBACK

def load_coefficients():
    path = choose_models_path()
    if not path.exists():
        print("No coefficients workbook found; TR plots will be skipped.")
        return pd.DataFrame()

    coef = pd.read_excel(path, sheet_name="coefficients")
    coef = norm(coef)

    # Identify common columns across step 05/07 variants
    sheet_col = "sheet" if "sheet" in coef.columns else pick_column(coef, ["phenophase","phase"])
    if sheet_col and sheet_col != "sheet":
        coef = coef.rename(columns={sheet_col: "sheet"})

    model_col = "model" if "model" in coef.columns else pick_column(coef, ["distribution","aft_model"])
    if model_col and model_col != "model":
        coef = coef.rename(columns={model_col: "model"})

    # lifelines AFT summary often has 'param' (alpha_/lambda_/mu_/sigma_/…)
    param_col = ("param" if "param" in coef.columns
                 else pick_column(coef, ["parameter","parameter_type","component"]))
    cov_col = ("covariate" if "covariate" in coef.columns
               else pick_column(coef, ["covariate","variable","name","term","index","parameter"]))

    if cov_col is None:
        print("No covariate/term column detected; TR plots will be skipped.")
        return pd.DataFrame()

    # Keep only primary time component (alpha_/lambda_/mu_), drop ancillary shape (sigma_/rho_/beta_)
    if param_col and param_col in coef.columns:
        mask_time = coef[param_col].astype(str).str.startswith(("alpha_","lambda_","mu_"))
        if mask_time.any():
            coef = coef[mask_time].copy()

    # Standardize covariate column name to 'term'
    if cov_col != "term":
        coef = coef.rename(columns={cov_col: "term"})

    # Focus on z-scored antecedent covariates aggregated to cutoff
    wanted = {"gdd_sum_to_cutoff_z","prcp_sum_to_cutoff_z","frost_days_to_cutoff_z","heat_days_to_cutoff_z"}
    coef = coef[coef["term"].isin(wanted)].copy()
    if coef.empty:
        print("No expected z-scored covariates found; TR plots will be skipped.")
        return coef

    # Ensure exp(coef) and TR CIs
    if "exp(coef)" in coef.columns and "exp_coef" not in coef.columns:
        coef = coef.rename(columns={"exp(coef)": "exp_coef"})
    if "exp_coef" not in coef.columns:
        if "coef" in coef.columns:
            coef["exp_coef"] = np.exp(pd.to_numeric(coef["coef"], errors="coerce"))
        else:
            coef["exp_coef"] = np.nan

    lo_col = pick_column(coef, ["coef_lower_95","coef_lower_95%","lower_95","coef_lower","lower_ci","ci_lower"])
    hi_col = pick_column(coef, ["coef_upper_95","coef_upper_95%","upper_95","coef_upper","upper_ci","ci_upper"])
    if lo_col and hi_col:
        lo = pd.to_numeric(coef[lo_col], errors="coerce")
        hi = pd.to_numeric(coef[hi_col], errors="coerce")
        coef["tr_lower"] = np.exp(lo)
        coef["tr_upper"] = np.exp(hi)
    elif "coef" in coef.columns and "se(coef)" in coef.columns:
        c  = pd.to_numeric(coef["coef"], errors="coerce")
        se = pd.to_numeric(coef["se(coef)"], errors="coerce")
        coef["tr_lower"] = np.exp(c - 1.96 * se)
        coef["tr_upper"] = np.exp(c + 1.96 * se)
    else:
        coef["tr_lower"] = np.nan
        coef["tr_upper"] = np.nan

    # Readable labels
    pretty = {
        "gdd_sum_to_cutoff_z":   "GDD to cutoff (z)",
        "prcp_sum_to_cutoff_z":  "Precip total to cutoff (z)",
        "frost_days_to_cutoff_z":"Frost days to cutoff (z)",
        "heat_days_to_cutoff_z": "Heat days to cutoff (z)",
    }
    coef["label"] = coef["term"].map(pretty).fillna(coef["term"])

    if "sheet" not in coef.columns:
        coef["sheet"] = "all"
    if "model" not in coef.columns:
        coef["model"] = "AFT"

    keep = ["sheet","model","term","label","exp_coef","tr_lower","tr_upper"]
    coef = coef[[c for c in keep if c in coef.columns]].copy()
    return coef

def sample_sizes(flowers: pd.DataFrame, fruits: pd.DataFrame) -> pd.DataFrame:
    def block(df, label):
        ev = df[df.get("event", 0) == 1]
        med_r = pd.to_numeric(ev.get("r", np.nan), errors="coerce")
        gdd   = pd.to_numeric(ev.get("gdd_sum_to_cutoff", np.nan), errors="coerce")
        return {
            "Endpoint": label,
            "N_with_weather": int(len(df)),
            "Events": int(ev.shape[0]),
            "Censored": int(len(df) - ev.shape[0]),
            "Event_rate": (float(df.get("event", pd.Series(dtype=float)).mean())
                           if len(df) and "event" in df.columns else np.nan),
            "Median_R_DOY_events": (float(med_r.median()) if ev.shape[0] else np.nan),
            "Mean_GDD_at_event": (float(gdd.mean()) if ev.shape[0] else np.nan),
        }
    return pd.DataFrame([block(flowers, "Open flowers"),
                         block(fruits,  "Ripe fruits")])

def tr_plot(coef_df, subset_sheet=None, out_path=Path("plot.tif")):
    df = coef_df.copy()
    if subset_sheet:
        df = df[df["sheet"].str.lower() == subset_sheet.lower()].copy()
        if df.empty:
            print(f"- TR plot skipped for {subset_sheet} (no coefficients).")
            return False

    # If multiple models, prefer Weibull; else choose first per term
    if "model" in df.columns and df["model"].nunique() > 1:
        if (df["model"] == "WeibullAFT").any():
            df = df[df["model"] == "WeibullAFT"].copy()
        else:
            df = df.sort_values("model").groupby("term", as_index=False).first()

    df = df.sort_values("term")
    y = np.arange(len(df))[::-1]
    x  = pd.to_numeric(df["exp_coef"], errors="coerce")
    lo = pd.to_numeric(df["tr_lower"], errors="coerce")
    hi = pd.to_numeric(df["tr_upper"], errors="coerce")
    labels = df["label"].tolist()

    fig, ax = plt.subplots(figsize=(5.0, 3.5), dpi=300)
    for yi, xi, l, h in zip(y, x, lo, hi):
        if np.isfinite(l) and np.isfinite(h):
            ax.hlines(yi, l, h, lw=2)
    ax.plot(x, y, "o", ms=5)
    ax.axvline(1.0, color="black", lw=1, ls="--")

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Time ratio (exp(coef))")
    ax.set_title(subset_sheet if subset_sheet else "All endpoints")

    finite_vals = pd.concat([x, lo, hi]).replace([np.inf, -np.inf], np.nan).dropna()
    if len(finite_vals):
        xmin = max(0.1, float(finite_vals.min()) * 0.9)
        xmax = min(10.0, float(finite_vals.max()) * 1.1)
        if xmin < xmax:
            ax.set_xlim(xmin, xmax)

    plt.tight_layout()
    fig.savefig(out_path, format="tiff", dpi=300, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print(f"- Saved TR plot: {out_path}")
    return True

def main():
    print("Loading inputs…")
    flowers, fruits = load_clean()
    coef = load_coefficients()

    print("Writing summaries…")
    t_sizes = sample_sizes(flowers, fruits)
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as xw:
        t_sizes.to_excel(xw, index=False, sheet_name="sample_sizes")
        if not coef.empty:
            coef.to_excel(xw, index=False, sheet_name="coefficients_used")

    print("Rendering time-ratio plots (TIFF 300 dpi)…")
    if coef.empty:
        print("- No usable coefficients found → skipping TR plots.")
    else:
        made_any = False
        made_any |= tr_plot(coef, "open_flowers", PLOT_OPEN)
        made_any |= tr_plot(coef, "ripe_fruits",  PLOT_FRUIT)
        if not made_any:
            tr_plot(coef, None, PLOT_ALL)

    print("\nOutputs")
    print(f"- Summary workbook: {OUT_XLSX}")
    if PLOT_OPEN.exists():  print(f"- TR plot: {PLOT_OPEN}")
    if PLOT_FRUIT.exists(): print(f"- TR plot: {PLOT_FRUIT}")
    if PLOT_ALL.exists():   print(f"- TR plot: {PLOT_ALL}")
    print("Done.")

if __name__ == "__main__":
    main()
