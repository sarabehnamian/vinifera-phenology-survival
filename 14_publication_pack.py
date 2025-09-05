# 14_publication_pack.py
# Build manuscript-ready tables & a short results blurb from your CI results.
# Inputs:
#   survival_analysis_results/interval_model_results_bivariate_CI.xlsx
#   (optional) survival_analysis_results/interval_model_results_simple.xlsx
# Outputs:
#   14_publication_pack/publication_pack_summary.xlsx
#   14_publication_pack/results_blurb.txt
#   survival_analysis_results/publication_pack_summary.xlsx (copy)

import sys
from pathlib import Path
import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Fix paths to use project root
PROJECT = Path.cwd()
AGG = PROJECT / "survival_analysis_results"
OUT_DIR = PROJECT / "14_publication_pack"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BIV_CI = AGG / "interval_model_results_bivariate_CI.xlsx"
SIMPLE = AGG / "interval_model_results_simple.xlsx"

def norm(df):
    df = df.copy()
    df.columns = (df.columns.str.strip()
                  .str.lower().str.replace(" ", "_").str.replace("-", "_"))
    return df

def load_sheet(path):
    xls = pd.ExcelFile(path)
    # choose first sheet that looks like results
    prefer = [s for s in xls.sheet_names if "result" in s.lower()]
    sheet = prefer[0] if prefer else xls.sheet_names[0]
    df = pd.read_excel(path, sheet_name=sheet)
    return norm(df)

def best_col(df, candidates, default=None):
    for c in candidates:
        if c in df.columns:
            return c
    return default

def prettify_cov(s):
    s = str(s)
    # collapse any z-suffix
    s = s.replace("_z", "")
    # friendly names
    s = s.replace("flowers_", "").replace("fruits_", "")
    s = s.replace("gdd_pre", "GDD (pre-season)")
    s = s.replace("prcp_pre", "Precipitation (pre-season)")
    s = s.replace("tmin_pre_mean", "Mean Tmin (pre-season)")
    s = s.replace("tmax_pre_mean", "Mean Tmax (pre-season)")
    s = s.replace("frost_pre", "Frost days (pre-season)")
    s = s.replace("heat_pre", "Heat days (pre-season)")
    s = s.replace("gdd_sum_to_cutoff", "GDD to cutoff")
    s = s.replace("prcp_sum_to_cutoff", "Precip to cutoff")
    return s

def make_table(df):
    # Figure out the key columns regardless of exact names
    df = df.copy()

    # Phenophase/sheet name
    ph_col = best_col(df, ["sheet","phenophase","phase"], "sheet")
    if ph_col not in df.columns:
        df[ph_col] = "unknown"

    # Model name
    model_col = best_col(df, ["model","family"], "model")
    if model_col not in df.columns:
        df[model_col] = "WeibullAFT"

    # Covariate
    cov_col = best_col(df, ["covariate","parameter","term"], "covariate")
    if cov_col not in df.columns:
        df[cov_col] = "gdd_pre"

    # Time ratio (point)
    tr_col = best_col(df, ["time_ratio", "time_ratio_(+1sd)", "tr", "ratio"], None)
    if tr_col is None:
        # some sheets store TR as exp(coef) or alike; try to compute if coef exists
        coef_col = best_col(df, ["coef","coefficient","estimate"], None)
        if coef_col and coef_col in df.columns:
            df["__tr"] = np.exp(df[coef_col].astype(float))
            tr_col = "__tr"
        else:
            # fall back to NaN
            df["__tr"] = np.nan
            tr_col = "__tr"

    # CI lower / upper
    lo_col = best_col(df, ["ci_lower","ci_low","lower_ci","lower","ci_l","ci_lower_95","ci_lower95"], None)
    hi_col = best_col(df, ["ci_upper","ci_high","upper_ci","upper","ci_u","ci_upper_95","ci_upper95"], None)

    # N rows
    n_col = best_col(df, ["n_rows","n","n_obs"], None)
    if n_col and n_col not in df.columns:
        n_col = None  # safety

    keep = [ph_col, model_col, cov_col, tr_col, lo_col, hi_col] + ([n_col] if n_col else [])
    keep = [k for k in keep if k and k in df.columns]
    tidy = df[keep].copy()

    # Rename to standard
    rename = {
        ph_col: "Phenophase",
        model_col: "Model",
        cov_col: "Covariate",
        tr_col: "Time ratio",
    }
    if lo_col: rename[lo_col] = "CI lower"
    if hi_col: rename[hi_col] = "CI upper"
    if n_col:  rename[n_col]  = "N"

    tidy = tidy.rename(columns=rename)

    # Format covariate names
    tidy["Covariate (pretty)"] = tidy["Covariate"].map(prettify_cov)

    # Round numerics
    for c in ["Time ratio","CI lower","CI upper"]:
        if c in tidy.columns:
            tidy[c] = pd.to_numeric(tidy[c], errors="coerce").round(3)

    # Significance flag (CI excludes 1)
    if "CI lower" in tidy.columns and "CI upper" in tidy.columns:
        L = pd.to_numeric(tidy["CI lower"], errors="coerce")
        U = pd.to_numeric(tidy["CI upper"], errors="coerce")
        tidy["Sig. (95% CI excludes 1)"] = np.where((L.notna() & U.notna()) & ((U < 1) | (L > 1)), "Yes", "No")

    # Order nicely
    order_cols = ["Phenophase","Model","Covariate (pretty)","Time ratio"]
    if "CI lower" in tidy.columns and "CI upper" in tidy.columns:
        order_cols += ["CI lower","CI upper"]
    if "N" in tidy.columns:
        order_cols += ["N"]
    if "Sig. (95% CI excludes 1)" in tidy.columns:
        order_cols += ["Sig. (95% CI excludes 1)"]

    # Sort by phenophase, then model
    order_cols = [c for c in order_cols if c in tidy.columns]
    tidy = tidy[order_cols].sort_values(["Phenophase","Model","Covariate (pretty)"]).reset_index(drop=True)
    return tidy

def write_blurb(tidy: pd.DataFrame, out_txt: Path):
    lines = []
    lines.append("Results (auto-generated summary)\n")
    for ph in tidy["Phenophase"].unique():
        sub = tidy[tidy["Phenophase"]==ph]
        lines.append(f"{ph}:")
        for _, row in sub.iterrows():
            model = row["Model"]
            covp  = row["Covariate (pretty)"]
            tr    = row.get("Time ratio", np.nan)
            lo    = row.get("CI lower", np.nan)
            up    = row.get("CI upper", np.nan)
            sig   = row.get("Sig. (95% CI excludes 1)", "")
            if pd.notna(tr):
                if pd.notna(lo) and pd.notna(up):
                    lines.append(f"  - {model}, {covp}: TR={tr:.3f} (95% CI {lo:.3f}–{up:.3f}){', significant' if sig=='Yes' else ''}.")
                else:
                    lines.append(f"  - {model}, {covp}: TR={tr:.3f}.")
        lines.append("")
    out_txt.write_text("\n".join(lines), encoding="utf-8")

def main():
    if not BIV_CI.exists():
        raise SystemExit(f"Missing {BIV_CI}")

    print("Loading bivariate CI results…")
    bivar = load_sheet(BIV_CI)
    print(f"Columns in CI file: {list(bivar.columns)[:12]} …")

    tidy = make_table(bivar)

    # Optionally append simple results (if present), for a one-stop summary
    if SIMPLE.exists():
        print("Loading simple (univariate) results…")
        simp = load_sheet(SIMPLE)
        # ensure we have something resembling TR columns
        if "time_ratio_(+1sd)" in simp.columns:
            simp = simp.rename(columns={"time_ratio_(+1sd)":"time_ratio"})
        simp_tidy = make_table(simp)
        # Tag model label if missing
        if "Model" not in simp_tidy.columns:
            simp_tidy["Model"] = "AFT (univariate)"
        tidy = pd.concat([tidy, simp_tidy], ignore_index=True)

    # Write XLSX to step folder and aggregated folder
    out_step = OUT_DIR / "publication_pack_summary.xlsx"
    out_agg = AGG / "publication_pack_summary.xlsx"
    
    with pd.ExcelWriter(out_step, engine="openpyxl") as xw:
        tidy.to_excel(xw, index=False, sheet_name="Model_Results")
    with pd.ExcelWriter(out_agg, engine="openpyxl") as xw:
        tidy.to_excel(xw, index=False, sheet_name="Model_Results")
    
    print(f"Saved tables:\n- {out_step}\n- {out_agg}")

    # Write a short text blurb (only in step folder)
    blurb = OUT_DIR / "results_blurb.txt"
    write_blurb(tidy, blurb)
    print(f"Saved blurb:\n- {blurb}")

if __name__ == "__main__":
    main()