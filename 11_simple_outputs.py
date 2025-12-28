# 11_simple_outputs.py  (folder-aware + robust)
# Formats step-10 univariate results and makes simple forest-style plots.
# Inputs  (preferred → fallback):
#   10_refit_simple_models/interval_model_results_simple.xlsx
#   survival_analysis_results/interval_model_results_simple.xlsx
# Outputs:
#   11_simple_outputs/interval_model_results_simple_formatted.xlsx
#   10_refit_simple_models/interval_model_results_simple_formatted.xlsx  (copy)
#   survival_analysis_results/interval_model_results_simple_formatted.xlsx (copy)
#   forest_*_simple.tif in all three folders (300 dpi)

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# --- paths (run from D:/American_Vitis) ---
PROJECT = Path.cwd()
STEP10   = PROJECT / "10_refit_simple_models"
AGG      = PROJECT / "survival_analysis_results"
STEP11   = PROJECT / "11_simple_outputs"
STEP11.mkdir(parents=True, exist_ok=True)
STEP10.mkdir(parents=True, exist_ok=True)
AGG.mkdir(parents=True, exist_ok=True)

# input: prefer step-10 result, fallback to agg folder
CAND_IN = [
    STEP10 / "interval_model_results_simple.xlsx",
    AGG    / "interval_model_results_simple.xlsx",
]
IN_RESULTS = next((p for p in CAND_IN if p.exists()), None)
if IN_RESULTS is None:
    raise SystemExit(
        "Missing input: neither "
        f"{CAND_IN[0]} nor {CAND_IN[1]} exists."
    )

# outputs (write primary in step-11, plus 2 copies)
OUT_PATHS = [
    STEP11 / "interval_model_results_simple_formatted.xlsx",
    STEP10 / "interval_model_results_simple_formatted.xlsx",
    AGG    / "interval_model_results_simple_formatted.xlsx",
]

def norm(df):
    df = df.copy()
    df.columns = (df.columns.str.strip().str.lower()
                  .str.replace(" ", "_").str.replace("-", "_"))
    return df

def load_results():
    df = pd.read_excel(IN_RESULTS, sheet_name="univariate_results")
    df = norm(df)

    # Normalize known column drifts
    if "n_rows" not in df.columns and "n_fit_rows" in df.columns:
        df["n_rows"] = df["n_fit_rows"]

    expected = [
        "sheet","model","covariate",
        "median_doy_at_z0","median_doy_at_z+1","time_ratio_(+1sd)","n_rows"
    ]
    for c in expected:
        if c not in df.columns:
            df[c] = np.nan

    # compute TR if medians present but TR missing
    need_tr = df["time_ratio_(+1sd)"].isna()
    ok_med  = df["median_doy_at_z0"].notna() & df["median_doy_at_z+1"].notna()
    df.loc[need_tr & ok_med, "time_ratio_(+1sd)"] = (
        df.loc[need_tr & ok_med, "median_doy_at_z+1"] /
        df.loc[need_tr & ok_med, "median_doy_at_z0"]
    )

    df["phenophase"] = df["sheet"].map({
        "open_flowers": "Open flowers",
        "ripe_fruits": "Ripe fruits"
    }).fillna(df["sheet"])

    df["covariate_pretty"] = (
        df["covariate"].astype(str).str.replace("_", " ").str.strip()
    )

    print(f"Loaded: {IN_RESULTS}")
    return df

def save_formatted(df):
    wanted = ["phenophase","model","covariate_pretty",
              "median_doy_at_z0","median_doy_at_z+1","time_ratio_(+1sd)","n_rows"]
    cols = [c for c in wanted if c in df.columns]
    tidy = df[cols].copy()

    for out in OUT_PATHS:
        with pd.ExcelWriter(out, engine="openpyxl") as xw:
            tidy.to_excel(xw, index=False, sheet_name="univariate_results")
            pd.DataFrame({"README":[
                "Time ratio TR = median_DOY(z=+1 SD) / median_DOY(z=0).",
                "TR < 1 → earlier (accelerated); TR > 1 → later (delayed).",
                "These are one-covariate pre-season GDD fits from Step 10.",
                f"Input source: {IN_RESULTS.name}"
            ]}).to_excel(xw, index=False, sheet_name="README")
        print(f"Saved: {out}")
    return tidy

def forest_point(tr, title, stem):
    if not np.isfinite(tr):
        return False

    # dynamic x-limits on log scale; always include 1.0
    lower = max(min(0.5, tr/1.8), 0.1)
    upper = max(2.0, tr*1.8)

    fig, ax = plt.subplots(figsize=(4.2, 2.5))
    ax.axvline(1.0, linestyle="--", linewidth=1.0)
    ax.plot([tr], [0], marker="o")
    ax.set_xscale("log")
    ax.set_yticks([])
    ax.set_xlim(lower, upper)
    ax.set_xlabel("Time ratio (per +1 SD)")
    ax.set_title(title, fontsize=10)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()

    for folder in (STEP11, STEP10, AGG):
        p = folder / f"{stem}.tif"
        fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plots: {stem}.tif (in 11/10/agg folders)")
    return True

def main():
    df = load_results()
    tidy = save_formatted(df)

    # Pick pre-season GDD rows (allow variants like 'flowers gdd pre', 'gdd pre')
    mask = tidy["covariate_pretty"].str.contains(r"\bgdd\s*pre\b", case=False, regex=True)
    made = False
    for _, row in tidy[mask].iterrows():
        tr = float(row["time_ratio_(+1sd)"]) if pd.notna(row["time_ratio_(+1sd)"]) else np.nan
        title = f"{row['phenophase']} • {row['model']} • GDD pre-season (+1 SD)"
        stem  = f"forest_{row['phenophase'].lower().replace(' ','_')}_{row['model']}_gdd_pre_simple"
        ok = forest_point(tr, title, stem)
        made = made or ok

    if not made:
        print("No plottable TR values (all NaN) — plots skipped.")

if __name__ == "__main__":
    main()
