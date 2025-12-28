# 13_bivariate_ci_plots.py  (folder-aware)
# Bivariate AFT (pre-season GDD + precip) with 95% CI forest plots
# Inputs (preferred → fallbacks):
#   survival_analysis_results/fixed_window_features.xlsx
#   10_refit_simple_models/fixed_window_features.xlsx
#   09_diagnostics_and_refit/fixed_window_features.xlsx
#   07_validate_sensitivity/survival_with_weather_clean.xlsx  [preferred]
#   03_merge_survival_with_weather/survival_with_weather.xlsx [fallback]
# Outputs:
#   survival_analysis_results/interval_model_results_bivariate_CI.xlsx
#   13_bivariate_ci_plots/interval_model_results_bivariate_CI.xlsx
#   13_bivariate_ci_plots/forest_*_bivariate_CI.tif  (300 dpi)

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ---- paths (run from D:/American_Vitis) ----
PROJECT = Path.cwd()
AGG     = PROJECT / "survival_analysis_results"
STEP13  = PROJECT / "13_bivariate_ci_plots"
AGG.mkdir(parents=True, exist_ok=True)
STEP13.mkdir(parents=True, exist_ok=True)

# survival inputs
SURV_CLEAN = PROJECT / "07_validate_sensitivity" / "survival_with_weather_clean.xlsx"
SURV_RAW   = PROJECT / "03_merge_survival_with_weather" / "survival_with_weather.xlsx"

# fixed-window features (prefer AGG, then step folders)
CAND_FIXED = [
    AGG / "fixed_window_features.xlsx",
    PROJECT / "10_refit_simple_models" / "fixed_window_features.xlsx",
    PROJECT / "09_diagnostics_and_refit" / "fixed_window_features.xlsx",
]
FIXED = next((p for p in CAND_FIXED if p.exists()), None)
if FIXED is None:
    raise SystemExit("fixed_window_features.xlsx not found in AGG/step10/step09")

# outputs
OUT_MAIN = AGG   / "interval_model_results_bivariate_CI.xlsx"
OUT_STEP = STEP13 / "interval_model_results_bivariate_CI.xlsx"

# ---- helpers ----
def norm(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (df.columns.str.strip().str.lower()
                  .str.replace(" ", "_").str.replace("-", "_"))
    return df

def zscore(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    mu = np.nanmean(x); sd = np.nanstd(x, ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(0.0, index=x.index)
    return (x - mu) / sd

def load_surv(sheet: str) -> pd.DataFrame:
    src = SURV_CLEAN if SURV_CLEAN.exists() else SURV_RAW
    df = norm(pd.read_excel(src, sheet_name=sheet))
    for need in ["site_id","year","l","r","event","first_obs_doy","last_obs_doy"]:
        if need not in df.columns:
            raise SystemExit(f"[{sheet}] missing '{need}' in {src.name}")
    df["site_id"] = df["site_id"].astype(str)
    df["year"]    = df["year"].astype(int)
    df["l"] = pd.to_numeric(df["l"], errors="coerce").clip(1, 366)
    df["r"] = pd.to_numeric(df["r"], errors="coerce").clip(1, 366)
    df["r_filled"] = df["r"].where(df["r"].notna(), np.inf)
    mask = np.isfinite(df["r_filled"]) & (df["r_filled"] <= df["l"])
    df.loc[mask, "r_filled"] = df.loc[mask, "l"] + 1e-6
    return df

def harmonize_window(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    df = norm(df)
    base = ["gdd_pre","prcp_pre","tmin_pre_mean","tmax_pre_mean",
            "frost_pre","heat_pre","days_present","coverage"]
    if f"{prefix}_gdd_pre" not in df.columns:
        df = df.rename(columns={b: f"{prefix}_{b}" for b in base if b in df.columns})
    df["site_id"] = df["site_id"].astype(str)
    df["year"]    = df["year"].astype(int)
    return df

def load_fixed_windows():
    fl = harmonize_window(pd.read_excel(FIXED, sheet_name="flowers_window"), "flowers")
    fr = harmonize_window(pd.read_excel(FIXED, sheet_name="fruits_window"),  "fruits")
    return fl, fr

def extract_tr_ci(summary: pd.DataFrame, xname: str) -> tuple[float,float,float]:
    """
    Robustly pull coef and SE for the main time component (lambda_/mu_/alpha_),
    then return TR and its 95% CI on the exp scale.
    """
    s = summary.copy()
    # Standardize column names
    s.columns = [c.lower().replace(" ", "_") for c in s.columns]

    # Try MultiIndex (param, covariate)
    if isinstance(s.index, pd.MultiIndex):
        param_level = s.index.get_level_values(0).astype(str)
        covar_level = s.index.get_level_values(1).astype(str)
        sel = (param_level.isin(["lambda_","mu_","alpha_"])) & (covar_level == xname)
        row = s.loc[sel].iloc[0]
    else:
        # Flat columns 'param' + 'covariate'
        if {"param","covariate"} <= set(s.columns):
            row = s[(s["param"].isin(["lambda_","mu_","alpha_"])) &
                    (s["covariate"] == xname)].iloc[0]
        else:
            # Fallback: first row containing xname
            row = s[s.index.astype(str).str.contains(xname)].iloc[0]

    coef = float(row["coef"])
    se   = float(row.get("se(coef)", row.get("se_coef", np.nan)))
    if not np.isfinite(se):
        # try precomputed CI columns
        lo = float(row.get("coef_lower_95%", row.get("coef_lower_95", np.nan)))
        hi = float(row.get("coef_upper_95%", row.get("coef_upper_95", np.nan)))
        return np.exp(coef), np.exp(lo), np.exp(hi)
    lo = coef - 1.96*se
    hi = coef + 1.96*se
    return np.exp(coef), np.exp(lo), np.exp(hi)

def fit_and_ci(name: str, surv: pd.DataFrame, feats: pd.DataFrame,
               prefix: str, min_cov: float = 0.70) -> pd.DataFrame:
    from lifelines import WeibullAFTFitter, LogLogisticAFTFitter

    for col, caster in [("site_id", str), ("year", int)]:
        if col in surv.columns:  surv[col]  = surv[col].map(caster)
        if col in feats.columns: feats[col] = feats[col].map(caster)

    df = surv.merge(feats, on=["site_id","year"], how="left")
    covcol = f"{prefix}_coverage"
    if covcol in df.columns:
        before = len(df)
        df = df[pd.to_numeric(df[covcol], errors="coerce") >= min_cov].copy()
        print(f"  {name}: kept {len(df)}/{before} rows with coverage ≥ {int(min_cov*100)}%")

    gdd = f"{prefix}_gdd_pre"
    prc = f"{prefix}_prcp_pre"
    need = ["l","r_filled", gdd, prc]
    if any(c not in df.columns for c in need):
        missing = [c for c in need if c not in df.columns]
        raise SystemExit(f"[{name}] missing feature columns: {missing}")

    df = df.dropna(subset=need).copy()
    df[gdd+"_z"] = zscore(df[gdd])
    df[prc+"_z"] = zscore(df[prc])

    X = [gdd+"_z", prc+"_z"]
    design = pd.concat([df[["l","r_filled"]], df[X]], axis=1)

    results = []
    for model_name, Fitter in [("WeibullAFT", WeibullAFTFitter),
                               ("LogLogisticAFT", LogLogisticAFTFitter)]:
        try:
            m = Fitter()
            m.fit_interval_censoring(design, lower_bound_col="l", upper_bound_col="r_filled")
            tr_gdd, lo_gdd, hi_gdd = extract_tr_ci(m.summary, X[0])
            tr_prc, lo_prc, hi_prc = extract_tr_ci(m.summary, X[1])

            results += [
                {"sheet": name, "model": model_name, "covariate": "Pre-season GDD (+1 SD)",
                 "TR": tr_gdd, "CI_low": lo_gdd, "CI_high": hi_gdd, "n": len(df)},
                {"sheet": name, "model": model_name, "covariate": "Pre-season precipitation (+1 SD)",
                 "TR": tr_prc, "CI_low": lo_prc, "CI_high": hi_prc, "n": len(df)},
            ]

            # plot with CIs -> saved in Step 13 folder
            fig, ax = plt.subplots(figsize=(4.8, 3.0), dpi=300)
            vals = [tr_gdd, tr_prc]
            errs = [[tr_gdd - lo_gdd, tr_prc - lo_prc], [hi_gdd - tr_gdd, hi_prc - tr_prc]]
            y = np.arange(2)[::-1]
            ax.errorbar(vals, y, xerr=errs, fmt="o", capsize=3)
            ax.axvline(1.0, ls="--", lw=1)
            ax.set_yticks(y)
            ax.set_yticklabels(["Pre-season GDD (+1 SD)", "Pre-season precipitation (+1 SD)"])
            ax.set_xlabel("Time ratio (median DOY); <1 earlier, >1 later")
            ax.set_title(f"{name}: {model_name} (95% CI)")
            plt.tight_layout()
            fig.savefig(STEP13 / f"forest_{name}_{model_name}_bivariate_CI.tif",
                        dpi=300, bbox_inches="tight")
            plt.close(fig)
        except Exception as e:
            results.append({"sheet":name,"model":model_name,"covariate":"(fit failed)",
                            "TR":np.nan,"CI_low":np.nan,"CI_high":np.nan,"n":len(df),
                            "error":str(e)})
    return pd.DataFrame(results)

# ---- main ----
def main():
    fl, fr = load_fixed_windows()
    surv_open = load_surv("open_flowers")
    surv_ripe = load_surv("ripe_fruits")

    print("Fitting bivariate AFT with CIs…")
    res_open = fit_and_ci("open_flowers", surv_open, fl, "flowers", 0.70)
    res_ripe = fit_and_ci("ripe_fruits",  surv_ripe, fr, "fruits",  0.70)
    res = pd.concat([res_open, res_ripe], ignore_index=True)

    for out in (OUT_MAIN, OUT_STEP):
        with pd.ExcelWriter(out, engine="openpyxl") as xw:
            res.to_excel(xw, index=False, sheet_name="TR_95CI")
            pd.DataFrame({"README":[
                "Bivariate AFT with pre-season GDD & precipitation (z-scored).",
                "Time ratio (TR) = exp(beta_time) per +1 SD; TR<1 earlier (accelerated), TR>1 later (delayed).",
                f"Fixed-window source: {FIXED.name}",
            ]}).to_excel(xw, index=False, sheet_name="README")
        print(f"Saved: {out}")

if __name__ == "__main__":
    main()
