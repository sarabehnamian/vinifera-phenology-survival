# 09_diagnostics_and_refit.py
# Diagnostics (corr, VIF) on current covariates + refit simple models
# with exogenous fixed-window weather (avoids "to-cutoff" endogeneity).
#
# Inputs (from earlier steps):
#   07_validate_sensitivity/survival_with_weather_clean.xlsx     [preferred]
#   03_merge_survival_with_weather/survival_with_weather.xlsx    [fallback]
#   02_fetch_nasa_power_weather/site_daily_weather.xlsx          (sheet: daily_weather)
#
# Outputs (in a subfolder named like this file, i.e., 09_diagnostics_and_refit/):
#   diagnostics.xlsx                 (index of per-sheet files)
#   diagnostics_open_flowers.xlsx    (Pearson/Spearman corr, VIF)
#   diagnostics_ripe_fruits.xlsx     (Pearson/Spearman corr, VIF)
#   fixed_window_features.xlsx       (per sheet)
#   univariate_results.xlsx          (time ratios + CI for fixed-window GDD_pre_z)
#   forest_open_flowers_fixed.tif
#   forest_ripe_fruits_fixed.tif
#   forest_open_flowers_current.tif  (joint model on current covars; illustrative)
#   forest_ripe_fruits_current.tif   (joint model on current covars; illustrative)

import sys, math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lifelines import LogLogisticAFTFitter, WeibullAFTFitter

# ---- console UTF-8 (Windows-friendly) ----
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ---- paths ----
ROOT = Path.cwd()

# resolve script-stem and output folder named like this file
try:
    SCRIPT_STEM = Path(__file__).stem
except NameError:
    SCRIPT_STEM = "09_diagnostics_and_refit"

OUTDIR = ROOT / SCRIPT_STEM
OUTDIR.mkdir(parents=True, exist_ok=True)

# inputs aligned to your step folders
SURV_CLEAN    = ROOT / "07_validate_sensitivity" / "survival_with_weather_clean.xlsx"
SURV_FALLBACK = ROOT / "03_merge_survival_with_weather" / "survival_with_weather.xlsx"
WX_XLSX       = ROOT / "02_fetch_nasa_power_weather" / "site_daily_weather.xlsx"

# fixed exogenous windows (ends before typical detection)
FLOWERS_END_DOY = 120   # well before median flower DOY (~150)
FRUITS_END_DOY  = 180   # before median fruit DOY (~205)

# current (endogenous-to-cutoff) covariates for diagnostics
CURRENT_COVARS = [
    "gdd_sum_to_cutoff", "prcp_sum_to_cutoff",
    "frost_days_to_cutoff", "heat_days_to_cutoff"
]

# ---- helpers ----
def norm(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (df.columns.str.strip().str.lower()
                  .str.replace(" ", "_").str.replace("-", "_"))
    return df

def zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mu = s.mean()
    sd = s.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return s - mu
    return (s - mu) / sd

def vif_table(X: pd.DataFrame) -> pd.DataFrame:
    """
    Compute VIF without extra deps: VIF_j = 1 / (1 - R^2_j),
    where R^2_j is from regressing X_j on X_-j (OLS).
    """
    X = X.dropna().copy()
    # drop exact duplicate columns if any
    if X.shape[1] > 1:
        X = X.loc[:, ~X.T.duplicated(keep="first").values]
    vifs = []
    for col in X.columns:
        y = X[col].values
        X_others = X.drop(columns=[col]).values
        if X_others.shape[1] == 0:
            vifs.append((col, np.nan))
            continue
        Xmat = np.column_stack([np.ones(len(X_others)), X_others])
        beta, *_ = np.linalg.lstsq(Xmat, y, rcond=None)
        yhat = Xmat @ beta
        ss_res = float(np.sum((y - yhat)**2))
        ss_tot = float(np.sum((y - y.mean())**2))
        R2 = 0.0 if ss_tot == 0 else (1 - ss_res/ss_tot)
        vif = np.inf if (1 - R2) <= 0 else 1.0/(1.0 - R2)
        vifs.append((col, float(vif)))
    return pd.DataFrame(vifs, columns=["variable","VIF"])

def clip_interval(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["l"] = pd.to_numeric(df["l"], errors="coerce").clip(1, 366)
    if "r" in df.columns:
        df["r"] = pd.to_numeric(df["r"], errors="coerce")
        mask_fin = df["r"].notna()
        df.loc[mask_fin, "r"] = df.loc[mask_fin, "r"].clip(1, 366)
        bad = mask_fin & (df["r"] <= df["l"])
        df.loc[bad, "r"] = df.loc[bad, "l"] + 1e-6
    return df

def lifelines_univariate(df: pd.DataFrame, covar: str, family: str = "weibull"):
    """Fit a single-covariate interval-censored AFT and return TR + CI dict."""
    d = df[["l","r", covar]].dropna().copy()
    if len(d) < 8:
        return None
    model = WeibullAFTFitter() if family.lower().startswith("w") else LogLogisticAFTFitter()
    model.fit_interval_censoring(d, lower_bound_col="l", upper_bound_col="r", show_progress=False)
    summ = model.summary.reset_index().rename(columns={"index":"param"})
    row = summ[summ["param"].str.contains(covar, regex=False)].head(1)
    if row.empty:
        return None
    coef = float(row["coef"].iloc[0])
    se   = float(row["se(coef)"].iloc[0])
    z    = 1.96
    tr   = math.exp(coef)
    lo   = math.exp(coef - z*se)
    hi   = math.exp(coef + z*se)
    return {
        "model": ("WeibullAFT" if isinstance(model, WeibullAFTFitter) else "LogLogisticAFT"),
        "covariate": covar,
        "time_ratio": tr,
        "ci_lower": lo,
        "ci_upper": hi
    }

def forest_plot(labels, est, lo, hi, title, outpath, xlim=None):
    fig, ax = plt.subplots(figsize=(6.2, 4.0), dpi=300)
    y = np.arange(len(labels))[::-1]
    ax.hlines(y, lo, hi, lw=2)
    ax.plot(est, y, "o", ms=5)
    ax.axvline(1.0, ls="--", lw=1, color="k")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Time ratio (exp(coef))", fontsize=11)
    ax.set_title(title, fontsize=12)
    if xlim:
        ax.set_xlim(*xlim)
    ax.grid(True, axis="x", alpha=0.25)
    plt.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)

# ---- load survival workbook (prefer validated/cleaned) ----
if SURV_CLEAN.exists():
    surv = pd.ExcelFile(SURV_CLEAN)
    print(f"Loaded validated survival workbook: {SURV_CLEAN}")
else:
    surv = pd.ExcelFile(SURV_FALLBACK)
    print(f"Loaded fallback survival workbook: {SURV_FALLBACK}")

# ---- load daily weather (step 02) ----
wx = pd.read_excel(WX_XLSX, sheet_name="daily_weather")
wx = norm(wx)
wx["date"] = pd.to_datetime(wx["date"], errors="coerce")
wx["year"] = wx["date"].dt.year.astype(int)
wx["doy"]  = wx["date"].dt.dayofyear.astype(int)

# ---- A) Diagnostics on current (to-cutoff) covariates ----
diag_index = []
for sheet in [s for s in surv.sheet_names if s.lower() in ("open_flowers","ripe_fruits")]:
    df = norm(pd.read_excel(surv, sheet_name=sheet))
    df = clip_interval(df)
    # if using fallback, enforce R<120 drop for ripe_fruits (carry-over)
    if sheet.lower() == "ripe_fruits" and "r" in df.columns:
        df = df[~(df["r"].notna() & (pd.to_numeric(df["r"], errors="coerce") < 120))].copy()

    available = [c for c in CURRENT_COVARS if c in df.columns]
    if not available:
        continue
    X = df[available].apply(pd.to_numeric, errors="coerce")

    pear = X.corr(method="pearson").round(3)
    spear = X.corr(method="spearman").round(3)
    vif = vif_table(X.apply(zscore))

    out_path = OUTDIR / f"diagnostics_{sheet}.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
        pear.to_excel(xw, sheet_name="pearson_corr")
        spear.to_excel(xw, sheet_name="spearman_corr")
        vif.to_excel(xw, sheet_name="VIF", index=False)
    diag_index.append(out_path.name)
    print(f"- saved diagnostics: {out_path}")

# small index file that points to per-sheet diagnostics
with pd.ExcelWriter(OUTDIR / "diagnostics.xlsx", engine="openpyxl") as xw:
    pd.DataFrame({"files": diag_index}).to_excel(xw, index=False, sheet_name="index")

# ---- B) Build exogenous fixed-window features ----
def fixed_features_for(df_intervals: pd.DataFrame, end_doy: int) -> pd.DataFrame:
    out = []
    for (sid, yr), g in df_intervals.groupby(["site_id","year"]):
        sid_str = str(sid)
        sub = wx[(wx["site_id"].astype(str) == sid_str) &
                 (wx["year"] == int(yr)) &
                 (wx["doy"] <= int(end_doy))]
        if sub.empty:
            continue
        out.append({
            "site_id": sid_str,
            "year": int(yr),
            "gdd_pre": float(pd.to_numeric(sub["gdd_base10"], errors="coerce").sum()),
            "prcp_pre": float(pd.to_numeric(sub["prcp_mm"], errors="coerce").sum()),
            "tmin_pre_mean": float(pd.to_numeric(sub["tmin_c"], errors="coerce").mean()),
            "tmax_pre_mean": float(pd.to_numeric(sub["tmax_c"], errors="coerce").mean()),
            "frost_pre_days": int((pd.to_numeric(sub["tmin_c"], errors="coerce") <= 0).sum()),
            "heat_pre_days": int((pd.to_numeric(sub["tmax_c"], errors="coerce") >= 35).sum()),
            "window_end_doy": int(end_doy),
            "days_in_window": int(len(sub)),
        })
    return pd.DataFrame(out)

fixed_sheets = {}
for sheet, end_doy in [("open_flowers", FLOWERS_END_DOY), ("ripe_fruits", FRUITS_END_DOY)]:
    if sheet not in surv.sheet_names:
        continue
    base = norm(pd.read_excel(surv, sheet_name=sheet))
    base["site_id"] = base["site_id"].astype(str)
    base = clip_interval(base)
    if sheet == "ripe_fruits" and "r" in base.columns:
        base = base[~(base["r"].notna() & (pd.to_numeric(base["r"], errors="coerce") < 120))].copy()

    feats = fixed_features_for(base[["site_id","year"]], end_doy)
    merged = base.merge(feats, on=["site_id","year"], how="left")

    # z-score primary pre-season metrics
    for c in ["gdd_pre","prcp_pre","frost_pre_days","heat_pre_days"]:
        if c in merged.columns:
            merged[c + "_z"] = zscore(merged[c])
    fixed_sheets[sheet] = merged

with pd.ExcelWriter(OUTDIR / "fixed_window_features.xlsx", engine="openpyxl") as xw:
    for k, v in fixed_sheets.items():
        v.to_excel(xw, index=False, sheet_name=k)
print(f"- saved fixed-window features: {OUTDIR / 'fixed_window_features.xlsx'}")

# ---- C) Univariate AFT using exogenous GDD_pre_z ----
uni_rows = []
for sheet, df in fixed_sheets.items():
    col = "gdd_pre_z"
    if col not in df.columns:
        continue
    d = df[["l","r", col]].dropna().copy()
    if len(d) < 10:
        continue
    for fam in ("weibull","loglogistic"):
        try:
            res = lifelines_univariate(d, col, family=fam)
            if res:
                res.update({"sheet": sheet, "n_rows": int(len(d))})
                uni_rows.append(res)
        except Exception:
            pass

if uni_rows:
    uni = pd.DataFrame(uni_rows)
    uni["time_ratio"] = uni["time_ratio"].round(3)
    uni["ci_lower"]   = uni["ci_lower"].round(3)
    uni["ci_upper"]   = uni["ci_upper"].round(3)
    uni = uni[["sheet","model","covariate","n_rows","time_ratio","ci_lower","ci_upper"]]
    uni_out = OUTDIR / "univariate_results.xlsx"
    uni.to_excel(uni_out, index=False)
    print(f"- saved univariate AFT (fixed window): {uni_out}")

    # Forest per sheet (prefer Weibull; else first available)
    for sheet in uni["sheet"].unique():
        u = uni[(uni["sheet"] == sheet) & (uni["model"] == "WeibullAFT")]
        if u.empty:
            u = uni[uni["sheet"] == sheet].iloc[[0]]
        labels = ["GDD pre-season (z)"]
        est = u["time_ratio"].values
        lo  = u["ci_lower"].values
        hi  = u["ci_upper"].values
        xlim = (0.75, 1.5) if sheet == "open_flowers" else (0.9, 1.2)
        forest_plot(
            labels, est, lo, hi,
            title=f"{sheet} — univariate AFT (fixed window)",
            outpath=OUTDIR / f"forest_{sheet}_fixed.tif",
            xlim=xlim
        )

# ---- D) Optional: quick forest for current covariates (joint, illustrative) ----
for sheet in [s for s in surv.sheet_names if s.lower() in ("open_flowers","ripe_fruits")]:
    df = norm(pd.read_excel(surv, sheet_name=sheet))
    df = clip_interval(df)
    avail = [c for c in CURRENT_COVARS if c in df.columns]
    if len(avail) == 0:
        continue
    Z = df[avail].apply(zscore)
    d = pd.concat([df[["l","r"]], Z.add_suffix("_z")], axis=1).dropna()
    if len(d) < 15:
        continue
    try:
        m = WeibullAFTFitter()
        m.fit_interval_censoring(d, lower_bound_col="l", upper_bound_col="r", show_progress=False)
        sm = m.summary.reset_index().rename(columns={"index":"param"})
        keep = sm[sm["param"].str.contains("_z")].copy()
        if not keep.empty:
            keep["tr"] = np.exp(keep["coef"])
            keep["lo"] = np.exp(keep["coef"] - 1.96 * keep["se(coef)"])
            keep["hi"] = np.exp(keep["coef"] + 1.96 * keep["se(coef)"])
            labels = (keep["param"].str.replace("_z", "", regex=False)
                                   .str.replace("_", " ")
                                   .str.replace("gdd sum to cutoff", "GDD to cutoff")
                                   .str.replace("prcp sum to cutoff", "Precip total to cutoff")
                                   .str.replace("heat days to cutoff", "Heat days to cutoff")
                                   .str.replace("frost days to cutoff", "Frost days to cutoff")).tolist()
            forest_plot(
                labels, keep["tr"].values, keep["lo"].values, keep["hi"].values,
                title=f"{sheet} — current covariates (joint)",
                outpath=OUTDIR / f"forest_{sheet}_current.tif",
                xlim=(0.75, 2.25) if sheet == "open_flowers" else (0.9, 1.25)
            )
    except Exception:
        pass

print("\nDone.")
print("Saved outputs in:", OUTDIR)
