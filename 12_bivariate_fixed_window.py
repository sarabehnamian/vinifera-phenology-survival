# 12_bivariate_fixed_window.py  (folder-aware + robust)
# Bivariate interval-censored AFT using pre-season features: GDD_pre + Prcp_pre
#
# Inputs (preferred → fallback):
#   10_refit_simple_models/fixed_window_features.xlsx
#   09_diagnostics_and_refit/fixed_window_features.xlsx
#   (else build from) 02_fetch_nasa_power_weather/site_daily_weather.xlsx  (sheet: daily_weather)
#   Survival: 07_validate_sensitivity/survival_with_weather_clean.xlsx → 03_merge_survival_with_weather/survival_with_weather.xlsx
#
# Outputs:
#   12_bivariate_fixed_window/interval_model_results_bivariate.xlsx
#   survival_analysis_results/interval_model_results_bivariate.xlsx (copy)
#   12_bivariate_fixed_window/fixed_window_features.xlsx (if built)
#   forest_*_bivariate.tif (TIFF 300 dpi)

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

STEP02 = PROJECT / "02_fetch_nasa_power_weather" / "site_daily_weather.xlsx"
STEP07 = PROJECT / "07_validate_sensitivity" / "survival_with_weather_clean.xlsx"
STEP03 = PROJECT / "03_merge_survival_with_weather" / "survival_with_weather.xlsx"
STEP10 = PROJECT / "10_refit_simple_models" / "fixed_window_features.xlsx"
STEP09 = PROJECT / "09_diagnostics_and_refit" / "fixed_window_features.xlsx"

OUT_DIR = PROJECT / "12_bivariate_fixed_window"
OUT_DIR.mkdir(parents=True, exist_ok=True)

AGG = PROJECT / "survival_analysis_results"
AGG.mkdir(parents=True, exist_ok=True)

OUT_MAIN = OUT_DIR / "interval_model_results_bivariate.xlsx"
OUT_COPY = AGG / "interval_model_results_bivariate.xlsx"

def norm(df):
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

# ---------- daily weather ----------
def load_daily_weather() -> pd.DataFrame:
    if not STEP02.exists():
        raise FileNotFoundError(f"Weather file not found: {STEP02}")
    wx = pd.read_excel(STEP02, sheet_name="daily_weather")
    wx = norm(wx)
    need = {"site_id","date","gdd_base10","tmin_c","tmax_c","prcp_mm"}
    miss = need - set(wx.columns)
    if miss:
        raise SystemExit(f"[weather] missing columns: {miss}")
    wx["site_id"] = wx["site_id"].astype(str)
    wx["date"] = pd.to_datetime(wx["date"], errors="coerce")
    wx = wx.dropna(subset=["site_id","date"]).copy()
    wx["year"] = wx["date"].dt.year.astype(int)
    wx["doy"]  = wx["date"].dt.dayofyear.astype(int)
    return wx

def build_fixed_windows(wx: pd.DataFrame, flowers_end=120, fruits_end=180):
    def agg(df, end_doy):
        sub = df[df["doy"].between(1, end_doy)].copy()
        if sub.empty: return None
        return dict(
            gdd_pre=float(pd.to_numeric(sub["gdd_base10"], errors="coerce").sum()),
            prcp_pre=float(pd.to_numeric(sub["prcp_mm"], errors="coerce").sum()),
            tmin_pre_mean=float(pd.to_numeric(sub["tmin_c"], errors="coerce").mean()),
            tmax_pre_mean=float(pd.to_numeric(sub["tmax_c"], errors="coerce").mean()),
            frost_pre=int((pd.to_numeric(sub["tmin_c"], errors="coerce") <= 0).sum()),
            heat_pre=int((pd.to_numeric(sub["tmax_c"], errors="coerce") >= 35).sum()),
            days_present=int(len(sub)),
            coverage=float(len(sub)/float(end_doy)),
            window_end_doy=int(end_doy),
        )
    rows_f, rows_r = [], []
    for (sid, yr), g in wx.groupby(["site_id","year"], sort=False):
        f = agg(g, flowers_end);  r = agg(g, fruits_end)
        if f: rows_f.append({"site_id":str(sid), "year":int(yr), **f})
        if r: rows_r.append({"site_id":str(sid), "year":int(yr), **r})
    flowers = norm(pd.DataFrame(rows_f))
    fruits  = norm(pd.DataFrame(rows_r))
    # save in our OUT_DIR and hand a copy back into the project root for reuse
    fixed_here = OUT_DIR / "fixed_window_features.xlsx"
    with pd.ExcelWriter(fixed_here, engine="openpyxl") as xw:
        if not flowers.empty: flowers.to_excel(xw, index=False, sheet_name="flowers_window")
        if not fruits.empty:  fruits.to_excel(xw,  index=False, sheet_name="fruits_window")
    print(f"  Saved fixed windows: {fixed_here}")
    return flowers, fruits

def _prefix_if_needed(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    Accepts either unprefixed columns (gdd_pre, prcp_pre, coverage, …)
    or already-prefixed (flowers_gdd_pre, fruits_prcp_pre, …).
    Returns a frame with prefixed names (e.g., flowers_gdd_pre, …) plus site_id/year.
    Also handles step-09 variant that used days_in_window/window_end_doy.
    """
    df = norm(df)
    df["site_id"] = df["site_id"].astype(str)
    df["year"]    = df["year"].astype(int)

    # if already prefixed, keep as-is
    if f"{prefix}_gdd_pre" in df.columns and f"{prefix}_prcp_pre" in df.columns:
        return df

    # build a mapping from unprefixed to prefixed
    base = {
        "gdd_pre":"gdd_pre",
        "prcp_pre":"prcp_pre",
        "tmin_pre_mean":"tmin_pre_mean",
        "tmax_pre_mean":"tmax_pre_mean",
        "frost_pre":"frost_pre",
        "heat_pre":"heat_pre",
        "coverage":"coverage",
        "days_present":"days_present",
        "days_in_window":"days_present",   # step-09 name
        "window_end_doy":"window_end_doy",
    }
    ren = {}
    for k, v in base.items():
        if k in df.columns:
            ren[k] = f"{prefix}_{v}"
    out = df.rename(columns=ren)

    # derive coverage if missing but we have days/window_end
    cov = f"{prefix}_coverage"
    if cov not in out.columns:
        if f"{prefix}_days_present" in out.columns and f"{prefix}_window_end_doy" in out.columns:
            dp = pd.to_numeric(out[f"{prefix}_days_present"], errors="coerce")
            wd = pd.to_numeric(out[f"{prefix}_window_end_doy"], errors="coerce").replace(0, np.nan)
            out[cov] = (dp / wd).clip(upper=1.0)

    return out

def load_fixed_windows_or_build(wx: pd.DataFrame):
    # candidates: step10 → step09 → build from daily wx
    if STEP10.exists():
        print(f"Using fixed windows from: {STEP10}")
        try:
            fl = pd.read_excel(STEP10, sheet_name="flowers_window")
            fr = pd.read_excel(STEP10, sheet_name="fruits_window")
        except Exception:
            print("  step-10 file incomplete; trying step-09…")
            fl = fr = None
    else:
        fl = fr = None

    if fl is None or fr is None:
        if STEP09.exists():
            print(f"Using fixed windows from: {STEP09}")
            try:
                # step-09 saved merged per-sheet; we only need site_id/year and window features
                fl = pd.read_excel(STEP09, sheet_name="open_flowers")
                fr = pd.read_excel(STEP09, sheet_name="ripe_fruits")
                # keep just keys + antecedent features if present
                fl = fl[[c for c in fl.columns if c.lower() in
                         {"site_id","year","gdd_pre","prcp_pre","tmin_pre_mean","tmax_pre_mean",
                          "frost_pre_days","heat_pre_days","days_in_window","window_end_doy","coverage"}]]
                # step-09 used *_pre_days names
                fl = fl.rename(columns={"frost_pre_days":"frost_pre","heat_pre_days":"heat_pre"})
                fr = fr[[c for c in fr.columns if c.lower() in
                         {"site_id","year","gdd_pre","prcp_pre","tmin_pre_mean","tmax_pre_mean",
                          "frost_pre_days","heat_pre_days","days_in_window","window_end_doy","coverage"}]]
                fr = fr.rename(columns={"frost_pre_days":"frost_pre","heat_pre_days":"heat_pre"})
            except Exception:
                print("  step-09 file not usable; will build from daily weather.")
                fl = fr = None

    if fl is None or fr is None:
        print("Building fixed-window features from daily weather…")
        return build_fixed_windows(wx)

    # harmonize to prefixed names
    fl = _prefix_if_needed(norm(fl), "flowers")
    fr = _prefix_if_needed(norm(fr), "fruits")
    return fl, fr

# ---------- survival ----------
def load_surv(sheet: str) -> pd.DataFrame:
    src = STEP07 if STEP07.exists() else STEP03
    df = norm(pd.read_excel(src, sheet_name=sheet))
    for need in ["site_id","year","l","r","event","first_obs_doy","last_obs_doy"]:
        if need not in df.columns:
            raise SystemExit(f"[{sheet}] missing '{need}' in {src.name}")
    df["site_id"] = df["site_id"].astype(str)
    df["year"]    = df["year"].astype(int)
    df["l"] = pd.to_numeric(df["l"], errors="coerce").clip(1, 366)
    df["r"] = pd.to_numeric(df["r"], errors="coerce").clip(1, 366)
    df["r_filled"] = df["r"].where(df["r"].notna(), np.inf)
    finite = np.isfinite(df["r_filled"])
    df.loc[finite, "r_filled"] = df.loc[finite, "r_filled"].clip(1, 366)
    bad = finite & (df["r_filled"] <= df["l"])
    df.loc[bad, "r_filled"] = df.loc[bad, "l"] + 1e-6
    return df

# ---------- modeling ----------
def fit_bivariate(name: str, surv: pd.DataFrame, feats: pd.DataFrame,
                  window_prefix="flowers", min_cov=0.70) -> pd.DataFrame:
    from lifelines import WeibullAFTFitter, LogLogisticAFTFitter

    # keys & merge
    for col, caster in [("site_id", str), ("year", int)]:
        if col in surv.columns:  surv[col]  = surv[col].map(caster)
        if col in feats.columns: feats[col] = feats[col].map(caster)
    df = surv.merge(feats, on=["site_id","year"], how="left")

    covcol = f"{window_prefix}_coverage"
    if covcol in df.columns:
        before = len(df)
        df = df[pd.to_numeric(df[covcol], errors="coerce") >= float(min_cov)].copy()
        print(f"  {name}: kept {len(df)}/{before} rows with coverage ≥ {int(min_cov*100)}%")

    gdd = f"{window_prefix}_gdd_pre"
    prc = f"{window_prefix}_prcp_pre"
    for c in [gdd, prc]:
        if c not in df.columns:
            raise SystemExit(f"[{name}] missing feature column: {c}")

    df = df.dropna(subset=["l", gdd, prc]).copy()
    df[gdd+"_z"] = zscore(df[gdd])
    df[prc+"_z"] = zscore(df[prc])

    Xcols = [gdd+"_z", prc+"_z"]
    design = df[["l","r_filled"] + Xcols].dropna().copy()
    if len(design) < 10:
        print(f"  {name}: <10 rows after cleaning — skipping fits.")
        return pd.DataFrame([{"sheet":name,"model":"WeibullAFT","n_rows":len(design)},
                             {"sheet":name,"model":"LogLogisticAFT","n_rows":len(design)}])

    # write diagnostics
    dbg = OUT_DIR / f"debug_bivariate_{name}.xlsx"
    with pd.ExcelWriter(dbg, engine="openpyxl") as xw:
        design.to_excel(xw, index=False, sheet_name="fit_data")
        pd.DataFrame({"col":design.columns,
                      "n_na":[design[c].isna().sum() for c in design.columns]}).to_excel(xw, index=False, sheet_name="na_check")
    print(f"  [diag] wrote {dbg.name}")

    results = []

    def summarize_fit(model_name, m):
        # TR via medians at z=0 vs +1 for each covariate
        base = {c:0.0 for c in Xcols}
        row = {"sheet": name, "model": model_name, "n_rows": len(design)}
        for x in Xcols:
            med0 = float(m.predict_median(pd.DataFrame([base])).iloc[0])
            alt = base.copy(); alt[x] = 1.0
            med1 = float(m.predict_median(pd.DataFrame([alt])).iloc[0])
            row[f"{x}_TR(+1SD)"] = med1/med0

        # Extract coef-based CIs if available
        try:
            summ = m.summary.reset_index().rename(columns={"index":"param"})
            for x in Xcols:
                mask = summ["param"].astype(str).str.contains(x, regex=False)
                if mask.any():
                    coef = float(summ.loc[mask, "coef"].iloc[0])
                    se   = float(summ.loc[mask, "se(coef)"].iloc[0])
                    row[f"{x}_CI_lo"] = float(np.exp(coef - 1.96*se))
                    row[f"{x}_CI_hi"] = float(np.exp(coef + 1.96*se))
        except Exception:
            pass
        return row

    def forest_plot(row, title_stem):
        # two points with CIs if present
        vals = [row.get(f"{Xcols[0]}_TR(+1SD)"), row.get(f"{Xcols[1]}_TR(+1SD)")]
        los  = [row.get(f"{Xcols[0]}_CI_lo", np.nan), row.get(f"{Xcols[1]}_CI_lo", np.nan)]
        his  = [row.get(f"{Xcols[0]}_CI_hi", np.nan), row.get(f"{Xcols[1]}_CI_hi", np.nan)]
        labels = ["Pre-season GDD (+1 SD)", "Pre-season precipitation (+1 SD)"]
        y = np.arange(2)[::-1]

        fig, ax = plt.subplots(figsize=(4.8, 3.0), dpi=300)
        for yi, lo, hi in zip(y, los, his):
            if np.isfinite(lo) and np.isfinite(hi):
                ax.hlines(yi, lo, hi, lw=2)
        ax.scatter(vals, y, s=45)
        ax.axvline(1.0, ls="--", lw=1)
        ax.set_yticks(y); ax.set_yticklabels(labels)
        ax.set_xlabel("Time ratio (per +1 SD)")
        ax.set_title(title_stem)
        # dynamic x-lims to keep the point visible; include 1.0
        finite = [v for v in vals+los+his if np.isfinite(v)]
        if finite:
            lo = max(0.5, min(finite)*0.8)
            hi = min(2.5, max(finite)*1.2)
            if lo < hi:
                ax.set_xlim(lo, hi)
        plt.tight_layout()
        fig.savefig(OUT_DIR / f"{title_stem.replace(' • ','_').replace(' ','_')}_bivariate.tif",
                    dpi=300, bbox_inches="tight")
        plt.close(fig)

    for model_name, Fitter in [("WeibullAFT", __import__("lifelines").WeibullAFTFitter),
                               ("LogLogisticAFT", __import__("lifelines").LogLogisticAFTFitter)]:
        try:
            print(f"  Fitting {model_name} with covariates: {', '.join(Xcols)}")
            m = Fitter()
            m.fit_interval_censoring(design, lower_bound_col="l", upper_bound_col="r_filled")
            row = summarize_fit(model_name, m)
            results.append(row)
            forest_plot(row, f"{name} • {model_name}")
        except Exception as e:
            print(f"  {model_name} failed: {e}")
            results.append({"sheet":name,"model":model_name,"n_rows":len(design),
                            f"{Xcols[0]}_TR(+1SD)":np.nan, f"{Xcols[1]}_TR(+1SD)":np.nan,
                            "error":str(e)})

    return pd.DataFrame(results)

def main():
    print("Loading daily weather…")
    wx = load_daily_weather()

    print("Loading/deriving fixed-window features…")
    fl, fr = load_fixed_windows_or_build(wx)

    print("Loading survival sheets…")
    surv_open = load_surv("open_flowers")
    surv_ripe = load_surv("ripe_fruits")

    print("\nPreparing & fitting: open_flowers")
    res_open = fit_bivariate("open_flowers", surv_open, fl, window_prefix="flowers", min_cov=0.70)

    print("\nPreparing & fitting: ripe_fruits")
    res_ripe = fit_bivariate("ripe_fruits",  surv_ripe, fr, window_prefix="fruits",  min_cov=0.70)

    res = pd.concat([res_open, res_ripe], ignore_index=True)

    for out in (OUT_MAIN, OUT_COPY):
        with pd.ExcelWriter(out, engine="openpyxl") as xw:
            res.to_excel(xw, index=False, sheet_name="bivariate_TR(+1SD)")
            pd.DataFrame({"README":[
                "Bivariate AFT (Weibull & Log-Logistic). Covariates: pre-season GDD and precipitation (z-scored).",
                "Right-censoring handled with R=+inf; degenerate intervals adjusted.",
                "TR(+1 SD) reported for each covariate holding the other at its mean (z=0).",
                "CIs are exp(coef ± 1.96*SE) from the AFT time component where available.",
                f"Fixed-window source: {('10/' if STEP10.exists() else ('09/' if STEP09.exists() else 'built from 02/'))}."
            ]}).to_excel(xw, index=False, sheet_name="README")
        print(f"Saved: {out}")

if __name__ == "__main__":
    main()
