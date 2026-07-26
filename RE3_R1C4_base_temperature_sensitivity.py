# RE3_R1C4_base_temperature_sensitivity.py
"""
Reviewer 1, Comment R1.4: base temperature choice
=================================================
"The 10 C base temperature is presented as the established standard for V. vinifera...
 This matters for the pipeline specifically: the base temperature choice directly
 affects the GDD values feeding the model."

This script recomputes pre-season GDD at Tb = 5, 8, 10, 12 C, refits the primary
univariate fixed-window interval-censored AFT models, and reports how the time
ratio changes. Methodology matches 12_bivariate_fixed_window.py exactly:
  windows DOY 1-120 (Open flowers) / 1-180 (Ripe fruits)
  coverage >= 0.70, bounds clipped to [1,366], R=inf for right-censoring
  GDD_pre z-scored within the analysis set
  Ripe fruits: R < 120 excluded (prior-season carry-over)

Outputs: RE3_R1C4_results/base_temperature_sensitivity.xlsx
         RE3_R1C4_results/RE3_R1C4_tb_sensitivity.png
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lifelines import WeibullAFTFitter, LogLogisticAFTFitter

import warnings
warnings.filterwarnings("ignore")

PROJECT = Path(__file__).resolve().parent
WX_FILE = PROJECT / "02_fetch_nasa_power_weather" / "site_daily_weather.xlsx"
SURV_FILE = PROJECT / "00_npn_survival_analysis" / "survival_intervals_ready.xlsx"
OUT = PROJECT / "RE3_R1C4_results"
OUT.mkdir(parents=True, exist_ok=True)

# fall back to flat project root (useful when files sit beside the script)
if not WX_FILE.exists():
    WX_FILE = PROJECT / "site_daily_weather.xlsx"
if not SURV_FILE.exists():
    SURV_FILE = PROJECT / "survival_intervals_ready.xlsx"

BASE_TEMPS = [5.0, 8.0, 10.0, 12.0]
ENDPOINTS = {
    "open_flowers": dict(window_end=120, drop_r_lt=None,  label="Open flowers"),
    "ripe_fruits":  dict(window_end=180, drop_r_lt=120.0, label="Ripe fruits"),
}
MIN_COVERAGE = 0.70


def norm(df):
    df = df.copy()
    df.columns = (df.columns.str.strip().str.lower()
                  .str.replace(" ", "_").str.replace("-", "_"))
    return df


def zscore(x):
    x = pd.to_numeric(x, errors="coerce")
    sd = np.nanstd(x, ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(0.0, index=x.index)
    return (x - np.nanmean(x)) / sd


def load_weather():
    wx = norm(pd.read_excel(WX_FILE, sheet_name="daily_weather"))
    wx["site_id"] = wx["site_id"].astype(str)
    wx["date"] = pd.to_datetime(wx["date"], errors="coerce")
    wx = wx.dropna(subset=["site_id", "date"]).copy()
    wx["year"] = wx["date"].dt.year.astype(int)
    wx["doy"] = wx["date"].dt.dayofyear.astype(int)
    wx["tmin_c"] = pd.to_numeric(wx["tmin_c"], errors="coerce")
    wx["tmax_c"] = pd.to_numeric(wx["tmax_c"], errors="coerce")
    return wx


def window_gdd(wx, end_doy, tbase):
    """Pre-season GDD at an arbitrary base temperature, per site-year."""
    sub = wx[wx["doy"].between(1, end_doy)].copy()
    sub["gdd"] = np.maximum((sub["tmax_c"] + sub["tmin_c"]) / 2.0 - tbase, 0.0)
    g = (sub.groupby(["site_id", "year"], sort=False)
            .agg(gdd_pre=("gdd", "sum"), days_present=("gdd", "size"))
            .reset_index())
    g["coverage"] = g["days_present"] / float(end_doy)
    return g


def load_survival(sheet, drop_r_lt=None):
    df = norm(pd.read_excel(SURV_FILE, sheet_name=sheet))
    df["site_id"] = df["site_id"].astype(str)
    df["year"] = df["year"].astype(int)
    df["l"] = pd.to_numeric(df["l"], errors="coerce").clip(1, 366)
    df["r"] = pd.to_numeric(df["r"], errors="coerce").clip(1, 366)
    if drop_r_lt is not None:
        df = df[~(df["r"].notna() & (df["r"] < drop_r_lt))].copy()
    df["r_filled"] = df["r"].where(df["r"].notna(), np.inf)
    finite = np.isfinite(df["r_filled"])
    bad = finite & (df["r_filled"] <= df["l"])
    df.loc[bad, "r_filled"] = df.loc[bad, "l"] + 1e-6
    return df


def fit_one(design):
    """Univariate interval-censored AFT; returns TR, CI, AIC for both families."""
    out = {}
    for mname, Fitter in [("Weibull", WeibullAFTFitter),
                          ("LogLogistic", LogLogisticAFTFitter)]:
        m = Fitter()
        m.fit_interval_censoring(design, lower_bound_col="l", upper_bound_col="r_filled")
        s = m.summary.reset_index()
        row = s[s["covariate"] == "gdd_pre_z"].iloc[0]
        beta, se = float(row["coef"]), float(row["se(coef)"])
        out[mname] = dict(
            tr=float(np.exp(beta)),
            lo=float(np.exp(beta - 1.96 * se)),
            hi=float(np.exp(beta + 1.96 * se)),
            aic=float(m.AIC_),
        )
    return out


def main():
    wx = load_weather()
    rows = []

    for sheet, cfg in ENDPOINTS.items():
        surv = load_survival(sheet, cfg["drop_r_lt"])
        for tb in BASE_TEMPS:
            feats = window_gdd(wx, cfg["window_end"], tb)
            df = surv.merge(feats, on=["site_id", "year"], how="left")
            df = df[pd.to_numeric(df["coverage"], errors="coerce") >= MIN_COVERAGE].copy()
            df = df.dropna(subset=["l", "gdd_pre"]).copy()
            df["gdd_pre_z"] = zscore(df["gdd_pre"])
            design = df[["l", "r_filled", "gdd_pre_z"]].dropna().copy()

            fits = fit_one(design)
            for mname, f in fits.items():
                rows.append(dict(
                    endpoint=cfg["label"], base_temp_C=tb, model=mname,
                    n=len(design),
                    median_gdd_pre=float(np.median(df["gdd_pre"])),
                    TR=round(f["tr"], 3),
                    CI_low=round(f["lo"], 3), CI_high=round(f["hi"], 3),
                    AIC=round(f["aic"], 1),
                    significant="yes" if f["hi"] < 1.0 else "no",
                ))
            print(f"  {cfg['label']:<13} Tb={tb:>4.1f}  n={len(design):>3}  "
                  f"medGDD={np.median(df['gdd_pre']):>7.1f}  "
                  f"Weib TR={fits['Weibull']['tr']:.3f}  "
                  f"LogLog TR={fits['LogLogistic']['tr']:.3f}")

    res = pd.DataFrame(rows)

    # ---- stability summary: spread of TR across base temperatures ----
    summ = (res.groupby(["endpoint", "model"])
               .agg(TR_min=("TR", "min"), TR_max=("TR", "max"),
                    TR_at_Tb10=("TR", lambda s: np.nan))
               .reset_index())
    tb10 = res[res["base_temp_C"] == 10.0].set_index(["endpoint", "model"])["TR"]
    summ["TR_at_Tb10"] = summ.set_index(["endpoint", "model"]).index.map(tb10)
    summ["TR_range"] = (summ["TR_max"] - summ["TR_min"]).round(3)
    summ["max_pct_dev_from_Tb10"] = (
        100 * np.maximum((summ["TR_max"] - summ["TR_at_Tb10"]).abs(),
                         (summ["TR_min"] - summ["TR_at_Tb10"]).abs())
        / summ["TR_at_Tb10"]).round(1)

    xlsx = OUT / "base_temperature_sensitivity.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as xw:
        res.to_excel(xw, index=False, sheet_name="tb_sensitivity")
        summ.to_excel(xw, index=False, sheet_name="stability_summary")
    print(f"\nSaved {xlsx}")

    # ---- figure ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=False)
    handles = None
    for ax, ep in zip(axes, res["endpoint"].unique()):
        sub = res[res["endpoint"] == ep]
        for mname, colr, off in [("Weibull", "#C0504D", -0.18),
                                 ("Log-logistic", "#4F81BD", 0.18)]:
            key = "Weibull" if mname == "Weibull" else "LogLogistic"
            d = sub[sub["model"] == key]
            ax.errorbar(d["base_temp_C"] + off, d["TR"],
                        yerr=[d["TR"] - d["CI_low"], d["CI_high"] - d["TR"]],
                        fmt="o", ms=5, capsize=3, lw=1.4, color=colr, label=mname)
        ax.axhline(1.0, ls="--", lw=1, color="grey", zorder=0)
        ax.axvline(10.0, ls=":", lw=1, color="black", zorder=0)
        ax.set_xticks(BASE_TEMPS)
        ax.set_xlim(3.6, 13.4)
        ax.set_xlabel("Base temperature $T_b$ (°C)")
        ax.set_ylabel("Time ratio (per +1 SD pre-season GDD)")
        ax.set_title(ep, fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()
    # one shared legend, outside the data area
    fig.legend(handles, labels, loc="lower center", ncol=2,
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    png = OUT / "RE3_R1C4_tb_sensitivity.png"
    fig.savefig(png, dpi=300)
    print(f"Saved {png}")

    print("\n--- stability summary ---")
    print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
