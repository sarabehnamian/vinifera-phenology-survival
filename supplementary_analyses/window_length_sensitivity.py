"""
window_length_sensitivity.py

Sensitivity of the fixed-window results to the LENGTH of the pre-season window.

Every other element of the pipeline is held fixed: the same >=70% coverage
filter, the same interval-bound handling, the same within-set z-scoring, and the
same univariate interval-censored AFT specification. At the published windows
(DOY 120 for Open flowers, DOY 180 for Ripe fruits) this reproduces the
published time ratios and AIC values exactly.

Inputs (relative to the project root):
    02_fetch_nasa_power_weather/site_daily_weather.xlsx   (sheet: daily_weather)
    07_validate_sensitivity/survival_with_weather_clean.xlsx
        (sheets: open_flowers, ripe_fruits)

Outputs (written to the project root, alongside the other sensitivity outputs):
    window_length_sensitivity.xlsx
    RE3_R2C_window_sensitivity.png

Run from anywhere:
    python supplementary_analyses/window_length_sensitivity.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lifelines import WeibullAFTFitter, LogLogisticAFTFitter

# ---------------------------------------------------------------- paths
# This file lives in supplementary_analyses/, so the project root is two up.
PROJECT = Path(__file__).resolve().parent.parent

WEATHER = PROJECT / "02_fetch_nasa_power_weather" / "site_daily_weather.xlsx"
SURVIVAL = PROJECT / "07_validate_sensitivity" / "survival_with_weather_clean.xlsx"

OUT_DIR = PROJECT
OUT_XLSX = OUT_DIR / "window_length_sensitivity.xlsx"
OUT_FIG = OUT_DIR / "RE3_R2C_window_sensitivity.png"

# ---------------------------------------------------------------- settings
MIN_COVERAGE = 0.70

GRID = {
    "open_flowers": dict(label="Open flowers", ends=[90, 105, 120, 135], primary=120),
    "ripe_fruits": dict(label="Ripe fruits", ends=[150, 165, 180, 195], primary=180),
}


def norm(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names to lowercase with underscores."""
    df = df.copy()
    df.columns = (df.columns.str.strip().str.lower()
                  .str.replace(" ", "_").str.replace("-", "_"))
    return df


def zscore(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    mu, sd = np.nanmean(x), np.nanstd(x, ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(0.0, index=x.index)
    return (x - mu) / sd


def load_weather() -> pd.DataFrame:
    if not WEATHER.exists():
        raise SystemExit(f"Weather file not found: {WEATHER}")
    wx = norm(pd.read_excel(WEATHER, sheet_name="daily_weather"))
    need = {"site_id", "date", "gdd_base10", "prcp_mm"}
    missing = need - set(wx.columns)
    if missing:
        raise SystemExit(f"[weather] missing columns: {missing}")
    wx["site_id"] = wx["site_id"].astype(str)
    wx["date"] = pd.to_datetime(wx["date"], errors="coerce")
    wx = wx.dropna(subset=["site_id", "date"]).copy()
    wx["year"] = wx["date"].dt.year.astype(int)
    wx["doy"] = wx["date"].dt.dayofyear.astype(int)
    return wx


def build_window(wx: pd.DataFrame, end_doy: int) -> pd.DataFrame:
    """Aggregate DOY 1..end_doy per site-year (same construction as step 10)."""
    rows = []
    for (sid, yr), grp in wx.groupby(["site_id", "year"], sort=False):
        sub = grp[grp["doy"].between(1, end_doy)]
        if sub.empty:
            continue
        rows.append(dict(
            site_id=str(sid),
            year=int(yr),
            gdd_pre=float(pd.to_numeric(sub["gdd_base10"], errors="coerce").sum()),
            prcp_pre=float(pd.to_numeric(sub["prcp_mm"], errors="coerce").sum()),
            days_present=int(len(sub)),
            coverage=float(len(sub) / float(end_doy)),
            window_end_doy=int(end_doy),
        ))
    return pd.DataFrame(rows)


def load_survival(sheet: str) -> pd.DataFrame:
    if not SURVIVAL.exists():
        raise SystemExit(f"Survival file not found: {SURVIVAL}")
    df = norm(pd.read_excel(SURVIVAL, sheet_name=sheet))
    for need in ("site_id", "year", "l", "r"):
        if need not in df.columns:
            raise SystemExit(f"[{sheet}] missing column '{need}'")
    df["site_id"] = df["site_id"].astype(str)
    df["year"] = df["year"].astype(int)
    df["l"] = pd.to_numeric(df["l"], errors="coerce").clip(1, 366)
    df["r"] = pd.to_numeric(df["r"], errors="coerce").clip(1, 366)
    # right censoring: R = +inf; fix degenerate intervals
    df["r_filled"] = df["r"].where(df["r"].notna(), np.inf)
    bad = np.isfinite(df["r_filled"]) & (df["r_filled"] <= df["l"])
    df.loc[bad, "r_filled"] = df.loc[bad, "l"] + 1e-6
    return df


def fit_one(endpoint: str, surv: pd.DataFrame, feats: pd.DataFrame, end_doy: int) -> dict:
    df = surv.merge(feats, on=["site_id", "year"], how="left")
    df = df[pd.to_numeric(df["coverage"], errors="coerce") >= MIN_COVERAGE].copy()
    df = df.dropna(subset=["l", "gdd_pre"]).copy()
    df["gdd_z"] = zscore(df["gdd_pre"])

    design = df[["l", "r_filled", "gdd_z"]].dropna().copy()
    n = len(design)
    n_censored = int(np.isinf(design["r_filled"]).sum())

    # observed events whose detection falls INSIDE the window: for these the
    # covariate is not strictly antecedent
    events = df[np.isfinite(df["r_filled"])]
    n_inside = int((events["r_filled"] <= end_doy).sum())

    row = dict(
        endpoint=endpoint,
        window_end_doy=end_doy,
        n=n,
        n_events=len(events),
        pct_censored=100.0 * n_censored / n if n else np.nan,
        events_inside_window=n_inside,
        median_gdd_pre=float(np.nanmedian(df["gdd_pre"])) if n else np.nan,
    )

    for name, Fitter in (("weibull", WeibullAFTFitter),
                         ("loglogistic", LogLogisticAFTFitter)):
        try:
            model = Fitter()
            model.fit_interval_censoring(design,
                                         lower_bound_col="l",
                                         upper_bound_col="r_filled")
            summary = model.summary.reset_index()
            mask = summary.iloc[:, 1].astype(str).str.contains("gdd_z", regex=False)
            coef = float(summary.loc[mask, "coef"].iloc[0])
            se = float(summary.loc[mask, "se(coef)"].iloc[0])
            row[f"{name}_TR"] = float(np.exp(coef))
            row[f"{name}_lo"] = float(np.exp(coef - 1.96 * se))
            row[f"{name}_hi"] = float(np.exp(coef + 1.96 * se))
            row[f"{name}_AIC"] = float(model.AIC_)
        except Exception as exc:  # noqa: BLE001 - report and continue
            row[f"{name}_TR"] = np.nan
            row[f"{name}_error"] = str(exc)
    return row


def make_figure(results: pd.DataFrame) -> None:
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
    })

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), dpi=300)
    handles = labels = None

    for ax, (endpoint, cfg) in zip(axes, GRID.items()):
        sub = results[results.endpoint == endpoint].sort_values("window_end_doy")
        x = sub.window_end_doy.values.astype(float)
        step = float(np.diff(x)[0])
        offset = step * 0.14

        for key, name, colour, marker, dx in (
            ("weibull", "Weibull", "#1f77b4", "o", -offset),
            ("loglogistic", "Log-logistic", "#d62728", "s", offset),
        ):
            tr = sub[f"{key}_TR"].values
            lo = sub[f"{key}_lo"].values
            hi = sub[f"{key}_hi"].values
            ax.errorbar(x + dx, tr, yerr=[tr - lo, hi - tr], fmt=marker,
                        color=colour, capsize=3, lw=1.4, ms=6, label=name)

        ax.axhline(1.0, ls="--", lw=1, color="0.45", zorder=0)
        ax.axvline(cfg["primary"], ls=":", lw=1.2, color="0.6", zorder=0)
        ax.set_xticks(x)
        ax.set_xlim(x[0] - step * 0.6, x[-1] + step * 0.6)
        ax.set_ylim(0.62, 1.24)
        ax.set_xlabel("Pre-season window end (DOY)")
        ax.set_title(cfg["label"], pad=26)

        top = ax.secondary_xaxis("top")
        top.set_xticks(x)
        top.set_xticklabels([str(int(v)) for v in sub.events_inside_window.values])
        top.tick_params(length=0, labelsize=9, colors="0.35")
        top.set_xlabel("Events falling inside window", fontsize=8.5,
                       color="0.35", labelpad=4)
        for spine in top.spines.values():
            spine.set_visible(False)

        if handles is None:
            handles, labels = ax.get_legend_handles_labels()

    axes[0].set_ylabel("Time ratio per +1 SD pre-season GDD")
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.015), fontsize=10)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(OUT_FIG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {OUT_FIG}")


def main() -> pd.DataFrame:
    wx = load_weather()

    rows = []
    for endpoint, cfg in GRID.items():
        surv = load_survival(endpoint)
        for end_doy in cfg["ends"]:
            feats = build_window(wx, end_doy)
            row = fit_one(endpoint, surv, feats, end_doy)
            row["is_primary"] = (end_doy == cfg["primary"])
            rows.append(row)
            print(f"{endpoint:14s} end={end_doy:4d}  n={row['n']:3d}  "
                  f"cens={row['pct_censored']:.1f}%  "
                  f"inside={row['events_inside_window']:2d}  "
                  f"W={row.get('weibull_TR', float('nan')):.3f}  "
                  f"LL={row.get('loglogistic_TR', float('nan')):.3f}")

    results = pd.DataFrame(rows)

    readme = [
        "Sensitivity of the univariate fixed-window AFT models to the pre-season window length.",
        "Pre-season GDD (base 10 C) summed over DOY 1..window_end_doy, z-scored within the",
        "analysis set, entered as the sole predictor in interval-censored Weibull and",
        "log-logistic AFT models.",
        f"Coverage filter: observed days / window length >= {MIN_COVERAGE:.2f}.",
        "events_inside_window = observed events whose first detection (R) falls inside the",
        "window; for these units the covariate is not strictly antecedent.",
        "TR is the time ratio per +1 SD; CIs are exp(coef +/- 1.96*SE).",
        "Rows with is_primary = True are the windows used throughout the manuscript.",
    ]

    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        results.to_excel(writer, index=False, sheet_name="window_sensitivity")
        pd.DataFrame({"README": readme}).to_excel(writer, index=False, sheet_name="README")
    print(f"Results saved: {OUT_XLSX}")

    make_figure(results)
    return results


if __name__ == "__main__":
    main()
