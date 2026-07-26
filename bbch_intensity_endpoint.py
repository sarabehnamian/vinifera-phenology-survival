# RE3_R1C8_bbch_intensity_endpoint.py
"""
Reviewer 1, Comment R1.8: BBCH mapping
======================================
First presence ("one or more open, fresh flowers") precedes BBCH 65 (50% capfall).
USA-NPN intensity category 50 asks what percentage of all fresh flowers are open,
and category 58 what percentage of all fruits are ripe -- so a BBCH-equivalent
endpoint can be built as the first visit reaching >= 50% intensity.

This script builds that endpoint, compares it with the published first-presence
endpoint, and refits the primary univariate fixed-window interval-censored AFT
models on it. Window/coverage/z-scoring identical to 12_bivariate_fixed_window.py.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from lifelines import WeibullAFTFitter, LogLogisticAFTFitter
import warnings
warnings.filterwarnings("ignore")

PROJECT = Path("/mnt/project")
OBS = PROJECT / "status_intensity_observation_data.csv"
WX_FILE = PROJECT / "site_daily_weather.xlsx"

HI = {"50-74%", "75-94%", "95% or more"}
ENDPOINTS = {
    "Open flowers": dict(window_end=120, drop_r_lt=None),
    "Ripe fruits":  dict(window_end=180, drop_r_lt=120.0),
}
MIN_COVERAGE = 0.70
TBASE = 10.0


def build_intervals(obs, phenophase, mode):
    """mode='presence' -> published endpoint; mode='bbch' -> first visit >=50%."""
    s = obs[obs["Phenophase_Name"] == phenophase].copy()
    s = s[s["Phenophase_Status"] != -1]
    if mode == "presence":
        s["hit"] = s["Phenophase_Status"] == 1
    else:
        s["hit"] = s["Intensity_Value"].isin(HI)
    rows = []
    for (ind, site, yr), g in s.groupby(["Individual_ID", "Site_ID", "year"]):
        g = g.sort_values("Day_of_Year")
        hits = g[g["hit"]]
        if len(hits):
            R = float(hits["Day_of_Year"].iloc[0])
            before = g[g["Day_of_Year"] < R]
            L = float(before["Day_of_Year"].iloc[-1]) if len(before) else R - 1.0
        else:
            R = np.nan
            L = float(g["Day_of_Year"].iloc[-1])
        rows.append(dict(individual_id=ind, site_id=str(site), year=int(yr),
                         L=L, R=R, event=int(np.isfinite(R))))
    return pd.DataFrame(rows)


def sanitise(df, drop_r_lt):
    df = df.copy()
    df["l"] = pd.to_numeric(df["L"], errors="coerce").clip(1, 366)
    df["r"] = pd.to_numeric(df["R"], errors="coerce").clip(1, 366)
    if drop_r_lt is not None:
        df = df[~(df["r"].notna() & (df["r"] < drop_r_lt))].copy()
    df["r_filled"] = df["r"].where(df["r"].notna(), np.inf)
    fin = np.isfinite(df["r_filled"])
    bad = fin & (df["r_filled"] <= df["l"])
    df.loc[bad, "r_filled"] = df.loc[bad, "l"] + 1e-6
    return df


def window_gdd(wx, end_doy):
    sub = wx[wx["doy"].between(1, end_doy)].copy()
    sub["gdd"] = np.maximum((sub["tmax_c"] + sub["tmin_c"]) / 2.0 - TBASE, 0.0)
    g = (sub.groupby(["site_id", "year"], sort=False)
            .agg(gdd_pre=("gdd", "sum"), days_present=("gdd", "size")).reset_index())
    g["coverage"] = g["days_present"] / float(end_doy)
    return g


def zscore(x):
    x = pd.to_numeric(x, errors="coerce")
    sd = np.nanstd(x, ddof=0)
    return pd.Series(0.0, index=x.index) if (not np.isfinite(sd) or sd == 0) \
        else (x - np.nanmean(x)) / sd


def fit(design):
    out = {}
    for name, F in [("Weibull", WeibullAFTFitter), ("LogLogistic", LogLogisticAFTFitter)]:
        m = F()
        m.fit_interval_censoring(design, lower_bound_col="l", upper_bound_col="r_filled")
        row = m.summary.reset_index().query("covariate=='gdd_pre_z'").iloc[0]
        b, se = float(row["coef"]), float(row["se(coef)"])
        out[name] = (np.exp(b), np.exp(b - 1.96 * se), np.exp(b + 1.96 * se))
    return out


def main():
    obs = pd.read_csv(OBS, low_memory=False)
    obs["year"] = pd.to_datetime(obs["Observation_Date"]).dt.year

    wx = pd.read_excel(WX_FILE, sheet_name="daily_weather")
    wx.columns = wx.columns.str.strip().str.lower().str.replace(" ", "_")
    wx["site_id"] = wx["site_id"].astype(str)
    wx["date"] = pd.to_datetime(wx["date"])
    wx["year"] = wx["date"].dt.year
    wx["doy"] = wx["date"].dt.dayofyear

    rows = []
    for ph, cfg in ENDPOINTS.items():
        feats = window_gdd(wx, cfg["window_end"])
        for mode, label in [("presence", "First presence (published)"),
                            ("bbch", "BBCH-equivalent (>=50% intensity)")]:
            iv = sanitise(build_intervals(obs, ph, mode), cfg["drop_r_lt"])
            df = iv.merge(feats, on=["site_id", "year"], how="left")
            df = df[pd.to_numeric(df["coverage"], errors="coerce") >= MIN_COVERAGE]
            df = df.dropna(subset=["l", "gdd_pre"]).copy()
            df["gdd_pre_z"] = zscore(df["gdd_pre"])
            design = df[["l", "r_filled", "gdd_pre_z"]].dropna()
            n_ev = int(np.isfinite(design["r_filled"]).sum())
            f = fit(design)
            rows.append(dict(
                endpoint=ph, definition=label, n=len(design), events=n_ev,
                censor_pct=round(100 * (1 - n_ev / len(design)), 1),
                median_R=round(float(np.nanmedian(df.loc[np.isfinite(df["r_filled"]), "r_filled"])), 1),
                Weib_TR=round(f["Weibull"][0], 3),
                Weib_CI=f"{f['Weibull'][1]:.3f}-{f['Weibull'][2]:.3f}",
                Weib_sig="yes" if f["Weibull"][2] < 1 else "no",
                LL_TR=round(f["LogLogistic"][0], 3),
                LL_CI=f"{f['LogLogistic'][1]:.3f}-{f['LogLogistic'][2]:.3f}",
                LL_sig="yes" if f["LogLogistic"][2] < 1 else "no"))

    res = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(res.to_string(index=False))
    res.to_excel("/home/claude/RE3_R1C8_bbch_endpoint.xlsx", index=False)


if __name__ == "__main__":
    main()
