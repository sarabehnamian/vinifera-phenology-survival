# RE3_R1C7_homogeneous_subset.py
"""
Reviewer 1, Comment R1.7: cultivar / plant-type heterogeneity
=============================================================
"The plants cover wine grapes, table grapes, and ornamental plantings, all pooled
 as a single species group... this heterogeneity likely contributes to the
 non-significant flowering results through unmodeled variation in thermal
 requirements across plants."

Test: refit the primary univariate fixed-window models on the homogeneous subset
(Individuals 63013/63014/63015 at Site 16610 -- three wine grapes, same site,
same management, same weather), which removes plant-type, management and site
heterogeneity simultaneously. If the flowering effect stays non-significant
there, heterogeneity is not the explanation; power is.

Methodology matches 12_bivariate_fixed_window.py exactly.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from lifelines import WeibullAFTFitter, LogLogisticAFTFitter
import warnings
warnings.filterwarnings("ignore")

PROJECT = Path("/mnt/project")
WX_FILE = PROJECT / "site_daily_weather.xlsx"
SURV_FILE = PROJECT / "survival_intervals_ready.xlsx"

HOMOG_INDIVIDUALS = [63013, 63014, 63015]   # wine grapes, Site 16610
ENDPOINTS = {
    "open_flowers": dict(window_end=120, drop_r_lt=None, label="Open flowers"),
    "ripe_fruits":  dict(window_end=180, drop_r_lt=120.0, label="Ripe fruits"),
}
MIN_COVERAGE = 0.70
TBASE = 10.0


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


def window_gdd(wx, end_doy, tbase=TBASE):
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
    out = {}
    for mname, Fitter in [("Weibull", WeibullAFTFitter),
                          ("LogLogistic", LogLogisticAFTFitter)]:
        m = Fitter()
        m.fit_interval_censoring(design, lower_bound_col="l", upper_bound_col="r_filled")
        s = m.summary.reset_index()
        row = s[s["covariate"] == "gdd_pre_z"].iloc[0]
        beta, se = float(row["coef"]), float(row["se(coef)"])
        out[mname] = dict(tr=float(np.exp(beta)),
                          lo=float(np.exp(beta - 1.96 * se)),
                          hi=float(np.exp(beta + 1.96 * se)))
    return out


def main():
    wx = load_weather()
    rows = []
    for sheet, cfg in ENDPOINTS.items():
        surv_full = load_survival(sheet, cfg["drop_r_lt"])
        idcol = "individual_id" if "individual_id" in surv_full.columns else None
        feats = window_gdd(wx, cfg["window_end"])

        for label, surv in [("Full sample", surv_full),
                            ("Homogeneous subset",
                             surv_full[surv_full[idcol].isin(HOMOG_INDIVIDUALS)]
                             if idcol else None)]:
            if surv is None:
                print("!! individual_id column not found; columns =",
                      list(surv_full.columns))
                return
            df = surv.merge(feats, on=["site_id", "year"], how="left")
            df = df[pd.to_numeric(df["coverage"], errors="coerce") >= MIN_COVERAGE].copy()
            df = df.dropna(subset=["l", "gdd_pre"]).copy()
            df["gdd_pre_z"] = zscore(df["gdd_pre"])
            design = df[["l", "r_filled", "gdd_pre_z"]].dropna().copy()
            n_ev = int(np.isfinite(design["r_filled"]).sum())
            fits = fit_one(design)
            for mname, f in fits.items():
                rows.append(dict(endpoint=cfg["label"], sample=label, model=mname,
                                 n=len(design), events=n_ev,
                                 censor_pct=round(100 * (1 - n_ev / len(design)), 1),
                                 TR=round(f["tr"], 3),
                                 CI_low=round(f["lo"], 3), CI_high=round(f["hi"], 3),
                                 significant="yes" if f["hi"] < 1.0 else "no"))

    res = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(res.to_string(index=False))
    res.to_excel("/home/claude/RE3_R1C7_homogeneous_subset.xlsx", index=False)


if __name__ == "__main__":
    main()
