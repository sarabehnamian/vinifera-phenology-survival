# RE3_R1C10_water_balance.py
# Reviewer 1, comment 10: cumulative precipitation is a poor proxy for plant
# water status across a CA-to-NY gradient; evapotranspiration and the timing of
# stress are more meaningful.
#
# Adds Hargreaves-Samani reference evapotranspiration (FAO-56 fallback method,
# which needs only Tmax, Tmin and extraterrestrial radiation) and refits the
# bivariate fixed-window models with the climatic water balance P - ET0 in
# place of raw precipitation. GDD and precipitation are taken from the existing
# feature file so the "current" column reproduces the published estimates.

import numpy as np
import pandas as pd
from lifelines import WeibullAFTFitter, LogLogisticAFTFitter

WEATHER = "site_daily_weather.xlsx"
FEATURES = "fixed_window_features.xlsx"
SURVIVAL = "survival_intervals_ready.xlsx"
OUT = "RE3_R1C10_water_balance.xlsx"

SITE_LAT = {1074: 39.367615, 6466: 37.109486, 16610: 37.991367,
            17574: 40.863052, 21207: 30.410679}
ENDPOINTS = [("ripe_fruits", "fruits_window", 180),
             ("open_flowers", "flowers_window", 120)]


def hargreaves_et0(df):
    """FAO-56 eq. 21 extraterrestrial radiation + Hargreaves-Samani ET0 (mm/d)."""
    J = df["doy"].to_numpy()
    phi = np.radians(df["lat"].to_numpy())
    dr = 1 + 0.033 * np.cos(2 * np.pi * J / 365.0)
    dec = 0.409 * np.sin(2 * np.pi * J / 365.0 - 1.39)
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(dec), -1, 1))
    Ra = ((24 * 60 / np.pi) * 0.0820 * dr
          * (ws * np.sin(phi) * np.sin(dec)
             + np.cos(phi) * np.cos(dec) * np.sin(ws)))       # MJ m-2 d-1
    tmean = (df["tmax_c"] + df["tmin_c"]) / 2
    trange = np.clip(df["tmax_c"] - df["tmin_c"], 0, None)
    return 0.0023 * (Ra / 2.45) * (tmean + 17.8) * np.sqrt(trange)


def fit(d, cols, label, rows, endpoint, spec):
    d = d.dropna(subset=cols).copy()
    d["L"] = d["L"].clip(1, 366)
    d["R"] = np.where(d["event"] == 1, d["R"].clip(1, 366), np.inf)
    z = []
    for c in cols:
        d["z_" + c] = (d[c] - d[c].mean()) / d[c].std()
        z.append("z_" + c)
    print(f"{label}  (n={len(d)}, events={int(d['event'].sum())})")
    for Fitter, name in [(WeibullAFTFitter, "Weibull"),
                         (LogLogisticAFTFitter, "Log-logistic")]:
        f = Fitter()
        f.fit_interval_censoring(d[["L", "R"] + z], "L", "R")
        out = []
        for c, zc in zip(cols, z):
            s = f.summary.xs(zc, level=1, drop_level=False)
            tr, lo, hi = (np.exp(s["coef"].iloc[0]),
                          np.exp(s["coef lower 95%"].iloc[0]),
                          np.exp(s["coef upper 95%"].iloc[0]))
            sig = "*" if (lo - 1) * (hi - 1) > 0 else ""
            out.append(f"{c} TR={tr:.3f} ({lo:.3f}-{hi:.3f}){sig}")
            rows.append(dict(endpoint=endpoint, spec=spec, model=name,
                             covariate=c, TR=round(tr, 3),
                             CI_low=round(lo, 3), CI_high=round(hi, 3),
                             n=len(d), events=int(d["event"].sum()),
                             AIC=round(f.AIC_, 1)))
        print(f"    {name:<13} " + "  |  ".join(out) + f"   AIC={f.AIC_:.1f}")


w = pd.read_excel(WEATHER, sheet_name="daily_weather")
w["date"] = pd.to_datetime(w["date"])
w["year"] = w["date"].dt.year
w["doy"] = w["date"].dt.dayofyear
w["lat"] = w["site_id"].map(SITE_LAT)
w["et0"] = hargreaves_et0(w)

rows, desc = [], []
for endpoint, sheet, end_doy in ENDPOINTS:
    # ET0 summed over the same fixed window, same >=70% coverage rule
    e = (w[w["doy"] <= end_doy].groupby(["site_id", "year"])
         .agg(et0_pre=("et0", "sum"), days=("doy", "size")).reset_index())
    e = e[e["days"] / end_doy >= 0.70].drop(columns="days")

    feats = pd.read_excel(FEATURES, sheet_name=sheet)
    feats = feats.loc[feats["coverage"] >= 0.70,
                      ["site_id", "year", "gdd_pre", "prcp_pre"]]
    feats = feats.merge(e, on=["site_id", "year"])
    feats["wb_pre"] = feats["prcp_pre"] - feats["et0_pre"]

    sv = pd.read_excel(SURVIVAL, sheet_name=endpoint)
    if endpoint == "ripe_fruits":
        sv = sv[~((sv["event"] == 1) & (sv["R"] < 120))]
    d = sv.merge(feats, on=["site_id", "year"])

    print("\n" + "=" * 76)
    print(f"{endpoint.upper()}   fixed window DOY 1-{end_doy}")
    print(f"  median precipitation {feats['prcp_pre'].median():.0f} mm | "
          f"ET0 {feats['et0_pre'].median():.0f} mm | "
          f"P-ET0 {feats['wb_pre'].median():.0f} mm")
    print(f"  corr(P, P-ET0) = {feats['prcp_pre'].corr(feats['wb_pre']):.3f}")
    print("  median P-ET0 by site (mm): "
          + str(feats.groupby('site_id')['wb_pre'].median().round(0).to_dict()))
    desc.append(dict(endpoint=endpoint, window_end_doy=end_doy,
                     median_prcp=round(feats["prcp_pre"].median()),
                     median_et0=round(feats["et0_pre"].median()),
                     median_wb=round(feats["wb_pre"].median()),
                     corr_P_vs_WB=round(feats["prcp_pre"].corr(feats["wb_pre"]), 3)))

    fit(d, ["gdd_pre", "prcp_pre"], "  [published]  GDD + precipitation",
        rows, endpoint, "GDD + precipitation")
    fit(d, ["gdd_pre", "wb_pre"], "  [new]        GDD + water balance",
        rows, endpoint, "GDD + water balance")

with pd.ExcelWriter(OUT) as xl:
    pd.DataFrame(rows).to_excel(xl, sheet_name="model_comparison", index=False)
    pd.DataFrame(desc).to_excel(xl, sheet_name="window_summary", index=False)
print(f"\nwrote {OUT}")
