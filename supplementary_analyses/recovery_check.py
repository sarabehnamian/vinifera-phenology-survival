# RE3_R1C9_recovery_check.py
# R1.9: unit-level exclusion vs observation-level screening for the
# single early ripening record (individual 63014, Site 16610, 2015).
# Prints the four time ratios needed for [TR-A]..[TR-D].

import numpy as np
import pandas as pd
from lifelines import WeibullAFTFitter, LogLogisticAFTFitter

SURV = r"D:\ZAmerican_Vitis_20260726\00_npn_survival_analysis\survival_intervals_ready.xlsx"
FWIN = r"D:\ZAmerican_Vitis_20260726\10_refit_simple_models\fixed_window_features.xlsx"

sv = pd.read_excel(SURV, sheet_name="ripe_fruits")
fw = pd.read_excel(FWIN, sheet_name="fruits_window")
fw = fw.loc[fw["coverage"] >= 0.70, ["site_id", "year", "gdd_pre"]]


def fit(df, label):
    d = df.merge(fw, on=["site_id", "year"]).dropna(subset=["gdd_pre"]).copy()
    d["L"] = d["L"].clip(1, 366)
    d["R"] = np.where(d["event"] == 1, d["R"].clip(1, 366), np.inf)
    d["z"] = (d["gdd_pre"] - d["gdd_pre"].mean()) / d["gdd_pre"].std()
    print(f"{label}: n={len(d)}, events={int(d['event'].sum())}")
    for Fitter, name in [(WeibullAFTFitter, "Weibull"),
                         (LogLogisticAFTFitter, "Log-logistic")]:
        f = Fitter()
        f.fit_interval_censoring(d[["L", "R", "z"]], "L", "R")
        s = f.summary.xs("z", level=1, drop_level=False)
        b, lo, hi = (s["coef"].iloc[0],
                     s["coef lower 95%"].iloc[0],
                     s["coef upper 95%"].iloc[0])
        print(f"    {name:<13} TR = {np.exp(b):.3f} "
              f"({np.exp(lo):.3f}-{np.exp(hi):.3f})")


# Current: whole plant-site-year dropped  -> [TR-A] Weibull, [TR-C] log-logistic
fit(sv[~((sv["event"] == 1) & (sv["R"] < 120))], "Unit-level exclusion (current)")

# Recovered: screen applied at observation level -> [TR-B], [TR-D]
rec = sv.copy()
mask = (rec["individual_id"] == 63014) & (rec["year"] == 2015)
rec.loc[mask, ["L", "R", "event"]] = [201, 208, 1]
fit(rec, "Observation-level screening (recovered)")
