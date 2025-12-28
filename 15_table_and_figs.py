# 15_table1_and_figs.py
# Build Table 1 + sanity-check scatter plots using pre-season windows.
# Robust to either prefixed ("flowers_gdd_pre") or unprefixed ("gdd_pre") columns
# in fixed_window_features.xlsx. Plots use Step-04 colors.

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ---------- Style & colors (match Step 04) ----------
plt.style.use("seaborn-v0_8-whitegrid")
COLOR_FLOWERS = "#E8743B"   # warm orange
COLOR_FRUITS  = "#6B5B95"   # purple
LINE_FLOWERS  = "darkred"
LINE_FRUITS   = "darkviolet"

# Slightly larger publication-ish fonts
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10
})

# ---------- Paths ----------
PROJECT = Path.cwd()
AGG = PROJECT / "survival_analysis_results"
OUT_DIR = PROJECT / "15_table1_and_figs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Inputs (prefer cleaned)
SURV_CLEAN = PROJECT / "07_validate_sensitivity" / "survival_with_weather_clean.xlsx"
SURV_RAW   = PROJECT / "03_merge_survival_with_weather" / "survival_with_weather.xlsx"

# Look for fixed_window_features.xlsx in multiple locations (priority order)
FIXED_CANDIDATES = [
    PROJECT / "10_refit_simple_models" / "fixed_window_features.xlsx",
    PROJECT / "12_bivariate_fixed_window" / "fixed_window_features.xlsx",
    AGG / "fixed_window_features.xlsx",
]
FIXED_FEATS = next((p for p in FIXED_CANDIDATES if p.exists()), None)
if FIXED_FEATS is None:
    raise SystemExit(f"fixed_window_features.xlsx not found in any of: {[str(p) for p in FIXED_CANDIDATES]}")

# Outputs
OUT_STEP = OUT_DIR / "publication_table1.xlsx"
OUT_AGG  = AGG     / "publication_table1.xlsx"

# ---------- Helpers ----------
def norm(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (df.columns.str.strip()
                  .str.lower().str.replace(" ", "_").str.replace("-", "_"))
    return df

def load_surv(sheet: str) -> pd.DataFrame:
    src = SURV_CLEAN if SURV_CLEAN.exists() else SURV_RAW
    df = norm(pd.read_excel(src, sheet_name=sheet))
    need = ["site_id", "year", "event", "l", "r"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise SystemExit(f"[{sheet}] missing columns {miss} in {src.name}")
    df["site_id"] = df["site_id"].astype(str)
    df["year"]    = df["year"].astype(int)
    df["l"] = pd.to_numeric(df["l"], errors="coerce").clip(1, 366)
    df["r"] = pd.to_numeric(df["r"], errors="coerce").clip(1, 366)
    return df

BASES = ["gdd_pre","prcp_pre","tmin_pre_mean","tmax_pre_mean",
         "frost_pre","heat_pre","coverage","days_present"]

def ensure_prefixed(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """If columns like 'gdd_pre' exist but 'flowers_gdd_pre' doesn't, add the prefix."""
    df = df.copy()
    for base in BASES:
        pref = f"{prefix}_{base}"
        if pref not in df.columns and base in df.columns:
            df.rename(columns={base: pref}, inplace=True)
    if "site_id" in df.columns: df["site_id"] = df["site_id"].astype(str)
    if "year"    in df.columns: df["year"]    = df["year"].astype(int)
    return df

def load_windows():
    print(f"  Using fixed windows from: {FIXED_FEATS}")
    fl = ensure_prefixed(norm(pd.read_excel(FIXED_FEATS, sheet_name="flowers_window")), "flowers")
    fr = ensure_prefixed(norm(pd.read_excel(FIXED_FEATS, sheet_name="fruits_window")),  "fruits")
    if "site_id" not in fl.columns or "year" not in fl.columns:
        raise SystemExit("[flowers_window] must contain site_id and year")
    if "site_id" not in fr.columns or "year" not in fr.columns:
        raise SystemExit("[fruits_window] must contain site_id and year")
    print("  flowers_window cols:", sorted([c for c in fl.columns if c.startswith("flowers_")])[:8], "…")
    print("  fruits_window  cols:", sorted([c for c in fr.columns if c.startswith("fruits_")])[:8],  "…")
    return fl, fr

def pearson_r(x, y):
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    m = x.notna() & y.notna()
    if m.sum() < 3: return np.nan
    x = x[m]; y = y[m]
    if x.std(ddof=0) == 0 or y.std(ddof=0) == 0: return np.nan
    return float(np.corrcoef(x, y)[0, 1])

def table1_for_phase(name, surv, feats, window_prefix, cov_threshold=0.70):
    N       = len(surv)
    events  = int((surv["event"] == 1).sum())
    cens    = int(N - events)
    event_rate = round(events / N, 3) if N else np.nan
    med_R   = float(surv.loc[surv["event"]==1, "r"].median()) if events > 0 else np.nan
    n_sites = int(surv["site_id"].nunique())

    cov_col = f"{window_prefix}_coverage"
    gdd_col = f"{window_prefix}_gdd_pre"

    feats_ok = feats.copy()
    if cov_col in feats_ok.columns:
        feats_ok = feats_ok[feats_ok[cov_col] >= cov_threshold].copy()

    if gdd_col not in feats_ok.columns:
        raise SystemExit(f"[{name}] expected column not found in features: {gdd_col}")

    dfj = surv.merge(feats_ok[["site_id","year", gdd_col]],
                     on=["site_id","year"], how="left")

    ev = dfj[dfj["event"] == 1]
    r_ev = pearson_r(ev[gdd_col], ev["r"])

    return {
        "Phenophase": "Open flowers" if name == "open_flowers" else "Ripe fruits",
        "N rows": N,
        "Events": events,
        "Censored": cens,
        "Event rate": event_rate,
        "Median event DOY": round(med_R, 1) if pd.notna(med_R) else np.nan,
        "Sites": n_sites,
        "Events-only r(GDD_pre, DOY)": round(r_ev, 3) if pd.notna(r_ev) else np.nan
    }, dfj

def scatter_plot(ev_df, gdd_col, title, out_path, point_color, line_color):
    """Events-only scatter with OLS line, Step-04 colors."""
    d = ev_df[(ev_df["event"] == 1) & ev_df[gdd_col].notna() & ev_df["r"].notna()].copy()
    if d.empty:
        print(f"Plot skipped (no events with {gdd_col}): {out_path.name}")
        return

    x = d[gdd_col].to_numpy(dtype=float)
    y = d["r"].to_numpy(dtype=float)

    # Safe Pearson r
    r = np.nan
    if len(x) >= 3 and np.std(x) > 0 and np.std(y) > 0:
        r = float(np.corrcoef(x, y)[0, 1])

    # Simple least-squares fit if possible
    line_xy = None
    if len(x) >= 2:
        coeffs = np.polyfit(x, y, deg=1)
        xx = np.linspace(x.min(), x.max(), 200)
        yy = coeffs[0] * xx + coeffs[1]
        line_xy = (xx, yy)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(x, y, alpha=0.85, s=60, color=point_color,
               edgecolors=line_color, linewidth=0.6)
    if line_xy:
        ax.plot(line_xy[0], line_xy[1], "--", color=line_color, linewidth=1.6, alpha=0.75)

    ax.set_xlabel("Pre-season GDD (base 10°C)")
    ax.set_ylabel("Event DOY")
    ax.set_title(title)
    if np.isfinite(r):
        ax.text(0.02, 0.96, f"r ≈ {r:.2f}", transform=ax.transAxes,
                va="top", ha="left", fontsize=10, color="black")

    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

# ---------- Main ----------
def main():
    print("Loading windows…")
    fl_win, fr_win = load_windows()
    print("Loading survival…")
    surv_fl = load_surv("open_flowers")
    surv_fr = load_surv("ripe_fruits")

    row_fl, join_fl = table1_for_phase("open_flowers", surv_fl, fl_win, "flowers")
    row_fr, join_fr = table1_for_phase("ripe_fruits",  surv_fr, fr_win, "fruits")

    table1 = pd.DataFrame([row_fl, row_fr])

    # Save Table 1 to both step folder and aggregated folder
    for out in (OUT_STEP, OUT_AGG):
        with pd.ExcelWriter(out, engine="openpyxl") as xw:
            table1.to_excel(xw, index=False, sheet_name="Table1")
            pd.DataFrame({"README":[
                "Table1 summarizes counts and median event timing.",
                "Events-only Pearson r uses pre-season GDD windows (flowers: DOY1–120, fruits: DOY1–180).",
                "Scatter plots reflect events only; no raw daily weather are exposed."
            ]}).to_excel(xw, index=False, sheet_name="README")
        print(f"Saved: {out}")

    # Save plots (Step-04 colors)
    scatter_plot(join_fl, "flowers_gdd_pre",
                 "Open flowers: pre-season GDD vs event DOY",
                 OUT_DIR / "gdd_vs_doy_open_flowers.tif",
                 point_color=COLOR_FLOWERS, line_color=LINE_FLOWERS)

    scatter_plot(join_fr, "fruits_gdd_pre",
                 "Ripe fruits: pre-season GDD vs event DOY",
                 OUT_DIR / "gdd_vs_doy_ripe_fruits.tif",
                 point_color=COLOR_FRUITS, line_color=LINE_FRUITS)

if __name__ == "__main__":
    main()
