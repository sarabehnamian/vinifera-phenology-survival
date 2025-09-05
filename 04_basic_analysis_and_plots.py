# 04_basic_analysis_and_plots.py
# Analyze the survival data with weather, perform QC checks, and create visualizations
# Input : survival_analysis_results/survival_with_weather.xlsx
# Output: survival_analysis_results/04_analysis_outputs/  (Excel + .tif plots + QC text)

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# ---------------- Setup & paths ----------------
# Use a cleaner style and professional color palette
plt.style.use("seaborn-v0_8-whitegrid")
# Define custom colors for consistency
COLOR_FLOWERS = '#E8743B'  # Warm orange for flowers
COLOR_FRUITS = '#6B5B95'   # Purple for fruits

IN_FILE = Path("03_merge_survival_with_weather/survival_with_weather.xlsx")
OUT_DIR = Path("04_basic_analysis_and_plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_XLSX = OUT_DIR / "analysis_results.xlsx"
QC_TXT   = OUT_DIR / "quality_checks.txt"

# --------------- Load -----------------
print("Loading survival data with weather...")
flowers_df = pd.read_excel(IN_FILE, sheet_name="open_flowers")
fruits_df  = pd.read_excel(IN_FILE, sheet_name="ripe_fruits")

# Ensure lower-case column names for safety
flowers_df.columns = flowers_df.columns.str.lower()
fruits_df.columns  = fruits_df.columns.str.lower()

# Quick presence checks
need_cols = {"site_id","year","event","r","gdd_sum_to_cutoff"}
for name, df in [("open_flowers", flowers_df), ("ripe_fruits", fruits_df)]:
    missing = need_cols - set(df.columns)
    if missing:
        raise SystemExit(f"[{name}] missing required columns: {missing}")

# Remove rows with no weather
flowers_clean = flowers_df.dropna(subset=["gdd_sum_to_cutoff"]).copy()
fruits_clean  = fruits_df.dropna(subset=["gdd_sum_to_cutoff"]).copy()

# Event-only subsets
flowers_events = flowers_clean[flowers_clean["event"] == 1].copy()
fruits_events  = fruits_clean[fruits_clean["event"] == 1].copy()

# ---------------- QC: basic numbers & anomaly detection ----------------
def safe_median(x):
    return float(np.nanmedian(x)) if len(x) else np.nan

basic_summary = {
    "open_flowers": {
        "rows": len(flowers_clean),
        "events": int(flowers_events.shape[0]),
        "event_rate": float(flowers_clean["event"].mean()) if len(flowers_clean) else np.nan,
        "median_R": safe_median(flowers_events["r"]),
    },
    "ripe_fruits": {
        "rows": len(fruits_clean),
        "events": int(fruits_events.shape[0]),
        "event_rate": float(fruits_clean["event"].mean()) if len(fruits_clean) else np.nan,
        "median_R": safe_median(fruits_events["r"]),
    },
}

# Pearson correlations (events only) between GDD and R
def pearson_corr(df, x, y):
    df2 = df[[x, y]].dropna()
    return float(df2[x].corr(df2[y])) if len(df2) >= 2 else np.nan

r_flowers = pearson_corr(flowers_events, "gdd_sum_to_cutoff", "r")
r_fruits  = pearson_corr(fruits_events,  "gdd_sum_to_cutoff", "r")

# Flag biologically implausible ripe-fruit events (likely carry-over)
EARLY_FRUIT_DOY = 120
fruits_early_flags = fruits_events[fruits_events["r"] < EARLY_FRUIT_DOY].copy()

# Prepare filtered fruits (for plots and secondary stats)
fruits_events_filt = fruits_events[fruits_events["r"] >= EARLY_FRUIT_DOY].copy()
r_fruits_filt      = pearson_corr(fruits_events_filt, "gdd_sum_to_cutoff", "r")
median_fruits_filt = safe_median(fruits_events_filt["r"])

# ---------------- Write QC text report ----------------
with open(QC_TXT, "w", encoding="utf-8") as fh:
    fh.write("Quality checks for survival_with_weather.xlsx\n")
    fh.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    fh.write("BASIC COUNTS & MEDIANS (events only)\n")
    fh.write(f"- Open flowers: rows={basic_summary['open_flowers']['rows']}, "
             f"events={basic_summary['open_flowers']['events']} "
             f"(rate={basic_summary['open_flowers']['event_rate']:.2f}), "
             f"median R≈{basic_summary['open_flowers']['median_R']:.0f} DOY\n")
    fh.write(f"- Ripe fruits:  rows={basic_summary['ripe_fruits']['rows']}, "
             f"events={basic_summary['ripe_fruits']['events']} "
             f"(rate={basic_summary['ripe_fruits']['event_rate']:.2f}), "
             f"median R≈{basic_summary['ripe_fruits']['median_R']:.0f} DOY\n\n")

    fh.write("GDD vs Timing (Pearson r, events only)\n")
    fh.write(f"- Open flowers: r ≈ {r_flowers:.2f}\n")
    fh.write(f"- Ripe fruits : r ≈ {r_fruits:.2f} (before filtering)\n")
    fh.write(f"- Ripe fruits : r ≈ {r_fruits_filt:.2f} (after dropping R<{EARLY_FRUIT_DOY})\n\n")

    if len(fruits_early_flags):
        fh.write(f"FLAGGED RIPE-FRUIT EVENTS WITH R < {EARLY_FRUIT_DOY} (likely carry-over fruit):\n")
        cols = [c for c in ["site_id","year","individual_id","l","r","gdd_sum_to_cutoff"] if c in fruits_early_flags.columns]
        fh.write(fruits_early_flags[cols].to_string(index=False))
        fh.write("\n\nRecommendation: drop ripe_fruits events with R < "
                 f"{EARLY_FRUIT_DOY}. These likely represent fruit remaining from the previous season.\n")
    else:
        fh.write(f"No ripe-fruit events with R < {EARLY_FRUIT_DOY} were found.\n")

print(f"QC report written to: {QC_TXT}")

# ---------------- Excel outputs ----------------
with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
    # Summary statistics (before filter)
    summary_stats = pd.DataFrame({
        "Phenophase": ["Open Flowers", "Ripe Fruits"],
        "Total_Obs":  [basic_summary["open_flowers"]["rows"], basic_summary["ripe_fruits"]["rows"]],
        "Events":     [basic_summary["open_flowers"]["events"], basic_summary["ripe_fruits"]["events"]],
        "Censored":   [
            basic_summary["open_flowers"]["rows"] - basic_summary["open_flowers"]["events"],
            basic_summary["ripe_fruits"]["rows"]  - basic_summary["ripe_fruits"]["events"]
        ],
        "Event_Rate": [basic_summary["open_flowers"]["event_rate"], basic_summary["ripe_fruits"]["event_rate"]],
        "Median_DOY_Events": [
            basic_summary["open_flowers"]["median_R"], basic_summary["ripe_fruits"]["median_R"]
        ],
        "Pearson_r(GDD,R)": [r_flowers, r_fruits],
    })
    summary_stats.to_excel(writer, sheet_name="Summary_Statistics", index=False)

    # Weather effects for events (before filter)
    def effects(df):
        return pd.Series({
            "GDD_at_Event_mean":   df["gdd_sum_to_cutoff"].mean(),
            "GDD_at_Event_sd":     df["gdd_sum_to_cutoff"].std(),
            "Prcp_total_mean":     df["prcp_sum_to_cutoff"].mean(),
            "Tmin_mean":           df["tmin_mean_to_cutoff"].mean(),
            "Tmax_mean":           df["tmax_mean_to_cutoff"].mean(),
            "Frost_days_mean":     df["frost_days_to_cutoff"].mean(),
            "Heat_days_mean":      df["heat_days_to_cutoff"].mean(),
        })

    wx_eff = pd.concat([
        effects(flowers_events).rename("Open Flowers"),
        effects(fruits_events).rename("Ripe Fruits"),
    ], axis=1)
    wx_eff.to_excel(writer, sheet_name="Weather_Effects", index=True)

    # Site analysis (event rates by site; before filter)
    def site_table(df, label):
        tmp = (df.groupby("site_id", as_index=False)
                 .agg(n=("event","size"), events=("event","sum")))
        tmp["event_rate"] = tmp["events"] / tmp["n"]
        tmp["phenophase"] = label
        return tmp

    site_flow = site_table(flowers_clean, "Open Flowers")
    site_fru  = site_table(fruits_clean,  "Ripe Fruits")
    site_df   = pd.concat([site_flow, site_fru], ignore_index=True)
    site_df.to_excel(writer, sheet_name="Site_Analysis", index=False)

    # QC flags sheet & filtered fruits summary
    fruits_early_flags.to_excel(writer, sheet_name="QC_Flags_Ripe_R<120", index=False)
    pd.DataFrame({
        "After_Filter_RipeFruits": [f"keep R≥{EARLY_FRUIT_DOY}"],
        "Events_kept": [len(fruits_events_filt)],
        "Median_R_kept": [median_fruits_filt],
        "Pearson_r_kept": [r_fruits_filt],
    }).to_excel(writer, sheet_name="RipeFruits_Filtered_Summary", index=False)

print(f"Analysis results saved to: {OUT_XLSX}")

# ---------------- Plots (300 dpi .TIF) ----------------
# Set better default font sizes
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['legend.fontsize'] = 10

# 1) Event timing distributions
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

if len(flowers_events):
    n, bins, patches = axes[0].hist(flowers_events["r"], bins=15, 
                                    color=COLOR_FLOWERS, edgecolor="white", 
                                    alpha=0.8, linewidth=1.5)
    median_val = np.median(flowers_events["r"])
    axes[0].axvline(median_val, color='darkred', linestyle='--', linewidth=2, 
                   label=f'Median = {median_val:.0f}')
    axes[0].set_xlabel("Day of Year")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title(f"Open Flowers (n={len(flowers_events)})")
    axes[0].grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    # Place legend above the plot
    axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), 
                  frameon=True, fancybox=True, shadow=False)
else:
    axes[0].text(0.5, 0.5, "No events", ha="center", va="center")

if len(fruits_events_filt):
    n, bins, patches = axes[1].hist(fruits_events_filt["r"], bins=15, 
                                    color=COLOR_FRUITS, edgecolor="white", 
                                    alpha=0.8, linewidth=1.5)
    median_val = np.median(fruits_events_filt["r"])
    axes[1].axvline(median_val, color='darkviolet', linestyle='--', linewidth=2,
                   label=f'Median = {median_val:.0f}')
    axes[1].set_xlabel("Day of Year")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title(f"Ripe Fruits (R≥{EARLY_FRUIT_DOY}, n={len(fruits_events_filt)})")
    axes[1].grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    # Place legend above the plot
    axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), 
                  frameon=True, fancybox=True, shadow=False)
else:
    axes[1].text(0.5, 0.5, "No events after filter", ha="center", va="center")

plt.suptitle("Phenological Event Timing Distributions", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(OUT_DIR / "event_timing_distributions.tif", dpi=300, bbox_inches="tight")
plt.close()
print("✓ Saved: event_timing_distributions.tif")

# 2) GDD vs Event Time
if len(flowers_events) or len(fruits_events_filt):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    if len(flowers_events):
        axes[0].scatter(flowers_events["gdd_sum_to_cutoff"], flowers_events["r"], 
                       alpha=0.7, s=60, color=COLOR_FLOWERS, edgecolors='darkred', linewidth=0.5)
        axes[0].set_xlabel("Cumulative GDD (base 10°C)")
        axes[0].set_ylabel("Day of Year (First Event)")
        axes[0].set_title(f"Open Flowers (r = {r_flowers:.2f})")
        axes[0].grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # Add regression line if correlation is significant
        if abs(r_flowers) > 0.2:
            z = np.polyfit(flowers_events["gdd_sum_to_cutoff"], flowers_events["r"], 1)
            p = np.poly1d(z)
            x_line = np.linspace(flowers_events["gdd_sum_to_cutoff"].min(), 
                               flowers_events["gdd_sum_to_cutoff"].max(), 100)
            axes[0].plot(x_line, p(x_line), '--', color='darkred', alpha=0.5, linewidth=1.5)

    if len(fruits_events_filt):
        axes[1].scatter(fruits_events_filt["gdd_sum_to_cutoff"], fruits_events_filt["r"], 
                       alpha=0.7, s=60, color=COLOR_FRUITS, edgecolors='darkviolet', linewidth=0.5)
        axes[1].set_xlabel("Cumulative GDD (base 10°C)")
        axes[1].set_ylabel("Day of Year (First Event)")
        axes[1].set_title(f"Ripe Fruits (R≥{EARLY_FRUIT_DOY}, r = {r_fruits_filt:.2f})")
        axes[1].grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # Add regression line
        if abs(r_fruits_filt) > 0.2:
            z = np.polyfit(fruits_events_filt["gdd_sum_to_cutoff"], fruits_events_filt["r"], 1)
            p = np.poly1d(z)
            x_line = np.linspace(fruits_events_filt["gdd_sum_to_cutoff"].min(), 
                               fruits_events_filt["gdd_sum_to_cutoff"].max(), 100)
            axes[1].plot(x_line, p(x_line), '--', color='darkviolet', alpha=0.5, linewidth=1.5)

    plt.suptitle("Growing Degree Days vs Phenological Timing", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "gdd_vs_timing.tif", dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ Saved: gdd_vs_timing.tif")

# 3) Event rates by site (before filter, side-by-side bars)
site_rates_flow = (flowers_clean.groupby("site_id")["event"].mean().rename("Open Flowers"))
site_rates_fru  = (fruits_clean.groupby("site_id")["event"].mean().rename("Ripe Fruits"))
site_rates = pd.concat([site_rates_flow, site_rates_fru], axis=1).fillna(0.0)

if len(site_rates):
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(site_rates.shape[0])
    width = 0.35
    
    bars1 = ax.bar(x - width/2, site_rates["Open Flowers"], width, 
                   label="Open Flowers", color=COLOR_FLOWERS, alpha=0.8, 
                   edgecolor='darkred', linewidth=1)
    bars2 = ax.bar(x + width/2, site_rates["Ripe Fruits"], width, 
                   label="Ripe Fruits", color=COLOR_FRUITS, alpha=0.8,
                   edgecolor='darkviolet', linewidth=1)
    
    ax.set_xticks(x)
    ax.set_xticklabels(site_rates.index.astype(str), rotation=45, ha="right")
    ax.set_ylim(0, 1.15)
    ax.set_xlabel("Site ID")
    ax.set_ylabel("Event Rate")
    ax.set_title("Event Rates by Site", fontsize=14)
    
    # Place legend outside the plot area (to the right)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, 
             fancybox=True, shadow=True)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    ax.grid(True, alpha=0.3, axis='y', linestyle='-', linewidth=0.5)
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    plt.savefig(OUT_DIR / "site_comparison.tif", dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ Saved: site_comparison.tif")

print("\n" + "="*54)
print("STEP 04 COMPLETE — outputs in:", OUT_DIR)
print("="*54)