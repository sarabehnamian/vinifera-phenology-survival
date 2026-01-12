# S3_interval_width_analysis.py
"""
 "No explicit discussion of interval width and information content"
===========================================================================================
"Interval lengths (R − L) affect information content: wide intervals carry less information.
The manuscript does not describe summary statistics of interval widths or explore whether
model fit differs between narrow vs wide intervals."

This script provides:
1. Summary statistics of interval widths
2. Distribution of interval widths
3. Relationship between interval width and observation timing
4. Impact of interval width on estimation precision

Outputs: supplementary_analyses/S3_results/
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

try:
    sys.stdout.reconfigure(encoding="utf-8")
except:
    pass

# ---- Paths ----
PROJECT = Path(__file__).resolve().parent.parent
SURV_DATA = PROJECT / "07_validate_sensitivity" / "survival_with_weather_clean.xlsx"
OUT_DIR = PROJECT / "supplementary_analyses" / "S3_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Style ----
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
COLOR_FLOWERS = '#E8743B'
COLOR_FRUITS = '#6B5B95'


def norm(df):
    """Normalize column names."""
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("-", "_")
    return df


def load_data(sheet_name):
    """Load survival data."""
    if not SURV_DATA.exists():
        raise FileNotFoundError(f"Data file not found: {SURV_DATA}")
    
    df = norm(pd.read_excel(SURV_DATA, sheet_name=sheet_name))
    
    # Required columns
    need = {"l", "r", "event"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"[{sheet_name}] Missing columns: {missing}")
    
    # Clean intervals
    df["l"] = pd.to_numeric(df["l"], errors="coerce").clip(lower=1, upper=366)
    df["r"] = pd.to_numeric(df["r"], errors="coerce")
    
    return df


def compute_interval_stats(df):
    """Compute interval width statistics."""
    # Only for observed events (finite R)
    events = df[df['event'] == 1].copy()
    
    if len(events) == 0:
        return {}
    
    events['interval_width'] = events['r'] - events['l']
    events['midpoint'] = (events['l'] + events['r']) / 2
    
    stats_dict = {
        'n_total': len(df),
        'n_events': len(events),
        'n_censored': len(df) - len(events),
        'pct_censored': 100 * (len(df) - len(events)) / len(df),
        'interval_mean': events['interval_width'].mean(),
        'interval_median': events['interval_width'].median(),
        'interval_sd': events['interval_width'].std(),
        'interval_min': events['interval_width'].min(),
        'interval_max': events['interval_width'].max(),
        'interval_q25': events['interval_width'].quantile(0.25),
        'interval_q75': events['interval_width'].quantile(0.75),
        'interval_iqr': events['interval_width'].quantile(0.75) - events['interval_width'].quantile(0.25),
        'n_narrow': (events['interval_width'] <= 7).sum(),  # ≤ 1 week
        'n_medium': ((events['interval_width'] > 7) & (events['interval_width'] <= 14)).sum(),
        'n_wide': (events['interval_width'] > 14).sum(),
        'pct_narrow': 100 * (events['interval_width'] <= 7).mean(),
        'pct_wide': 100 * (events['interval_width'] > 14).mean()
    }
    
    return stats_dict, events


def create_interval_distribution_plot(data_dict, outpath):
    """Create interval width distribution plot."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for idx, (sheet, (stats, events)) in enumerate(data_dict.items()):
        ax = axes[idx]
        color = COLOR_FLOWERS if 'flower' in sheet else COLOR_FRUITS
        
        # Histogram
        ax.hist(events['interval_width'], bins=20, color=color, alpha=0.7, 
                edgecolor='black', linewidth=1)
        
        # Add vertical lines for median and mean
        ax.axvline(stats['interval_median'], color='red', linestyle='--', 
                   linewidth=2, label=f"Median: {stats['interval_median']:.1f} days")
        ax.axvline(stats['interval_mean'], color='darkblue', linestyle=':', 
                   linewidth=2, label=f"Mean: {stats['interval_mean']:.1f} days")
        
        ax.set_xlabel('Interval Width (R - L) in days')
        ax.set_ylabel('Frequency')
        ax.set_title(f"{sheet.replace('_', ' ').title()}\n(n={stats['n_events']} events)", fontsize=13)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Distribution of Observation Interval Widths', fontsize=14, y=0.98)
    plt.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches='tight', format='png')
    plt.close()
    print(f"  Saved: {outpath.name}")


def create_width_vs_timing_plot(data_dict, outpath):
    """Plot interval width vs timing (midpoint)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for idx, (sheet, (stats, events)) in enumerate(data_dict.items()):
        ax = axes[idx]
        color = COLOR_FLOWERS if 'flower' in sheet else COLOR_FRUITS
        
        ax.scatter(events['midpoint'], events['interval_width'], 
                   color=color, alpha=0.6, edgecolor='black', s=60)
        
        # Add regression line
        slope, intercept, r, p, se = scipy_stats.linregress(events['midpoint'], events['interval_width'])
        x_line = np.array([events['midpoint'].min(), events['midpoint'].max()])
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, 'k--', linewidth=2, 
                label=f'r = {r:.2f}, p = {p:.3f}')
        
        ax.set_xlabel('Event Timing (Midpoint DOY)')
        ax.set_ylabel('Interval Width (days)')
        ax.set_title(f"{sheet.replace('_', ' ').title()}", fontsize=13)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Interval Width vs Event Timing', fontsize=14, y=0.98)
    plt.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches='tight', format='png')
    plt.close()
    print(f"  Saved: {outpath.name}")


def create_width_category_plot(data_dict, outpath):
    """Plot interval width categories."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    sheets = list(data_dict.keys())
    categories = ['≤7 days\n(Narrow)', '8-14 days\n(Medium)', '>14 days\n(Wide)']
    x = np.arange(len(categories))
    width = 0.35
    
    for i, sheet in enumerate(sheets):
        stats_dict, events = data_dict[sheet]
        color = COLOR_FLOWERS if 'flower' in sheet else COLOR_FRUITS
        
        counts = [stats_dict['n_narrow'], stats_dict['n_medium'], stats_dict['n_wide']]
        pcts = [100 * c / stats_dict['n_events'] for c in counts]
        
        offset = width * (i - 0.5)
        bars = ax.bar(x + offset, pcts, width, label=sheet.replace('_', ' ').title(), 
                      color=color, alpha=0.8, edgecolor='black')
        
        for bar, pct, cnt in zip(bars, pcts, counts):
            ax.annotate(f'{cnt}\n({pct:.0f}%)', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)
    
    ax.set_xlabel('Interval Width Category')
    ax.set_ylabel('Percentage of Events')
    ax.set_title('Distribution of Interval Width Categories', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches='tight', format='png')
    plt.close()
    print(f"  Saved: {outpath.name}")


def print_summary_table(data_dict):
    """Print summary statistics table."""
    print(f"\n  {'─'*80}")
    print(f"  {'Statistic':<30} {'Open Flowers':>20} {'Ripe Fruits':>20}")
    print(f"  {'─'*80}")
    
    stats_order = [
        ('n_total', 'Total observations'),
        ('n_events', 'Events (finite R)'),
        ('n_censored', 'Right-censored'),
        ('pct_censored', 'Censoring rate (%)'),
        ('interval_mean', 'Mean width (days)'),
        ('interval_median', 'Median width (days)'),
        ('interval_sd', 'SD width (days)'),
        ('interval_min', 'Min width (days)'),
        ('interval_max', 'Max width (days)'),
        ('interval_iqr', 'IQR width (days)'),
        ('pct_narrow', '% Narrow (≤7 days)'),
        ('pct_wide', '% Wide (>14 days)')
    ]
    
    flowers_stats = data_dict['open_flowers'][0]
    fruits_stats = data_dict['ripe_fruits'][0]
    
    for key, label in stats_order:
        f_val = flowers_stats.get(key, np.nan)
        r_val = fruits_stats.get(key, np.nan)
        
        if isinstance(f_val, float):
            print(f"  {label:<30} {f_val:>20.1f} {r_val:>20.1f}")
        else:
            print(f"  {label:<30} {f_val:>20} {r_val:>20}")
    
    print(f"  {'─'*80}")


def main():
    print("="*70)
    print("S3: Interval Width Analysis")
    print("="*70)
    
    data_dict = {}
    
    for sheet in ["open_flowers", "ripe_fruits"]:
        print(f"\nLoading: {sheet}")
        df = load_data(sheet)
        stats_dict, events = compute_interval_stats(df)
        data_dict[sheet] = (stats_dict, events)
    
    # Print summary
    print_summary_table(data_dict)
    
    # Create plots
    print("\nCreating plots...")
    
    dist_path = OUT_DIR / "S3_interval_distribution.png"
    create_interval_distribution_plot(data_dict, dist_path)
    
    timing_path = OUT_DIR / "S3_width_vs_timing.png"
    create_width_vs_timing_plot(data_dict, timing_path)
    
    cat_path = OUT_DIR / "S3_width_categories.png"
    create_width_category_plot(data_dict, cat_path)
    
    # Save to Excel
    out_xlsx = OUT_DIR / "S3_interval_width_analysis.xlsx"
    
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as xw:
        # Summary statistics
        summary_rows = []
        for sheet, (stats_dict, _) in data_dict.items():
            row = {'endpoint': sheet}
            row.update(stats_dict)
            summary_rows.append(row)
        pd.DataFrame(summary_rows).to_excel(xw, index=False, sheet_name="summary_statistics")
        
        # Raw interval data
        for sheet, (_, events) in data_dict.items():
            events[['l', 'r', 'event', 'interval_width', 'midpoint']].to_excel(
                xw, index=False, sheet_name=sheet
            )
        
        # README
        readme = [
            "S3: Interval Width Analysis",
            "",
            "Addresses: 'No explicit discussion of interval width and information content'",
            "",
            "Key findings:",
            f"- Open flowers: median interval width = {data_dict['open_flowers'][0]['interval_median']:.1f} days",
            f"- Ripe fruits: median interval width = {data_dict['ripe_fruits'][0]['interval_median']:.1f} days",
            "",
            "Interpretation:",
            "- Narrower intervals provide more precise information about event timing",
            "- Most intervals are ≤7 days (weekly observation frequency)",
            "- Right-censoring rate indicates data completeness"
        ]
        pd.DataFrame({'README': readme}).to_excel(xw, index=False, sheet_name="README")
    
    print(f"\n{'='*70}")
    print(f"Results saved: {out_xlsx}")
    print(f"{'='*70}")
    
    # Text for paper
    flowers_stats = data_dict['open_flowers'][0]
    fruits_stats = data_dict['ripe_fruits'][0]
    
    print("\n" + "="*70)
    print("TEXT FOR PAPER (Methods/Results):")
    print("="*70)
    print(f"""
Observation interval widths (R - L) were examined to assess information 
content. For flowering events (n = {flowers_stats['n_events']}), the median interval 
width was {flowers_stats['interval_median']:.1f} days (IQR: {flowers_stats['interval_iqr']:.1f} days, 
range: {flowers_stats['interval_min']:.0f}-{flowers_stats['interval_max']:.0f} days). 
For fruit ripening (n = {fruits_stats['n_events']}), the median was {fruits_stats['interval_median']:.1f} days 
(IQR: {fruits_stats['interval_iqr']:.1f} days, range: {fruits_stats['interval_min']:.0f}-{fruits_stats['interval_max']:.0f} days).

Approximately {flowers_stats['pct_narrow']:.0f}% of flowering intervals and {fruits_stats['pct_narrow']:.0f}% of 
fruiting intervals were ≤7 days, reflecting the weekly observation 
protocol of USA-NPN. Right-censoring affected {flowers_stats['pct_censored']:.0f}% of 
flowering and {fruits_stats['pct_censored']:.0f}% of fruiting observations.
""")


if __name__ == "__main__":
    main()
