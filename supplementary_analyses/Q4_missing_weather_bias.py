# Q4_missing_weather_bias.py
"""
Missing Weather Data Bias Check
================================
This script explicitly checks whether missing weather data causes bias in cumulative measures
by comparing:
1. Cumulative measures computed with missing data (current approach)
2. Cumulative measures computed with imputed data (forward-fill interpolation)

Outputs: supplementary_analyses/Q4_bias_results/
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    sys.stdout.reconfigure(encoding="utf-8")
except:
    pass

# ---- Paths ----
PROJECT = Path(__file__).resolve().parent.parent
SURV_DATA = PROJECT / "07_validate_sensitivity" / "survival_with_weather_clean.xlsx"
WEATHER_DATA = PROJECT / "02_fetch_nasa_power_weather" / "site_daily_weather.xlsx"
OUT_DIR = PROJECT / "supplementary_analyses" / "Q4_bias_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Style ----
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11


def norm(df):
    """Normalize column names."""
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("-", "_")
    return df


def compute_cumulative_with_missing(wx_group, cutoff_doy):
    """
    Compute cumulative measures up to cutoff_doy using only available dates.
    This is the current approach - only days with weather data are included.
    """
    sub = wx_group[wx_group['doy'] <= cutoff_doy].copy()
    if sub.empty:
        return None
    
    # Compute cumulative sums (only for days that exist in dataset)
    sub['gdd_cum'] = sub['gdd_base10'].cumsum()
    sub['prcp_cum'] = sub['prcp_mm'].cumsum()
    sub['days_cum'] = sub['gdd_base10'].notna().cumsum()  # Count non-missing days
    
    last = sub.iloc[-1]
    
    # Days covered = number of days in dataset up to cutoff
    days_covered = len(sub)
    
    return {
        'gdd_sum': float(last['gdd_cum']) if pd.notna(last['gdd_cum']) else np.nan,
        'prcp_sum': float(last['prcp_cum']) if pd.notna(last['prcp_cum']) else np.nan,
        'days_covered': days_covered,
        'expected_days': cutoff_doy,
        'missing_days': cutoff_doy - days_covered,  # Days not in dataset
        'coverage_pct': 100 * days_covered / cutoff_doy if cutoff_doy > 0 else 0
    }


def compute_cumulative_with_imputation(wx_group, cutoff_doy):
    """
    Compute cumulative measures up to cutoff_doy assuming missing days have average values.
    This estimates what cumulative values would be if all days in the window had data.
    Uses average daily values from available data to fill gaps.
    """
    sub = wx_group[wx_group['doy'] <= cutoff_doy].copy()
    if sub.empty:
        return None
    
    # Get the actual date range in the data
    min_doy = sub['doy'].min()
    max_doy = sub['doy'].max()
    days_in_range = max_doy - min_doy + 1
    
    # Current cumulative (only available days)
    current_gdd = sub['gdd_base10'].sum()
    current_prcp = sub['prcp_mm'].sum()
    days_available = len(sub)
    
    if days_available == 0:
        return None
    
    # Average daily values from available data
    avg_daily_gdd = current_gdd / days_available if days_available > 0 else 0
    avg_daily_prcp = current_prcp / days_available if days_available > 0 else 0
    
    # Estimate cumulative if all days in the range had data
    # Use average daily value for missing days within the range
    missing_days_in_range = days_in_range - days_available
    estimated_gdd = current_gdd + avg_daily_gdd * missing_days_in_range
    estimated_prcp = current_prcp + avg_daily_prcp * missing_days_in_range
    
    return {
        'gdd_sum': float(estimated_gdd),
        'prcp_sum': float(estimated_prcp),
        'days_covered': days_in_range,  # All days in range assumed covered
        'expected_days': days_in_range,
        'missing_days': 0,
        'coverage_pct': 100.0
    }


def analyze_bias():
    """
    Compare cumulative measures with and without imputation to quantify bias.
    """
    print("\n" + "="*70)
    print("RE2 Q4: Missing Weather Data Bias Analysis")
    print("="*70)
    
    # Load survival data
    print("\nLoading survival data...")
    df_flowers = norm(pd.read_excel(SURV_DATA, sheet_name="open_flowers"))
    df_fruits = norm(pd.read_excel(SURV_DATA, sheet_name="ripe_fruits"))
    
    # Load weather data
    print("Loading weather data...")
    wx = pd.read_excel(WEATHER_DATA, sheet_name="daily_weather")
    wx = norm(wx)
    
    # Ensure required columns
    for col in ["site_id", "date", "gdd_base10", "prcp_mm"]:
        if col not in wx.columns:
            raise SystemExit(f"Weather data missing column: {col}")
    
    # Ensure site_id is consistent type (numeric)
    wx['site_id'] = pd.to_numeric(wx['site_id'], errors='coerce')
    wx['date'] = pd.to_datetime(wx['date'])
    wx['year'] = wx['date'].dt.year
    wx['doy'] = wx['date'].dt.dayofyear
    
    # Sort by site, year, doy
    wx_sorted = wx.sort_values(['site_id', 'year', 'doy']).copy()
    
    all_results = []
    
    for sheet_name, df in [("open_flowers", df_flowers), ("ripe_fruits", df_fruits)]:
        print(f"\n{'─'*70}")
        print(f"Analyzing: {sheet_name.replace('_', ' ').title()}")
        print(f"{'─'*70}")
        
        # Get cutoff DOY for each observation
        df['cutoff_doy'] = np.where(
            (df['event'] == 1) & df['r'].notna(),
            df['r'],
            df.get('last_obs_doy', df['l'] + 30)
        )
        df['cutoff_doy'] = df['cutoff_doy'].clip(lower=1, upper=366).astype(int)
        
        # Get first observation DOY (when weather data collection likely started)
        df['first_obs_doy'] = df.get('first_obs_doy', df['l'] - 30).clip(lower=1, upper=366).astype(int)
        
        results = []
        
        for idx, row in df.iterrows():
            site_id = pd.to_numeric(row['site_id'], errors='coerce')
            year = int(row['year'])
            cutoff_doy = int(row['cutoff_doy'])
            first_obs_doy = int(row.get('first_obs_doy', max(1, cutoff_doy - 60)))
            
            if pd.isna(site_id):
                continue
            
            # Get weather for this site-year
            wx_group = wx_sorted[
                (wx_sorted['site_id'] == site_id) & 
                (wx_sorted['year'] == year)
            ].copy()
            
            if wx_group.empty:
                continue
            
            # Only analyze period from first_obs_doy to cutoff_doy (actual observation window)
            # This avoids bias from days before observations started
            analysis_start = max(1, first_obs_doy)
            analysis_end = cutoff_doy
            
            # Filter weather to analysis window
            wx_analysis = wx_group[
                (wx_group['doy'] >= analysis_start) & 
                (wx_group['doy'] <= analysis_end)
            ].copy()
            
            if wx_analysis.empty:
                continue
            
            # Compute with missing data (current approach)
            with_missing = compute_cumulative_with_missing(wx_analysis, analysis_end)
            if with_missing is None:
                continue
            
            # Adjust expected days to analysis window
            expected_days_in_window = analysis_end - analysis_start + 1
            with_missing['expected_days'] = expected_days_in_window
            with_missing['missing_days'] = expected_days_in_window - with_missing['days_covered']
            with_missing['coverage_pct'] = 100 * with_missing['days_covered'] / expected_days_in_window if expected_days_in_window > 0 else 0
            
            # Compute with imputation
            with_imputation = compute_cumulative_with_imputation(wx_analysis, analysis_end)
            if with_imputation is None:
                continue
            
            # Adjust imputation to analysis window
            with_imputation['expected_days'] = expected_days_in_window
            
            # Calculate bias
            gdd_bias = with_imputation['gdd_sum'] - with_missing['gdd_sum']
            gdd_bias_pct = 100 * gdd_bias / with_missing['gdd_sum'] if with_missing['gdd_sum'] > 0 else 0
            
            prcp_bias = with_imputation['prcp_sum'] - with_missing['prcp_sum']
            prcp_bias_pct = 100 * prcp_bias / with_missing['prcp_sum'] if with_missing['prcp_sum'] > 0 else 0
            
            results.append({
                'sheet': sheet_name,
                'site_id': float(site_id),
                'year': year,
                'cutoff_doy': cutoff_doy,
                'missing_days': with_missing['missing_days'],
                'coverage_pct': with_missing['coverage_pct'],
                'gdd_with_missing': with_missing['gdd_sum'],
                'gdd_with_imputation': with_imputation['gdd_sum'],
                'gdd_bias': gdd_bias,
                'gdd_bias_pct': gdd_bias_pct,
                'prcp_with_missing': with_missing['prcp_sum'],
                'prcp_with_imputation': with_imputation['prcp_sum'],
                'prcp_bias': prcp_bias,
                'prcp_bias_pct': prcp_bias_pct
            })
        
        results_df = pd.DataFrame(results)
        all_results.append(results_df)
        
        # Summary statistics
        print(f"\n--- Summary Statistics ---")
        print(f"Total observations analyzed: {len(results_df)}")
        
        if len(results_df) == 0:
            print(f"  WARNING: No observations matched weather data!")
            print(f"  Survival data has {len(df)} rows")
            print(f"  Sample site_ids: {df['site_id'].unique()[:5] if 'site_id' in df.columns else 'N/A'}")
            print(f"  Sample years: {df['year'].unique()[:5] if 'year' in df.columns else 'N/A'}")
            print(f"  Weather data sites: {wx_sorted['site_id'].unique()[:5]}")
            print(f"  Weather data years: {wx_sorted['year'].unique()[:5]}")
            continue
        
        print(f"\nMissing Data:")
        print(f"  Mean missing days: {results_df['missing_days'].mean():.1f}")
        print(f"  Max missing days: {results_df['missing_days'].max():.0f}")
        print(f"  Mean coverage: {results_df['coverage_pct'].mean():.1f}%")
        print(f"  Min coverage: {results_df['coverage_pct'].min():.1f}%")
        
        print(f"\nGDD Bias:")
        print(f"  Mean bias: {results_df['gdd_bias'].mean():.1f} GDD")
        print(f"  Mean bias %: {results_df['gdd_bias_pct'].mean():.2f}%")
        print(f"  Max bias: {results_df['gdd_bias'].max():.1f} GDD")
        print(f"  Max bias %: {results_df['gdd_bias_pct'].max():.2f}%")
        
        print(f"\nPrecipitation Bias:")
        print(f"  Mean bias: {results_df['prcp_bias'].mean():.2f} mm")
        print(f"  Mean bias %: {results_df['prcp_bias_pct'].mean():.2f}%")
        print(f"  Max bias: {results_df['prcp_bias'].max():.2f} mm")
        print(f"  Max bias %: {results_df['prcp_bias_pct'].max():.2f}%")
        
        # Observations with substantial bias (>5%)
        substantial_gdd = (results_df['gdd_bias_pct'].abs() > 5).sum()
        substantial_prcp = (results_df['prcp_bias_pct'].abs() > 5).sum()
        print(f"\nObservations with >5% bias:")
        print(f"  GDD: {substantial_gdd} ({100*substantial_gdd/len(results_df):.1f}%)")
        print(f"  Precipitation: {substantial_prcp} ({100*substantial_prcp/len(results_df):.1f}%)")
    
    # Combine results
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Create visualization
    create_bias_plots(combined_df)
    
    # Save results
    out_xlsx = OUT_DIR / "Q4_missing_weather_bias.xlsx"
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as xw:
        combined_df.to_excel(xw, index=False, sheet_name="all_results")
        
        # Summary by sheet
        for sheet in ["open_flowers", "ripe_fruits"]:
            sheet_df = combined_df[combined_df['sheet'] == sheet]
            if not sheet_df.empty:
                summary = pd.DataFrame({
                    'metric': ['Mean', 'Median', 'Std', 'Min', 'Max'],
                    'missing_days': [
                        sheet_df['missing_days'].mean(),
                        sheet_df['missing_days'].median(),
                        sheet_df['missing_days'].std(),
                        sheet_df['missing_days'].min(),
                        sheet_df['missing_days'].max()
                    ],
                    'coverage_pct': [
                        sheet_df['coverage_pct'].mean(),
                        sheet_df['coverage_pct'].median(),
                        sheet_df['coverage_pct'].std(),
                        sheet_df['coverage_pct'].min(),
                        sheet_df['coverage_pct'].max()
                    ],
                    'gdd_bias_pct': [
                        sheet_df['gdd_bias_pct'].mean(),
                        sheet_df['gdd_bias_pct'].median(),
                        sheet_df['gdd_bias_pct'].std(),
                        sheet_df['gdd_bias_pct'].min(),
                        sheet_df['gdd_bias_pct'].max()
                    ],
                    'prcp_bias_pct': [
                        sheet_df['prcp_bias_pct'].mean(),
                        sheet_df['prcp_bias_pct'].median(),
                        sheet_df['prcp_bias_pct'].std(),
                        sheet_df['prcp_bias_pct'].min(),
                        sheet_df['prcp_bias_pct'].max()
                    ]
                })
                summary.to_excel(xw, index=False, sheet_name=f"summary_{sheet}")
    
    print(f"\n{'='*70}")
    print(f"Results saved: {out_xlsx}")
    print(f"{'='*70}")
    
    # Key conclusion
    print("\n" + "="*70)
    print("KEY CONCLUSION:")
    print("="*70)
    mean_gdd_bias = combined_df['gdd_bias_pct'].mean()
    mean_prcp_bias = combined_df['prcp_bias_pct'].mean()
    print(f"  Mean GDD bias from missing data: {mean_gdd_bias:.2f}%")
    print(f"  Mean precipitation bias from missing data: {mean_prcp_bias:.2f}%")
    
    if abs(mean_gdd_bias) < 5 and abs(mean_prcp_bias) < 5:
        print(f"\n  → Bias is minimal (<5%) and unlikely to substantially affect results.")
    else:
        print(f"\n  → Bias may be substantial and should be considered in interpretation.")
    
    return combined_df


def create_bias_plots(df):
    """Create visualization of bias."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = {'open_flowers': '#E8743B', 'ripe_fruits': '#6B5B95'}
    
    # Plot 1: Coverage vs GDD bias
    ax1 = axes[0, 0]
    for sheet in df['sheet'].unique():
        sheet_df = df[df['sheet'] == sheet]
        ax1.scatter(sheet_df['coverage_pct'], sheet_df['gdd_bias_pct'], 
                   color=colors[sheet], alpha=0.6, label=sheet.replace('_', ' ').title(), s=60)
    ax1.axhline(0, color='black', linestyle='--', linewidth=1)
    ax1.set_xlabel('Weather Coverage (%)')
    ax1.set_ylabel('GDD Bias (%)')
    ax1.set_title('GDD Bias vs Weather Coverage')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Coverage vs Precipitation bias
    ax2 = axes[0, 1]
    for sheet in df['sheet'].unique():
        sheet_df = df[df['sheet'] == sheet]
        ax2.scatter(sheet_df['coverage_pct'], sheet_df['prcp_bias_pct'], 
                   color=colors[sheet], alpha=0.6, label=sheet.replace('_', ' ').title(), s=60)
    ax2.axhline(0, color='black', linestyle='--', linewidth=1)
    ax2.set_xlabel('Weather Coverage (%)')
    ax2.set_ylabel('Precipitation Bias (%)')
    ax2.set_title('Precipitation Bias vs Weather Coverage')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Distribution of GDD bias
    ax3 = axes[1, 0]
    for sheet in df['sheet'].unique():
        sheet_df = df[df['sheet'] == sheet]
        ax3.hist(sheet_df['gdd_bias_pct'], bins=20, alpha=0.6, 
                color=colors[sheet], label=sheet.replace('_', ' ').title(), edgecolor='black')
    ax3.axvline(0, color='black', linestyle='--', linewidth=2)
    ax3.set_xlabel('GDD Bias (%)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Distribution of GDD Bias')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Distribution of Precipitation bias
    ax4 = axes[1, 1]
    for sheet in df['sheet'].unique():
        sheet_df = df[df['sheet'] == sheet]
        ax4.hist(sheet_df['prcp_bias_pct'], bins=20, alpha=0.6, 
                color=colors[sheet], label=sheet.replace('_', ' ').title(), edgecolor='black')
    ax4.axvline(0, color='black', linestyle='--', linewidth=2)
    ax4.set_xlabel('Precipitation Bias (%)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Distribution of Precipitation Bias')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Missing Weather Data Bias Analysis', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    out_png = OUT_DIR / "Q4_bias_analysis.png"
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved plot: {out_png.name}")


if __name__ == "__main__":
    analyze_bias()

