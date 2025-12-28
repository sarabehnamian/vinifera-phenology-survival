# RE2_P3_uncertain_observations.py
"""
Reviewer 2, Point 3: Effect of excluding uncertain observations on timing information.

Question addressed:
- How does excluding uncertain observations affect the timing information for each year?
- What is the impact on interval construction and survival endpoints?
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ==================== PATHS ====================
SCRIPT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = SCRIPT_DIR / "revision" / "RE2_P3_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Data files
OBS_DATA = SCRIPT_DIR / "data" / "status_intensity_observation_data.csv"
SURV_DATA = SCRIPT_DIR / "03_merge_survival_with_weather" / "survival_with_weather.xlsx"

# Output files
OUT_XLSX = OUTPUT_DIR / "RE2_P3_uncertain_observations_analysis.xlsx"
OUT_PNG = OUTPUT_DIR / "RE2_P3_uncertain_impact.png"

# Colors
COLOR_FLOWERS = '#E8743B'
COLOR_FRUITS = '#6B5B95'

# ==================== HELPER FUNCTIONS ====================
def norm(df):
    """Normalize column names to lowercase."""
    df.columns = df.columns.str.lower()
    return df

def parse_status(x):
    """Parse phenophase status to boolean or NaN."""
    if pd.isna(x):
        return np.nan
    
    # Numeric-like
    try:
        xi = int(float(x))
        if xi == 1:
            return True
        if xi == 0:
            return False
        if xi == -1:
            return np.nan  # Uncertain
    except Exception:
        pass
    
    # String-like
    s = str(x).strip().lower()
    if s in {"yes", "y", "true", "present"}:
        return True
    if s in {"no", "n", "false", "absent"}:
        return False
    if s in {"-1", "uncertain"}:
        return np.nan  # Uncertain
    
    return np.nan

def analyze_uncertain_observations():
    """Analyze uncertain observations and their impact on timing."""
    print("Loading observation data...")
    df = norm(pd.read_csv(OBS_DATA))
    
    # Parse dates
    df['observation_date'] = pd.to_datetime(df['observation_date'], errors='coerce')
    df = df.dropna(subset=['observation_date']).copy()
    df['year'] = df['observation_date'].dt.year
    df['doy'] = df['observation_date'].dt.dayofyear
    
    # Parse status
    df['status_parsed'] = df['phenophase_status'].map(parse_status)
    
    # Filter to our phenophases
    target_phenophases = ['Open flowers', 'Ripe fruits']
    df_filtered = df[df['phenophase_name'].str.contains('|'.join(target_phenophases), case=False, na=False)].copy()
    
    # Classify observations
    df_filtered['is_uncertain'] = df_filtered['status_parsed'].isna()
    df_filtered['is_yes'] = (df_filtered['status_parsed'] == True)
    df_filtered['is_no'] = (df_filtered['status_parsed'] == False)
    
    # Summary by phenophase and year
    summary_by_phase_year = df_filtered.groupby(['phenophase_name', 'year'], as_index=False).agg(
        total_obs=('observation_date', 'count'),
        n_uncertain=('is_uncertain', 'sum'),
        n_yes=('is_yes', 'sum'),
        n_no=('is_no', 'sum')
    )
    
    # Calculate percentages
    summary_by_phase_year['pct_uncertain'] = (summary_by_phase_year['n_uncertain'] / summary_by_phase_year['total_obs'] * 100).round(2)
    summary_by_phase_year['pct_yes'] = (summary_by_phase_year['n_yes'] / summary_by_phase_year['total_obs'] * 100).round(2)
    summary_by_phase_year['pct_no'] = (summary_by_phase_year['n_no'] / summary_by_phase_year['total_obs'] * 100).round(2)
    
    # Overall summary
    overall_summary = df_filtered.groupby('phenophase_name', as_index=False).agg(
        total_obs=('observation_date', 'count'),
        n_uncertain=('is_uncertain', 'sum'),
        n_yes=('is_yes', 'sum'),
        n_no=('is_no', 'sum')
    )
    overall_summary['pct_uncertain'] = (overall_summary['n_uncertain'] / overall_summary['total_obs'] * 100).round(2)
    
    # Analyze impact on intervals
    # For each individual-year-phenophase, check if excluding uncertain affects interval bounds
    impact_analysis = []
    
    for phenophase in target_phenophases:
        phen_df = df_filtered[df_filtered['phenophase_name'].str.contains(phenophase, case=False, na=False)].copy()
        
        for (ind_id, site_id, year), group in phen_df.groupby(['individual_id', 'site_id', 'year']):
            # All observations (including uncertain)
            all_dates = sorted(group['doy'].unique())
            all_yes_dates = sorted(group[group['is_yes']]['doy'].unique())
            
            # Without uncertain (only yes/no)
            group_clean = group[~group['is_uncertain']].copy()
            clean_dates = sorted(group_clean['doy'].unique())
            clean_yes_dates = sorted(group_clean[group_clean['is_yes']]['doy'].unique())
            
            # Check if first yes date changes
            first_yes_all = all_yes_dates[0] if len(all_yes_dates) > 0 else None
            first_yes_clean = clean_yes_dates[0] if len(clean_yes_dates) > 0 else None
            
            # Check if interval bounds would change
            if len(all_yes_dates) > 0:
                # Find last "no" before first "yes"
                no_before_yes_all = group[(group['is_no']) & (group['doy'] < first_yes_all)]
                no_before_yes_clean = group_clean[(group_clean['is_no']) & (group_clean['doy'] < first_yes_clean)] if first_yes_clean else pd.DataFrame()
                
                L_all = no_before_yes_all['doy'].max() if len(no_before_yes_all) > 0 else (all_dates[0] if len(all_dates) > 0 else None)
                L_clean = no_before_yes_clean['doy'].max() if len(no_before_yes_clean) > 0 else (clean_dates[0] if len(clean_dates) > 0 else None)
                
                R_all = first_yes_all
                R_clean = first_yes_clean
                
                interval_changed = (L_all != L_clean) or (R_all != R_clean)
                
                impact_analysis.append({
                    'phenophase': phenophase,
                    'individual_id': ind_id,
                    'site_id': site_id,
                    'year': year,
                    'n_uncertain': group['is_uncertain'].sum(),
                    'L_with_uncertain': L_all,
                    'R_with_uncertain': R_all,
                    'L_without_uncertain': L_clean,
                    'R_without_uncertain': R_clean,
                    'interval_changed': interval_changed
                })
    
    impact_df = pd.DataFrame(impact_analysis)
    
    return df_filtered, summary_by_phase_year, overall_summary, impact_df

def create_impact_plot(summary_by_phase_year, impact_df, output_path):
    """Create plot showing impact of excluding uncertain observations."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Percentage of uncertain observations by year
    for phenophase in ['Open flowers', 'Ripe fruits']:
        phen_data = summary_by_phase_year[summary_by_phase_year['phenophase_name'].str.contains(phenophase, case=False, na=False)]
        color = COLOR_FLOWERS if 'flower' in phenophase.lower() else COLOR_FRUITS
        ax1.plot(phen_data['year'], phen_data['pct_uncertain'], 
                marker='o', label=phenophase, color=color, linewidth=2, markersize=6)
    
    ax1.set_xlabel('Year', fontsize=12)
    ax1.set_ylabel('% Uncertain Observations', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Number of uncertain observations by year
    for phenophase in ['Open flowers', 'Ripe fruits']:
        phen_data = summary_by_phase_year[summary_by_phase_year['phenophase_name'].str.contains(phenophase, case=False, na=False)]
        color = COLOR_FLOWERS if 'flower' in phenophase.lower() else COLOR_FRUITS
        ax2.bar(phen_data['year'] + (0.2 if 'flower' in phenophase.lower() else -0.2),
                phen_data['n_uncertain'], width=0.4, label=phenophase, color=color, alpha=0.7)
    
    ax2.set_xlabel('Year', fontsize=12)
    ax2.set_ylabel('Number of Uncertain Observations', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Impact on intervals (how many changed)
    if len(impact_df) > 0:
        impact_summary = impact_df.groupby('phenophase', as_index=False).agg(
            total_intervals=('interval_changed', 'count'),
            intervals_changed=('interval_changed', 'sum'),
            intervals_unchanged=('interval_changed', lambda x: (~x).sum())
        )
        
        x_pos = np.arange(len(impact_summary))
        width = 0.35
        
        changed = impact_summary['intervals_changed'].values
        unchanged = impact_summary['intervals_unchanged'].values
        
        colors = [COLOR_FLOWERS if 'flower' in p.lower() else COLOR_FRUITS for p in impact_summary['phenophase']]
        
        ax3.bar(x_pos - width/2, unchanged, width, label='Unchanged', color='lightgray', alpha=0.7)
        ax3.bar(x_pos + width/2, changed, width, label='Changed', color=colors, alpha=0.7)
        
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels([p[:15] for p in impact_summary['phenophase']], fontsize=9)
        ax3.set_ylabel('Number of Intervals', fontsize=12)
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add percentage labels
        for i, (unch, chg, tot) in enumerate(zip(unchanged, changed, impact_summary['total_intervals'])):
            ax3.text(i - width/2, unch + 0.5, f'{int(unch)}', ha='center', va='bottom', fontsize=9)
            ax3.text(i + width/2, chg + 0.5, f'{int(chg)}', ha='center', va='bottom', fontsize=9)
            pct_changed = (chg / tot * 100) if tot > 0 else 0
            ax3.text(i, max(unch, chg) + 2, f'{pct_changed:.1f}%', ha='center', va='bottom', 
                    fontsize=10, fontweight='bold')
    
    # Plot 4: Distribution of uncertain observations by DOY
    if len(impact_df) > 0:
        for phenophase in ['Open flowers', 'Ripe fruits']:
            phen_impact = impact_df[impact_df['phenophase'].str.contains(phenophase, case=False, na=False)]
            if len(phen_impact) > 0:
                uncertain_counts = phen_impact['n_uncertain'].value_counts().sort_index()
                color = COLOR_FLOWERS if 'flower' in phenophase.lower() else COLOR_FRUITS
                ax4.bar(uncertain_counts.index, uncertain_counts.values, 
                       alpha=0.7, label=phenophase, color=color, width=0.8)
        
        ax4.set_xlabel('Number of Uncertain Observations per Interval', fontsize=12)
        ax4.set_ylabel('Frequency', fontsize=12)
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

# ==================== MAIN ====================
def main():
    print("=" * 60)
    print("RE2_P3: Uncertain Observations Impact Analysis")
    print("=" * 60)
    
    # Analyze uncertain observations
    print("\n1. Analyzing uncertain observations...")
    df_filtered, summary_by_phase_year, overall_summary, impact_df = analyze_uncertain_observations()
    
    print(f"\n   Overall Summary:")
    for _, row in overall_summary.iterrows():
        print(f"   {row['phenophase_name']}:")
        print(f"     Total observations: {row['total_obs']}")
        print(f"     Uncertain: {row['n_uncertain']} ({row['pct_uncertain']}%)")
        print(f"     Yes: {row['n_yes']}")
        print(f"     No: {row['n_no']}")
    
    if len(impact_df) > 0:
        print(f"\n2. Impact on intervals:")
        impact_summary = impact_df.groupby('phenophase', as_index=False).agg(
            total=('interval_changed', 'count'),
            changed=('interval_changed', 'sum')
        )
        for _, row in impact_summary.iterrows():
            pct = (row['changed'] / row['total'] * 100) if row['total'] > 0 else 0
            print(f"   {row['phenophase']}: {row['changed']}/{row['total']} intervals changed ({pct:.1f}%)")
    
    # Create plot
    print("\n3. Creating plot...")
    create_impact_plot(summary_by_phase_year, impact_df, OUT_PNG)
    print(f"   Saved: {OUT_PNG}")
    
    # Write Excel output
    print("\n4. Writing Excel output...")
    with pd.ExcelWriter(OUT_XLSX, engine='openpyxl') as xw:
        overall_summary.to_excel(xw, sheet_name='Overall_Summary', index=False)
        summary_by_phase_year.to_excel(xw, sheet_name='Summary_By_Phase_Year', index=False)
        if len(impact_df) > 0:
            impact_df.to_excel(xw, sheet_name='Interval_Impact', index=False)
            impact_summary = impact_df.groupby('phenophase', as_index=False).agg(
                total_intervals=('interval_changed', 'count'),
                intervals_changed=('interval_changed', 'sum'),
                intervals_unchanged=('interval_changed', lambda x: (~x).sum()),
                mean_uncertain_per_interval=('n_uncertain', 'mean')
            )
            impact_summary['pct_changed'] = (impact_summary['intervals_changed'] / impact_summary['total_intervals'] * 100).round(2)
            impact_summary.to_excel(xw, sheet_name='Impact_Summary', index=False)
    
    print(f"   Saved: {OUT_XLSX}")
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()

