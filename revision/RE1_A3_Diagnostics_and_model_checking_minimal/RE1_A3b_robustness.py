# RE1_A3b_robustness_analysis.py
"""
Reviewer 1, Comment A3 (continued): "Robustness to distributional mis-specification"
=====================================================================================
"Discussion of robustness to distributional mis-specification, especially 
with such a small sample (n = 50 plant–site–years)."

This script provides:
1. Comparison across multiple parametric families
2. Sensitivity to outliers (leave-k-out analysis)
3. Sensitivity to censoring assumptions
4. Parameter stability across subsamples

Outputs: revision/RE1_A3b_results/
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

try:
    sys.stdout.reconfigure(encoding="utf-8")
except:
    pass

# ---- Paths ----
PROJECT = Path(__file__).resolve().parent.parent
SURV_DATA = PROJECT / "07_validate_sensitivity" / "survival_with_weather_clean.xlsx"
OUT_DIR = PROJECT / "revision" / "RE1_A3b_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Style ----
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
COLOR_WEIBULL = '#E8743B'
COLOR_LOGLOGISTIC = '#6B5B95'
COLOR_LOGNORMAL = '#2E86AB'


def norm(df):
    """Normalize column names."""
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("-", "_")
    return df


def load_data(sheet_name):
    """Load survival data."""
    df = norm(pd.read_excel(SURV_DATA, sheet_name=sheet_name))
    return df


def fit_multiple_distributions(df):
    """Fit multiple parametric distributions for comparison."""
    from lifelines import WeibullAFTFitter, LogLogisticAFTFitter, LogNormalAFTFitter
    
    df_fit = df[['l', 'r']].copy()
    df_fit.columns = ['lower', 'upper']
    df_fit['upper'] = df_fit['upper'].fillna(400)
    
    results = []
    
    # Weibull
    try:
        wf = WeibullAFTFitter()
        wf.fit_interval_censoring(df_fit, lower_bound_col='lower', upper_bound_col='upper')
        results.append({
            'distribution': 'Weibull',
            'median': wf.median_survival_time_,
            'AIC': wf.AIC_,
            'log_likelihood': wf.log_likelihood_,
            'converged': True
        })
    except Exception as e:
        results.append({'distribution': 'Weibull', 'converged': False, 'error': str(e)})
    
    # Log-Logistic
    try:
        llf = LogLogisticAFTFitter()
        llf.fit_interval_censoring(df_fit, lower_bound_col='lower', upper_bound_col='upper')
        results.append({
            'distribution': 'Log-Logistic',
            'median': llf.median_survival_time_,
            'AIC': llf.AIC_,
            'log_likelihood': llf.log_likelihood_,
            'converged': True
        })
    except Exception as e:
        results.append({'distribution': 'Log-Logistic', 'converged': False, 'error': str(e)})
    
    # Log-Normal
    try:
        lnf = LogNormalAFTFitter()
        lnf.fit_interval_censoring(df_fit, lower_bound_col='lower', upper_bound_col='upper')
        results.append({
            'distribution': 'Log-Normal',
            'median': lnf.median_survival_time_,
            'AIC': lnf.AIC_,
            'log_likelihood': lnf.log_likelihood_,
            'converged': True
        })
    except Exception as e:
        results.append({'distribution': 'Log-Normal', 'converged': False, 'error': str(e)})
    
    return pd.DataFrame(results)


def leave_k_out_sensitivity(df, k=1, n_iterations=50):
    """Leave-k-out analysis for outlier sensitivity."""
    from lifelines import LogLogisticAFTFitter
    
    n = len(df)
    medians = []
    
    for i in range(min(n_iterations, n if k == 1 else n_iterations)):
        # Remove k random observations
        if k == 1:
            idx_remove = [i]
        else:
            idx_remove = np.random.choice(n, size=k, replace=False)
        
        df_sub = df.drop(df.index[idx_remove])
        
        df_fit = df_sub[['l', 'r']].copy()
        df_fit.columns = ['lower', 'upper']
        df_fit['upper'] = df_fit['upper'].fillna(400)
        
        try:
            llf = LogLogisticAFTFitter()
            llf.fit_interval_censoring(df_fit, lower_bound_col='lower', upper_bound_col='upper')
            medians.append(llf.median_survival_time_)
        except:
            continue
    
    if len(medians) < 5:
        return None
    
    return {
        'k': k,
        'n_iterations': len(medians),
        'median_mean': np.mean(medians),
        'median_sd': np.std(medians),
        'median_range': (np.min(medians), np.max(medians)),
        'cv': np.std(medians) / np.mean(medians) * 100
    }


def censoring_sensitivity(df):
    """Sensitivity to different censoring assumptions."""
    from lifelines import LogLogisticAFTFitter
    
    results = []
    
    # Original
    df_orig = df[['l', 'r']].copy()
    df_orig.columns = ['lower', 'upper']
    df_orig['upper'] = df_orig['upper'].fillna(400)
    
    try:
        llf = LogLogisticAFTFitter()
        llf.fit_interval_censoring(df_orig, lower_bound_col='lower', upper_bound_col='upper')
        results.append({
            'scenario': 'Original (R_max=400)',
            'median': llf.median_survival_time_,
            'AIC': llf.AIC_
        })
    except:
        pass
    
    # Different R_max values for right-censored
    for r_max in [300, 350, 450, 500]:
        df_test = df[['l', 'r']].copy()
        df_test.columns = ['lower', 'upper']
        df_test['upper'] = df_test['upper'].fillna(r_max)
        
        try:
            llf = LogLogisticAFTFitter()
            llf.fit_interval_censoring(df_test, lower_bound_col='lower', upper_bound_col='upper')
            results.append({
                'scenario': f'R_max={r_max}',
                'median': llf.median_survival_time_,
                'AIC': llf.AIC_
            })
        except:
            pass
    
    return pd.DataFrame(results)


def subsample_stability(df, fractions=[0.5, 0.6, 0.7, 0.8, 0.9], n_reps=30):
    """Test parameter stability across random subsamples."""
    from lifelines import LogLogisticAFTFitter
    
    results = []
    n = len(df)
    
    for frac in fractions:
        size = int(n * frac)
        medians = []
        
        for _ in range(n_reps):
            idx = np.random.choice(n, size=size, replace=False)
            df_sub = df.iloc[idx]
            
            df_fit = df_sub[['l', 'r']].copy()
            df_fit.columns = ['lower', 'upper']
            df_fit['upper'] = df_fit['upper'].fillna(400)
            
            try:
                llf = LogLogisticAFTFitter()
                llf.fit_interval_censoring(df_fit, lower_bound_col='lower', upper_bound_col='upper')
                medians.append(llf.median_survival_time_)
            except:
                continue
        
        if len(medians) >= 5:
            results.append({
                'fraction': frac,
                'n_obs': size,
                'n_successful': len(medians),
                'median_mean': np.mean(medians),
                'median_sd': np.std(medians),
                'cv': np.std(medians) / np.mean(medians) * 100
            })
    
    return pd.DataFrame(results)


def create_robustness_plot(dist_results, leave_out, censor_sens, sheet_name, outpath):
    """Create robustness analysis plot."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Distribution comparison
    ax1 = axes[0, 0]
    valid = dist_results[dist_results['converged'] == True]
    if len(valid) > 0:
        x = range(len(valid))
        colors = [COLOR_WEIBULL if 'Weibull' in d else 
                  COLOR_LOGLOGISTIC if 'Log-Logistic' in d else 
                  COLOR_LOGNORMAL for d in valid['distribution']]
        bars = ax1.bar(x, valid['median'], color=colors, alpha=0.8, edgecolor='black')
        ax1.set_xticks(x)
        ax1.set_xticklabels(valid['distribution'], rotation=45, ha='right')
        ax1.set_ylabel('Median Event Time (DOY)')
        ax1.set_title('Median Across Distributions', fontweight='bold')
        
        # Add AIC labels
        for bar, aic in zip(bars, valid['AIC']):
            ax1.annotate(f'AIC:{aic:.0f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)
        ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. Leave-k-out sensitivity
    ax2 = axes[0, 1]
    if leave_out:
        ax2.bar(['Leave-1-out'], [leave_out['cv']], color=COLOR_LOGLOGISTIC, 
                alpha=0.8, edgecolor='black')
        ax2.set_ylabel('Coefficient of Variation (%)')
        ax2.set_title('Outlier Sensitivity (CV of Median)', fontweight='bold')
        ax2.annotate(f'CV = {leave_out["cv"]:.1f}%\n(n={leave_out["n_iterations"]} iter)', 
                    xy=(0, leave_out['cv']), xytext=(0.2, leave_out['cv']*0.8),
                    fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Censoring sensitivity
    ax3 = axes[1, 0]
    if len(censor_sens) > 0:
        x = range(len(censor_sens))
        ax3.plot(x, censor_sens['median'], 'o-', color=COLOR_LOGLOGISTIC, 
                linewidth=2, markersize=8)
        ax3.set_xticks(x)
        ax3.set_xticklabels(censor_sens['scenario'], rotation=45, ha='right')
        ax3.set_ylabel('Median Event Time (DOY)')
        ax3.set_title('Sensitivity to Censoring Assumption', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Add range annotation
        med_range = censor_sens['median'].max() - censor_sens['median'].min()
        ax3.annotate(f'Range: {med_range:.1f} days', xy=(0.05, 0.95), 
                    xycoords='axes fraction', fontsize=10)
    
    # 4. Summary box
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = f"""
ROBUSTNESS SUMMARY - {sheet_name.replace('_', ' ').title()}

Distribution Sensitivity:
"""
    if len(valid) > 0:
        med_range = valid['median'].max() - valid['median'].min()
        summary_text += f"  • Median range: {med_range:.1f} days across distributions\n"
        best_dist = valid.loc[valid['AIC'].idxmin(), 'distribution']
        summary_text += f"  • Best by AIC: {best_dist}\n"
    
    if leave_out:
        summary_text += f"""
Outlier Sensitivity:
  • CV = {leave_out['cv']:.1f}% (leave-1-out)
  • Median range: [{leave_out['median_range'][0]:.1f}, {leave_out['median_range'][1]:.1f}]
"""
    
    if len(censor_sens) > 0:
        cens_range = censor_sens['median'].max() - censor_sens['median'].min()
        summary_text += f"""
Censoring Assumption Sensitivity:
  • Median varies by {cens_range:.1f} days across R_max values
  • {"Robust" if cens_range < 10 else "Moderate sensitivity"} to censoring assumption
"""
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle(f'Robustness Analysis - {sheet_name.replace("_", " ").title()}',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches='tight', format='png')
    plt.close()
    print(f"  Saved: {outpath.name}")


def main():
    print("="*70)
    print("RE1_A3b: Robustness and Sensitivity Analysis")
    print("="*70)
    
    all_results = {}
    
    for sheet in ["open_flowers", "ripe_fruits"]:
        print(f"\n{'─'*70}")
        print(f"Processing: {sheet.upper()}")
        print(f"{'─'*70}")
        
        df = load_data(sheet)
        
        # 1. Multiple distributions
        print("\n  [1/4] Fitting multiple distributions...")
        dist_results = fit_multiple_distributions(df)
        valid = dist_results[dist_results['converged'] == True]
        if len(valid) > 0:
            print(f"    Converged: {', '.join(valid['distribution'].tolist())}")
            print(f"    Median range: {valid['median'].min():.1f} - {valid['median'].max():.1f}")
        
        # 2. Leave-k-out
        print("\n  [2/4] Leave-1-out sensitivity...")
        leave_out = leave_k_out_sensitivity(df, k=1)
        if leave_out:
            print(f"    CV = {leave_out['cv']:.1f}%")
        
        # 3. Censoring sensitivity
        print("\n  [3/4] Censoring assumption sensitivity...")
        censor_sens = censoring_sensitivity(df)
        if len(censor_sens) > 0:
            print(f"    Median range: {censor_sens['median'].min():.1f} - {censor_sens['median'].max():.1f}")
        
        # 4. Subsample stability
        print("\n  [4/4] Subsample stability...")
        subsample = subsample_stability(df)
        if len(subsample) > 0:
            print(f"    CV at 50% subsample: {subsample[subsample['fraction']==0.5]['cv'].values[0]:.1f}%")
        
        # Create plot
        plot_path = OUT_DIR / f"RE1_A3b_robustness_{sheet}.png"
        create_robustness_plot(dist_results, leave_out, censor_sens, sheet, plot_path)
        
        all_results[sheet] = {
            'distributions': dist_results,
            'leave_out': leave_out,
            'censoring': censor_sens,
            'subsample': subsample
        }
    
    # Save results
    out_xlsx = OUT_DIR / "RE1_A3b_robustness_analysis.xlsx"
    
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as xw:
        for sheet, res in all_results.items():
            res['distributions'].to_excel(xw, index=False, sheet_name=f"{sheet}_distributions")
            res['censoring'].to_excel(xw, index=False, sheet_name=f"{sheet}_censoring")
            if len(res['subsample']) > 0:
                res['subsample'].to_excel(xw, index=False, sheet_name=f"{sheet}_subsample")
    
    print(f"\n{'='*70}")
    print(f"Results saved: {out_xlsx}")
    print(f"{'='*70}")
    
    # Text for response
    print("\n" + "="*70)
    print("TEXT FOR RESPONSE TO REVIEWERS:")
    print("="*70)
    
    flowers = all_results['open_flowers']
    fruits = all_results['ripe_fruits']
    
    print(f"""
We conducted comprehensive robustness analyses to address concerns about 
distributional mis-specification with small samples:

1. **Distribution Sensitivity**: We compared Weibull, Log-Logistic, and 
   Log-Normal AFT models. Median estimates varied by less than 
   {flowers['distributions'][flowers['distributions']['converged']==True]['median'].max() - flowers['distributions'][flowers['distributions']['converged']==True]['median'].min():.0f} days 
   for flowering and {fruits['distributions'][fruits['distributions']['converged']==True]['median'].max() - fruits['distributions'][fruits['distributions']['converged']==True]['median'].min():.0f} days 
   for fruiting across distributions.

2. **Outlier Sensitivity**: Leave-one-out analysis showed CV = 
   {flowers['leave_out']['cv']:.1f}% (flowering) and {fruits['leave_out']['cv']:.1f}% (fruiting), 
   indicating reasonable stability to individual observations.

3. **Censoring Assumptions**: Varying the right-censoring bound (R_max) 
   from 300 to 500 changed median estimates by less than 
   {flowers['censoring']['median'].max() - flowers['censoring']['median'].min():.0f} days, 
   demonstrating robustness to censoring specification.

These analyses support the reliability of our findings despite the 
modest sample size.
""")


if __name__ == "__main__":
    main()
