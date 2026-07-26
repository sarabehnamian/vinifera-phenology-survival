# A1_systematic_comparison.py
"""
Systematic Comparison to Alternative Methods
============================================
This script provides SYSTEMATIC COMPARISON TO ALTERNATIVES:
1. First-date OLS (naive - discards censored observations)
2. Midpoint OLS (naive - ignores censoring structure)
3. Kaplan-Meier (non-parametric baseline, midpoint approximation)
4. Turnbull estimator (non-parametric, proper interval censoring)
5. Cox PH (semi-parametric, midpoint approximation)
6. Weibull AFT (parametric, proper interval censoring)
7. Log-Logistic AFT (parametric, proper interval censoring)

Key outputs:
- Sample size comparison (showing naive methods waste data)
- AIC/BIC for principled model selection
- Median DOY comparison across methods

Outputs: supplementary_analyses/A1_results/
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
PROJECT = Path(__file__).resolve().parent.parent  # Go up from supplementary_analyses/ to project root
SURV_DATA = PROJECT / "07_validate_sensitivity" / "survival_with_weather_clean.xlsx"
OUT_DIR = PROJECT / "supplementary_analyses" / "A1_results"
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
    
    # Right-censoring: set R = +inf where missing
    df["r_filled"] = df["r"].where(df["r"].notna(), np.inf)
    
    # Fix degenerate intervals
    finite_r = np.isfinite(df["r_filled"])
    df.loc[finite_r, "r_filled"] = df.loc[finite_r, "r_filled"].clip(lower=1, upper=366)
    bad = finite_r & (df["r_filled"] <= df["l"])
    df.loc[bad, "r_filled"] = df.loc[bad, "l"] + 1e-6
    
    return df


def fit_first_date_ols(df):
    """Naive OLS using first-detected date (R). DISCARDS censored observations."""
    events = df[(df["event"] == 1) & np.isfinite(df["r_filled"])].copy()
    n_discarded = len(df) - len(events)
    
    return {
        "method": "First-date OLS",
        "n_total": len(df),
        "n_used": len(events),
        "n_discarded": n_discarded,
        "pct_discarded": 100 * n_discarded / len(df),
        "median_DOY": float(events["r_filled"].median()) if len(events) > 0 else np.nan,
        "handles_censoring": "No",
        "covariates": "Yes",
        "AIC": np.nan,
        "BIC": np.nan,
        "status": "success"
    }


def fit_midpoint_ols(df):
    """Naive OLS using interval midpoint. Uses all data but ignores censoring structure."""
    df_mid = df.copy()
    df_mid["midpoint"] = np.where(
        np.isfinite(df_mid["r_filled"]),
        (df_mid["l"] + df_mid["r_filled"]) / 2.0,
        df_mid["l"] + 30
    )
    
    return {
        "method": "Midpoint OLS",
        "n_total": len(df),
        "n_used": len(df),
        "n_discarded": 0,
        "pct_discarded": 0,
        "median_DOY": float(df_mid["midpoint"].median()),
        "handles_censoring": "Partial",
        "covariates": "Yes",
        "AIC": np.nan,
        "BIC": np.nan,
        "status": "success"
    }


def fit_kaplan_meier(df):
    """Kaplan-Meier (non-parametric baseline, midpoint approximation)."""
    try:
        from lifelines import KaplanMeierFitter
        
        df_km = df.copy()
        df_km["duration"] = np.where(
            np.isfinite(df_km["r_filled"]),
            (df_km["l"] + df_km["r_filled"]) / 2.0,
            df_km["l"] + 30
        )
        
        kmf = KaplanMeierFitter()
        kmf.fit(df_km["duration"], event_observed=df_km["event"])
        
        median = kmf.median_survival_time_
        if np.isinf(median):
            median = np.nan
        
        return {
            "method": "Kaplan-Meier*",
            "n_total": len(df),
            "n_used": len(df),
            "n_discarded": 0,
            "pct_discarded": 0,
            "median_DOY": float(median) if not np.isnan(median) else np.nan,
            "handles_censoring": "Yes",
            "covariates": "No",
            "AIC": np.nan,
            "BIC": np.nan,
            "status": "success"
        }
    except Exception as e:
        print(f"    Kaplan-Meier fit failed: {e}")
        return {
            "method": "Kaplan-Meier*",
            "n_total": len(df),
            "n_used": len(df),
            "n_discarded": 0,
            "pct_discarded": 0,
            "median_DOY": np.nan,
            "handles_censoring": "Yes",
            "covariates": "No",
            "status": f"failed: {e}"
        }


def fit_turnbull(df):
    """
    Turnbull estimator (non-parametric, proper interval censoring).
    Implements the self-consistency algorithm for interval-censored data.
    """
    try:
        # Prepare intervals: [L, R] for interval-censored, [L, inf] for right-censored
        intervals = []
        for idx, row in df.iterrows():
            l_val = row["l"]
            r_val = row["r_filled"]
            if np.isfinite(r_val):
                intervals.append((l_val, r_val))
            else:
                intervals.append((l_val, np.inf))
        
        # Get unique interval endpoints
        endpoints = set()
        for l_val, r_val in intervals:
            endpoints.add(l_val)
            if np.isfinite(r_val):
                endpoints.add(r_val)
        endpoints = sorted(endpoints)
        
        if len(endpoints) < 2:
            return {
                "method": "Turnbull",
                "n_total": len(df),
                "n_used": len(df),
                "n_discarded": 0,
                "pct_discarded": 0,
                "median_DOY": np.nan,
                "handles_censoring": "Yes",
                "covariates": "No",
                "AIC": np.nan,
                "BIC": np.nan,
                "status": "failed: insufficient data"
            }
        
        # Initialize probability masses (Turnbull's self-consistency algorithm)
        n_intervals = len(intervals)
        n_endpoints = len(endpoints) - 1  # Number of intervals between endpoints
        
        # Create indicator matrix: which intervals contain which endpoint intervals
        indicator = np.zeros((n_intervals, n_endpoints), dtype=bool)
        for i, (l_val, r_val) in enumerate(intervals):
            for j in range(n_endpoints):
                if endpoints[j] >= l_val and (not np.isfinite(r_val) or endpoints[j+1] <= r_val):
                    indicator[i, j] = True
        
        # Initialize equal probability masses
        p = np.ones(n_endpoints) / n_endpoints
        
        # Self-consistency iteration (EM algorithm)
        max_iter = 1000
        tol = 1e-6
        for iteration in range(max_iter):
            p_old = p.copy()
            
            # E-step: compute expected counts
            expected_counts = np.zeros(n_endpoints)
            for j in range(n_endpoints):
                # Sum over observations that contain interval j
                mask = indicator[:, j]
                if mask.sum() > 0:
                    # For each observation containing interval j, weight by p[j] / sum(p over all intervals in that observation)
                    for i in range(n_intervals):
                        if indicator[i, j]:
                            # Sum of probabilities for all intervals contained in observation i
                            prob_sum = (indicator[i, :] @ p)
                            if prob_sum > 0:
                                expected_counts[j] += p[j] / prob_sum
            
            # M-step: normalize
            if expected_counts.sum() > 0:
                p = expected_counts / expected_counts.sum()
            else:
                break
            
            # Check convergence
            if np.max(np.abs(p - p_old)) < tol:
                break
        
        # Compute survival function
        survival = np.ones(len(endpoints))
        for i in range(1, len(endpoints)):
            survival[i] = survival[i-1] - p[i-1]
        
        # Find median (where survival = 0.5)
        median = np.nan
        for i in range(len(survival) - 1):
            if survival[i] >= 0.5 and survival[i+1] < 0.5:
                # Linear interpolation
                frac = (survival[i] - 0.5) / (survival[i] - survival[i+1])
                median = endpoints[i] + frac * (endpoints[i+1] - endpoints[i])
                break
        
        # If median not found, use last point where survival > 0.5
        if np.isnan(median):
            for i in range(len(survival) - 1, -1, -1):
                if survival[i] >= 0.5:
                    median = endpoints[i]
                    break
        
        return {
            "method": "Turnbull",
            "n_total": len(df),
            "n_used": len(df),
            "n_discarded": 0,
            "pct_discarded": 0,
            "median_DOY": float(median) if not np.isnan(median) else np.nan,
            "handles_censoring": "Yes",
            "covariates": "No",
            "AIC": np.nan,
            "BIC": np.nan,
            "status": "success"
        }
    except Exception as e:
        print(f"    Turnbull fit failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "method": "Turnbull",
            "n_total": len(df),
            "n_used": len(df),
            "n_discarded": 0,
            "pct_discarded": 0,
            "median_DOY": np.nan,
            "handles_censoring": "Yes",
            "covariates": "No",
            "status": f"failed: {e}"
        }


def fit_cox_ph(df):
    """Cox PH (semi-parametric, midpoint approximation)."""
    try:
        from lifelines import CoxPHFitter
        
        df_cox = df.copy()
        df_cox["duration"] = np.where(
            np.isfinite(df_cox["r_filled"]),
            (df_cox["l"] + df_cox["r_filled"]) / 2.0,
            df_cox["l"] + 30
        )
        df_cox["observed"] = df_cox["event"].astype(int)
        
        cox = CoxPHFitter()
        cox.fit(df_cox[["duration", "observed"]], duration_col="duration", event_col="observed")
        
        surv = cox.baseline_survival_
        median_idx = np.where(surv.values.flatten() <= 0.5)[0]
        median = float(surv.index[median_idx[0]]) if len(median_idx) > 0 else np.nan
        
        return {
            "method": "Cox PH*",
            "n_total": len(df),
            "n_used": len(df),
            "n_discarded": 0,
            "pct_discarded": 0,
            "median_DOY": median,
            "handles_censoring": "Yes",
            "covariates": "No (baseline)",
            "AIC": np.nan,
            "BIC": np.nan,
            "status": "success"
        }
    except Exception as e:
        print(f"    Cox PH fit failed: {e}")
        return {
            "method": "Cox PH*",
            "n_total": len(df),
            "n_used": len(df),
            "n_discarded": 0,
            "pct_discarded": 0,
            "median_DOY": np.nan,
            "handles_censoring": "Yes",
            "covariates": "No (baseline)",
            "status": f"failed: {e}"
        }


def fit_weibull_aft(df):
    """Weibull AFT with proper interval censoring."""
    try:
        from lifelines import WeibullAFTFitter
        
        # Create clean dataframe with only interval columns
        # Keep np.inf for right-censored observations (lifelines accepts this)
        df_aft = pd.DataFrame({
            "lower": df["l"].values,
            "upper": df["r_filled"].values  # Keep np.inf for right-censored
        })
        
        # Remove any rows with NaN in lower (shouldn't happen, but safety check)
        df_aft = df_aft.dropna(subset=["lower"])
        
        wf = WeibullAFTFitter()
        wf.fit_interval_censoring(df_aft, lower_bound_col="lower", upper_bound_col="upper")
        
        n_params = len(wf.params_)
        n_obs = len(df)
        bic = n_params * np.log(n_obs) - 2 * wf.log_likelihood_
        
        return {
            "method": "Weibull AFT",
            "n_total": len(df),
            "n_used": len(df),
            "n_discarded": 0,
            "pct_discarded": 0,
            "median_DOY": float(wf.median_survival_time_),
            "handles_censoring": "Yes",
            "covariates": "No (baseline)",
            "log_likelihood": wf.log_likelihood_,
            "n_parameters": n_params,
            "AIC": wf.AIC_,
            "BIC": bic,
            "status": "success"
        }
    except Exception as e:
        print(f"    Weibull AFT fit failed: {e}")
        return {"method": "Weibull AFT", "n_total": len(df), "n_used": 0, "status": f"failed: {e}"}


def fit_loglogistic_aft(df):
    """Log-Logistic AFT with proper interval censoring."""
    try:
        from lifelines import LogLogisticAFTFitter
        
        # Create clean dataframe with only interval columns
        # Keep np.inf for right-censored observations (lifelines accepts this)
        df_aft = pd.DataFrame({
            "lower": df["l"].values,
            "upper": df["r_filled"].values  # Keep np.inf for right-censored
        })
        
        # Remove any rows with NaN in lower (shouldn't happen, but safety check)
        df_aft = df_aft.dropna(subset=["lower"])
        
        llf = LogLogisticAFTFitter()
        llf.fit_interval_censoring(df_aft, lower_bound_col="lower", upper_bound_col="upper")
        
        n_params = len(llf.params_)
        n_obs = len(df)
        bic = n_params * np.log(n_obs) - 2 * llf.log_likelihood_
        
        return {
            "method": "Log-Logistic AFT",
            "n_total": len(df),
            "n_used": len(df),
            "n_discarded": 0,
            "pct_discarded": 0,
            "median_DOY": float(llf.median_survival_time_),
            "handles_censoring": "Yes",
            "covariates": "No (baseline)",
            "log_likelihood": llf.log_likelihood_,
            "n_parameters": n_params,
            "AIC": llf.AIC_,
            "BIC": bic,
            "status": "success"
        }
    except Exception as e:
        print(f"    Log-Logistic AFT fit failed: {e}")
        return {"method": "Log-Logistic AFT", "n_total": len(df), "n_used": 0, "status": f"failed: {e}"}


def create_comparison_plot(results, sheet_name, outpath):
    """Create comparison plot."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5))
    
    valid = [r for r in results if "success" in str(r.get("status", ""))]
    methods = [r["method"] for r in valid]
    # Remove asterisks from method names for plot display
    methods_display = [m.replace('*', '') for m in methods]
    n_used = [r.get("n_used", 0) for r in valid]
    n_disc = [r.get("n_discarded", 0) for r in valid]
    medians = [r.get("median_DOY", np.nan) for r in valid]
    
    color = COLOR_FLOWERS if "flower" in sheet_name.lower() else COLOR_FRUITS
    x = np.arange(len(methods))
    
    # Plot 1: Sample sizes
    ax1.bar(x, n_used, label="Used", color=color, alpha=0.9, edgecolor='black', linewidth=1)
    ax1.bar(x, n_disc, bottom=n_used, label="Discarded", color='lightgray', 
            alpha=0.9, edgecolor='black', linewidth=1)
    
    for i, (u, d) in enumerate(zip(n_used, n_disc)):
        if d > 0:
            ax1.annotate(f'{100*d/(u+d):.0f}%\nlost', xy=(i, u+d), ha='center', 
                        va='bottom', fontsize=9, color='red', 
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'))
    
    # Add more space at top
    max_val = max([u + d for u, d in zip(n_used, n_disc)])
    ax1.set_ylim(0, max_val * 1.25)  # 25% padding at top
    
    ax1.set_ylabel("Number of Observations", fontsize=12)
    ax1.set_title("Sample Size: Used vs Discarded", fontsize=13)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods_display, rotation=45, ha="right", fontsize=10)
    ax1.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Median DOY
    valid_med = [(m, med) for m, med in zip(methods_display, medians) if not np.isnan(med)]
    if valid_med:
        m_names, m_vals = zip(*valid_med)
        bars = ax2.bar(range(len(m_names)), m_vals, color=color, alpha=0.8, 
                       edgecolor='black', linewidth=1)
        for bar, val in zip(bars, m_vals):
            ax2.annotate(f'{val:.1f}', xy=(bar.get_x() + bar.get_width()/2, val),
                        xytext=(0, 5), textcoords="offset points", ha='center', 
                        va='bottom', fontsize=10)
        
        # Add more space at top
        max_val = max(m_vals)
        min_val = min(m_vals)
        range_val = max_val - min_val
        ax2.set_ylim(min_val - range_val * 0.05, max_val + range_val * 0.25)  # 25% padding at top
        
        ax2.set_ylabel("Median DOY", fontsize=12)
        ax2.set_title("Median Phenology Timing by Method", fontsize=13)
        ax2.set_xticks(range(len(m_names)))
        ax2.set_xticklabels(m_names, rotation=45, ha="right", fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f"{sheet_name.replace('_', ' ').title()}", fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 0.95, 0.96])  # Leave space on right for legend, top for suptitle
    fig.savefig(outpath, dpi=300, bbox_inches="tight", format='png')
    plt.close()
    print(f"  Saved: {outpath.name}")


def print_table(results, sheet_name):
    """Print comparison table."""
    print(f"\n  {'─'*95}")
    print(f"  {'Method':<20} {'n_used':>8} {'n_disc':>8} {'%lost':>7} {'Censor':>8} {'Covar':>10} {'Median':>10} {'AIC':>10}")
    print(f"  {'─'*95}")
    
    for r in results:
        if "success" not in str(r.get("status", "")):
            continue
        method = r['method']
        n_used = r.get('n_used', 0)
        n_disc = r.get('n_discarded', 0)
        pct = r.get('pct_discarded', 0)
        censor = r.get('handles_censoring', '?')
        covar = str(r.get('covariates', '?'))[:10]
        median = r.get('median_DOY', np.nan)
        aic = r.get('AIC', np.nan)
        
        median_str = f"{median:>10.1f}" if not np.isnan(median) else "       —"
        aic_str = f"{aic:>10.1f}" if not np.isnan(aic) else "       —"
        
        print(f"  {method:<20} {n_used:>8} {n_disc:>8} {pct:>6.1f}% {censor:>8} "
              f"{covar:>10} {median_str} {aic_str}")
    
    print(f"  {'─'*95}")
    print(f"  * Uses midpoint approximation for interval censoring")


def main():
    print("="*70)
    print("A1: Systematic Comparison to Alternative Methods")
    print("="*70)
    
    all_results = []
    
    for sheet in ["open_flowers", "ripe_fruits"]:
        print(f"\n{'─'*70}")
        print(f"Processing: {sheet.upper()}")
        print(f"{'─'*70}")
        
        df = load_data(sheet)
        n_events = int(df["event"].sum())
        print(f"  Data: {len(df)} obs ({n_events} events, {len(df)-n_events} censored)")
        
        results = []
        
        print("  [1/6] First-date OLS...")
        results.append(fit_first_date_ols(df))
        
        print("  [2/6] Midpoint OLS...")
        results.append(fit_midpoint_ols(df))
        
        print("  [3/7] Kaplan-Meier...")
        results.append(fit_kaplan_meier(df))
        
        print("  [4/7] Turnbull estimator...")
        results.append(fit_turnbull(df))
        
        print("  [5/7] Cox PH...")
        results.append(fit_cox_ph(df))
        
        print("  [6/7] Weibull AFT...")
        results.append(fit_weibull_aft(df))
        
        print("  [7/7] Log-Logistic AFT...")
        results.append(fit_loglogistic_aft(df))
        
        # Add sheet name
        for r in results:
            r["sheet"] = sheet
        
        print_table(results, sheet)
        
        # Plot
        plot_path = OUT_DIR / f"A1_comparison_{sheet}.png"
        create_comparison_plot(results, sheet, plot_path)
        
        all_results.extend(results)
    
    # Save to Excel
    out_xlsx = OUT_DIR / "A1_systematic_comparison.xlsx"
    summary_df = pd.DataFrame(all_results)
    
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as xw:
        summary_df.to_excel(xw, index=False, sheet_name="all_results")
        
        # Clean table for paper
        paper_cols = ["sheet", "method", "n_total", "n_used", "n_discarded", "pct_discarded",
                      "handles_censoring", "covariates", "median_DOY", "AIC", "BIC"]
        paper_df = summary_df[[c for c in paper_cols if c in summary_df.columns]]
        paper_df.to_excel(xw, index=False, sheet_name="table_for_paper")
    
    print(f"\n{'='*70}")
    print(f"Results saved: {out_xlsx}")
    print(f"{'='*70}")
    
    # Key message
    print("\n" + "="*70)
    print("KEY FINDINGS:")
    print("="*70)
    first_date = [r for r in all_results if r["method"] == "First-date OLS" and "success" in str(r.get("status",""))]
    if first_date:
        avg_lost = np.mean([r["pct_discarded"] for r in first_date])
        print(f"  → First-date OLS discards {avg_lost:.0f}% of observations")
    print(f"  → Interval-censored AFT uses ALL data with proper likelihood")
    
    aft_results = [r for r in all_results if "AFT" in r["method"] and not np.isnan(r.get("AIC", np.nan))]
    if aft_results:
        weibull_aic = [r["AIC"] for r in aft_results if "Weibull" in r["method"]]
        ll_aic = [r["AIC"] for r in aft_results if "Log-Logistic" in r["method"]]
        if weibull_aic and ll_aic:
            better = "Log-Logistic" if np.mean(ll_aic) < np.mean(weibull_aic) else "Weibull"
            print(f"  → AIC favors {better} over alternative")


if __name__ == "__main__":
    main()
