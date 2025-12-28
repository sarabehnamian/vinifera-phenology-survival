# RE1_A3_diagnostics.py
"""
Reviewer 1, Comment A3: "Diagnostics and model checking are minimal"
=====================================================================
"For a statistics journal, I would expect:
- Comparison to non-parametric interval-censored estimators (Turnbull) [see A1]
- Examination of goodness-of-fit (e.g., residuals, QQ-plots, hazard shapes)
- Discussion of robustness to distributional mis-specification
- Principled justification for model selection"

This script provides:
1. Cox-Snell residuals for Weibull and Log-Logistic models
2. QQ-plots for distributional fit
3. Hazard shape plots
4. AIC/BIC comparison for model selection
5. Predicted vs observed comparison

Outputs: revision/RE1_A3_results/
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

try:
    sys.stdout.reconfigure(encoding="utf-8")
except:
    pass

# ---- Paths ----
PROJECT = Path(__file__).resolve().parent.parent
SURV_DATA = PROJECT / "07_validate_sensitivity" / "survival_with_weather_clean.xlsx"
OUT_DIR = PROJECT / "revision" / "RE1_A3_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Style ----
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
COLOR_FLOWERS = '#E8743B'  # Orange for flowers
COLOR_FRUITS = '#6B5B95'    # Purple for fruits


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


def fit_models(df):
    """Fit Weibull and Log-Logistic AFT models."""
    from lifelines import WeibullAFTFitter, LogLogisticAFTFitter
    
    # Create clean dataframe with only interval columns
    # Keep np.inf for right-censored observations (lifelines accepts this)
    df_fit = pd.DataFrame({
        "lower": df["l"].values,
        "upper": df["r_filled"].values  # Keep np.inf for right-censored
    })
    
    # Remove any rows with NaN in lower (shouldn't happen, but safety check)
    df_fit = df_fit.dropna(subset=["lower"])
    
    results = {}
    
    # Weibull
    try:
        wf = WeibullAFTFitter()
        wf.fit_interval_censoring(df_fit, lower_bound_col="lower", upper_bound_col="upper")
        n_params = len(wf.params_)
        bic = n_params * np.log(len(df_fit)) - 2 * wf.log_likelihood_
        results['Weibull'] = {
            'fitter': wf,
            'log_likelihood': wf.log_likelihood_,
            'AIC': wf.AIC_,
            'BIC': bic,
            'n_parameters': n_params,
            'lambda_': float(wf.params_.loc['lambda_', 'Intercept']),
            'rho_': float(wf.params_.loc['rho_', 'Intercept'])
        }
    except Exception as e:
        print(f"  Weibull fit failed: {e}")
    
    # Log-Logistic
    try:
        llf = LogLogisticAFTFitter()
        llf.fit_interval_censoring(df_fit, lower_bound_col="lower", upper_bound_col="upper")
        n_params = len(llf.params_)
        bic = n_params * np.log(len(df_fit)) - 2 * llf.log_likelihood_
        results['LogLogistic'] = {
            'fitter': llf,
            'log_likelihood': llf.log_likelihood_,
            'AIC': llf.AIC_,
            'BIC': bic,
            'n_parameters': n_params,
            'alpha_': float(llf.params_.loc['alpha_', 'Intercept']),
            'beta_': float(llf.params_.loc['beta_', 'Intercept'])
        }
    except Exception as e:
        print(f"  Log-Logistic fit failed: {e}")
    
    return results


def compute_cox_snell_residuals(df, model_results, model_name):
    """
    Compute Cox-Snell residuals for interval-censored data.
    For proper interval censoring, we use the midpoint approximation.
    """
    fitter = model_results[model_name]['fitter']
    
    df_cs = df.copy()
    # Use midpoint for interval-censored
    df_cs['t'] = np.where(
        np.isfinite(df_cs['r_filled']),
        (df_cs['l'] + df_cs['r_filled']) / 2.0,
        df_cs['l'] + 30
    )
    
    # Cumulative hazard at observed/imputed time
    # For AFT models: H(t) = -log(S(t))
    try:
        # Predict survival at each time point
        times = df_cs['t'].values
        
        if model_name == 'Weibull':
            # Weibull: S(t) = exp(-(t/lambda)^rho)
            lambda_ = np.exp(model_results['Weibull']['lambda_'])
            rho_ = np.exp(model_results['Weibull']['rho_'])
            survival = np.exp(-(times / lambda_) ** rho_)
        else:
            # Log-Logistic: S(t) = 1 / (1 + (t/alpha)^beta)
            alpha_ = np.exp(model_results['LogLogistic']['alpha_'])
            beta_ = np.exp(model_results['LogLogistic']['beta_'])
            survival = 1 / (1 + (times / alpha_) ** beta_)
        
        # Cox-Snell residuals: r_i = H(t_i) = -log(S(t_i))
        cs_residuals = -np.log(np.clip(survival, 1e-10, 1))
        
        return cs_residuals, df_cs['event'].values
    
    except Exception as e:
        print(f"  Error computing residuals: {e}")
        return None, None


def create_qq_plot(residuals, events, model_name, sheet_name, outpath):
    """Create QQ-plot for Cox-Snell residuals."""
    if residuals is None:
        return
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # For Cox-Snell residuals, should follow Exp(1) if model is correct
    # Use only events for cleaner visualization
    event_residuals = residuals[events == 1]
    
    # Theoretical quantiles (exponential)
    n = len(event_residuals)
    theoretical = -np.log(1 - (np.arange(1, n+1) - 0.5) / n)
    observed = np.sort(event_residuals)
    
    # Use endpoint-specific color with shades for different models
    base_color = COLOR_FLOWERS if 'flower' in sheet_name.lower() else COLOR_FRUITS
    if 'Weibull' in model_name:
        color = base_color
    else:
        # Use darker shade for Log-Logistic
        import matplotlib.colors as mcolors
        rgb = mcolors.hex2color(base_color)
        color = mcolors.rgb2hex([max(0, c * 0.7) for c in rgb])  # Darker shade
    
    ax.scatter(theoretical, observed, alpha=0.6, color=color, edgecolor='black', s=50)
    
    # Add 45-degree line
    max_val = max(theoretical.max(), observed.max())
    ax.plot([0, max_val], [0, max_val], 'k--', linewidth=2, label='Perfect fit')
    
    ax.set_xlabel('Theoretical Quantiles (Exp(1))')
    ax.set_ylabel('Cox-Snell Residuals')
    ax.set_title(f'{model_name} - Cox-Snell QQ Plot\n{sheet_name.replace("_", " ").title()}', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches='tight', format='png')
    plt.close()
    print(f"  Saved: {outpath.name}")


def create_hazard_plot(model_results, sheet_name, outpath):
    """Create hazard function comparison plot."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    times = np.linspace(50, 250, 200)
    
    # Use endpoint-specific color with shades for different models
    base_color = COLOR_FLOWERS if 'flower' in sheet_name.lower() else COLOR_FRUITS
    import matplotlib.colors as mcolors
    
    for idx, model_name in enumerate([('Weibull', base_color), ('LogLogistic', None)]):
        if model_name[0] not in model_results:
            continue
        
        # Use base color for Weibull, darker shade for Log-Logistic
        if idx == 0:
            color = base_color
        else:
            rgb = mcolors.hex2color(base_color)
            color = mcolors.rgb2hex([max(0, c * 0.7) for c in rgb])  # Darker shade
        
        if model_name[0] == 'Weibull':
            lambda_ = np.exp(model_results['Weibull']['lambda_'])
            rho_ = np.exp(model_results['Weibull']['rho_'])
            # Weibull hazard: h(t) = (rho/lambda) * (t/lambda)^(rho-1)
            hazard = (rho_ / lambda_) * (times / lambda_) ** (rho_ - 1)
        else:
            alpha_ = np.exp(model_results['LogLogistic']['alpha_'])
            beta_ = np.exp(model_results['LogLogistic']['beta_'])
            # Log-Logistic hazard: h(t) = (beta/alpha) * (t/alpha)^(beta-1) / (1 + (t/alpha)^beta)
            hazard = (beta_ / alpha_) * (times / alpha_) ** (beta_ - 1) / (1 + (times / alpha_) ** beta_)
        
        ax.plot(times, hazard, color=color, linewidth=2.5, label=model_name[0])
    
    ax.set_xlabel('Day of Year')
    ax.set_ylabel('Hazard Rate h(t)')
    ax.set_title(f'Hazard Functions - {sheet_name.replace("_", " ").title()}', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(50, 250)
    
    plt.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches='tight', format='png')
    plt.close()
    print(f"  Saved: {outpath.name}")


def create_survival_comparison_plot(df, model_results, sheet_name, outpath):
    """Compare fitted survival curves to Kaplan-Meier."""
    from lifelines import KaplanMeierFitter
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Kaplan-Meier (midpoint approximation)
    df_km = df.copy()
    df_km['duration'] = np.where(
        np.isfinite(df_km['r_filled']),
        (df_km['l'] + df_km['r_filled']) / 2.0,
        df_km['l'] + 30
    )
    
    kmf = KaplanMeierFitter()
    kmf.fit(df_km['duration'], event_observed=df_km['event'])
    kmf.plot_survival_function(ax=ax, color='black', linewidth=2, label='Kaplan-Meier')
    
    # Fitted models
    times = np.linspace(50, 250, 200)
    
    # Use endpoint-specific color with shades for different models
    base_color = COLOR_FLOWERS if 'flower' in sheet_name.lower() else COLOR_FRUITS
    import matplotlib.colors as mcolors
    
    for idx, model_name in enumerate(['Weibull', 'LogLogistic']):
        if model_name not in model_results:
            continue
        
        # Use base color for Weibull, darker shade for Log-Logistic
        if idx == 0:
            color = base_color
        else:
            rgb = mcolors.hex2color(base_color)
            color = mcolors.rgb2hex([max(0, c * 0.7) for c in rgb])  # Darker shade
        
        if model_name == 'Weibull':
            lambda_ = np.exp(model_results['Weibull']['lambda_'])
            rho_ = np.exp(model_results['Weibull']['rho_'])
            survival = np.exp(-(times / lambda_) ** rho_)
        else:
            alpha_ = np.exp(model_results['LogLogistic']['alpha_'])
            beta_ = np.exp(model_results['LogLogistic']['beta_'])
            survival = 1 / (1 + (times / alpha_) ** beta_)
        
        ax.plot(times, survival, color=color, linewidth=2.5, linestyle='--', label=f'{model_name} AFT')
    
    ax.set_xlabel('Day of Year')
    ax.set_ylabel('Survival Probability S(t)')
    ax.set_title(f'Survival Function Comparison - {sheet_name.replace("_", " ").title()}', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(50, 250)
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches='tight', format='png')
    plt.close()
    print(f"  Saved: {outpath.name}")


def create_model_selection_table(all_results):
    """Create model selection table with AIC/BIC."""
    rows = []
    for sheet, models in all_results.items():
        for model_name, res in models.items():
            rows.append({
                'Endpoint': sheet.replace('_', ' ').title(),
                'Model': model_name,
                'Log-Likelihood': res['log_likelihood'],
                'K (parameters)': res['n_parameters'],
                'AIC': res['AIC'],
                'BIC': res['BIC']
            })
    
    if not rows:
        return pd.DataFrame(columns=['Endpoint', 'Model', 'Log-Likelihood', 'K (parameters)', 'AIC', 'BIC'])
    
    df = pd.DataFrame(rows)
    
    # Add delta AIC/BIC
    for sheet in df['Endpoint'].unique():
        mask = df['Endpoint'] == sheet
        min_aic = df.loc[mask, 'AIC'].min()
        min_bic = df.loc[mask, 'BIC'].min()
        df.loc[mask, 'ΔAIC'] = df.loc[mask, 'AIC'] - min_aic
        df.loc[mask, 'ΔBIC'] = df.loc[mask, 'BIC'] - min_bic
    
    return df


def main():
    print("="*70)
    print("RE1_A3: Diagnostics and Model Checking")
    print("="*70)
    
    all_model_results = {}
    all_diagnostics = []
    
    for sheet in ["open_flowers", "ripe_fruits"]:
        print(f"\n{'─'*70}")
        print(f"Processing: {sheet.upper()}")
        print(f"{'─'*70}")
        
        # Load data
        df = load_data(sheet)
        print(f"  Data: {len(df)} obs ({int(df['event'].sum())} events)")
        
        # Fit models
        print("  Fitting models...")
        model_results = fit_models(df)
        all_model_results[sheet] = model_results
        
        # Print AIC/BIC
        print(f"\n  Model Selection Criteria:")
        print(f"  {'Model':<15} {'LL':>12} {'AIC':>12} {'BIC':>12}")
        print(f"  {'-'*55}")
        for name, res in model_results.items():
            print(f"  {name:<15} {res['log_likelihood']:>12.2f} {res['AIC']:>12.2f} {res['BIC']:>12.2f}")
        
        # Determine best model
        if 'Weibull' in model_results and 'LogLogistic' in model_results:
            best_aic = 'Log-Logistic' if model_results['LogLogistic']['AIC'] < model_results['Weibull']['AIC'] else 'Weibull'
            best_bic = 'Log-Logistic' if model_results['LogLogistic']['BIC'] < model_results['Weibull']['BIC'] else 'Weibull'
            print(f"\n  → Best by AIC: {best_aic}")
            print(f"  → Best by BIC: {best_bic}")
        
        # Cox-Snell residuals and QQ plots
        print("\n  Computing diagnostics...")
        for model_name in model_results:
            residuals, events = compute_cox_snell_residuals(df, model_results, model_name)
            
            if residuals is not None:
                # QQ plot
                qq_path = OUT_DIR / f"RE1_A3_qq_{model_name.lower()}_{sheet}.png"
                create_qq_plot(residuals, events, model_name, sheet, qq_path)
                
                # Store diagnostics
                all_diagnostics.append({
                    'sheet': sheet,
                    'model': model_name,
                    'mean_residual': np.mean(residuals[events == 1]),
                    'std_residual': np.std(residuals[events == 1]),
                    'expected_mean': 1.0,  # Exp(1) has mean 1
                })
        
        # Hazard plot
        hazard_path = OUT_DIR / f"RE1_A3_hazard_{sheet}.png"
        create_hazard_plot(model_results, sheet, hazard_path)
        
        # Survival comparison plot
        surv_path = OUT_DIR / f"RE1_A3_survival_comparison_{sheet}.png"
        create_survival_comparison_plot(df, model_results, sheet, surv_path)
    
    # Save results
    out_xlsx = OUT_DIR / "RE1_A3_diagnostics.xlsx"
    
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as xw:
        # Model selection table
        model_table = create_model_selection_table(all_model_results)
        model_table.to_excel(xw, index=False, sheet_name="model_selection")
        
        # Diagnostics
        pd.DataFrame(all_diagnostics).to_excel(xw, index=False, sheet_name="residual_diagnostics")
        
        # README
        readme = [
            "RE1_A3: Diagnostics and Model Checking",
            "",
            "Addresses reviewer concern about minimal diagnostics.",
            "",
            "Contents:",
            "1. model_selection: AIC/BIC comparison for Weibull vs Log-Logistic",
            "2. residual_diagnostics: Cox-Snell residual summary statistics",
            "",
            "Plots generated:",
            "- QQ plots for Cox-Snell residuals (should follow Exp(1) line)",
            "- Hazard function comparison (Weibull vs Log-Logistic)",
            "- Survival function comparison (KM vs fitted models)",
            "",
            "Interpretation:",
            "- Points on QQ plot close to 45° line indicate good fit",
            "- Mean of Cox-Snell residuals should be ≈ 1 for good fit",
            "- Lower AIC/BIC indicates better model"
        ]
        pd.DataFrame({'README': readme}).to_excel(xw, index=False, sheet_name="README")
    
    print(f"\n{'='*70}")
    print(f"Results saved: {out_xlsx}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
