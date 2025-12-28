# RE1_S2_model_priority.py
"""
Reviewer 1, Comment S2: "Unclear priority among multiple models"
================================================================
"Different parts of the Results section describe:
- Multivariate AFT models with to-cutoff covariates;
- Univariate fixed-window models;
- Bivariate fixed-window models.
It remains unclear which model(s) the authors consider their primary inference 
model(s). There should be a clear statement: 'Our main model is X for reason Y; 
models A and B are only diagnostic / sensitivity analyses.'"

This script provides:
1. Comprehensive comparison of all model types
2. AIC/BIC ranking to identify primary model
3. Clear justification for model selection
4. Classification of models as primary vs diagnostic/sensitivity

Outputs: revision/RE1_S2_results/
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
OUT_DIR = PROJECT / "revision" / "RE1_S2_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Model result files
MULTIVARIATE = PROJECT / "05_fit_interval_models" / "interval_model_results.xlsx"
UNIVARIATE = PROJECT / "10_refit_simple_models" / "interval_model_results_simple.xlsx"
BIVARIATE = PROJECT / "12_bivariate_fixed_window" / "interval_model_results_bivariate.xlsx"

# Outputs
OUT_XLSX = OUT_DIR / "RE1_S2_model_priority_comparison.xlsx"
OUT_PLOT = OUT_DIR / "RE1_S2_model_comparison.png"

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


def load_multivariate_results():
    """Load multivariate AFT models with to-cutoff covariates."""
    if not MULTIVARIATE.exists():
        return pd.DataFrame()
    
    try:
        # Try to load coefficients sheet
        coef = norm(pd.read_excel(MULTIVARIATE, sheet_name="coefficients"))
        quick = norm(pd.read_excel(MULTIVARIATE, sheet_name="model_quickview"))
        
        results = []
        for sheet in ["open_flowers", "ripe_fruits"]:
            sheet_coef = coef[coef["sheet"].str.lower() == sheet.lower()].copy()
            sheet_quick = quick[quick["sheet"].str.lower() == sheet.lower()].copy()
            
            for model in ["WeibullAFT", "LogLogisticAFT"]:
                model_coef = sheet_coef[sheet_coef["model"].str.contains(model, case=False, na=False)]
                if not model_coef.empty:
                    # Extract AIC/BIC from quickview
                    aic_col = f"{model}_aic"
                    bic_col = f"{model}_bic"
                    ll_col = f"{model}_log_likelihood"
                    
                    aic = float(sheet_quick[aic_col].iloc[0]) if aic_col in sheet_quick.columns else np.nan
                    bic = float(sheet_quick[bic_col].iloc[0]) if bic_col in sheet_quick.columns else np.nan
                    ll = float(sheet_quick[ll_col].iloc[0]) if ll_col in sheet_quick.columns else np.nan
                    
                    # Count parameters (intercept + shape + covariates)
                    n_params = len(model_coef[model_coef["covariate"].notna()]) + 2  # +2 for intercept and shape
                    
                    results.append({
                        "sheet": sheet,
                        "model_type": "Multivariate (to-cutoff)",
                        "model_family": model,
                        "n_covariates": 4,  # gdd, prcp, frost, heat
                        "covariate_type": "Endogenous (to-event)",
                        "n_parameters": n_params,
                        "log_likelihood": ll,
                        "AIC": aic,
                        "BIC": bic,
                        "status": "endogenous_bias"
                    })
        
        return pd.DataFrame(results)
    except Exception as e:
        print(f"  Warning: Could not load multivariate results: {e}")
        return pd.DataFrame()


def fit_univariate_models():
    """Fit univariate fixed-window models and extract AIC/BIC."""
    from lifelines import LogLogisticAFTFitter, WeibullAFTFitter
    from scipy.stats import zscore
    
    # Load survival data
    surv_clean = PROJECT / "07_validate_sensitivity" / "survival_with_weather_clean.xlsx"
    if not surv_clean.exists():
        return pd.DataFrame()
    
    # Load fixed window features
    fixed_features = PROJECT / "10_refit_simple_models" / "fixed_window_features.xlsx"
    if not fixed_features.exists():
        return pd.DataFrame()
    
    results = []
    
    for sheet_name in ["open_flowers", "ripe_fruits"]:
        try:
            surv_df = norm(pd.read_excel(surv_clean, sheet_name=sheet_name))
            window_prefix = "flowers" if "flower" in sheet_name else "fruits"
            
            # Load fixed window features
            window_df = norm(pd.read_excel(fixed_features, sheet_name=f"{window_prefix}_window"))
            
            # Merge
            df = surv_df.merge(window_df, on=["site_id", "year"], how="inner")
            
            # Get GDD column (check both prefixed and non-prefixed)
            gdd_col = f"{window_prefix}_gdd_pre" if f"{window_prefix}_gdd_pre" in df.columns else "gdd_pre"
            if gdd_col not in df.columns:
                continue
            
            # Prepare data
            df["r_filled"] = df["r"].where(df["r"].notna(), np.inf)
            df = df.dropna(subset=["l", gdd_col])
            df[f"{gdd_col}_z"] = zscore(df[gdd_col])
            
            design = df[["l", "r_filled", f"{gdd_col}_z"]].dropna()
            
            if len(design) < 10:
                continue
            
            # Fit both models
            for model_name, Fitter in [("LogLogisticAFT", LogLogisticAFTFitter),
                                       ("WeibullAFT", WeibullAFTFitter)]:
                try:
                    m = Fitter()
                    m.fit_interval_censoring(design, lower_bound_col="l", upper_bound_col="r_filled")
                    
                    n_params = len(m.params_)
                    n_obs = len(design)
                    bic = n_params * np.log(n_obs) - 2 * m.log_likelihood_
                    
                    results.append({
                        "sheet": sheet_name,
                        "model_type": "Univariate (fixed-window)",
                        "model_family": model_name,
                        "n_covariates": 1,
                        "covariate_type": "Exogenous (fixed-window)",
                        "n_parameters": n_params,
                        "log_likelihood": m.log_likelihood_,
                        "AIC": m.AIC_,
                        "BIC": bic,
                        "status": "primary_candidate"
                    })
                except Exception as e:
                    print(f"    {model_name} fit failed for {sheet_name}: {e}")
        
        except Exception as e:
            print(f"  Error processing {sheet_name}: {e}")
    
    return pd.DataFrame(results)


def fit_bivariate_models():
    """Fit bivariate fixed-window models and extract AIC/BIC."""
    from lifelines import LogLogisticAFTFitter, WeibullAFTFitter
    from scipy.stats import zscore
    
    # Load survival data
    surv_clean = PROJECT / "07_validate_sensitivity" / "survival_with_weather_clean.xlsx"
    if not surv_clean.exists():
        return pd.DataFrame()
    
    # Load fixed window features
    fixed_features = PROJECT / "10_refit_simple_models" / "fixed_window_features.xlsx"
    if not fixed_features.exists():
        return pd.DataFrame()
    
    results = []
    
    for sheet_name in ["open_flowers", "ripe_fruits"]:
        try:
            surv_df = norm(pd.read_excel(surv_clean, sheet_name=sheet_name))
            window_prefix = "flowers" if "flower" in sheet_name else "fruits"
            
            # Load fixed window features
            window_df = norm(pd.read_excel(fixed_features, sheet_name=f"{window_prefix}_window"))
            
            # Merge
            df = surv_df.merge(window_df, on=["site_id", "year"], how="inner")
            
            # Get GDD and precipitation columns (check both prefixed and non-prefixed)
            gdd_col = f"{window_prefix}_gdd_pre" if f"{window_prefix}_gdd_pre" in df.columns else "gdd_pre"
            prcp_col = f"{window_prefix}_prcp_pre" if f"{window_prefix}_prcp_pre" in df.columns else "prcp_pre"
            
            if gdd_col not in df.columns or prcp_col not in df.columns:
                continue
            
            # Prepare data
            df["r_filled"] = df["r"].where(df["r"].notna(), np.inf)
            df = df.dropna(subset=["l", gdd_col, prcp_col])
            df[f"{gdd_col}_z"] = zscore(df[gdd_col])
            df[f"{prcp_col}_z"] = zscore(df[prcp_col])
            
            design = df[["l", "r_filled", f"{gdd_col}_z", f"{prcp_col}_z"]].dropna()
            
            if len(design) < 10:
                continue
            
            # Fit both models
            for model_name, Fitter in [("LogLogisticAFT", LogLogisticAFTFitter),
                                       ("WeibullAFT", WeibullAFTFitter)]:
                try:
                    m = Fitter()
                    m.fit_interval_censoring(design, lower_bound_col="l", upper_bound_col="r_filled")
                    
                    n_params = len(m.params_)
                    n_obs = len(design)
                    bic = n_params * np.log(n_obs) - 2 * m.log_likelihood_
                    
                    results.append({
                        "sheet": sheet_name,
                        "model_type": "Bivariate (fixed-window)",
                        "model_family": model_name,
                        "n_covariates": 2,
                        "covariate_type": "Exogenous (fixed-window)",
                        "n_parameters": n_params,
                        "log_likelihood": m.log_likelihood_,
                        "AIC": m.AIC_,
                        "BIC": bic,
                        "status": "primary_candidate"
                    })
                except Exception as e:
                    print(f"    {model_name} fit failed for {sheet_name}: {e}")
        
        except Exception as e:
            print(f"  Error processing {sheet_name}: {e}")
    
    return pd.DataFrame(results)


def classify_models(df):
    """Classify models as primary vs diagnostic/sensitivity."""
    df = df.copy()
    
    # For each sheet, rank by BIC (lower is better)
    df["rank_by_bic"] = df.groupby("sheet")["BIC"].rank(method="min", na_option="keep")
    
    # Primary model: Best BIC among exogenous (fixed-window) models
    exogenous_mask = df["covariate_type"] == "Exogenous (fixed-window)"
    
    for sheet in df["sheet"].unique():
        sheet_mask = (df["sheet"] == sheet) & exogenous_mask
        sheet_df = df[sheet_mask].copy()
        
        if not sheet_df.empty and sheet_df["BIC"].notna().any():
            best_bic = sheet_df["BIC"].min()
            best_idx = sheet_df[sheet_df["BIC"] == best_bic].index[0]
            df.loc[best_idx, "classification"] = "PRIMARY"
            df.loc[sheet_mask & (df.index != best_idx), "classification"] = "Alternative (exogenous)"
        
        # Endogenous models are diagnostic (show bias)
        endogenous_mask = (df["sheet"] == sheet) & (df["covariate_type"] == "Endogenous (to-event)")
        df.loc[endogenous_mask, "classification"] = "Diagnostic (endogeneity demonstration)"
    
    # Fill remaining as "Not classified"
    df["classification"] = df["classification"].fillna("Not classified")
    
    return df


def create_comparison_plot(df, outpath):
    """Create model comparison visualization."""
    import matplotlib.colors as mcolors
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for idx, sheet in enumerate(["open_flowers", "ripe_fruits"]):
        sheet_df = df[df["sheet"] == sheet].copy()
        if sheet_df.empty:
            continue
        
        base_color = COLOR_FLOWERS if "flower" in sheet.lower() else COLOR_FRUITS
        
        # Plot 1: AIC comparison
        ax1 = axes[idx, 0]
        valid_aic = sheet_df[sheet_df["AIC"].notna()].copy()
        if not valid_aic.empty:
            valid_aic = valid_aic.sort_values("AIC")
            
            # Assign colors based on model type
            colors_aic = []
            for _, row in valid_aic.iterrows():
                if "Multivariate" in row["model_type"]:
                    colors_aic.append('lightgray')  # Diagnostic models in gray
                elif "Univariate" in row["model_type"]:
                    colors_aic.append(base_color)  # Base color for univariate
                elif "Bivariate" in row["model_type"]:
                    # Darker shade for bivariate
                    rgb = mcolors.hex2color(base_color)
                    colors_aic.append(mcolors.rgb2hex([max(0, c * 0.7) for c in rgb]))
                else:
                    colors_aic.append(base_color)
            
            bars = ax1.barh(range(len(valid_aic)), valid_aic["AIC"], color=colors_aic, 
                           alpha=0.8, edgecolor='black', linewidth=1)
            
            # Highlight primary model
            primary_mask = valid_aic["classification"] == "PRIMARY"
            if primary_mask.any():
                primary_idx = valid_aic[primary_mask].index[0]
                bar_idx = list(valid_aic.index).index(primary_idx)
                bars[bar_idx].set_edgecolor('red')
                bars[bar_idx].set_linewidth(3)
            
            ax1.set_yticks(range(len(valid_aic)))
            ax1.set_yticklabels([f"{row['model_type']}\n{row['model_family']}" for _, row in valid_aic.iterrows()], fontsize=12)
            ax1.set_xlabel("AIC", fontsize=12)
            ax1.set_title(f"{sheet.replace('_', ' ').title()} - AIC Comparison", fontsize=13)
            ax1.grid(True, alpha=0.3, axis='x')
            ax1.invert_yaxis()
        
        # Plot 2: BIC comparison
        ax2 = axes[idx, 1]
        valid_bic = sheet_df[sheet_df["BIC"].notna()].copy()
        if not valid_bic.empty:
            valid_bic = valid_bic.sort_values("BIC")
            
            # Assign colors based on model type
            colors_bic = []
            for _, row in valid_bic.iterrows():
                if "Multivariate" in row["model_type"]:
                    colors_bic.append('lightgray')  # Diagnostic models in gray
                elif "Univariate" in row["model_type"]:
                    colors_bic.append(base_color)  # Base color for univariate
                elif "Bivariate" in row["model_type"]:
                    # Darker shade for bivariate
                    rgb = mcolors.hex2color(base_color)
                    colors_bic.append(mcolors.rgb2hex([max(0, c * 0.7) for c in rgb]))
                else:
                    colors_bic.append(base_color)
            
            bars = ax2.barh(range(len(valid_bic)), valid_bic["BIC"], color=colors_bic, 
                           alpha=0.8, edgecolor='black', linewidth=1)
            
            # Highlight primary model
            primary_mask = valid_bic["classification"] == "PRIMARY"
            if primary_mask.any():
                primary_idx = valid_bic[primary_mask].index[0]
                bar_idx = list(valid_bic.index).index(primary_idx)
                bars[bar_idx].set_edgecolor('red')
                bars[bar_idx].set_linewidth(3)
            
            ax2.set_yticks(range(len(valid_bic)))
            ax2.set_yticklabels([f"{row['model_type']}\n{row['model_family']}" for _, row in valid_bic.iterrows()], fontsize=12)
            ax2.set_xlabel("BIC", fontsize=12)
            ax2.set_title(f"{sheet.replace('_', ' ').title()} - BIC Comparison", fontsize=13)
            ax2.grid(True, alpha=0.3, axis='x')
            ax2.invert_yaxis()
    
    plt.suptitle("Model Comparison: Identifying Primary Inference Model", fontsize=14, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(outpath, dpi=300, bbox_inches="tight", format='png')
    plt.close()
    print(f"  Saved: {outpath.name}")


def main():
    print("="*70)
    print("RE1_S2: Model Priority and Classification")
    print("="*70)
    print("\nAddresses: 'Unclear priority among multiple models'")
    
    print("\nLoading model results...")
    
    # Load all model types
    print("  Loading multivariate (to-cutoff) models...")
    multi_df = load_multivariate_results()
    print(f"    Found {len(multi_df)} models")
    
    print("  Fitting univariate (fixed-window) models...")
    uni_df = fit_univariate_models()
    print(f"    Found {len(uni_df)} models")
    
    print("  Fitting bivariate (fixed-window) models...")
    bi_df = fit_bivariate_models()
    print(f"    Found {len(bi_df)} models")
    
    # Combine
    all_models = pd.concat([multi_df, uni_df, bi_df], ignore_index=True)
    
    if all_models.empty:
        print("  ERROR: No model results found!")
        return
    
    print(f"  Loaded {len(all_models)} model configurations")
    
    # Classify models
    all_models = classify_models(all_models)
    
    # Print summary
    print("\n" + "="*70)
    print("MODEL CLASSIFICATION SUMMARY")
    print("="*70)
    
    for sheet in ["open_flowers", "ripe_fruits"]:
        sheet_df = all_models[all_models["sheet"] == sheet].copy()
        if sheet_df.empty:
            continue
        
        print(f"\n{sheet.replace('_', ' ').title()}:")
        print("-" * 70)
        
        # Primary model
        primary = sheet_df[sheet_df["classification"] == "PRIMARY"]
        if not primary.empty:
            p = primary.iloc[0]
            print(f"  PRIMARY MODEL:")
            print(f"    {p['model_type']} - {p['model_family']}")
            print(f"    AIC: {p['AIC']:.2f}, BIC: {p['BIC']:.2f}")
            print(f"    Justification: Lowest BIC among exogenous (fixed-window) models")
            print(f"    Covariates: {p['n_covariates']} ({p['covariate_type']})")
        
        # Diagnostic models
        diagnostic = sheet_df[sheet_df["classification"].str.contains("Diagnostic", na=False)]
        if not diagnostic.empty:
            print(f"\n  DIAGNOSTIC MODELS (endogeneity demonstration):")
            for _, d in diagnostic.iterrows():
                print(f"    {d['model_type']} - {d['model_family']}")
                print(f"      Purpose: Demonstrate bias in to-event covariates")
        
        # Alternative models
        alt = sheet_df[sheet_df["classification"].str.contains("Alternative", na=False)]
        if not alt.empty:
            print(f"\n  ALTERNATIVE MODELS (exogenous, not primary):")
            for _, a in alt.iterrows():
                print(f"    {a['model_type']} - {a['model_family']}")
                print(f"      AIC: {a['AIC']:.2f}, BIC: {a['BIC']:.2f}")
    
    # Create plot
    print("\nCreating comparison plot...")
    create_comparison_plot(all_models, OUT_PLOT)
    
    # Save to Excel
    print(f"\nSaving results to: {OUT_XLSX}")
    
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as xw:
        # Full comparison
        all_models.to_excel(xw, index=False, sheet_name="all_models")
        
        # Summary by classification
        summary_rows = []
        for sheet in ["open_flowers", "ripe_fruits"]:
            sheet_df = all_models[all_models["sheet"] == sheet].copy()
            for classification in sheet_df["classification"].unique():
                class_df = sheet_df[sheet_df["classification"] == classification]
                summary_rows.append({
                    "sheet": sheet,
                    "classification": classification,
                    "n_models": len(class_df),
                    "best_AIC": class_df["AIC"].min() if class_df["AIC"].notna().any() else np.nan,
                    "best_BIC": class_df["BIC"].min() if class_df["BIC"].notna().any() else np.nan,
                    "model_types": ", ".join(class_df["model_type"].unique())
                })
        
        pd.DataFrame(summary_rows).to_excel(xw, index=False, sheet_name="summary")
        
        # Primary models only
        primary_models = all_models[all_models["classification"] == "PRIMARY"].copy()
        if not primary_models.empty:
            primary_models.to_excel(xw, index=False, sheet_name="primary_models")
        
        # README
        readme_text = [
            "MODEL PRIORITY AND CLASSIFICATION",
            "Addresses: 'Unclear priority among multiple models' (Reviewer 1, S2)",
            "",
            "MODEL TYPES COMPARED:",
            "1. Multivariate (to-cutoff): AFT with 4 endogenous covariates (GDD, precip, frost, heat to event)",
            "2. Univariate (fixed-window): AFT with 1 exogenous covariate (pre-season GDD)",
            "3. Bivariate (fixed-window): AFT with 2 exogenous covariates (pre-season GDD + precip)",
            "",
            "CLASSIFICATION:",
            "- PRIMARY: Best BIC among exogenous (fixed-window) models",
            "- DIAGNOSTIC: Endogenous models (demonstrate endogeneity bias)",
            "- ALTERNATIVE: Other exogenous models (not primary)",
            "",
            "JUSTIFICATION:",
            "Primary model selected based on:",
            "1. Exogenous covariates (avoid endogeneity bias)",
            "2. Lowest BIC (best model fit with penalty for complexity)",
            "3. Biological interpretability",
            "",
            "Multivariate to-cutoff models are diagnostic only - they demonstrate",
            "the endogeneity bias problem but are not used for inference."
        ]
        pd.DataFrame({"README": readme_text}).to_excel(xw, index=False, sheet_name="README")
    
    print(f"\n{'='*70}")
    print("RE1_S2 COMPLETE")
    print(f"{'='*70}")
    print(f"\nOutputs:")
    print(f"  Excel: {OUT_XLSX}")
    print(f"  Plot: {OUT_PLOT.name}")
    print(f"\nKey message for reviewer:")
    print(f"  → Primary model clearly identified based on BIC and covariate type")
    print(f"  → Diagnostic models (endogenous) clearly separated from inference models")
    print(f"  → Clear justification provided for model selection")


if __name__ == "__main__":
    main()

