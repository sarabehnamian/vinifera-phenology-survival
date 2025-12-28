# Vinifera Phenology Survival Analysis

A comprehensive analysis pipeline for studying grapevine (Vitis vinifera) phenology and survival patterns using observational data and weather variables.

## Overview

This project analyzes the relationship between environmental factors and grapevine phenological events and survival. The analysis combines observational phenology data with NASA POWER weather data to understand how climate variables influence vine development and survival patterns.

## Project Structure

The analysis is organized in two ways:

### 1. Python Package (New)

The `vinifera_phenology/` directory contains a **formal Python package** that can be installed and used as a library. See `vinifera_phenology/README.md` for package-specific documentation.

**Installation:**
```bash
pip install -e .
```

**Usage:**
```python
from vinifera_phenology import survival_analysis, weather

# Process observation data into intervals
intervals = survival_analysis.process_phenology_data(
    csv_path="data/status_intensity_observation_data.csv",
    phenophases=["Open flowers", "Ripe fruits"]
)
```

### 2. Sequential Scripts (Original Workflow)

The original sequential pipeline of Python scripts (00-15) is also available for step-by-step execution:

#### Data Preparation

* `00_npn_survival_analysis.py` - Initial survival analysis setup
* `01_analyze_observation_dates.py` - Analysis of observation timing patterns
* `02_fetch_nasa_power_weather.py` - Retrieval of NASA POWER weather data
* `03_merge_survival_with_weather.py` - Integration of survival and weather datasets

#### Statistical Analysis

* `04_basic_analysis_and_plots.py` - Exploratory data analysis and visualization
* `05_fit_interval_models.py` - Interval censoring survival models
* `06_format_model_outputs.py` - Model output formatting and processing
* `07_validate_sensitivity.py` - Sensitivity analysis and model validation
* `08_effect_summaries.py` - Summary of treatment effects
* `09_diagnostics_and_refit.py` - Model diagnostics and refitting
* `10_refit_simple_models.py` - Simplified model refitting
* `11_simple_outputs.py` - Basic output generation

#### Advanced Analysis & Visualization

* `12_bivariate_fixed_window.py` - Bivariate analysis with fixed time windows
* `13_bivariate_ci_plots.py` - Confidence interval plotting for bivariate models
* `14_publication_pack.py` - Publication-ready output generation
* `15_table_and_figs.py` - Final tables and figures for publication

**Usage of Sequential Scripts**: Run the scripts in numerical order (00-15) to execute the complete analysis pipeline. Each script is designed to build upon the outputs of previous scripts, so maintaining the sequence is important for proper execution.

## Data Sources

* **Phenology Data**: We monitored 7 individual European grapevine (Vitis vinifera L.) plants across 5 sites, yielding 41,274 phenophase status observations between 30 March 2012 and 16 May 2025. These observations were collected by 16 observers using the USA–NPN Status & Intensity protocol. Although 41,274 daily records were available, interval-censoring requires collapsing them to one record per phenophase transition.
* **Weather Data**: NASA POWER meteorological data including temperature, precipitation, and other climate variables
* **Survival Data**: Information on vine survival and mortality events

### Data Availability

**Note**: The phenological observation data used in this analysis is not publicly available due to data sharing restrictions. Researchers interested in accessing the data should contact the authors directly to discuss potential collaboration or data sharing agreements.

**Synthetic Data**: Synthetic data mimicking the structure of the real data is available in `revision/synthetic_data/` for reproducibility purposes. See `revision/synthetic_data/` for details.

## Key Features

* **Python Package**: Formal library structure for programmatic use (see `vinifera_phenology/`)
* **Survival Analysis**: Implementation of interval-censored survival models
* **Weather Integration**: Incorporation of comprehensive meteorological variables
* **Bivariate Modeling**: Analysis of relationships between multiple environmental factors
* **Sensitivity Analysis**: Robust validation of model assumptions and parameters
* **Publication Output**: Generation of publication-ready figures and tables

## Requirements

The analysis requires Python with the following key packages:

* pandas
* numpy
* matplotlib
* seaborn
* scipy
* survival analysis libraries (lifelines, scikit-survival)
* NASA POWER API integration tools

Install all requirements:

```bash
pip install -r requirements.txt
```

Or install the package with dependencies:

```bash
pip install -e .
```

## Usage

### Option 1: Use the Python Package (Recommended)

```python
from vinifera_phenology import survival_analysis, weather

# Process data
intervals = survival_analysis.process_phenology_data("data/observations.csv")

# Fetch weather
weather_data = weather.fetch_weather_for_sites(sites_df)
```

### Option 2: Run Sequential Scripts

Run the scripts in numerical order (00-15) to execute the complete analysis pipeline:

```bash
python 00_npn_survival_analysis.py
python 01_analyze_observation_dates.py
# ... continue through all scripts
python 15_table_and_figs.py
```

Each script is designed to build upon the outputs of previous scripts, so maintaining the sequence is important for proper execution.

## Output

The analysis generates:

* Statistical models of phenology-survival relationships
* Visualizations of temporal patterns and trends
* Confidence intervals and effect size estimates
* Publication-ready tables and figures
* Sensitivity analysis results

## Research Applications

This analysis framework can be applied to:

* Climate change impact assessment on viticulture
* Optimization of vineyard management practices
* Understanding phenological responses to environmental variation
* Development of predictive models for grape growing regions

## Contributing

Contributions to improve the analysis pipeline are welcome. Please ensure that any modifications maintain the sequential structure and compatibility with existing data formats.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this analysis pipeline in your research, please cite: [Add appropriate citation information when available]

## Contact

For questions or collaboration opportunities, please contact:

**Sara Behnamian**  
Email: sara.behnamian@sund.ku.dk
