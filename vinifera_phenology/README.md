# vinifera_phenology Package

A Python package for analyzing Vitis vinifera phenology data using interval-censored survival analysis.

## Installation

Install the package in development mode:

```bash
pip install -e .
```

Or install from the repository root:

```bash
cd /path/to/vinifera-phenology-survival
pip install -e .
```

## Quick Start

```python
from vinifera_phenology import survival_analysis, weather

# Process observation data into intervals
intervals = survival_analysis.process_phenology_data(
    csv_path="data/status_intensity_observation_data.csv",
    phenophases=["Open flowers", "Ripe fruits"],
    output_path="output/intervals.xlsx"
)

# Fetch weather data for sites
import pandas as pd
sites_df = pd.read_excel("sites.xlsx", sheet_name="Weather_Coordinates")
weather_data = weather.fetch_weather_for_sites(
    sites_df,
    output_path="output/weather.xlsx"
)

# Merge weather with intervals
from vinifera_phenology.weather import merge_weather_with_intervals

# Add cutoff DOY to intervals
intervals_df = intervals["Open flowers"].copy()
intervals_df["cutoff_doy"] = intervals_df.apply(
    lambda row: row["R"] if row["event"] == 1 and pd.notna(row["R"]) else row["last_obs_doy"],
    axis=1
)

enriched = merge_weather_with_intervals(intervals_df, weather_data)
```

## Package Modules

### `survival_analysis`

Functions for converting USA-NPN observation data into interval-censored survival format.

**Main Functions:**
- `process_phenology_data()` - Main function to process observation CSV into intervals
- `prepare_observation_data()` - Load and prepare observation data
- `build_intervals()` - Build interval-censored table for a phenophase

**Example:**
```python
from vinifera_phenology import survival_analysis

intervals = survival_analysis.process_phenology_data(
    csv_path="data/observations.csv",
    phenophases=["Open flowers", "Ripe fruits"]
)
```

### `weather`

Functions for fetching and processing weather data from NASA POWER.

**Main Functions:**
- `fetch_weather_for_sites()` - Fetch weather data for multiple sites
- `fetch_weather_for_site()` - Fetch weather for a single site
- `merge_weather_with_intervals()` - Merge weather data with survival intervals

**Example:**
```python
from vinifera_phenology import weather

weather_data = weather.fetch_weather_for_sites(sites_df)
```

### `utils`

Utility functions for data processing.

**Functions:**
- `normalize_columns()` - Normalize DataFrame column names
- `parse_status()` - Parse phenophase status values
- `zscore()` - Standardize a series to z-scores

### `models`

Model fitting functions (placeholder - requires lifelines library for full functionality).

## Command-Line Interface

The package includes a command-line interface:

```bash
vinifera-survival data/observations.csv output/intervals.xlsx
```

## Dependencies

Core dependencies:
- pandas >= 1.3.0
- numpy >= 1.20.0
- openpyxl >= 3.0.0
- requests >= 2.25.0

Optional dependencies (for model fitting):
- lifelines >= 0.27.0

Optional dependencies (for plotting):
- matplotlib >= 3.3.0
- seaborn >= 0.11.0

## Documentation

For detailed usage examples, see `example_usage.py` in this directory.

For the complete analysis workflow using sequential scripts, see the main repository README.md.

## Relationship to Sequential Scripts

This package provides a programmatic interface to the functionality in the sequential scripts (00-15). The scripts can still be run independently for step-by-step analysis, while the package allows for more flexible, programmatic use.

## License

MIT License - see LICENSE file in repository root.

