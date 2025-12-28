# Vitis vinifera Phenology Analysis Package

A Python package for analyzing phenology data using interval-censored survival analysis.

## Installation

```bash
pip install vinifera-phenology
```

Or install from source:

```bash
git clone <repository-url>
cd vinifera-phenology
pip install -e .
```

## Quick Start

```python
from vinifera_phenology import survival_analysis, weather

# Process observation data into intervals
intervals = survival_analysis.process_phenology_data(
    csv_path="data/status_intensity_observation_data.csv",
    phenophases=["Open flowers", "Ripe fruits"]
)

# Fetch weather data
sites_df = pd.read_excel("sites.xlsx")
weather_data = weather.fetch_weather_for_sites(sites_df)

# Merge weather with intervals
from vinifera_phenology.weather import merge_weather_with_intervals
enriched = merge_weather_with_intervals(
    intervals["Open flowers"],
    weather_data
)
```

## Package Structure

- `survival_analysis`: Functions for converting observations to interval-censored format
- `weather`: Functions for fetching and processing weather data from NASA POWER
- `models`: Model fitting functions (requires lifelines)
- `utils`: Utility functions for data processing

## Documentation

Full documentation and examples are available in the original analysis scripts (00-15) which demonstrate the complete workflow.

## Citation

If you use this package, please cite the associated publication.

## License

MIT License

