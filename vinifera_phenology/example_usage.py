"""
Example usage of the vinifera_phenology package.

This demonstrates how to use the package programmatically instead of running scripts.
"""

from pathlib import Path
from vinifera_phenology import survival_analysis, weather

# Example 1: Process observation data
print("Example 1: Processing observation data...")
intervals = survival_analysis.process_phenology_data(
    csv_path="data/status_intensity_observation_data.csv",
    phenophases=["Open flowers", "Ripe fruits"],
    output_path="output/intervals.xlsx"
)

print(f"Created intervals for: {list(intervals.keys())}")

# Example 2: Fetch weather data
print("\nExample 2: Fetching weather data...")
import pandas as pd

# Load site information
sites_df = pd.read_excel("01_analyze_observation_dates/date_time_summary.xlsx", 
                         sheet_name="Weather_Coordinates")

weather_data = weather.fetch_weather_for_sites(
    sites_df,
    output_path="output/weather.xlsx"
)

print(f"Fetched weather data for {weather_data['site_id'].nunique()} sites")

# Example 3: Merge weather with intervals
print("\nExample 3: Merging weather with intervals...")
from vinifera_phenology.weather import merge_weather_with_intervals

# Add cutoff DOY to intervals
intervals_df = intervals["Open flowers"].copy()
intervals_df["cutoff_doy"] = intervals_df.apply(
    lambda row: row["R"] if row["event"] == 1 and pd.notna(row["R"]) else row["last_obs_doy"],
    axis=1
)

enriched = merge_weather_with_intervals(
    intervals_df,
    weather_data
)

print(f"Enriched {len(enriched)} intervals with weather data")
print(f"Columns: {list(enriched.columns)}")

