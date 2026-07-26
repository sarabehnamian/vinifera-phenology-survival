#!/usr/bin/env python3
"""
Generate synthetic phenology observation data mimicking USA-NPN structure.

This script creates a synthetic dataset that mimics the structure of the real
phenology data, allowing readers to reproduce the workflow without access to
the actual proprietary data.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd


# ---------- Configuration ----------

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR
OUTPUT_CSV = OUTPUT_DIR / "synthetic_status_intensity_observation_data.csv"

# Synthetic data parameters
N_SITES = 5
N_INDIVIDUALS_PER_SITE = 3
YEARS = [2015, 2016, 2017, 2018, 2019]
SPECIES = "Vitis vinifera"

# Phenophase definitions
PHENOPHASES = {
    "Open flowers": {
        "phenophase_id": 501,
        "phenophase_category": "Flowers",
        "typical_start_doy": 120,  # Late April
        "typical_end_doy": 180,   # Late June
        "duration_days": 30,       # Typical flowering duration
    },
    "Ripe fruits": {
        "phenophase_id": 390,
        "phenophase_category": "Fruits",
        "typical_start_doy": 200,  # Mid July
        "typical_end_doy": 280,     # Early October
        "duration_days": 45,        # Typical fruiting duration
    }
}

# Site locations (synthetic but realistic)
SITES = [
    {"site_id": "SYN001", "site_name": "Synthetic Site 1", "latitude": 38.5, "longitude": -122.3, "state": "CA", "elevation": 150},
    {"site_id": "SYN002", "site_name": "Synthetic Site 2", "latitude": 40.2, "longitude": -85.5, "state": "IN", "elevation": 250},
    {"site_id": "SYN003", "site_name": "Synthetic Site 3", "latitude": 39.8, "longitude": -76.8, "state": "PA", "elevation": 180},
    {"site_id": "SYN004", "site_name": "Synthetic Site 4", "latitude": 36.1, "longitude": -79.4, "state": "NC", "elevation": 220},
    {"site_id": "SYN005", "site_name": "Synthetic Site 5", "latitude": 42.3, "longitude": -71.1, "state": "MA", "elevation": 100},
]


# ---------- Helper functions ----------

def generate_observation_dates(year: int, phenophase_name: str, 
                                typical_start: int, typical_end: int,
                                n_visits: int = None) -> list[datetime]:
    """
    Generate observation dates for a phenophase in a given year.
    Creates more frequent visits during the expected phenophase period.
    """
    if n_visits is None:
        # Random number of visits between 5 and 20
        n_visits = np.random.randint(5, 21)
    
    # Start observations earlier than typical start, end later than typical end
    obs_start_doy = max(1, typical_start - 30)
    obs_end_doy = min(365, typical_end + 30)
    
    # Create date range
    year_start = datetime(year, 1, 1)
    start_date = year_start + timedelta(days=obs_start_doy - 1)
    end_date = year_start + timedelta(days=obs_end_doy - 1)
    
    # Generate dates with higher density during expected period
    dates = []
    for _ in range(n_visits):
        # Weight towards the middle of the observation window
        if np.random.random() < 0.6:
            # 60% chance: sample from expected period
            doy = np.random.randint(typical_start, typical_end + 1)
        else:
            # 40% chance: sample from full observation window
            doy = np.random.randint(obs_start_doy, obs_end_doy + 1)
        
        date = year_start + timedelta(days=doy - 1)
        dates.append(date)
    
    # Sort dates
    dates.sort()
    return dates


def generate_phenophase_status(dates: list[datetime], year: int,
                               phenophase_name: str, typical_start: int,
                               typical_end: int, duration: int) -> list[int]:
    """
    Generate phenophase status (0=absent, 1=present) for each date.
    Creates realistic transitions from absent to present.
    """
    # Determine actual event timing (with some year-to-year variation)
    base_event_doy = typical_start + np.random.randint(-10, 11)
    event_date = datetime(year, 1, 1) + timedelta(days=base_event_doy - 1)
    
    # Determine end of phenophase
    end_event_doy = base_event_doy + duration + np.random.randint(-5, 6)
    end_event_date = datetime(year, 1, 1) + timedelta(days=end_event_doy - 1)
    
    statuses = []
    for date in dates:
        if date < event_date:
            status = 0  # Not yet occurred
        elif date > end_event_date:
            # After phenophase ends, could be 0 or 1 depending on phenophase
            # For fruits, might still be present; for flowers, likely absent
            if phenophase_name == "Ripe fruits":
                status = 1 if np.random.random() < 0.3 else 0
            else:
                status = 0
        else:
            status = 1  # Present during phenophase
        
        statuses.append(status)
    
    return statuses


def create_observation_row(obs_id: int, site_info: dict, individual_id: int,
                           year: int, date: datetime, phenophase_name: str,
                           phenophase_info: dict, status: int) -> dict:
    """Create a single observation row matching USA-NPN format."""
    doy = date.timetuple().tm_yday
    
    return {
        "Observation_ID": obs_id,
        "Dataset_ID": -9999,
        "ObservedBy_Person_ID": np.random.randint(1000, 5000),
        "Submission_ID": np.random.randint(30000, 60000),
        "SubmittedBy_Person_ID": np.random.randint(1000, 5000),
        "Submission_Datetime": date.strftime("%Y-%m-%d %H:%M:%S"),
        "UpdatedBy_Person_ID": -9999,
        "Update_Datetime": -9999,
        "Partner_Group": -9999,
        "Site_ID": site_info["site_id"],
        "Site_Name": site_info["site_name"],
        "Latitude": site_info["latitude"],
        "Longitude": site_info["longitude"],
        "Elevation_in_Meters": site_info["elevation"],
        "State": site_info["state"],
        "Species_ID": 96,
        "Genus": "Vitis",
        "Species": "vinifera",
        "Common_Name": "wine grape",
        "Kingdom": "Plantae",
        "Species_Functional_Type": "Deciduous broadleaf",
        "Species_Category": "Crop",
        "Lifecycle_Duration": "Perennial",
        "Growth_Habit": "Shrub/Vine",
        "USDA_PLANTS_Symbol": "VIVI5",
        "ITIS_Number": 28629,
        "Individual_ID": individual_id,
        "Plant_Nickname": f"plant-{individual_id}",
        "Patch": -9999,
        "Protocol_ID": 233,
        "Phenophase_ID": phenophase_info["phenophase_id"],
        "Phenophase_Category": phenophase_info["phenophase_category"],
        "Phenophase_Description": phenophase_name,
        "Phenophase_Name": phenophase_name,
        "Phenophase_Definition_ID": phenophase_info["phenophase_id"],
        "Species-Specific_Info_ID": np.random.randint(5000, 8000),
        "Observation_Date": date.strftime("%Y-%m-%d"),
        "Observation_Time": "12:00:00",
        "Day_of_Year": doy,
        "Phenophase_Status": status,
        "Intensity_Category_ID": np.random.choice([39, 40, 41, 42, 48, 50, 56, 58, 59]) if status == 1 else -9999,
        "Intensity_Value": np.random.choice(["3 to 10", "11 to 100", "101 to 1,000"]) if status == 1 else -9999,
        "Abundance_Value": -9999,
        "Site_Visit_ID": np.random.randint(10000, 20000),
        "Observation_Comments": -9999,
        "Observed_Status_Conflict_Flag": -9999,
        "Status_Conflict_Related_Records": "'-9999'",
    }


# ---------- Main generation function ----------

def generate_synthetic_data() -> pd.DataFrame:
    """Generate synthetic phenology observation data."""
    print("Generating synthetic phenology observation data...")
    print(f"Sites: {N_SITES}, Individuals per site: {N_INDIVIDUALS_PER_SITE}, Years: {YEARS}")
    
    observations = []
    obs_id_counter = 1000000  # Start observation IDs
    
    np.random.seed(42)  # For reproducibility
    
    for site_idx, site_info in enumerate(SITES[:N_SITES]):
        site_id = site_info["site_id"]
        print(f"  Processing site {site_id}...")
        
        for individual_idx in range(N_INDIVIDUALS_PER_SITE):
            individual_id = int(f"{site_idx+1}{individual_idx+1:03d}")  # e.g., 1001, 1002, etc.
            
            for year in YEARS:
                for phenophase_name, phenophase_info in PHENOPHASES.items():
                    # Generate observation dates
                    dates = generate_observation_dates(
                        year=year,
                        phenophase_name=phenophase_name,
                        typical_start=phenophase_info["typical_start_doy"],
                        typical_end=phenophase_info["typical_end_doy"],
                    )
                    
                    if not dates:
                        continue
                    
                    # Generate statuses
                    statuses = generate_phenophase_status(
                        dates=dates,
                        year=year,
                        phenophase_name=phenophase_name,
                        typical_start=phenophase_info["typical_start_doy"],
                        typical_end=phenophase_info["typical_end_doy"],
                        duration=phenophase_info["duration_days"],
                    )
                    
                    if len(statuses) != len(dates):
                        continue
                    
                    # Create observation rows
                    for date, status in zip(dates, statuses):
                        row = create_observation_row(
                            obs_id=obs_id_counter,
                            site_info=site_info,
                            individual_id=individual_id,
                            year=year,
                            date=date,
                            phenophase_name=phenophase_name,
                            phenophase_info=phenophase_info,
                            status=status,
                        )
                        observations.append(row)
                        obs_id_counter += 1
    
    if not observations:
        raise ValueError("No observations generated! Check the generation logic.")
    
    df = pd.DataFrame(observations)
    print(f"Generated {len(df)} observations")
    print(f"  Sites: {df['Site_ID'].nunique()}")
    print(f"  Individuals: {df['Individual_ID'].nunique()}")
    print(f"  Years: {sorted(df['Observation_Date'].str[:4].unique())}")
    print(f"  Phenophases: {df['Phenophase_Name'].unique().tolist()}")
    
    return df


# ---------- Main execution ----------

def main() -> str:
    """Main function to generate and save synthetic data."""
    # Create output directory if needed
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Generate data
    df = generate_synthetic_data()
    
    # Save to CSV
    print(f"\nSaving synthetic data to: {OUTPUT_CSV}")
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(df)} rows to {OUTPUT_CSV}")
    
    # Print summary statistics
    print("\n--- Summary Statistics ---")
    print(f"Total observations: {len(df):,}")
    print(f"Unique sites: {df['Site_ID'].nunique()}")
    print(f"Unique individuals: {df['Individual_ID'].nunique()}")
    print(f"Date range: {df['Observation_Date'].min()} to {df['Observation_Date'].max()}")
    
    print("\nObservations by phenophase:")
    print(df.groupby('Phenophase_Name').size())
    
    print("\nObservations by status:")
    print(df.groupby('Phenophase_Status').size())
    
    print("\nObservations by year:")
    print(df.groupby(df['Observation_Date'].str[:4]).size())
    
    return str(OUTPUT_CSV)


if __name__ == "__main__":
    output_path = main()
    print(f"\nSynthetic data generation complete!")
    print(f"  Output file: {output_path}")

