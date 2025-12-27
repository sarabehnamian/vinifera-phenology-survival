#!/usr/bin/env python3
"""
Generate all synthetic ancillary data files matching the structure of real USA-NPN data.

This script creates synthetic versions of all ancillary files that are consistent
with the synthetic observation data, allowing full workflow reproduction.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd


# ---------- Configuration ----------

SCRIPT_DIR = Path(__file__).resolve().parent
SYNTHETIC_DATA_DIR = SCRIPT_DIR
SYNTHETIC_OBS_FILE = SYNTHETIC_DATA_DIR / "synthetic_status_intensity_observation_data.csv"

# Set random seed for reproducibility
np.random.seed(42)


# ---------- Helper functions ----------

def load_synthetic_observations() -> pd.DataFrame:
    """Load the synthetic observation data."""
    if not SYNTHETIC_OBS_FILE.exists():
        raise FileNotFoundError(
            f"Synthetic observation data not found: {SYNTHETIC_OBS_FILE}\n"
            "Please run generate_synthetic_phenology_data.py first."
        )
    df = pd.read_csv(SYNTHETIC_OBS_FILE)
    return df


def generate_ancillary_site_data(df_obs: pd.DataFrame) -> pd.DataFrame:
    """Generate synthetic site data."""
    sites = df_obs[["Site_ID", "Site_Name", "Latitude", "Longitude", "State", "Elevation_in_Meters"]].drop_duplicates()
    
    site_data = []
    for _, row in sites.iterrows():
        site_data.append({
            "Site_ID": row["Site_ID"],
            "Site_Type": np.random.choice(["Personal", "Group"]),
            "Site_Name": row["Site_Name"],
            "State": row["State"],
            "Latitude": row["Latitude"],
            "Longitude": row["Longitude"],
            "Lat_Long_Datum": "WGS84",
            "Lat_Long_Source": "Google Maps",
            "Elevation_in_Meters": row["Elevation_in_Meters"],
            "Elevation_Source": "Auto-USGS-WebService",
            "Degree_of_Development_Surrounding_Site": np.random.choice(["Urban", "Suburban", "Rural"]),
            "Landscape_Description": np.random.choice([
                "Landscaped area (such as vegetable or flower garden, lawn, arboretum)",
                "Orchard or tree farm",
                "Other (please describe in the comments section)"
            ]),
            "Proximity_to_Road": "",
            "Proximity_to_Permanent_Water": "",
            "Area_of_Site": "",
            "Type_of_Forest_at_Site": "",
            "Presence_of_Slope": "",
            "Location_Relative_to_Slope": "",
            "Slope_Aspect": "",
            "Presence_of_Domesticated_Cats": "",
            "Presence_of_Domesticated_Dogs": "",
            "Presence_of_Domesticated_Animals": "",
            "Presence_of_Garden": "",
            "Presence_of_Bird_Feeder": "",
            "Presence_of_Nesting_Box": "",
            "Presence_of_Fruit": "",
            "Presence_of_Birdbath": "",
            "Presence_of_Other_Features_Designed_to_Attract_Animals": "",
            "Site_Comments": "",
            "Site_Registration_Date": "2015-01-01",
        })
    
    return pd.DataFrame(site_data)


def generate_ancillary_individual_plant_data(df_obs: pd.DataFrame) -> pd.DataFrame:
    """Generate synthetic individual plant data."""
    individuals = df_obs[["Individual_ID", "Plant_Nickname", "Species"]].drop_duplicates()
    
    individual_data = []
    for _, row in individuals.iterrows():
        individual_data.append({
            "Individual_ID": row["Individual_ID"],
            "Scientific_Name": row["Species"],
            "Plant_Nickname": row["Plant_Nickname"],
            "Patch": "",
            "Patch_Size": "",
            "Shade_Status": np.random.choice(["Full sun", "Mostly sun", "Partial shade", "Mostly shade"]),
            "Wild": np.random.choice([0, 1]),
            "Watered": np.random.choice([0, 1]),
            "Fertilized": np.random.choice([0, 1]),
            "Gender": np.random.choice(["Both", "Male", "Female", ""]),
            "Planting_Date": "",
            "Plant_Comments": "",
            "Plant_Registration_Date": "2015-01-01",
            "Death_Date_Observed": "",
            "Last_Date_Observed_Alive": "",
            "Death_Reason": "",
            "Death_Comments": "",
            "Plant_Image_URL": "",
            "Plant_Image_Upload_Date": "",
        })
    
    return pd.DataFrame(individual_data)


def generate_ancillary_phenophase_data() -> pd.DataFrame:
    """Generate synthetic phenophase data."""
    phenophase_data = [
        {
            "Phenophase_ID": 501,
            "Phenophase_Description": "Open flowers",
            "Phenophase_Definition_IDs": "'501'",
            "Phenophase_Names": "Open flowers",
            "Phenophase_Revision_Comments": "",
        },
        {
            "Phenophase_ID": 390,
            "Phenophase_Description": "Ripe fruits",
            "Phenophase_Definition_IDs": "'390'",
            "Phenophase_Names": "Ripe fruits",
            "Phenophase_Revision_Comments": "",
        },
    ]
    return pd.DataFrame(phenophase_data)


def generate_ancillary_phenophase_definition_data() -> pd.DataFrame:
    """Generate synthetic phenophase definition data."""
    phenophase_def_data = [
        {
            "Phenophase_Definition_ID": 501,
            "Dataset_ID": 12,
            "Phenophase_ID": 501,
            "Phenophase_Name": "Open flowers",
            "Phenophase_Definition": "Flowers are open and reproductive parts are visible.",
            "Phenophase_Definition_Start_Date": "2008-08-25 00:00:00",
            "Phenophase_Definition_End_Date": "2009-03-01 00:00:00",
            "Phenophase_Definition_Comments": "",
        },
        {
            "Phenophase_Definition_ID": 390,
            "Dataset_ID": 12,
            "Phenophase_ID": 390,
            "Phenophase_Name": "Ripe fruits",
            "Phenophase_Definition": "Fruits have reached full ripeness and are ready for harvest.",
            "Phenophase_Definition_Start_Date": "2008-08-25 00:00:00",
            "Phenophase_Definition_End_Date": "2009-03-01 00:00:00",
            "Phenophase_Definition_Comments": "",
        },
    ]
    return pd.DataFrame(phenophase_def_data)


def generate_ancillary_site_visit_data(df_obs: pd.DataFrame) -> pd.DataFrame:
    """Generate synthetic site visit data."""
    site_visits = df_obs[["Site_Visit_ID"]].drop_duplicates()
    
    visit_data = []
    for visit_id in site_visits["Site_Visit_ID"].unique():
        visit_data.append({
            "Site_Visit_ID": int(visit_id),
            "Travel_Time": "",
            "Total_Observation_Time": f"{np.random.randint(10, 60)} minutes",
            "Animal_Search_Time": "",
            "Num_Observers_Searching": "",
            "Animal_Search_Method": "",
            "Snow_on_Ground": 0,
            "Percent_Snow_Cover": "",
            "Snow_in_Tree_Canopy": 0,
            "Site_Visit_Comments": "",
        })
    
    return pd.DataFrame(visit_data)


def generate_ancillary_intensity_data() -> pd.DataFrame:
    """Generate synthetic intensity data (read from real file structure)."""
    # These are standard intensity categories - use a subset
    intensity_data = [
        {
            "Intensity_Category_ID": 39,
            "Intensity_Category_Name": "Breaking leaf buds",
            "Intensity_Question": "How many buds are breaking?",
            "Intensity_Value_Options": "Less than 3; 3 to 10; More than 10",
        },
        {
            "Intensity_Category_ID": 40,
            "Intensity_Category_Name": "Leaves",
            "Intensity_Question": "What proportion of the canopy is full with leaves?",
            "Intensity_Value_Options": "Less than 5%; 5-24%; 25-49%; 50-74%; 75-94%; 95% or more",
        },
        {
            "Intensity_Category_ID": 41,
            "Intensity_Category_Name": "Increasing leaf size",
            "Intensity_Question": "What proportion of full size are most leaves?",
            "Intensity_Value_Options": "25-49%; 50-74%; 75-94%; 95% or more; Less than 25%",
        },
        {
            "Intensity_Category_ID": 42,
            "Intensity_Category_Name": "Colored leaves",
            "Intensity_Question": "What proportion of the canopy is still full with green leaves?",
            "Intensity_Value_Options": "Less than 5%; 5-24%; 25-49%; 50-74%; 75-94%; 95% or more",
        },
        {
            "Intensity_Category_ID": 48,
            "Intensity_Category_Name": "Flowers or flower buds",
            "Intensity_Question": "How many flowers or flower buds are present?",
            "Intensity_Value_Options": "Less than 3; 3 to 10; 11 to 100; 101 to 1,000; More than 1,000",
        },
        {
            "Intensity_Category_ID": 50,
            "Intensity_Category_Name": "Open flowers",
            "Intensity_Question": "How many open flowers are present?",
            "Intensity_Value_Options": "Less than 3; 3 to 10; 11 to 100; 101 to 1,000; More than 1,000",
        },
        {
            "Intensity_Category_ID": 56,
            "Intensity_Category_Name": "Fruits",
            "Intensity_Question": "How many fruits are present?",
            "Intensity_Value_Options": "Less than 3; 3 to 10; 11 to 100; 101 to 1,000; More than 1,000",
        },
        {
            "Intensity_Category_ID": 58,
            "Intensity_Category_Name": "Ripe fruits",
            "Intensity_Question": "How many ripe fruits are present?",
            "Intensity_Value_Options": "Less than 3; 3 to 10; 11 to 100; 101 to 1,000; More than 1,000",
        },
        {
            "Intensity_Category_ID": 59,
            "Intensity_Category_Name": "Recent fruit or seed drop",
            "Intensity_Question": "How many fruits or seeds have dropped recently?",
            "Intensity_Value_Options": "Less than 3; 3 to 10; 11 to 100; 101 to 1,000; More than 1,000",
        },
    ]
    return pd.DataFrame(intensity_data)


def generate_ancillary_person_data(df_obs: pd.DataFrame) -> pd.DataFrame:
    """Generate synthetic person data."""
    person_ids = set()
    person_ids.update(df_obs["ObservedBy_Person_ID"].unique())
    person_ids.update(df_obs["SubmittedBy_Person_ID"].unique())
    person_ids.update(df_obs["UpdatedBy_Person_ID"].unique())
    person_ids.discard(-9999)  # Remove placeholder
    
    person_data = []
    for person_id in sorted(person_ids):
        person_data.append({
            "Person_ID": int(person_id),
            "Read_Online_Training_Materials": "",
            "Trained_in_Person": "",
            "Place_of_Training": "",
            "Ecological_Experience": "",
            "Eco_Experience_Comments": "",
            "Self-Described_Naturalist": "",
            "Naturalist_Skill_Level": "",
            "Participation_as_Part_of_Job": "",
            "Type_of_Job": "",
            "Job_Comments": "",
            "LPL_Certified_Date": "",
        })
    
    return pd.DataFrame(person_data)


def generate_ancillary_dataset_data() -> pd.DataFrame:
    """Generate synthetic dataset data."""
    dataset_data = [
        {
            "Dataset_ID": -9999,
            "Dataset_Name": "Synthetic Dataset",
            "Dataset_Description": "Synthetic phenology data for reproducibility",
            "Contact_Name": "Synthetic Data Generator",
            "Contact_Institution": "USA-NPN",
            "Contact_Email": "synthetic@usanpn.org",
            "Contact_Phone": "",
            "Contact_Address": "",
            "Dataset_Comments": "This is synthetic data generated for workflow reproduction",
            "Dataset_Documentation_URL": "",
        },
    ]
    return pd.DataFrame(dataset_data)


def generate_ancillary_species_protocol_data() -> pd.DataFrame:
    """Generate synthetic species protocol data."""
    species_protocol_data = [
        {
            "Species_ID": 96,
            "Protocol_ID": 233,
            "Species_Protocol_Comments": "",
        },
    ]
    return pd.DataFrame(species_protocol_data)


def generate_ancillary_species_specific_info_data(df_obs: pd.DataFrame) -> pd.DataFrame:
    """Generate synthetic species-specific info data."""
    species_specific_ids = df_obs["Species-Specific_Info_ID"].unique()
    species_specific_ids = [x for x in species_specific_ids if x != -9999]
    
    species_specific_data = []
    for info_id in sorted(species_specific_ids)[:10]:  # Limit to first 10
        species_specific_data.append({
            "Species-Specific_Info_ID": int(info_id),
            "Species_ID": 96,
            "Protocol_ID": 233,
            "Species-Specific_Info_Comments": "",
        })
    
    return pd.DataFrame(species_specific_data)


def generate_ancillary_protocol_data() -> pd.DataFrame:
    """Generate synthetic protocol data."""
    protocol_data = [
        {
            "Protocol_ID": 233,
            "Protocol_Name": "Status and Intensity",
            "Protocol_Description": "Standard USA-NPN Status and Intensity protocol",
            "Protocol_Comments": "",
        },
    ]
    return pd.DataFrame(protocol_data)


def generate_search_parameters() -> pd.DataFrame:
    """Generate synthetic search parameters."""
    search_params = [
        {"Parameter": "Data Type:", "Setting": "Status and Intensity"},
        {"Parameter": "Start Date:", "Setting": "2015-01-01"},
        {"Parameter": "End Date:", "Setting": "2019-12-31"},
        {"Parameter": "States:", "Setting": ""},
        {"Parameter": "Species:", "Setting": "wine grape (Vitis vinifera)"},
        {"Parameter": "Phenophase Categories:", "Setting": "Flowers, Fruits"},
        {"Parameter": "Output Fields:", "Setting": "All"},
        {"Parameter": "Partner Groups:", "Setting": ""},
        {"Parameter": "Integrated Datasets:", "Setting": "Nature's Notebook"},
        {"Parameter": "Stations:", "Setting": ""},
    ]
    return pd.DataFrame(search_params)


# ---------- Main execution ----------

def main():
    """Generate all synthetic ancillary data files."""
    print("Loading synthetic observation data...")
    df_obs = load_synthetic_observations()
    print(f"Loaded {len(df_obs)} observations")
    
    SYNTHETIC_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Generate all ancillary files
    files_to_generate = [
        ("ancillary_site_data.csv", generate_ancillary_site_data),
        ("ancillary_individual_plant_data.csv", generate_ancillary_individual_plant_data),
        ("ancillary_phenophase_data.csv", generate_ancillary_phenophase_data),
        ("ancillary_phenophase_definition_data.csv", generate_ancillary_phenophase_definition_data),
        ("ancillary_site_visit_data.csv", generate_ancillary_site_visit_data),
        ("ancillary_intensity_data.csv", generate_ancillary_intensity_data),
        ("ancillary_person_data.csv", generate_ancillary_person_data),
        ("ancillary_dataset_data.csv", generate_ancillary_dataset_data),
        ("ancillary_species_protocol_data.csv", generate_ancillary_species_protocol_data),
        ("ancillary_species-specific_info_data.csv", generate_ancillary_species_specific_info_data),
        ("ancillary_protocol_data.csv", generate_ancillary_protocol_data),
        ("search_parameters.csv", generate_search_parameters),
    ]
    
    print("\nGenerating ancillary files...")
    for filename, generator_func in files_to_generate:
        print(f"  Generating {filename}...")
        try:
            if "df_obs" in generator_func.__code__.co_varnames:
                df = generator_func(df_obs)
            else:
                df = generator_func()
            
            output_path = SYNTHETIC_DATA_DIR / f"synthetic_{filename}"
            df.to_csv(output_path, index=False)
            print(f"    Saved {len(df)} rows to {output_path.name}")
        except Exception as e:
            print(f"    Error generating {filename}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\nSynthetic ancillary data generation complete!")
    print(f"All files saved to: {SYNTHETIC_DATA_DIR}")


if __name__ == "__main__":
    main()

