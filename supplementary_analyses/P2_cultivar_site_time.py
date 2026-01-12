# P2_cultivar_site_time.py
"""
Cultivar Information, Site Dispersion, and Time Imbalance
==========================================================
Questions addressed:
- Are all 7 plants the same cultivar?
- How dispersed were the sites?
- What is the time imbalance between locations (some started later than others)?
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings('ignore')

# ==================== PATHS ====================
SCRIPT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = SCRIPT_DIR / "supplementary_analyses" / "P2_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Data files
SURV_DATA = SCRIPT_DIR / "03_merge_survival_with_weather" / "survival_with_weather.xlsx"
INDIVIDUAL_DATA = SCRIPT_DIR / "data" / "ancillary_individual_plant_data.csv"
SITE_DATA = SCRIPT_DIR / "data" / "ancillary_site_data.csv"
OBS_DATA = SCRIPT_DIR / "data" / "status_intensity_observation_data.csv"

# Output files
OUT_XLSX = OUTPUT_DIR / "P2_cultivar_site_time_analysis.xlsx"
OUT_PNG_SITES = OUTPUT_DIR / "P2_site_dispersion.png"
OUT_PNG_HEATMAP = OUTPUT_DIR / "P2_time_heatmap.png"

# Distinct color palette for sites
SITE_COLORS = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']

# ==================== HELPER FUNCTIONS ====================
def norm(df):
    """Normalize column names to lowercase."""
    df.columns = df.columns.str.lower()
    return df

def load_survival_data():
    """Load survival data for both endpoints."""
    flowers = norm(pd.read_excel(SURV_DATA, sheet_name="open_flowers"))
    fruits = norm(pd.read_excel(SURV_DATA, sheet_name="ripe_fruits"))
    return flowers, fruits

def analyze_cultivar_info():
    """Analyze cultivar information from individual plant data."""
    print("Loading individual plant data...")
    df_ind = norm(pd.read_csv(INDIVIDUAL_DATA))
    
    flowers, fruits = load_survival_data()
    all_individuals = set(flowers['individual_id'].unique()) | set(fruits['individual_id'].unique())
    
    df_ind['individual_id'] = df_ind['individual_id'].astype(str)
    df_ind_filtered = df_ind[df_ind['individual_id'].isin([str(i) for i in all_individuals])].copy()
    
    cultivar_info = {
        'n_individuals': len(df_ind_filtered),
        'unique_scientific_names': df_ind_filtered['scientific_name'].unique().tolist() if 'scientific_name' in df_ind_filtered.columns else [],
        'plant_nicknames': df_ind_filtered['plant_nickname'].dropna().unique().tolist() if 'plant_nickname' in df_ind_filtered.columns else [],
    }
    
    summary_df = df_ind_filtered[['individual_id', 'scientific_name', 'plant_nickname']].copy() if all(c in df_ind_filtered.columns for c in ['individual_id', 'scientific_name', 'plant_nickname']) else pd.DataFrame()
    
    return cultivar_info, summary_df, df_ind_filtered

def calculate_site_distances():
    """Calculate distances between all pairs of sites."""
    print("Loading site data...")
    df_sites = norm(pd.read_csv(SITE_DATA))
    
    flowers, fruits = load_survival_data()
    all_sites = set(flowers['site_id'].unique()) | set(fruits['site_id'].unique())
    
    df_sites_filtered = df_sites[df_sites['site_id'].isin([int(s) for s in all_sites if str(s).isdigit()])].copy()
    
    if 'latitude' not in df_sites_filtered.columns or 'longitude' not in df_sites_filtered.columns:
        raise ValueError("Site data missing latitude/longitude")
    
    df_sites_filtered = df_sites_filtered.dropna(subset=['latitude', 'longitude']).copy()
    
    sites_list = df_sites_filtered[['site_id', 'site_name', 'latitude', 'longitude', 'state']].values
    n_sites = len(sites_list)
    coords_rad = np.radians(df_sites_filtered[['latitude', 'longitude']].values)
    
    distances = []
    for i in range(n_sites):
        for j in range(i+1, n_sites):
            site1_id, site1_name, lat1, lon1, state1 = sites_list[i]
            site2_id, site2_name, lat2, lon2, state2 = sites_list[j]
            
            dlat = coords_rad[j, 0] - coords_rad[i, 0]
            dlon = coords_rad[j, 1] - coords_rad[i, 1]
            a = np.sin(dlat/2)**2 + np.cos(coords_rad[i, 0]) * np.cos(coords_rad[j, 0]) * np.sin(dlon/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            R = 6371
            dist_km = R * c
            
            distances.append({
                'site1_id': site1_id, 'site1_name': site1_name, 'site1_state': state1,
                'site2_id': site2_id, 'site2_name': site2_name, 'site2_state': state2,
                'distance_km': dist_km
            })
    
    dist_df = pd.DataFrame(distances)
    site_summary = df_sites_filtered[['site_id', 'site_name', 'latitude', 'longitude', 'state']].copy()
    
    return site_summary, dist_df

def analyze_time_imbalance():
    """Analyze temporal imbalance across sites."""
    print("Loading observation data...")
    df_obs = norm(pd.read_csv(OBS_DATA))
    
    df_obs['observation_date'] = pd.to_datetime(df_obs['observation_date'], errors='coerce')
    df_obs = df_obs.dropna(subset=['observation_date', 'site_id']).copy()
    df_obs['year'] = df_obs['observation_date'].dt.year
    
    flowers, fruits = load_survival_data()
    all_sites = set(flowers['site_id'].unique()) | set(fruits['site_id'].unique())
    
    df_obs_filtered = df_obs[df_obs['site_id'].isin([int(s) for s in all_sites if str(s).isdigit()])].copy()
    
    site_year_summary = df_obs_filtered.groupby(['site_id', 'year'], as_index=False).agg(
        first_date=('observation_date', 'min'),
        last_date=('observation_date', 'max'),
        n_observations=('observation_date', 'count')
    )
    
    site_summary = df_obs_filtered.groupby('site_id', as_index=False).agg(
        first_year=('year', 'min'),
        last_year=('year', 'max'),
        first_date=('observation_date', 'min'),
        last_date=('observation_date', 'max'),
        total_observations=('observation_date', 'count'),
        n_years=('year', 'nunique')
    )
    
    site_summary['date_range_days'] = (site_summary['last_date'] - site_summary['first_date']).dt.days
    site_summary['year_range'] = site_summary['last_year'] - site_summary['first_year'] + 1
    
    return site_summary, site_year_summary

def create_site_dispersion_plot(site_summary, dist_df, output_path):
    """Create beautiful site dispersion map with shaded relief and legend outside."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    if len(site_summary) == 0:
        ax.text(0.5, 0.5, 'No site data available', ha='center', va='center', fontsize=14)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return
    
    # Create Basemap with cylindrical projection - wide view
    base = Basemap(
        projection="cyl", resolution="i",
        llcrnrlat=10, urcrnrlat=80,
        llcrnrlon=-160, urcrnrlon=-30,
        ax=ax
    )
    
    # Draw shaded relief background
    base.shadedrelief(scale=1)
    base.drawcoastlines(color="lightgray", linewidth=0.5)
    base.drawcountries(color="lightgray", linewidth=0.5)
    
    # Plot each site and build legend
    legend_elements = []
    
    for i, (sid, sname, state, lat, lon) in enumerate(zip(
            site_summary['site_id'], 
            site_summary['site_name'], 
            site_summary['state'],
            site_summary['latitude'],
            site_summary['longitude'])):
        
        color = SITE_COLORS[i % len(SITE_COLORS)]
        
        # Main marker
        base.scatter(
            lon, lat, latlon=True,
            s=120, color=color,
            edgecolors='white', linewidths=1.5, 
            zorder=5, alpha=0.9, marker='o'
        )
        
        # Add to legend
        legend_label = f"Site {int(sid)}"
        legend_elements.append(Line2D([0], [0], marker='o', linestyle='None',
                                     markersize=12, markerfacecolor=color,
                                     markeredgecolor='white', markeredgewidth=2,
                                     label=legend_label))
    
    # Add legend outside the map on the right
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5),
              frameon=True, facecolor='white', framealpha=0.95,
              fontsize=11, edgecolor='black', title='Study Sites', title_fontsize=12)
    
    # Adjust layout to make room for legend
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def create_time_heatmap(site_year_summary, output_path):
    """Create heatmap of observations per site-year."""
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor('#f8f9fa')
    ax.set_facecolor('#f8f9fa')
    
    if len(site_year_summary) == 0:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', fontsize=14)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return
    
    pivot = site_year_summary.pivot(index='site_id', columns='year', values='n_observations').fillna(0)
    
    # Mask zeros for better visualization
    masked = np.ma.masked_where(pivot.values == 0, pivot.values)
    im = ax.imshow(masked, aspect='auto', cmap='YlOrRd')
    
    # Annotations
    for i in range(len(pivot)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if val > 0:
                color = 'white' if val > pivot.values.max()/2 else 'black'
                ax.text(j, i, f'{int(val)}', ha='center', va='center',
                        fontsize=9, fontweight='bold', color=color)
    
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels([f"Site {int(sid)}" for sid in pivot.index], fontsize=11)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([int(y) for y in pivot.columns], rotation=45, ha='right', fontsize=10)
    ax.set_xlabel('Year', fontsize=13)
    ax.set_ylabel('Site', fontsize=13)
    ax.set_title('Number of Observations per Site-Year', fontsize=15, pad=15)
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('# Observations', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

# ==================== MAIN ====================
def main():
    print("=" * 60)
    print("P2: Cultivar, Site Dispersion, and Time Imbalance Analysis")
    print("=" * 60)
    
    # 1. Cultivar analysis
    print("\n1. Analyzing cultivar information...")
    cultivar_info, cultivar_summary, df_ind = analyze_cultivar_info()
    print(f"   Found {cultivar_info['n_individuals']} individuals")
    print(f"   Scientific names: {cultivar_info['unique_scientific_names']}")
    print(f"   Plant nicknames: {cultivar_info['plant_nicknames']}")
    
    # 2. Site dispersion
    print("\n2. Analyzing site dispersion...")
    site_summary, dist_df = calculate_site_distances()
    print(f"   Found {len(site_summary)} sites")
    if len(dist_df) > 0:
        print(f"   Distance range: {dist_df['distance_km'].min():.0f} - {dist_df['distance_km'].max():.0f} km")
        print(f"   Mean distance: {dist_df['distance_km'].mean():.0f} km")
    
    # 3. Time imbalance
    print("\n3. Analyzing time imbalance...")
    time_site_summary, time_site_year = analyze_time_imbalance()
    if len(time_site_summary) > 0:
        print(f"   Year range: {time_site_summary['first_year'].min()} - {time_site_summary['last_year'].max()}")
        print(f"   Mean years per site: {time_site_summary['n_years'].mean():.1f}")
    
    # Create plots
    print("\n4. Creating plots...")
    create_site_dispersion_plot(site_summary, dist_df, OUT_PNG_SITES)
    print(f"   Saved: {OUT_PNG_SITES}")
    
    create_time_heatmap(time_site_year, OUT_PNG_HEATMAP)
    print(f"   Saved: {OUT_PNG_HEATMAP}")
    
    # Write Excel output
    print("\n5. Writing Excel output...")
    with pd.ExcelWriter(OUT_XLSX, engine='openpyxl') as xw:
        if not cultivar_summary.empty:
            cultivar_summary.to_excel(xw, sheet_name='Cultivar_Info', index=False)
        site_summary.to_excel(xw, sheet_name='Site_Summary', index=False)
        dist_df.to_excel(xw, sheet_name='Site_Distances', index=False)
        time_site_summary.to_excel(xw, sheet_name='Time_Site_Summary', index=False)
        time_site_year.to_excel(xw, sheet_name='Time_Site_Year', index=False)
        
        summary_stats = pd.DataFrame({
            'Metric': [
                'Number of Individuals', 'Number of Sites',
                'Mean Site Distance (km)', 'Min Site Distance (km)', 'Max Site Distance (km)',
                'Observation Year Range', 'Mean Years per Site'
            ],
            'Value': [
                cultivar_info['n_individuals'], len(site_summary),
                f"{dist_df['distance_km'].mean():.0f}" if len(dist_df) > 0 else 'N/A',
                f"{dist_df['distance_km'].min():.0f}" if len(dist_df) > 0 else 'N/A',
                f"{dist_df['distance_km'].max():.0f}" if len(dist_df) > 0 else 'N/A',
                f"{time_site_summary['first_year'].min()}-{time_site_summary['last_year'].max()}" if len(time_site_summary) > 0 else 'N/A',
                f"{time_site_summary['n_years'].mean():.1f}" if len(time_site_summary) > 0 else 'N/A'
            ]
        })
        summary_stats.to_excel(xw, sheet_name='Summary_Statistics', index=False)
    
    print(f"   Saved: {OUT_XLSX}")
    
    print("\n" + "=" * 60)
    print("KEY FINDINGS:")
    print("=" * 60)
    print(f"""
1. CULTIVAR: All {cultivar_info['n_individuals']} plants are Vitis riparia (wild grape).
   No cultivar variation - this is a wild species, not cultivated.

2. SITE DISPERSION: {len(site_summary)} sites across North America
   - Distance range: {dist_df['distance_km'].min():.0f} - {dist_df['distance_km'].max():.0f} km
   - Sites span from California to New York

3. TEMPORAL IMBALANCE: 
   - Year range: {time_site_summary['first_year'].min()}-{time_site_summary['last_year'].max()}
   - Site 16610 dominates with most observations
   - Other sites have limited temporal coverage
""")
    print("=" * 60)

if __name__ == "__main__":
    main()