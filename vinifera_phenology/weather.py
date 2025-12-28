"""
Weather data fetching and processing functions.

Fetches weather data from NASA POWER and processes it for phenology analysis.
"""

from __future__ import annotations

import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .utils import normalize_columns


def check_coords(lat: float, lon: float):
    """Validate latitude and longitude coordinates."""
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError(f"Invalid coords lat={lat}, lon={lon}")


def fetch_power_chunk(
    lat: float,
    lon: float,
    start_dt: date,
    end_dt: date,
    retries: int = 3,
    pause: float = 0.75
) -> pd.DataFrame:
    """
    Fetch one inclusive date chunk from NASA POWER daily endpoint.
    
    Parameters
    ----------
    lat : float
        Latitude
    lon : float
        Longitude
    start_dt : date
        Start date
    end_dt : date
        End date
    retries : int
        Number of retry attempts
    pause : float
        Pause between retries (seconds)
    
    Returns
    -------
    pd.DataFrame
        Weather data with columns: date, tmin_c, tmax_c, prcp_mm
    """
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        "start": start_dt.strftime("%Y%m%d"),
        "end": end_dt.strftime("%Y%m%d"),
        "latitude": lat,
        "longitude": lon,
        "parameters": "T2M_MIN,T2M_MAX,PRECTOTCORR",
        "community": "AG",
        "format": "JSON",
    }
    last_err = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=60)
            r.raise_for_status()
            j = r.json()
            p = j["properties"]["parameter"]
            # Align on union of available dates across variables
            dates = sorted(
                set(p.get("T2M_MAX", {}).keys())
                | set(p.get("T2M_MIN", {}).keys())
                | set(p.get("PRECTOTCORR", {}).keys())
            )
            if not dates:
                return pd.DataFrame(columns=["date", "tmin_c", "tmax_c", "prcp_mm"])
            df = pd.DataFrame({
                "date": pd.to_datetime(dates),
                "tmin_c": [p.get("T2M_MIN", {}).get(d, np.nan) for d in dates],
                "tmax_c": [p.get("T2M_MAX", {}).get(d, np.nan) for d in dates],
                "prcp_mm": [p.get("PRECTOTCORR", {}).get(d, np.nan) for d in dates],
            })
            return df
        except Exception as e:
            last_err = e
            time.sleep(pause * (attempt + 1))
    raise RuntimeError(f"NASA POWER request failed after {retries} attempts: {last_err}")


def year_chunks(dmin: date, dmax: date):
    """Yield inclusive (start,end) by calendar year between dmin..dmax."""
    y = dmin.year
    while y <= dmax.year:
        s = date(y, 1, 1) if y > dmin.year else dmin
        e = date(y, 12, 31) if y < dmax.year else dmax
        yield s, e
        y += 1


def fetch_weather_for_site(
    site_id: str,
    latitude: float,
    longitude: float,
    start_date: date,
    end_date: date
) -> pd.DataFrame:
    """
    Fetch weather data for a single site.
    
    Parameters
    ----------
    site_id : str
        Site identifier
    latitude : float
        Site latitude
    longitude : float
        Site longitude
    start_date : date
        Start date for weather data
    end_date : date
        End date for weather data
    
    Returns
    -------
    pd.DataFrame
        Weather data with site_id column
    """
    check_coords(latitude, longitude)
    
    if end_date < start_date:
        return pd.DataFrame()

    site_frames = []
    for s, e in year_chunks(start_date, end_date):
        df = fetch_power_chunk(latitude, longitude, s, e)
        if not df.empty:
            site_frames.append(df)
        time.sleep(0.2)  # polite pause

    if not site_frames:
        return pd.DataFrame()

    wx = (
        pd.concat(site_frames, ignore_index=True)
        .drop_duplicates("date")
        .sort_values("date")
    )

    wx = wx[
        (wx["date"] >= pd.Timestamp(start_date)) & (wx["date"] <= pd.Timestamp(end_date))
    ].copy()

    if wx.empty:
        return pd.DataFrame()

    wx["site_id"] = site_id
    wx["gdd_base10"] = ((wx["tmax_c"] + wx["tmin_c"]) / 2.0 - 10.0).clip(lower=0)
    
    return wx[["site_id", "date", "tmin_c", "tmax_c", "prcp_mm", "gdd_base10"]]


def fetch_weather_for_sites(
    sites_df: pd.DataFrame,
    output_path: str | Path | None = None
) -> pd.DataFrame:
    """
    Fetch weather data for multiple sites.
    
    Parameters
    ----------
    sites_df : pd.DataFrame
        DataFrame with columns: site_id, latitude, longitude, weather_start_date, weather_end_date
    output_path : str | Path | None
        Optional path to save Excel output
    
    Returns
    -------
    pd.DataFrame
        Combined weather data for all sites
    """
    sites_df = normalize_columns(sites_df)
    required = {"site_id", "latitude", "longitude", "weather_start_date", "weather_end_date"}
    missing = required - set(sites_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    sites_df = sites_df.dropna(subset=list(required)).copy()
    sites_df["weather_start_date"] = pd.to_datetime(sites_df["weather_start_date"], errors="coerce")
    sites_df["weather_end_date"] = pd.to_datetime(sites_df["weather_end_date"], errors="coerce")
    sites_df = sites_df.dropna(subset=["weather_start_date", "weather_end_date"])

    rows = []
    for _, row in sites_df.iterrows():
        sid = row["site_id"]
        lat, lon = float(row["latitude"]), float(row["longitude"])
        dmin = row["weather_start_date"].date()
        dmax = row["weather_end_date"].date()

        wx = fetch_weather_for_site(sid, lat, lon, dmin, dmax)
        if not wx.empty:
            rows.append(wx)

    if not rows:
        raise ValueError("No weather data fetched (check coordinates/dates).")

    weather = (
        pd.concat(rows, ignore_index=True)
        .drop_duplicates(["site_id", "date"])
        .sort_values(["site_id", "date"])
    )

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(output_path, engine="openpyxl") as xw:
            weather.to_excel(xw, index=False, sheet_name="daily_weather")

    return weather


def merge_weather_with_intervals(
    intervals_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    cutoff_col: str = "cutoff_doy"
) -> pd.DataFrame:
    """
    Merge survival intervals with weather data, computing cumulative metrics up to cutoff.
    
    Parameters
    ----------
    intervals_df : pd.DataFrame
        Interval-censored data with site_id, year, and cutoff_doy columns
    weather_df : pd.DataFrame
        Daily weather data with site_id, date columns
    cutoff_col : str
        Name of column containing cutoff DOY
    
    Returns
    -------
    pd.DataFrame
        Intervals enriched with weather covariates
    """
    intervals_df = normalize_columns(intervals_df.copy())
    weather_df = normalize_columns(weather_df.copy())
    
    weather_df["site_id"] = weather_df["site_id"].astype(str)
    weather_df["date"] = pd.to_datetime(weather_df["date"])
    weather_df["year"] = weather_df["date"].dt.year.astype(int)
    weather_df["doy"] = weather_df["date"].dt.dayofyear.astype(int)

    # Precompute cumulative metrics
    wx_sorted = weather_df.sort_values(["site_id", "year", "doy"]).copy()
    wx_sorted["day_count"] = 1
    wx_sorted["frost_day"] = (wx_sorted["tmin_c"] <= 0).astype(int)
    wx_sorted["heat_day"] = (wx_sorted["tmax_c"] >= 35).astype(int)

    group_cols = ["site_id", "year"]
    for newc, src in {
        "gdd_cum": "gdd_base10",
        "prcp_cum": "prcp_mm",
        "tmin_sum_cum": "tmin_c",
        "tmax_sum_cum": "tmax_c",
        "days_cum": "day_count",
        "frost_days_cum": "frost_day",
        "heat_days_cum": "heat_day",
    }.items():
        wx_sorted[newc] = wx_sorted.groupby(group_cols, sort=False)[src].cumsum()

    def take_up_to_cutoff(sub: pd.DataFrame, cutoff_doy: int):
        """Return cumulative metrics up to (and including) cutoff_doy."""
        sub2 = sub[sub["doy"] <= int(cutoff_doy)]
        if sub2.empty:
            return None
        last = sub2.iloc[-1]
        days = float(last["days_cum"])
        return {
            "gdd_sum_to_cutoff": float(last["gdd_cum"]),
            "prcp_sum_to_cutoff": float(last["prcp_cum"]),
            "tmin_mean_to_cutoff": float(last["tmin_sum_cum"] / days),
            "tmax_mean_to_cutoff": float(last["tmax_sum_cum"] / days),
            "frost_days_to_cutoff": int(last["frost_days_cum"]),
            "heat_days_to_cutoff": int(last["heat_days_cum"]),
            "wx_days_covered": int(days),
            "wx_max_doy_used": int(last["doy"]),
        }

    intervals_df["site_id"] = intervals_df["site_id"].astype(str)
    
    # Compute cutoff DOY if not present
    if cutoff_col not in intervals_df.columns:
        if "r" in intervals_df.columns and "last_obs_doy" in intervals_df.columns:
            intervals_df[cutoff_col] = np.where(
                (intervals_df.get("event", 0) == 1) & intervals_df["r"].notna(),
                intervals_df["r"],
                intervals_df["last_obs_doy"]
            )
        else:
            raise ValueError(f"Cannot determine cutoff DOY. Need '{cutoff_col}' or 'r'/'last_obs_doy' columns.")

    intervals_df[cutoff_col] = intervals_df[cutoff_col].clip(lower=1, upper=366).astype(int)

    # Initialize weather columns
    covar_cols = [
        "gdd_sum_to_cutoff", "prcp_sum_to_cutoff",
        "tmin_mean_to_cutoff", "tmax_mean_to_cutoff",
        "frost_days_to_cutoff", "heat_days_to_cutoff",
        "wx_days_covered", "wx_max_doy_used"
    ]
    for c in covar_cols:
        intervals_df[c] = np.nan

    wx_groups = {k: v for k, v in wx_sorted.groupby(group_cols, sort=False)}

    for idx, row in intervals_df.iterrows():
        key = (row["site_id"], int(row["year"]))
        sub = wx_groups.get(key)
        if sub is None or sub.empty:
            continue
        vals = take_up_to_cutoff(sub, int(row[cutoff_col]))
        if vals is None:
            continue
        for k, v in vals.items():
            intervals_df.at[idx, k] = v

    # Cast ints cleanly
    for c in ["frost_days_to_cutoff", "heat_days_to_cutoff", "wx_days_covered", "wx_max_doy_used"]:
        if c in intervals_df.columns:
            intervals_df[c] = pd.to_numeric(intervals_df[c], errors="coerce").fillna(0).astype(int)

    return intervals_df

