# 02_fetch_nasa_power_weather.py
# Real daily weather from NASA POWER for each site over its observed dates,
# using the Weather_Coordinates sheet you already created.
# Join key: site_id + date
# Output: 02_fetch_nasa_power_weather/site_daily_weather.xlsx
# Requires: requests, pandas, numpy, openpyxl

import sys, time, requests
import pandas as pd, numpy as np
from pathlib import Path
from datetime import date

# --- console encoding (Windows) ---
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# --- paths ---
SUMMARY_XLSX = Path("01_analyze_observation_dates/date_time_summary.xlsx")
OUT_DIR = Path("02_fetch_nasa_power_weather")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / "site_daily_weather.xlsx"

def norm(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (df.columns.str.strip().str.lower()
                  .str.replace(" ", "_").str.replace("-", "_"))
    return df

def check_coords(lat, lon):
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError(f"Invalid coords lat={lat}, lon={lon}")

def fetch_power_chunk(lat, lon, start_dt: date, end_dt: date, retries=3, pause=0.75) -> pd.DataFrame:
    """
    Fetch one inclusive date chunk from NASA POWER daily endpoint.
    Returns DataFrame with columns: date, tmin_c, tmax_c, prcp_mm
    """
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        "start": start_dt.strftime("%Y%m%d"),
        "end":   end_dt.strftime("%Y%m%d"),
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
            dates = sorted(set(p.get("T2M_MAX", {}).keys())
                           | set(p.get("T2M_MIN", {}).keys())
                           | set(p.get("PRECTOTCORR", {}).keys()))
            if not dates:
                return pd.DataFrame(columns=["date","tmin_c","tmax_c","prcp_mm"])
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

# --- load Weather_Coordinates from your summary workbook ---
print(f"Reading: {SUMMARY_XLSX}")
try:
    wx_coords_raw = pd.read_excel(SUMMARY_XLSX, sheet_name="Weather_Coordinates")
except ValueError as e:
    raise SystemExit("Sheet 'Weather_Coordinates' not found. Re-run 01_analyze_observation_dates.py") from e

wx_coords = norm(wx_coords_raw)
required = {"site_id","latitude","longitude","weather_start_date","weather_end_date"}
missing = required - set(wx_coords.columns)
if missing:
    raise SystemExit(f"Weather_Coordinates missing columns: {missing}")

# Clean dates and coords
wx_coords = wx_coords.dropna(subset=["site_id","latitude","longitude","weather_start_date","weather_end_date"]).copy()
wx_coords["weather_start_date"] = pd.to_datetime(wx_coords["weather_start_date"], errors="coerce")
wx_coords["weather_end_date"]   = pd.to_datetime(wx_coords["weather_end_date"],   errors="coerce")
wx_coords = wx_coords.dropna(subset=["weather_start_date","weather_end_date"])

rows = []
print(f"Sites to fetch: {wx_coords['site_id'].nunique()}")

for _, row in wx_coords.iterrows():
    sid = row["site_id"]
    lat, lon = float(row["latitude"]), float(row["longitude"])
    check_coords(lat, lon)

    dmin = row["weather_start_date"].date()
    dmax = row["weather_end_date"].date()
    if dmax < dmin:
        continue

    site_frames = []
    for s, e in year_chunks(dmin, dmax):
        df = fetch_power_chunk(lat, lon, s, e)
        if not df.empty:
            site_frames.append(df)
        time.sleep(0.2)  # polite pause

    if not site_frames:
        continue

    wx = (pd.concat(site_frames, ignore_index=True)
            .drop_duplicates("date")
            .sort_values("date"))

    wx = wx[(wx["date"] >= pd.Timestamp(dmin)) & (wx["date"] <= pd.Timestamp(dmax))].copy()
    if wx.empty:
        continue

    wx["site_id"] = sid
    wx["gdd_base10"] = ((wx["tmax_c"] + wx["tmin_c"]) / 2.0 - 10.0).clip(lower=0)
    rows.append(wx[["site_id","date","tmin_c","tmax_c","prcp_mm","gdd_base10"]])

if not rows:
    raise SystemExit("No POWER data fetched (check coordinates/dates in Weather_Coordinates).")

weather = (pd.concat(rows, ignore_index=True)
             .drop_duplicates(["site_id","date"])
             .sort_values(["site_id","date"]))

# --- write output ---
with pd.ExcelWriter(OUT, engine="openpyxl") as xw:
    weather.to_excel(xw, index=False, sheet_name="daily_weather")
    summary = (weather.groupby("site_id")
                      .agg(start=("date","min"),
                           end=("date","max"),
                           days=("date","count"),
                           tmin_mean=("tmin_c","mean"),
                           tmax_mean=("tmax_c","mean"),
                           prcp_total=("prcp_mm","sum"),
                           gdd_sum=("gdd_base10","sum"))
                      .round(2)
                      .reset_index())
    summary.to_excel(xw, index=False, sheet_name="summary_by_site")
    pd.DataFrame({"README":[
        "Source: NASA POWER daily (parameters: T2M_MIN,T2M_MAX,PRECTOTCORR; community=AG).",
        "Units: °C (temps) and mm/day (precip). Join on site_id + date.",
        "GDD base 10 °C = max(((Tmax+Tmin)/2 - 10), 0).",
        "Date span per site comes from Weather_Coordinates (your observed phenology span)."
    ]}).to_excel(xw, index=False, sheet_name="README")

print(f"Saved real weather to: {OUT}")
