# -*- coding: utf-8 -*-
"""
fetch_era5_study_period.py
==========================
Retrieve the ERA5 daily reanalysis series for the study period
(2015-01-01 to 2022-12-31) for the Haizhu grid cell from the Open-Meteo
Archive API and store it as data_era5/era5_haizhu.json. This file is the
observational input required by run_analysis.py.
"""
import json, os, time, urllib.request, urllib.parse

LAT, LON = 23.093145, 113.34728
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data_era5")
os.makedirs(OUT, exist_ok=True)
ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"


def main():
    path = os.path.join(OUT, "era5_haizhu.json")
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        print("era5_haizhu.json already present, skipped")
        return
    params = {
        "latitude": LAT, "longitude": LON,
        "daily": "temperature_2m_mean,precipitation_sum,relative_humidity_2m_mean",
        "start_date": "2015-01-01", "end_date": "2022-12-31",
        "timezone": "Asia/Shanghai",
    }
    url = ARCHIVE + "?" + urllib.parse.urlencode(params)
    d = None
    for attempt in range(6):
        try:
            with urllib.request.urlopen(url, timeout=180) as r:
                d = json.load(r)
            break
        except Exception as e:
            wait = 20 * (attempt + 1)
            print(f"[{type(e).__name__}] backing off {wait}s: {str(e)[:60]}", flush=True)
            time.sleep(wait)
    if d is None:
        raise RuntimeError("exhausted retries")
    daily = d["daily"]
    out = {
        "source": "Open-Meteo Archive API (ERA5 reanalysis)",
        "grid": [LAT, LON], "period": ["2015-01-01", "2022-12-31"],
        "time": daily["time"],
        "temperature_2m_mean": daily["temperature_2m_mean"],
        "precipitation_sum": daily["precipitation_sum"],
        "relative_humidity_2m_mean": daily["relative_humidity_2m_mean"],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"saved era5_haizhu.json: {len(daily['time'])} days")


if __name__ == "__main__":
    main()
