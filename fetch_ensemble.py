# -*- coding: utf-8 -*-
"""
fetch_ensemble.py
=================
Robust, resumable retrieval of:
  (1) the ERA5 observational reference (1995-2014, Open-Meteo Archive API),
      the calibration baseline for bias correction;
  (2) daily temperature of 7 CMIP6 models (official Open-Meteo Climate API
      model list) for the historical (1995-2014) and future (2041-2050)
      windows.

Note: the Open-Meteo Climate API provides a single (high-emission-
approximate) projection with no SSP scenario split, so each model is fetched
for the historical and future windows only. Model names are the exact
strings accepted by the API.

Rate-limit handling: serial requests, 3 s sleep between calls, exponential
backoff retry on 429/5xx. Existing files are skipped, so the script is
resumable.

Output: raw_cmip6/{model}__{window}.json and era5_obs_1995_2014.json
"""
import json, os, time, urllib.request, urllib.parse

LAT, LON = 23.093145, 113.34728
HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw_cmip6")
os.makedirs(RAW, exist_ok=True)

CLIMATE = "https://climate-api.open-meteo.com/v1/climate"
ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"

# 7 models supported by the Open-Meteo Climate API (exact strings)
MODELS = [
    "CMCC_CM2_VHR4",
    "FGOALS_f3_H",
    "HiRAM_SIT_HR",
    "MRI_AGCM3_2_S",
    "EC_Earth3P_HR",
    "MPI_ESM1_2_XR",
    "NICAM16_8S",
]

WINDOWS = [
    ("hist",    "1995-01-01", "2014-12-31"),
    ("future",  "2041-01-01", "2050-12-31"),
]


def fetch_with_retry(url, timeout=180, max_tries=6):
    backoff = [10, 20, 40, 60, 90, 120]
    for i in range(max_tries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = backoff[min(i, len(backoff) - 1)]
                print(f"    [429] backing off {wait}s (try {i+1}/{max_tries})", flush=True)
                time.sleep(wait)
                continue
            if 500 <= e.code < 600:
                wait = backoff[min(i, len(backoff) - 1)]
                print(f"    [5xx {e.code}] backing off {wait}s", flush=True)
                time.sleep(wait)
                continue
            raise
        except Exception as e:
            wait = backoff[min(i, len(backoff) - 1)]
            print(f"    [{type(e).__name__}] backing off {wait}s: {str(e)[:60]}", flush=True)
            time.sleep(wait)
    raise RuntimeError("exhausted retries")


def fetch_era5_obs():
    path = os.path.join(HERE, "era5_obs_1995_2014.json")
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        print("ERA5 obs already present, skipped")
        return
    print("== Fetching ERA5 observational reference 1995-2014 ==", flush=True)
    params = {
        "latitude": LAT, "longitude": LON,
        "daily": "temperature_2m_mean",
        "start_date": "1995-01-01", "end_date": "2014-12-31",
        "timezone": "Asia/Shanghai",
    }
    url = ARCHIVE + "?" + urllib.parse.urlencode(params)
    d = fetch_with_retry(url)
    daily = d["daily"]
    out = {
        "source": "Open-Meteo Archive API (ERA5 reanalysis)",
        "grid": [LAT, LON], "period": ["1995-01-01", "2014-12-31"],
        "time": daily["time"], "temperature_2m_mean": daily["temperature_2m_mean"],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"  saved {os.path.basename(path)}: {len(daily['time'])} days", flush=True)


def fetch_model(model):
    for wname, s, e in WINDOWS:
        fname = f"{model}__{wname}.json"
        fpath = os.path.join(RAW, fname)
        if os.path.exists(fpath) and os.path.getsize(fpath) > 1000:
            print(f"  [{model}/{wname}] already present, skipped", flush=True)
            continue
        print(f"  [{model}/{wname}] fetching {s}~{e}", flush=True)
        params = {
            "latitude": LAT, "longitude": LON,
            "daily": "temperature_2m_mean",
            "models": model,
            "start_date": s, "end_date": e,
            "timezone": "Asia/Shanghai",
        }
        url = CLIMATE + "?" + urllib.parse.urlencode(params)
        try:
            d = fetch_with_retry(url)
            daily = d["daily"]
            out = {
                "model": model, "window": wname,
                "period": [s, e], "grid": [LAT, LON],
                "time": daily["time"],
                "temperature_2m_mean": daily["temperature_2m_mean"],
            }
            with open(fpath, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False)
            valid = sum(1 for x in daily["temperature_2m_mean"] if x is not None)
            print(f"    saved {fname}: {valid} valid days", flush=True)
        except Exception as ex:
            print(f"    !! [{model}/{wname}] failed: {ex}", flush=True)
        time.sleep(3)


def main():
    fetch_era5_obs()
    time.sleep(3)
    for model in MODELS:
        print(f"== Model {model} ==", flush=True)
        fetch_model(model)
        time.sleep(3)
    got = sorted(os.listdir(RAW))
    print(f"\nDone. raw files present: {len(got)}")
    print("List:", got)


if __name__ == "__main__":
    main()
