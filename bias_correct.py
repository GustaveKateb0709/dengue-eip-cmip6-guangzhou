# -*- coding: utf-8 -*-
"""
bias_correct.py
===============
Bias-correct daily near-surface temperature from 7 CMIP6 models and build a
multi-model ensemble.

Method: monthly quantile delta mapping (QDM; Cannon et al. 2015),
non-parametric implementation.
  - Calibration baseline: ERA5 observational daily temperature 1995-2014
    (era5_obs_1995_2014.json)
  - For each model m and calendar month mo:
        sh = sorted(model historical daily T of month mo)   # model historical ECDF
        so = sorted(ERA5 observational daily T of month mo) # observed ECDF
        for any future value x, its quantile in the model history is
        q = #{sh <= x}/N (clipped to [0, 1])
        mapping: x* = so(q) + (x - sh(q))
        where so(q) and sh(q) are the observed/model-historical values at
        quantile q.
  - This shifts the model's future distribution onto the observed
    distribution while preserving the model's own relative change
    (x - sh(q)), so systematic bias is removed without inflating or
    discarding the warming signal, and without the divergent linear
    extrapolation of naive quantile mapping outside the calibrated range.
  - Monthly climatology: mean of each series by calendar month
  - Monthly delta (bias-corrected):
        delta_BC[m, mo] = mean of BC future month mo - mean of ERA5 month mo
  - Monthly delta (raw):
        delta_raw[m, mo] = mean of model future month mo - mean of model
        historical month mo
  - Ensemble: cross-model mean and standard deviation of delta_BC /
    delta_raw (propagating structural uncertainty)

Note: the Open-Meteo Climate API provides a single (high-emission-
approximate) projection with no SSP scenario split, so each model has one
future window only. Output: cmip6_ensemble_bc.json
"""
import json
import os
import glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw_cmip6")


def _monthly_groups(daily_t, time_str):
    arr = np.asarray(daily_t, dtype=float)
    months = np.array([int(t[5:7]) for t in time_str])
    out = {}
    for m in range(1, 13):
        vals = arr[months == m]
        vals = vals[np.isfinite(vals)]
        out[m] = vals
    return out


def _monthly_clim(daily_t, time_str):
    g = _monthly_groups(daily_t, time_str)
    return np.array([g[m].mean() if len(g[m]) else np.nan for m in range(1, 13)])


def qdm_correct(hist_vals, obs_vals):
    """Monthly quantile delta mapping (QDM) corrector: takes model daily
    values and returns a callable that returns corrected daily values.

    The quantile q is located in the model's historical ECDF, mapped onto the
    same quantile of the observed distribution, and the model's own relative
    deviation (x - sh(q)) is added back, preserving relative change while
    removing systematic bias.
    """
    sh = np.sort(np.asarray(hist_vals, dtype=float))
    so = np.sort(np.asarray(obs_vals, dtype=float))
    sh = sh[np.isfinite(sh)]
    so = so[np.isfinite(so)]
    if len(sh) < 30 or len(so) < 30:
        # too few samples: fall back to additive mean-bias correction
        def addc(x):
            x = np.asarray(x, dtype=float)
            return x + (np.nanmean(so) - np.nanmean(sh))
        return addc
    n = len(sh)
    ranks = (np.arange(n) + 0.5) / n

    def correct(x):
        x = np.asarray(x, dtype=float)
        q = np.clip(np.searchsorted(sh, x, side="right") / n, 0.0, 1.0)
        obs_q = np.interp(q, ranks, so)
        mod_q = np.interp(q, ranks, sh)
        return obs_q + (x - mod_q)
    return correct


def load_obs():
    here = os.path.dirname(os.path.abspath(__file__))
    cand = [os.path.join(here, "era5_obs_1995_2014.json"),
            os.path.join(here, "..", "03_Supplementary", "era5_obs_1995_2014.json")]
    p = next((c for c in cand if os.path.exists(c)), cand[0])
    d = json.load(open(p, encoding="utf-8"))
    return d["temperature_2m_mean"], d["time"]


def load_model(model):
    files = glob.glob(os.path.join(RAW, f"{model}__*.json"))
    data = {}
    for fp in files:
        d = json.load(open(fp, encoding="utf-8"))
        data[d["window"]] = d
    return data


def correct_model(model, obs_t, obs_time):
    data = load_model(model)
    if not all(k in data for k in ("hist", "future")):
        return None
    obs_g = _monthly_groups(obs_t, obs_time)
    hist_d = data["hist"]
    fut_d = data["future"]
    bc_fut = np.full(12, np.nan)
    raw_fut = _monthly_clim(fut_d["temperature_2m_mean"], fut_d["time"])
    raw_hist = _monthly_clim(hist_d["temperature_2m_mean"], hist_d["time"])
    for m in range(1, 13):
        hg = _monthly_groups(hist_d["temperature_2m_mean"], hist_d["time"])[m]
        og = obs_g[m]
        if len(hg) < 30 or len(og) < 30:
            continue
        correct = qdm_correct(hg, og)
        fg = _monthly_groups(fut_d["temperature_2m_mean"], fut_d["time"])[m]
        if len(fg) >= 10:
            bc_fut[m - 1] = correct(fg).mean()
    raw_anom = raw_fut - raw_hist
    return {"bc_future": bc_fut, "raw_anomaly": raw_anom}


def main():
    obs_t, obs_time = load_obs()
    obs_clim = _monthly_clim(obs_t, obs_time)
    print(f"ERA5 observational monthly climatology, annual mean: {np.nanmean(obs_clim):.2f} C")

    models = sorted({os.path.basename(f).split("__")[0]
                     for f in glob.glob(os.path.join(RAW, "*__*.json"))})
    print(f"Models to process: {models}")

    per_model = {}
    for model in models:
        r = correct_model(model, obs_t, obs_time)
        if r is None:
            print(f"  {model}: missing window, skipped")
            continue
        per_model[model] = r
        bc = r["bc_future"]
        raw = r["raw_anomaly"]
        print(f"  {model}: BC annual delta {np.nanmean(bc - obs_clim):+.2f} C, "
              f"raw annual delta {np.nanmean(raw):+.2f} C")

    done = list(per_model.keys())
    # delta_BC is defined as corrected future mean - ERA5 observed mean
    bc_stack = np.array([per_model[m]["bc_future"] - obs_clim for m in done])
    raw_stack = np.array([per_model[m]["raw_anomaly"] for m in done])

    out = {
        "meta": {
            "method": "monthly quantile delta mapping (QDM, non-parametric; Cannon et al. 2015)",
            "obs": "ERA5 1995-2014 (Open-Meteo Archive)",
            "models": done,
            "n_models": len(done),
            "hist_window": "1995-2014",
            "fut_window": "2041-2050",
            "note": "Open-Meteo Climate API provides a single (high-emission-approx) "
                    "projection; no ssp scenario split available",
            "grid": [23.093145, 113.34728],
        },
        "obs_monthly_clim": {m + 1: round(float(obs_clim[m]), 3) for m in range(12)},
        "ensemble_bc_monthly_anomaly": {m + 1: round(float(np.nanmean(bc_stack[:, m])), 3) for m in range(12)},
        "ensemble_bc_spread_std": {m + 1: round(float(np.nanstd(bc_stack[:, m])), 3) for m in range(12)},
        "ensemble_raw_monthly_anomaly": {m + 1: round(float(np.nanmean(raw_stack[:, m])), 3) for m in range(12)},
        "ensemble_raw_spread_std": {m + 1: round(float(np.nanstd(raw_stack[:, m])), 3) for m in range(12)},
        "annual_mean_bc": round(float(np.nanmean(bc_stack)), 3),
        "annual_mean_raw": round(float(np.nanmean(raw_stack)), 3),
        "annual_spread_std_bc": round(float(np.nanstd(np.nanmean(bc_stack, axis=1))), 3),
        "per_model": {m: {
            "bc_monthly_anomaly": {k + 1: (None if np.isnan(per_model[m]["bc_future"][k] - obs_clim[k]) else round(float(per_model[m]["bc_future"][k] - obs_clim[k]), 3)) for k in range(12)},
            "raw_monthly_anomaly": {k + 1: (None if np.isnan(per_model[m]["raw_anomaly"][k]) else round(float(per_model[m]["raw_anomaly"][k]), 3)) for k in range(12)},
        } for m in done},
    }
    with open(os.path.join(HERE, "cmip6_ensemble_bc.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nWrote cmip6_ensemble_bc.json, n_models={len(done)}")
    print(f"Ensemble BC annual delta = {out['annual_mean_bc']:+.2f} C "
          f"(inter-model sd {out['annual_spread_std_bc']:.2f} C)")


if __name__ == "__main__":
    main()
