# -*- coding: utf-8 -*-
"""eip_vc_model.py
================
Mechanistic model of the dengue extrinsic incubation period (EIP) and
temperature-dependent transmission potential (R0).

Temperature forcings (biting rate b(T), EIP rate sigma_v(T), and the
next-generation-matrix R0) follow published parameterisations: the
temperature-driven mechanistic dengue study for Guangzhou (Xu et al. 2017),
classical temperature-dependent development and transmission forms
(Briere 1999; Mordecai et al. 2017; Mordecai et al. 2019), and the
next-generation-matrix R0 framework (van den Driessche and Watmough 2002).
All biological parameters carry their literature sources.

Climate forcings:
- Observational baseline: ERA5 reanalysis daily temperature (Open-Meteo
  Archive API, Guangzhou grid cells).
- Future scenario: multi-model CMIP6 ensemble (Open-Meteo Climate API;
  historical 1995-2014 and future 2041-2050, a single high-emission-
  approximate projection with no SSP scenario split), bias-corrected by
  monthly quantile delta mapping (QDM) against ERA5 1995-2014 and averaged
  across models into monthly deltas; the inter-model standard deviation
  represents structural uncertainty (see bias_correct.py and
  cmip6_ensemble_bc.json).
"""

import json
import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. Baseline parameters for the R0 model
# ---------------------------------------------------------------------------
BASE = {
    "Nh": 10000.0,            # host population size (only the Nv/Nh ratio matters)
    "Nv_ratio": 10.0,         # baseline Nv/Nh ratio (high-density outbreak setting)
    "mu_h": 1.0 / (70.0 * 365.0),
    "sigma_h": 1.0 / 5.5,     # human latent period, ~5.5 d
    "gamma_h": 1.0 / 5.0,     # human infectious period, ~5 d
    "beta_vh": 0.70,          # mosquito-to-human transmission probability per bite
    "beta_hv": 0.60,          # human-to-mosquito transmission probability per bite
    "mu_v": 1.0 / 12.0,       # adult mosquito daily mortality (lifetime ~12 d)
    "T_ref": 28.0,
}

# Biologically valid temperature range for the EIP (enzyme-kinetics form
# calibrated for Aedes albopictus)
EIP_T_MIN = 13.0
EIP_T_MAX = 35.0


# ---------------------------------------------------------------------------
# 2. Temperature response functions
# ---------------------------------------------------------------------------
def biting_rate(T):
    """Daily biting rate b(T) for Aedes albopictus (Xu et al. 2017)."""
    b = 0.0043 * T + 0.0943
    return float(min(max(b, 0.18), 0.23))


def eip_rate(T):
    """EIP *rate* sigma_v(T) in 1/day, enzyme-kinetics (Schoolfield) form.

    Tk is temperature in Kelvin, R = 1.987 cal/(deg mol). Returns
    1 / EIP(days).
    """
    T = np.asarray(T, dtype=float)
    Tk = T + 273.15
    R = 1.987
    num = 24.0 * 0.00333 * (Tk / 298.0) * np.exp(60513.2 / R * (1.0 / 298.0 - 1.0 / Tk))
    den = 1.0 + np.exp(705550.0 / R * (1.0 / 308.352 - 1.0 / Tk))
    return num / den


def eip_days(T):
    """EIP in days, EIP(T) = 1 / sigma_v(T). NaN outside the valid range."""
    T = np.asarray(T, dtype=float)
    out = np.full_like(T, np.nan, dtype=float)
    valid = (T >= EIP_T_MIN) & (T <= EIP_T_MAX)
    sv = eip_rate(T[valid])
    ok = sv > 1e-6
    out[valid] = np.where(ok, 1.0 / sv, np.nan)
    return out


# ---------------------------------------------------------------------------
# 3. Transmission potential R0 (next-generation matrix,
#    van den Driessche & Watmough 2002)
# ---------------------------------------------------------------------------
def _next_generation(T, p):
    b = biting_rate(T)
    sv = eip_rate(T)
    Nh = p["Nh"]
    Nv = p["Nv_ratio"] * Nh
    beta_vh = p["beta_vh"]
    beta_hv = p["beta_hv"]
    sigma_h = p["sigma_h"]
    gamma_h = p["gamma_h"]
    mu_h = p["mu_h"]
    mu_v = p["mu_v"]

    F = np.zeros((4, 4))
    F[0, 3] = beta_vh * b                 # new human infections (via Iv)
    F[2, 1] = beta_hv * b * (Nv / Nh)     # new mosquito infections (via Ih)

    V = np.zeros((4, 4))
    V[0, 0] = sigma_h + mu_h
    V[1, 0] = -sigma_h
    V[1, 1] = gamma_h + mu_h
    V[2, 2] = sv + mu_v
    V[3, 2] = -sv
    V[3, 3] = mu_v

    return F @ np.linalg.inv(V)


def r0(T, p=None):
    """Basic reproduction number R0(T). T may be a scalar or an array."""
    if p is None:
        p = BASE
    T = np.asarray(T, dtype=float)
    scalar = T.ndim == 0
    T = np.atleast_1d(T)
    out = np.empty_like(T, dtype=float)
    for i, t in enumerate(T):
        K = _next_generation(t, p)
        out[i] = float(np.max(np.abs(np.linalg.eigvals(K))))
    return out[0] if scalar else out


# ---------------------------------------------------------------------------
# 4. Response window after an imported case
# ---------------------------------------------------------------------------
def response_window(T, viremia_days=4.5):
    """Response window (days) = EIP + human viremic period.

    Interpretation: after one infectious traveller arrives, this is the time
    before local mosquitoes can transmit, i.e. the lead time available to
    control teams. viremia_days uses the dengue midpoint estimate of 4.5 d.
    """
    return eip_days(T) + viremia_days


# ---------------------------------------------------------------------------
# 5. Data interfaces
# ---------------------------------------------------------------------------
def load_era5_daily(json_path):
    """Read an Open-Meteo-format ERA5 daily JSON into a DataFrame
    [date, tmean, rain, rh]."""
    with open(json_path, "r", encoding="utf-8") as f:
        d = json.load(f)
    daily = d["daily"]
    df = pd.DataFrame({
        "date": pd.to_datetime(daily["time"]),
        "tmean": np.asarray(daily["temperature_2m_mean"], dtype=float),
        "rain": np.asarray(daily["precipitation_sum"], dtype=float),
        "rh": np.asarray(daily["relative_humidity_2m_mean"], dtype=float),
    })
    return df


def load_ensemble_bc(path=None):
    """Read the bias-corrected multi-model ensemble output
    cmip6_ensemble_bc.json produced by bias_correct.py. Looks in this
    directory first, then in ../03_Supplementary."""
    if path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        cand = [os.path.join(here, "cmip6_ensemble_bc.json"),
                os.path.join(here, "..", "03_Supplementary", "cmip6_ensemble_bc.json")]
        path = next((c for c in cand if os.path.exists(c)), cand[0])
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def cmip6_ensemble_deltas():
    """Monthly deltas of the bias-corrected multi-model ensemble (deg C,
    dict month -> delta), from the 7 CMIP6 models (historical 1995-2014,
    future 2041-2050) after monthly QDM bias correction against ERA5
    1995-2014, averaged across models."""
    a = load_ensemble_bc()
    return {int(m): float(v) for m, v in a["ensemble_bc_monthly_anomaly"].items()}


def cmip6_ensemble_spread():
    """Inter-model standard deviation of the monthly deltas (deg C),
    representing structural uncertainty."""
    a = load_ensemble_bc()
    return {int(m): float(v) for m, v in a["ensemble_bc_spread_std"].items()}


def cmip6_scaled_deltas(target_annual):
    """Scale the bias-corrected ensemble monthly *shape* to a given annual
    mean (deg C), keeping the ensemble's real seasonal structure and changing
    only the magnitude. Used for the warming-envelope sensitivity analysis."""
    d = cmip6_ensemble_deltas()
    cur = sum(d.values()) / 12.0
    k = target_annual / cur
    return {m: v * k for m, v in d.items()}


def uniform_warming_deltas(target_annual):
    """Uniform-warming sensitivity: apply the same delta every month so the
    annual mean equals target_annual (deg C). Examines the effect of "overall
    warming of X deg C" on season length and the response window without
    reference to any specific model's seasonal structure."""
    return {m: float(target_annual) for m in range(1, 13)}
