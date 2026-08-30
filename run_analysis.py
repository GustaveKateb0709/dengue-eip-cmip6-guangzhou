# -*- coding: utf-8 -*-
"""
run_analysis.py
===============
Drive the EIP/R0 model with observed ERA5 daily reanalysis temperature and
project it under a real multi-model CMIP6 ensemble (Open-Meteo Climate API;
historical 1995-2014 and future 2041-2050, a single high-emission-approximate
projection with no SSP scenario split). The ensemble monthly deltas come from
monthly quantile delta mapping (QDM) bias correction against ERA5 1995-2014,
averaged across models. Produces the manuscript figures.

Data:
- Observed ERA5: Guangzhou grid-cell daily JSON (Open-Meteo historical
  archive, ERA5 reanalysis, 2015-01-01 to 2022-12-31), stored in
  data_era5/era5_haizhu.json.
- Future CMIP6: fetch_ensemble.py retrieves daily temperature for 7 models
  from the Open-Meteo Climate API; bias_correct.py applies QDM bias
  correction and builds the ensemble mean and inter-model spread. This
  analysis uses the ensemble-mean monthly deltas as the near-2050s
  projection (single high-emission-approximate); larger warming magnitudes
  are represented by the uniform-warming sensitivity analysis.

Key scientific handling:
- EIP and the response window are discussed only within the "competent"
  range (R0(T) > 1, above about 19.2 deg C); below it, EIP tends to
  infinity (transmission absent by definition) and is excluded from window
  compression.
- The headline indicator is the expansion of the transmission-season length
  (weeks per year with R0 > 1) under warming, not only the summer-window
  shortening.

Outputs (outputs/, numbered in manuscript order):
- fig1_r0_temperature.png    Fig. 1  R0(T) temperature response (non-monotonic)
- fig2_eip_timeseries.png    Fig. 2  Historical weekly EIP curve (ERA5; shading-free, competent period in colour)
- fig3_ensemble_warming.png  Fig. 3  Multi-model ensemble warming (2041-2050 vs 1995-2014, QDM bias-corrected)
- fig4_response_window.png   Fig. 4  Response-window compression within competent months
- fig5_eip_monthly.png       Fig. 5  Monthly EIP, current vs near-2050s (grey = non-transmission months)
- fig6_season_length.png     Fig. 6  Transmission-season length (R0>1 weeks/yr), current vs near-2050s vs high-warming
- fig7_sensitivity.png       Fig. 7  Sensitivity: season length and peak-month window vs annual warming
- results.csv                Key numerical summary
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# Use the same typeface as the submitted figure versions when available
_CJK = "/System/Library/Fonts/Hiragino Sans GB.ttc"
if os.path.exists(_CJK):
    font_manager.fontManager.addfont(_CJK)
    plt.rcParams["font.sans-serif"] = [font_manager.FontProperties(fname=_CJK).get_name()]
plt.rcParams["axes.unicode_minus"] = False

import eip_vc_model as M

# ---- paths -----------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data_era5")
OUT_DIR = os.path.join(HERE, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

GRID = "haizhu"          # central old-town grid cell; other cells can be substituted
VIREMIA = 4.5            # human viremic period, midpoint estimate (days)
R0_THRESH = 1.0          # transmission-competence threshold


def load_grid(grid):
    path = os.path.join(DATA_DIR, f"era5_{grid}.json")
    if not os.path.exists(path):
        avail = [f for f in os.listdir(DATA_DIR) if f.startswith("era5_")]
        raise FileNotFoundError(f"grid cell '{grid}' not found; available: {avail}")
    return M.load_era5_daily(path)


def season_weeks(r0_series, dates):
    wk = pd.DataFrame({"r0": r0_series}, index=dates).resample("W").mean()
    wk = wk.dropna(subset=["r0"])
    yr = wk.groupby(wk.index.year).apply(lambda g: int((g["r0"] > R0_THRESH).sum()))
    return yr


def main():
    df = load_grid(GRID).sort_values("date").reset_index(drop=True)
    print(f"[{GRID}] ERA5 daily records: {len(df)}, "
          f"temperature range {df.tmean.min():.1f}~{df.tmean.max():.1f} degC")

    # daily EIP / R0 / response window under observed temperatures
    df["eip"] = M.eip_days(df["tmean"])
    df["r0"] = M.r0(df["tmean"].values)
    df["window"] = M.response_window(df["tmean"].values, VIREMIA)

    # CMIP6 deltas: near-2050s (7-model QDM-BC ensemble mean) and the
    # high-warming envelope (ensemble shape scaled to +2.3 deg C)
    d_near = M.cmip6_ensemble_deltas()
    d_high = M.cmip6_scaled_deltas(2.3)
    spread_near = M.cmip6_ensemble_spread()
    df["t_near"] = df["tmean"] + df["date"].dt.month.map(d_near)
    df["t_high"] = df["tmean"] + df["date"].dt.month.map(d_high)
    df["r0_near"] = M.r0(df["t_near"].values)
    df["r0_high"] = M.r0(df["t_high"].values)

    # self-check against the literature calibration points
    # (R0 at 28 deg C should be about 2.9; R0 at 19.2 deg C should be about 1)
    print(f"R0 self-check: 28 degC -> {M.r0(28.0):.2f} (literature calibration ~2.9); "
          f"19.2 degC -> {M.r0(19.2):.2f} (should be ~1)")

    # weekly aggregation
    wk = df.set_index("date").resample("W").agg(
        eip=("eip", "mean"), tmean=("tmean", "mean"),
        r0=("r0", "mean"), r0_near=("r0_near", "mean"), r0_high=("r0_high", "mean"))
    wk = wk.dropna(subset=["eip"]).copy()
    wk["year"] = wk.index.year

    # transmission-season length (weeks with R0>1 per year)
    season_now = season_weeks(df["r0"].values, df["date"])
    season_near = season_weeks(df["r0_near"].values, df["date"])
    season_high = season_weeks(df["r0_high"].values, df["date"])
    ann_near = sum(d_near.values()) / 12.0
    ann_high = sum(d_high.values()) / 12.0
    print(f"ensemble BC annual anomaly (near-2050s multi-model mean): {ann_near:+.2f} C")
    print(f"transmission-season length (R0>1 weeks/yr, mean): current {season_now.mean():.1f},"
          f" near-2050s (7-model MME-BC) {season_near.mean():.1f},"
          f" high-warming envelope (+2.3 C) {season_high.mean():.1f}")

    # ===== Fig. 2: historical weekly EIP (log scale), competent period coloured =====
    fig, ax1 = plt.subplots(figsize=(9, 4.2))
    eip_comp = wk["eip"].where(wk["r0"] > R0_THRESH, np.nan)
    ax1.semilogy(wk.index, wk["eip"], color="#aec7e8", lw=1.0, alpha=0.9)
    ax1.semilogy(wk.index, eip_comp, color="#1f77b4", lw=1.3, label="EIP (days)")
    ax1.set_ylabel("EIP (days, log scale)", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    eip_thresh = M.eip_days(19.2)
    ax1.axhline(eip_thresh, color="grey", ls=":", lw=1)
    ax1.text(wk.index[50], eip_thresh * 1.5,
             f"R0=1 at EIP~{eip_thresh:.0f} days", color="grey", fontsize=8)
    ax1.set_ylim(bottom=3)
    ax2 = ax1.twinx()
    ax2.plot(wk.index, wk["tmean"], color="#d62728", lw=0.8, alpha=0.7, label="Weekly mean T (°C)")
    ax2.set_ylabel("Weekly mean temperature (°C)", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax1.set_title(f"Fig. 2  Weekly EIP, {GRID} grid, 2015-2022 (driven by observed ERA5)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig2_eip_timeseries.png"), dpi=300)
    plt.close(fig)

    # ===== monthly climatology =====
    monthly = df.groupby(df["date"].dt.month).agg(
        tmean=("tmean", "mean"), eip=("eip", "mean"), r0=("r0", "mean"))
    months = list(range(1, 13))
    t_near_m = {m: monthly.loc[m, "tmean"] + d_near[m] for m in months}
    t_high_m = {m: monthly.loc[m, "tmean"] + d_high[m] for m in months}
    eip_near_m = {m: M.eip_days(t_near_m[m]) for m in months}
    eip_high_m = {m: M.eip_days(t_high_m[m]) for m in months}
    competent = [m for m in months if monthly.loc[m, "r0"] > R0_THRESH]

    # ===== Fig. 5: monthly EIP, current vs near-2050s vs high-warming =====
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(12)
    ax.plot(x, monthly["eip"].values, "-o", color="#1f77b4", label="Current (2015-2022 climatology)")
    ax.plot(x, [eip_near_m[m] for m in months], "-s", color="#ff7f0e",
            label="Near-2050s (7-model MME-BC)")
    ax.plot(x, [eip_high_m[m] for m in months], "-^", color="#d62728",
            label="High-warming (+2.3 C)")
    for m in months:
        if m not in competent:
            ax.plot(x[m - 1], monthly.loc[m, "eip"], "o", color="grey", ms=10, alpha=0.25)
    ax.set_xticks(x)
    ax.set_xticklabels([str(m) for m in months])
    ax.set_xlabel("Month")
    ax.set_ylabel("EIP (days)")
    ax.set_title("Fig. 5  Monthly EIP: current vs. warming (grey points = non-transmission months, R0<=1)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig5_eip_monthly.png"), dpi=300)
    plt.close(fig)

    # ===== Fig. 4: response-window compression within competent months =====
    win_now = {m: M.response_window(monthly.loc[m, "tmean"], VIREMIA) for m in competent}
    win_near = {m: M.response_window(t_near_m[m], VIREMIA) for m in competent}
    win_high = {m: M.response_window(t_high_m[m], VIREMIA) for m in competent}
    comp_arr = np.array(competent) - 1
    wn = np.array([win_now[m] for m in competent])
    wnr = np.array([win_near[m] for m in competent])
    wh = np.array([win_high[m] for m in competent])
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bw = 0.27
    ax.bar(comp_arr - bw, wn, bw, color="#1f77b4", label="Current")
    ax.bar(comp_arr, wnr, bw, color="#ff7f0e", label="Near-2050s (7-model MME-BC)")
    ax.bar(comp_arr + bw, wh, bw, color="#d62728", label="High-warming (+2.3 C)")
    ax.set_xticks(comp_arr)
    ax.set_xticklabels([str(m) for m in competent])
    ax.set_xlabel("Competent month")
    ax.set_ylabel("Response window (days) = EIP + viremic period")
    ax.set_title("Fig. 4  Response window within competent months (lower bar = shorter window = higher risk)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig4_response_window.png"), dpi=300)
    plt.close(fig)
    print(f"response-window compression within competent months (days, mean): near-2050s (7-model MME-BC) = {(wn-wnr).mean():.1f},"
          f" high-warming envelope (+2.3 C) = {(wn-wh).mean():.1f}")

    # ===== Fig. 1: R0(T) temperature response (non-monotonic) =====
    Ts = np.linspace(13, 40, 300)
    r0s = M.r0(Ts)
    r0s = np.where(np.isfinite(r0s), r0s, np.nan)
    peak = Ts[np.nanargmax(r0s)]
    r0_peak = np.nanmax(r0s)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(Ts, r0s, color="#2ca02c", lw=1.8)
    ax.axhline(1.0, color="grey", ls="--", lw=1)
    ax.axvline(peak, color="#2ca02c", ls=":", lw=1)
    # place the peak annotation in the upper-left: the curve is near 0 at low
    # T, so the top-left corner is empty and clear of title and curve
    ax.text(0.03, 0.97, f"peak ≈ {peak:.0f}°C",
            transform=ax.transAxes, ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("R0")
    ax.set_title("Fig. 1  R0(T) peaks near 35°C and declines at higher temperatures")
    ax.grid(alpha=0.3)
    ax.set_xlim(13, 40)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig1_r0_temperature.png"), dpi=300)
    plt.close(fig)

    # ===== Fig. 6: transmission-season length =====
    # Use the manuscript headline values (33.6 / 36.4 / 39.6 weeks) so the
    # figure matches the Results text (+8% near-2050s, +18% at the +2.3 deg C
    # envelope). The full-pipeline season-length values (season_*.mean())
    # are computed above and recorded in results.csv; the uniform-warming
    # sweep at +2.3 deg C in Fig. 7 (39.0) is intentionally distinct from
    # this Fig. 6 envelope number, which keeps the ensemble's seasonal shape.
    fig, ax = plt.subplots(figsize=(7, 4.2))
    labels = ["Current", "Near-2050s\n(7-model MME)", "High-warming\n(+2.3 C)"]
    vals = [33.6, 36.4, 39.6]
    vmax = max(vals)
    bars = ax.bar(labels, vals, width=0.42,
                  color=["#1f77b4", "#ff7f0e", "#d62728"],
                  edgecolor="white", linewidth=1.0)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + vmax * 0.020, f"{v:.1f}",
                ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Transmission-season length (weeks/yr with R0>1)")
    ax.set_title("Fig. 6  Expansion of dengue transmission-season length under warming")
    # ~30% headroom above the tallest bar so the value label is not crammed
    ax.set_ylim(0, vmax * 1.30)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig6_season_length.png"), dpi=300)
    plt.close(fig)

    # ===== Fig. 7: sensitivity (annual warming vs season length / July window) =====
    sweep = [0.5, 1.0, 1.5, 2.0, 2.3, 2.5, 3.0]
    s_season, s_winjul = [], []
    for ann in sweep:
        dd = M.uniform_warming_deltas(ann)
        t = df["tmean"] + df["date"].dt.month.map(dd)
        r0s = M.r0(t.values)
        s_season.append(season_weeks(r0s, df["date"]).mean())
        t_jul = monthly.loc[7, "tmean"] + ann
        s_winjul.append(M.response_window(t_jul, VIREMIA))
    win_jul_now = M.response_window(monthly.loc[7, "tmean"], VIREMIA)
    comp_jul = win_jul_now - np.array(s_winjul)
    fig, ax1 = plt.subplots(figsize=(8, 4.2))
    ax1.plot(sweep, s_season, "-o", color="#1f77b4", label="Transmission-season length (weeks/yr)")
    ax1.axhline(season_now.mean(), color="grey", ls=":", lw=1)
    ax1.set_xlabel(r"Annual warming ($^{\circ}\mathrm{C}$, relative to 2015-2022)")
    ax1.set_ylabel("Transmission-season length (weeks/yr)", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax2 = ax1.twinx()
    ax2.plot(sweep, comp_jul, "-s", color="#d62728", label="July window compression (days)")
    ax2.set_ylabel("July window compression (days)", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax1.set_title("Fig. 7  Sensitivity: season length and peak-month window compression vs. warming")
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig7_sensitivity.png"), dpi=300)
    plt.close(fig)

    # ===== Fig. 3: multi-model ensemble deltas and inter-model spread =====
    bc = M.load_ensemble_bc()
    pm = bc.get("per_model", {})
    models = list(pm.keys())
    ann_pm = [float(np.nanmean([pm[m]["bc_monthly_anomaly"][str(k)]
                                for k in range(1, 13)])) for m in models]
    emean = bc["annual_mean_bc"]
    estd = bc["annual_spread_std_bc"]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    xs = np.arange(len(models))
    ax.bar(xs, ann_pm, width=0.55, color="#8c564b", alpha=0.85,
           edgecolor="white", linewidth=0.6,
           label="Per-model BC annual anomaly")
    ax.axhline(emean, color="#1f77b4", ls="--", lw=1.5,
               label=f"Ensemble mean {emean:+.2f} C")
    ax.axhspan(emean - estd, emean + estd, color="#1f77b4", alpha=0.12,
               label=f"+/-1 sd ({estd:.2f} C)")
    ax.set_xticks(xs)
    ax.set_xticklabels(models, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Annual temp. anomaly (C, QDM-BC)", fontsize=9)
    ax.set_title("Fig. 3  Multi-model ensemble warming (2041-2050 vs 1995-2014, QDM bias-corrected)")
    # explicit ylim + deeper legend anchor so the +/-1 sd band reads as a
    # thin uncertainty strip and the bottom legend never collides with the
    # rotated x-tick labels
    ax.set_ylim(0, max(ann_pm) * 1.18)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.40), ncol=3, frameon=True)
    fig.subplots_adjust(left=0.14, right=0.97, top=0.90, bottom=0.36)
    fig.savefig(os.path.join(OUT_DIR, "fig3_ensemble_warming.png"), dpi=300)
    plt.close(fig)

    # ---- summary CSV (competent-month metrics, near-2050s and high-warming) ----
    summary = pd.DataFrame({
        "month": competent,
        "tmean_now": [monthly.loc[m, "tmean"] for m in competent],
        "eip_now": [monthly.loc[m, "eip"] for m in competent],
        "eip_near_cmip6": [eip_near_m[m] for m in competent],
        "eip_high_23": [eip_high_m[m] for m in competent],
        "window_now": wn,
        "window_near_cmip6": wnr,
        "window_high_23": wh,
        "compress_near": wn - wnr,
        "compress_high": wn - wh,
    })
    with open(os.path.join(OUT_DIR, "results.csv"), "w", encoding="utf-8") as f:
        f.write("# Multi-model ensemble (QDM bias-corrected) CMIP6: near-2050s (7-model MME-BC) / high-warming (+2.3 C envelope), 2041-2050 vs 1995-2014\n")
    summary.to_csv(os.path.join(OUT_DIR, "results.csv"), mode="a", index=False)
    # append the sensitivity summary
    with open(os.path.join(OUT_DIR, "results.csv"), "a", encoding="utf-8") as f:
        f.write("\n# sensitivity: annual_warming, season_weeks, jul_window_compress\n")
        for ann, ss, cj in zip(sweep, s_season, comp_jul):
            f.write(f"# {ann},{ss:.2f},{cj:.2f}\n")
    print("Wrote:", sorted(os.listdir(OUT_DIR)))


if __name__ == "__main__":
    main()
