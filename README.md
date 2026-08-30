# dengue-climate-mechanistic-projection

Code and data for the study "Future dengue outbreak risk in Guangzhou under climate change: a mechanistic projection coupling temperature-dependent vector competence with reanalysis and CMIP6 meteorology".

## Contents

- `fetch_ensemble.py` - retrieves the ERA5 observational reference and seven CMIP6 models from the Open-Meteo APIs (resumable, with rate-limit backoff)
- `bias_correct.py` - monthly quantile delta mapping (QDM) bias correction against ERA5 1995-2014; builds the multi-model ensemble (`cmip6_ensemble_bc.json`)
- `eip_vc_model.py` - mechanistic EIP / R0 / response-window model
- `run_analysis.py` - main analysis; generates the manuscript figures (`figures/`) and `results.csv`
- `raw_cmip6/` - raw daily temperature of the seven CMIP6 models (hist 1995-2014, future 2041-2050)
- `era5_obs_1995_2014.json` - ERA5 observational baseline
- `data_era5/era5_haizhu.json` - ERA5 daily temperature for the Haizhu grid cell, 2015-2022

## Reproduce

```bash
python bias_correct.py     # writes cmip6_ensemble_bc.json
python run_analysis.py     # writes figures and results.csv into outputs/
```

Requires Python 3 with numpy, pandas, and matplotlib.
