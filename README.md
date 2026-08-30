# dengue-eip-cmip6-guangzhou

Code and data for the study "Future dengue outbreak risk in Guangzhou under climate change: a mechanistic projection coupling temperature-dependent vector competence with reanalysis and CMIP6 meteorology".

## Contents

Analysis code:

- `fetch_ensemble.py` - retrieves the ERA5 observational reference and seven CMIP6 models from the Open-Meteo APIs (resumable, with rate-limit backoff)
- `fetch_era5_study_period.py` - retrieves ERA5 daily temperature for the Haizhu grid cell, 2015-2022, and writes `data_era5/era5_haizhu.json`
- `bias_correct.py` - monthly quantile delta mapping (QDM) bias correction against ERA5 1995-2014; builds the multi-model ensemble (`cmip6_ensemble_bc.json`)
- `eip_vc_model.py` - mechanistic EIP / R0 / response-window model
- `run_analysis.py` - main analysis; regenerates the manuscript figures and `results.csv`

Data shipped with the repository:

- `raw_cmip6/` - raw daily temperature of the seven CMIP6 models (historical 1995-2014, future 2041-2050)
- `era5_obs_1995_2014.json` - ERA5 observational baseline used for bias correction
- `data_era5/era5_haizhu.json` - ERA5 daily temperature for the Haizhu grid cell, 2015-2022
- `cmip6_ensemble_bc.json` - bias-corrected multi-model ensemble produced by `bias_correct.py`
- `results.csv` - the numerical results reported in the manuscript

The raw forcing data are included so that the pipeline can be rerun without network access. They can also be regenerated from source with the two fetch scripts.

## Reproduce

```
python bias_correct.py     # writes cmip6_ensemble_bc.json
python run_analysis.py     # writes figures and results.csv
```

To rebuild the raw inputs from the Open-Meteo APIs instead of using the copies in this repository:

```
python fetch_ensemble.py
python fetch_era5_study_period.py
```

Requires Python 3 with numpy, pandas, and matplotlib.
