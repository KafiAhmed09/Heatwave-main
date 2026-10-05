# Heatwave TMAX Forecasting — Rebuilt Model Package

This package replaces the original notebook scores with one leakage-controlled experiment for six Bangladesh stations. It forecasts daily maximum temperature (`TMAX`) for horizons 1–7 days, evaluates heatwave days and events separately, and reserves January–June 2024 as a final holdout.

## Result in one paragraph

Dynamic harmonic ARIMA (`Dynamic_ARIMA`) was selected before opening the final test because it had the lowest mean validation RMSE across 2022 and 2023 (2.242°C, versus 2.254°C for `Ridge_AR`). On the untouched 2024H1 holdout, `Ridge_AR` was numerically best (MAE 1.965°C; RMSE 2.679°C), followed by `Dynamic_ARIMA` (MAE 2.012°C; RMSE 2.742°C). Their paired station-week MAE difference was not supported at 95% confidence. For the primary heatwave definition, persistence produced the best final event F1 (0.514), while `Dynamic_ARIMA` reached event F1 0.387. Temperature-error ranking and event-detection ranking therefore answer different questions.

## Experiment protocol

- Target: station-level daily `TMAX`; no `TAVG` scores are mixed into the comparison.
- Forecast task: non-overlapping seven-day rolling-origin blocks. At each weekly origin, later days in the block are recursive forecasts; observed history is available again at the next origin.
- Outer folds:
  - train through 2021, validate on 2022;
  - expand through 2022, validate on 2023;
  - expand through 2023, final test on 1 January–30 June 2024.
- Hyperparameters: tuned only within each outer training period using its final 182 days as an inner temporal validation window.
- Selection: lowest mean RMSE over the two validation folds. The 2024H1 holdout is never used for model selection.
- Uncertainty: station-week block bootstrap for MAE/RMSE; 90% split-conformal intervals calibrated from 2022–2023 validation residuals.
- Randomness: seed 42; deterministic TensorFlow operations requested; chronological LSTM training with `shuffle=False`.

The compared methods are persistence, 365-day seasonal naive, station/day-of-year climatology, regularized autoregression (`Ridge_AR`), decision tree, random forest, dynamic harmonic ARIMA with annual Fourier terms, Prophet, and a compact station-specific LSTM.

## Heatwave outcome definition

The primary operational definition is `TMAX > 36°C` for at least three consecutive station-days; `TMAX >= 36°C` is included as a sensitivity analysis. This is anchored to Bangladesh Meteorological Department descriptions of a heatwave as temperature exceeding 36°C for at least three consecutive days ([BMD study](https://bmd.gov.bd/file/2024/08/14/pdf/171461.pdf); [BMD climate report](https://server6.bmd.gov.bd/file/2025/11/20/pdf/195905.pdf)). The source wording also includes a large-area condition, which cannot be established from only six station series; all package results are explicitly station-level operationalizations.

The last two days of each evaluation window are excluded from heatwave classification metrics to avoid right-censoring: a hot day at the boundary cannot be confirmed as part of a three-day run without two later observations. Temperature-regression metrics retain every test date.

## Data audit

- 12,048 rows: six stations × 2,008 complete daily dates, 1 January 2019–30 June 2024.
- No null cells, exact duplicate rows, duplicate station-date keys, or missing calendar dates.
- One row has `TMAX < TMIN` (Gopalganj, 3 May 2022); 72 `TAVG` values fall outside the reported min/max interval.
- No unverified correction or imputation was made. Modeling uses `TMAX` and its own history, so the `TAVG` inconsistencies do not enter the feature set.
- This record is too short to support a defensible climate projection to 2037. The rebuilt task is short-range forecasting, not climate projection.

## Reproduce

From this directory, using Python 3.12:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/heatwave_model_redo.py --csv data/cleandata.csv --output outputs
python src/validate_outputs.py --output outputs
```

The full run trains Prophet and 18 station-fold LSTM fits and can take substantial CPU time. The included notebook reads the completed outputs for rapid review.

## Package map

- `Heatwave_Model_Redo.ipynb`: executed, reader-facing analysis notebook.
- `src/heatwave_model_redo.py`: full from-scratch pipeline.
- `src/validate_outputs.py`: independent structural and metric-recalculation checks.
- `src/rebuild_arima_only.py`: convergence-safe ARIMA refit/rebuild utility.
- `data/cleandata.csv`: exact source-data copy used for the run.
- `outputs/metrics/`: fold, station, horizon, uncertainty, paired-comparison, tuning, and heatwave metrics.
- `outputs/predictions/`: all daily forecasts and labeled heatwave predictions.
- `outputs/figures/`: publication-ready PNG figures and confusion matrices.
- `outputs/audit/`: data audit, protocol, environment, original-notebook audit, and validation report.
- `MODEL_HANDOFF.md`: paper-revision interpretation and claims that are safe to make.

## Important limitations

Only temperature is available: humidity, wind, rainfall, pressure, and numerical weather prediction fields are absent. Heatwave positives are imbalanced and concentrated by station. These statistical forecasts are not a substitute for an operational meteorological forecast system.
