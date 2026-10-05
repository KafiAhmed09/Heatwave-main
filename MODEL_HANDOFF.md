# Model-redo handoff for the conference-paper revision

## Claims supported by the rebuilt experiment

1. **Validation selection:** Dynamic harmonic ARIMA was selected by the prespecified rule, with mean 2022–2023 validation RMSE 2.242°C and MAE 1.652°C.
2. **Final holdout:** On January–June 2024, `Ridge_AR` had the lowest point errors (MAE 1.965°C, RMSE 2.679°C, R² 0.745), followed by `Dynamic_ARIMA` (MAE 2.012°C, RMSE 2.742°C, R² 0.733).
3. **No decisive Ridge-versus-ARIMA separation:** Relative to validation-selected `Dynamic_ARIMA`, the paired block-bootstrap Ridge MAE difference was −0.048°C (95% CI −0.123 to 0.023). Negative means Ridge was numerically better; the interval crosses zero.
4. **Separation from the remaining methods:** On final-test MAE, `Dynamic_ARIMA` was significantly better at the 95% block-bootstrap level than random forest, climatology, Prophet, decision tree, LSTM, persistence, and seasonal naive.
5. **Heatwave events remain difficult:** Under the primary station-level definition, persistence had final heatwave-day F1 0.587 and event F1 0.514. `Dynamic_ARIMA` had day F1 0.442 and event F1 0.387. The stronger continuous-temperature model was more precise but less sensitive near the threshold.
6. **Uncertainty is imperfect:** The 90% conformal interval for `Dynamic_ARIMA` had 85.3% final-test coverage (half-width 3.687°C), indicating undercoverage under the 2024 distribution shift.

## Final-test regression table

| Model | MAE (°C) | RMSE (°C) | MAPE (%) | sMAPE (%) | R² | Bias (°C) |
|---|---:|---:|---:|---:|---:|---:|
| Ridge_AR | 1.965 | 2.679 | 6.902 | 6.565 | 0.745 | 0.320 |
| Dynamic_ARIMA | 2.012 | 2.742 | 7.109 | 6.750 | 0.733 | 0.383 |
| Random_Forest | 2.146 | 2.910 | 7.432 | 7.150 | 0.699 | −0.123 |
| Climatology | 2.193 | 2.931 | 7.629 | 7.321 | 0.694 | 0.018 |
| LSTM | 2.260 | 2.952 | 7.779 | 7.532 | 0.690 | −0.221 |
| Prophet | 2.198 | 2.954 | 7.847 | 7.394 | 0.690 | 0.534 |
| Decision_Tree | 2.259 | 3.028 | 7.838 | 7.565 | 0.674 | 0.022 |
| Persistence | 2.447 | 3.308 | 8.500 | 8.340 | 0.611 | −0.045 |
| Seasonal_Naive | 3.167 | 3.991 | 10.913 | 10.559 | 0.433 | 0.613 |

## Claims to remove or rewrite

- Do not reuse any error rate from the original notebooks. Their targets and validation designs were inconsistent, and several contained leakage.
- Do not call Prophet the best model; it was seventh by validation RMSE and sixth by final RMSE in the rebuilt comparison.
- Do not claim a defensible forecast to 2037 from this dataset. Five and a half years of station temperature data support the short-range task evaluated here, not a climate projection.
- Do not describe the station-level labels as official nationwide BMD declarations. The large-area component is not observable in this dataset.
- Do not claim that the lowest-RMSE model is also the best heatwave detector. Persistence produced the highest event F1 on the final holdout.
- Do not select `Ridge_AR` retrospectively as the single winner solely because it led the final holdout. Report the prespecified validation selection and the holdout result together.

## Reporting notes

- Name the design “expanding-window, seven-day rolling-origin evaluation.”
- State that forecasts are recursive within each week and refreshed with observed history at the next weekly origin.
- Report validation and final-test results separately.
- Pair point metrics with station-week bootstrap intervals and the paired comparison.
- Present strict `>36°C` results as primary and `>=36°C` as a threshold sensitivity check.
- Explain the two-day right-censoring exclusion in heatwave classification metrics.
- Use “station-level heatwave event” throughout.

The CSV tables in `outputs/metrics/` are the numerical source of truth for the next paper phase.
