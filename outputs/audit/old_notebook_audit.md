# Audit of the original modeling notebooks

| Notebook | Material issue | Consequence |
|---|---|---|
| Prophet error-rate | Uses `TAVG` as the target while the paper reports `TMAX`. | The reported Prophet scores are not comparable with the other models. |
| Prophet core | Forecasts 15 years from roughly 5.5 years of data; event code requires more than three days (`>3`), splits an event when severity changes, and leaves gaps at 38–39°C and 40–41°C. | Long-horizon and heatwave-event outputs are unsupported or incorrectly classified. |
| Random Forest | Uses a shuffled `train_test_split` and only station/year/month/day features. | Future dates leak into training and the score is optimistic for forecasting. |
| Decision Tree | Uses the same shuffled split and date-only feature setup. | Future dates leak into training and the result is not a time-series forecast. |
| LSTM | Fits scaling on the full dataset, uses the test set as validation, and the global sequence crosses station boundaries; the later station models also fit scalers before splitting. | Multiple leakage paths and physically invalid sequences make the score unusable. |
| ARIMA | Uses an 80/20 split and seasonal period `m=52` on daily data. | The seasonal specification does not represent annual daily seasonality and the comparison protocol differs from the tree models. |

The redevelopment pipeline replaces every reported score; none of the original error rates are carried forward.
