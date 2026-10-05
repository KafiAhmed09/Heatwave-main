#!/usr/bin/env python3
"""Validate saved model-redo outputs without refitting the models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_MODELS = {
    "Climatology",
    "Decision_Tree",
    "Dynamic_ARIMA",
    "LSTM",
    "Persistence",
    "Prophet",
    "Random_Forest",
    "Ridge_AR",
    "Seasonal_Naive",
}


def validate(output_root: Path) -> dict[str, object]:
    predictions = pd.read_csv(
        output_root / "predictions" / "all_model_predictions.csv",
        parse_dates=["origin_date", "date"],
    )
    saved_metrics = pd.read_csv(output_root / "metrics" / "regression_metrics_by_fold.csv")
    arima_status = pd.read_csv(output_root / "metrics" / "arima_convergence_status.csv")

    recalculated = (
        predictions.groupby(["fold", "model"])
        .apply(
            lambda group: pd.Series(
                {
                    "mae": np.mean(np.abs(group["y_true"] - group["y_pred"])),
                    "rmse": np.sqrt(np.mean((group["y_true"] - group["y_pred"]) ** 2)),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    comparison = saved_metrics.merge(
        recalculated,
        on=["fold", "model"],
        suffixes=("_saved", "_recalculated"),
    )

    expected_rows = len(EXPECTED_MODELS) * 6 * (365 + 365 + 182)
    checks = {
        "prediction_rows": int(len(predictions)),
        "expected_prediction_rows": int(expected_rows),
        "model_set_matches": set(predictions["model"].unique()) == EXPECTED_MODELS,
        "duplicate_forecast_keys": int(
            predictions.duplicated(["fold", "model", "station", "date"]).sum()
        ),
        "missing_observations": int(predictions["y_true"].isna().sum()),
        "missing_predictions": int(predictions["y_pred"].isna().sum()),
        "inconsistent_observations_across_models": int(
            (predictions.groupby(["fold", "station", "date"])["y_true"].nunique() > 1).sum()
        ),
        "forecast_horizons": sorted(int(value) for value in predictions["horizon_day"].unique()),
        "fold_date_ranges": {
            fold: {
                "start": str(group["date"].min().date()),
                "end": str(group["date"].max().date()),
            }
            for fold, group in predictions.groupby("fold")
        },
        "maximum_mae_recalculation_difference": float(
            np.max(np.abs(comparison["mae_saved"] - comparison["mae_recalculated"]))
        ),
        "maximum_rmse_recalculation_difference": float(
            np.max(np.abs(comparison["rmse_saved"] - comparison["rmse_recalculated"]))
        ),
        "arima_delivered_station_fold_fits": int(len(arima_status)),
        "arima_all_delivered_fits_converged": bool(arima_status["converged"].all()),
        "arima_fallback_station_fold_fits": int((arima_status["fallback_rank"] > 1).sum()),
    }
    checks["all_critical_checks_pass"] = bool(
        checks["prediction_rows"] == checks["expected_prediction_rows"]
        and checks["model_set_matches"]
        and checks["duplicate_forecast_keys"] == 0
        and checks["missing_observations"] == 0
        and checks["missing_predictions"] == 0
        and checks["inconsistent_observations_across_models"] == 0
        and checks["forecast_horizons"] == list(range(1, 8))
        and checks["maximum_mae_recalculation_difference"] < 1e-12
        and checks["maximum_rmse_recalculation_difference"] < 1e-12
        and checks["arima_all_delivered_fits_converged"]
    )
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    results = validate(arguments.output.resolve())
    destination = arguments.output.resolve() / "audit" / "output_validation.json"
    destination.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    if not results["all_critical_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
