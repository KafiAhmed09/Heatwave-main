#!/usr/bin/env python3
"""Refit the ARIMA component with strict convergence checks and rebuild outputs.

This utility is safe to run after the main pipeline because it replaces only
Dynamic_ARIMA predictions and then deterministically recomputes downstream
metrics, comparisons, heatwave labels, figures, and the run summary.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import pandas as pd

from heatwave_model_redo import (
    FOLDS,
    arima_outer_predictions,
    create_figures,
    final_test_uncertainty,
    heatwave_metrics_for_predictions,
    load_and_validate_data,
    make_regression_tables,
    paired_block_bootstrap,
    tune_arima_orders_by_station,
)


def rebuild(csv_path: Path, output_root: Path) -> None:
    warnings.filterwarnings("ignore")
    df, audit, _ = load_and_validate_data(csv_path)
    predictions_path = output_root / "predictions" / "all_model_predictions.csv"
    predictions = pd.read_csv(predictions_path, parse_dates=["origin_date", "date"])
    predictions = predictions[predictions["model"] != "Dynamic_ARIMA"].copy()

    tuning_path = output_root / "metrics" / "inner_tuning_results.csv"
    tuning = pd.read_csv(tuning_path)
    tuning = tuning[tuning["model"] != "Dynamic_ARIMA"].copy()

    hyperparameter_path = output_root / "metrics" / "model_hyperparameters.csv"
    hyperparameters = pd.read_csv(hyperparameter_path)
    hyperparameters = hyperparameters[hyperparameters["model"] != "Dynamic_ARIMA"].copy()

    new_predictions: list[pd.DataFrame] = []
    new_tuning: list[pd.DataFrame] = []
    status_frames: list[pd.DataFrame] = []
    new_hyperparameters: list[dict[str, str]] = []
    for fold in FOLDS:
        print(f"ARIMA convergence rebuild: {fold.name}", flush=True)
        rankings, fold_tuning = tune_arima_orders_by_station(df, pd.Timestamp(fold.train_end))
        fold_tuning["fold"] = fold.name
        fold_predictions, fold_status = arima_outer_predictions(
            df,
            pd.Timestamp(fold.train_end),
            pd.Timestamp(fold.test_start),
            pd.Timestamp(fold.test_end),
            fold.name,
            rankings,
        )
        new_predictions.append(fold_predictions)
        new_tuning.append(fold_tuning)
        status_frames.append(fold_status)
        for _, status in fold_status.iterrows():
            new_hyperparameters.append(
                {
                    "fold": fold.name,
                    "model": "Dynamic_ARIMA",
                    "station": str(status["station"]),
                    "parameters": json.dumps(
                        {
                            "order": str(status["order"]),
                            "annual_fourier_order": 3,
                            "validation_rank_used": int(status["fallback_rank"]),
                        },
                        sort_keys=True,
                    ),
                }
            )

    predictions = pd.concat([predictions, *new_predictions], ignore_index=True)
    predictions = predictions.sort_values(["fold", "model", "station", "date"]).reset_index(drop=True)
    predictions.to_csv(predictions_path, index=False)
    pd.concat([tuning, *new_tuning], ignore_index=True).to_csv(tuning_path, index=False)
    pd.concat([hyperparameters, pd.DataFrame(new_hyperparameters)], ignore_index=True).to_csv(
        hyperparameter_path, index=False
    )
    status = pd.concat(status_frames, ignore_index=True)
    status.to_csv(output_root / "metrics" / "arima_convergence_status.csv", index=False)
    if not status["converged"].all():
        raise RuntimeError("At least one delivered ARIMA fit did not converge")

    overall, station_metrics, horizon_metrics = make_regression_tables(predictions, FOLDS)
    overall.to_csv(output_root / "metrics" / "regression_metrics_by_fold.csv", index=False)
    station_metrics.to_csv(output_root / "metrics" / "regression_metrics_by_station.csv", index=False)
    horizon_metrics.to_csv(output_root / "metrics" / "regression_metrics_by_horizon.csv", index=False)

    strict_metrics, strict_event_station, strict_labeled = heatwave_metrics_for_predictions(
        predictions, df, FOLDS, inclusive=False
    )
    inclusive_metrics, inclusive_event_station, inclusive_labeled = heatwave_metrics_for_predictions(
        predictions, df, FOLDS, inclusive=True
    )
    heatwave_metrics = pd.concat([strict_metrics, inclusive_metrics], ignore_index=True)
    event_station_metrics = pd.concat([strict_event_station, inclusive_event_station], ignore_index=True)
    labeled_predictions = pd.concat([strict_labeled, inclusive_labeled], ignore_index=True)
    heatwave_metrics.to_csv(output_root / "metrics" / "heatwave_day_and_event_metrics.csv", index=False)
    event_station_metrics.to_csv(output_root / "metrics" / "heatwave_event_metrics_by_station.csv", index=False)
    labeled_predictions.to_csv(output_root / "predictions" / "heatwave_labeled_predictions.csv", index=False)

    validation_summary, uncertainty, selected_model = final_test_uncertainty(predictions, overall)
    validation_summary.to_csv(output_root / "metrics" / "validation_model_summary.csv", index=False)
    uncertainty.to_csv(output_root / "metrics" / "final_test_uncertainty.csv", index=False)
    paired = paired_block_bootstrap(predictions, selected_model)
    paired.to_csv(output_root / "metrics" / "final_test_paired_block_comparisons.csv", index=False)
    create_figures(
        predictions,
        overall,
        heatwave_metrics,
        labeled_predictions,
        uncertainty,
        selected_model,
        output_root / "figures",
    )

    summary = {
        "selected_model_from_validation": selected_model,
        "selection_rule": "Lowest mean validation RMSE across 2022 and 2023; final 2024H1 test is not used for selection.",
        "dataset": audit,
        "final_test_regression": overall[overall["fold"] == "test_2024H1"].sort_values("rmse").to_dict(orient="records"),
        "final_test_heatwave_primary": heatwave_metrics[
            (heatwave_metrics["fold"] == "test_2024H1")
            & (heatwave_metrics["definition"] == "strict_gt_36")
        ].sort_values("event_f1", ascending=False).to_dict(orient="records"),
        "arima_convergence": {
            "all_delivered_fits_converged": bool(status["converged"].all()),
            "station_specific_orders": True,
            "fallback_rule": "Use the next-best converged inner-validation order when the top-ranked order fails on the full outer training fold.",
        },
        "limitations": [
            "Only station-level temperature variables are available; humidity, wind, rainfall, pressure, and forecast-model fields are absent.",
            "The data cover 2019-01-01 through 2024-06-30, which is too short for climate projection or a defensible 2037 forecast.",
            "The BMD definition includes a large-area component that cannot be tested with six station series.",
            "Heatwave positives are imbalanced and station-concentrated, particularly in Rajshahi.",
            "Seven-day recursive forecasts are short-range statistical forecasts, not numerical weather predictions.",
        ],
    }
    (output_root / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"Validation-selected model after converged ARIMA rebuild: {selected_model}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    rebuild(arguments.csv.resolve(), arguments.output.resolve())
