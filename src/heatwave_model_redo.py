#!/usr/bin/env python3
"""Leakage-free heatwave/TMAX forecasting redevelopment pipeline.

The pipeline rebuilds the original experiment around station-level, seven-day
rolling-origin forecasts. It deliberately separates continuous TMAX forecasting
from heatwave day/event evaluation.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import platform
import random
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/heatwave-model-mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

SEED = 42
THRESHOLD_C = 36.0
MIN_CONSECUTIVE_DAYS = 3
FORECAST_BLOCK_DAYS = 7
LAGS = (1, 2, 3, 7, 14, 30, 365)
ROLL_WINDOWS = (3, 7, 14, 30)
STATION_DISPLAY = {
    "Ambagan(ctg)": "Chattogram (Ambagan)",
    "Gopalgonj": "Gopalganj",
}


@dataclass(frozen=True)
class Fold:
    name: str
    role: str
    train_end: str
    test_start: str
    test_end: str


FOLDS = (
    Fold("val_2022", "validation", "2021-12-31", "2022-01-01", "2022-12-31"),
    Fold("val_2023", "validation", "2022-12-31", "2023-01-01", "2023-12-31"),
    Fold("test_2024H1", "final_test", "2023-12-31", "2024-01-01", "2024-06-30"),
)

NUMERIC_FEATURES = [
    *(f"lag_{lag}" for lag in LAGS),
    *(f"roll_mean_{window}" for window in ROLL_WINDOWS),
    "roll_std_7",
    "roll_std_30",
    "doy_sin_1",
    "doy_cos_1",
    "doy_sin_2",
    "doy_cos_2",
    "doy_sin_3",
    "doy_cos_3",
    "trend_years",
]
FEATURE_COLUMNS = ["STATION", *NUMERIC_FEATURES]


def set_global_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dirs(output_root: Path) -> dict[str, Path]:
    dirs = {
        "root": output_root,
        "metrics": output_root / "metrics",
        "predictions": output_root / "predictions",
        "figures": output_root / "figures",
        "audit": output_root / "audit",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return dirs


def load_and_validate_data(csv_path: Path) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    expected_columns = ["DATE", "STATION", "TMAX", "TMIN", "TAVG"]
    if list(df.columns) != expected_columns:
        raise ValueError(f"Unexpected columns: {list(df.columns)}")
    df["DATE"] = pd.to_datetime(df["DATE"], format="%m/%d/%Y", errors="raise")
    for column in ["TMAX", "TMIN", "TAVG"]:
        df[column] = pd.to_numeric(df[column], errors="raise")
    df = df.sort_values(["STATION", "DATE"]).reset_index(drop=True)

    station_rows: list[dict[str, Any]] = []
    for station, station_df in df.groupby("STATION", sort=True):
        expected_dates = pd.date_range(station_df["DATE"].min(), station_df["DATE"].max(), freq="D")
        missing_dates = expected_dates.difference(pd.DatetimeIndex(station_df["DATE"]))
        station_rows.append(
            {
                "station": station,
                "display_station": STATION_DISPLAY.get(station, station),
                "rows": len(station_df),
                "unique_dates": station_df["DATE"].nunique(),
                "start_date": station_df["DATE"].min().date().isoformat(),
                "end_date": station_df["DATE"].max().date().isoformat(),
                "missing_calendar_dates": len(missing_dates),
                "threshold_days_gt_36": int((station_df["TMAX"] > THRESHOLD_C).sum()),
                "threshold_days_ge_36": int((station_df["TMAX"] >= THRESHOLD_C).sum()),
            }
        )
    station_profile = pd.DataFrame(station_rows)

    midpoint_error = (df["TAVG"] - (df["TMAX"] + df["TMIN"]) / 2).abs()
    audit = {
        "source_file": csv_path.name,
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "stations": int(df["STATION"].nunique()),
        "start_date": df["DATE"].min().date().isoformat(),
        "end_date": df["DATE"].max().date().isoformat(),
        "calendar_days_per_station": int(df.groupby("STATION").size().iloc[0]),
        "null_cells": int(df.isna().sum().sum()),
        "exact_duplicate_rows": int(df.duplicated().sum()),
        "duplicate_station_dates": int(df.duplicated(["STATION", "DATE"]).sum()),
        "invalid_tmax_below_tmin": int((df["TMAX"] < df["TMIN"]).sum()),
        "tavg_outside_tmin_tmax": int(((df["TAVG"] < df["TMIN"]) | (df["TAVG"] > df["TMAX"])).sum()),
        "tavg_midpoint_abs_error_mean": float(midpoint_error.mean()),
        "tavg_midpoint_abs_error_max": float(midpoint_error.max()),
        "tmax_min": float(df["TMAX"].min()),
        "tmax_max": float(df["TMAX"].max()),
        "complete_daily_calendar_all_stations": bool((station_profile["missing_calendar_dates"] == 0).all()),
        "analysis_target": "TMAX",
        "primary_heatwave_definition": "TMAX > 36 C for at least 3 consecutive station-days",
        "sensitivity_heatwave_definition": "TMAX >= 36 C for at least 3 consecutive station-days",
        "spatial_coverage_caveat": "Station-level data cannot test BMD's large-area declaration component.",
    }
    return df, audit, station_profile


def calendar_feature_dict(date: pd.Timestamp) -> dict[str, float]:
    day_of_year = date.dayofyear
    result: dict[str, float] = {}
    for harmonic in (1, 2, 3):
        angle = 2 * math.pi * harmonic * day_of_year / 365.25
        result[f"doy_sin_{harmonic}"] = math.sin(angle)
        result[f"doy_cos_{harmonic}"] = math.cos(angle)
    result["trend_years"] = (date - pd.Timestamp("2019-01-01")).days / 365.25
    return result


def build_training_features(df: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for station, station_df in df.groupby("STATION", sort=True):
        station_df = station_df.sort_values("DATE").copy()
        target = station_df["TMAX"]
        features = pd.DataFrame({"DATE": station_df["DATE"], "STATION": station})
        for lag in LAGS:
            features[f"lag_{lag}"] = target.shift(lag)
        shifted = target.shift(1)
        for window in ROLL_WINDOWS:
            features[f"roll_mean_{window}"] = shifted.rolling(window, min_periods=window).mean()
        features["roll_std_7"] = shifted.rolling(7, min_periods=7).std(ddof=0)
        features["roll_std_30"] = shifted.rolling(30, min_periods=30).std(ddof=0)
        for harmonic in (1, 2, 3):
            angle = 2 * np.pi * harmonic * station_df["DATE"].dt.dayofyear / 365.25
            features[f"doy_sin_{harmonic}"] = np.sin(angle)
            features[f"doy_cos_{harmonic}"] = np.cos(angle)
        features["trend_years"] = (station_df["DATE"] - pd.Timestamp("2019-01-01")).dt.days / 365.25
        features["y"] = target.to_numpy()
        frames.append(features)
    feature_frame = pd.concat(frames, ignore_index=True)
    return feature_frame.dropna(subset=NUMERIC_FEATURES).reset_index(drop=True)


def feature_row_from_history(station: str, date: pd.Timestamp, history: list[float]) -> pd.DataFrame:
    if len(history) < max(LAGS):
        raise ValueError(f"At least {max(LAGS)} historical days are required")
    row: dict[str, Any] = {"STATION": station}
    for lag in LAGS:
        row[f"lag_{lag}"] = float(history[-lag])
    for window in ROLL_WINDOWS:
        row[f"roll_mean_{window}"] = float(np.mean(history[-window:]))
    row["roll_std_7"] = float(np.std(history[-7:], ddof=0))
    row["roll_std_30"] = float(np.std(history[-30:], ddof=0))
    row.update(calendar_feature_dict(date))
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def make_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    numeric_transformer: Any = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer(
        transformers=[
            ("station", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["STATION"]),
            ("numeric", numeric_transformer, NUMERIC_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_ridge(alpha: float) -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", make_preprocessor(scale_numeric=True)),
            ("model", Ridge(alpha=alpha)),
        ]
    )


def make_decision_tree(max_depth: int | None, min_samples_leaf: int) -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", make_preprocessor(scale_numeric=False)),
            (
                "model",
                DecisionTreeRegressor(
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    random_state=SEED,
                ),
            ),
        ]
    )


def make_random_forest(max_depth: int | None, min_samples_leaf: int, max_features: float) -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", make_preprocessor(scale_numeric=False)),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=150,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    random_state=SEED,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def fit_panel_model(model: Pipeline, feature_frame: pd.DataFrame, train_end: pd.Timestamp) -> Pipeline:
    train_rows = feature_frame[feature_frame["DATE"] <= train_end]
    model.fit(train_rows[FEATURE_COLUMNS], train_rows["y"])
    return model


def weekly_blocks(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.DatetimeIndex]:
    dates = pd.date_range(start, end, freq="D")
    return [dates[index : index + FORECAST_BLOCK_DAYS] for index in range(0, len(dates), FORECAST_BLOCK_DAYS)]


def recursive_panel_predictions(
    model: Pipeline,
    df: pd.DataFrame,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    model_name: str,
    fold_name: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    stations = sorted(df["STATION"].unique())
    indexed = {
        station: station_df.set_index("DATE")["TMAX"].sort_index()
        for station, station_df in df.groupby("STATION", sort=True)
    }
    for block in weekly_blocks(test_start, test_end):
        origin = block[0] - pd.Timedelta(days=1)
        histories = {
            station: indexed[station].loc[:origin].astype(float).tolist()
            for station in stations
        }
        for horizon, date in enumerate(block, start=1):
            feature_rows = pd.concat(
                [feature_row_from_history(station, date, histories[station]) for station in stations],
                ignore_index=True,
            )
            block_predictions = model.predict(feature_rows)
            for station, prediction in zip(stations, block_predictions):
                prediction = float(prediction)
                histories[station].append(prediction)
                rows.append(
                    prediction_row(
                        fold_name,
                        model_name,
                        station,
                        origin,
                        date,
                        horizon,
                        float(indexed[station].loc[date]),
                        prediction,
                    )
                )
    return pd.DataFrame(rows)


def prediction_row(
    fold: str,
    model: str,
    station: str,
    origin_date: pd.Timestamp,
    target_date: pd.Timestamp,
    horizon: int,
    y_true: float,
    y_pred: float,
) -> dict[str, Any]:
    return {
        "fold": fold,
        "model": model,
        "station": station,
        "display_station": STATION_DISPLAY.get(station, station),
        "origin_date": origin_date,
        "date": target_date,
        "horizon_day": horizon,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def baseline_predictions(
    df: pd.DataFrame,
    train_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    fold_name: str,
) -> list[pd.DataFrame]:
    stations = sorted(df["STATION"].unique())
    indexed = {
        station: station_df.set_index("DATE")["TMAX"].sort_index()
        for station, station_df in df.groupby("STATION", sort=True)
    }

    # Circular 15-day station/day-of-year climatology using training data only.
    climatology: dict[tuple[str, int], float] = {}
    training = df[df["DATE"] <= train_end].copy()
    training["doy"] = training["DATE"].dt.dayofyear
    for station, station_df in training.groupby("STATION", sort=True):
        for doy in range(1, 367):
            circular_distance = np.minimum(
                np.abs(station_df["doy"].to_numpy() - doy),
                366 - np.abs(station_df["doy"].to_numpy() - doy),
            )
            values = station_df.loc[circular_distance <= 7, "TMAX"]
            climatology[(station, doy)] = float(values.mean())

    outputs: dict[str, list[dict[str, Any]]] = {
        "Persistence": [],
        "Seasonal_Naive": [],
        "Climatology": [],
    }
    for block in weekly_blocks(test_start, test_end):
        origin = block[0] - pd.Timedelta(days=1)
        for station in stations:
            actual_series = indexed[station]
            last_observed = float(actual_series.loc[origin])
            for horizon, date in enumerate(block, start=1):
                actual = float(actual_series.loc[date])
                seasonal_date = date - pd.Timedelta(days=365)
                seasonal_value = float(actual_series.loc[seasonal_date])
                values = {
                    "Persistence": last_observed,
                    "Seasonal_Naive": seasonal_value,
                    "Climatology": climatology[(station, date.dayofyear)],
                }
                for model_name, prediction in values.items():
                    outputs[model_name].append(
                        prediction_row(
                            fold_name,
                            model_name,
                            station,
                            origin,
                            date,
                            horizon,
                            actual,
                            prediction,
                        )
                    )
    return [pd.DataFrame(rows) for rows in outputs.values()]


def tune_panel_model(
    feature_frame: pd.DataFrame,
    df: pd.DataFrame,
    outer_train_end: pd.Timestamp,
    candidates: list[tuple[dict[str, Any], Callable[[], Pipeline]]],
    model_name: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    inner_start = outer_train_end - pd.Timedelta(days=181)
    inner_base_end = inner_start - pd.Timedelta(days=1)
    tuning_rows: list[dict[str, Any]] = []
    for parameters, factory in candidates:
        candidate = fit_panel_model(factory(), feature_frame, inner_base_end)
        predictions = recursive_panel_predictions(
            candidate,
            df,
            inner_start,
            outer_train_end,
            model_name,
            "inner_tuning",
        )
        score = float(math.sqrt(mean_squared_error(predictions["y_true"], predictions["y_pred"])))
        tuning_rows.append({"model": model_name, "parameters": json.dumps(parameters, sort_keys=True), "inner_rmse": score})
    tuning_df = pd.DataFrame(tuning_rows).sort_values(["inner_rmse", "parameters"]).reset_index(drop=True)
    best_parameters = json.loads(tuning_df.iloc[0]["parameters"])
    return best_parameters, tuning_df


def exogenous_fourier(dates: Iterable[pd.Timestamp]) -> pd.DataFrame:
    dates = pd.DatetimeIndex(dates)
    data: dict[str, np.ndarray] = {}
    for harmonic in (1, 2, 3):
        angle = 2 * np.pi * harmonic * dates.dayofyear.to_numpy() / 365.25
        data[f"sin_{harmonic}"] = np.sin(angle)
        data[f"cos_{harmonic}"] = np.cos(angle)
    data["trend_years"] = ((dates - pd.Timestamp("2019-01-01")).days.to_numpy() / 365.25)
    return pd.DataFrame(data, index=dates)


def fit_sarimax(series: pd.Series, order: tuple[int, int, int]):
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    series = series.asfreq("D")
    exog = exogenous_fourier(series.index)
    model = SARIMAX(
        series.astype(float),
        exog=exog,
        order=order,
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False,
        simple_differencing=False,
    )
    return model.fit(disp=False, maxiter=300)


def sarimax_predictions_from_result(
    result: Any,
    actual_series: pd.Series,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    station: str,
    model_name: str,
    fold_name: str,
) -> tuple[pd.DataFrame, Any]:
    rows: list[dict[str, Any]] = []
    current = result
    for block in weekly_blocks(test_start, test_end):
        origin = block[0] - pd.Timedelta(days=1)
        future_exog = exogenous_fourier(block)
        forecast = np.asarray(current.get_forecast(steps=len(block), exog=future_exog).predicted_mean)
        for horizon, (date, prediction) in enumerate(zip(block, forecast), start=1):
            rows.append(
                prediction_row(
                    fold_name,
                    model_name,
                    station,
                    origin,
                    date,
                    horizon,
                    float(actual_series.loc[date]),
                    float(prediction),
                )
            )
        block_actual = actual_series.loc[block]
        current = current.append(block_actual, exog=future_exog, refit=False)
    return pd.DataFrame(rows), current


def tune_arima_orders_by_station(
    df: pd.DataFrame,
    outer_train_end: pd.Timestamp,
) -> tuple[dict[str, list[tuple[int, int, int]]], pd.DataFrame]:
    """Rank converged ARIMA candidates separately for each station."""
    candidates = [(1, 0, 0), (2, 0, 0), (3, 0, 0), (1, 0, 1), (2, 0, 1)]
    inner_start = outer_train_end - pd.Timedelta(days=181)
    inner_base_end = inner_start - pd.Timedelta(days=1)
    rows: list[dict[str, Any]] = []
    rankings: dict[str, list[tuple[int, int, int]]] = {}
    for station, station_df in df.groupby("STATION", sort=True):
        series = station_df.set_index("DATE")["TMAX"].sort_index()
        station_rows: list[dict[str, Any]] = []
        for order in candidates:
            converged = False
            score = float("inf")
            iterations: int | None = None
            try:
                result = fit_sarimax(series.loc[:inner_base_end], order)
                converged = bool((getattr(result, "mle_retvals", {}) or {}).get("converged", False))
                iterations = (getattr(result, "mle_retvals", {}) or {}).get("iterations")
                if converged:
                    predictions, _ = sarimax_predictions_from_result(
                        result,
                        series,
                        inner_start,
                        outer_train_end,
                        station,
                        "Dynamic_ARIMA",
                        "inner_tuning",
                    )
                    score = float(math.sqrt(mean_squared_error(predictions["y_true"], predictions["y_pred"])))
            except Exception:
                converged = False
            station_rows.append(
                {
                    "model": "Dynamic_ARIMA",
                    "station": station,
                    "parameters": json.dumps({"order": order}),
                    "inner_rmse": score,
                    "converged": converged,
                    "iterations": iterations,
                }
            )
        finite_rows = [row for row in station_rows if np.isfinite(row["inner_rmse"])]
        if not finite_rows:
            raise RuntimeError(f"No converged ARIMA candidate for {station} through {outer_train_end.date()}")
        finite_rows.sort(key=lambda row: (row["inner_rmse"], row["parameters"]))
        rankings[station] = [tuple(json.loads(row["parameters"])["order"]) for row in finite_rows]
        rows.extend(station_rows)
    tuning = pd.DataFrame(rows).sort_values(["station", "inner_rmse", "parameters"]).reset_index(drop=True)
    return rankings, tuning


def arima_outer_predictions(
    df: pd.DataFrame,
    train_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    fold_name: str,
    order_rankings: dict[str, list[tuple[int, int, int]]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    station_predictions: list[pd.DataFrame] = []
    status_rows: list[dict[str, Any]] = []
    for station, station_df in df.groupby("STATION", sort=True):
        series = station_df.set_index("DATE")["TMAX"].sort_index()
        result = None
        selected_order: tuple[int, int, int] | None = None
        fallback_rank: int | None = None
        for rank, order in enumerate(order_rankings[station], start=1):
            candidate_result = fit_sarimax(series.loc[:train_end], order)
            if bool((getattr(candidate_result, "mle_retvals", {}) or {}).get("converged", False)):
                result = candidate_result
                selected_order = order
                fallback_rank = rank
                break
        if result is None or selected_order is None or fallback_rank is None:
            raise RuntimeError(f"No ranked ARIMA candidate converged for {station} in {fold_name}")
        predictions, _ = sarimax_predictions_from_result(
            result,
            series,
            test_start,
            test_end,
            station,
            "Dynamic_ARIMA",
            fold_name,
        )
        station_predictions.append(predictions)
        mle = getattr(result, "mle_retvals", {}) or {}
        status_rows.append(
            {
                "fold": fold_name,
                "station": station,
                "order": str(selected_order),
                "converged": bool(mle.get("converged", False)),
                "iterations": mle.get("iterations"),
                "fallback_rank": fallback_rank,
                "aic": float(result.aic),
                "bic": float(result.bic),
            }
        )
    return pd.concat(station_predictions, ignore_index=True), pd.DataFrame(status_rows)


def fit_prophet(train_df: pd.DataFrame, parameters: dict[str, float]):
    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
    logging.getLogger("prophet").setLevel(logging.ERROR)
    from prophet import Prophet

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="additive",
        n_changepoints=15,
        changepoint_prior_scale=float(parameters["changepoint_prior_scale"]),
        seasonality_prior_scale=float(parameters["seasonality_prior_scale"]),
        uncertainty_samples=0,
    )
    model.fit(train_df.rename(columns={"DATE": "ds", "TMAX": "y"})[["ds", "y"]])
    return model


def tune_prophet(df: pd.DataFrame, outer_train_end: pd.Timestamp) -> tuple[dict[str, float], pd.DataFrame]:
    candidates = [
        {"changepoint_prior_scale": changepoint, "seasonality_prior_scale": seasonality}
        for changepoint in (0.01, 0.05, 0.2)
        for seasonality in (1.0, 10.0)
    ]
    inner_start = outer_train_end - pd.Timedelta(days=181)
    inner_base_end = inner_start - pd.Timedelta(days=1)
    rows: list[dict[str, Any]] = []
    for parameters in candidates:
        all_true: list[float] = []
        all_pred: list[float] = []
        for _, station_df in df.groupby("STATION", sort=True):
            train = station_df[station_df["DATE"] <= inner_base_end]
            test = station_df[(station_df["DATE"] >= inner_start) & (station_df["DATE"] <= outer_train_end)]
            model = fit_prophet(train, parameters)
            forecast = model.predict(test[["DATE"]].rename(columns={"DATE": "ds"}))
            all_true.extend(test["TMAX"].astype(float).tolist())
            all_pred.extend(forecast["yhat"].astype(float).tolist())
        score = float(math.sqrt(mean_squared_error(all_true, all_pred)))
        rows.append({"model": "Prophet", "parameters": json.dumps(parameters, sort_keys=True), "inner_rmse": score})
    tuning = pd.DataFrame(rows).sort_values(["inner_rmse", "parameters"]).reset_index(drop=True)
    selected = json.loads(tuning.iloc[0]["parameters"])
    return selected, tuning


def prophet_outer_predictions(
    df: pd.DataFrame,
    train_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    fold_name: str,
    parameters: dict[str, float],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for station, station_df in df.groupby("STATION", sort=True):
        train = station_df[station_df["DATE"] <= train_end]
        test = station_df[(station_df["DATE"] >= test_start) & (station_df["DATE"] <= test_end)].copy()
        model = fit_prophet(train, parameters)
        forecast = model.predict(test[["DATE"]].rename(columns={"DATE": "ds"}))
        prediction_by_date = dict(zip(test["DATE"], forecast["yhat"].astype(float)))
        actual_by_date = dict(zip(test["DATE"], test["TMAX"].astype(float)))
        for block in weekly_blocks(test_start, test_end):
            origin = block[0] - pd.Timedelta(days=1)
            for horizon, date in enumerate(block, start=1):
                rows.append(
                    prediction_row(
                        fold_name,
                        "Prophet",
                        station,
                        origin,
                        date,
                        horizon,
                        actual_by_date[date],
                        prediction_by_date[date],
                    )
                )
    return pd.DataFrame(rows)


def import_tensorflow():
    import tensorflow as tf

    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    return tf


def make_lstm(tf: Any, lookback: int = 30, units: int = 16):
    tf.keras.utils.set_random_seed(SEED)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(lookback, 1)),
            tf.keras.layers.LSTM(units),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    return model


def sequence_arrays(values: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_values: list[np.ndarray] = []
    y_values: list[float] = []
    target_indices: list[int] = []
    for target_index in range(lookback, len(values)):
        x_values.append(values[target_index - lookback : target_index])
        y_values.append(float(values[target_index]))
        target_indices.append(target_index)
    x = np.asarray(x_values, dtype=np.float32).reshape(-1, lookback, 1)
    y = np.asarray(y_values, dtype=np.float32)
    return x, y, np.asarray(target_indices)


def train_station_lstm(series: pd.Series, outer_train_end: pd.Timestamp) -> tuple[Any, float, float, int]:
    tf = import_tensorflow()
    tf.keras.backend.clear_session()
    lookback = 30
    validation_start = outer_train_end - pd.Timedelta(days=181)
    outer_train = series.loc[:outer_train_end].astype(float)
    base_train = outer_train.loc[: validation_start - pd.Timedelta(days=1)]
    base_mean = float(base_train.mean())
    base_std = float(base_train.std(ddof=0)) or 1.0
    scaled_for_tuning = ((outer_train - base_mean) / base_std).to_numpy()
    x_all, y_all, target_indices = sequence_arrays(scaled_for_tuning, lookback)
    target_dates = outer_train.index[target_indices]
    train_mask = target_dates < validation_start
    validation_mask = target_dates >= validation_start

    tuning_model = make_lstm(tf, lookback=lookback, units=16)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True, min_delta=1e-4
    )
    history = tuning_model.fit(
        x_all[train_mask],
        y_all[train_mask],
        validation_data=(x_all[validation_mask], y_all[validation_mask]),
        epochs=50,
        batch_size=32,
        shuffle=False,
        verbose=0,
        callbacks=[early_stopping],
    )
    best_epoch = int(np.argmin(history.history["val_loss"]) + 1)

    final_mean = float(outer_train.mean())
    final_std = float(outer_train.std(ddof=0)) or 1.0
    final_scaled = ((outer_train - final_mean) / final_std).to_numpy()
    x_final, y_final, _ = sequence_arrays(final_scaled, lookback)
    tf.keras.backend.clear_session()
    final_model = make_lstm(tf, lookback=lookback, units=16)
    final_model.fit(
        x_final,
        y_final,
        epochs=best_epoch,
        batch_size=32,
        shuffle=False,
        verbose=0,
    )
    return final_model, final_mean, final_std, best_epoch


def lstm_outer_predictions(
    df: pd.DataFrame,
    train_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    fold_name: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    configs: list[dict[str, Any]] = []
    for station, station_df in df.groupby("STATION", sort=True):
        series = station_df.set_index("DATE")["TMAX"].sort_index()
        model, mean, std, epochs = train_station_lstm(series, train_end)
        configs.append(
            {
                "fold": fold_name,
                "model": "LSTM",
                "station": station,
                "parameters": json.dumps(
                    {
                        "lookback_days": 30,
                        "units": 16,
                        "batch_size": 32,
                        "learning_rate": 0.001,
                        "selected_epochs": epochs,
                    },
                    sort_keys=True,
                ),
            }
        )
        for block in weekly_blocks(test_start, test_end):
            origin = block[0] - pd.Timedelta(days=1)
            history = ((series.loc[:origin].astype(float) - mean) / std).tolist()
            for horizon, date in enumerate(block, start=1):
                x_input = np.asarray(history[-30:], dtype=np.float32).reshape(1, 30, 1)
                prediction_scaled = float(model.predict(x_input, verbose=0).reshape(-1)[0])
                prediction = prediction_scaled * std + mean
                history.append(prediction_scaled)
                rows.append(
                    prediction_row(
                        fold_name,
                        "LSTM",
                        station,
                        origin,
                        date,
                        horizon,
                        float(series.loc[date]),
                        float(prediction),
                    )
                )
    return pd.DataFrame(rows), configs


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    errors = y_true - y_pred
    denominator = np.maximum(np.abs(y_true), 1e-8)
    smape_denominator = np.maximum(np.abs(y_true) + np.abs(y_pred), 1e-8)
    return {
        "n": int(len(y_true)),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mape_pct": float(np.mean(np.abs(errors) / denominator) * 100),
        "smape_pct": float(np.mean(2 * np.abs(errors) / smape_denominator) * 100),
        "r2": float(r2_score(y_true, y_pred)),
        "bias": float(np.mean(y_pred - y_true)),
    }


def make_regression_tables(predictions: pd.DataFrame, folds: Iterable[Fold]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fold_role = {fold.name: fold.role for fold in folds}
    overall_rows: list[dict[str, Any]] = []
    station_rows: list[dict[str, Any]] = []
    horizon_rows: list[dict[str, Any]] = []
    for (fold, model), group in predictions.groupby(["fold", "model"], sort=True):
        overall_rows.append({"fold": fold, "role": fold_role[fold], "model": model, **regression_metrics(group.y_true.to_numpy(), group.y_pred.to_numpy())})
    for (fold, model, station), group in predictions.groupby(["fold", "model", "station"], sort=True):
        station_rows.append({"fold": fold, "role": fold_role[fold], "model": model, "station": station, **regression_metrics(group.y_true.to_numpy(), group.y_pred.to_numpy())})
    for (fold, model, horizon), group in predictions.groupby(["fold", "model", "horizon_day"], sort=True):
        horizon_rows.append({"fold": fold, "role": fold_role[fold], "model": model, "horizon_day": horizon, **regression_metrics(group.y_true.to_numpy(), group.y_pred.to_numpy())})
    return pd.DataFrame(overall_rows), pd.DataFrame(station_rows), pd.DataFrame(horizon_rows)


def label_heatwave_sequence(
    dates: pd.Series,
    temperatures: pd.Series,
    inclusive: bool,
) -> pd.DataFrame:
    sequence = pd.DataFrame({"date": pd.to_datetime(dates), "temperature": temperatures.astype(float)}).sort_values("date").reset_index(drop=True)
    hot = sequence["temperature"] >= THRESHOLD_C if inclusive else sequence["temperature"] > THRESHOLD_C
    new_run = hot.ne(hot.shift(fill_value=False)) | sequence["date"].diff().ne(pd.Timedelta(days=1))
    run_id = new_run.cumsum()
    run_length = hot.groupby(run_id).transform("sum")
    sequence["heatwave_day"] = hot & (run_length >= MIN_CONSECUTIVE_DAYS)
    sequence["event_id"] = 0
    event_counter = 0
    previous_heatwave = False
    previous_date: pd.Timestamp | None = None
    for index, row in sequence.iterrows():
        current = bool(row["heatwave_day"])
        if current and (not previous_heatwave or previous_date is None or row["date"] - previous_date != pd.Timedelta(days=1)):
            event_counter += 1
        if current:
            sequence.at[index, "event_id"] = event_counter
        previous_heatwave = current
        previous_date = row["date"]
    return sequence


def event_table(labels: pd.DataFrame, station: str, evaluation_start: pd.Timestamp, evaluation_end: pd.Timestamp) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for event_id, group in labels[labels["event_id"] > 0].groupby("event_id"):
        if group["date"].max() < evaluation_start or group["date"].min() > evaluation_end:
            continue
        rows.append(
            {
                "station": station,
                "event_id": int(event_id),
                "start_date": group["date"].min(),
                "end_date": group["date"].max(),
                "duration_days": int(len(group)),
                "maximum_temperature": float(group["temperature"].max()),
            }
        )
    return pd.DataFrame(rows)


def match_events(observed: pd.DataFrame, predicted: pd.DataFrame) -> dict[str, Any]:
    candidates: list[tuple[int, int, int]] = []
    for observed_index, observed_event in observed.reset_index(drop=True).iterrows():
        for predicted_index, predicted_event in predicted.reset_index(drop=True).iterrows():
            overlap_start = max(observed_event["start_date"], predicted_event["start_date"])
            overlap_end = min(observed_event["end_date"], predicted_event["end_date"])
            overlap_days = (overlap_end - overlap_start).days + 1
            if overlap_days > 0:
                candidates.append((overlap_days, observed_index, predicted_index))
    candidates.sort(reverse=True)
    matched_observed: set[int] = set()
    matched_predicted: set[int] = set()
    onset_errors: list[int] = []
    for _, observed_index, predicted_index in candidates:
        if observed_index in matched_observed or predicted_index in matched_predicted:
            continue
        matched_observed.add(observed_index)
        matched_predicted.add(predicted_index)
        onset_error = (
            predicted.reset_index(drop=True).iloc[predicted_index]["start_date"]
            - observed.reset_index(drop=True).iloc[observed_index]["start_date"]
        ).days
        onset_errors.append(int(onset_error))
    true_positive = len(matched_observed)
    false_positive = len(predicted) - true_positive
    false_negative = len(observed) - true_positive
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    event_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "observed_events": int(len(observed)),
        "predicted_events": int(len(predicted)),
        "matched_events": int(true_positive),
        "missed_events": int(false_negative),
        "false_alarm_events": int(false_positive),
        "event_precision": float(precision),
        "event_recall": float(recall),
        "event_f1": float(event_f1),
        "mean_onset_error_days": float(np.mean(onset_errors)) if onset_errors else np.nan,
        "mean_absolute_onset_error_days": float(np.mean(np.abs(onset_errors))) if onset_errors else np.nan,
    }


def heatwave_metrics_for_predictions(
    predictions: pd.DataFrame,
    df: pd.DataFrame,
    folds: Iterable[Fold],
    inclusive: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    definition_name = "inclusive_ge_36" if inclusive else "strict_gt_36"
    fold_lookup = {fold.name: fold for fold in folds}
    metric_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    labeled_prediction_frames: list[pd.DataFrame] = []
    for (fold_name, model), model_predictions in predictions.groupby(["fold", "model"], sort=True):
        fold = fold_lookup[fold_name]
        test_start = pd.Timestamp(fold.test_start)
        test_end = pd.Timestamp(fold.test_end)
        metric_end = test_end - pd.Timedelta(days=2)
        observed_events_all: list[pd.DataFrame] = []
        predicted_events_all: list[pd.DataFrame] = []
        station_daily: list[pd.DataFrame] = []
        for station, station_predictions in model_predictions.groupby("station", sort=True):
            station_predictions = station_predictions.sort_values("date")
            prehistory = (
                df[(df["STATION"] == station) & (df["DATE"] < test_start)]
                .sort_values("DATE")
                .tail(2)[["DATE", "TMAX"]]
            )
            actual_sequence = pd.concat(
                [
                    prehistory.rename(columns={"DATE": "date", "TMAX": "temperature"}),
                    station_predictions[["date", "y_true"]].rename(columns={"y_true": "temperature"}),
                ],
                ignore_index=True,
            )
            predicted_sequence = pd.concat(
                [
                    prehistory.rename(columns={"DATE": "date", "TMAX": "temperature"}),
                    station_predictions[["date", "y_pred"]].rename(columns={"y_pred": "temperature"}),
                ],
                ignore_index=True,
            )
            actual_labels = label_heatwave_sequence(actual_sequence["date"], actual_sequence["temperature"], inclusive)
            predicted_labels = label_heatwave_sequence(predicted_sequence["date"], predicted_sequence["temperature"], inclusive)
            actual_events = event_table(actual_labels, station, test_start, metric_end)
            predicted_events = event_table(predicted_labels, station, test_start, metric_end)
            observed_events_all.append(actual_events)
            predicted_events_all.append(predicted_events)

            actual_by_date = actual_labels.set_index("date")["heatwave_day"]
            predicted_by_date = predicted_labels.set_index("date")["heatwave_day"]
            daily = station_predictions.copy()
            daily["actual_heatwave_day"] = daily["date"].map(actual_by_date).fillna(False).astype(bool)
            daily["predicted_heatwave_day"] = daily["date"].map(predicted_by_date).fillna(False).astype(bool)
            daily = daily[daily["date"] <= metric_end].copy()
            station_daily.append(daily)

        daily_all = pd.concat(station_daily, ignore_index=True)
        observed_all = pd.concat(observed_events_all, ignore_index=True) if any(len(frame) for frame in observed_events_all) else pd.DataFrame(columns=["station", "event_id", "start_date", "end_date", "duration_days", "maximum_temperature"])
        predicted_all = pd.concat(predicted_events_all, ignore_index=True) if any(len(frame) for frame in predicted_events_all) else pd.DataFrame(columns=["station", "event_id", "start_date", "end_date", "duration_days", "maximum_temperature"])

        y_true = daily_all["actual_heatwave_day"].astype(int).to_numpy()
        y_pred = daily_all["predicted_heatwave_day"].astype(int).to_numpy()
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if tn + fp else np.nan
        event_metric_parts: list[dict[str, Any]] = []
        for station in sorted(df["STATION"].unique()):
            station_observed = observed_all[observed_all["station"] == station]
            station_predicted = predicted_all[predicted_all["station"] == station]
            station_match = match_events(station_observed, station_predicted)
            event_rows.append({"fold": fold_name, "model": model, "definition": definition_name, "station": station, **station_match})
            event_metric_parts.append(station_match)
        event_totals = {
            "observed_events": sum(item["observed_events"] for item in event_metric_parts),
            "predicted_events": sum(item["predicted_events"] for item in event_metric_parts),
            "matched_events": sum(item["matched_events"] for item in event_metric_parts),
            "missed_events": sum(item["missed_events"] for item in event_metric_parts),
            "false_alarm_events": sum(item["false_alarm_events"] for item in event_metric_parts),
        }
        event_precision = event_totals["matched_events"] / event_totals["predicted_events"] if event_totals["predicted_events"] else 0.0
        event_recall = event_totals["matched_events"] / event_totals["observed_events"] if event_totals["observed_events"] else 0.0
        event_f1 = 2 * event_precision * event_recall / (event_precision + event_recall) if event_precision + event_recall else 0.0
        onset_values = [item["mean_absolute_onset_error_days"] for item in event_metric_parts if not np.isnan(item["mean_absolute_onset_error_days"])]

        metric_rows.append(
            {
                "fold": fold_name,
                "role": fold.role,
                "model": model,
                "definition": definition_name,
                "n_station_days": int(len(y_true)),
                "positive_station_days": int(y_true.sum()),
                "predicted_positive_station_days": int(y_pred.sum()),
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "specificity": float(specificity),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
                "roc_auc": float(roc_auc_score(y_true, daily_all["y_pred"])) if len(np.unique(y_true)) == 2 else np.nan,
                "pr_auc": float(average_precision_score(y_true, daily_all["y_pred"])) if y_true.sum() else np.nan,
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                **event_totals,
                "event_precision": float(event_precision),
                "event_recall": float(event_recall),
                "event_f1": float(event_f1),
                "mean_absolute_onset_error_days": float(np.mean(onset_values)) if onset_values else np.nan,
            }
        )
        daily_all["definition"] = definition_name
        labeled_prediction_frames.append(daily_all)
    return pd.DataFrame(metric_rows), pd.DataFrame(event_rows), pd.concat(labeled_prediction_frames, ignore_index=True)


def bootstrap_interval(values: np.ndarray, statistic: Callable[[np.ndarray], float], seed: int = SEED, n_bootstrap: int = 2000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_bootstrap)
    for index in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        estimates[index] = statistic(sample)
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def block_bootstrap_error_interval(
    prediction_group: pd.DataFrame,
    metric: str,
    seed: int,
    n_bootstrap: int = 2000,
) -> tuple[float, float]:
    """Bootstrap complete station-week forecast blocks to preserve dependence."""
    working = prediction_group.copy()
    working["block_id"] = working["station"] + "|" + working["origin_date"].dt.strftime("%Y-%m-%d")
    errors_by_block = {
        block_id: (group["y_true"] - group["y_pred"]).to_numpy(dtype=float)
        for block_id, group in working.groupby("block_id", sort=True)
    }
    block_ids = np.asarray(list(errors_by_block), dtype=object)
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_bootstrap)
    for index in range(n_bootstrap):
        sampled_ids = rng.choice(block_ids, size=len(block_ids), replace=True)
        sampled_errors = np.concatenate([errors_by_block[block_id] for block_id in sampled_ids])
        if metric == "mae":
            estimates[index] = np.mean(np.abs(sampled_errors))
        elif metric == "rmse":
            estimates[index] = np.sqrt(np.mean(sampled_errors**2))
        else:
            raise ValueError(f"Unsupported metric: {metric}")
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def final_test_uncertainty(predictions: pd.DataFrame, overall_metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    validation_metrics = overall_metrics[overall_metrics["role"] == "validation"]
    validation_summary = (
        validation_metrics.groupby("model")[["mae", "rmse", "mape_pct", "smape_pct", "r2"]]
        .agg(["mean", "std"])
    )
    validation_summary.columns = [f"{metric}_{stat}" for metric, stat in validation_summary.columns]
    validation_summary = validation_summary.reset_index().sort_values(["rmse_mean", "mae_mean"])
    selected_model = str(validation_summary.iloc[0]["model"])

    validation_predictions = predictions[predictions["fold"].isin(["val_2022", "val_2023"])].copy()
    final_predictions = predictions[predictions["fold"] == "test_2024H1"].copy()
    uncertainty_rows: list[dict[str, Any]] = []
    for model, final_group in final_predictions.groupby("model", sort=True):
        validation_errors = np.abs(
            validation_predictions.loc[validation_predictions["model"] == model, "y_true"]
            - validation_predictions.loc[validation_predictions["model"] == model, "y_pred"]
        )
        q90 = float(np.quantile(validation_errors, 0.90, method="higher"))
        errors = (final_group["y_true"] - final_group["y_pred"]).to_numpy()
        absolute_errors = np.abs(errors)
        squared_errors = errors**2
        mae_ci = block_bootstrap_error_interval(final_group, "mae", seed=SEED)
        rmse_ci = block_bootstrap_error_interval(final_group, "rmse", seed=SEED + 1)
        lower = final_group["y_pred"].to_numpy() - q90
        upper = final_group["y_pred"].to_numpy() + q90
        coverage = np.mean((final_group["y_true"].to_numpy() >= lower) & (final_group["y_true"].to_numpy() <= upper))
        uncertainty_rows.append(
            {
                "model": model,
                "mae": float(np.mean(absolute_errors)),
                "mae_ci95_lower": mae_ci[0],
                "mae_ci95_upper": mae_ci[1],
                "rmse": float(np.sqrt(np.mean(squared_errors))),
                "rmse_ci95_lower": rmse_ci[0],
                "rmse_ci95_upper": rmse_ci[1],
                "conformal_nominal_coverage": 0.90,
                "conformal_half_width_c": q90,
                "final_test_coverage": float(coverage),
                "mean_interval_width_c": float(2 * q90),
            }
        )
    return validation_summary, pd.DataFrame(uncertainty_rows), selected_model


def paired_block_bootstrap(predictions: pd.DataFrame, selected_model: str, n_bootstrap: int = 2000) -> pd.DataFrame:
    final_predictions = predictions[predictions["fold"] == "test_2024H1"].copy()
    final_predictions["block_id"] = final_predictions["station"] + "|" + final_predictions["origin_date"].dt.strftime("%Y-%m-%d")
    block_mae = (
        final_predictions.assign(abs_error=lambda frame: (frame["y_true"] - frame["y_pred"]).abs())
        .groupby(["block_id", "model"])["abs_error"]
        .mean()
        .unstack("model")
        .dropna()
    )
    rng = np.random.default_rng(SEED)
    selected = block_mae[selected_model].to_numpy()
    rows: list[dict[str, Any]] = []
    for model in block_mae.columns:
        comparator = block_mae[model].to_numpy()
        differences = comparator - selected
        bootstrap_estimates = np.empty(n_bootstrap)
        for index in range(n_bootstrap):
            sample_indices = rng.integers(0, len(differences), len(differences))
            bootstrap_estimates[index] = differences[sample_indices].mean()
        lower, upper = np.quantile(bootstrap_estimates, [0.025, 0.975])
        rows.append(
            {
                "selected_model": selected_model,
                "comparator_model": model,
                "paired_blocks": int(len(differences)),
                "mean_mae_difference_comparator_minus_selected": float(differences.mean()),
                "ci95_lower": float(lower),
                "ci95_upper": float(upper),
                "difference_supported_at_95pct": bool(lower > 0 or upper < 0),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_mae_difference_comparator_minus_selected")


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def create_figures(
    predictions: pd.DataFrame,
    overall_metrics: pd.DataFrame,
    heatwave_metrics: pd.DataFrame,
    labeled_predictions: pd.DataFrame,
    uncertainty: pd.DataFrame,
    selected_model: str,
    figures_dir: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "axes.edgecolor": "#334155",
            "axes.labelcolor": "#172033",
            "xtick.color": "#334155",
            "ytick.color": "#334155",
        }
    )
    colors = {"actual": "#172033", "predicted": "#2563EB", "interval": "#BFDBFE", "accent": "#D97706"}

    final_metrics = overall_metrics[overall_metrics["fold"] == "test_2024H1"].sort_values("rmse")
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    positions = np.arange(len(final_metrics))
    width = 0.38
    ax.bar(positions - width / 2, final_metrics["mae"], width, label="MAE", color="#2563EB", edgecolor="#1E3A8A")
    ax.bar(positions + width / 2, final_metrics["rmse"], width, label="RMSE", color="#D97706", edgecolor="#92400E")
    ax.set_xticks(positions, final_metrics["model"], rotation=35, ha="right")
    ax.set_ylabel("Error (°C)")
    fig.suptitle("Final-test TMAX forecast errors by model", y=0.98, fontsize=13)
    fig.text(
        0.5,
        0.925,
        "1 January–30 June 2024; 7-day rolling-origin forecasts across six stations",
        ha="center",
        color="#475569",
    )
    ax.grid(axis="y", color="#E2E8F0", linewidth=0.8)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    save_figure(fig, figures_dir / "final_test_model_errors.png")

    selected = predictions[(predictions["fold"] == "test_2024H1") & (predictions["model"] == selected_model)].copy()
    interval_half_width = float(uncertainty.loc[uncertainty["model"] == selected_model, "conformal_half_width_c"].iloc[0])
    stations = sorted(selected["station"].unique())
    fig, axes = plt.subplots(3, 2, figsize=(11, 9), sharex=True)
    for ax, station in zip(axes.flat, stations):
        station_data = selected[selected["station"] == station].sort_values("date")
        x = station_data["date"].to_numpy()
        y_pred = station_data["y_pred"].to_numpy(dtype=float)
        ax.fill_between(x, y_pred - interval_half_width, y_pred + interval_half_width, color=colors["interval"], alpha=0.55, label="90% conformal interval")
        ax.plot(x, station_data["y_true"], color=colors["actual"], linewidth=1.15, label="Observed")
        ax.plot(x, y_pred, color=colors["predicted"], linewidth=1.05, label="Predicted")
        ax.axhline(THRESHOLD_C, color=colors["accent"], linestyle="--", linewidth=0.9, label="36°C threshold")
        ax.set_title(STATION_DISPLAY.get(station, station))
        ax.set_ylabel("TMAX (°C)")
        ax.grid(color="#E2E8F0", linewidth=0.6)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.suptitle(f"Observed and predicted TMAX: {selected_model}", y=0.992, fontsize=13)
    fig.text(0.5, 0.957, "Final test: 1 January–30 June 2024; shaded bands calibrated on 2022–2023 validation residuals", ha="center", color="#475569")
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.935), ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.895))
    save_figure(fig, figures_dir / "selected_model_observed_vs_predicted.png")

    final_predictions = predictions[predictions["fold"] == "test_2024H1"].copy()
    residual_order = final_metrics["model"].tolist()
    residual_values = [
        (final_predictions.loc[final_predictions["model"] == model, "y_pred"] - final_predictions.loc[final_predictions["model"] == model, "y_true"]).to_numpy()
        for model in residual_order
    ]
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    boxes = ax.boxplot(residual_values, tick_labels=residual_order, showfliers=False, patch_artist=True)
    for box in boxes["boxes"]:
        box.set_facecolor("#DBEAFE")
        box.set_edgecolor("#1D4ED8")
    ax.axhline(0, color="#172033", linewidth=0.9)
    ax.set_ylabel("Prediction residual (predicted − observed, °C)")
    fig.suptitle("Final-test residual distributions", y=0.98, fontsize=13)
    fig.text(0.5, 0.925, "Boxes show central spread; outliers hidden for legibility", ha="center", color="#475569")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", color="#E2E8F0", linewidth=0.8)
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    save_figure(fig, figures_dir / "final_test_residual_distributions.png")

    heatwave_selected = labeled_predictions[
        (labeled_predictions["fold"] == "test_2024H1")
        & (labeled_predictions["model"] == selected_model)
        & (labeled_predictions["definition"] == "strict_gt_36")
    ]
    fig, axes = plt.subplots(3, 2, figsize=(11, 8.4), sharex=True)
    for ax, station in zip(axes.flat, stations):
        station_data = heatwave_selected[heatwave_selected["station"] == station].sort_values("date")
        ax.plot(station_data["date"], station_data["y_true"], color="#94A3B8", linewidth=0.9, label="Observed TMAX")
        actual_days = station_data[station_data["actual_heatwave_day"]]
        predicted_days = station_data[station_data["predicted_heatwave_day"]]
        ax.scatter(actual_days["date"], actual_days["y_true"], s=14, color="#172033", label="Observed heatwave day", zorder=3)
        ax.scatter(predicted_days["date"], predicted_days["y_pred"], s=20, marker="x", color="#D97706", label="Predicted heatwave day", zorder=4)
        ax.axhline(THRESHOLD_C, color="#2563EB", linestyle="--", linewidth=0.8)
        ax.set_title(STATION_DISPLAY.get(station, station))
        ax.set_ylabel("TMAX (°C)")
        ax.grid(color="#E2E8F0", linewidth=0.6)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.suptitle(f"Heatwave-day timeline: {selected_model}", y=0.992, fontsize=13)
    fig.text(0.5, 0.957, "Primary definition: TMAX > 36°C for at least three consecutive station-days", ha="center", color="#475569")
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.935), ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.895))
    save_figure(fig, figures_dir / "selected_model_heatwave_timeline.png")

    final_heatwave = heatwave_metrics[(heatwave_metrics["fold"] == "test_2024H1") & (heatwave_metrics["definition"] == "strict_gt_36")]
    for _, row in final_heatwave.iterrows():
        matrix = np.array([[row["tn"], row["fp"]], [row["fn"], row["tp"]]], dtype=int)
        fig, ax = plt.subplots(figsize=(4.4, 4.0))
        image = ax.imshow(matrix, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{matrix[i, j]:,}", ha="center", va="center", color="#172033", fontsize=12)
        ax.set_xticks([0, 1], ["No heatwave", "Heatwave"])
        ax.set_yticks([0, 1], ["No heatwave", "Heatwave"])
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("Observed label")
        ax.set_title(f"Heatwave-day confusion matrix: {row['model']}")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        safe_name = str(row["model"]).lower().replace(" ", "_")
        save_figure(fig, figures_dir / f"confusion_matrix_{safe_name}.png")


def write_old_notebook_audit(path: Path) -> None:
    text = """# Audit of the original modeling notebooks

| Notebook | Material issue | Consequence |
|---|---|---|
| Prophet error-rate | Uses `TAVG` as the target while the paper reports `TMAX`. | The reported Prophet scores are not comparable with the other models. |
| Prophet core | Forecasts 15 years from roughly 5.5 years of data; event code requires more than three days (`>3`), splits an event when severity changes, and leaves gaps at 38–39°C and 40–41°C. | Long-horizon and heatwave-event outputs are unsupported or incorrectly classified. |
| Random Forest | Uses a shuffled `train_test_split` and only station/year/month/day features. | Future dates leak into training and the score is optimistic for forecasting. |
| Decision Tree | Uses the same shuffled split and date-only feature setup. | Future dates leak into training and the result is not a time-series forecast. |
| LSTM | Fits scaling on the full dataset, uses the test set as validation, and the global sequence crosses station boundaries; the later station models also fit scalers before splitting. | Multiple leakage paths and physically invalid sequences make the score unusable. |
| ARIMA | Uses an 80/20 split and seasonal period `m=52` on daily data. | The seasonal specification does not represent annual daily seasonality and the comparison protocol differs from the tree models. |

The redevelopment pipeline replaces every reported score; none of the original error rates are carried forward.
"""
    path.write_text(text, encoding="utf-8")


def environment_versions() -> dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "matplotlib": matplotlib.__version__,
    }
    try:
        import statsmodels

        versions["statsmodels"] = statsmodels.__version__
    except Exception:
        versions["statsmodels"] = "unavailable"
    try:
        import prophet

        versions["prophet"] = prophet.__version__
    except Exception:
        versions["prophet"] = "unavailable"
    try:
        import tensorflow as tf

        versions["tensorflow"] = tf.__version__
    except Exception:
        versions["tensorflow"] = "unavailable"
    return versions


def run_pipeline(csv_path: Path, output_root: Path) -> dict[str, Any]:
    set_global_seed()
    dirs = ensure_dirs(output_root)
    df, audit, station_profile = load_and_validate_data(csv_path)
    feature_frame = build_training_features(df)
    (dirs["audit"] / "data_quality_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    station_profile.to_csv(dirs["audit"] / "station_data_profile.csv", index=False)
    write_old_notebook_audit(dirs["audit"] / "old_notebook_audit.md")

    experiment_config = {
        "seed": SEED,
        "target": "TMAX",
        "forecast_block_days": FORECAST_BLOCK_DAYS,
        "rolling_origin_update": "Parameters fixed within each outer fold; observed history/state updated at each weekly origin.",
        "folds": [asdict(fold) for fold in FOLDS],
        "lags": list(LAGS),
        "rolling_windows": list(ROLL_WINDOWS),
        "primary_heatwave_threshold_operator": ">",
        "sensitivity_heatwave_threshold_operator": ">=",
        "heatwave_threshold_c": THRESHOLD_C,
        "minimum_consecutive_days": MIN_CONSECUTIVE_DAYS,
        "model_selection_rule": "Lowest mean validation RMSE across 2022 and 2023; final 2024H1 test is not used for selection.",
    }
    (dirs["audit"] / "experiment_config.json").write_text(json.dumps(experiment_config, indent=2), encoding="utf-8")
    (dirs["audit"] / "environment_versions.json").write_text(json.dumps(environment_versions(), indent=2), encoding="utf-8")

    all_predictions: list[pd.DataFrame] = []
    all_hyperparameters: list[dict[str, Any]] = []
    tuning_frames: list[pd.DataFrame] = []
    arima_status_frames: list[pd.DataFrame] = []

    for fold in FOLDS:
        print(f"\n=== {fold.name}: train through {fold.train_end}, evaluate {fold.test_start} to {fold.test_end} ===", flush=True)
        train_end = pd.Timestamp(fold.train_end)
        test_start = pd.Timestamp(fold.test_start)
        test_end = pd.Timestamp(fold.test_end)
        all_predictions.extend(baseline_predictions(df, train_end, test_start, test_end, fold.name))

        ridge_candidates = [({"alpha": alpha}, lambda alpha=alpha: make_ridge(alpha)) for alpha in (0.1, 1.0, 10.0, 100.0)]
        ridge_parameters, ridge_tuning = tune_panel_model(feature_frame, df, train_end, ridge_candidates, "Ridge_AR")
        ridge_tuning["fold"] = fold.name
        tuning_frames.append(ridge_tuning)
        ridge_model = fit_panel_model(make_ridge(float(ridge_parameters["alpha"])), feature_frame, train_end)
        all_predictions.append(recursive_panel_predictions(ridge_model, df, test_start, test_end, "Ridge_AR", fold.name))
        all_hyperparameters.append({"fold": fold.name, "model": "Ridge_AR", "station": "ALL", "parameters": json.dumps(ridge_parameters, sort_keys=True)})
        print(f"Ridge selected {ridge_parameters}", flush=True)

        tree_grid = [(depth, leaf) for depth in (5, 10, None) for leaf in (5, 15, 30)]
        tree_candidates = [
            ({"max_depth": depth, "min_samples_leaf": leaf}, lambda depth=depth, leaf=leaf: make_decision_tree(depth, leaf))
            for depth, leaf in tree_grid
        ]
        tree_parameters, tree_tuning = tune_panel_model(feature_frame, df, train_end, tree_candidates, "Decision_Tree")
        tree_tuning["fold"] = fold.name
        tuning_frames.append(tree_tuning)
        tree_model = fit_panel_model(
            make_decision_tree(tree_parameters["max_depth"], int(tree_parameters["min_samples_leaf"])),
            feature_frame,
            train_end,
        )
        all_predictions.append(recursive_panel_predictions(tree_model, df, test_start, test_end, "Decision_Tree", fold.name))
        all_hyperparameters.append({"fold": fold.name, "model": "Decision_Tree", "station": "ALL", "parameters": json.dumps(tree_parameters, sort_keys=True)})
        print(f"Decision Tree selected {tree_parameters}", flush=True)

        forest_grid = [(depth, leaf, features) for depth in (10, None) for leaf in (2, 8) for features in (0.7,)]
        forest_candidates = [
            (
                {"n_estimators": 150, "max_depth": depth, "min_samples_leaf": leaf, "max_features": features},
                lambda depth=depth, leaf=leaf, features=features: make_random_forest(depth, leaf, features),
            )
            for depth, leaf, features in forest_grid
        ]
        forest_parameters, forest_tuning = tune_panel_model(feature_frame, df, train_end, forest_candidates, "Random_Forest")
        forest_tuning["fold"] = fold.name
        tuning_frames.append(forest_tuning)
        forest_model = fit_panel_model(
            make_random_forest(
                forest_parameters["max_depth"],
                int(forest_parameters["min_samples_leaf"]),
                float(forest_parameters["max_features"]),
            ),
            feature_frame,
            train_end,
        )
        all_predictions.append(recursive_panel_predictions(forest_model, df, test_start, test_end, "Random_Forest", fold.name))
        all_hyperparameters.append({"fold": fold.name, "model": "Random_Forest", "station": "ALL", "parameters": json.dumps(forest_parameters, sort_keys=True)})
        print(f"Random Forest selected {forest_parameters}", flush=True)

        arima_rankings, arima_tuning = tune_arima_orders_by_station(df, train_end)
        arima_tuning["fold"] = fold.name
        tuning_frames.append(arima_tuning)
        arima_predictions, arima_status = arima_outer_predictions(
            df, train_end, test_start, test_end, fold.name, arima_rankings
        )
        all_predictions.append(arima_predictions)
        arima_status_frames.append(arima_status)
        for _, status in arima_status.iterrows():
            all_hyperparameters.append(
                {
                    "fold": fold.name,
                    "model": "Dynamic_ARIMA",
                    "station": status["station"],
                    "parameters": json.dumps(
                        {
                            "order": status["order"],
                            "annual_fourier_order": 3,
                            "validation_rank_used": int(status["fallback_rank"]),
                        },
                        sort_keys=True,
                    ),
                }
            )
        print("Dynamic ARIMA station orders selected with convergence checks", flush=True)

        prophet_parameters, prophet_tuning = tune_prophet(df, train_end)
        prophet_tuning["fold"] = fold.name
        tuning_frames.append(prophet_tuning)
        all_predictions.append(prophet_outer_predictions(df, train_end, test_start, test_end, fold.name, prophet_parameters))
        all_hyperparameters.append({"fold": fold.name, "model": "Prophet", "station": "ALL", "parameters": json.dumps(prophet_parameters, sort_keys=True)})
        print(f"Prophet selected {prophet_parameters}", flush=True)

        lstm_predictions, lstm_parameters = lstm_outer_predictions(df, train_end, test_start, test_end, fold.name)
        all_predictions.append(lstm_predictions)
        all_hyperparameters.extend(lstm_parameters)
        print("LSTM completed", flush=True)

    predictions = pd.concat(all_predictions, ignore_index=True)
    predictions["origin_date"] = pd.to_datetime(predictions["origin_date"])
    predictions["date"] = pd.to_datetime(predictions["date"])
    predictions = predictions.sort_values(["fold", "model", "station", "date"]).reset_index(drop=True)
    predictions.to_csv(dirs["predictions"] / "all_model_predictions.csv", index=False)
    pd.DataFrame(all_hyperparameters).to_csv(dirs["metrics"] / "model_hyperparameters.csv", index=False)
    pd.concat(tuning_frames, ignore_index=True).to_csv(dirs["metrics"] / "inner_tuning_results.csv", index=False)
    pd.concat(arima_status_frames, ignore_index=True).to_csv(
        dirs["metrics"] / "arima_convergence_status.csv", index=False
    )

    overall_metrics, station_metrics, horizon_metrics = make_regression_tables(predictions, FOLDS)
    overall_metrics.to_csv(dirs["metrics"] / "regression_metrics_by_fold.csv", index=False)
    station_metrics.to_csv(dirs["metrics"] / "regression_metrics_by_station.csv", index=False)
    horizon_metrics.to_csv(dirs["metrics"] / "regression_metrics_by_horizon.csv", index=False)

    strict_metrics, strict_event_station_metrics, strict_labeled = heatwave_metrics_for_predictions(predictions, df, FOLDS, inclusive=False)
    inclusive_metrics, inclusive_event_station_metrics, inclusive_labeled = heatwave_metrics_for_predictions(predictions, df, FOLDS, inclusive=True)
    heatwave_metrics = pd.concat([strict_metrics, inclusive_metrics], ignore_index=True)
    event_station_metrics = pd.concat([strict_event_station_metrics, inclusive_event_station_metrics], ignore_index=True)
    labeled_predictions = pd.concat([strict_labeled, inclusive_labeled], ignore_index=True)
    heatwave_metrics.to_csv(dirs["metrics"] / "heatwave_day_and_event_metrics.csv", index=False)
    event_station_metrics.to_csv(dirs["metrics"] / "heatwave_event_metrics_by_station.csv", index=False)
    labeled_predictions.to_csv(dirs["predictions"] / "heatwave_labeled_predictions.csv", index=False)

    validation_summary, uncertainty, selected_model = final_test_uncertainty(predictions, overall_metrics)
    validation_summary.to_csv(dirs["metrics"] / "validation_model_summary.csv", index=False)
    uncertainty.to_csv(dirs["metrics"] / "final_test_uncertainty.csv", index=False)
    paired_comparison = paired_block_bootstrap(predictions, selected_model)
    paired_comparison.to_csv(dirs["metrics"] / "final_test_paired_block_comparisons.csv", index=False)

    create_figures(
        predictions,
        overall_metrics,
        heatwave_metrics,
        labeled_predictions,
        uncertainty,
        selected_model,
        dirs["figures"],
    )

    summary = {
        "selected_model_from_validation": selected_model,
        "selection_rule": experiment_config["model_selection_rule"],
        "dataset": audit,
        "final_test_regression": overall_metrics[overall_metrics["fold"] == "test_2024H1"].sort_values("rmse").to_dict(orient="records"),
        "final_test_heatwave_primary": heatwave_metrics[(heatwave_metrics["fold"] == "test_2024H1") & (heatwave_metrics["definition"] == "strict_gt_36")].sort_values("event_f1", ascending=False).to_dict(orient="records"),
        "limitations": [
            "Only station-level temperature variables are available; humidity, wind, rainfall, pressure, and forecast-model fields are absent.",
            "The data cover 2019-01-01 through 2024-06-30, which is too short for climate projection or a defensible 2037 forecast.",
            "The BMD definition includes a large-area component that cannot be tested with six station series.",
            "Heatwave positives are imbalanced and station-concentrated, particularly in Rajshahi.",
            "Seven-day recursive forecasts are short-range statistical forecasts, not numerical weather predictions.",
        ],
    }
    (dirs["root"] / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\nValidation-selected model: {selected_model}", flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True, help="Path to cleandata.csv")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run_pipeline(arguments.csv.resolve(), arguments.output.resolve())
