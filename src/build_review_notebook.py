#!/usr/bin/env python3
"""Build and execute the reader-facing model-redo review notebook."""

from __future__ import annotations

import base64
import contextlib
import io
import os
import traceback
from pathlib import Path

import IPython.display as ipython_display
import nbformat as nbf


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


def execute_in_process(notebook: nbf.NotebookNode, working_directory: Path) -> None:
    """Execute cells in order when the sandbox prohibits all Jupyter sockets."""
    namespace: dict[str, object] = {"__name__": "__main__"}
    old_directory = Path.cwd()
    original_display = ipython_display.display
    try:
        os.chdir(working_directory)
        execution_count = 0
        for cell_index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            execution_count += 1
            outputs: list[nbf.NotebookNode] = []

            def capture_display(*objects, **_: object) -> None:
                for obj in objects:
                    if isinstance(obj, ipython_display.Image):
                        png = obj._repr_png_()
                        if png is None:
                            outputs.append(
                                nbf.v4.new_output(
                                    "display_data",
                                    data={"text/plain": repr(obj)},
                                    metadata={},
                                )
                            )
                        else:
                            image_metadata = {}
                            if isinstance(png, tuple):
                                png, image_metadata = png
                            outputs.append(
                                nbf.v4.new_output(
                                    "display_data",
                                    data={
                                        "image/png": png if isinstance(png, str) else base64.b64encode(png).decode("ascii"),
                                        "text/plain": "<embedded PNG figure>",
                                    },
                                    metadata=image_metadata,
                                )
                            )
                    else:
                        html = getattr(obj, "_repr_html_", lambda: None)()
                        data = {"text/plain": repr(obj)}
                        if html is not None:
                            data["text/html"] = html
                        outputs.append(nbf.v4.new_output("display_data", data=data, metadata={}))

            ipython_display.display = capture_display
            stdout = io.StringIO()
            stderr = io.StringIO()
            try:
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    exec(compile(cell["source"], f"notebook-cell-{cell_index}", "exec"), namespace)
            except Exception as error:
                cell["outputs"] = outputs + [
                    nbf.v4.new_output(
                        "error",
                        ename=type(error).__name__,
                        evalue=str(error),
                        traceback=traceback.format_exc().splitlines(),
                    )
                ]
                cell["execution_count"] = execution_count
                raise
            if stdout.getvalue():
                outputs.insert(0, nbf.v4.new_output("stream", name="stdout", text=stdout.getvalue()))
            if stderr.getvalue():
                outputs.append(nbf.v4.new_output("stream", name="stderr", text=stderr.getvalue()))
            cell["outputs"] = outputs
            cell["execution_count"] = execution_count
        notebook["metadata"]["execution_backend"] = "in-process fallback; Jupyter sockets prohibited by workspace sandbox"
    finally:
        ipython_display.display = original_display
        os.chdir(old_directory)


def build_notebook(package_root: Path) -> Path:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    }
    notebook["cells"] = [
        markdown(
            """
# Heatwave TMAX forecasting: rebuilt model experiment

## tl;dr

This notebook reviews the completed leakage-controlled rerun. `Dynamic_ARIMA` was selected using 2022–2023 validation RMSE (2.242°C), while `Ridge_AR` was numerically best on the untouched January–June 2024 test (RMSE 2.679°C versus 2.742°C). Their paired station-week MAE difference was not statistically supported at 95%. Heatwave event detection was harder: persistence had the highest final event F1 (0.514), so continuous-temperature accuracy and event detection must be reported separately.

The notebook reads the saved, fully reproducible outputs. The complete training implementation is in `src/heatwave_model_redo.py`.
"""
        ),
        markdown("## Context & methods\n\nThe source data contain station-level daily `TMAX`, `TMIN`, and `TAVG`. This rebuild targets only `TMAX` and uses expanding outer folds with seven-day rolling origins: validation years 2022 and 2023, followed by a final 2024H1 holdout."),
        code(
            """
from pathlib import Path
import json
import pandas as pd
from IPython.display import Image, display

ROOT = Path.cwd()
OUTPUTS = ROOT / "outputs"
METRICS = OUTPUTS / "metrics"
FIGURES = OUTPUTS / "figures"
AUDIT = OUTPUTS / "audit"

with (OUTPUTS / "run_summary.json").open() as handle:
    summary = json.load(handle)
with (AUDIT / "data_quality_audit.json").open() as handle:
    data_audit = json.load(handle)
with (AUDIT / "experiment_config.json").open() as handle:
    experiment = json.load(handle)

validation = pd.read_csv(METRICS / "validation_model_summary.csv")
regression = pd.read_csv(METRICS / "regression_metrics_by_fold.csv")
uncertainty = pd.read_csv(METRICS / "final_test_uncertainty.csv")
paired = pd.read_csv(METRICS / "final_test_paired_block_comparisons.csv")
heatwave = pd.read_csv(METRICS / "heatwave_day_and_event_metrics.csv")
station_profile = pd.read_csv(AUDIT / "station_data_profile.csv")

print(f"Validation-selected model: {summary['selected_model_from_validation']}")
print(f"Data range: {data_audit['start_date']} to {data_audit['end_date']}")
"""
        ),
        markdown(
            """
### Key assumptions

- Forecasts are recursive inside each seven-day block; the next block may use newly observed history.
- Hyperparameters are chosen within each outer training period, never from its evaluation dates.
- The final 2024H1 data are not used for model selection.
- The primary station-level heatwave definition is `TMAX > 36°C` for at least three consecutive days; `>=36°C` is a sensitivity analysis.
- The nationwide “large-area” component of the BMD wording cannot be evaluated from six stations.
- The final two dates of a fold are excluded from heatwave classification to avoid right-censoring of a possible three-day run. Regression metrics use every date.
"""
        ),
        markdown("## Data"),
        code(
            """
audit_rows = pd.DataFrame(
    {
        "Item": [
            "Rows", "Stations", "Date range", "Null cells", "Duplicate station-dates",
            "Missing calendar dates", "TMAX below TMIN", "TAVG outside min/max"
        ],
        "Result": [
            f"{data_audit['rows']:,}", data_audit['stations'],
            f"{data_audit['start_date']} to {data_audit['end_date']}",
            data_audit['null_cells'], data_audit['duplicate_station_dates'],
            0 if data_audit['complete_daily_calendar_all_stations'] else "present",
            data_audit['invalid_tmax_below_tmin'], data_audit['tavg_outside_tmin_tmax']
        ],
    }
)
display(audit_rows)
display(station_profile)
"""
        ),
        markdown(
            """
The single `TMAX < TMIN` row and the `TAVG` inconsistencies were retained rather than silently corrected. `TAVG` is not a target or predictor in the rebuilt experiment. The record ends on 30 June 2024—not August—and is not long enough for a defensible climate projection to 2037.
"""
        ),
        markdown("## Results\n\n### Validation determines the selected model"),
        code(
            """
validation_view = validation.sort_values("rmse_mean")[
    ["model", "mae_mean", "rmse_mean", "mape_pct_mean", "smape_pct_mean", "r2_mean"]
].rename(columns={
    "model": "Model", "mae_mean": "Mean MAE (°C)", "rmse_mean": "Mean RMSE (°C)",
    "mape_pct_mean": "Mean MAPE (%)", "smape_pct_mean": "Mean sMAPE (%)", "r2_mean": "Mean R²"
})
display(validation_view.round(3))
"""
        ),
        markdown("### Untouched 2024H1 final test"),
        code(
            """
final_view = regression[regression["role"] == "final_test"].sort_values("rmse")[
    ["model", "mae", "rmse", "mape_pct", "smape_pct", "r2", "bias"]
].rename(columns={
    "model": "Model", "mae": "MAE (°C)", "rmse": "RMSE (°C)",
    "mape_pct": "MAPE (%)", "smape_pct": "sMAPE (%)", "r2": "R²", "bias": "Bias (°C)"
})
display(final_view.round(3))
display(Image(filename=str(FIGURES / "final_test_model_errors.png"), width=900))
"""
        ),
        markdown(
            """
`Ridge_AR` leads the holdout numerically, but that result does not override the prespecified selection rule. The paired comparison below defines the difference as comparator MAE minus selected-model MAE; positive values favor `Dynamic_ARIMA`.
"""
        ),
        code(
            """
paired_view = paired[[
    "comparator_model", "mean_mae_difference_comparator_minus_selected",
    "ci95_lower", "ci95_upper", "difference_supported_at_95pct"
]].rename(columns={
    "comparator_model": "Comparator", "mean_mae_difference_comparator_minus_selected": "Δ MAE (°C)",
    "ci95_lower": "95% CI lower", "ci95_upper": "95% CI upper",
    "difference_supported_at_95pct": "95% supported"
})
display(paired_view.round(3))
"""
        ),
        markdown("### Predictive uncertainty and station traces"),
        code(
            """
uncertainty_view = uncertainty.sort_values("rmse")[[
    "model", "mae_ci95_lower", "mae_ci95_upper", "rmse_ci95_lower", "rmse_ci95_upper",
    "conformal_half_width_c", "final_test_coverage"
]].rename(columns={
    "model": "Model", "mae_ci95_lower": "MAE CI lower", "mae_ci95_upper": "MAE CI upper",
    "rmse_ci95_lower": "RMSE CI lower", "rmse_ci95_upper": "RMSE CI upper",
    "conformal_half_width_c": "90% half-width (°C)", "final_test_coverage": "Final coverage"
})
display(uncertainty_view.round(3))
display(Image(filename=str(FIGURES / "selected_model_observed_vs_predicted.png"), width=1000))
"""
        ),
        markdown("### Heatwave-day and event performance"),
        code(
            """
primary_hw = heatwave[(heatwave["role"] == "final_test") & (heatwave["definition"] == "strict_gt_36")]
primary_view = primary_hw.sort_values("event_f1", ascending=False)[[
    "model", "precision", "recall", "f1", "observed_events", "predicted_events",
    "matched_events", "event_precision", "event_recall", "event_f1"
]].rename(columns={
    "model": "Model", "precision": "Day precision", "recall": "Day recall", "f1": "Day F1",
    "observed_events": "Observed events", "predicted_events": "Predicted events",
    "matched_events": "Matched events", "event_precision": "Event precision",
    "event_recall": "Event recall", "event_f1": "Event F1"
})
display(primary_view.round(3))
display(Image(filename=str(FIGURES / "selected_model_heatwave_timeline.png"), width=1000))
"""
        ),
        markdown("## Validation checks"),
        code(
            """
with (AUDIT / "output_validation.json").open() as handle:
    validation_checks = json.load(handle)
display(pd.DataFrame(validation_checks.items(), columns=["Check", "Result"]))
assert validation_checks["all_critical_checks_pass"]
"""
        ),
        markdown(
            """
## Takeaways

1. The rebuilt evidence supports dynamic harmonic ARIMA as the validation-selected model and Ridge autoregression as the numerically strongest 2024H1 holdout model; their difference is not decisive.
2. Prophet is not the leading model in this experiment, and the original notebook scores should not be reused.
3. Heatwave event detection is a separate, harder task. Persistence has the strongest event F1, while the smooth statistical models often miss threshold crossings.
4. The 90% conformal band undercovers in 2024H1, so uncertainty claims need a distribution-shift caveat.
5. The data support the evaluated 1–7-day station forecast, not a 2037 climate projection or an official nationwide BMD heatwave declaration.
"""
        ),
    ]

    destination = package_root / "Heatwave_Model_Redo.ipynb"
    nbf.write(notebook, destination)
    execute_in_process(notebook, package_root)
    nbf.write(notebook, destination)
    return destination


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    print(build_notebook(root))
