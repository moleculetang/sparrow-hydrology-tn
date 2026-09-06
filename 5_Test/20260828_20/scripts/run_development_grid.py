"""Fit shared reservoir hyperparameters without opening held-out outcomes."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
STAGE = ROOT / "5_Test" / "20260828_20"
SCRIPT_DIR = STAGE / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from reservoir_development_core import (  # noqa: E402
    PARENT,
    ROLE_AUDIT,
    load_static_context,
    metric_summary,
    prepare_reservoir_metadata,
    simulate_candidate,
    station_monthly_prediction,
)


WORKERS = 5
TAU_GRID = (60.0, 180.0, 365.0)
ALPHA_GRID = (0.25, 0.50, 0.75)
DRAWDOWN_GRID = (0.05, 0.15)

_CONTEXT: dict[str, object] | None = None
_METADATA: dict[str, dict[str, object]] | None = None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def initialize_worker() -> None:
    global _CONTEXT, _METADATA
    _CONTEXT = load_static_context("2018-12-31")
    _METADATA = prepare_reservoir_metadata(_CONTEXT)


def score_candidate(candidate: dict[str, object]) -> dict[str, object]:
    if _CONTEXT is None or _METADATA is None:
        raise RuntimeError("Worker context was not initialized")
    total_rate, _, _ = simulate_candidate(
        _CONTEXT,
        _METADATA,
        str(candidate["layer"]),
        float(candidate["tau_day"]),
        float(candidate["inflow_response"]),
        float(candidate["drawdown_fraction"]),
        keep_diagnostics=False,
    )
    prediction = station_monthly_prediction(_CONTEXT, total_rate)
    result = dict(candidate)
    result.update({f"fit_{key}": value for key, value in metric_summary(
        _CONTEXT["obs_monthly"], prediction, (2010, 2015)
    ).items()})
    result.update({f"selection_{key}": value for key, value in metric_summary(
        _CONTEXT["obs_monthly"], prediction, (2016, 2018)
    ).items()})
    return result


def noninferiority(candidate: dict[str, float], parent: dict[str, float]) -> dict[str, bool]:
    checks = {
        "pooled_log_RMSE": candidate["pooled_log_RMSE"] - parent["pooled_log_RMSE"] <= 0.01,
        "pooled_NSE": candidate["pooled_NSE"] - parent["pooled_NSE"] >= -0.02,
        "station_median_NSE": candidate["station_median_NSE"] - parent["station_median_NSE"] >= -0.03,
        "station_median_absolute_PBIAS_pct": (
            candidate["station_median_absolute_PBIAS_pct"]
            - parent["station_median_absolute_PBIAS_pct"]
            <= 2.0
        ),
    }
    checks["all"] = all(checks.values())
    return checks


def main() -> None:
    outputs = STAGE / "outputs"
    reports = STAGE / "reports"
    locks = STAGE / "locks"
    for folder in [outputs, reports, locks]:
        folder.mkdir(parents=True, exist_ok=True)

    context = load_static_context("2018-12-31")
    metadata = prepare_reservoir_metadata(context)
    parent_prediction = station_monthly_prediction(context, context["routed_total_rate"])
    parent_fit = metric_summary(context["obs_monthly"], parent_prediction, (2010, 2015))
    parent_selection = metric_summary(context["obs_monthly"], parent_prediction, (2016, 2018))

    candidates: list[dict[str, object]] = []
    for layer, tau, alpha, drawdown in itertools.product(
        ("R1", "R2"), TAU_GRID, ALPHA_GRID, DRAWDOWN_GRID
    ):
        candidates.append(
            {
                "candidate_id": f"{layer}_tau{tau:g}_a{alpha:.2f}_d{drawdown:.2f}",
                "layer": layer,
                "tau_day": tau,
                "inflow_response": alpha,
                "drawdown_fraction": drawdown,
            }
        )

    rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=WORKERS, initializer=initialize_worker) as pool:
        futures = {pool.submit(score_candidate, candidate): candidate for candidate in candidates}
        for index, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            rows.append(row)
            print(
                f"[{index:02d}/{len(candidates)}] {row['candidate_id']} "
                f"fit={row['fit_pooled_log_RMSE']:.6f} "
                f"selection={row['selection_pooled_log_RMSE']:.6f}",
                flush=True,
            )
    scores = pd.DataFrame(rows).sort_values(["layer", "fit_pooled_log_RMSE", "candidate_id"])
    if len(scores) != len(candidates) or scores["candidate_id"].duplicated().any():
        raise RuntimeError("Candidate grid is incomplete or duplicated")
    scores.to_parquet(outputs / "development_candidate_scores.parquet", index=False)

    selected_rows = []
    for layer in ["R1", "R2"]:
        selected_rows.append(
            scores[scores["layer"] == layer]
            .sort_values(["fit_pooled_log_RMSE", "candidate_id"], kind="mergesort")
            .iloc[0]
            .to_dict()
        )
    selected = {str(row["layer"]): row for row in selected_rows}

    layer_selection_metrics: dict[str, dict[str, float]] = {"R0": parent_selection}
    layer_fit_metrics: dict[str, dict[str, float]] = {"R0": parent_fit}
    noninferiority_checks: dict[str, dict[str, bool]] = {}
    for layer, row in selected.items():
        fit_metrics = {
            key.removeprefix("fit_"): float(value)
            for key, value in row.items()
            if key.startswith("fit_")
        }
        selection_metrics = {
            key.removeprefix("selection_"): float(value)
            for key, value in row.items()
            if key.startswith("selection_")
        }
        layer_fit_metrics[layer] = fit_metrics
        layer_selection_metrics[layer] = selection_metrics
        noninferiority_checks[layer] = noninferiority(selection_metrics, parent_selection)

    if noninferiority_checks["R2"]["all"]:
        tn_layer = "R2"
    elif noninferiority_checks["R1"]["all"]:
        tn_layer = "R1"
    else:
        tn_layer = "R0"
    empirically_best = min(
        layer_selection_metrics,
        key=lambda layer: layer_selection_metrics[layer]["pooled_log_RMSE"],
    )
    empirical_improvement = (
        layer_selection_metrics[empirically_best]["pooled_log_RMSE"]
        < parent_selection["pooled_log_RMSE"]
    )
    if tn_layer == "R0":
        status = "RESERVOIR_STRUCTURE_INFERIOR_RETAIN_R0"
    elif empirical_improvement and empirically_best == tn_layer:
        status = "STATE_CONSISTENT_RESERVOIR_INTERFACE_EMPIRICALLY_SUPPORTED_SHARED_PARAMETERS_ONLY"
    else:
        status = "STATE_CONSISTENT_RESERVOIR_INTERFACE_NONINFERIOR_STRUCTURAL_PRIOR_ONLY"

    metadata_rows = []
    for entity, item in metadata.items():
        metadata_rows.append(
            {
                "reservoir_entity_id": entity,
                **{
                    key: (";".join(map(str, value)) if isinstance(value, tuple) else value)
                    for key, value in item.items()
                },
            }
        )
    pd.DataFrame(metadata_rows).to_parquet(outputs / "locked_reservoir_static_metadata.parquet", index=False)

    lock = {
        "stage": "20260828_20",
        "status": status,
        "development_only": True,
        "held_out_2019_2022_opened": False,
        "four_spatial_station_observations_opened": False,
        "tn_observations_opened": False,
        "workers": WORKERS,
        "candidate_count": len(candidates),
        "station_count": len(context["stations"]),
        "excluded_control_reach_stations": ["武宣（二）", "平山（三）"],
        "fit_period": "2010-2015",
        "selection_period": "2016-2018",
        "selected_candidates": {
            layer: {
                "candidate_id": row["candidate_id"],
                "tau_day": float(row["tau_day"]),
                "inflow_response": float(row["inflow_response"]),
                "drawdown_fraction": float(row["drawdown_fraction"]),
            }
            for layer, row in selected.items()
        },
        "fit_metrics": layer_fit_metrics,
        "selection_metrics": layer_selection_metrics,
        "noninferiority_against_R0": noninferiority_checks,
        "tn_interface_layer": tn_layer,
        "empirically_best_layer": empirically_best,
        "empirically_best_improves_primary_metric": empirical_improvement,
        "reservoir_specific_release_gate": "NOT_AUTHORIZED_NO_ELIGIBLE_TOTAL_RELEASE_SERIES",
        "claim_boundary": (
            "Only shared hyperparameters are calibrated from downstream flow. Per-reservoir "
            "operations and releases are not identified. A noninferior R1/R2 selection supports "
            "a conservative structural TN interface, not a validated reconstruction of real rules."
        ),
        "hashes": {
            "contract": sha256(STAGE / "experiment_contract.json"),
            "parent": sha256(PARENT),
            "observation_role_registry": sha256(ROLE_AUDIT),
            "candidate_scores": sha256(outputs / "development_candidate_scores.parquet"),
            "core_code": sha256(SCRIPT_DIR / "reservoir_development_core.py"),
            "runner_code": sha256(Path(__file__)),
        },
    }
    (locks / "development_structure_parameter_lock.json").write_text(
        json.dumps(lock, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (reports / "development_decision.json").write_text(
        json.dumps(lock, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(lock, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
