"""Complete only the physical-null edges of the Stage-20 grid."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import itertools
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(r"E:\SPARROW")
STAGE = ROOT / "5_Test" / "20260828_21"
S20 = ROOT / "5_Test" / "20260828_20"
sys.path.insert(0, str(S20 / "scripts"))

from reservoir_development_core import (  # noqa: E402
    PARENT,
    ROLE_AUDIT,
    load_static_context,
    metric_summary,
    station_monthly_prediction,
)
from run_development_grid import initialize_worker, noninferiority, score_candidate  # noqa: E402


WORKERS = 5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    outputs = STAGE / "outputs"
    reports = STAGE / "reports"
    locks = STAGE / "locks"
    for folder in [outputs, reports, locks]:
        folder.mkdir(parents=True, exist_ok=True)

    old = pd.read_parquet(S20 / "outputs" / "development_candidate_scores.parquet")
    new_candidates = []
    for layer, tau, alpha, drawdown in itertools.product(
        ("R1", "R2"), (60.0, 180.0, 365.0), (0.0, 0.25, 0.50, 0.75), (0.0, 0.05, 0.15)
    ):
        candidate_id = f"{layer}_tau{tau:g}_a{alpha:.2f}_d{drawdown:.2f}"
        if candidate_id not in set(old["candidate_id"]):
            new_candidates.append(
                {
                    "candidate_id": candidate_id,
                    "layer": layer,
                    "tau_day": tau,
                    "inflow_response": alpha,
                    "drawdown_fraction": drawdown,
                }
            )
    if len(new_candidates) != 36:
        raise RuntimeError(f"Expected 36 physical-null completion candidates, found {len(new_candidates)}")

    rows = []
    with ProcessPoolExecutor(max_workers=WORKERS, initializer=initialize_worker) as pool:
        futures = {pool.submit(score_candidate, candidate): candidate for candidate in new_candidates}
        for index, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            rows.append(row)
            print(
                f"[{index:02d}/{len(new_candidates)}] {row['candidate_id']} "
                f"fit={row['fit_pooled_log_RMSE']:.6f} "
                f"selection={row['selection_pooled_log_RMSE']:.6f}",
                flush=True,
            )
    new = pd.DataFrame(rows)
    new.to_parquet(outputs / "boundary_completion_candidate_scores.parquet", index=False)
    complete = pd.concat([old, new], ignore_index=True).sort_values(
        ["layer", "fit_pooled_log_RMSE", "candidate_id"]
    )
    if len(complete) != 72 or complete["candidate_id"].duplicated().any():
        raise RuntimeError("Completed grid must contain 72 unique candidates")
    complete.to_parquet(outputs / "completed_candidate_scores.parquet", index=False)

    context = load_static_context("2018-12-31")
    parent_prediction = station_monthly_prediction(context, context["routed_total_rate"])
    parent_fit = metric_summary(context["obs_monthly"], parent_prediction, (2010, 2015))
    parent_selection = metric_summary(context["obs_monthly"], parent_prediction, (2016, 2018))
    selected = {
        layer: complete[complete["layer"] == layer]
        .sort_values(["fit_pooled_log_RMSE", "candidate_id"], kind="mergesort")
        .iloc[0]
        .to_dict()
        for layer in ["R1", "R2"]
    }
    fit_metrics = {"R0": parent_fit}
    selection_metrics = {"R0": parent_selection}
    noninferiority_checks = {}
    for layer, row in selected.items():
        fit_metrics[layer] = {
            key.removeprefix("fit_"): float(value)
            for key, value in row.items() if key.startswith("fit_")
        }
        selection_metrics[layer] = {
            key.removeprefix("selection_"): float(value)
            for key, value in row.items() if key.startswith("selection_")
        }
        noninferiority_checks[layer] = noninferiority(selection_metrics[layer], parent_selection)

    tn_layer = "R2" if noninferiority_checks["R2"]["all"] else (
        "R1" if noninferiority_checks["R1"]["all"] else "R0"
    )
    zero_effects = {
        layer: {
            "inflow_response_is_zero": float(row["inflow_response"]) == 0.0,
            "seasonal_drawdown_is_zero": float(row["drawdown_fraction"]) == 0.0,
        }
        for layer, row in selected.items()
    }
    lock = {
        "stage": "20260828_21",
        "status": "BOUNDARY_COMPLETE_DEVELOPMENT_LOCKED",
        "held_out_2019_2022_opened": False,
        "four_spatial_station_observations_opened": False,
        "tn_observations_opened": False,
        "old_candidate_count": len(old),
        "added_candidate_count": len(new),
        "completed_candidate_count": len(complete),
        "selected_candidates": {
            layer: {
                "candidate_id": row["candidate_id"],
                "tau_day": float(row["tau_day"]),
                "inflow_response": float(row["inflow_response"]),
                "drawdown_fraction": float(row["drawdown_fraction"]),
            }
            for layer, row in selected.items()
        },
        "zero_effect_interpretation": zero_effects,
        "fit_metrics": fit_metrics,
        "selection_metrics": selection_metrics,
        "noninferiority_against_R0": noninferiority_checks,
        "tn_interface_layer": tn_layer,
        "reservoir_specific_release_gate": "NOT_AUTHORIZED_NO_ELIGIBLE_TOTAL_RELEASE_SERIES",
        "claim_boundary": (
            "The physical null was explicitly tested. Nonzero shared parameters can be interpreted "
            "only as downstream-flow-supported shared behavior; no individual operation is identified."
        ),
        "hashes": {
            "contract": sha256(STAGE / "experiment_contract.json"),
            "stage20_scores": sha256(S20 / "outputs" / "development_candidate_scores.parquet"),
            "new_scores": sha256(outputs / "boundary_completion_candidate_scores.parquet"),
            "complete_scores": sha256(outputs / "completed_candidate_scores.parquet"),
            "parent": sha256(PARENT),
            "observation_role_registry": sha256(ROLE_AUDIT),
            "runner_code": sha256(Path(__file__)),
        },
    }
    for path in [locks / "boundary_complete_development_lock.json", reports / "boundary_completion_decision.json"]:
        path.write_text(json.dumps(lock, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(lock, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
