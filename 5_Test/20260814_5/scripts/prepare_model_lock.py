from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from runtime_guard import assert_sparrow_runtime


RUNTIME = assert_sparrow_runtime()
ROOT = Path(__file__).resolve().parents[1]
STAGE1 = ROOT.parent / "20260814_1"
STAGE2 = ROOT.parent / "20260814_2"
STAGE3 = ROOT.parent / "20260814_3"
STAGE4 = ROOT.parent / "20260814_4"
LOCK = ROOT / "model_lock.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if LOCK.exists():
        raise RuntimeError("model_lock.json already exists and is immutable")
    (ROOT / "inputs" / "registries").mkdir(parents=True, exist_ok=True)
    (ROOT / "reports").mkdir(parents=True, exist_ok=True)
    (ROOT / "outputs").mkdir(parents=True, exist_ok=True)

    development = pd.read_parquet(
        STAGE1 / "inputs" / "development_indata_2006_2018.parquet",
        columns=["q_site", "year", "Q_obsv_cfs", "explicit_upstream_net_cfs"],
    )
    development = development[
        development.Q_obsv_cfs.notna() & development.Q_obsv_cfs.gt(0) & development.year.le(2018)
    ].copy()
    development.q_site = development.q_site.astype(str)
    thresholds = development.groupby("q_site", as_index=False).agg(
        q20_cfs=("Q_obsv_cfs", lambda x: float(x.quantile(0.20))),
        q75_cfs=("Q_obsv_cfs", lambda x: float(x.quantile(0.75))),
        upstream_positive_input_q75_cfs=(
            "explicit_upstream_net_cfs",
            lambda x: float(x.quantile(0.75)),
        ),
        development_observations=("Q_obsv_cfs", "size"),
    )
    threshold_path = ROOT / "inputs" / "registries" / "locked_station_thresholds_2006_2018.parquet"
    thresholds.to_parquet(threshold_path, index=False)
    registry = {
        "frozen_before_locked_prediction_access": True,
        "threshold_period": ["2006-01", "2018-12"],
        "stations": int(len(thresholds)),
        "threshold_sha256": sha256(threshold_path),
        "lowflow_definition": "locked_actual_cfs <= frozen_station_q20_cfs",
        "tail_definition": "frozen_station_Q75_local_peak_then_2_to_6_consecutive_monotone_months_below_frozen_upstream_input_Q75_excluding_edges_and_right_censor",
        "q20_q75_recomputed_after_access": False,
    }
    registry_path = ROOT / "locked_metric_registry_manifest.json"
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")

    parameter_manifest = json.loads((STAGE1 / "reports" / "parameter_manifest.json").read_text(encoding="utf-8"))
    files = {
        "experiment_contract": STAGE1 / "experiment_contract.md",
        "scenario_contract": STAGE1 / "scenario_contract.json",
        "metrics_contract": STAGE1 / "metrics_contract.json",
        "parameter_manifest": STAGE1 / "reports" / "parameter_manifest.json",
        "spinup_contract": STAGE1 / "reports" / "spinup_contract.json",
        "environment": STAGE1 / "reports" / "environment.json",
        "instrumentation_neutrality": STAGE1 / "reports" / "instrumentation_neutrality.json",
        "topology": STAGE1 / "inputs" / "topology" / "topology_edges.csv",
        "parent_input": STAGE1 / "inputs" / "parent_indata.parquet",
        "fixed_states": STAGE1 / "outputs" / "fixed_branch_local_states.parquet",
        "stage2_gate": STAGE2 / "reports" / "stage2_flow_gate.json",
        "stage3_gate": STAGE3 / "reports" / "conditional_flow_gate.json",
        "stage4_skip": STAGE4 / "SKIPPED.json",
        "locked_registry": registry_path,
        "locked_thresholds": threshold_path,
        "confirmation_code": ROOT / "scripts" / "run_locked_confirmation.py",
        "groundwater_code": ROOT / "scripts" / "aggregate_locked_groundwater.py",
    }
    missing = [name for name, path in files.items() if not path.exists()]
    if missing:
        raise RuntimeError(f"Cannot lock; missing artifacts: {missing}")
    lock = {
        "scenario_id": "20260814_5",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "terminal_state": "operational_retained_interface_provisional",
        "operational_flow_model_id": "H0_hybrid",
        "operational_model_reason": "no_fixed_or_conditional_single_production_candidate_passed_complete_development_gate",
        "interface_branch_id": "main",
        "interface_status": "provisional_unresolved",
        "groundwater_status": "non_identifying",
        "sas_old_status": "retained_in_operational_H0; single_production_ablation_not_triggered",
        "performance_type": "locked_time_confirmation",
        "confirmation_runs_allowed": 1,
        "locked_period": ["2019-01", "2022-12"],
        "parameters": parameter_manifest,
        "runtime": RUNTIME,
        "thread_limits": {k: os.environ.get(k) for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]},
        "locked_metric_registry": registry,
        "artifact_sha256": {name: sha256(path) for name, path in files.items()},
        "confirmation_command": "conda --no-plugins run -n sparrow python E:/SPARROW/5_Test/20260814_5/scripts/run_locked_confirmation.py",
        "groundwater_command": "conda --no-plugins run -n sparrow python E:/SPARROW/5_Test/20260814_5/scripts/aggregate_locked_groundwater.py",
        "lock_written_before_locked_value_access": True,
        "model_recall_after_confirmation_allowed": False,
    }
    LOCK.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    print(json.dumps({"lock": str(LOCK), "locked": True, "artifacts": len(files)}, indent=2))


if __name__ == "__main__":
    main()
