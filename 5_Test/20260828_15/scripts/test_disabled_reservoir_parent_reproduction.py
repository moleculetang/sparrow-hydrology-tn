"""Verify that reservoir-aware routing exactly degenerates to the parent.

This preflight reads no observations and fits no parameters.  It reconstructs
the published routed fast/slow discharge from the frozen local components with
all reservoirs disabled.  Exact equality is required before any reservoir fit.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
OPERATOR_DIR = Path(r"E:\SPARROW\5_Test\20260828_14\scripts")
if str(OPERATOR_DIR) not in sys.path:
    sys.path.insert(0, str(OPERATOR_DIR))

from reservoir_network_router import (  # noqa: E402
    ReservoirRuntime,
    build_reach_graph,
    route_network_steps,
)
from reservoir_operator import ReservoirRule, ReservoirState  # noqa: E402


ROOT = Path(r"E:\SPARROW")
STAGE = ROOT / "5_Test" / "20260828_15"
PARENT = (
    ROOT
    / "5_Test"
    / "20260828_9"
    / "outputs"
    / "canonical_reach_daily_2006_2024.parquet"
)
TOPOLOGY = (
    ROOT
    / "5_Test"
    / "20260814_1"
    / "inputs"
    / "topology"
    / "topology_edges.csv"
)
TOPOLOGY_OVERRIDES = (
    ROOT
    / "5_Test"
    / "20260828_13"
    / "inputs"
    / "manual_reservoir_topology_overrides.csv"
)
PRIORITY_TOPOLOGY_AUDIT = (
    ROOT
    / "5_Test"
    / "20260828_13"
    / "outputs"
    / "priority_reservoir_topology_audit.parquet"
)
NETWORK_ROUTER = STAGE / "scripts" / "reservoir_network_router.py"
RESERVOIR_OPERATOR = ROOT / "5_Test" / "20260828_14" / "scripts" / "reservoir_operator.py"
EXPECTED_PARENT_SHA256 = (
    "a8c2b57d35de374a57798ab3e837dbf5c75cbd644275b6e52f888b20e16ca471"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_graph(reach_ids: list[int]) -> tuple[list[int], dict[int, tuple[int, float]]]:
    topology = pd.read_csv(TOPOLOGY)
    topology["reach_id"] = topology["reach_id"].astype(int)
    nodes = set(reach_ids)
    downstream: dict[int, tuple[int, float]] = {}
    for row in topology.itertuples(index=False):
        reach = int(row.reach_id)
        if reach not in nodes or pd.isna(row.downstream_reach):
            continue
        target = int(row.downstream_reach)
        if target not in nodes:
            raise ValueError(f"Reach {reach} points outside the registered domain")
        fraction = float(row.frac)
        if not 0 < fraction <= 1:
            raise ValueError(f"Invalid routing fraction on Reach {reach}: {fraction}")
        downstream[reach] = (target, fraction)

    indegree = {reach: 0 for reach in reach_ids}
    for target, _ in downstream.values():
        indegree[target] += 1
    queue = sorted(reach for reach, degree in indegree.items() if degree == 0)
    order: list[int] = []
    while queue:
        reach = queue.pop(0)
        order.append(reach)
        if reach in downstream:
            target = downstream[reach][0]
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
                queue.sort()
    if len(order) != len(reach_ids):
        raise ValueError("230-Reach topology is not a complete directed acyclic graph")
    return order, downstream


def disabled_rule() -> ReservoirRule:
    return ReservoirRule(
        capacity_m3=0.0,
        dead_storage_m3=0.0,
        target_storage_m3=0.0,
        baseline_release_m3_step=0.0,
        climatological_inflow_m3_step=0.0,
        inflow_response=0.0,
        storage_recovery_per_step=0.0,
        minimum_release_m3_step=0.0,
        maximum_controlled_release_m3_step=0.0,
        enabled=False,
    )


def load_disabled_reservoirs() -> list[ReservoirRuntime]:
    audit = pd.read_parquet(PRIORITY_TOPOLOGY_AUDIT)
    required = {
        "reservoir_entity_id",
        "waterbody_type",
        "operator_authorized",
        "registered_operator_inflow_reaches",
        "registered_unique_outflow_reach",
        "local_capture_fraction",
    }
    missing = required.difference(audit.columns)
    if missing:
        raise RuntimeError(f"Priority topology audit is missing fields: {sorted(missing)}")
    authorized = audit[audit["operator_authorized"].fillna(False)].copy()
    authorized = authorized[
        authorized["waterbody_type"] == "artificial_reservoir"
    ]
    runtimes: list[ReservoirRuntime] = []
    for row in authorized.itertuples(index=False):
        inflow_text = str(row.registered_operator_inflow_reaches)
        control_reaches = tuple(
            int(value.strip()) for value in inflow_text.split(";") if value.strip()
        )
        if not control_reaches:
            raise RuntimeError(
                f"Authorized reservoir {row.reservoir_entity_id} has no control Reach"
            )
        runtimes.append(
            ReservoirRuntime(
                entity_id=str(row.reservoir_entity_id),
                control_reach_ids=control_reaches,
                outflow_reach_id=int(row.registered_unique_outflow_reach),
                base_rule=disabled_rule(),
                state=ReservoirState.empty(),
                local_capture_fraction=float(row.local_capture_fraction),
            )
        )
    if len(runtimes) != 13:
        raise RuntimeError(f"Expected 13 authorized disabled reservoirs, found {len(runtimes)}")
    return runtimes


def main() -> None:
    parent_hash = sha256(PARENT)
    if parent_hash != EXPECTED_PARENT_SHA256:
        raise RuntimeError(f"Frozen parent hash changed: {parent_hash}")
    columns = [
        "date",
        "reach_id",
        "local_fast_response_m3_s",
        "local_slow_response_m3_s",
        "routed_fast_response_m3_s",
        "routed_slow_response_m3_s",
        "routed_total_m3_s",
    ]
    frame = pd.read_parquet(PARENT, columns=columns)
    if frame.duplicated(["date", "reach_id"]).any():
        raise RuntimeError("Parent contains duplicate date-Reach keys")
    frame = frame.sort_values(["date", "reach_id"], kind="mergesort").reset_index(drop=True)
    reach_ids = sorted(frame["reach_id"].unique().astype(int).tolist())
    dates = pd.Index(frame["date"].drop_duplicates())
    if reach_ids != list(range(1, 231)):
        raise RuntimeError("Parent does not contain exactly Reach IDs 1..230")
    if len(frame) != len(dates) * len(reach_ids):
        raise RuntimeError("Parent date-Reach grid is incomplete")
    if frame[columns[2:]].isna().any().any():
        raise RuntimeError("Parent routing fields contain missing values")

    local = frame[
        ["local_fast_response_m3_s", "local_slow_response_m3_s"]
    ].to_numpy(np.float64).reshape(len(dates), len(reach_ids), 2)
    expected = frame[
        ["routed_fast_response_m3_s", "routed_slow_response_m3_s"]
    ].to_numpy(np.float64).reshape(len(dates), len(reach_ids), 2)
    topology_frame = pd.read_csv(TOPOLOGY)
    graph = build_reach_graph(topology_frame, reach_ids)
    disabled_reservoirs = load_disabled_reservoirs()
    routed = route_network_steps(
        local[:, :, 0],
        local[:, :, 1],
        graph,
        disabled_reservoirs,
        dates=dates,
        values_are_volumes_per_step=False,
        keep_reservoir_diagnostics=False,
    )
    rebuilt = np.stack([routed.routed_fast, routed.routed_slow], axis=2)

    difference = rebuilt - expected
    rebuilt_total = rebuilt.sum(axis=2)
    expected_total = frame["routed_total_m3_s"].to_numpy(np.float64).reshape(
        len(dates), len(reach_ids)
    )
    result = {
        "stage": "20260828_15",
        "test": "disabled_reservoir_parent_reproduction",
        "status": "PASS" if np.array_equal(rebuilt, expected) and np.array_equal(rebuilt_total, expected_total) else "FAIL",
        "parent_sha256": parent_hash,
        "topology_sha256": sha256(TOPOLOGY),
        "topology_overrides_sha256": sha256(TOPOLOGY_OVERRIDES),
        "priority_topology_audit_sha256": sha256(PRIORITY_TOPOLOGY_AUDIT),
        "network_router_sha256": sha256(NETWORK_ROUTER),
        "reservoir_operator_sha256": sha256(RESERVOIR_OPERATOR),
        "test_script_sha256": sha256(Path(__file__)),
        "dates": int(len(dates)),
        "reaches": int(len(reach_ids)),
        "rows": int(len(frame)),
        "max_abs_fast_difference_m3_s": float(np.max(np.abs(difference[:, :, 0]))),
        "max_abs_slow_difference_m3_s": float(np.max(np.abs(difference[:, :, 1]))),
        "max_abs_total_difference_m3_s": float(np.max(np.abs(rebuilt_total - expected_total))),
        "bitwise_fast_slow_equal": bool(np.array_equal(rebuilt, expected)),
        "bitwise_total_equal": bool(np.array_equal(rebuilt_total, expected_total)),
        "registered_disabled_reservoir_count": len(disabled_reservoirs),
        "registered_disabled_reservoir_ids": [
            runtime.entity_id for runtime in disabled_reservoirs
        ],
        "formal_network_router_exercised": True,
        "disabled_reservoir_step_operator_exercised": True,
        "observations_read": False,
        "parameters_fitted": False,
    }
    output = STAGE / "reports" / "disabled_reservoir_parent_reproduction.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "PASS":
        raise SystemExit("BLOCKED_PARENT_REPRODUCTION_MISMATCH")


if __name__ == "__main__":
    main()
