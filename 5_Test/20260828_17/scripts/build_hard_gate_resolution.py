from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(r"E:\SPARROW")
STAGE = ROOT / "5_Test/20260828_17"
TOPOLOGY = ROOT / "5_Test/20260828_13/outputs/priority_reservoir_topology_audit.parquet"
EVIDENCE = STAGE / "inputs/reservoir_hard_gate_evidence.csv"
ROUTER = ROOT / "5_Test/20260828_15/scripts/reservoir_network_router.py"
ROUTER_TESTS = ROOT / "5_Test/20260828_15/reports/reservoir_network_router_test_results.json"
PARENT_TEST = ROOT / "5_Test/20260828_15/reports/disabled_reservoir_parent_reproduction.json"
Q72_PARENT_AREA = ROOT / "5_Test/20260823_27/outputs/q72_full_state_tn_bridge_2006_2022.parquet"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    topology = pd.read_parquet(TOPOLOGY)
    evidence = pd.read_csv(EVIDENCE)
    router_tests = json.loads(ROUTER_TESTS.read_text(encoding="utf-8"))
    parent_test = json.loads(PARENT_TEST.read_text(encoding="utf-8"))
    priority = topology.loc[
        topology["priority_tier"] == "priority_13_artificial_reservoirs"
    ].set_index("reservoir_entity_id")

    required_facts = {
        ("GRAND_7196", "first_impoundment"),
        ("GRAND_7196", "phase1_total_capacity"),
        ("GRAND_7036", "first_generation"),
        ("LOCAL_DATENGXIA", "first_impoundment"),
        ("GRAND_5703", "official_control_area"),
        ("GRAND_5758", "official_control_area"),
    }
    facts = set(zip(evidence["reservoir_entity_id"], evidence["fact_name"]))
    if not required_facts.issubset(facts):
        raise RuntimeError(f"Missing hard-gate evidence: {sorted(required_facts - facts)}")

    official_area = (
        evidence.loc[evidence["fact_name"] == "official_control_area"]
        .set_index("reservoir_entity_id")["value_numeric"]
        .astype(float)
    )
    model_area = priority["model_control_area_km2"].astype(float)
    q72_area = (
        pd.read_parquet(Q72_PARENT_AREA, columns=["reach_id", "catchment_area_km2"])
        .drop_duplicates("reach_id")
        .set_index("reach_id")["catchment_area_km2"]
        .astype(float)
    )
    guishi_fraction = float(model_area["GRAND_5703"] / official_area["GRAND_5703"])
    chengbihe_fraction = float(
        model_area["GRAND_5720"] / official_area["GRAND_5720"]
    )
    baipenzhu_fraction = float(
        official_area["GRAND_5758"] / q72_area.loc[17]
    )

    test_names = {row["test"] for row in router_tests["tests"]}
    partial_tests = {
        "test_disabled_partial_capture_is_bitwise_parent",
        "test_baipenzhu_partial_local_capture_and_bypass_close",
        "test_partial_capture_retains_all_upstream_water",
        "test_invalid_partial_capture_is_rejected",
    }
    partial_router_ready = (
        router_tests.get("status") == "PASS"
        and partial_tests.issubset(test_names)
        and parent_test.get("status") == "PASS"
        and parent_test.get("bitwise_fast_slow_equal") is True
        and parent_test.get("bitwise_total_equal") is True
    )

    rows = [
        {
            "reservoir_entity_id": "GRAND_7196",
            "reservoir_name_zh": "龙滩水电站水库",
            "gate": "activation_and_capacity_version",
            "status": "RESOLVED",
            "operator_start": "2006-09-30",
            "fit_eligible_2010_2018": True,
            "extension_operator_eligible": True,
            "model_action": "use_phase1_capacity_16200_million_m3",
            "numeric_factor": 1.0,
        },
        {
            "reservoir_entity_id": "GRAND_7036",
            "reservoir_name_zh": "长洲水利枢纽水库",
            "gate": "activation_date",
            "status": "RESOLVED_CONSERVATIVE_START",
            "operator_start": "2007-10-30",
            "fit_eligible_2010_2018": True,
            "extension_operator_eligible": True,
            "model_action": "identity_before_first_verified_operation",
            "numeric_factor": 1.0,
        },
        {
            "reservoir_entity_id": "LOCAL_DATENGXIA",
            "reservoir_name_zh": "大藤峡水利枢纽",
            "gate": "post_development_activation",
            "status": "RESOLVED_EXTENSION_ONLY",
            "operator_start": "2020-03-11",
            "fit_eligible_2010_2018": False,
            "extension_operator_eligible": True,
            "model_action": "hierarchical_prior_only_no_reservoir_specific_fit",
            "numeric_factor": 1.0,
        },
        {
            "reservoir_entity_id": "GRAND_5758",
            "reservoir_name_zh": "白盆珠水库",
            "gate": "reach17_interior_dam",
            "status": "RESOLVED_IMPLEMENTATION_AND_EVIDENCE",
            "operator_start": "before_2006",
            "fit_eligible_2010_2018": bool(partial_router_ready),
            "extension_operator_eligible": bool(partial_router_ready),
            "model_action": "capture_all_upstream_plus_fraction_of_reach17_local",
            "numeric_factor": baipenzhu_fraction,
        },
        {
            "reservoir_entity_id": "GRAND_5703",
            "reservoir_name_zh": "龟石水库",
            "gate": "partial_modeled_control_area",
            "status": "RESOLVED_WITH_DOMAIN_FRACTION_SCALING",
            "operator_start": "before_2006",
            "fit_eligible_2010_2018": True,
            "extension_operator_eligible": True,
            "model_action": "scale_storage_zone_priors_no_added_water",
            "numeric_factor": guishi_fraction,
        },
        {
            "reservoir_entity_id": "GRAND_5720",
            "reservoir_name_zh": "澄碧河水库",
            "gate": "partial_modeled_control_area",
            "status": "RESOLVED_WITH_UNCERTAIN_DOMAIN_FRACTION_SCALING",
            "operator_start": "before_2006",
            "fit_eligible_2010_2018": True,
            "extension_operator_eligible": True,
            "model_action": "scale_storage_zone_priors_and_run_1800_2100_km2_sensitivity",
            "numeric_factor": chengbihe_fraction,
        },
        {
            "reservoir_entity_id": "GRAND_5718",
            "reservoir_name_zh": "岩滩水库",
            "gate": "shared_two_arm_physical_evidence",
            "status": "RESOLVED_BY_NAMED_NETWORK_ARMS_DAM_GEOMETRY_AND_AREA",
            "operator_start": "before_2006",
            "fit_eligible_2010_2018": True,
            "extension_operator_eligible": True,
            "model_action": "one_shared_storage_two_inflow_arms_one_release",
            "numeric_factor": 1.0,
        },
    ]
    resolution = pd.DataFrame(rows)
    if not 0 < baipenzhu_fraction < 1:
        raise RuntimeError("Baipenzhu local-capture fraction is outside (0, 1)")
    if not 0 < guishi_fraction <= 1:
        raise RuntimeError("Guishi represented-domain fraction is outside (0, 1]")
    if not 0 < chengbihe_fraction <= 1:
        raise RuntimeError("Chengbihe represented-domain fraction is outside (0, 1]")

    output = STAGE / "outputs/reservoir_hard_gate_resolution.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    resolution.to_parquet(output, index=False)

    unresolved = resolution.loc[
        resolution["status"].astype(str).str.contains("BLOCKED|PENDING")
    ]
    report = {
        "stage": "20260828_17",
        "status": (
            "HARD_GATES_RESOLVED_AUTHORITY_SYNC_REQUIRED"
            if len(unresolved) == 0
            else "HARD_GATE_PROGRESS_FORMAL_FIT_STILL_CLOSED"
        ),
        "resolved_gate_count": int(len(resolution) - len(unresolved)),
        "unresolved_gate_count": int(len(unresolved)),
        "unresolved_entities": unresolved["reservoir_entity_id"].tolist(),
        "baipenzhu_local_capture_fraction": baipenzhu_fraction,
        "guishi_represented_domain_fraction": guishi_fraction,
        "chengbihe_represented_domain_fraction": chengbihe_fraction,
        "partial_router_tests_pass": bool(partial_router_ready),
        "disabled_parent_bitwise_pass": bool(parent_test.get("status") == "PASS"),
        "formal_fit_authorized": False,
        "hashes": {
            "evidence": sha256(EVIDENCE),
            "topology": sha256(TOPOLOGY),
            "router": sha256(ROUTER),
            "router_tests": sha256(ROUTER_TESTS),
            "parent_test": sha256(PARENT_TEST),
            "q72_parent_area_source": sha256(Q72_PARENT_AREA),
            "output": sha256(output),
        },
        "next_action": (
            "Update the authoritative registry and formal contract, then run the pre-fit integration smoke test."
            if len(unresolved) == 0
            else "Resolve remaining hard gates before formal fitting."
        ),
    }
    report_path = STAGE / "reports/hard_gate_resolution.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
