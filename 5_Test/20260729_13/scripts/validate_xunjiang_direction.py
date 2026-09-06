from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


RUN_DIR = Path(__file__).resolve().parents[1]
ROOT = RUN_DIR.parents[1]
PARENT = ROOT / "5_Test" / "20260729_12"
REPORT = RUN_DIR / "reports" / "xunjiang_direction_gate"
MANIFEST = RUN_DIR / "inputs_manifest" / "provenance_manifest.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    gate = json.loads((REPORT / "gate.json").read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    current = pd.read_parquet(RUN_DIR / "inputs" / "reach_slope_direction_final.parquet")
    parent = pd.read_parquet(PARENT / "inputs" / "reach_slope_direction_adjudicated.parquet")
    static = pd.read_parquet(RUN_DIR / "inputs" / "reach_static_direction_final.parquet")
    named = pd.read_csv(REPORT / "named_confluence_evidence.csv", encoding="utf-8-sig")
    nodes = pd.read_csv(REPORT / "node_27_counterfactual.csv", encoding="utf-8-sig")
    areas = pd.read_csv(REPORT / "area_closure_counterfactual.csv", encoding="utf-8-sig")
    external = (REPORT / "external_web_cross_validation.md").read_text(encoding="utf-8")
    merged = current[["reach_id", "slope_m_m"]].merge(
        parent[["reach_id", "slope_m_m"]],
        on="reach_id",
        suffixes=("_current", "_parent"),
        validate="one_to_one",
    )
    target = current.loc[current["reach_id"].astype(int) == 31].iloc[0]
    product_hashes = all(
        Path(r["path"]).exists() and sha256(Path(r["path"])) == r["sha256"]
        for r in manifest["products"]
    )
    checks = {
        "required_products_exist": all(p.exists() for p in [
            REPORT / "gate.json",
            REPORT / "technical_report.md",
            REPORT / "named_confluence_evidence.csv",
            REPORT / "node_27_counterfactual.csv",
            REPORT / "area_closure_counterfactual.csv",
            REPORT / "external_web_cross_validation.md",
            REPORT / "run_manifest.json",
            RUN_DIR / "inputs" / "reach_slope_direction_final.csv",
            RUN_DIR / "inputs" / "reach_slope_direction_final.parquet",
            RUN_DIR / "inputs" / "reach_static_direction_final.parquet",
            MANIFEST,
        ]),
        "reach_counts_230": len(current) == len(parent) == len(static) == 230,
        "reach_ids_unique": current["reach_id"].is_unique and static["reach_id"].is_unique,
        "slope_values_bitwise_identical": np.array_equal(
            merged["slope_m_m_current"].to_numpy(),
            merged["slope_m_m_parent"].to_numpy(),
        ),
        "no_slope_change_flag": not current[
            "slope_value_changed_by_direction_audit"
        ].astype(bool).any(),
        "reach_31_gate_passes": bool(target["direction_gate_pass"]),
        "reach_31_direction_retained": target["recommended_model_direction"] == "retain_fnode_to_tnode",
        "named_confluence_matches": bool(named.loc[0, "named_confluence_match"]),
        "official_web_cross_validation_consistent": (
            gate["external_cross_validation"]["consistent_with_local_direction"]
            and "浔江、桂江相汇成西江" in external
            and "wuzhou.gov.cn" in gate["external_cross_validation"]["url"]
            and gate["external_cross_validation"]["decision_role"]
            == "non_decisive_external_cross_validation"
        ),
        "counterfactual_has_two_outgoing": int(
            nodes.loc[nodes["scenario"] == "reverse_reach_31", "outgoing_count"].iloc[0]
        ) == 2,
        "counterfactual_area_break_material": areas[
            "area_closure_residual_km2_alternative"
        ].abs().max() >= 1000.0,
        "gate_and_action_consistent": (
            gate["passed"]
            and gate["authorized_next_action"] == "BUILD_Q78_NAT_CONSERVATION_CORE"
        ),
        "product_hashes_match_manifest": product_hashes,
        "production_topology_unmodified": manifest["production_topology_mutated"] is False,
        "flow_acc_not_used_as_truth": manifest["flow_acc_used_as_direction_truth"] is False,
    }
    checks = {k: bool(v) for k, v in checks.items()}
    result = {
        "run_id": "20260729_13",
        "validated_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": "conda sparrow",
        "checks": checks,
        "passed_checks": sum(checks.values()),
        "total_checks": len(checks),
        "artifact_validation_passed": all(checks.values()),
        "scientific_gate_passed": bool(gate["passed"]),
    }
    (REPORT / "validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))
    if not result["artifact_validation_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
