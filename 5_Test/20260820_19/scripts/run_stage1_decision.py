from __future__ import annotations

import json

import pandas as pd

import hierarchical19_shared as h


def main() -> None:
    h.require_runtime()
    gates = pd.read_parquet(h.OUT / "temporal_gate_matrix.parquet")
    params = pd.read_parquet(h.OUT / "temporal_fold_parameters.parquet")

    def count(mechanism: str, block: str, column: str) -> int:
        rows = gates.loc[
            gates.layer.eq("P1") & gates.mechanism.eq(mechanism) & gates.block.eq(block)
        ]
        return int(rows[column].sum())

    candidates = {}
    for mechanism in ("H1_GLOBAL", "H2_COVARIATE", "H3_TREE_PARTIAL_POOL"):
        values = {
            "station_noninferior": count(mechanism, "station", "noninferior"),
            "tree_noninferior": count(mechanism, "tree", "noninferior"),
            "station_improved": count(mechanism, "station", "predictively_improved"),
            "tree_improved": count(mechanism, "tree", "predictively_improved"),
        }
        values["temporal_process_gate_pass"] = bool(
            values["station_noninferior"] >= 10
            and values["tree_noninferior"] >= 10
            and (values["station_improved"] >= 8 or values["tree_improved"] >= 8)
        )
        candidates[mechanism] = values

    h1_success = bool(params.loc[params.structure.eq("H1_GLOBAL"), "outer_success"].all())
    admitted = "H1_GLOBAL" if candidates["H1_GLOBAL"]["temporal_process_gate_pass"] and h1_success else "H0_PARENT"
    decision = {
        "status": "PASS" if admitted == "H1_GLOBAL" else "STOP_NO_HYDRAULIC_STRUCTURE_ADMITTED",
        "candidate_counts": candidates,
        "registered_parsimony_decision": admitted,
        "interpretation": (
            "H1 global hydraulic exposure admitted; H2/H3 rejected before spatial evaluation because their P1 tree-block evidence failed. "
            "Hierarchical spatial pooling proceeds only in the P2R readout."
        ),
        "nested_LOSO_LOTO_authorized": bool(admitted == "H1_GLOBAL"),
        "temperature_used": False,
        "TN_2022_read": False,
    }
    h.dump_json(h.REPORTS / "stage1_structure_decision.json", decision)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
