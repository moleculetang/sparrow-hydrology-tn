from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


RUN = Path(__file__).resolve().parents[1]
REPORT = RUN / "reports" / "et_evidence_audit"
OUTPUT = RUN / "outputs" / "et_evidence_comparison.parquet"
MANIFEST = RUN / "inputs_manifest" / "provenance_manifest.json"
VALIDATION = REPORT / "validation.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported JSON type: {type(value).__name__}")


def main() -> None:
    gate = json.loads((REPORT / "gate.json").read_text(encoding="utf-8"))
    data = pd.read_parquet(OUTPUT)
    pairwise = pd.read_csv(
        REPORT / "pairwise_et_metrics.csv", encoding="utf-8-sig"
    )
    reach = pd.read_csv(
        REPORT / "reach_pairwise_et_metrics.csv",
        encoding="utf-8-sig",
    )
    source = pd.read_csv(
        REPORT / "et_source_summary.csv", encoding="utf-8-sig"
    )
    gap = pd.read_csv(
        REPORT / "gap_decomposition.csv", encoding="utf-8-sig"
    )
    unit = json.loads(
        (REPORT / "unit_semantic_audit.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    product_hashes_valid = True
    for item in manifest["products"]:
        path = Path(item["path"])
        product_hashes_valid &= (
            path.exists()
            and path.stat().st_size == item["bytes"]
            and sha256(path) == item["sha256"]
        )

    pairs = set(pairwise["pair"])
    model_pml = pairwise.set_index("pair").loc["model_vs_pml"]
    era5_pml = pairwise.set_index("pair").loc["era5_vs_pml"]
    checks = {
        "output_rows_35880": len(data) == 35880,
        "output_reaches_230": data["reach_id"].nunique() == 230,
        "output_months_156": data[["year", "month"]].drop_duplicates().shape[0]
        == 156,
        "output_keys_unique": not data.duplicated(
            ["reach_id", "year", "month"]
        ).any(),
        "output_values_finite": bool(
            np.isfinite(
                data[
                    [
                        "AET_candidate_mm",
                        "AET_diagnostic_mm",
                        "pml_aet_mm",
                        "PET_mm",
                        "P_mm",
                    ]
                ].to_numpy(float)
            ).all()
        ),
        "three_pairwise_comparisons": pairs
        == {"model_vs_era5", "model_vs_pml", "era5_vs_pml"},
        "reach_pairwise_rows_690": len(reach) == 690,
        "source_summary_rows_3": len(source) == 3,
        "gap_decomposition_closes": abs(
            gap.set_index("component").loc[
                "era5_minus_pml_product_disagreement",
                "area_weighted_mm_month",
            ]
            + gap.set_index("component").loc[
                "pml_minus_model_residual_gap",
                "area_weighted_mm_month",
            ]
            - gap.set_index("component").loc[
                "original_era5_minus_model_gap",
                "area_weighted_mm_month",
            ]
        )
        <= 1e-10,
        "era5_unit_reconstruction_supported": (
            unit["era5_land"]["maximum_reconstruction_error_mm"] <= 1e-5
        ),
        "model_pml_original_gate_passes": (
            abs(model_pml["domain_volume_relative_bias"]) <= 0.2
            and model_pml["reach_absolute_bias_median"] <= 0.3
            and model_pml[
                "reach_fraction_absolute_bias_le_40pct"
            ]
            >= 0.7
            and model_pml["monthly_correlation_median"] >= 0.6
            and model_pml["climatology_correlation_median"] >= 0.8
        ),
        "era5_pml_total_disagreement_material": abs(
            era5_pml["domain_volume_relative_bias"]
        )
        >= 0.2,
        "gate_decision_exact": gate["decision"]
        == "PML_PRIMARY_REFERENCE_MODEL_AET_ACCEPTED",
        "next_action_exact": gate["authorized_next_action"]
        == "TEST_Q78_NAT_GROUNDWATER_BASEFLOW_SIGNATURE_WITH_PML_AET_REFERENCE",
        "restricted_inputs_not_read": (
            not gate["station_observations_read"]
            and not gate["management_fluxes_read"]
            and not gate["period_2019_2022_read"]
        ),
        "q78_full_gate_not_overclaimed": not gate[
            "q78_full_physical_gate_passed"
        ],
        "manifest_product_hashes_valid": bool(product_hashes_valid),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    passed = bool(all(checks.values()))
    payload = {
        "run_id": RUN.name,
        "checks": checks,
        "checks_passed": int(sum(checks.values())),
        "checks_total": len(checks),
        "scientific_gate_decision": gate["decision"],
        "passed": passed,
    }
    VALIDATION.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            default=json_default,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            default=json_default,
        )
    )
    if not passed:
        raise SystemExit(4)


if __name__ == "__main__":
    main()
