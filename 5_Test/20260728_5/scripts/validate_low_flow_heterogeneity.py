from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


RUN = Path(__file__).resolve().parents[1]
OUT = RUN / "reports" / "low_flow_heterogeneity"
REPORTS = RUN / "reports"
EXPECTED_FOLDS = {
    "fit_2006_2011_eval_2012_2013",
    "fit_2006_2013_eval_2014_2015",
    "fit_2006_2015_eval_2016_2018",
}
EXCLUSIONS = {"劳村站", "富罗（二）站", "隆安站"}
PROTECTED = "石角站"
STATIC_MODELS = {
    "deployable_static",
    "static_plus_location_sensitivity",
    "static_plus_gauged_signature_upper_bound",
}


def truth(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().eq("true")


def main() -> int:
    fold = pd.read_csv(
        OUT / "station_fold_low_flow_bias.csv",
        encoding="utf-8-sig",
    )
    stations = pd.read_csv(
        OUT / "station_heterogeneity_classification.csv",
        encoding="utf-8-sig",
    )
    models = pd.read_csv(
        OUT / "separability_model_summary.csv",
        encoding="utf-8-sig",
    )
    predictions = pd.read_csv(
        OUT / "separability_oof_predictions.csv",
        encoding="utf-8-sig",
    )
    coefficients = pd.read_csv(
        OUT / "separability_full_fit_coefficients.csv",
        encoding="utf-8-sig",
    )
    permutations = pd.read_csv(
        OUT / "separability_permutation_null.csv",
        encoding="utf-8-sig",
    )
    scientific = json.loads(
        (OUT / "scientific_gate.json").read_text(encoding="utf-8")
    )
    summary = json.loads((OUT / "analysis_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (RUN / "inputs_manifest" / "source_file_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    predecessor = json.loads(
        (RUN.parent / "20260728_4" / "reports" / "gate.json").read_text(
            encoding="utf-8"
        )
    )

    reservoir = truth(stations["reservoir_related_bool"])
    quality = truth(stations["data_quality_suspicious"])
    stable = truth(stations["stable_low_flow_target"])
    eligible = truth(stations["eligible_for_separability"])
    model_completed = models["status"].eq("completed")
    completed_models = set(models.loc[model_completed, "model_id"])

    stable_recomputed = (
        truth(stations["eligible_for_heterogeneity_label"])
        & stations["positive_fold_fraction"].ge(2.0 / 3.0)
        & stations["median_low_flow_bias_pct"].ge(10.0)
        & stations["material_negative_fold_fraction"].eq(0)
        & ~reservoir
        & ~quality
    )
    model_prediction_coverage = True
    permutation_coverage = True
    for row in models[model_completed].itertuples(index=False):
        model_predictions = predictions[predictions["model_id"].eq(row.model_id)]
        model_permutations = permutations[permutations["model_id"].eq(row.model_id)]
        model_prediction_coverage &= bool(
            len(model_predictions) == int(row.rows)
            and model_predictions["q_site"].nunique() == int(row.rows)
            and model_predictions["oof_probability"].between(0, 1).all()
        )
        permutation_coverage &= bool(
            len(model_permutations) == int(row.permutations)
            and model_permutations["permutation"].nunique()
            == int(row.permutations)
            and model_permutations["roc_auc"].between(0, 1).all()
        )

    candidate_attribute_columns = [
        "log_inc_area_km2",
        "log_tot_area_km2",
        "log_length_km",
        "local_area_fraction",
        "upstream_count",
        "headwater",
        "terminal",
        "frac",
        "climate_ppt_mean",
        "climate_pet_mean",
        "climate_aet_mean",
        "climate_aridity",
        "climate_surplus_ratio",
        "climate_ppt_cv",
        "climate_wetness_mean",
        "reach_class",
    ]
    gates = {
        "source_manifest_complete": bool(
            manifest.get("all_present")
            and int(manifest.get("files", 0)) == 8
        ),
        "predecessor_integrity_gate_passed": bool(predecessor.get("gate_passed")),
        "three_strict_oof_folds_present": set(fold["fold_id"]) == EXPECTED_FOLDS,
        "oof_fold_station_key_unique": not bool(
            fold.duplicated(["fold_id", "q_site"]).any()
        ),
        "no_2019_2022_evidence": bool(
            int(summary["maximum_oof_year"]) <= 2018
            and int(summary["maximum_signature_year"]) <= 2011
            and int(summary["maximum_climate_attribute_year"]) <= 2011
        ),
        "fold_biases_finite": bool(
            np.isfinite(
                fold[
                    ["low_flow_volume_bias_pct", "low_flow_log_rmse"]
                ].to_numpy(dtype=float)
            ).all()
        ),
        "station_classification_unique": bool(
            stations["q_site"].is_unique
            and stations["reach_id"].notna().all()
        ),
        "stable_target_definition_exact": bool(
            (stable.to_numpy() == stable_recomputed.to_numpy()).all()
        ),
        "all_required_heterogeneity_classes_reported": set(
            pd.read_csv(
                OUT / "heterogeneity_class_summary.csv",
                encoding="utf-8-sig",
            )["heterogeneity_class"]
        )
        == {
            "reservoir_or_regulation_possible",
            "data_quality_suspicious",
            "stable_low_flow_overprediction",
            "material_sign_reversal",
            "episodic_low_flow_overprediction",
            "not_stably_overpredicted",
        },
        "excluded_stations_absent": not bool(stations["q_site"].isin(EXCLUSIONS).any()),
        "protected_station_present": bool(stations["q_site"].eq(PROTECTED).any()),
        "reservoir_rows_reported_not_targeted": bool(
            reservoir.any()
            and not stable[reservoir].any()
            and not eligible[reservoir].any()
        ),
        "quality_rows_reported_not_targeted": bool(
            quality.any()
            and not stable[quality].any()
            and not eligible[quality].any()
        ),
        "deployable_attributes_complete_for_eligible_rows": not bool(
            stations.loc[eligible, candidate_attribute_columns].isna().any().any()
        ),
        "three_preregistered_models_accounted_for": set(models["model_id"])
        == STATIC_MODELS,
        "completed_model_oof_coverage_complete": bool(model_prediction_coverage),
        "permutation_null_complete": bool(permutation_coverage),
        "no_station_identity_in_model_features": not bool(
            coefficients["feature"]
            .astype(str)
            .str.contains("q_site|station|reach_id", case=False, regex=True)
            .any()
        ),
        "static_gate_uses_deployable_model_only": bool(
            scientific["deployable_static_attribute_gate"]["metrics"][
                "model_id"
            ]
            == "deployable_static"
            and scientific["location_sensitivity_can_promote"] is False
            and scientific["gauged_signature_upper_bound"][
                "deployable_to_ungauged_reaches"
            ]
            is False
        ),
        "no_formal_model_or_prediction_rewrite": not any(
            name in stations.columns
            for name in ["Q_new_cfs", "Q_corrected_cfs", "gate_weight"]
        ),
    }
    passed = bool(all(gates.values()))
    gate_payload = {
        "run_id": RUN.name,
        "phase_id": "low_flow_heterogeneity_and_attribute_separability",
        "logical_parent_run": "20260727_6",
        "diagnostic_predecessor": "20260728_4",
        "diagnostic_only": True,
        "promotion_permitted": False,
        "gates": gates,
        "gate_passed": passed,
        "scientific_result": {
            "stable_low_flow_target_supported": scientific["stable_signal"][
                "supported"
            ],
            "deployable_static_attribute_gate_supported": scientific[
                "deployable_static_attribute_gate"
            ]["supported"],
            "gauged_signature_upper_bound_supported": scientific[
                "gauged_signature_upper_bound"
            ]["supported"],
            "decision": scientific["decision"],
        },
        "decision": "pass_to_next_phase" if passed else "stop_for_audit_failure",
        "validated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    (REPORTS / "gate.json").write_text(
        json.dumps(gate_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        f"# {RUN.name} Gate",
        "",
        f"- hard QA: **{'PASS' if passed else 'FAIL'}**",
        f"- stable low-flow target supported: **{gate_payload['scientific_result']['stable_low_flow_target_supported']}**",
        f"- deployable static gate supported: **{gate_payload['scientific_result']['deployable_static_attribute_gate_supported']}**",
        f"- gauged signature upper bound supported: **{gate_payload['scientific_result']['gauged_signature_upper_bound_supported']}**",
        "",
        "## Hard QA",
        "",
    ]
    lines.extend(
        f"- {'PASS' if value else 'FAIL'}: `{name}`"
        for name, value in gates.items()
    )
    lines.extend(
        [
            "",
            "## Separability models",
            "",
            models.to_markdown(index=False),
            "",
        ]
    )
    (REPORTS / "gate.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(gate_payload, ensure_ascii=False, indent=2))
    return 0 if passed else 4


if __name__ == "__main__":
    raise SystemExit(main())
