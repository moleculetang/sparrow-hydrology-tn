from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


RUN = Path(__file__).resolve().parents[1]
REPORTS = RUN / "reports"
OUT = REPORTS / "q72_q78_complementarity"
EXPECTED_FOLDS = {
    "fit_2006_2011_eval_2012_2013": (2011, 2012, 2013),
    "fit_2006_2013_eval_2014_2015": (2013, 2014, 2015),
    "fit_2006_2015_eval_2016_2018": (2015, 2016, 2018),
}
EXCLUSIONS = {"劳村站", "富罗（二）站", "隆安站"}
PROTECTED = "石角站"


def markdown_bool(value: bool) -> str:
    return "PASS" if value else "FAIL"


def main() -> int:
    oof = pd.read_csv(OUT / "oof_predictions_2012_2018.csv", encoding="utf-8-sig")
    fit = pd.read_csv(OUT / "fold_fit_manifest.csv", encoding="utf-8-sig")
    coefficients = pd.read_csv(OUT / "q78_fold_coefficients.csv", encoding="utf-8-sig")
    mass = pd.read_csv(OUT / "mass_balance_audit.csv", encoding="utf-8-sig")
    thresholds = pd.read_csv(OUT / "training_flow_thresholds.csv", encoding="utf-8-sig")
    cells = pd.read_csv(OUT / "station_regime_fold_oracle.csv", encoding="utf-8-sig")
    eligibility = pd.read_csv(OUT / "conditional_fusion_eligibility.csv", encoding="utf-8-sig")
    scientific = json.loads(
        (OUT / "conditional_fusion_eligibility.json").read_text(encoding="utf-8")
    )
    summary = json.loads((OUT / "analysis_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (RUN / "inputs_manifest" / "source_file_manifest.json").read_text(
            encoding="utf-8"
        )
    )

    key = ["fold_id", "q_site", "reach_id", "year", "month"]
    fold_boundaries_ok = True
    for fold_id, (train_end, eval_start, eval_end) in EXPECTED_FOLDS.items():
        fit_part = fit[fit["fold_id"].eq(fold_id)]
        oof_part = oof[oof["fold_id"].eq(fold_id)]
        threshold_part = thresholds[thresholds["fold_id"].eq(fold_id)]
        fold_boundaries_ok &= bool(
            len(fit_part) == 1
            and int(fit_part["train_end"].iloc[0]) == train_end
            and int(fit_part["training_year_max"].iloc[0]) <= train_end
            and train_end < eval_start
            and int(oof_part["year"].min()) == eval_start
            and int(oof_part["year"].max()) == eval_end
            and int(threshold_part["train_end"].max()) == train_end
        )

    finite_columns = [
        "Q_obsv_cfs",
        "Q72_pred_cfs",
        "Q78_mass_cfs",
        "Q_original_fusion_cfs",
    ]
    finite_predictions = bool(
        np.isfinite(oof[finite_columns].to_numpy(dtype=float)).all()
    )
    nonnegative_predictions = bool(
        (oof[["Q72_pred_cfs", "Q78_mass_cfs", "Q_original_fusion_cfs"]] >= 0)
        .all()
        .all()
    )
    reservoir_cells_in_gate = cells[
        cells["reservoir_flag"].astype(str).str.lower().eq("true")
        & cells["qualified_cell"].astype(str).str.lower().eq("true")
    ]
    gates = {
        "source_manifest_complete": bool(
            manifest.get("all_present")
            and int(manifest.get("files", 0)) == 12
        ),
        "three_folds_complete": set(fit["fold_id"]) == set(EXPECTED_FOLDS)
        and set(oof["fold_id"]) == set(EXPECTED_FOLDS)
        and set(mass["fold_id"]) == set(EXPECTED_FOLDS),
        "strict_fold_time_boundaries": fold_boundaries_ok,
        "no_2019_2022_evidence_rows": bool(
            int(oof["year"].max()) <= 2018
            and int(fit["training_year_max"].max()) <= 2015
            and int(mass["routed_year_max"].max()) <= 2018
        ),
        "oof_prediction_key_unique": not bool(oof.duplicated(key).any()),
        "oof_predictions_complete": not bool(
            oof[finite_columns + ["reach_class", "flow_regime", "class_alpha"]]
            .isna()
            .any()
            .any()
        ),
        "predictions_finite": finite_predictions,
        "predictions_nonnegative": nonnegative_predictions,
        "q78_coefficients_nonnegative": bool(
            (coefficients["coefficient"] >= -1.0e-15).all()
        ),
        "q78_optimizer_succeeded": bool(
            fit["optimizer_success"]
            .astype(str)
            .str.lower()
            .eq("true")
            .all()
        ),
        "mass_balance_residual_at_most_1e_9": bool(
            mass["max_abs_mass_balance_residual_cfs"].max() <= 1.0e-9
        ),
        "excluded_stations_absent": not bool(oof["q_site"].isin(EXCLUSIONS).any()),
        "protected_station_present_in_every_fold": bool(
            oof[oof["q_site"].eq(PROTECTED)]["fold_id"].nunique()
            == len(EXPECTED_FOLDS)
        ),
        "reservoir_stations_reported": bool(oof["reservoir_flag"].astype(bool).any()),
        "reservoir_stations_excluded_from_scientific_gate": bool(
            len(reservoir_cells_in_gate) > 0
        ),
        "training_frozen_regime_thresholds_complete": bool(
            not thresholds[
                ["training_q25_cfs", "training_q75_cfs", "training_months"]
            ]
            .isna()
            .any()
            .any()
            and (thresholds["training_q25_cfs"] <= thresholds["training_q75_cfs"])
            .all()
        ),
        "all_regimes_audited": set(eligibility["flow_regime"])
        == {"all", "low", "middle", "high"},
        "oracle_alpha_grid_respected": bool(
            cells["oracle_alpha"].between(0.0, 1.0).all()
            and np.allclose(cells["oracle_alpha"] * 100, np.round(cells["oracle_alpha"] * 100))
        ),
        "oracle_weights_not_promoted": bool(
            scientific.get("oracle_weights_deployable") is False
            and summary.get("oracle_weights_deployable") is False
        ),
    }
    # Eligibility is recomputed only from non-reservoir cells in the build
    # script. Verify its station counts directly against those cells.
    reservoir_exclusion_verified = True
    for row in eligibility.itertuples(index=False):
        expected = (
            cells[
                cells["flow_regime"].eq(row.flow_regime)
                & ~cells["reservoir_flag"].astype(str).str.lower().eq("true")
                & cells["qualified_cell"].astype(str).str.lower().eq("true")
            ]["q_site"]
            .nunique()
        )
        reservoir_exclusion_verified &= int(row.eligible_stations) == int(expected)
    gates["reservoir_stations_excluded_from_scientific_gate"] = bool(
        reservoir_exclusion_verified
    )

    passed = bool(all(gates.values()))
    gate_payload = {
        "run_id": RUN.name,
        "phase_id": "q72_q78_oof_complementarity",
        "reference_run": "20260727_6",
        "diagnostic_only": True,
        "promotion_permitted": False,
        "gates": gates,
        "gate_passed": passed,
        "scientific_result": {
            "conditional_fusion_upper_bound_supported": scientific[
                "conditional_fusion_upper_bound_supported"
            ],
            "eligible_regimes": scientific["eligible_regimes"],
            "decision": scientific["decision"],
            "oracle_weights_deployable": False,
        },
        "decision": "pass_to_next_phase" if passed else "stop_for_audit_failure",
        "validated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    (REPORTS / "gate.json").write_text(
        json.dumps(gate_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    gate_lines = [
        f"# {RUN.name} Gate",
        "",
        f"- reference: `20260727_6`",
        f"- diagnostic integrity: **{'PASS' if passed else 'FAIL'}**",
        f"- scientific upper bound supported: **{scientific['conditional_fusion_upper_bound_supported']}**",
        f"- eligible regimes: `{', '.join(scientific['eligible_regimes']) or 'none'}`",
        "- oracle weights are diagnostic only and are not deployable.",
        "",
        "## Hard QA",
        "",
    ]
    for name, value in gates.items():
        gate_lines.append(f"- {markdown_bool(bool(value))}: `{name}`")
    gate_lines.extend(
        [
            "",
            "## Pre-registered scientific gate",
            "",
            eligibility.to_markdown(index=False),
            "",
        ]
    )
    (REPORTS / "gate.md").write_text("\n".join(gate_lines), encoding="utf-8")
    print(json.dumps(gate_payload, ensure_ascii=False, indent=2))
    return 0 if passed else 4


if __name__ == "__main__":
    raise SystemExit(main())
