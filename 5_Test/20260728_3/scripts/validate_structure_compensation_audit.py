from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


RUN = Path(__file__).resolve().parents[1]
SOURCE = RUN / "reports" / "structure_compensation"
EXPECTED_CANDIDATES = {"20260727_13", "20260727_15"}
EXPECTED_PERIODS = {
    "fit_2006_2015",
    "selection_2016_2018",
    "development_2006_2018",
}
ALLOWED_CLASSIFICATIONS = {
    "weak_structural_leverage",
    "compensation_dominated",
    "structural_effect_retained",
    "mixed_or_uncertain",
}


def all_true(frame: pd.DataFrame, columns: list[str]) -> bool:
    return bool(
        all(
            frame[column]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin({"true", "1", "yes"})
            .all()
            for column in columns
        )
    )


def main() -> None:
    manifest = json.loads(
        (
            RUN
            / "inputs_manifest"
            / "compensation_source_manifest.json"
        ).read_text(encoding="utf-8")
    )
    summary = pd.read_csv(
        SOURCE / "compensation_summary.csv",
        encoding="utf-8-sig",
    )
    verification = pd.read_csv(
        SOURCE / "fit_reconstruction_verification.csv",
        encoding="utf-8-sig",
    )
    integrity = pd.read_csv(
        SOURCE / "audit_integrity.csv",
        encoding="utf-8-sig",
    )
    predictions = pd.read_csv(
        SOURCE / "compensation_predictions_2006_2018.csv",
        encoding="utf-8-sig",
    )
    changed = pd.read_csv(
        SOURCE / "changed_design_columns.csv",
        encoding="utf-8-sig",
    )

    gates = {
        "source_inputs_identical": bool(
            manifest["all_input_hashes_identical"]
        ),
        "reconstructed_runs_exact": bool(
            len(verification) == 3
            and verification["run_id"].astype(str).isin(
                {"20260727_6", *EXPECTED_CANDIDATES}
            ).all()
            and all_true(verification, ["key_match", "passed"])
            and verification["max_abs_eta_difference"].le(1.0e-9).all()
        ),
        "candidate_and_period_coverage": bool(
            set(summary["candidate_run"].astype(str))
            == EXPECTED_CANDIDATES
            and set(summary["period"].astype(str)) == EXPECTED_PERIODS
            and len(summary) == 6
        ),
        "audit_integrity": bool(
            len(integrity) == 2
            and set(integrity["candidate_run"].astype(str))
            == EXPECTED_CANDIDATES
            and all_true(
                integrity,
                [
                    "keys_match_baseline",
                    "feature_lists_match_baseline",
                    "stations_match_baseline",
                    "matrix_shape_match",
                    "unchanged_coefficients_exact_in_partial",
                    "partial_coefficients_finite",
                    "predictions_finite",
                ],
            )
            and integrity["changed_design_columns"].gt(0).all()
            and integrity["maximum_year_used"].eq(2018).all()
        ),
        "prediction_key_unique": bool(
            not predictions.duplicated(
                ["candidate_run", "q_site", "year", "month"]
            ).any()
        ),
        "no_2019_2022_rows": bool(
            predictions["year"].min() == 2006
            and predictions["year"].max() == 2018
        ),
        "prediction_values_finite": bool(
            np.isfinite(
                predictions[
                    [
                        "eta_baseline",
                        "eta_frozen_coefficients",
                        "eta_partial_refit",
                        "eta_full_refit",
                    ]
                ].to_numpy(dtype=float)
            ).all()
        ),
        "changed_column_index_complete": bool(
            len(changed) > 0
            and set(changed["candidate_run"].astype(str))
            == EXPECTED_CANDIDATES
            and changed.groupby("candidate_run")[
                "column_index"
            ].nunique().nunique()
            == 1
        ),
        "classification_complete": bool(
            summary["classification"].notna().all()
            and set(summary["classification"].astype(str)).issubset(
                ALLOWED_CLASSIFICATIONS
            )
        ),
    }
    passed = bool(all(gates.values()))
    development = summary.loc[
        summary["period"].eq("development_2006_2018")
    ].copy()
    result = {
        "run_id": RUN.name,
        "phase_id": "structure_compensation_audit",
        "reference_run": "20260727_6",
        "candidate_runs": sorted(EXPECTED_CANDIDATES),
        "diagnostic_only": True,
        "promotion_permitted": False,
        "gates": gates,
        "gate_passed": passed,
        "decision": (
            "pass_to_next_phase"
            if passed
            else "stop_keep_20260727_6"
        ),
        "development_2006_2018": development.to_dict("records"),
    }
    (RUN / "reports" / "gate.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        f"# {RUN.name} Structure-Compensation Gate",
        "",
        f"- reference: 20260727_6",
        f"- diagnostic only: True",
        f"- promotion permitted: False",
    ]
    lines.extend(f"- {name}: {value}" for name, value in gates.items())
    lines.extend(
        [
            f"- final gate: {'PASS' if passed else 'FAIL'}",
            f"- decision: {result['decision']}",
            "",
            "## Development-period interpretation",
            "",
            "| candidate | leverage | partial CI | full CI | classification |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )
    for row in development.to_dict("records"):
        lines.append(
            f"| {row['candidate']} | "
            f"{row['structure_leverage_median_abs_delta_logq']:.6f} | "
            f"{row['partial_compensation_index']:.6f} | "
            f"{row['full_compensation_index']:.6f} | "
            f"{row['classification']} |"
        )
    text = "\n".join(lines) + "\n"
    (RUN / "reports" / "gate.md").write_text(text, encoding="utf-8")
    print(text)
    if not passed:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
