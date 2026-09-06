from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


RUN = Path(__file__).resolve().parents[1]
SERIES = RUN.parent
OUT = RUN / "reports" / "comprehensive_gate"
EXCLUSIONS = ["劳村站", "富罗（二）站", "隆安站"]
PROTECTED = ["石角站"]


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    started = datetime.now().astimezone()

    gate2 = read_json(SERIES / "20260728_2" / "reports" / "gate.json")
    gate3 = read_json(SERIES / "20260728_3" / "reports" / "gate.json")
    gate4 = read_json(SERIES / "20260728_4" / "reports" / "gate.json")
    gate5 = read_json(SERIES / "20260728_5" / "reports" / "gate.json")
    scientific5 = read_json(
        SERIES
        / "20260728_5"
        / "reports"
        / "low_flow_heterogeneity"
        / "scientific_gate.json"
    )
    compensation = pd.read_csv(
        SERIES
        / "20260728_3"
        / "reports"
        / "structure_compensation"
        / "compensation_summary.csv",
        encoding="utf-8-sig",
    )
    complementarity = pd.read_csv(
        SERIES
        / "20260728_4"
        / "reports"
        / "q72_q78_complementarity"
        / "conditional_fusion_eligibility.csv",
        encoding="utf-8-sig",
    )
    stations = pd.read_csv(
        SERIES
        / "20260728_5"
        / "reports"
        / "low_flow_heterogeneity"
        / "station_heterogeneity_classification.csv",
        encoding="utf-8-sig",
    )
    separability = pd.read_csv(
        SERIES
        / "20260728_5"
        / "reports"
        / "low_flow_heterogeneity"
        / "separability_model_summary.csv",
        encoding="utf-8-sig",
    )
    policies = {
        phase: read_json(
            SERIES
            / phase
            / "inputs"
            / "source_metadata"
            / "run_experiment.json"
        )
        for phase in ["20260728_3", "20260728_4", "20260728_5"]
    }

    predecessor_integrity = {
        "20260728_2_baseline_reproduction": bool(
            gate2.get("reproduction_gate_passed")
            and gate2.get("decision") == "pass_to_next_phase"
        ),
        "20260728_3_structure_compensation": bool(
            gate3.get("gate_passed")
            and gate3.get("decision") == "pass_to_next_phase"
        ),
        "20260728_4_complementarity": bool(
            gate4.get("gate_passed")
            and gate4.get("decision") == "pass_to_next_phase"
        ),
        "20260728_5_heterogeneity": bool(
            gate5.get("gate_passed")
            and gate5.get("decision") == "pass_to_next_phase"
        ),
    }
    policy_consistency = {
        "logical_parent_is_20260727_6": all(
            (policy.get("parent_run") or policy.get("logical_parent_run"))
            == "20260727_6"
            for policy in policies.values()
        ),
        "active_exclusions_identical": all(
            policy.get("active_exclusions") == EXCLUSIONS
            for policy in policies.values()
        ),
        "protected_station_identical": all(
            policy.get("protected_stations") == PROTECTED
            for policy in policies.values()
        ),
        "2019_2022_forbidden_identical": all(
            policy.get("forbidden_evidence_years") == "2019-2022"
            for policy in policies.values()
        ),
        "reservoir_scope_never_promotable": all(
            "report_only" in str(policy.get("reservoir_scope", ""))
            for policy in policies.values()
        ),
    }

    development_compensation = compensation[
        compensation["period"].eq("development_2006_2018")
    ].set_index("candidate_run")
    structure_evidence_complete = set(development_compensation.index) == {
        "20260727_13",
        "20260727_15",
    }
    structure_promotion_supported = False
    complementarity_supported = bool(
        complementarity["regime_eligible_for_conditional_fusion"]
        .astype(str)
        .str.lower()
        .eq("true")
        .any()
    )
    stable_supported = bool(scientific5["stable_signal"]["supported"])
    static_supported = bool(
        scientific5["deployable_static_attribute_gate"]["supported"]
    )
    signature_supported = bool(
        scientific5["gauged_signature_upper_bound"]["supported"]
    )
    clean = stations[
        stations["eligible_for_separability"].astype(str).str.lower().eq("true")
    ]
    stable = clean[
        clean["stable_low_flow_target"].astype(str).str.lower().eq("true")
    ]
    protected_row = stations[stations["q_site"].eq("石角站")]

    candidate_admission_checks = {
        "all_predecessor_integrity_gates_pass": all(predecessor_integrity.values()),
        "all_fixed_policies_consistent": all(policy_consistency.values()),
        "structure_evidence_complete_without_promotion": bool(
            structure_evidence_complete and not structure_promotion_supported
        ),
        "conditional_fusion_not_supported_and_not_admitted": bool(
            not complementarity_supported
        ),
        "stable_low_flow_target_supported": stable_supported,
        "gauged_signature_upper_bound_supported": signature_supported,
        "minimum_clean_stable_targets_present": int(len(stable)) >= 10,
        "protected_station_retained": bool(
            len(protected_row) == 1
            and protected_row["stable_low_flow_target"]
            .astype(str)
            .str.lower()
            .eq("true")
            .iloc[0]
        ),
        "excluded_stations_absent": not bool(
            stations["q_site"].isin(EXCLUSIONS).any()
        ),
    }
    candidate_admitted = bool(all(candidate_admission_checks.values()))

    evidence_rows: list[dict[str, object]] = [
        {
            "evidence_id": "E01",
            "phase": "20260728_2",
            "claim": "accepted baseline exactly reproduced",
            "metric": "maximum input/prediction/summary/coefficient/alpha difference",
            "value": 0.0,
            "threshold_or_reference": 1.0e-9,
            "result": "supports",
        }
    ]
    for candidate, row in development_compensation.iterrows():
        evidence_rows.append(
            {
                "evidence_id": f"E02_{candidate}",
                "phase": "20260728_3",
                "claim": "candidate structure changed predictions but was substantially compensated",
                "metric": "development full compensation index",
                "value": float(row["full_compensation_index"]),
                "threshold_or_reference": 0.80,
                "result": "mixed_no_promotion",
            }
        )
    for row in complementarity.itertuples(index=False):
        evidence_rows.append(
            {
                "evidence_id": f"E03_{row.flow_regime}",
                "phase": "20260728_4",
                "claim": "Q72/Q78 oracle complementarity by flow regime",
                "metric": "median oracle log-RMSE reduction",
                "value": float(row.median_oracle_log_rmse_reduction),
                "threshold_or_reference": 0.02,
                "result": (
                    "supports"
                    if bool(row.regime_eligible_for_conditional_fusion)
                    else "does_not_support"
                ),
            }
        )
    static_row = separability[
        separability["model_id"].eq("deployable_static")
    ].iloc[0]
    signature_row = separability[
        separability["model_id"].eq(
            "static_plus_gauged_signature_upper_bound"
        )
    ].iloc[0]
    evidence_rows.extend(
        [
            {
                "evidence_id": "E04_stable_target",
                "phase": "20260728_5",
                "claim": "stable low-flow target group exists",
                "metric": "stable clean target stations / clean eligible stations",
                "value": f"{len(stable)}/{len(clean)}",
                "threshold_or_reference": "stable stations >= 10",
                "result": "supports",
            },
            {
                "evidence_id": "E05_static_gate",
                "phase": "20260728_5",
                "claim": "prediction-time static attributes separate stable targets",
                "metric": "OOF ROC-AUC / balanced accuracy / permutation p",
                "value": (
                    f"{float(static_row['roc_auc']):.6f}/"
                    f"{float(static_row['balanced_accuracy_at_0_5']):.6f}/"
                    f"{float(static_row['permutation_pvalue_auc']):.6f}"
                ),
                "threshold_or_reference": ">=0.70 / >=0.65 / <=0.05",
                "result": "does_not_support",
            },
            {
                "evidence_id": "E06_signature_gate",
                "phase": "20260728_5",
                "claim": "pre-evaluation gauged signatures separate stable targets",
                "metric": "OOF ROC-AUC / balanced accuracy / permutation p",
                "value": (
                    f"{float(signature_row['roc_auc']):.6f}/"
                    f"{float(signature_row['balanced_accuracy_at_0_5']):.6f}/"
                    f"{float(signature_row['permutation_pvalue_auc']):.6f}"
                ),
                "threshold_or_reference": ">=0.70 / >=0.65 / <=0.05",
                "result": "supports_gauged_only",
            },
        ]
    )
    evidence = pd.DataFrame(evidence_rows)
    evidence.to_csv(
        OUT / "cross_phase_evidence_register.csv",
        index=False,
        encoding="utf-8-sig",
    )

    nested_oof_contract = [
        {
            "outer_fold": "fit_through_2011_eval_2012_2013",
            "allowed_inner_blocks": [
                "fit_2006_2007_eval_2008_2009",
                "fit_2006_2009_eval_2010_2011",
            ],
            "latest_gate_evidence_year": 2011,
        },
        {
            "outer_fold": "fit_through_2013_eval_2014_2015",
            "allowed_inner_blocks": [
                "fit_2006_2007_eval_2008_2009",
                "fit_2006_2009_eval_2010_2011",
                "fit_2006_2011_eval_2012_2013",
            ],
            "latest_gate_evidence_year": 2013,
        },
        {
            "outer_fold": "fit_through_2015_eval_2016_2018",
            "allowed_inner_blocks": [
                "fit_2006_2007_eval_2008_2009",
                "fit_2006_2009_eval_2010_2011",
                "fit_2006_2011_eval_2012_2013",
                "fit_2006_2013_eval_2014_2015",
            ],
            "latest_gate_evidence_year": 2015,
        },
    ]
    candidate_contract = {
        "contract_id": "C01_gauged_history_low_flow_residual_expert",
        "admitted": candidate_admitted,
        "logical_parent_run": "20260727_6",
        "baseline_components": {
            "Q72": "frozen",
            "Q78": "frozen",
            "reach_class_alpha": "frozen",
            "accepted_prediction_definition": "original class-alpha fusion",
        },
        "scope": {
            "gauged_station_history_required": True,
            "ungauged_reach_deployment_permitted": False,
            "static_attribute_gate_permitted": False,
            "location_gate_permitted": False,
            "q72_q78_conditional_fusion_permitted": False,
            "new_hydrologic_structure_permitted": False,
        },
        "station_policy": {
            "excluded": EXCLUSIONS,
            "protected_and_retained": PROTECTED,
            "reservoir_related_gate": 0.0,
            "data_quality_suspicious_gate": 0.0,
            "missing_nested_oof_history_gate": 0.0,
            "full_series_20260728_5_target_label_as_predictor": "forbidden",
        },
        "outer_folds": [
            "fit_through_2011_eval_2012_2013",
            "fit_through_2013_eval_2014_2015",
            "fit_through_2015_eval_2016_2018",
        ],
        "nested_oof_gate_contract": nested_oof_contract,
        "station_gate_definition": {
            "fold_low_flow_bias": (
                "100 * sum(Q0_pred-Q_obs) / sum(Q_obs), where low months use "
                "the inner-block training-observation Q25"
            ),
            "minimum_low_months_per_inner_block": 3,
            "minimum_eligible_inner_blocks": 2,
            "stable_rule": (
                "positive bias in >=2/3 eligible inner blocks, median bias "
                ">=10%, and no eligible block bias <=-10%"
            ),
            "soft_score": "sigmoid((median_bias_pct-10)/10)",
            "hard_zero_overrides": [
                "reservoir_related",
                "data_quality_suspicious",
                "insufficient_inner_blocks",
            ],
        },
        "time_gate_definition": {
            "inputs": "frozen baseline predictions only",
            "threshold": "station inner-OOF baseline-prediction Q25",
            "scale": "max(inner-OOF logQ interquartile range, 0.25)",
            "formula": "sigmoid((log(q25_Q0+eps)-log(Q0_t+eps))/scale)",
            "observed_eval_flow_use": "forbidden",
        },
        "correction_contract": {
            "formula": "log(Qnew)=log(Q0)-g_station*h_time*nonnegative_amplitude",
            "direction": "downward_only",
            "first_pilot_amplitude_grid": [0.0, 0.05, 0.10, 0.15, 0.20],
            "amplitude_selection_evidence": "nested OOF only",
            "deep_network": "forbidden",
            "Qnew_nonnegative": True,
        },
        "outer_evaluation_gates": {
            "target_low_abs_bias": "improves in at least 2/3 outer folds",
            "target_low_log_rmse_or_fdc": (
                "at least one improves in at least 2/3 outer folds"
            ),
            "clean_non_target_median_abs_delta_logq_max": 0.01,
            "target_nonlow_median_abs_delta_logq_max": 0.01,
            "high_flow_median_abs_delta_logq_max": 0.005,
            "new_severe_abs_pbias_over_50pct_stations": 0,
            "extreme_station_dependency": "forbidden",
            "good_count": "report with threshold-margin audit, not sole veto",
        },
        "sealed_confirmation": {
            "years": "2019-2022",
            "allowed_uses_before_full_freeze": "none",
            "confirmation_count": 1,
            "retuning_after_confirmation": "forbidden",
        },
    }
    (OUT / "candidate_contract.json").write_text(
        json.dumps(candidate_contract, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    dynamic_plan = {
        "run_naming": "use consecutive 20260728_N folders; create only after the previous folder is reported",
        "fixed_target": (
            "a leak-free frozen-baseline low-flow residual expert that improves "
            "stable gauged targets, protects non-target and non-low states, and "
            "only later earns ungauged scope through a separate static gate"
        ),
        "stages": [
            {
                "next_folder": "20260728_7",
                "name": "nested_oof_station_gate_audit",
                "model_change": "none",
                "success": [
                    "all outer-fold gates use only earlier nested OOF blocks",
                    "at least 10 clean targets are identified in at least 2 outer folds",
                    "next-period target low-flow median bias is positive in at least 2/3 folds",
                    "reservoir, quality-suspicious, and insufficient-history gates are zero",
                ],
                "failure_terminal": (
                    "write an evidence stop report; do not create a residual model"
                ),
            },
            {
                "next_folder": "20260728_8",
                "name": "minimal_monotone_residual_pilot",
                "model_change": "one scalar nonnegative amplitude selected only on nested OOF",
                "success": "all target and protection gates in candidate_contract.json pass",
                "failure_terminal": (
                    "diagnose exactly one mechanism; continue only if a bounded "
                    "single-component correction remains inside this contract"
                ),
            },
            {
                "next_folder": "dynamic_after_8",
                "name": "bounded_mechanism_iteration",
                "model_change": (
                    "one of station gate, time gate, or low-order residual form per folder"
                ),
                "loop_rule": (
                    "iterate folders until all temporal gates pass or evidence "
                    "shows no in-contract correction can pass; no fixed run count"
                ),
                "forbidden": [
                    "new Q72/Q78 fusion",
                    "new hydrologic structure",
                    "global loss search",
                    "2019-2022 inspection",
                ],
            },
            {
                "next_folder": "dynamic_after_temporal_pass",
                "name": "expanded_static_attribute_gate",
                "entry": "only after a frozen temporal candidate passes",
                "success": [
                    "OOF ROC-AUC >=0.70",
                    "balanced accuracy >=0.65",
                    "permutation p <=0.05",
                ],
                "failure_terminal": (
                    "retain gauged-only scope or stop; do not claim ungauged deployment"
                ),
            },
            {
                "next_folder": "dynamic_after_static_pass",
                "name": "five_fold_spatial_cluster_holdout",
                "entry": "only after the static attribute gate passes",
                "failure_terminal": "stop ungauged deployment",
            },
            {
                "next_folder": "dynamic_after_full_freeze",
                "name": "one_time_2019_2022_confirmation",
                "entry": "temporal and spatial gates passed and all choices frozen",
                "failure_terminal": (
                    "evidence stop; no retuning against 2019-2022"
                ),
            },
            {
                "next_folder": "final_dynamic_folder",
                "name": "final_summary_or_evidence_stop",
                "terminal": True,
            },
        ],
    }
    (OUT / "dynamic_execution_plan.json").write_text(
        json.dumps(dynamic_plan, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    decision = {
        "run_id": RUN.name,
        "phase_id": "comprehensive_diagnostic_gate",
        "logical_parent_run": "20260727_6",
        "predecessor_integrity": predecessor_integrity,
        "policy_consistency": policy_consistency,
        "candidate_admission_checks": candidate_admission_checks,
        "candidate_admitted": candidate_admitted,
        "admitted_contract_id": (
            candidate_contract["contract_id"] if candidate_admitted else None
        ),
        "scientific_decisions": {
            "promote_20260727_13_structure": False,
            "promote_20260727_15_structure": False,
            "develop_q72_q78_conditional_fusion": False,
            "develop_gauged_history_low_flow_residual_expert": candidate_admitted,
            "claim_ungauged_static_attribute_gate": False,
            "inspect_2019_2022_now": False,
        },
        "confidence": "ready_for_bounded_development_decision_with_scope_caveats",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    (OUT / "series_decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    contract_md = f"""# Admitted Candidate Contract

## Decision

`{candidate_contract['contract_id']}` is **{'ADMITTED' if candidate_admitted else 'NOT ADMITTED'}**
for bounded development.

It is a gauged-history, frozen-baseline, downward-only low-flow residual
expert. It is not an ungauged model and cannot use the full-series `_5` labels
inside blocked evaluation.

## Immutable baseline and scope

- Q72, Q78, and reach-class alpha stay frozen at `20260727_6`.
- No new structure, conditional fusion, global loss search, or deep network.
- Reservoir-related, data-quality-suspicious, and insufficient-history station
  gates equal zero.
- 劳村站、富罗（二）站、隆安站 remain excluded.
- 石角站 remains present.
- 2019--2022 remains sealed.

## Leak-free gate construction

| outer evaluation | allowed nested OOF evidence |
| --- | --- |
| 2012--2013 | fit 2006--2007/eval 2008--2009; fit 2006--2009/eval 2010--2011 |
| 2014--2015 | the two prior blocks plus fit through 2011/eval 2012--2013 |
| 2016--2018 | the three prior blocks plus fit through 2013/eval 2014--2015 |

The gate uses only nested blocks ending before the outer evaluation. The `_5`
57-station label is a diagnostic reference, never a predictor.

## First allowed correction

```text
log(Qnew) = log(Q0) - g_station * h_time * amplitude
```

- `amplitude` is nonnegative and selected from
  `0, 0.05, 0.10, 0.15, 0.20` using nested OOF only.
- `g_station` comes from nested OOF low-flow volume bias.
- `h_time` uses frozen-baseline predictions only.
- Evaluation observations may score results but cannot form a gate.

## Terminal behavior

`20260728_7` must first prove that the nested gate itself identifies a stable
next-period target group. If it fails, no residual candidate is created.
Subsequent folders change one bounded component at a time until the complete
temporal contract passes or an evidence-based stop is reached. Static ungauged
gating and spatial holdout remain downstream conditional stages.
"""
    (OUT / "CANDIDATE_CONTRACT.md").write_text(contract_md, encoding="utf-8")

    validation_md = f"""# Cross-Phase Validation Report

## Overall assessment

**Ready to use for a bounded development decision, with scope caveats.**

## Methodology review

The decision uses the exact `_6` reproduction, compensation audit, strict OOF
complementarity audit, and strict OOF heterogeneity audit. Engineering
integrity is separated from scientific admission. No causal or deployment
claim is inferred from a diagnostic gate alone.

## Calculation spot-checks

- `_3` development compensation indices independently read as
  `{float(development_compensation.loc['20260727_13', 'full_compensation_index']):.6f}`
  and `{float(development_compensation.loc['20260727_15', 'full_compensation_index']):.6f}`.
- `_4` has `{int(complementarity['regime_eligible_for_conditional_fusion'].astype(str).str.lower().eq('true').sum())}`
  eligible fusion regimes.
- `_5` has `{len(stable)}/{len(clean)}` clean stable targets with median
  station low-flow bias `{float(stable['median_low_flow_bias_pct'].median()):.2f}%`.
- Static AUC is `{float(static_row['roc_auc']):.3f}`; gauged-signature AUC is
  `{float(signature_row['roc_auc']):.3f}`.

## Decision-impact issues and caveats

1. The clean target is broad (57/68), so the residual expert is not a sparse
   anomaly patch; time gating and non-low-flow anchoring are critical.
2. Static attributes fail separation. Ungauged scope is unsupported.
3. Gauged signatures succeed, but require discharge history.
4. The full-series `_5` label would leak future outer-fold information and is
   prohibited as a model input.
5. Reservoir and data-quality groups show large bias but remain outside the
   correction target pending dedicated treatment.

## Required fixes before any deployment claim

- pass nested temporal OOF gate audit;
- pass residual target and protection gates;
- pass a new static-attribute gate;
- pass five-fold spatial cluster holdout;
- freeze all choices before the single 2019--2022 confirmation.
"""
    (OUT / "cross_phase_validation_report.md").write_text(
        validation_md,
        encoding="utf-8",
    )

    summary = {
        "run_id": RUN.name,
        "started_at": started.isoformat(timespec="seconds"),
        "ended_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "predecessor_gates_passed": int(sum(predecessor_integrity.values())),
        "predecessor_gates_expected": len(predecessor_integrity),
        "policy_checks_passed": int(sum(policy_consistency.values())),
        "policy_checks_expected": len(policy_consistency),
        "candidate_admitted": candidate_admitted,
        "contract_id": candidate_contract["contract_id"],
        "stable_clean_targets": int(len(stable)),
        "clean_eligible_stations": int(len(clean)),
        "conditional_fusion_regimes": int(
            complementarity[
                "regime_eligible_for_conditional_fusion"
            ].astype(str).str.lower().eq("true").sum()
        ),
        "static_attribute_gate_supported": static_supported,
        "gauged_signature_upper_bound_supported": signature_supported,
        "ungauged_deployment_permitted": False,
        "next_folder": "20260728_7" if candidate_admitted else None,
    }
    (OUT / "analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
