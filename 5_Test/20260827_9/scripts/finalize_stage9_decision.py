"""Close the program at Stage 9 when a pre-registered development gate fails."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260827_9"
REPORTS = RUN / "reports"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    development = json.loads((REPORTS / "stage9_temporal_component_decision.json").read_text(encoding="utf-8"))
    if development["status"] == "PASS_TEMPORAL_COMPONENT_DEVELOPMENT":
        raise RuntimeError("Development passed; spatial deletion refits are required instead of closing")
    failed = [name for name, passed in development["checks"].items() if not passed]
    parent = development["lambda_zero"]
    candidate = development["candidate"]
    output_operator = development["output_operator_same_cohort"]
    final = {
        "stage": "20260827_9",
        "status": "STATE_CONSISTENT_SIG2P_NOT_AUTHORIZED",
        "selected_lambda_S": development["selected_lambda_S"],
        "selected_seed": development["selected_seed"],
        "failed_registered_gates": failed,
        "total_flow_finding": {
            "daily_pooled_NSE_delta": candidate["daily_summary"]["pooled_NSE"] - parent["daily_summary"]["pooled_NSE"],
            "monthly_pooled_NSE_delta": candidate["monthly_summary"]["pooled_NSE"] - parent["monthly_summary"]["pooled_NSE"],
            "monthly_station_median_NSE_delta": candidate["monthly_summary"]["station_median_NSE"] - parent["monthly_summary"]["station_median_NSE"],
            "daily_pooled_log_RMSE_delta": candidate["daily_summary"]["pooled_log_RMSE"] - parent["daily_summary"]["pooled_log_RMSE"]
        },
        "component_finding": {
            "lambda_zero_BFI_spearman": parent["BFI_spearman"],
            "candidate_BFI_spearman": candidate["BFI_spearman"],
            "lambda_zero_BFI_RMSE": parent["BFI_RMSE"],
            "candidate_BFI_RMSE": candidate["BFI_RMSE"],
            "same_cohort_output_operator_BFI_RMSE": output_operator["BFI_RMSE"],
            "interpretation": "The state operator improves spatial ordering and total flow but does not move enough water into state-consistent slow storage to meet the registered absolute-magnitude gate."
        },
        "actions": {
            "run_spatial_deletion_refits": False,
            "read_2019_2022_observations": False,
            "read_four_spatial_station_observations": False,
            "extend_candidate_to_2023_2024": False,
            "promote_candidate_to_TN": False,
            "retained_total_flow_baseline": "20260827_6",
            "retained_TN_state_interface": "raw DYN2P states/fluxes only",
            "SIG2P_O_role": "instantaneous split/signature diagnostic only"
        },
        "program_closed": True,
        "authorized_successor": None
    }
    write_json(REPORTS / "final_decision.json", final)
    report = f"""# 20260827_7–9 State-consistent SIG2P-S/P final report

## Decision

`{final['status']}`. The state-internal operator is numerically valid and improves several development metrics, but it fails the registered BFI magnitude gate. The program therefore stops before spatial deletion refits and before formal observations are opened.

## What worked

- Lambda-zero reproduces the original DYN2P fluxes and states exactly.
- Daily land mass error remains below `1e-10 mm`; all states and fluxes are nonnegative.
- All three seeds selected `lambda_S=0.5`; no boundary selection occurred.
- 2017–2018 daily pooled NSE changes from {parent['daily_summary']['pooled_NSE']:.4f} to {candidate['daily_summary']['pooled_NSE']:.4f}.
- Monthly pooled NSE changes from {parent['monthly_summary']['pooled_NSE']:.4f} to {candidate['monthly_summary']['pooled_NSE']:.4f}; station-median NSE changes from {parent['monthly_summary']['station_median_NSE']:.4f} to {candidate['monthly_summary']['station_median_NSE']:.4f}.
- BFI Spearman improves from {parent['BFI_spearman']:.3f} to {candidate['BFI_spearman']:.3f}; high-flow-fast and slow-memory directions both pass at 100% of evaluable stations.

## Why it does not pass

BFI RMSE improves only from {parent['BFI_RMSE']:.3f} to {candidate['BFI_RMSE']:.3f}, above the registered `0.20` limit. The same-cohort output-only SIG2P-O diagnostic reaches {output_operator['BFI_RMSE']:.3f}, showing that a final-output split can force the desired magnitude much more strongly, but it still has no corresponding corrected lower-store state.

The result is therefore not a numerical failure. It is a structural trade-off: the conservative state equation preserves realistic lag and slightly improves discharge, but the registered BFI score does not identify enough state-consistent slow-path transfer under the selected global strength.

## Locked consequence

- `20260827_6` remains the total-flow baseline.
- TN may use only raw DYN2P inventories and fluxes when a true storage lag is required.
- Existing SIG2P-O remains a component/signature diagnostic and must not be presented as a corrected slow-water inventory.
- 2019–2022 and the four spatial stations were not opened for this rejected candidate.
"""
    (REPORTS / "technical_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(final, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
