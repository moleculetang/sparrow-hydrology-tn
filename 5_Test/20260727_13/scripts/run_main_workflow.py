from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from experiment_settings import NONLINEAR_RECESSION_EXPONENT


ROOT = Path(r"E:\SPARROW")
RUN = Path(__file__).resolve().parents[1]
SCRIPTS = RUN / "scripts"
REPORTS = RUN / "reports"
LOGS = RUN / "logs"
DAILY_LOG = LOGS / f"{RUN.name}_main_workflow.log"
POLICY_PATH = RUN / "inputs" / "source_metadata" / "station_screening_policy.csv"
EXPERIMENT_METADATA_PATH = RUN / "inputs" / "source_metadata" / "run_experiment.json"

HYPERPARAMS = (
    "rho=0.70;wm=480;et_gamma=0.75;sas_rho=0.93;young_k=1.5;"
    "storage_scale=720;prod_capacity=240;runoff_gamma=2.5;quick_rho=0.25;"
    "base_rho=0.85;base_release=0.10;fixed_sigma=3.0;production_sigma=1.5;"
    "group_sigma=1.5;multistore_sigma=0.3;hysteresis_sigma=3.0;"
    "station_sigma=1.0;slope_sigma=0.15;regime_slope_sigma=0.25;"
    f"flow_contrast_weight=1.0;nonlinear_recession_exponent={NONLINEAR_RECESSION_EXPONENT}"
)

STEPS = [
    ("input_build", "build_input_panel.py"),
    ("base_regression", "fit_base_regression.py"),
    ("mass_skeleton", "build_mass_skeleton.py"),
    ("mass_global_nonnegative", "fit_global_mass_model.py"),
    ("mass_reach_class", "fit_reach_class_mass_model.py"),
    ("mass_map_shrunk", "fit_map_shrunk_mass_model.py"),
    ("alpha_sweep", "run_mass_constraint_sweep.py"),
    ("station_adaptive_comparator", "run_station_adaptive_comparator.py"),
    ("main_model", "run_reach_class_main_model.py"),
    ("main_diagnostics_and_figures", "build_main_diagnostics_and_figures.py"),
    ("station_performance_diagnostics", "build_station_performance_diagnostics.py"),
    ("main_model_full_document", "build_main_model_full_document.py"),
]


def read_exclusion_policy() -> tuple[list[str], str]:
    policy = pd.read_csv(POLICY_PATH, encoding="utf-8-sig")
    mask = policy["exclude_before_training"].astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})
    names = policy.loc[mask, "station_name"].astype(str).tolist()
    return names, "、".join(names) if names else "none"


def read_experiment_metadata() -> dict[str, object]:
    defaults: dict[str, object] = {
        "parent_run": "20260620_44",
        "generation_type": "dynamic_station_screening_reproducibility_baseline",
        "candidate_station": "",
        "screening_iteration": 0,
        "evidence_run": "",
        "decision_before_run": "baseline_reproduction",
    }
    if EXPERIMENT_METADATA_PATH.exists():
        defaults.update(json.loads(EXPERIMENT_METADATA_PATH.read_text(encoding="utf-8")))
    return defaults


def write_root_hyperparam_manifest() -> None:
    excluded_names, excluded_text = read_exclusion_policy()
    metadata = read_experiment_metadata()
    REPORTS.mkdir(parents=True, exist_ok=True)
    path = REPORTS / "run_manifest.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_id",
                "parent_run",
                "generation_type",
                "candidate_station",
                "screening_iteration",
                "evidence_run",
                "decision_before_run",
                "baseline_run",
                "current_best_run",
                "model_change",
                "train_split",
                "inner_validation_split",
                "strict_validation_split",
                "selected_hyperparameters",
                "promotion_decision",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "run_id": RUN.name,
                "parent_run": metadata["parent_run"],
                "generation_type": metadata["generation_type"],
                "candidate_station": metadata["candidate_station"],
                "screening_iteration": metadata["screening_iteration"],
                "evidence_run": metadata["evidence_run"],
                "decision_before_run": metadata["decision_before_run"],
                "baseline_run": "20260620_44",
                "current_best_run": metadata["parent_run"],
                "model_change": metadata.get(
                    "model_change",
                    "experiment-specific change documented in run_experiment.json",
                ),
                "train_split": "2006-2015",
                "inner_validation_split": "2016-2018",
                "strict_validation_split": "2019-2022",
                "selected_hyperparameters": HYPERPARAMS,
                "promotion_decision": metadata.get(
                    "decision_after_run",
                    "pending_experiment_gate",
                ),
                "notes": (
                    f"Single-factor model-development run against {metadata['parent_run']}. "
                    f"Active exclusions ({len(excluded_names)}): {excluded_text}. "
                    "Reservoir-related stations are report-only. "
                    "Years 2019-2022 are confirmation-only and are not used in fitting or selection."
                ),
            }
        )


def cleanup_report_layout() -> None:
    input_dir = REPORTS / "input_preprocessing"
    workflow_dir = REPORTS / "workflow"
    input_dir.mkdir(parents=True, exist_ok=True)
    workflow_dir.mkdir(parents=True, exist_ok=True)
    workflow_files = {
        "run_manifest.csv",
        "workflow_step_status.csv",
        "reproducibility_comparison.csv",
    }

    for path in REPORTS.iterdir():
        if not path.is_file():
            continue
        destination = workflow_dir / path.name if path.name in workflow_files else input_dir / path.name
        if destination.exists():
            destination.unlink()
        path.replace(destination)

    # Stage-0 and later ablations must remain auditable.  Preserve all
    # intermediate reports/figures and only remove Python bytecode caches.
    for path in [SCRIPTS / "__pycache__"]:
        if path.exists():
            shutil.rmtree(path)
    for path in (SCRIPTS / "components").glob("__pycache__"):
        if path.exists():
            shutil.rmtree(path)


def run_step(label: str, script_name: str) -> dict[str, object]:
    script = SCRIPTS / script_name
    started = datetime.now()
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(RUN),
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    ended = datetime.now()
    (LOGS / f"{label}.stdout.log").write_text(proc.stdout, encoding="utf-8")
    (LOGS / f"{label}.stderr.log").write_text(proc.stderr, encoding="utf-8")
    return {
        "step": label,
        "script": script_name,
        "returncode": proc.returncode,
        "started_at": started.isoformat(timespec="seconds"),
        "ended_at": ended.isoformat(timespec="seconds"),
        "elapsed_seconds": round((ended - started).total_seconds(), 3),
    }


def read_final_metrics() -> dict[str, object]:
    summary_path = REPORTS / "main_model" / "reach_class_light_constraint_summary.csv"
    if not summary_path.exists():
        return {}
    row = pd.read_csv(summary_path, encoding="utf-8-sig").iloc[0].to_dict()
    out = {
        "validation_stations": int(row["validation_stations"]),
        "class_median_NSElog": float(row["class_median_NSElog"]),
        "class_median_KGE": float(row["class_median_KGE"]),
        "class_median_absPBIAS": float(row["class_median_absPBIAS"]),
        "class_good_count": int(row["class_good_count"]),
        "class_median_alpha": float(row["class_median_alpha"]),
        "class_mean_alpha": float(row["class_mean_alpha"]),
        "distance_to_mass_reduction_pct_vs_alpha0": float(row["distance_to_mass_reduction_pct_vs_alpha0"]),
        "alpha0_median_NSElog": float(row["alpha0_median_NSElog"]),
        "global01_median_NSElog": float(row["global01_median_NSElog"]),
        "mass_alpha1_median_NSElog": float(row["mass_alpha1_median_NSElog"]),
    }
    return out


def write_workflow_logs(step_rows: list[dict[str, object]], final: dict[str, object]) -> None:
    _, excluded_text = read_exclusion_policy()
    metadata = read_experiment_metadata()
    pd.DataFrame(step_rows).to_csv(REPORTS / "workflow_step_status.csv", index=False, encoding="utf-8-sig")
    lines = [
        f"# {RUN.name} Main Workflow Log",
        "",
        f"Updated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Purpose",
        "",
        f"Run {metadata['generation_type']} against accepted parent {metadata['parent_run']}. Active exclusions are loaded from station_screening_policy.csv: {excluded_text}.",
        "",
        "Station-reach fixed metadata under `inputs/source_metadata/station_reach_match_fixed.csv` preserve previously inspected topology corrections, including Gaoyao reach 19 and Wuzhou reach 18.",
        "",
        "## Step Status",
        "",
    ]
    for row in step_rows:
        lines.append(
            f"- {row['step']}: returncode={row['returncode']}, elapsed={row['elapsed_seconds']}s, script={row['script']}"
        )
    if final:
        lines.extend(
            [
                "",
                "## Final Mainline Metrics",
                "",
                f"- validation stations: {final['validation_stations']}",
                f"- median NSElog: {final['class_median_NSElog']:.6f}",
                f"- median KGE: {final['class_median_KGE']:.6f}",
                f"- median |PBIAS|: {final['class_median_absPBIAS']:.6f}",
                f"- good stations: {final['class_good_count']}",
                f"- median alpha: {final['class_median_alpha']:.6f}",
                f"- mean alpha: {final['class_mean_alpha']:.6f}",
                f"- distance-to-mass reduction vs alpha0: {final['distance_to_mass_reduction_pct_vs_alpha0']:.2f}%",
            ]
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
        "- This folder is one sequential single-factor experiment in the model-development chain.",
            "- It does not run layered upstream transfer.",
        "- It does not use validation observed upstream flow, target observed flow, or post-hoc residual correction.",
        f"- Stations excluded before all training steps: {excluded_text}.",
        ]
    )
    text = "\n".join(lines) + "\n"
    (LOGS / "main_workflow_log.md").write_text(text, encoding="utf-8")
    with DAILY_LOG.open("a", encoding="utf-8") as f:
        f.write("\n\n" + text)


def write_final_summary(final: dict[str, object]) -> None:
    if not final:
        return
    rows = [
        {
            "model": "Q_main",
            "group": "all_stations",
            "period": "strict_2019_2022",
            "station_count": final["validation_stations"],
            "median_NSElog": final["class_median_NSElog"],
            "median_KGE": final["class_median_KGE"],
            "median_absPBIAS": final["class_median_absPBIAS"],
            "good_count": final["class_good_count"],
            "interpretation": "Q72 process-informed hierarchical MAP plus Q78_mass light mass constraint; promotion requires the common-station ablation gate",
        }
    ]
    pd.DataFrame(rows).to_csv(REPORTS / "final_model_summary.csv", index=False, encoding="utf-8-sig")
    lines = [
        f"# {RUN.name} Final Model Summary",
        "",
        "## Conclusion",
        "",
        "This folder is one sequential single-factor experiment in the model-development chain.",
        "",
        "It keeps the 20260620_44 model structure and loads all active exclusions from station_screening_policy.csv before input construction.",
        "",
        "## Strict Validation Result",
        "",
        "| model | group | median NSElog | median KGE | median abs PBIAS | good stations |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
        f"| Q_main | all stations | {final['class_median_NSElog']:.6f} | {final['class_median_KGE']:.6f} | {final['class_median_absPBIAS']:.6f} | {final['class_good_count']} / {final['validation_stations']} |",
        "",
        "## Model Boundary",
        "",
        "```text",
        "Q_main = Q72 process-informed hierarchical MAP log-flow learner + Q78_mass light mass-conserving pull",
        "```",
        "",
        "Not included:",
        "",
        "```text",
        "Q_layered_transfer",
        "observed upstream validation flow",
        "observed target validation flow",
        "post-hoc Bayesian residual correction",
        "```",
    ]
    (REPORTS / "final_model_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    LOGS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    write_root_hyperparam_manifest()
    step_rows: list[dict[str, object]] = []
    for label, script_name in STEPS:
        row = run_step(label, script_name)
        step_rows.append(row)
        if int(row["returncode"]) != 0:
            write_workflow_logs(step_rows, {})
            print(f"FAILED {label}; see logs/{label}.stderr.log")
            return int(row["returncode"])
    final = read_final_metrics()
    write_workflow_logs(step_rows, final)
    cleanup_report_layout()
    write_final_summary(final)
    print((LOGS / "main_workflow_log.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
