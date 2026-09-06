from __future__ import annotations

import csv
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(r"E:\SPARROW")
RUN = Path(__file__).resolve().parents[1]
SCRIPTS = RUN / "scripts"
REPORTS = RUN / "reports"
LOGS = RUN / "logs"
DAILY_LOG = ROOT / "5_Test" / "20260618.log"

HYPERPARAMS = (
    "rho=0.70;wm=480;et_gamma=0.75;sas_rho=0.93;young_k=1.5;"
    "storage_scale=720;prod_capacity=240;runoff_gamma=2.5;quick_rho=0.25;"
    "base_rho=0.85;base_release=0.10;fixed_sigma=3.0;production_sigma=1.5;"
    "group_sigma=1.5;multistore_sigma=0.3;hysteresis_sigma=3.0;"
    "station_sigma=1.0;slope_sigma=0.15;regime_slope_sigma=0.25;"
    "flow_contrast_weight=1.0"
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


def write_root_hyperparam_manifest() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    path = REPORTS / "run_manifest.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_id",
                "parent_run",
                "baseline_run",
                "current_best_run",
                "generation_type",
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
                "parent_run": "none",
                "baseline_run": "clean_Qmain_pure_simulation",
                "current_best_run": "clean_Qmain_pure_simulation",
                "generation_type": "clean_complete_reproducibility_mainline",
                "model_change": "rebuild_Qmain_from_source_inputs_inside_this_folder; no_layered_upstream_transfer; keep_CHM_PRE_V2_1_PPT; keep_CMFD_FAO56_ET0_PET; keep_ERA5_Land_monthly_evaporation_AET_2006_2022",
                "train_split": "2006-2015",
                "inner_validation_split": "2016-2018",
                "strict_validation_split": "2019-2022",
                "selected_hyperparameters": HYPERPARAMS,
                "promotion_decision": "pure_simulation_Qmain_baseline_for_20260618_subject_learner_replacement",
                "notes": "Local hyperparameter manifest used by the base-regression component inside this clean case. This folder is intended to run without depending on previous test-folder model outputs. Station-reach fixed metadata are stored under inputs/source_metadata and preserve the previously inspected topology corrections such as Gaoyao reach 19 and Wuzhou reach 18. CMFD provides PET/ET0 drivers; ERA5-Land provides observed/reanalysis evaporation AET. This run intentionally does not run layered upstream transfer or any post-hoc residual correction.",
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

    for path in [REPORTS / "intermediate", RUN / "figure" / "intermediate", SCRIPTS / "__pycache__"]:
        if path.exists():
            shutil.rmtree(path)
    for path in RUN.glob("README_202*.md"):
        path.unlink()
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
    pd.DataFrame(step_rows).to_csv(REPORTS / "workflow_step_status.csv", index=False, encoding="utf-8-sig")
    lines = [
        f"# {RUN.name} Clean Reproducibility Baseline Log",
        "",
        f"Updated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Purpose",
        "",
        "Run the pure-simulation Q_main hydrologic flow baseline in a single clean folder without reading previous test-folder model outputs. CHM_PRE V2.1 reach-catchment monthly rainfall, CMFD V2.0 six-variable FAO56 Penman-Monteith ET0/PET, and ERA5-Land monthly evaporation AET are retained. The case then runs base regression, mass-conserving intermediate baselines, alpha sweep, station-adaptive comparator, final reach-class light mass constraint, station-level diagnostics, and the full main-model document locally. It intentionally does not run layered upstream transfer.",
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
            "- This folder is the pure-simulation Q_main baseline.",
            "- It does not run layered upstream transfer.",
            "- It does not use validation observed upstream flow, target observed flow, or post-hoc residual correction.",
        ]
    )
    text = "\n".join(lines) + "\n"
    (LOGS / "run_log.md").write_text(text, encoding="utf-8")
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
            "interpretation": "pure simulation baseline: Q72 process-informed hierarchical MAP plus Q78_mass light mass constraint",
        }
    ]
    pd.DataFrame(rows).to_csv(REPORTS / "final_model_summary.csv", index=False, encoding="utf-8-sig")
    lines = [
        f"# {RUN.name} Final Model Summary",
        "",
        "## Conclusion",
        "",
        "This folder is the clean pure-simulation Q_main baseline for the 20260618 subject-learner replacement experiments.",
        "",
        "It intentionally excludes layered upstream transfer and post-hoc residual correction.",
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
    print((LOGS / "run_log.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
