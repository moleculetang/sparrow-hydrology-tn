from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


TEST_ROOT = Path(r"E:\SPARROW\5_Test")
CONTROL_RUN = TEST_ROOT / "20260721_1"
STATE_PATH = CONTROL_RUN / "reports" / "dynamic_station_screening" / "chain_state.json"
CONTROLLER = CONTROL_RUN / "scripts" / "station_set_trial_controller.py"
COMPARE = CONTROL_RUN / "scripts" / "compare_station_ablation.py"
FOLDS = (
    "fit_2006_2011_eval_2012_2013",
    "fit_2006_2013_eval_2014_2015",
    "fit_2006_2015_eval_2016_2018",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create/resume and finish one joint-ablation or re-add trial.")
    mode = parser.add_mutually_exclusive_group(required=False)
    mode.add_argument("--stations-json")
    mode.add_argument("--stations", help="Semicolon-separated station names; safer than JSON in Windows shells.")
    mode.add_argument("--readd-station")
    mode.add_argument("--group-readd-stations", help="Semicolon-separated locked interaction-group members.")
    parser.add_argument("--reason", default="interaction_screen")
    parser.add_argument("--run-id")
    parser.add_argument("--force-main", action="store_true")
    parser.add_argument("--force-folds", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def run(command: list[str], cwd: Path) -> None:
    print("RUN", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    args = parse_args()
    state = read_json(STATE_PATH)
    active = str(state.get("active_trial") or "")
    if not active:
        if args.stations_json or args.stations:
            stations_json = args.stations_json or json.dumps(
                [x.strip() for x in str(args.stations).split(";") if x.strip()], ensure_ascii=False
            )
            command = [sys.executable, str(CONTROLLER), "create-group", "--stations-json", stations_json, "--reason", args.reason]
        elif args.readd_station:
            command = [sys.executable, str(CONTROLLER), "create-readd", "--station", args.readd_station]
        elif args.group_readd_stations:
            stations_json = json.dumps(
                [x.strip() for x in str(args.group_readd_stations).split(";") if x.strip()], ensure_ascii=False
            )
            command = [sys.executable, str(CONTROLLER), "create-group-readd", "--stations-json", stations_json]
        else:
            raise ValueError("Provide --stations-json or --readd-station when no active set trial exists")
        if args.run_id:
            command.extend(["--run-id", args.run_id])
        run(command, CONTROL_RUN)
        state = read_json(STATE_PATH)
        active = str(state["active_trial"])
    trial = TEST_ROOT / active
    experiment = read_json(trial / "inputs" / "source_metadata" / "run_experiment.json")
    generation = str(experiment.get("generation_type"))
    if generation not in {
        "dynamic_station_screening_joint_ablation",
        "dynamic_station_screening_readd",
        "dynamic_station_screening_group_readd",
    }:
        raise RuntimeError(f"Active run {active} is not a station-set trial: {generation}")

    main_output = trial / "reports" / "main_model" / "reach_class_selected_predictions_long.csv"
    if args.force_main or not main_output.exists():
        run([sys.executable, str(trial / "scripts" / "run_main_workflow.py")], trial)
    else:
        print(f"RESUME main workflow already complete: {main_output}", flush=True)
    fold_outputs = [
        trial / "reports" / "station_screening" / "blocked_folds" / fold / "evaluation_predictions.csv" for fold in FOLDS
    ]
    if args.force_folds or not all(path.exists() for path in fold_outputs):
        run([sys.executable, str(trial / "scripts" / "run_blocked_q72_screening.py")], trial)
    else:
        print("RESUME all blocked folds already complete", flush=True)

    candidates_json = json.dumps(experiment["candidate_stations"], ensure_ascii=False)
    if generation == "dynamic_station_screening_joint_ablation":
        reference = TEST_ROOT / str(experiment["metric_reference_run"])
        run(
            [sys.executable, str(COMPARE), "--base", str(reference), "--trial", str(trial), "--candidates-json", candidates_json],
            trial,
        )
        run([sys.executable, str(CONTROLLER), "finalize-group", "--trial", active], CONTROL_RUN)
    else:
        excluded_parent = TEST_ROOT / str(experiment["parent_run"])
        output_name = "group_readd_comparison" if generation == "dynamic_station_screening_group_readd" else "readd_comparison"
        output = trial / "reports" / "station_screening" / output_name
        run(
            [
                sys.executable,
                str(COMPARE),
                "--base",
                str(trial),
                "--trial",
                str(excluded_parent),
                "--experiment-run",
                str(trial),
                "--output-dir",
                str(output),
                "--candidates-json",
                candidates_json,
            ],
            trial,
        )
        finalize_command = "finalize-group-readd" if generation == "dynamic_station_screening_group_readd" else "finalize-readd"
        run([sys.executable, str(CONTROLLER), finalize_command, "--trial", active], CONTROL_RUN)
    print(json.dumps(read_json(STATE_PATH), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
