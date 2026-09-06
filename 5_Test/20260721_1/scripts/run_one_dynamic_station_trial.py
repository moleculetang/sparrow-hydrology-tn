from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


TEST_ROOT = Path(r"E:\SPARROW\5_Test")
CONTROL_RUN = TEST_ROOT / "20260721_1"
CONTROL_SCRIPT = CONTROL_RUN / "scripts" / "dynamic_station_screening_controller.py"
STATE_PATH = CONTROL_RUN / "reports" / "dynamic_station_screening" / "chain_state.json"
FOLDS = (
    "fit_2006_2011_eval_2012_2013",
    "fit_2006_2013_eval_2014_2015",
    "fit_2006_2015_eval_2016_2018",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create/resume and finish exactly one dynamic station-ablation trial.")
    parser.add_argument("--candidate")
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
        command = [sys.executable, str(CONTROL_SCRIPT), "create"]
        if args.candidate:
            command.extend(["--candidate", args.candidate])
        if args.run_id:
            command.extend(["--run-id", args.run_id])
        run(command, CONTROL_RUN)
        state = read_json(STATE_PATH)
        active = str(state["active_trial"])
    trial = TEST_ROOT / active
    experiment = read_json(trial / "inputs" / "source_metadata" / "run_experiment.json")
    if str(experiment.get("generation_type")) != "dynamic_station_screening_single_station_ablation":
        raise RuntimeError(f"Active run {active} is not a single-station ablation")

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

    reference = TEST_ROOT / str(experiment["metric_reference_run"])
    run(
        [
            sys.executable,
            str(trial / "scripts" / "compare_station_ablation.py"),
            "--base",
            str(reference),
            "--trial",
            str(trial),
            "--candidate",
            str(experiment["candidate_station"]),
        ],
        trial,
    )
    run([sys.executable, str(CONTROL_SCRIPT), "finalize", "--trial", active], CONTROL_RUN)
    final_state = read_json(STATE_PATH)
    decision = read_json(trial / "reports" / "station_screening" / "ablation_comparison" / "decision.json")
    print(
        json.dumps(
            {
                "finished_trial": active,
                "candidate": experiment["candidate_station"],
                "decision": decision["decision"],
                "accepted_by": decision.get("accepted_by", ""),
                "accepted_parent_run": final_state["accepted_parent_run"],
                "next_run_id": final_state["next_run_id"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
