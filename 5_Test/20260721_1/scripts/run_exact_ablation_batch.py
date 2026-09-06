from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


CONTROL_RUN = Path(r"E:\SPARROW\5_Test\20260721_1")
STATE_PATH = CONTROL_RUN / "reports" / "dynamic_station_screening" / "chain_state.json"
RUN_ONE = CONTROL_RUN / "scripts" / "run_one_dynamic_station_trial.py"
CONTINUABLE_PHASES = {"full_ablation_escalation", "priority_candidate_ablation"}


def read_state() -> dict[str, object]:
    return json.loads(STATE_PATH.read_text(encoding="utf-8"))


def main() -> None:
    initial = read_state()
    if initial.get("active_trial"):
        raise RuntimeError(f"Cannot start batch with active trial {initial['active_trial']}")
    initial_parent = str(initial["accepted_parent_run"])
    completed = 0
    while True:
        before = read_state()
        phase = str(before.get("phase"))
        if str(before["accepted_parent_run"]) != initial_parent:
            print("BATCH_STOP_ACCEPTED_PARENT_CHANGED", flush=True)
            break
        if phase not in CONTINUABLE_PHASES:
            print(f"BATCH_STOP_PHASE={phase}", flush=True)
            break
        if before.get("active_trial"):
            raise RuntimeError(f"Unexpected active trial before launch: {before['active_trial']}")

        proc = subprocess.run([sys.executable, str(RUN_ONE)], cwd=CONTROL_RUN)
        after = read_state()
        completed += 1
        print(
            json.dumps(
                {
                    "batch_completed": completed,
                    "run": after.get("last_completed_trial"),
                    "decision": after.get("last_decision"),
                    "accepted_parent": after.get("accepted_parent_run"),
                    "phase": after.get("phase"),
                    "next_run_id": after.get("next_run_id"),
                    "runner_returncode": proc.returncode,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if proc.returncode != 0:
            print("BATCH_STOP_RUNNER_ERROR_OR_QUEUE_EXHAUSTED", flush=True)
            break
        if str(after["accepted_parent_run"]) != initial_parent:
            print("BATCH_STOP_ACCEPTED_PARENT_CHANGED", flush=True)
            break
        if str(after.get("phase")) not in CONTINUABLE_PHASES:
            print(f"BATCH_STOP_PHASE={after.get('phase')}", flush=True)
            break


if __name__ == "__main__":
    main()
