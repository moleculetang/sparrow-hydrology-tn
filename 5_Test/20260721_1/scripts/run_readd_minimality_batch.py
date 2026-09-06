from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


CONTROL_RUN = Path(r"E:\SPARROW\5_Test\20260721_1")
STATE_PATH = CONTROL_RUN / "reports" / "dynamic_station_screening" / "chain_state.json"
SET_CONTROLLER = CONTROL_RUN / "scripts" / "station_set_trial_controller.py"
RUN_ONE_SET = CONTROL_RUN / "scripts" / "run_one_station_set_trial.py"


def read_state() -> dict[str, object]:
    return json.loads(STATE_PATH.read_text(encoding="utf-8"))


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=CONTROL_RUN, check=True)


def main() -> None:
    completed = 0
    while True:
        state = read_state()
        if state.get("active_trial"):
            raise RuntimeError(f"Unexpected active trial: {state['active_trial']}")
        phase = str(state.get("phase"))
        if phase == "readd_audit_next_pass_required":
            run([sys.executable, str(SET_CONTROLLER), "start-readd-audit"])
            continue
        if phase == "full_influence_audit_required":
            print("READD_BATCH_STOP_MINIMALITY_STABLE", flush=True)
            break
        if phase != "final_exclusion_readd_audit":
            print(f"READD_BATCH_STOP_PHASE={phase}", flush=True)
            break

        pending = [str(x) for x in state.get("pending_minimization_stations", [])]
        if not pending:
            raise RuntimeError("Re-add phase has no pending stations")
        station = pending[0]
        run([sys.executable, str(RUN_ONE_SET), "--readd-station", station])
        completed += 1
        after = read_state()
        print(
            json.dumps(
                {
                    "readd_batch_completed": completed,
                    "run": after.get("last_completed_trial"),
                    "station": station,
                    "decision": after.get("last_decision"),
                    "accepted_parent": after.get("accepted_parent_run"),
                    "phase": after.get("phase"),
                    "pending": after.get("pending_minimization_stations"),
                    "next_run_id": after.get("next_run_id"),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
