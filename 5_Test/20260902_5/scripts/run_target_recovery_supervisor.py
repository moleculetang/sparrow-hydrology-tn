from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pandas as pd


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / r"5_Test\20260902_5"
WORK = RUN / "work"
LOGS = RUN / "logs" / "target_recovery"
REPORTS = RUN / "reports"
WORKER = RUN / r"scripts\run_nested_worker.py"
FINALIZER = RUN / r"scripts\finalize_nested.py"
FULL_REFIT = ROOT / r"5_Test\20260902_6\scripts\run_full_refit.py"
STATUS = REPORTS / "target_recovery_status.json"
MAX_WORKERS = 8
KKT_LIMIT = 1e-5

sys.path.insert(0, str(RUN / "scripts"))
import run_nested_worker as nested  # noqa: E402


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def passed_folds() -> set[str]:
    passed: set[str] = set()
    for path in WORK.glob("h22_nested_parameters_*.parquet"):
        frame = pd.read_parquet(path)
        if "projected_kkt_max" not in frame:
            continue
        valid = frame.loc[frame.projected_kkt_max <= KKT_LIMIT, "fold_id"]
        passed.update(valid.astype(str))
    return passed


def required_folds() -> list[str]:
    observations = nested.s19.build_observations()
    folds = nested.s19.build_folds(observations, "full")
    folds = folds.loc[~folds.holdout_type.eq("TEMPORAL")]
    return folds.fold_id.astype(str).tolist()


def parameter_path(fold_id: str) -> Path:
    safe = "".join(character if character.isalnum() or character in "_-" else "_" for character in fold_id)
    return WORK / f"h22_nested_parameters_target_{safe}.parquet"


def validate_fold(fold_id: str) -> float:
    path = parameter_path(fold_id)
    if not path.exists():
        raise RuntimeError(f"{fold_id}: worker exited without parameter checkpoint")
    frame = pd.read_parquet(path)
    if len(frame) != 1 or str(frame.fold_id.iloc[0]) != fold_id:
        raise RuntimeError(f"{fold_id}: invalid target checkpoint identity")
    kkt = float(frame.projected_kkt_max.iloc[0])
    if not (kkt <= KKT_LIMIT):
        raise RuntimeError(f"{fold_id}: projected KKT {kkt} exceeds {KKT_LIMIT}")
    return kkt


def run_command(arguments: list[str], stdout_path: Path, stderr_path: Path) -> None:
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        result = subprocess.run(arguments, stdout=stdout, stderr=stderr, cwd=ROOT, check=False)
    if result.returncode != 0:
        detail = stderr_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(arguments)}\n{detail}")


def main() -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    required = required_folds()
    initial_passed = passed_folds()
    pending = [fold_id for fold_id in required if fold_id not in initial_passed]
    active: dict[str, tuple[subprocess.Popen, object, object]] = {}
    completed: dict[str, float] = {}
    failures: dict[str, str] = {}
    environment = os.environ.copy()
    environment.update({
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
        "NUMEXPR_NUM_THREADS": "2",
    })

    def write_status(state: str) -> None:
        atomic_json({
            "status": state,
            "required_fold_count": len(required),
            "initial_passed_count": len(set(required) & initial_passed),
            "pending_count": len(pending),
            "active": sorted(active),
            "completed_this_run": completed,
            "failures": failures,
            "updated_at_epoch": time.time(),
        }, STATUS)

    write_status("RUNNING_TARGET_RECOVERY")
    while pending or active:
        while pending and len(active) < MAX_WORKERS:
            fold_id = pending.pop(0)
            stdout_path = LOGS / f"{fold_id}.stdout.log"
            stderr_path = LOGS / f"{fold_id}.stderr.log"
            stdout = stdout_path.open("w", encoding="utf-8")
            stderr = stderr_path.open("w", encoding="utf-8")
            process = subprocess.Popen(
                [sys.executable, "-u", str(WORKER), "--fold-id", fold_id],
                stdout=stdout,
                stderr=stderr,
                cwd=ROOT,
                env=environment,
            )
            active[fold_id] = (process, stdout, stderr)
            print(json.dumps({"event": "started", "fold_id": fold_id, "pid": process.pid}), flush=True)
        time.sleep(2)
        for fold_id, (process, stdout, stderr) in list(active.items()):
            returncode = process.poll()
            if returncode is None:
                continue
            stdout.close()
            stderr.close()
            del active[fold_id]
            if returncode != 0:
                error_path = LOGS / f"{fold_id}.stderr.log"
                detail = error_path.read_text(encoding="utf-8", errors="replace")[-4000:]
                failures[fold_id] = f"exit_code={returncode}\n{detail}"
                print(json.dumps({"event": "failed", "fold_id": fold_id, "exit_code": returncode}), flush=True)
            else:
                try:
                    kkt = validate_fold(fold_id)
                    completed[fold_id] = kkt
                    print(json.dumps({"event": "completed", "fold_id": fold_id, "kkt": kkt}), flush=True)
                except Exception as error:  # preserve all other completed workers
                    failures[fold_id] = str(error)
                    print(json.dumps({"event": "failed_validation", "fold_id": fold_id, "detail": str(error)}), flush=True)
            write_status("RUNNING_TARGET_RECOVERY")

    if failures:
        write_status("FAILED_TARGET_RECOVERY")
        raise RuntimeError(f"Target recovery failed for {len(failures)} folds: {sorted(failures)}")

    final_passed = passed_folds()
    missing = [fold_id for fold_id in required if fold_id not in final_passed]
    if missing:
        write_status("FAILED_INCOMPLETE_CHECKPOINTS")
        raise RuntimeError(f"Missing {len(missing)} folds after target recovery: {missing}")
    part_files = list(WORK.glob("*.part"))
    if part_files:
        write_status("FAILED_PART_FILES")
        raise RuntimeError(f"Atomic part files remain: {[path.name for path in part_files]}")

    write_status("FINALIZING_STAGE5")
    run_command(
        [sys.executable, str(FINALIZER)],
        LOGS / "finalize_nested.stdout.log",
        LOGS / "finalize_nested.stderr.log",
    )
    decision_path = REPORTS / "spatial_decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if decision.get("status") == "PASS_H22_SPATIAL_GATES":
        write_status("RUNNING_FULL_REFIT")
        run_command(
            [sys.executable, str(FULL_REFIT)],
            LOGS / "full_refit.stdout.log",
            LOGS / "full_refit.stderr.log",
        )
        write_status("COMPLETE_PROMOTED")
    else:
        write_status("COMPLETE_NOT_PROMOTED")


if __name__ == "__main__":
    main()
