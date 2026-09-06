from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(r"E:\SPARROW")
TEST = ROOT / "5_Test"
HERE = TEST / "20260824_8"
REPORTS = HERE / "reports"
LOCKS = HERE / "locks"
CONTRACT = HERE / "experiment_contract.json"

EVIDENCE = {
    "f1_decision": TEST / "20260824_4" / "reports" / "f1_decision.json",
    "f1_verification": TEST / "20260824_4" / "reports" / "f1_independent_verification.json",
    "f2_decision": TEST / "20260824_5" / "reports" / "f2_decision.json",
    "f2_verification": TEST / "20260824_5" / "reports" / "f2_independent_verification.json",
    "f3_decision": TEST / "20260824_6" / "reports" / "f3_decision.json",
    "f3_verification": TEST / "20260824_6" / "reports" / "f3_independent_verification.json",
    "spatial_decision": TEST / "20260824_7" / "reports" / "nested_spatial_decision.json",
    "spatial_verification": TEST / "20260824_7" / "reports" / "nested_spatial_independent_verification.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    if Path(sys.prefix).name.lower() != "sparrow":
        raise RuntimeError("sparrow environment required")
    missing = [str(path) for path in [CONTRACT, *EVIDENCE.values()] if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)
    values = {name: load(path) for name, path in EVIDENCE.items()}
    required = {
        "f1_decision": "F1_MEAN_LAYER_TEMPORAL_UPGRADE_SUPPORTED",
        "f1_verification": "PASS",
        "f2_decision": "NO_REGISTERED_F2_UPGRADE_SUPPORTED",
        "f2_verification": "PASS",
        "f3_decision": "NO_REGISTERED_F3_UPGRADE_SUPPORTED",
        "f3_verification": "PASS",
        "spatial_decision": "F1_MONITORED_STATION_TEMPORAL_ONLY",
        "spatial_verification": "PASS",
    }
    actual = {name: str(value.get("status")) for name, value in values.items()}
    if actual != required:
        raise RuntimeError({"expected": required, "actual": actual})
    if any(value.get("TN_2022_values_read", value.get("TN_2022_read", False)) for value in values.values()):
        raise RuntimeError("STOP_2022_TN_PRELOCK_READ")
    hashes = {str(CONTRACT): sha256(CONTRACT)}
    hashes.update({str(path): sha256(path) for path in EVIDENCE.values()})
    lock = {
        "lock": "development_mechanism_lock",
        "written_before_2022_TN_values_read": True,
        "evidence_status": actual,
        "selected_final_architecture": "Q72_F00_S0_OR_S1_12_SIX_MU_FIXED_T1_H1_GLOBAL_GAUSSIAN_CQ_HINGE",
        "reference_architecture": "Q72_F00_S0_OR_S1_12_SIX_MU_FIXED_T1_H1_GLOBAL_GAUSSIAN_PROCESS_PARENT",
        "upgrade_scope": "MONITORED_STATION_PREDICTION_UPGRADE_ONLY",
        "spatial_transfer_status": "NOT_SUPPORTED",
        "temperature": "CLOSED_NOT_USED",
        "F2": "CLOSED_NOT_SUPPORTED",
        "F3": "CLOSED_NOT_SUPPORTED",
        "joint_F2_F3": "CLOSED_TRIGGER_NOT_MET",
        "development_support_period": [2016, 2021],
        "OOF_evaluation_years": [2018, 2019, 2020, 2021],
        "retrospective_locked_year": 2022,
        "TN_2022_values_read": False,
        "files": hashes,
        "aggregate_sha256": hashlib.sha256("\n".join(f"{k}|{v}" for k, v in sorted(hashes.items())).encode()).hexdigest(),
    }
    LOCKS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    target = LOCKS / "development_mechanism_lock.json"
    target.write_text(json.dumps(lock, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORTS / "development_mechanism_lock.json").write_text(json.dumps(lock, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(lock, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
