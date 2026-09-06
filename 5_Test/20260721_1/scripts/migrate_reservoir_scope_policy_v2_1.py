from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


TEST_ROOT = Path(r"E:\SPARROW\5_Test")
CONTROL = TEST_ROOT / "20260721_1"
CONTROL_REPORT = CONTROL / "reports" / "dynamic_station_screening"
STATE_PATH = CONTROL_REPORT / "chain_state.json"
DECISION_POLICY = CONTROL / "inputs" / "source_metadata" / "station_screening_decision_policy_v2.json"
COMPARATOR = CONTROL / "scripts" / "compare_station_ablation.py"
EVIDENCE = CONTROL / "scripts" / "build_station_screening_evidence.py"
AUDIT_SCRIPT = CONTROL / "scripts" / "run_all_station_influence_audit.py"


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def composite_hash(station_policy: Path) -> str:
    digest = hashlib.sha256()
    digest.update(b"station-screening-policy\0")
    digest.update(station_policy.read_bytes())
    digest.update(b"\0decision-policy\0")
    digest.update(DECISION_POLICY.read_bytes())
    return digest.hexdigest()


def main() -> None:
    migrated_at = datetime.now().isoformat(timespec="seconds")
    state = read_json(STATE_PATH)
    policy = read_json(DECISION_POLICY)
    version = str(policy["policy_version"])
    if version != "v2.1":
        raise RuntimeError(f"Expected decision policy v2.1, found {version}")
    base_hash = composite_hash(CONTROL / "inputs" / "source_metadata" / "station_screening_policy.csv")
    accepted_parent = TEST_ROOT / str(state["accepted_parent_run"])
    accepted_hash = composite_hash(accepted_parent / "inputs" / "source_metadata" / "station_screening_policy.csv")

    for number in range(3, 20):
        trial = TEST_ROOT / f"20260721_{number}"
        if not trial.exists():
            continue
        shutil.copy2(DECISION_POLICY, trial / "inputs" / "source_metadata" / DECISION_POLICY.name)
        if (trial / "scripts" / "compare_station_ablation.py").exists():
            shutil.copy2(COMPARATOR, trial / "scripts" / "compare_station_ablation.py")
        if (trial / "scripts" / "build_station_screening_evidence.py").exists():
            shutil.copy2(EVIDENCE, trial / "scripts" / "build_station_screening_evidence.py")
        if (trial / "scripts" / "run_all_station_influence_audit.py").exists():
            shutil.copy2(AUDIT_SCRIPT, trial / "scripts" / "run_all_station_influence_audit.py")
        manifest_path = trial / "inputs" / "source_metadata" / "run_experiment.json"
        if manifest_path.exists():
            manifest = read_json(manifest_path)
            manifest["decision_policy_version"] = version
            manifest["reservoir_scope_migrated_at"] = migrated_at
            if number <= 18:
                manifest["accepted_policy_sha256"] = base_hash
            elif number == 19:
                manifest["accepted_policy_sha256"] = accepted_hash
            write_json(manifest_path, manifest)

    for number in range(3, 18):
        trial = TEST_ROOT / f"20260721_{number}"
        experiment = read_json(trial / "inputs" / "source_metadata" / "run_experiment.json")
        subprocess.run(
            [
                sys.executable,
                str(trial / "scripts" / "compare_station_ablation.py"),
                "--base",
                str(TEST_ROOT / "20260721_2"),
                "--trial",
                str(trial),
                "--candidate",
                str(experiment["candidate_station"]),
            ],
            cwd=trial,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        readme_path = trial / "README.md"
        text = readme_path.read_text(encoding="utf-8").replace("Policy v2 re-evaluation", "Policy v2.1 re-evaluation")
        text = text.replace("fixed policy v2 review", "fixed policy v2.1 review")
        text = text.replace("Policy v2 also permits", "Policy v2.1 also permits")
        readme_path.write_text(text, encoding="utf-8")

    ledger_path = CONTROL_REPORT / "decision_ledger.csv"
    ledger = pd.read_csv(ledger_path, encoding="utf-8-sig")
    ledger["policy_version"] = version
    ledger["policy_sha256"] = base_hash
    ledger.to_csv(ledger_path, index=False, encoding="utf-8-sig")
    status_path = CONTROL_REPORT / "station_status_ledger.csv"
    status = pd.read_csv(status_path, encoding="utf-8-sig")
    status["decision"] = status["decision"].astype(str).str.replace(r"policy_v2_[0-9a-f]{12}", f"policy_v2_1_{base_hash[:12]}", regex=True)
    status.to_csv(status_path, index=False, encoding="utf-8-sig")

    summary_path = CONTROL_REPORT / "policy_v2_recalculation_summary.json"
    summary = read_json(summary_path)
    summary.update(
        {
            "policy_version": version,
            "reservoir_scope_migrated_at": migrated_at,
            "base_policy_sha256": base_hash,
            "accepted_policy_sha256": accepted_hash,
            "manual_reservoir_name_tokens": policy["reservoir_scope"]["defer_station_name_contains"],
        }
    )
    write_json(summary_path, summary)
    summary_csv_path = CONTROL_REPORT / "policy_v2_recalculation_summary.csv"
    summary_csv = pd.read_csv(summary_csv_path, encoding="utf-8-sig")
    summary_csv["policy_version"] = version
    summary_csv.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")

    state.update({"policy_version": version, "policy_sha256": accepted_hash})
    write_json(STATE_PATH, state)
    print(
        json.dumps(
            {
                "migrated": True,
                "policy_version": version,
                "base_policy_sha256": base_hash,
                "accepted_policy_sha256": accepted_hash,
                "active_trial": state.get("active_trial"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
