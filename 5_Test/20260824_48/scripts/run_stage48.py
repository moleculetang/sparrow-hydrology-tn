"""Preregistered Stage48 joint-opening and temperature-readiness audit."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test/20260824_48"
REPORTS = RUN / "reports"
LOCKS = RUN / "locks"
STAGE47_LOCK = ROOT / "5_Test/20260824_47/locks/stage47_lock.json"
GW = ROOT / "5_Test/20260814_9/inputs/model_ready/static/groundwater_temperature_benz_2024_by_reach.parquet"
LAKE = ROOT / "5_Test/20260814_9/inputs/model_ready/monthly/lake_mix_layer_temperature_era5_land_by_reach_month_2006_2022.parquet"
TEMP_QA = ROOT / "5_Test/20260814_9/inputs/provenance/temperature_covariates_qa.json"
HYDRO = ROOT / "5_Test/20260828_9/outputs/canonical_reach_monthly_2006_2024.parquet"
CONTRACT = RUN / "experiment_contract.json"
MANIFEST = RUN / "program_manifest.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def main() -> None:
    lock = json.loads(STAGE47_LOCK.read_text(encoding="utf-8"))
    gw = pd.read_parquet(GW)
    lake = pd.read_parquet(LAKE)
    hydro = pd.read_parquet(HYDRO)
    qa = json.loads(TEMP_QA.read_text(encoding="utf-8"))
    eligible = set(lock.get("eligible_candidates", []))
    joint_authorized = {"REG3_OBSERVATION", "DYN_HYDRO_DELIVERY"}.issubset(eligible)
    checks = {
        "stage47_passed": bool(lock.get("status") == "PASS_STAGE47_SPATIAL_VALIDATION"),
        "groundwater_rows_230": bool(len(gw) == 230 and gw.reach_id.nunique() == 230),
        "groundwater_is_static_role": bool("static" in " ".join(gw.groundwater_temperature_temporal_role.astype(str).unique()).lower()),
        "lake_rows_230x17x12": bool(len(lake) == 230 * 17 * 12 and lake.reach_id.nunique() == 230),
        "lake_period_2006_2022": bool(int(lake.year.min()) == 2006 and int(lake.year.max()) == 2022),
        "lake_proxy_label_present": bool(lake.surface_temperature_role.astype(str).str.lower().str.contains("proxy").all()),
        "temperature_source_qa_pass": bool(qa.get("status") == "PASS"),
        "canonical_hydrology_2006_2024": bool(int(hydro.year.min()) == 2006 and int(hydro.year.max()) == 2024),
        "canonical_hydrology_230_reaches": bool(hydro.reach_id.nunique() == 230),
        "canonical_travel_time_present": bool("channel_bankfull_travel_time_central_day" in hydro.columns),
    }
    engineering_pass = all(checks.values())
    aquatic_q10_authorized = False
    if joint_authorized:
        status = "JOINT_AUTHORIZED_BUT_NOT_RUN_BY_THIS_AUDIT"
    else:
        status = "PASS_STAGE48_JOINT_CLOSED_TEMPERATURE_BLOCKED_READY_FOR_STAGE49"
    if not engineering_pass:
        status = "FAIL_STAGE48_ENGINEERING"
    decision = {
        "stage": "20260824_48",
        "status": status,
        "stage47_eligible_candidates": sorted(eligible),
        "joint_candidate_authorized": joint_authorized,
        "joint_candidate_action": "RUN_REG3_PLUS_DYN" if joint_authorized else "CLOSED_BY_PREREGISTERED_STAGE47_GATE",
        "temperature": {
            "groundwater_role": "static spatial thermal-state/trend covariate; not monthly forcing",
            "lake_role": "lake/reservoir mixed-layer proxy sensitivity only",
            "lake_period": [int(lake.year.min()), int(lake.year.max())],
            "formal_tn_period": [2021, 2024],
            "all_reach_aquatic_q10_authorized": aquatic_q10_authorized,
            "reason": "No formal all-reach river-water temperature forcing; lake proxy ends in 2022 and is scope-mismatched."
        },
        "checks": checks,
        "input_hashes": {str(path): sha256(path) for path in [STAGE47_LOCK, GW, LAKE, TEMP_QA, HYDRO, CONTRACT, MANIFEST]},
        "authorized_successor": "20260824_49" if engineering_pass and not joint_authorized else None,
    }
    atomic_json(decision, REPORTS / "stage48_decision.json")
    atomic_json({
        "stage": "20260824_48",
        "status": status,
        "joint_candidate_authorized": joint_authorized,
        "all_reach_aquatic_q10_authorized": aquatic_q10_authorized,
        "decision_sha256": sha256(REPORTS / "stage48_decision.json"),
        "authorized_successor": decision["authorized_successor"],
    }, LOCKS / "stage48_lock.json")
    report = f"""# `20260824_48` conditional joint and temperature readiness audit

Status: `{status}`.

- Stage47 spatially eligible candidates: `{sorted(eligible)}`.
- REG3+DYN joint candidate authorized: `{joint_authorized}`.
- All-reach aquatic Q10 authorized: `{aquatic_q10_authorized}`.
- Groundwater temperature remains a static spatial thermal-state/trend covariate, not monthly forcing.
- ERA5-Land `lmlt` is a lake/reservoir mixed-layer proxy covering 2006–2022; it is neither observed river temperature nor complete for the 2021–2024 formal fit period.
- Canonical hydrology covers 230 reaches through 2024 and already supplies flow-derived channel travel time; no reference discharge is used.

Therefore no temperature coefficient or joint candidate is fitted in this stage. The registered successor is Stage49 final synthesis and lock.
"""
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "technical_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    if not engineering_pass:
        raise RuntimeError(status)


if __name__ == "__main__":
    main()
