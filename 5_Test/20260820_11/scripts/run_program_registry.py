from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
HERE = ROOT / "5_Test" / "20260820_11"
OUT = HERE / "outputs"
REPORTS = HERE / "reports"
RESIDUAL_PATH = ROOT / "5_Test" / "20260818_5" / "outputs" / "seasonal_residual_metrics.parquet"
MONTHLY_PATH = ROOT / "5_Test" / "20260815_2" / "outputs" / "reach_month_n_inputs_hydrology_1961_2022.parquet"
PARENT_LOCK = ROOT / "5_Test" / "20260820_10" / "final_lock.json"
PARENT_SHARED = ROOT / "5_Test" / "20260818_1" / "scripts" / "legacy18_shared.py"
OBS_PATH = ROOT / "5_Test" / "20260815_1" / "outputs" / "tn_observation_registry_2016_2022.parquet"
FOLD_PATH = ROOT / "5_Test" / "20260815_1" / "outputs" / "tn_fold_registry.parquet"
REGISTERED = (2, 3, 7, 10, 11, 12)
LOCKED_SIGN = {2: -1, 3: 1, 7: 1, 10: -1, 11: 1, 12: 1}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    parent = json.loads(PARENT_LOCK.read_text(encoding="utf-8"))
    if parent.get("selected_mechanism") != "PARENT":
        raise RuntimeError("STOP_PARENT_20260820_10_NOT_RETAINED")

    inputs = [RESIDUAL_PATH, MONTHLY_PATH, PARENT_LOCK, PARENT_SHARED, OBS_PATH, FOLD_PATH]
    dump(REPORTS / "input_hash_manifest.json", {str(path): sha256(path) for path in inputs})

    residual = pd.read_parquet(RESIDUAL_PATH)
    locked = (
        residual.loc[residual.month.isin(REGISTERED)]
        .groupby("month", as_index=False)
        .agg(
            median_B=("median_effect", "median"),
            min_B=("median_effect", "min"),
            max_B=("median_effect", "max"),
            model_count=("model_id", "nunique"),
        )
    )
    locked["locked_expected_sign"] = locked.month.map(LOCKED_SIGN).astype(int)
    locked["observed_sign"] = np.sign(locked.median_B).astype(int)
    locked["sign_reproduced"] = locked.observed_sign.eq(locked.locked_expected_sign)

    monthly = pd.read_parquet(MONTHLY_PATH)
    dev = monthly.loc[monthly.year.between(2016, 2021)].copy()
    if (dev["negative_legacy_eligible_n_surplus_kg_n_month"] > 0).any():
        raise RuntimeError("STOP_UNREGISTERED_NEGATIVE_REACH_YEAR")
    hydro = (
        dev.groupby("month", as_index=False)
        .agg(
            uniform_annual12_diffuse_n_kg=("positive_legacy_eligible_n_surplus_kg_n_month", "sum"),
            local_total_q_mm=("q_local_total_mm", "mean"),
            local_quick_q_mm=("quick_release_mm", "mean"),
            local_gw_q_mm=("gw_discharge_mm", "mean"),
        )
    )
    hydro["uniform_weight"] = 1.0 / 12.0
    denominator = hydro.local_quick_q_mm + hydro.local_gw_q_mm
    hydro["quick_fraction"] = np.divide(hydro.local_quick_q_mm, denominator, out=np.zeros(len(hydro)), where=denominator > 1e-12)
    hydro["gw_fraction"] = 1.0 - hydro.quick_fraction
    audit = hydro.merge(locked, on="month", how="left", validate="one_to_one")
    audit.to_parquet(OUT / "a0_hypothesis_plausibility_audit.parquet", index=False)

    d23 = float(locked.set_index("month").loc[2, "median_B"] - locked.set_index("month").loc[3, "median_B"])
    d101112 = float(
        locked.set_index("month").loc[10, "median_B"]
        - 0.5 * (locked.set_index("month").loc[11, "median_B"] + locked.set_index("month").loc[12, "median_B"])
    )
    reproduced = bool(locked.sign_reproduced.all() and len(locked) == 6)
    result = {
        "scenario_id": "20260820_11",
        "status": "PASS",
        "decision": "hypothesis_plausible" if reproduced and d23 < 0 and d101112 < 0 else "hypothesis_not_strengthened",
        "decision_is_gate_for_A0": False,
        "registered_months": list(REGISTERED),
        "residual_definition": "ln1p_observed_minus_ln1p_predicted",
        "all_registered_signs_reproduced": reproduced,
        "D_2_to_3": d23,
        "D_10_to_11_12": d101112,
        "negative_surplus_rows_2016_2021": int((dev.negative_legacy_eligible_n_surplus_kg_n_month > 0).sum()),
        "interpretation_boundary": "descriptive development evidence only; not an observed fertilizer calendar and not an independent confirmation",
        "TN_2022_materialized": False,
    }
    dump(REPORTS / "a0_hypothesis_plausibility_audit.json", result)

    manifest_path = HERE / "program_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["scenarios"][0]["status"] = "passed"
    manifest["scenarios"][0]["decision"] = result["decision"]
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    report = f"""# A0 hypothesis plausibility audit\n\n- Status: **PASS**\n- Decision: **{result['decision']}** (non-gating)\n- Registered signs reproduced: **{reproduced}**\n- `D_2_to_3`: {d23:.6f}\n- `D_10_to_11_12`: {d101112:.6f}\n- Negative surplus rows in 2016–2021: 0\n\nThis audit reproduces an already-registered development residual fingerprint. It does not fit `a,b`, does not access 2022 TN, and cannot identify a fertilizer calendar or true monthly source availability.\n"""
    (REPORTS / "technical_report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
