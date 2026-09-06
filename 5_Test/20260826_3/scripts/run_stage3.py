from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260826_3"
OUT = RUN / "outputs"
REPORT = RUN / "reports"
sys.path.insert(0, str(RUN / "scripts"))
from hydrologic_structures import DEFAULT_PARAMETERS, RUNNERS, periodic_spinup  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def response_centroid(response: np.ndarray) -> float:
    total = float(response.sum())
    return float(np.dot(np.arange(len(response), dtype=float), response) / total) if total > 0 else float("nan")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True); REPORT.mkdir(parents=True, exist_ok=True)
    days = np.arange(365 * 4)
    p_spin = 3.5 + 2.5 * np.sin(2 * np.pi * days / 365.25)
    p_spin = np.maximum(0.0, p_spin)
    pet_spin = 2.2 + 1.8 * np.sin(2 * np.pi * (days - 90) / 365.25)
    pet_spin = np.maximum(0.0, pet_spin)
    pulse_p = np.zeros(900); pulse_p[0] = 100.0
    pulse_pet = np.zeros_like(pulse_p)
    rows: list[dict[str, object]] = []
    full_results: dict[str, object] = {}
    for model, runner in RUNNERS.items():
        parameters = DEFAULT_PARAMETERS[model]
        spin_state, spin_audit = periodic_spinup(runner, p_spin, pet_spin, parameters)
        pulse = runner(pulse_p, pulse_pet, parameters, spin_state)
        control = runner(np.zeros_like(pulse_p), pulse_pet, parameters, spin_state)
        split = 317
        first = runner(pulse_p[:split], pulse_pet[:split], parameters, spin_state)
        second = runner(pulse_p[split:], pulse_pet[split:], parameters, first.final_state_mm)
        restart_response = np.vstack([first.response_mm, second.response_mm])
        restart_states = np.vstack([first.states_mm, second.states_mm])
        incremental_response = np.maximum(0.0, pulse.response_mm - control.response_mm)
        centroid = [response_centroid(incremental_response[:, component]) for component in range(3)]
        mass_error = max(float(np.max(np.abs(pulse.mass_error_mm))), float(spin_audit["max_abs_mass_error_mm"]))
        restart_delta = max(float(np.max(np.abs(pulse.response_mm - restart_response))), float(np.max(np.abs(pulse.states_mm - restart_states))))
        nonnegative = bool((pulse.response_mm >= -1e-12).all() and (pulse.states_mm >= -1e-12).all())
        ordering = bool(centroid[0] < centroid[1] < centroid[2])
        passed = bool(spin_audit["converged"] and mass_error <= 1e-10 and restart_delta <= 1e-12 and nonnegative and ordering)
        row = {
            "model_id": model,
            "spinup_converged": bool(spin_audit["converged"]),
            "spinup_cycles": int(spin_audit["cycles"]),
            "max_abs_mass_error_mm": mass_error,
            "restart_max_abs_delta": restart_delta,
            "nonnegative": nonnegative,
            "fast_centroid_day": centroid[0],
            "intermediate_centroid_day": centroid[1],
            "slow_centroid_day": centroid[2],
            "response_ordering_passed": ordering,
            "all_preflight_passed": passed,
        }
        rows.append(row); full_results[model] = {"parameters": parameters.tolist(), "spinup": spin_audit, "tests": row}
    tests = pd.DataFrame(rows)
    tests.to_parquet(OUT / "synthetic_structure_preflight.parquet", index=False)

    # HBV parent semantics are inherited only after exact reproduction in Stage 2.
    parent_reproduction = json.loads((ROOT / "5_Test" / "20260826_2" / "reports" / "parent_exact_reproduction.json").read_text(encoding="utf-8"))
    parent_pass = bool(parent_reproduction["exact_reproduction_passed"] and parent_reproduction["full_period_land_mass_max_abs_error_mm"] <= 1e-10)
    registry = {
        "HBV3_PARENT": {"implementation": "frozen 20260825 Raven-ordered HBV core", "preflight": "Stage 2 exact replay", "passed": parent_pass},
        "RAVEN_SACSMA3": {"implementation": "mass-conserving SAC-SMA response subset", "not_claimed": "bitwise full operational NWS SAC-SMA"},
        "RAVEN_TOPMODEL3": {"implementation": "TOPMODEL-style saturation area and nonlinear baseflow response"},
        "HYPE3L_HYDROLOGY": {"implementation": "HYPE-style hydrology-only three-layer soil response", "not_claimed": "full HYPE nutrient or lake model"},
        "MTRS3": {"implementation": "registered mass-conserving multi-timescale response stores"},
        "common_output": ["fast_response", "intermediate_response", "slow_response"],
        "claim_boundary": "model response components; not direct observations of source water or age",
    }
    write_json(REPORT / "equation_implementation_registry.json", registry)
    write_json(REPORT / "synthetic_structure_preflight.json", full_results)
    all_pass = bool(parent_pass and tests.all_preflight_passed.all())
    decision = {
        "stage": "20260826_3",
        "status": "PASS_ALL_STRUCTURE_NUMERICAL_AND_SEMANTIC_PREFLIGHTS" if all_pass else "STAGE3_PREFLIGHT_FAILED",
        "parent_preflight_passed": parent_pass,
        "candidate_preflight_passed": int(tests.all_preflight_passed.sum()),
        "candidate_count": int(len(tests)),
        "observed_discharge_used": False,
        "TN_used": False,
        "authorized_successor": "20260826_4" if all_pass else None,
    }
    write_json(REPORT / "stage3_decision.json", decision)
    failed = tests.loc[~tests.all_preflight_passed, "model_id"].tolist()
    report = f"""# 20260826_3 结构实现与合成预检

## 结论

状态：`{decision['status']}`。四个挑战结构中`{decision['candidate_preflight_passed']}/{decision['candidate_count']}`通过周期spin-up、逐日质量闭合、非负性、restart等价和脉冲响应快<中<慢语义测试；失败列表：`{failed}`。HBV父模型沿用Stage 2的逐字段零差重放测试。

本阶段没有读取观测流量或TN。`SAC-SMA3`、`TOPMODEL3`和`HYPE3L`名称表示其标准储库角色与产汇流方程族；报告明确记录了实现子集，避免把本地实现误称为完整业务软件。所有输出仍只称模型响应分量。
"""
    (REPORT / "technical_report.md").write_text(report, encoding="utf-8")
    write_json(REPORT / "integrity.json", {str(path): sha256(path) for path in [RUN / "experiment_contract.json", RUN / "scripts" / "hydrologic_structures.py", OUT / "synthetic_structure_preflight.parquet", REPORT / "stage3_decision.json", REPORT / "technical_report.md"]})
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
