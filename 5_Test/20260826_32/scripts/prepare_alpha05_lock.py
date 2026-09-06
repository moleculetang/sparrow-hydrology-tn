"""Create the disclosed post-retrospective alpha=0.5 engineering repair lock."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260826_32"
OUT = RUN / "outputs"
REPORTS = RUN / "reports"
STAGE28 = ROOT / "5_Test" / "20260826_28"
sys.path[:0] = [str(STAGE28 / "scripts"), str(ROOT / "5_Test" / "20260826_29" / "scripts")]

from run_stage28 import FinalTwoPath, RAW_PARAMETER_NAMES, state_dict_frame  # noqa: E402
from run_stage29 import restore_state  # noqa: E402


ALPHA = 0.5
PARENT_LOCK = ROOT / "5_Test" / "20260825_7" / "reports" / "full_development_parameter_lock.json"
FINAL_LOCK = STAGE28 / "reports" / "full_development_parameter_lock.json"
FINAL_STATE = STAGE28 / "outputs" / "full_development_model_state.parquet"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    contract = {
        "stage": "20260826_32",
        "role": "SECOND_AND_FINAL_INTEGRITY_REPAIR",
        "post_retrospective": True,
        "independent_validation_claim_forbidden": True,
        "allowed_change": "fixed alpha=0.5 common shrinkage of final raw-parameter offsets and gate strength toward the old parent",
        "development_evidence": {
            "retained_fraction_of_final_monthly_log_MSE_gain": 0.719,
            "monthly_pooled_NSE_drop_vs_parent": 0.001004,
            "monthly_station_median_NSE_change_vs_parent": 0.003512,
            "monthly_station_median_absolute_PBIAS": 14.392341,
        },
        "forbidden": ["alpha search on 2019-2022", "new weights", "regional alpha", "station correction", "third repair"],
        "TN_read": False,
    }
    write_json(RUN / "experiment_contract.json", contract)
    parent = json.loads(PARENT_LOCK.read_text(encoding="utf-8"))
    final = json.loads(FINAL_LOCK.read_text(encoding="utf-8"))
    parent_raw = np.asarray([parent["raw_parameters"][name] for name in RAW_PARAMETER_NAMES], dtype=np.float64)
    final_raw = np.asarray([final["raw_parameters"][name] for name in RAW_PARAMETER_NAMES], dtype=np.float64)
    repaired_raw = parent_raw + ALPHA * (final_raw - parent_raw)
    model = FinalTwoPath(int(final["selected_seed"]))
    restore_state(model, pd.read_parquet(FINAL_STATE))
    final_strength = float(model.gate.strength().detach())
    repaired_strength = ALPHA * final_strength
    probability = 2.0 * repaired_strength
    repaired_gate_raw = math.log(probability / (1.0 - probability))
    repaired_offset = np.arctanh(np.clip((repaired_raw - parent_raw) / 1.5, -0.999999999, 0.999999999))
    with torch.no_grad():
        model.raw_offset.copy_(torch.tensor(repaired_offset, dtype=torch.float64))
        model.gate.raw_gate_strength.copy_(torch.tensor(repaired_gate_raw, dtype=torch.float64))
    state_dict_frame(model).to_parquet(OUT / "repaired_model_state.parquet", index=False)
    lock = {
        "stage": "20260826_32", "status": "POST_RETROSPECTIVE_ALPHA05_ENGINEERING_LOCK",
        "alpha": ALPHA, "selected_seed": int(final["selected_seed"]),
        "raw_parameters": dict(zip(RAW_PARAMETER_NAMES, repaired_raw.tolist())),
        "gate_strength": repaired_strength,
        "source_final_lock": str(FINAL_LOCK),
        "retrospective_used_to_estimate_alpha": False,
        "retrospective_failure_trigger_disclosed": True,
        "independent_validation_claim": False,
        "TN_read": False,
    }
    write_json(REPORTS / "repair_parameter_lock.json", lock)
    print(json.dumps(lock, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
