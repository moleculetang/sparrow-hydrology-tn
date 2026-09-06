from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    obs = frame["actual"].to_numpy(float)
    pred = frame["predict"].to_numpy(float)
    log_obs, log_pred = np.log(obs), np.log(pred)
    return {
        "n": len(frame),
        "raw_nse": 1 - np.sum((pred-obs)**2) / np.sum((obs-obs.mean())**2),
        "log_nse": 1 - np.sum((log_pred-log_obs)**2) / np.sum((log_obs-log_obs.mean())**2),
        "pbias_pct": 100*np.sum(pred-obs)/np.sum(obs),
        "rmse": np.sqrt(np.mean((pred-obs)**2)),
        "log_rmse": np.sqrt(np.mean((log_pred-log_obs)**2)),
    }


def main() -> None:
    gate_path = ROOT/"logs"/"G5_G8_dynamic_inference.json"
    inference_gate = json.loads(gate_path.read_text(encoding="utf-8")) if gate_path.exists() else {}
    i1_scientifically_admissible = bool(inference_gate) and all(row.get("identifiability_pre_gate", False) for row in inference_gate.get("folds", []))
    scenarios = [name for name in ["B0", "I0"] if (ROOT/"outputs"/name/"q72_three_fold_oof_predictions.parquet").exists()]
    if i1_scientifically_admissible and (ROOT/"outputs"/"I1"/"q72_three_fold_oof_predictions.parquet").exists():
        scenarios.append("I1")
    rows, fold_rows = [], []
    frames = {}
    for name in scenarios:
        frame = pd.read_parquet(ROOT/"outputs"/name/"q72_three_fold_oof_predictions.parquet")
        frames[name] = frame
        rows.append({"scenario": name, **metrics(frame)})
        for fold, part in frame.groupby("fold_id"):
            fold_rows.append({"scenario": name, "fold_id": fold, **metrics(part)})
    pd.DataFrame(rows).to_csv(ROOT/"reports"/"scenario_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(fold_rows).to_csv(ROOT/"reports"/"fold_metrics.csv", index=False, encoding="utf-8-sig")
    key = ["comid", "q_site", "year", "month", "fold_id"]
    identity = {"scenarios": scenarios, "rows": {k: len(v) for k,v in frames.items()}, "i1_scientifically_admissible": i1_scientifically_admissible}
    if "B0" in frames:
        base = frames["B0"].sort_values(key).reset_index(drop=True)
        identity["all_keys_equal_b0"] = all(base[key].equals(frame.sort_values(key).reset_index(drop=True)[key]) for frame in frames.values())
    (ROOT/"logs"/"oof_identity.json").write_text(json.dumps(identity, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
