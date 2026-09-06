from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
NEW_PATH = ROOT / "outputs" / "W1" / "q72_three_fold_oof_predictions.parquet"
OLD_PATH = Path(r"E:\SPARROW\5_Test\20260813_36\outputs\A1\q72_three_fold_oof_predictions.parquet")
KEY = ["comid", "q_site", "year", "month", "fold_id"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def nse(obs: np.ndarray, pred: np.ndarray) -> float:
    return 1.0 - float(np.sum((pred - obs) ** 2)) / float(np.sum((obs - obs.mean()) ** 2))


def metrics(frame: pd.DataFrame, population: str, scenario: str) -> list[dict[str, object]]:
    rows = []
    groups = [("pooled", frame)] + list(frame.groupby("fold_id", sort=True))
    for fold_id, part in groups:
        obs = part.actual.to_numpy(float)
        pred = part.predict.to_numpy(float)
        floor = max(1e-6, 0.001 * float(np.median(obs[obs > 0])))
        rows.append({
            "population": population,
            "scenario": scenario,
            "fold_id": fold_id,
            "rows": len(part),
            "raw_nse": nse(obs, pred),
            "log_nse": nse(np.log(obs + floor), np.log(pred + floor)),
            "rmse_cfs": float(np.sqrt(np.mean((pred - obs) ** 2))),
            "pbias_percent": 100.0 * float(np.sum(pred - obs)) / float(np.sum(obs)),
            "log_floor_cfs": floor,
        })
    return rows


def main() -> None:
    new = pd.read_parquet(NEW_PATH)
    old = pd.read_parquet(OLD_PATH)
    for frame in (new, old):
        frame["q_site"] = frame["q_site"].astype(str)
    common_identity = ["comid", "year", "month", "fold_id"]
    joined = old.merge(
        new,
        on=common_identity,
        suffixes=("_old", "_new"),
        how="inner",
        validate="one_to_one",
    )
    old_common = joined[[*common_identity, "q_site_old", "actual_old", "predict_old"]].rename(
        columns={"q_site_old": "q_site", "actual_old": "actual", "predict_old": "predict"}
    )
    new_common = joined[[*common_identity, "q_site_new", "actual_new", "predict_new"]].rename(
        columns={"q_site_new": "q_site", "actual_new": "actual", "predict_new": "predict"}
    )
    result = []
    result += metrics(old, "parent_full_8822", "reach193_A1")
    result += metrics(new, "spinup_full_8822", "periodic_spinup_W1")
    result += metrics(old_common, "common_reach_month_fold", "reach193_A1")
    result += metrics(new_common, "common_reach_month_fold", "periodic_spinup_W1")
    metric_frame = pd.DataFrame(result)
    metric_frame.to_csv(ROOT / "fold_metrics.csv", index=False, encoding="utf-8-sig")

    station_rows = []
    for scenario, frame in [("reach193_A1", old), ("periodic_spinup_W1", new)]:
        for (comid, q_site), part in frame.groupby(["comid", "q_site"], sort=True):
            obs = part.actual.to_numpy(float)
            pred = part.predict.to_numpy(float)
            floor = max(1e-6, 0.001 * float(np.median(obs[obs > 0])))
            station_rows.append({
                "scenario": scenario, "comid": comid, "q_site": q_site, "rows": len(part),
                "raw_nse": nse(obs, pred),
                "log_nse": nse(np.log(obs + floor), np.log(pred + floor)),
                "rmse_cfs": float(np.sqrt(np.mean((pred - obs) ** 2))),
            })
    pd.DataFrame(station_rows).to_csv(ROOT / "station_metrics.csv", index=False, encoding="utf-8-sig")

    payload = {
        "terminal": "DETERMINISTIC_SPINUP_REPAIR_COMPLETE",
        "parent_rows": len(old),
        "repaired_rows": len(new),
        "common_reach_month_fold_rows": len(joined),
        "added_evaluation_rows": len(new) - len(joined),
        "evaluation_key_population_unchanged": bool(len(joined) == len(old) == len(new)),
        "parent_oof_sha256": sha256(OLD_PATH),
        "repaired_oof_sha256": sha256(NEW_PATH),
        "strict_dual_nse_target_met": bool(
            ((metric_frame.population == "spinup_full_8822") &
             (metric_frame.fold_id != "pooled") &
             (metric_frame.raw_nse > 0.95) &
             (metric_frame.log_nse > 0.95)).sum() == 3
        ),
    }
    (ROOT / "terminal_gate.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
