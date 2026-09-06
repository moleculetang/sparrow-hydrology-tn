from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


RUN = Path(r"E:\SPARROW\5_Test\20260813_30")
PARENT = Path(r"E:\SPARROW\5_Test\20260813_25\outputs\P1\q72_three_fold_oof_predictions.parquet")
NEW = RUN / "outputs" / "P1" / "q72_three_fold_oof_predictions.parquet"
KEY = ["comid", "q_site", "year", "month", "fold_id"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def metrics(frame: pd.DataFrame) -> dict[str, float]:
    obs = frame.actual.to_numpy(float)
    pred = frame.predict.to_numpy(float)
    log_obs = np.log(obs)
    log_pred = np.log(pred)
    return {
        "raw_nse": float(1.0 - np.square(obs - pred).sum() / np.square(obs - obs.mean()).sum()),
        "log_nse": float(1.0 - np.square(log_obs - log_pred).sum() / np.square(log_obs - log_obs.mean()).sum()),
        "rmse": float(np.sqrt(np.mean(np.square(obs - pred)))),
        "log_rmse": float(np.sqrt(np.mean(np.square(log_obs - log_pred)))),
        "pbias_pct": float(100.0 * (pred - obs).sum() / obs.sum()),
    }


def main() -> None:
    parent = pd.read_parquet(PARENT).sort_values(KEY).reset_index(drop=True)
    new = pd.read_parquet(NEW).sort_values(KEY).reset_index(drop=True)
    key_equal = parent[KEY].equals(new[KEY])
    actual_max = float(np.max(np.abs(parent.actual.to_numpy(float) - new.actual.to_numpy(float))))
    rows = []
    for label, frame in [("parent", parent), ("d8_supported", new)]:
        for fold, group in frame.groupby("fold_id", sort=True):
            rows.append({"scenario": label, "fold_id": fold, "rows": len(group), **metrics(group)})
        rows.append({"scenario": label, "fold_id": "pooled", "rows": len(frame), **metrics(frame)})
    table = pd.DataFrame(rows)
    table.to_csv(RUN / "reports" / "rebuilt_p1_fold_metrics.csv", index=False, encoding="utf-8-sig")
    old = table[table.scenario.eq("parent")].set_index("fold_id")
    repaired = table[table.scenario.eq("d8_supported")].set_index("fold_id")
    delta = repaired.drop(columns=["scenario", "rows"]) - old.drop(columns=["scenario", "rows"])
    delta.reset_index().to_csv(RUN / "reports" / "rebuilt_p1_metric_deltas.csv", index=False, encoding="utf-8-sig")
    folds = repaired.drop(index="pooled")
    payload = {
        "parent_oof_sha256": sha256(PARENT),
        "repaired_oof_sha256": sha256(NEW),
        "rows_parent": int(len(parent)),
        "rows_repaired": int(len(new)),
        "key_equal": bool(key_equal),
        "actual_max_abs_difference": actual_max,
        "fold_metrics": repaired.reset_index().to_dict(orient="records"),
        "pooled_metrics": repaired.loc["pooled"].to_dict(),
        "all_three_raw_nse_gt_0_95": bool((folds.raw_nse > 0.95).all()),
        "all_three_log_nse_gt_0_95": bool((folds.log_nse > 0.95).all()),
        "strict_dual_nse_target_met": bool(((folds.raw_nse > 0.95) & (folds.log_nse > 0.95)).all()),
    }
    payload["engineering_population_gate_pass"] = bool(len(parent) == len(new) == 8738 and key_equal and actual_max <= 1e-12)
    (RUN / "reports" / "rebuilt_p1_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if not payload["engineering_population_gate_pass"]:
        raise RuntimeError(payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
