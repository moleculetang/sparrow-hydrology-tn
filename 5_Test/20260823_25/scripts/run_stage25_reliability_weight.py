from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
TEST = ROOT / "5_Test"
RUN = TEST / "20260823_25"
OUT = RUN / "outputs"
REPORT = RUN / "reports"
sys.path.insert(0, str(TEST / "20260823_17" / "scripts"))
from regionalization import station_metrics, summary_metrics  # noqa: E402


PRED = TEST / "20260823_24" / "outputs" / "zero_history_spatial_predictions.parquet"
WEIGHTS = [i / 10 for i in range(11)]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def bootstrap_upper(frame: pd.DataFrame, pred: str, reference: str, n_boot: int = 10000) -> tuple[float, float, float]:
    a = station_metrics(frame, pred).set_index("q_site").RMSE_log
    b = station_metrics(frame, reference).set_index("q_site").RMSE_log
    diff = (a - b).dropna().to_numpy(float)
    rng = np.random.default_rng(20260823)
    boot = np.array([rng.choice(diff, len(diff), replace=True).mean() for _ in range(n_boot)])
    return float(diff.mean()), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    contract = json.loads((RUN / "experiment_contract.json").read_text(encoding="utf-8"))
    if contract["status"] != "registered_before_weight_selection":
        raise RuntimeError("Weight contract not pre-registered")
    pred = pd.read_parquet(PRED)
    q72 = summary_metrics(pred, "q72_routed_total_cfs")
    log_q72 = np.log1p(pred.q72_routed_total_cfs.to_numpy(float))
    log_spatial = np.log1p(pred.Q_GAUGE_ZERO_HISTORY_cfs.to_numpy(float))
    rows, prediction_parts = [], []
    for weight in WEIGHTS:
        col = f"Q_hybrid_{int(weight*10):02d}"
        pred[col] = np.maximum(np.expm1(log_q72 + weight * (log_spatial - log_q72)), 0.0)
        metrics = summary_metrics(pred, col)
        delta, lower, upper = bootstrap_upper(pred, col, "q72_routed_total_cfs")
        gates = {
            "bootstrap_noninferior": upper < 0.03,
            "pooled_NSE_noninferior": q72["NSE"] - metrics["NSE"] <= 0.02,
            "station_median_NSE_noninferior": q72["station_median_NSE"] - metrics["station_median_NSE"] <= 0.03,
            "PBIAS_worsening_within_5pp": abs(metrics["PBIAS_pct"]) - abs(q72["PBIAS_pct"]) <= 5.0,
        }
        rows.append({
            "weight": weight, **metrics, "delta_station_mean_RMSE_log_vs_Q72": delta,
            "bootstrap_ci95_lower": lower, "bootstrap_ci95_upper": upper,
            **gates, "eligible": all(gates.values()),
        })
    grid = pd.DataFrame(rows)
    eligible = grid[grid.eligible]
    selected = float(eligible.weight.max()) if len(eligible) else 0.0
    selected_col = f"Q_hybrid_{int(selected*10):02d}"
    result = pred[["q_site", "reach_id", "year", "month", "Q_obsv_cfs", "q72_routed_total_cfs", "Q_GAUGE_ZERO_HISTORY_cfs", selected_col]].copy()
    result = result.rename(columns={selected_col: "Q_UNKNOWN_RELIABILITY_BLEND_cfs"})
    result["selected_unknown_weight"] = selected
    result.to_parquet(OUT / "unknown_gauge_reliability_predictions.parquet", index=False)
    grid.to_parquet(OUT / "unknown_gauge_weight_grid.parquet", index=False)
    decision = {
        "stage": "20260823_25",
        "status": "NONZERO_SPATIAL_WEIGHT_SUPPORTED" if selected > 0 else "SPATIAL_WEIGHT_ZERO_Q72_FALLBACK_LOCKED",
        "selected_unknown_weight": selected,
        "known_gauge_weight": 1.0,
        "eligible_weight_count": int(len(eligible)),
        "spatial_search_stops": bool(selected == 0),
        "reach_flow_for_unknown_reaches": "Q72_CONSERVING",
        "external_four_stations_read": False,
    }
    (REPORT / "stage25_decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORT / "technical_report.md").write_text(
        "# 20260823_25 unknown-gauge reliability weight\n\n"
        + f"Status: `{decision['status']}`; selected weight={selected}.\n\n"
        + grid.to_markdown(index=False) + "\n\n"
        + "Known gauges retain weight 1 after historical conditioning. Unknown gauges use the selected weight; this layer never alters Reach water routing.\n",
        encoding="utf-8",
    )
    (REPORT / "integrity.json").write_text(json.dumps({
        "contract_sha256": sha256(RUN / "experiment_contract.json"),
        "input_predictions_sha256": sha256(PRED),
        "grid_sha256": sha256(OUT / "unknown_gauge_weight_grid.parquet"),
    }, indent=2), encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    print(grid.to_string(index=False))


if __name__ == "__main__":
    main()
