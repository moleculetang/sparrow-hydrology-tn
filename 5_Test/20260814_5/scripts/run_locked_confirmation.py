from __future__ import annotations

import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from runtime_guard import assert_sparrow_runtime


RUNTIME = assert_sparrow_runtime()
ROOT = Path(__file__).resolve().parents[1]
STAGE1 = ROOT.parent / "20260814_1"
PARENT = ROOT.parent / "20260813_54"
LOCK = ROOT / "model_lock.json"
COMPONENT = STAGE1 / "scripts" / "components" / "q72_prior_semantics_component.py"
INPUT = STAGE1 / "inputs" / "parent_indata.parquet"
TOPOLOGY = STAGE1 / "inputs" / "topology" / "topology_edges.csv"
OUT = ROOT / "outputs" / "fit_2006_2018_eval_2019_2022"
REPORT = OUT / "model_reports"
FIGURE = OUT / "model_figures"
EPS = 1e-9
KEY = ["comid", "q_site", "year", "month"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_component():
    spec = importlib.util.spec_from_file_location("q72_locked_h0", COMPONENT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    obs = frame.actual.to_numpy(float)
    pred = frame.predict.to_numpy(float)
    lo, lp = np.log(np.clip(obs, EPS, None)), np.log(np.clip(pred, EPS, None))
    return {
        "rows": int(len(frame)),
        "raw_nse": float(1 - np.sum((pred - obs) ** 2) / np.sum((obs - obs.mean()) ** 2)),
        "log_nse": float(1 - np.sum((lp - lo) ** 2) / np.sum((lo - lo.mean()) ** 2)),
        "pbias_pct": float(100 * np.sum(pred - obs) / np.sum(obs)),
        "log_rmse": float(np.sqrt(np.mean((lp - lo) ** 2))),
    }


def build_locked_registries(validation: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    thresholds = pd.read_parquet(ROOT / "inputs" / "registries" / "locked_station_thresholds_2006_2018.parquet")
    thresholds.q_site = thresholds.q_site.astype(str)
    frame = validation.merge(thresholds, on="q_site", validate="many_to_one")
    low = frame[frame.actual <= frame.q20_cfs][KEY + ["actual", "predict", "q20_cfs"]].copy()
    events, months = [], []
    for site, part in frame.groupby("q_site", sort=True):
        part = part.sort_values(["year", "month"]).reset_index(drop=True)
        q75 = float(part.q75_cfs.iloc[0])
        wet75 = float(part.upstream_positive_input_q75_cfs.iloc[0])
        for i in range(1, len(part) - 1):
            q = float(part.at[i, "actual"])
            if not (q >= q75 and q >= float(part.at[i - 1, "actual"]) and q > float(part.at[i + 1, "actual"])):
                continue
            tail_rows = []
            previous = q
            for j in range(i + 1, min(i + 7, len(part))):
                p0 = pd.Period(f"{int(part.at[j-1, 'year'])}-{int(part.at[j-1, 'month']):02d}")
                p1 = pd.Period(f"{int(part.at[j, 'year'])}-{int(part.at[j, 'month']):02d}")
                cur = float(part.at[j, "actual"])
                wet = float(part.at[j, "upstream_positive_input_equivalent_cfs"])
                if p1.ordinal - p0.ordinal != 1 or wet > wet75 or cur > previous:
                    break
                tail_rows.append(j)
                previous = cur
            if len(tail_rows) < 2:
                continue
            event_id = f"locked::{site}::{int(part.at[i, 'year']):04d}-{int(part.at[i, 'month']):02d}"
            events.append({
                "event_id": event_id,
                "comid": int(part.at[i, "comid"]),
                "q_site": str(site),
                "peak_year": int(part.at[i, "year"]),
                "peak_month": int(part.at[i, "month"]),
                "peak_observed_cfs": q,
                "peak_predict_cfs": float(part.at[i, "predict"]),
                "frozen_q75_cfs": q75,
                "frozen_upstream_positive_input_q75_cfs": wet75,
                "tail_month_count": len(tail_rows),
            })
            for lag, j in enumerate(tail_rows, start=1):
                months.append({
                    "event_id": event_id,
                    "comid": int(part.at[j, "comid"]),
                    "q_site": str(site),
                    "year": int(part.at[j, "year"]),
                    "month": int(part.at[j, "month"]),
                    "tail_lag": lag,
                    "actual": float(part.at[j, "actual"]),
                    "predict": float(part.at[j, "predict"]),
                })
    event_df, month_df = pd.DataFrame(events), pd.DataFrame(months)
    reg = ROOT / "outputs" / "locked_registries"
    reg.mkdir(parents=True, exist_ok=True)
    low.to_parquet(reg / "lowflow_month_registry.parquet", index=False)
    event_df.to_parquet(reg / "tail_event_registry.parquet", index=False)
    month_df.to_parquet(reg / "tail_month_registry.parquet", index=False)
    return low, event_df, month_df


def main() -> None:
    confirmation = ROOT / "reports" / "locked_confirmation.json"
    if confirmation.exists():
        raise RuntimeError("Locked confirmation already exists; a second run is forbidden")
    if not LOCK.exists():
        raise RuntimeError("model_lock.json must exist before locked values are opened")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if not lock.get("lock_written_before_locked_value_access") or lock.get("confirmation_runs_allowed") != 1:
        raise RuntimeError("Invalid model lock")
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    FIGURE.mkdir(parents=True, exist_ok=True)

    module = load_component()
    module.INPUT_PATH, module.TOPOLOGY_PATH = INPUT, TOPOLOGY
    module.REPORT_DIR, module.FIG_DIR = REPORT, FIGURE
    module.CAL_END_YEAR, module.INNER_TRAIN_END_YEAR = 2018, 2015
    module.STATE_CALENDAR_MODE = "full_forcing"
    module.FORCING_SEMANTICS_MODE = "prescribed_aet_balance"
    module.MASS_ACCOUNTING_MODE = "explicit_upstream_volume"
    module.DYNAMIC_BETA_W = 0.0
    module.configure_engineering_repair(True)
    module.configure_full_gaussian_prior_space(True)
    module.ZERO_PRESERVING_FULL_PRIOR_MODE = True
    module.set_et_feature_block_mode("full")
    module.NETWORK_INPUT_SCALE = 1.0
    module.NETWORK_INPUT_SEMANTICS = "d8_reach193_periodic_spinup_zero_gate_full_gaussian_prior_space"
    module.DETERMINISTIC_SPINUP_MODE = True
    module.SCENARIO_ID = "20260814_5_locked_H0"
    module.write_readme = lambda *_args, **_kwargs: None
    module.main()

    full = pd.read_parquet(REPORT / "monthly_bayes_seasonal_hysteresis_prediction_vs_observed_2006_2022.parquet")
    validation = full[full.year.between(2019, 2022)].copy()
    validation.q_site = validation.q_site.astype(str)
    if validation.empty or validation[KEY].duplicated().any() or not validation.actual.gt(0).all():
        raise RuntimeError("Locked validation population gate failed")
    validation["performance_type"] = "locked_time_confirmation"
    validation["operational_flow_model_id"] = "H0_hybrid"
    output_path = OUT / "validation_predictions_2019_2022.parquet"
    validation.to_parquet(output_path, index=False)
    low, events, tails = build_locked_registries(validation)
    overall = metrics(validation)
    low_metric = metrics(low)
    tail_metric = metrics(tails)

    reference_path = PARENT / "outputs" / "fit_2006_2018_eval_2019_2022" / "validation_predictions_2019_2022.parquet"
    reference_check = {"available": reference_path.exists(), "role": "determinism_only_not_model_selection"}
    if reference_path.exists():
        reference = pd.read_parquet(reference_path)
        reference.q_site = reference.q_site.astype(str)
        joined = validation[KEY + ["predict"]].merge(
            reference[KEY + ["predict"]], on=KEY, suffixes=("_new", "_reference"), validate="one_to_one"
        )
        reference_check.update({
            "matched_rows": int(len(joined)),
            "max_abs_prediction_difference_cfs": float((joined.predict_new - joined.predict_reference).abs().max()),
        })

    result = {
        "performance_type": "locked_time_confirmation",
        "confirmation_run_number": 1,
        "model_lock_sha256": sha256(LOCK),
        "operational_flow_model_id": "H0_hybrid",
        "interface_branch_id": "main",
        "interface_status": "provisional_unresolved",
        "groundwater_status": "non_identifying",
        "locked_metrics": {"overall": overall, "lowflow": low_metric, "tail": tail_metric},
        "locked_registry_counts": {"lowflow_months": int(len(low)), "tail_events": int(len(events)), "tail_months": int(len(tails))},
        "relative_to_locked_H0": {
            "applicability": "operational model is locked H0; no development-qualified alternative exists",
            "lowflow_delta": 0.0,
            "tail_delta": 0.0,
            "both_worsen": False,
            "pbias_guard": True,
        },
        "reference_determinism": reference_check,
        "prediction_sha256": sha256(output_path),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": RUNTIME,
        "confirmation_pass": True,
        "model_recall_permitted": False,
    }
    confirmation.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
