from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
TEST = ROOT / "5_Test"
HERE = TEST / "20260824_8"
OUT = HERE / "outputs"
REPORTS = HERE / "reports"
LOCKS = HERE / "locks"
H_PATH = TEST / "20260820_19" / "scripts" / "hierarchical19_shared.py"
F1_PATH = TEST / "20260824_4" / "scripts" / "run_f1_readout.py"
Q72 = TEST / "20260820_2" / "outputs" / "q72_routed_monthly_discharge_2006_2022.parquet"
OBS = TEST / "20260815_1" / "outputs" / "tn_observation_registry_2016_2022.parquet"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def add_full_q_feature(frame: pd.DataFrame) -> pd.DataFrame:
    q = pd.read_parquet(
        Q72,
        filters=[("year", ">=", 2016), ("year", "<=", 2021)],
        columns=["reach_id", "year", "month", "q72_outlet_discharge_m3_s"],
    )
    q["log_q72"] = np.log(q.q72_outlet_discharge_m3_s.clip(lower=1e-12))
    median = q.groupby("reach_id", observed=True).log_q72.median().rename("training_all_month_log_q_median")
    out = frame.merge(q, on=["reach_id", "year", "month"], validate="many_to_one")
    out = out.merge(median, on="reach_id", validate="many_to_one")
    if len(out) != len(frame) or out[["log_q72", "training_all_month_log_q_median"]].isna().any().any():
        raise RuntimeError("STOP_FULL_DEVELOPMENT_Q_FEATURE")
    out["cq_z"] = out.log_q72 - out.training_all_month_log_q_median
    out["cq_low"] = np.minimum(out.cq_z, 0.0)
    out["cq_high"] = np.maximum(out.cq_z, 0.0)
    return out


def main() -> None:
    if Path(sys.prefix).name.lower() != "sparrow":
        raise RuntimeError("sparrow environment required")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        if os.environ.get(name) != "1":
            raise RuntimeError(f"{name}=1 required")
    mechanism_path = LOCKS / "development_mechanism_lock.json"
    mechanism = json.loads(mechanism_path.read_text(encoding="utf-8"))
    if not mechanism["written_before_2022_TN_values_read"] or mechanism["TN_2022_values_read"]:
        raise RuntimeError("STOP_MECHANISM_LOCK")

    h = load_module(H_PATH, "final8_h19")
    f1 = load_module(F1_PATH, "final8_f1")
    shared = h.parent_shared()
    observations = h.development_observations()
    if set(observations.year.unique()) != {2016, 2017, 2018, 2019, 2020, 2021}:
        raise RuntimeError("STOP_DEVELOPMENT_YEAR_BOUNDARY")
    OUT.mkdir(parents=True, exist_ok=True)
    parameter_rows = []
    effect_rows = []
    prediction_rows = []
    h1_rows = []
    for model_id in h.FORMAL_MODELS:
        router = h.build_router(model_id, shared)
        h1 = h.fit_structure(router, observations, "H1_GLOBAL", shared)
        vf = float(np.asarray(h1["parameters"])[0])
        frame = add_full_q_feature(router.frame(observations, np.full(len(router.reach_ids), vf)))
        h1_rows.append({
            "model_id": model_id,
            "v_f_m_per_day": vf,
            "H1_training_objective": float(h1["objective"]),
            "H1_training_data_objective": float(h1["data_objective"]),
            "H1_eta_quick_sequential_fit": float(h1["eta"][0]),
            "H1_eta_gw_sequential_fit": float(h1["eta"][1]),
            "H1_outer_success": bool(h1["outer_success"]),
            "H1_inner_success": bool(h1["diagnostic"]["success"]),
            "H1_vf_boundary": bool(vf >= 0.49),
            "H1_eta_boundary": bool(h1["diagnostic"]["eta_boundary"]),
        })
        for layer in ("P1", "P2"):
            fits = {
                "GAUSSIAN_PROCESS_PARENT": f1.gaussian_fit(frame, layer, False, shared),
                "GAUSSIAN_CQ_HINGE": f1.gaussian_fit(frame, layer, True, shared),
            }
            for arm, fit in fits.items():
                pred = f1.predict(frame, fit, layer)
                pred["model_id"] = model_id
                pred["layer"] = layer
                pred["arm"] = arm
                pred["v_f_m_per_day"] = vf
                prediction_rows.append(pred)
                parameter_rows.append({
                    "model_id": model_id,
                    "layer": layer,
                    "arm": arm,
                    "v_f_m_per_day": vf,
                    "eta_quick": float(fit["eta"][0]),
                    "eta_gw": float(fit["eta"][1]),
                    "beta_low": float(fit["beta"][0]),
                    "beta_high": float(fit["beta"][1]),
                    "success": bool(fit["success"]),
                    "nfev": int(fit["nfev"]),
                    "objective": float(fit["objective"]),
                    "eta_boundary": bool(fit["eta_boundary"]),
                    "beta_boundary": bool(fit["beta_boundary"]),
                    "station_effect_count": len(fit["effects"]),
                })
                for station, value in fit["effects"].items():
                    effect_rows.append({
                        "model_id": model_id,
                        "layer": layer,
                        "arm": arm,
                        "station_key": str(station),
                        "station_effect": float(value),
                    })
        print(json.dumps({"full_development_complete": model_id, "v_f": vf}), flush=True)

    parameters = pd.DataFrame(parameter_rows)
    effects = pd.DataFrame(effect_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    h1_parameters = pd.DataFrame(h1_rows)
    if len(parameters) != 48 or len(h1_parameters) != 12 or not parameters.success.all():
        raise RuntimeError("STOP_FULL_DEVELOPMENT_FIT_COMPLETENESS")
    parameters.to_parquet(OUT / "full_development_parameters.parquet", index=False)
    effects.to_parquet(OUT / "full_development_station_effects.parquet", index=False)
    predictions.to_parquet(OUT / "full_development_training_predictions.parquet", index=False)
    h1_parameters.to_parquet(OUT / "full_development_H1_parameters.parquet", index=False)

    artifacts = [
        OUT / "full_development_parameters.parquet",
        OUT / "full_development_station_effects.parquet",
        OUT / "full_development_training_predictions.parquet",
        OUT / "full_development_H1_parameters.parquet",
    ]
    inputs = [mechanism_path, H_PATH, F1_PATH, Q72, OBS, Path(__file__)]
    file_hashes = {str(path): sha256(path) for path in [*inputs, *artifacts]}
    lock = {
        "lock": "full_development_parameter_lock",
        "written_before_2022_TN_values_read": True,
        "development_TN_years": [2016, 2017, 2018, 2019, 2020, 2021],
        "models": 12,
        "parameter_rows": len(parameters),
        "selected_prediction_arm": "GAUSSIAN_CQ_HINGE",
        "reference_arm": "GAUSSIAN_PROCESS_PARENT",
        "layers": ["P1", "P2"],
        "all_readout_optimizers_successful": bool(parameters.success.all()),
        "H1_outer_success_all": bool(h1_parameters.H1_outer_success.all()),
        "H1_inner_success_all": bool(h1_parameters.H1_inner_success.all()),
        "H1_vf_range_m_per_day": [float(h1_parameters.v_f_m_per_day.min()), float(h1_parameters.v_f_m_per_day.max())],
        "eta_boundary_rows": int(parameters.eta_boundary.sum()),
        "beta_boundary_rows": int(parameters.beta_boundary.sum()),
        "TN_2022_values_read": False,
        "files": file_hashes,
        "aggregate_sha256": hashlib.sha256("\n".join(f"{k}|{v}" for k, v in sorted(file_hashes.items())).encode()).hexdigest(),
    }
    target = LOCKS / "full_development_parameter_lock.json"
    target.write_text(json.dumps(lock, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORTS / "full_development_parameter_lock.json").write_text(json.dumps(lock, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(lock, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
