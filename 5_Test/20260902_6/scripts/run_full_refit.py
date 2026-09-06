"""Conditional 2021-2024 H22 full refit and 230-reach production export."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / r"5_Test\20260902_6"
OUT, REPORTS, LOCKS = (RUN / name for name in ("outputs", "reports", "locks"))
STAGE5_LOCK = ROOT / r"5_Test\20260902_5\locks\stage5_lock.json"
STAGE5_DECISION = ROOT / r"5_Test\20260902_5\reports\spatial_decision.json"
TEMPORAL_PAR = ROOT / r"5_Test\20260902_4\outputs\fold_parameters.parquet"
CONTRACT = RUN / "experiment_contract.json"
MANIFEST = ROOT / r"5_Test\20260902_1\program_manifest.json"
for path in [
    ROOT / r"5_Test\20260902_5\scripts", ROOT / r"5_Test\20260902_4\scripts",
    ROOT / r"5_Test\20260824_41\scripts", ROOT / r"5_Test\20260824_44\scripts",
    ROOT / r"5_Test\20260824_19\scripts", ROOT / r"5_Test\20260824_46\scripts",
]:
    sys.path.insert(0, str(path))
from run_nested_worker import fit_fold  # noqa: E402
from spatial_head_model import FoldDesign  # noqa: E402
from stage46_models import Stage46Model  # noqa: E402
import run_stage19 as s19  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def all_reach_frame() -> pd.DataFrame:
    index = pd.MultiIndex.from_product([range(2021, 2025), range(1, 13), range(1, 231)], names=["year", "month", "reach_id"])
    frame = index.to_frame(index=False)
    frame["downstream_fraction_on_reach"] = 1.0
    return frame


def main() -> None:
    for path in (OUT, REPORTS, LOCKS):
        path.mkdir(parents=True, exist_ok=True)
    lock = json.loads(STAGE5_LOCK.read_text(encoding="utf-8"))
    if lock.get("status") != "PASS_H22_SPATIAL_GATES" or not lock.get("spatially_eligible"):
        raise RuntimeError("Stage6 is not authorized: H22 did not pass Stage5")
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(8)
    observations = s19.build_observations()
    train = observations.loc[observations.year.between(2021, 2024)].copy().reset_index(drop=True)
    parent_row = (
        pd.read_parquet(TEMPORAL_PAR)
        .loc[lambda frame: frame.candidate.eq("H22_TRANSFER_HEAD") & frame.fold_id.eq("T3")]
        .iloc[0]
    )
    model = Stage46Model("MINERAL_LIFETIME")
    result = fit_fold(model, train, "FULL_2021_2024", parent_row)
    if float(result["kkt"]) > 1e-5:
        raise RuntimeError(f"Full refit KKT failed: {result['kkt']}")
    physical = torch.tensor(np.asarray(result["physical"], dtype=float))
    gamma = torch.tensor(np.asarray(result["gamma"], dtype=float))
    design = FoldDesign.build("H22_TRANSFER_HEAD", train)
    reach_effect = design.effect_reach(gamma).detach().numpy()
    all_reach = all_reach_frame()
    model._obs_index_cache.clear(); model._q_feature_cache.clear()
    with torch.no_grad():
        raw_log, population_base = model.evaluate(all_reach, physical, 2021, 2024)
    all_reach["raw_process_tn_mg_l"] = np.maximum(np.expm1(raw_log.detach().numpy()), 0.0)
    all_reach["population_transferable_tn_mg_l"] = np.maximum(
        np.expm1(population_base.detach().numpy() + reach_effect[all_reach.reach_id.to_numpy(int) - 1]), 0.0
    )
    all_reach["transferable_offset_log_unit"] = reach_effect[all_reach.reach_id.to_numpy(int) - 1]
    all_reach["model_id"] = "TN_H22_MINERAL_LIFETIME_V1"
    effect_map = dict(zip(result["stations"], map(float, result["effects"])))
    model._obs_index_cache.clear(); model._q_feature_cache.clear()
    with torch.no_grad():
        _, station_base = model.evaluate(train, physical, 2021, 2024)
    station_log = station_base.detach().numpy() + reach_effect[train.reach_id.to_numpy(int) - 1]
    station = train[["station_key", "reach_id", "terminal_tree_id", "year", "month", "tn_mg_l"]].copy()
    station["population_transferable_tn_mg_l"] = np.maximum(np.expm1(station_log), 0.0)
    station["gauged_conditional_tn_mg_l"] = np.maximum(np.expm1(station_log + np.asarray([effect_map[str(key)] for key in train.station_key])), 0.0)
    parameter = pd.DataFrame([{
        "model_id": "TN_H22_MINERAL_LIFETIME_V1", "train_start_year": 2021, "train_end_year": 2024,
        "train_rows": len(train), "train_stations": train.station_key.nunique(),
        "objective": result["objective"], "projected_kkt_max": result["kkt"],
        "process_projected_kkt_max": result["process_projected_kkt"],
        "site_gradient_kkt_max": result["site_gradient_kkt"], "gamma_kkt_max": result["gamma_kkt"],
        "max_abs_transferable_offset": float(np.max(np.abs(reach_effect))),
        "polish_blocks": result["polish_blocks"], "all_starts_json": result["all_starts_json"],
        **dict(zip(model.names(), map(float, result["physical"]))),
    }])
    gamma_frame = pd.DataFrame({"feature": design.fields, "gamma": result["gamma"]})
    station_effect = pd.DataFrame({"station_key": result["stations"], "station_residual_log_unit": result["effects"]})
    offset = pd.DataFrame({"reach_id": range(1, 231), "transferable_offset_log_unit": reach_effect})
    structural = model.structural_diagnostics(physical)
    paths = {
        "predictions_230": OUT / "tn_h22_mineral_lifetime_230_reach_monthly_2021_2024.parquet",
        "station_fit": OUT / "full_development_station_fit.parquet",
        "parameters": OUT / "full_development_parameters.parquet",
        "gamma": OUT / "full_development_gamma.parquet",
        "station_residuals": OUT / "full_development_station_residuals.parquet",
        "reach_offsets": OUT / "full_development_reach_offsets.parquet",
    }
    for name, frame in {
        "predictions_230": all_reach, "station_fit": station, "parameters": parameter,
        "gamma": gamma_frame, "station_residuals": station_effect, "reach_offsets": offset,
    }.items():
        atomic_parquet(frame, paths[name])
    checks = {
        "stage5_authorized": True,
        "kkt_le_1e_5": bool(float(result["kkt"]) <= 1e-5),
        "prediction_rows_230x48": len(all_reach) == 230 * 48,
        "prediction_reaches_230": all_reach.reach_id.nunique() == 230,
        "prediction_period_2021_2024": (int(all_reach.year.min()), int(all_reach.year.max())) == (2021, 2024),
        "predictions_finite_nonnegative": bool(np.isfinite(all_reach.population_transferable_tn_mg_l).all() and (all_reach.population_transferable_tn_mg_l >= 0).all()),
        "offset_within_1_log_unit": bool(np.max(np.abs(reach_effect)) <= 1.0),
        "land_mass_closure_le_1e_10": structural["land_mass_closure_relative"] <= 1e-10,
        "channel_mass_closure_le_1e_10": structural["channel_closure_relative"] <= 1e-10,
    }
    status = "PROMOTED_TN_H22_MINERAL_LIFETIME_V1" if all(checks.values()) else "FAIL_FULL_REFIT_ENGINEERING"
    decision = {
        "stage": "20260902_6", "status": status, "checks": checks,
        "structural_diagnostics": structural, "design_audit": design.audit(),
        "input_hashes": {str(path): sha256(path) for path in (CONTRACT, STAGE5_LOCK, STAGE5_DECISION, TEMPORAL_PAR)},
        "output_hashes": {name: sha256(path) for name, path in paths.items()},
        "method_mainline": "U3_P90_S90",
        "promoted_model": "TN_H22_MINERAL_LIFETIME_V1" if all(checks.values()) else None,
        "previous_production_product": "STAGE32_L0",
    }
    atomic_json(decision, REPORTS / "final_decision.json")
    atomic_json({
        "status": status,
        "model_id": decision["promoted_model"],
        "method_mainline": "U3_P90_S90",
        "process_parent": "MINERAL_LIFETIME",
        "spatial_head": "H22_TRANSFER_HEAD",
        "prediction_product": str(paths["predictions_230"]),
        "decision_sha256": sha256(REPORTS / "final_decision.json"),
        "does_not_modify_historical_locks": True,
    }, LOCKS / "final_program_lock.json")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest["stage_status"]["20260902_5"] = "completed_pass"
    manifest["stage_status"]["20260902_6"] = "completed_promoted" if all(checks.values()) else "completed_engineering_fail"
    manifest["status"] = "complete" if all(checks.values()) else "complete_without_spatial_promotion"
    manifest["final_model"] = decision["promoted_model"]
    atomic_json(manifest, MANIFEST)
    report = [
        "# `20260902_6` final refit and promotion", "", f"Status: `{status}`.", "",
        "The model retains the `MINERAL_LIFETIME` TN process and adds the validated H22 transferable spatial head. The strongly shrunk station residual is retained only for monitored-site conditional predictions; the 230-reach product uses the transferable population layer.", "",
        f"Full-development KKT: `{float(result['kkt']):.3e}`; maximum absolute transferable offset: `{float(np.max(np.abs(reach_effect))):.4f}` log unit.", "",
        f"Production product: `{paths['predictions_230']}`.",
    ]
    (REPORTS / "technical_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    if not all(checks.values()):
        raise RuntimeError(status)


if __name__ == "__main__":
    main()
