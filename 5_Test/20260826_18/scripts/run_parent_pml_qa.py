"""Build frozen-parent state summaries and audit PML AET as soft validation.

Run with the dedicated hydro_dpl_20260826 interpreter.  This script reads no
discharge and no TN.  It does not fit or select any parameter.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260826_18"
OUT = RUN / "outputs"
REPORTS = RUN / "reports"
FORCING = ROOT / "5_Test" / "20260825_2" / "outputs" / "daily_hbv_forcing_2006_2022.parquet"
PARAMETER_LOCK = ROOT / "5_Test" / "20260825_7" / "reports" / "full_development_parameter_lock.json"
PML = ROOT / "5_Test" / "20260825_2" / "outputs" / "pml_v2_2a_monthly_aet_by_reach_2006_2022.parquet"
PML_REGISTRY = ROOT / "5_Test" / "20260825_2" / "outputs" / "pml_v2_2a_source_registry.parquet"
Q72_BRIDGE = ROOT / "5_Test" / "20260823_27" / "outputs" / "q72_full_state_tn_bridge_2006_2022.parquet"

sys.path.insert(0, str(ROOT / "5_Test" / "20260826_14" / "scripts"))
from torch_hbv import periodic_spinup, simulate_ordered_hbv  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3 or np.std(x[valid]) <= 0 or np.std(y[valid]) <= 0:
        return float("nan")
    return float(np.corrcoef(x[valid], y[valid])[0, 1])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    reach_ids = list(range(1, 231))
    forcing = pd.read_parquet(
        FORCING,
        columns=["reach_id", "date", "precipitation_daily_mm", "pet_fao56_mm_day"],
    )
    forcing["date"] = pd.to_datetime(forcing.date)
    forcing = forcing.loc[forcing.date.dt.year.between(2006, 2018)].copy()
    dates = pd.date_range("2006-01-01", "2018-12-31", freq="D")
    precipitation = forcing.pivot(index="date", columns="reach_id", values="precipitation_daily_mm").reindex(index=dates, columns=reach_ids)
    pet = forcing.pivot(index="date", columns="reach_id", values="pet_fao56_mm_day").reindex(index=dates, columns=reach_ids)
    if precipitation.isna().any().any() or pet.isna().any().any():
        raise RuntimeError("Frozen daily forcing is incomplete for 230 Reaches in 2006-2018")

    parameter_lock = json.loads(PARAMETER_LOCK.read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    physical = torch.tensor(
        [parameter_lock["physical_parameters"][name] for name in (
            "fc_mm", "beta", "lp", "perc_mm_day", "uzl_mm", "tau0_day", "delta_tau10_day", "delta_tau21_day"
        )],
        dtype=torch.float64,
        device=device,
    )
    p = torch.tensor(precipitation.to_numpy(float), dtype=torch.float64, device=device)
    e = torch.tensor(pet.to_numpy(float), dtype=torch.float64, device=device)
    spin_mask = dates.year <= 2009
    initial, spinup = periodic_spinup(p[spin_mask], e[spin_mask], physical, 1.0e-8, 500)
    with torch.no_grad():
        result = simulate_ordered_hbv(
            p,
            e,
            physical,
            initial,
            collect_components=False,
            collect_storage=True,
            collect_diagnostic_fluxes=True,
        )
    storage = result.storage_mm.cpu().numpy()
    aet = result.diagnostic_fluxes_mm_day["aet"].cpu().numpy()
    daily = pd.DataFrame(
        {
            "date": np.repeat(dates.to_numpy(), len(reach_ids)),
            "reach_id": np.tile(np.asarray(reach_ids), len(dates)),
            "model_aet_mm_day": aet.reshape(-1),
            "model_soil_storage_mm": storage[:, :, 0].reshape(-1),
            "model_fast_response_storage_mm": storage[:, :, 1].reshape(-1),
            "model_slow_response_storage_mm": storage[:, :, 2].reshape(-1),
        }
    )
    daily["model_total_storage_mm"] = daily[
        ["model_soil_storage_mm", "model_fast_response_storage_mm", "model_slow_response_storage_mm"]
    ].sum(axis=1)
    daily["year"] = daily.date.dt.year
    daily["month"] = daily.date.dt.month
    monthly = daily.groupby(["reach_id", "year", "month"], as_index=False).agg(
        model_aet_mm_month=("model_aet_mm_day", "sum"),
        model_soil_storage_mm=("model_soil_storage_mm", "mean"),
        model_fast_response_storage_mm=("model_fast_response_storage_mm", "mean"),
        model_slow_response_storage_mm=("model_slow_response_storage_mm", "mean"),
        model_total_storage_mm=("model_total_storage_mm", "mean"),
    )
    monthly.to_parquet(OUT / "parent_state_monthly_2006_2018.parquet", index=False)

    pml = pd.read_parquet(PML).loc[lambda frame: frame.year.between(2006, 2018)].copy()
    merged = monthly.merge(pml.drop(columns=["role"]), on=["reach_id", "year", "month"], validate="one_to_one")
    development = merged.loc[merged.year.between(2010, 2018)].copy()
    metric_rows = []
    for reach, frame in development.groupby("reach_id"):
        observed = frame.pml_aet_mm_month.to_numpy(float)
        modeled = frame.model_aet_mm_month.to_numpy(float)
        metric_rows.append(
            {
                "reach_id": int(reach),
                "n_month": int(len(frame)),
                "pearson_r": safe_corr(observed, modeled),
                "bias_mm_month": float(np.mean(modeled - observed)),
                "mae_mm_month": float(np.mean(np.abs(modeled - observed))),
                "rmse_mm_month": float(np.sqrt(np.mean((modeled - observed) ** 2))),
            }
        )
    metrics = pd.DataFrame(metric_rows)
    metrics.to_parquet(OUT / "pml_parent_reach_metrics.parquet", index=False)

    area = pd.read_parquet(Q72_BRIDGE, columns=["reach_id", "catchment_area_km2"]).drop_duplicates("reach_id")
    if area.reach_id.nunique() != 230 or area.catchment_area_km2.le(0).any():
        raise RuntimeError("Local catchment-area weights are invalid")
    basin = development.merge(area, on="reach_id", validate="many_to_one")
    basin["weighted_model"] = basin.model_aet_mm_month * basin.catchment_area_km2
    basin["weighted_pml"] = basin.pml_aet_mm_month * basin.catchment_area_km2
    basin_monthly = basin.groupby(["year", "month"], as_index=False).agg(
        area_km2=("catchment_area_km2", "sum"),
        weighted_model=("weighted_model", "sum"),
        weighted_pml=("weighted_pml", "sum"),
    )
    basin_monthly["model_aet_mm_month"] = basin_monthly.weighted_model / basin_monthly.area_km2
    basin_monthly["pml_aet_mm_month"] = basin_monthly.weighted_pml / basin_monthly.area_km2
    basin_monthly = basin_monthly.drop(columns=["weighted_model", "weighted_pml"])
    basin_monthly.to_parquet(OUT / "pml_parent_basin_monthly_2010_2018.parquet", index=False)

    pml_registry = pd.read_parquet(PML_REGISTRY).loc[lambda frame: frame.year.between(2006, 2018)]
    report = {
        "product": "PML V2.2a",
        "role": "MONTHLY_AET_SOFT_VALIDATION_ONLY_NOT_FORCING_OR_SELECTION",
        "source_years": [int(pml_registry.year.min()), int(pml_registry.year.max())],
        "source_file_count": int(len(pml_registry)),
        "source_hash_registry_sha256": sha256(PML_REGISTRY),
        "model_state_period": [2006, 2018],
        "comparison_period": [2010, 2018],
        "reach_count": int(development.reach_id.nunique()),
        "comparison_rows": int(len(development)),
        "missing_values": int(development[["model_aet_mm_month", "pml_aet_mm_month"]].isna().sum().sum()),
        "negative_values": int((development[["model_aet_mm_month", "pml_aet_mm_month"]] < 0).sum().sum()),
        "reach_pearson_r_median": float(metrics.pearson_r.median()),
        "reach_pearson_r_mean": float(metrics.pearson_r.mean()),
        "reach_bias_mm_month_median": float(metrics.bias_mm_month.median()),
        "basin_pearson_r": safe_corr(
            basin_monthly.pml_aet_mm_month.to_numpy(float), basin_monthly.model_aet_mm_month.to_numpy(float)
        ),
        "basin_bias_mm_month": float((basin_monthly.model_aet_mm_month - basin_monthly.pml_aet_mm_month).mean()),
        "parent_spinup": spinup,
        "simulation_device": str(device),
        "parent_max_abs_mass_error_mm": float(result.max_abs_mass_error_mm),
        "qa_pass": bool(
            development.reach_id.nunique() == 230
            and len(development) == 230 * 9 * 12
            and development[["model_aet_mm_month", "pml_aet_mm_month"]].notna().all().all()
            and (development[["model_aet_mm_month", "pml_aet_mm_month"]] >= 0).all().all()
            and spinup["converged"]
            and float(result.max_abs_mass_error_mm) <= 1.0e-9
        ),
        "interpretation_limit": "PML is an independent model product, not direct AET observation; correlations are descriptive and have no post-hoc promotion threshold.",
    }
    (REPORTS / "pml_state_product_qa.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if not report["qa_pass"]:
        raise RuntimeError(report)


if __name__ == "__main__":
    main()
