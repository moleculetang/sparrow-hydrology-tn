from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
TEST = ROOT / "5_Test"
RUN = TEST / "20260823_35"
OUT = RUN / "outputs"
REPORT = RUN / "reports"
DAILY_P_OLD = TEST / "20260807_5" / "inputs" / "derived" / "chm_pre_v2_daily_by_reach_2006_2018.parquet"
DAILY_P_ROOT = ROOT / "0_reach_topology" / "data" / "raw" / "atmosphere" / "precipitation" / "chm_pre_v2" / "daily"
WEIGHTS = TEST / "20260813_30" / "inputs" / "spatial" / "grid_overlap_weights.parquet"
DAILY_Q = TEST / "20260823_28" / "outputs" / "daily_hydrograph_separation.parquet"
BRIDGE = TEST / "20260823_27" / "outputs" / "q72_full_state_tn_bridge_2006_2022.parquet"
SUPPORT = TEST / "20260823_28" / "outputs" / "gauge_support_hydrology.parquet"
PRIOR_DAILY = TEST / "20260808_4" / "validation.json"
PRIOR_README = TEST / "20260808_4" / "README.md"
PROGRAM = TEST / "20260823_27" / "program_manifest.json"
EPS = 1e-12


def decode_daily_dates(handle: h5py.File) -> pd.DatetimeIndex:
    values = np.asarray(handle["time"][:], dtype=int)
    units_raw = handle["time"].attrs["units"]
    units = units_raw.decode() if isinstance(units_raw, bytes) else str(units_raw)
    prefix = "days since "
    if not units.startswith(prefix):
        raise RuntimeError(f"Unsupported CHM time units: {units}")
    origin = pd.Timestamp(units[len(prefix):])
    return pd.DatetimeIndex(origin + pd.to_timedelta(values, unit="D"))


def build_d8_daily_precipitation() -> tuple[pd.DataFrame, dict[str, object]]:
    weights = pd.read_parquet(WEIGHTS)
    weights = weights.loc[weights["product"].eq("CHM")].sort_values(
        ["reach_id", "grid_i", "grid_j"]
    ).reset_index(drop=True)
    if weights.reach_id.nunique() != 230:
        raise RuntimeError("Frozen D8 CHM weights do not cover 230 reaches")
    reach_ids = weights.reach_id.to_numpy(int)
    unique_reaches, starts = np.unique(reach_ids, return_index=True)
    if not np.array_equal(unique_reaches, np.arange(1, 231)):
        raise RuntimeError("Unexpected D8 Reach IDs")
    weight_values = weights.weight.to_numpy(float)
    weight_sums = np.add.reduceat(weight_values, starts)
    if float(np.max(np.abs(weight_sums - 1.0))) > 1e-8:
        raise RuntimeError("D8 polygon weights do not sum to one")
    ilat = weights.grid_i.to_numpy(int)
    ilon = weights.grid_j.to_numpy(int)
    frames = []
    source_rows = []
    for year in range(2006, 2023):
        path = DAILY_P_ROOT / f"CHM_PRE_V2_daily_{year}.nc"
        with h5py.File(path, "r") as handle:
            dates = decode_daily_dates(handle)
            if len(dates) not in {365, 366} or dates[0] != pd.Timestamp(year, 1, 1) or dates[-1] != pd.Timestamp(year, 12, 31):
                raise RuntimeError(f"Incomplete CHM daily calendar: {path}")
            annual = np.empty((len(dates), 230), dtype=np.float64)
            min_coverage = 1.0
            for day_index in range(len(dates)):
                field = np.asarray(handle["prec"][day_index], dtype=float)
                values = field[ilat, ilon]
                valid = np.isfinite(values) & (values < 1e19)
                valid_weights = weight_values * valid
                coverage = np.add.reduceat(valid_weights, starts)
                min_coverage = min(min_coverage, float(coverage.min()))
                if float(coverage.min()) < 0.999:
                    raise RuntimeError(f"Insufficient CHM support in {year}: {coverage.min()}")
                numerator = np.add.reduceat(np.where(valid, values * weight_values, 0.0), starts)
                annual[day_index] = numerator / coverage
        frame = pd.DataFrame({
            "reach_id": np.tile(unique_reaches, len(dates)),
            "date": np.repeat(dates.to_numpy(), len(unique_reaches)),
            "precipitation_daily_mm": annual.reshape(-1),
        })
        frames.append(frame)
        source_rows.append({
            "year": year, "path": str(path), "sha256": sha256(path),
            "days": len(dates), "minimum_valid_weight": min_coverage,
        })
    daily = pd.concat(frames, ignore_index=True).sort_values(["reach_id", "date"]).reset_index(drop=True)
    pd.DataFrame(source_rows).to_parquet(OUT / "daily_precipitation_source_registry.parquet", index=False)
    return daily, {
        "weight_source": str(WEIGHTS),
        "weight_sha256": sha256(WEIGHTS),
        "mapping_method": "20260813_30 frozen D8 full_polygon_overlap",
        "source_years": [2006, 2022],
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extraterrestrial_radiation_weight(latitude_deg: np.ndarray, day_of_year: np.ndarray) -> np.ndarray:
    latitude = np.radians(latitude_deg)
    dr = 1.0 + 0.033 * np.cos(2.0 * np.pi * day_of_year / 365.0)
    declination = 0.409 * np.sin(2.0 * np.pi * day_of_year / 365.0 - 1.39)
    sunset = np.arccos(np.clip(-np.tan(latitude) * np.tan(declination), -1.0, 1.0))
    radiation = dr * (
        sunset * np.sin(latitude) * np.sin(declination)
        + np.cos(latitude) * np.cos(declination) * np.sin(sunset)
    )
    return np.maximum(radiation, EPS)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    contract = json.loads((RUN / "experiment_contract.json").read_text(encoding="utf-8"))
    if contract["status"] != "registered_before_audit":
        raise RuntimeError("Stage35 audit was not registered")
    daily_p, daily_p_lineage = build_d8_daily_precipitation()
    daily_p["date"] = pd.to_datetime(daily_p.date)
    bridge = pd.read_parquet(BRIDGE)
    monthly = bridge[bridge.year.between(2006, 2022)][[
        "reach_id", "year", "month", "precipitation_mm", "prescribed_aet_mm", "catchment_area_km2"
    ]].copy()
    weights = pd.read_parquet(WEIGHTS)
    weights = weights.loc[weights["product"].eq("CHM")].copy()
    reach_lat = weights.groupby("reach_id").apply(
        lambda group: np.average(group.fragment_centroid_lat.to_numpy(float), weights=group.weight.to_numpy(float)),
        include_groups=False,
    ).rename("latitude_deg").reset_index()
    forcing = daily_p.copy()
    forcing["year"] = forcing.date.dt.year
    forcing["month"] = forcing.date.dt.month
    forcing["days_in_month"] = forcing.date.dt.days_in_month
    forcing["day_of_year"] = forcing.date.dt.dayofyear
    forcing = forcing.merge(monthly, on=["reach_id", "year", "month"], validate="many_to_one")
    forcing = forcing.merge(reach_lat, on="reach_id", validate="many_to_one")
    forcing["aet_uniform_daily_mm"] = forcing.prescribed_aet_mm / forcing.days_in_month
    forcing["solar_raw_weight"] = extraterrestrial_radiation_weight(forcing.latitude_deg.to_numpy(float), forcing.day_of_year.to_numpy(float))
    forcing["solar_month_weight_sum"] = forcing.groupby(["reach_id", "year", "month"]).solar_raw_weight.transform("sum")
    forcing["aet_solar_weighted_daily_mm"] = forcing.prescribed_aet_mm * forcing.solar_raw_weight / forcing.solar_month_weight_sum
    daily_sum = forcing.groupby(["reach_id", "year", "month"], as_index=False).agg(
        precipitation_daily_sum_mm=("precipitation_daily_mm", "sum"),
        aet_uniform_sum_mm=("aet_uniform_daily_mm", "sum"),
        aet_solar_sum_mm=("aet_solar_weighted_daily_mm", "sum"),
        daily_rows=("date", "size"),
    ).merge(monthly, on=["reach_id", "year", "month"], validate="one_to_one")
    daily_sum["precipitation_closure_error_mm"] = daily_sum.precipitation_daily_sum_mm - daily_sum.precipitation_mm
    daily_sum["aet_uniform_closure_error_mm"] = daily_sum.aet_uniform_sum_mm - daily_sum.prescribed_aet_mm
    daily_sum["aet_solar_closure_error_mm"] = daily_sum.aet_solar_sum_mm - daily_sum.prescribed_aet_mm
    daily_q = pd.read_parquet(DAILY_Q)
    daily_q["date"] = pd.to_datetime(daily_q.date)
    station_meta = pd.read_parquet(SUPPORT)[["station_norm", "reach_id", "downstream_fraction_on_reach"]].drop_duplicates("station_norm")
    station_meta["station_norm"] = station_meta.station_norm.astype(str)
    daily_q["station_norm"] = daily_q.station_norm.astype(str)
    daily_q = daily_q.merge(station_meta, on="station_norm", validate="many_to_one")
    daily_q["forcing_available"] = daily_q.date.dt.year.between(2010, 2022)
    daily_q["period"] = np.select(
        [daily_q.date.dt.year.between(2010, 2018), daily_q.date.dt.year.between(2019, 2022)],
        ["development_2010_2018", "frozen_check_2019_2022"],
        default="outside_daily_forcing_experiment",
    )
    coverage = daily_q[daily_q.forcing_available].groupby(["station_norm", "reach_id", "period"], as_index=False).agg(
        first_date=("date", "min"), last_date=("date", "max"), rows=("date", "size"),
        valid_total_flow=("q_m3_s", lambda values: int(np.isfinite(values).sum())),
        valid_proxy=("delayed_proxy_median_m3_s", lambda values: int(np.isfinite(values).sum())),
    )
    forcing_keep = [
        "reach_id", "date", "year", "month", "precipitation_daily_mm", "aet_uniform_daily_mm",
        "aet_solar_weighted_daily_mm", "prescribed_aet_mm", "precipitation_mm", "catchment_area_km2",
        "latitude_deg",
    ]
    forcing[forcing_keep].sort_values(["reach_id", "date"]).to_parquet(OUT / "daily_reach_forcing_2006_2022.parquet", index=False)
    daily_q.sort_values(["station_norm", "date"]).to_parquet(OUT / "daily_station_flow_proxy_registry.parquet", index=False)
    daily_sum.to_parquet(OUT / "daily_monthly_forcing_closure.parquet", index=False)
    coverage.to_parquet(OUT / "daily_station_coverage.parquet", index=False)
    prior = json.loads(PRIOR_DAILY.read_text(encoding="utf-8"))
    audit = {
        "stage": "20260823_35",
        "daily_precipitation": {
            "source": str(DAILY_P_ROOT), "reach_count": int(forcing.reach_id.nunique()),
            "first_date": str(forcing.date.min().date()), "last_date": str(forcing.date.max().date()),
            "rows": int(len(forcing)), "missing_values": int(forcing.precipitation_daily_mm.isna().sum()),
            **daily_p_lineage,
        },
        "daily_discharge": {
            "source": str(DAILY_Q), "station_count_all_2010_2022": int(pd.read_parquet(DAILY_Q, columns=["station_norm"]).station_norm.nunique()),
            "station_count_overlap": int(daily_q[daily_q.forcing_available].station_norm.nunique()),
            "overlap_rows": int(daily_q.forcing_available.sum()),
        },
        "daily_ET": {
            "authoritative_product_available": False,
            "primary_schedule": "uniform monthly-prescribed AET conservation",
            "sensitivity_schedule": "extraterrestrial-radiation weighted monthly-prescribed AET conservation",
            "claim": "ARTIFICIAL_DAILY_ET_SENSITIVITY_ONLY",
        },
        "forcing_closure": {
            "precipitation_max_abs_mm": float(daily_sum.precipitation_closure_error_mm.abs().max()),
            "uniform_AET_max_abs_mm": float(daily_sum.aet_uniform_closure_error_mm.abs().max()),
            "solar_AET_max_abs_mm": float(daily_sum.aet_solar_closure_error_mm.abs().max()),
        },
        "prior_daily_state_experiment": {
            "stage": "20260808_4", "decision": prior["decision"],
            "distinction": "prior experiment froze physical parameters and predicted monthly flow; the authorized experiment jointly fits daily physical fast/delayed states to daily total flow and a soft delayed-flow proxy",
        },
        "frozen_check_contract": "Raw CHM_PRE daily forcing is available through 2022. Daily discharge from 2019-2022 is withheld from fitting and read only after the development parameter lock.",
    }
    gates = contract["hard_gates"]
    gate_results = {
        "precipitation_closure": audit["forcing_closure"]["precipitation_max_abs_mm"] <= float(gates["precipitation_monthly_closure_max_abs_mm"]),
        "uniform_AET_closure": audit["forcing_closure"]["uniform_AET_max_abs_mm"] <= float(gates["AET_monthly_closure_max_abs_mm"]),
        "solar_AET_closure": audit["forcing_closure"]["solar_AET_max_abs_mm"] <= float(gates["AET_monthly_closure_max_abs_mm"]),
        "reach_count": audit["daily_precipitation"]["reach_count"] == int(gates["reach_count"]),
        "station_count": audit["daily_discharge"]["station_count_overlap"] >= int(gates["daily_flow_station_min"]),
        "development_years": 9 >= int(gates["development_years_min"]),
        "frozen_check_years": 4 >= int(gates["frozen_check_years_min"]),
    }
    audit["gates"] = gate_results
    audit["status"] = "DAILY_TIMING_STATE_SENSITIVITY_EXPERIMENT_FEASIBLE" if all(gate_results.values()) else "DAILY_EXPERIMENT_BLOCKED"
    (REPORT / "stage35_daily_data_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    report = (
        "# 20260823_35 日尺度数据审计\n\n"
        f"状态：`{audit['status']}`。可运行的是2010–2022 daily-P fast/delayed-state experiment，不是完整日能量—水量平衡。\n\n"
        "- 日降水：230 Reach，2006–2022；使用当前Q72相同的D8 full-polygon-overlap空间权重；\n"
        f"- 重叠日流量：{audit['daily_discharge']['station_count_overlap']}站、{audit['daily_discharge']['overlap_rows']}行；\n"
        "- 日ET：无权威产品，仅允许两种月总量严格闭合的人工分配；\n"
        "- 2019–2022：日降水完整，只在development参数锁后读取同期日流量作时间外检验。\n\n"
        "历史上的 `20260808_4` 已否定“冻结参数下日降雨顺序显著改善月Q72”；本轮只允许检验更强但不同的日流量+日分割联合物理训练。\n"
    )
    (REPORT / "technical_report.md").write_text(report, encoding="utf-8")
    (REPORT / "integrity.json").write_text(json.dumps({
        "contract_sha256": sha256(RUN / "experiment_contract.json"),
        "daily_precip_weight_sha256": sha256(WEIGHTS), "legacy_daily_precip_sha256": sha256(DAILY_P_OLD),
        "daily_discharge_sha256": sha256(DAILY_Q),
        "bridge_sha256": sha256(BRIDGE), "forcing_output_sha256": sha256(OUT / "daily_reach_forcing_2006_2022.parquet"),
        "program_manifest_sha256": sha256(PROGRAM),
    }, indent=2), encoding="utf-8")
    if not all(gate_results.values()):
        raise RuntimeError(audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
