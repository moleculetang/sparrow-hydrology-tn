from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow
import scipy
import shapely
import xarray as xr
from shapely.geometry import box


ROOT = Path(r"E:\SPARROW\5_Test\20260815_1")
OUT = ROOT / "outputs"
REPORTS = ROOT / "reports"
S6 = Path(r"E:\SPARROW\5_Test\20260814_6")
S1 = Path(r"E:\SPARROW\5_Test\20260814_1")
CANON = S6 / "outputs" / "structural_canonical_main_interface_2006_2022.parquet"
TN = Path(r"E:\SPARROW\1_Inputs\WaterQualityData\model_ready\tn_station_month_strict_baseline_2006_2022.parquet")
GROUNDWATER = S1 / "inputs" / "spatial" / "groundwater_catchment_monthly_2005_2018.parquet"
CATCHMENTS = S1 / "inputs" / "spatial" / "reach_catchments.shp"
TOPOLOGY = S1 / "inputs" / "topology" / "topology_edges.csv"
WATERGAP = Path(
    r"E:\SPARROW\0_reach_topology\data\raw\hydrology\groundwater_recharge"
    r"\watergap_v2_2e_naturalized_diffuse_recharge_1901_2022\data"
    r"\watergap22e_gswp3-era5_qrdif_nosoc_monthly_1901_2022.nc"
)
EXPECTED_CANON_SHA = "a0b47562bae42129a6b55571e28cc5fdfc1001ee740f9c9287e6f2f70db065c2"
SEED = 20260815
N_BOOT = 10_000


def require_runtime() -> None:
    expected = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    wrong = {k: os.environ.get(k) for k, v in expected.items() if os.environ.get(k) != v}
    if wrong:
        raise RuntimeError(f"Thread limits are not frozen: {wrong}")
    if Path(sys.prefix).name.lower() != "sparrow":
        raise RuntimeError(f"Expected conda environment sparrow, found {sys.prefix}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dump_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parent_files() -> dict[str, Path]:
    files = {
        "canonical_interface": CANON,
        "s6_interface_lock": S6 / "structural_interface_lock.json",
        "s6_dual_model_manifest": S6 / "dual_model_manifest.json",
        "tn_strict_baseline": TN,
        "groundwater_prepared": GROUNDWATER,
        "topology": TOPOLOGY,
        "watergap_raw": WATERGAP,
    }
    for suffix in [".shp", ".shx", ".dbf", ".prj", ".cpg"]:
        files[f"catchments_{suffix[1:]}"] = CATCHMENTS.with_suffix(suffix)
    return files


def hash_manifest() -> dict[str, object]:
    result: dict[str, object] = {}
    for key, path in parent_files().items():
        if not path.exists():
            raise FileNotFoundError(path)
        result[key] = {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
    return result


def terminal_map() -> dict[int, int]:
    topology = pd.read_csv(TOPOLOGY)
    downstream: dict[int, int | None] = {}
    for row in topology.itertuples(index=False):
        value = row.downstream_reach
        downstream[int(row.reach_id)] = None if pd.isna(value) or str(value).strip() == "" else int(float(str(value).split(",")[0]))
    result: dict[int, int] = {}
    for reach in downstream:
        current = reach
        seen: set[int] = set()
        while downstream.get(current) is not None:
            if current in seen:
                raise RuntimeError(f"Topology cycle at {current}")
            seen.add(current)
            current = int(downstream[current])
        result[reach] = current
    return result


def build_climatology(hydro: pd.DataFrame) -> pd.DataFrame:
    fields = [
        "positive_input_mm",
        "quick_generated_mm",
        "soil_overflow_to_quick_mm",
        "gw_recharge_mm",
        "gw_discharge_mm",
        "gw_response_state_end_mm",
        "source_water_capacity_mm",
        "catchment_area_km2",
    ]
    source = hydro.loc[hydro.year.between(2006, 2015), ["comid", "year", "month", *fields]].copy()
    if sorted(source.year.unique().tolist()) != list(range(2006, 2016)):
        raise RuntimeError("Q72 climatology is not exactly 2006-2015")
    if len(source) != 230 * 120:
        raise RuntimeError(f"Expected 27,600 Q72 climatology source rows, found {len(source)}")
    climate = source.groupby(["comid", "month"], as_index=False)[fields].mean()
    climate = climate.rename(columns={"comid": "reach_id"})
    climate["climatology_start_year"] = 2006
    climate["climatology_end_year"] = 2015
    climate["soil_contact_water_mm"] = climate.soil_overflow_to_quick_mm + climate.gw_recharge_mm
    climate["quick_bypass_fraction"] = np.divide(
        climate.quick_generated_mm,
        climate.positive_input_mm,
        out=np.zeros(len(climate), dtype=float),
        where=climate.positive_input_mm.to_numpy() > 1e-12,
    )
    if len(climate) != 230 * 12 or not climate.quick_bypass_fraction.between(-1e-12, 1 + 1e-12).all():
        raise RuntimeError("Invalid Q72 monthly climatology")
    climate.to_parquet(OUT / "q72_monthly_climatology_2006_2015.parquet", index=False)
    return climate


def build_tn_registry() -> tuple[pd.DataFrame, pd.DataFrame]:
    tn = pd.read_parquet(TN).copy()
    required = ["station", "station_key", "year", "month", "tn_mg_l", "reach_id", "strict"]
    missing = [c for c in required if c not in tn]
    if missing:
        raise RuntimeError(f"TN registry missing {missing}")
    tn = tn.loc[tn.strict.astype(bool) & tn.year.between(2016, 2022)].copy()
    tn["reach_id"] = tn.reach_id.astype(int)
    key = ["station_key", "year", "month"]
    if tn.duplicated(key).any() or (~np.isfinite(tn.tn_mg_l)).any() or (tn.tn_mg_l <= 0).any():
        raise RuntimeError("TN observations fail uniqueness/positivity checks")
    tn["observation_role"] = np.where(tn.year.eq(2022), "locked_final", "development_available")
    tn["target_semantics"] = "TN_concentration_mg_L"
    tn = tn.sort_values(key).reset_index(drop=True)
    tn.to_parquet(OUT / "tn_observation_registry_2016_2022.parquet", index=False)

    fold_defs = [("F1", 2017, 2018), ("F2", 2018, 2019), ("F3", 2019, 2020), ("F4", 2020, 2021)]
    pieces = []
    for fold_id, train_end, eval_year in fold_defs:
        x = tn.loc[tn.year.between(2016, eval_year), key + ["station", "reach_id", "tn_mg_l"]].copy()
        x["fold_id"] = fold_id
        x["fold_role"] = np.where(x.year <= train_end, "train", "evaluate")
        x["train_start_year"] = 2016
        x["train_end_year"] = train_end
        x["evaluation_year"] = eval_year
        pieces.append(x)
    folds = pd.concat(pieces, ignore_index=True)
    if folds.year.eq(2022).any():
        raise RuntimeError("Locked 2022 observations leaked into fold registry")
    folds.to_parquet(OUT / "tn_fold_registry.parquet", index=False)
    return tn, folds


def build_watergap_weights(ds: xr.Dataset, catchments: gpd.GeoDataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    bounds = catchments.to_crs(4326).total_bounds
    lat = ds.lat.values.astype(float)
    lon = ds.lon.values.astype(float)
    lat_idx = np.flatnonzero((lat >= bounds[1] - 0.75) & (lat <= bounds[3] + 0.75))
    lon_idx = np.flatnonzero((lon >= bounds[0] - 0.75) & (lon <= bounds[2] + 0.75))
    records = []
    nlon = len(lon_idx)
    for ilocal, iglobal in enumerate(lat_idx):
        for jlocal, jglobal in enumerate(lon_idx):
            records.append(
                {
                    "cell_pos": ilocal * nlon + jlocal,
                    "lat_index": int(iglobal),
                    "lon_index": int(jglobal),
                    "geometry": box(lon[jglobal] - 0.25, lat[iglobal] - 0.25, lon[jglobal] + 0.25, lat[iglobal] + 0.25),
                }
            )
    cells = gpd.GeoDataFrame(records, crs=4326).to_crs(catchments.crs)
    left = catchments[["reach_id", "geometry"]].copy()
    pairs = gpd.sjoin(left, cells, how="inner", predicate="intersects").reset_index(drop=True)
    cell_geom = cells.geometry
    areas = []
    for row in pairs.itertuples(index=False):
        areas.append(row.geometry.intersection(cell_geom.iloc[int(row.index_right)]).area)
    pairs["intersection_area_m2"] = np.asarray(areas, dtype=float)
    pairs = pairs.loc[pairs.intersection_area_m2 > 0].copy()
    pairs["cell_pos"] = pairs.index_right.map(cells.cell_pos).astype(int)
    pairs["weight"] = pairs.intersection_area_m2 / pairs.groupby("reach_id").intersection_area_m2.transform("sum")
    closure = pairs.groupby("reach_id").weight.sum()
    if len(closure) != 230 or float((closure - 1).abs().max()) > 1e-12:
        raise RuntimeError("WaterGAP catchment weight closure failed")
    weights = pairs[["reach_id", "cell_pos", "lat_index", "lon_index", "intersection_area_m2", "weight"]].sort_values(["reach_id", "cell_pos"])
    weights.to_parquet(OUT / "watergap_catchment_overlap_weights.parquet", index=False)
    return weights, lat_idx, lon_idx


def aggregate_watergap(catchments: gpd.GeoDataFrame) -> pd.DataFrame:
    ds = xr.open_dataset(WATERGAP, decode_times=False)
    try:
        weights, lat_idx, lon_idx = build_watergap_weights(ds, catchments)
        dates = pd.date_range("1901-01-01", periods=ds.sizes["time"], freq="MS")
        mask = (dates.year >= 2006) & (dates.year <= 2022)
        time_idx = np.flatnonzero(mask)
        arr = ds.qrdif.isel(time=time_idx, lat=lat_idx, lon=lon_idx).values.astype(float).reshape(len(time_idx), -1)
    finally:
        ds.close()
    seconds = (dates[mask].days_in_month.to_numpy(dtype=float) * 86400.0)[:, None]
    arr_mm = arr * seconds
    rows = []
    selected_dates = dates[mask]
    for reach_id, group in weights.groupby("reach_id", sort=True):
        pos = group.cell_pos.to_numpy(dtype=int)
        w = group.weight.to_numpy(dtype=float)
        values = arr_mm[:, pos]
        valid = np.isfinite(values)
        denom = valid @ w
        numer = np.nansum(values * w[None, :], axis=1)
        reach_value = np.divide(numer, denom, out=np.full(len(numer), np.nan), where=denom > 0)
        rows.append(
            pd.DataFrame(
                {
                    "reach_id": int(reach_id),
                    "year": selected_dates.year,
                    "month": selected_dates.month,
                    "watergap_recharge_mm": reach_value,
                    "valid_weight_sum": denom,
                }
            )
        )
    result = pd.concat(rows, ignore_index=True).sort_values(["reach_id", "year", "month"])
    if len(result) != 230 * 204 or result.watergap_recharge_mm.isna().any():
        raise RuntimeError("WaterGAP aggregation is incomplete")
    result.to_parquet(OUT / "watergap_recharge_by_reach_2006_2022.parquet", index=False)
    return result


def reach_correlations(frame: pd.DataFrame, x: str, y: str, minimum: int = 12) -> pd.DataFrame:
    rows = []
    for reach_id, group in frame.groupby("reach_id", sort=True):
        z = group[[x, y]].dropna()
        corr = float(z[x].corr(z[y])) if len(z) >= minimum and z[x].std() > 0 and z[y].std() > 0 else np.nan
        rows.append({"reach_id": int(reach_id), "correlation": corr, "n": int(len(z))})
    return pd.DataFrame(rows)


def tree_bootstrap(correlations: pd.DataFrame, reach_terminal: dict[int, int], seed_offset: int) -> dict[str, object]:
    x = correlations.dropna(subset=["correlation"]).copy()
    x["terminal_tree_id"] = x.reach_id.map(reach_terminal)
    if x.terminal_tree_id.isna().any() or x.empty:
        raise RuntimeError("Cannot map diagnostic correlations to terminal trees")
    blocks = {int(k): g.correlation.to_numpy(dtype=float) for k, g in x.groupby("terminal_tree_id")}
    ids = np.array(sorted(blocks), dtype=int)
    rng = np.random.default_rng(SEED + seed_offset)
    dist = np.empty(N_BOOT, dtype=float)
    for i in range(N_BOOT):
        selected = rng.choice(ids, size=len(ids), replace=True)
        dist[i] = np.median(np.concatenate([blocks[int(k)] for k in selected]))
    low, high = np.percentile(dist, [2.5, 97.5])
    status = "consistent" if low > 0 else "contradictory" if high < 0 else "non_identifying"
    return {
        "reaches": int(len(x)),
        "terminal_tree_count": int(len(ids)),
        "terminal_tree_ids": ids.tolist(),
        "median_reach_correlation": float(np.median(x.correlation)),
        "fraction_reaches_positive": float((x.correlation > 0).mean()),
        "bootstrap_replicates": N_BOOT,
        "ci95_low": float(low),
        "ci95_high": float(high),
        "status": status,
    }


def monthly_anomaly(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    x = frame.copy()
    for column in columns:
        climatology = x.groupby(["reach_id", "month"])[column].transform("mean")
        x[f"{column}_anomaly"] = x[column] - climatology
    return x


def external_diagnostics(hydro: pd.DataFrame, wg: pd.DataFrame, reach_terminal: dict[int, int]) -> dict[str, object]:
    q72 = hydro.rename(columns={"comid": "reach_id"})
    q72["reach_id"] = q72.reach_id.astype(int)
    water = q72[["reach_id", "year", "month", "gw_recharge_mm"]].merge(
        wg, on=["reach_id", "year", "month"], validate="one_to_one"
    )
    water = monthly_anomaly(water, ["gw_recharge_mm", "watergap_recharge_mm"])
    water_corr = reach_correlations(water, "gw_recharge_mm_anomaly", "watergap_recharge_mm_anomaly", 120)
    water_corr.to_csv(REPORTS / "watergap_q72_reach_correlations.csv", index=False, encoding="utf-8-sig")
    water_report = tree_bootstrap(water_corr, reach_terminal, 1)
    q_clim = water.groupby("month").gw_recharge_mm.mean()
    w_clim = water.groupby("month").watergap_recharge_mm.mean()
    water_report["pooled_seasonal_correlation"] = float(q_clim.corr(w_clim))
    water_report["period"] = "2006-2022"
    water_report["role"] = "external_recharge_benchmark_only"

    gw = pd.read_parquet(GROUNDWATER).rename(columns={"reach_id": "reach_id"})
    gw = gw.loc[gw.year.between(2006, 2017), ["reach_id", "year", "month", "groundwater_area_weighted_median_m"]].copy()
    state = q72.loc[q72.year.between(2006, 2017), ["reach_id", "year", "month", "gw_response_state_end_mm"]]
    head = state.merge(gw, on=["reach_id", "year", "month"], validate="one_to_one").sort_values(["reach_id", "year", "month"])
    for column in ["gw_response_state_end_mm", "groundwater_area_weighted_median_m"]:
        head[f"delta_{column}"] = head.groupby("reach_id")[column].diff()
    boundary = ((head.year == 2009) & (head.month == 1)) | ((head.year == 2010) & (head.month == 1))
    dynamic = head.loc[~boundary].copy()
    dynamic_corr = reach_correlations(
        dynamic,
        "delta_gw_response_state_end_mm",
        "delta_groundwater_area_weighted_median_m",
        60,
    )
    dynamic_corr.to_csv(REPORTS / "groundwater_dynamic_reach_correlations.csv", index=False, encoding="utf-8-sig")
    dynamic_report = tree_bootstrap(dynamic_corr, reach_terminal, 2)
    dynamic_report["period"] = "2006-2017_excluding_2008Dec_to_2009Jan_and_2009Dec_to_2010Jan"

    stable = head.loc[head.year.between(2010, 2017)].copy()
    stable = monthly_anomaly(stable, ["gw_response_state_end_mm", "groundwater_area_weighted_median_m"])
    stable_corr = reach_correlations(
        stable,
        "gw_response_state_end_mm_anomaly",
        "groundwater_area_weighted_median_m_anomaly",
        72,
    )
    stable_corr.to_csv(REPORTS / "groundwater_stable_anomaly_reach_correlations.csv", index=False, encoding="utf-8-sig")
    stable_report = tree_bootstrap(stable_corr, reach_terminal, 3)
    stable_report["period"] = "2010-2017"

    if dynamic_report["status"] == stable_report["status"] and dynamic_report["status"] in {"consistent", "contradictory"}:
        groundwater_status = dynamic_report["status"]
    else:
        groundwater_status = "non_identifying"
    groundwater_report = {
        "dynamic_change": dynamic_report,
        "stable_anomaly": stable_report,
        "status": groundwater_status,
        "role": "unresolved_auxiliary_validation",
        "does_not_replace_s6_groundwater_status": True,
    }
    if water_report["status"] == groundwater_status and groundwater_status in {"consistent", "contradictory"}:
        combined = groundwater_status
    else:
        combined = "non_identifying"
    result = {
        "watergap_recharge_vs_q72_recharge": water_report,
        "groundwater_level_vs_q72_response_state": groundwater_report,
        "hydrology_external_consistency": combined,
        "decision_role": "diagnostic_only_not_in_N_loss_Q72_calibration_or_mu_selection",
        "s6_groundwater_status_remains": "non_identifying",
    }
    dump_json(REPORTS / "external_hydrology_diagnostics.json", result)
    return result


def main() -> None:
    require_runtime()
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    start = hash_manifest()
    dump_json(REPORTS / "parent_hashes_start.json", start)
    if start["canonical_interface"]["sha256"] != EXPECTED_CANON_SHA:
        raise RuntimeError("Canonical interface hash does not match the structural lock")

    hydro = pd.read_parquet(CANON)
    if len(hydro) != 46_920 or hydro.comid.nunique() != 230 or sorted(hydro.year.unique()) != list(range(2006, 2023)):
        raise RuntimeError("Canonical Q72 interface has unexpected coverage")
    climate = build_climatology(hydro)
    tn, folds = build_tn_registry()
    catchments = gpd.read_file(CATCHMENTS)
    catchments["reach_id"] = catchments.reach_id.astype(int)
    if len(catchments) != 230 or catchments.crs is None or not catchments.crs.is_projected:
        raise RuntimeError("Expected 230 equal-area catchments")
    wg = aggregate_watergap(catchments)
    reach_terminal = terminal_map()
    diagnostics = external_diagnostics(hydro, wg, reach_terminal)

    runtime = {
        "python": sys.version,
        "platform": platform.platform(),
        "sys_prefix": sys.prefix,
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "pyarrow": pyarrow.__version__,
            "geopandas": gpd.__version__,
            "shapely": shapely.__version__,
            "xarray": xr.__version__,
        },
        "thread_limits": {k: os.environ.get(k) for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]},
    }
    dump_json(REPORTS / "runtime_environment.json", runtime)
    end = hash_manifest()
    dump_json(REPORTS / "parent_hashes_end.json", end)
    unchanged = start == end
    if not unchanged:
        raise RuntimeError("An upstream artifact changed during stage 1")
    summary = {
        "scenario_id": "20260815_1",
        "status": "complete",
        "q72_climatology": {"rows": len(climate), "reaches": int(climate.reach_id.nunique()), "months": int(climate.month.nunique()), "source_years": [2006, 2015]},
        "tn_registry": {"rows": len(tn), "stations": int(tn.station_key.nunique()), "reaches": int(tn.reach_id.nunique()), "development_rows": int((tn.year <= 2021).sum()), "locked_2022_rows": int((tn.year == 2022).sum())},
        "fold_registry_rows": len(folds),
        "watergap": {"rows": len(wg), "reaches": int(wg.reach_id.nunique()), "months": 204},
        "external_hydrology_status": diagnostics["hydrology_external_consistency"],
        "external_evidence_used_for_n_selection": False,
        "parent_hashes_unchanged": unchanged,
    }
    dump_json(REPORTS / "stage1_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
