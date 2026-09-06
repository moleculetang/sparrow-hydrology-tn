from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gamma


@dataclass(frozen=True)
class ElementNResult:
    years: np.ndarray
    ma: np.ndarray
    mp: np.ndarray
    mmin: np.ndarray
    leaching_rate: np.ndarray
    subsurface_load: np.ndarray
    wastewater_load: np.ndarray
    total_load: np.ndarray


def _single(table: pd.DataFrame, name: str) -> float:
    values = table.loc[table["ParamName"].eq(name), "GlobalValue"].dropna().to_numpy(dtype=float)
    if len(values) != 1:
        raise ValueError(f"Expected exactly one GlobalValue for {name}; found {len(values)}")
    return float(values[0])


def _by_land_use(table: pd.DataFrame, name: str) -> np.ndarray:
    row = table.loc[table["ParamName"].eq(name), ["LU_Other", "LU_Pasture", "LU_Cropland"]]
    if len(row) != 1:
        raise ValueError(f"Expected exactly one land-use row for {name}; found {len(row)}")
    values = row.iloc[0].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite land-use parameter for {name}")
    return values


def _assign_land_use(cropland_ratio: float, pasture_ratio: float, bins: int) -> np.ndarray:
    centers = (np.arange(bins, dtype=float) + 0.5) / bins
    result = np.zeros(bins, dtype=np.int8)
    result[centers <= cropland_ratio] = 2
    result[(centers > cropland_ratio) & (centers <= cropland_ratio + pasture_ratio)] = 1
    return result


def _read_inputs(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    parameters = pd.read_excel(root / "para.xlsx")
    data = root / "input_data"
    return (
        parameters,
        pd.read_csv(data / "input_population.csv"),
        pd.read_csv(data / "input_q.csv"),
        pd.read_csv(data / "input_ns_all_lu.csv"),
        pd.read_csv(data / "input_landuse_ratios.csv"),
    )


def run_elementn_reference(root: str | Path) -> ElementNResult:
    """Replicate the local ``main.m`` numerical updates without plotting or file writes."""
    root = Path(root)
    parameters, population, discharge, surplus, landuse = _read_inputs(root)
    dt = _single(parameters, "dt")
    n_years = int(_single(parameters, "num_simulation_years"))
    watershed_area = _single(parameters, "watershed_area_total")
    bins = int(_single(parameters, "num_s_bins"))
    if not all(len(frame) == n_years for frame in (population, discharge, surplus, landuse)):
        raise ValueError("ELEMeNT input lengths do not match num_simulation_years")
    expected_years = np.arange(n_years)
    for frame in (population, discharge, surplus, landuse):
        if not np.array_equal(pd.to_numeric(frame["Year"], errors="raise").to_numpy(), expected_years):
            raise ValueError("ELEMeNT input years must be 0..n-1")

    lu_crop = int(_single(parameters, "LU_ID_Cropland"))
    lu_pasture = int(_single(parameters, "LU_ID_Pasture"))
    lu_other = int(_single(parameters, "LU_ID_Other"))
    if (lu_other, lu_pasture, lu_crop) != (0, 1, 2):
        raise ValueError("Reference implementation requires the source-code land-use IDs 0/1/2")
    gamma_shape = _single(parameters, "gamma_shape")
    gamma_scale = _single(parameters, "gamma_scale")
    den_sub = _single(parameters, "den_sub")
    max_travel = int(_single(parameters, "max_travel_time_steps"))
    lambda_pop = _single(parameters, "lambda_pop")
    per_capita_n = _single(parameters, "per_capita_N")

    ka = _by_land_use(parameters, "min_ka")
    kp = _by_land_use(parameters, "min_kp")
    h = _by_land_use(parameters, "h_factor")
    mp_pristine = _by_land_use(parameters, "Mp_s_pris")
    den_soil = _by_land_use(parameters, "den_s_source")
    porosity = _by_land_use(parameters, "n_porosity")
    soil_water = _by_land_use(parameters, "sw_param")
    soil_depth = _by_land_use(parameters, "V_soil_depth")
    initial_ma = np.array([_single(parameters, "Ma_initial_other"), _single(parameters, "Ma_initial_past"), _single(parameters, "Ma_initial_crop")])
    initial_mp = np.array([_single(parameters, "Mp_initial_other"), _single(parameters, "Mp_initial_past"), _single(parameters, "Mp_initial_crop")])
    initial_mmin = np.array([_single(parameters, "Mmin_initial_other"), _single(parameters, "Mmin_initial_past"), _single(parameters, "Mmin_initial_crop")])

    ma = np.zeros((bins, n_years), dtype=float)
    mp = np.zeros((bins, n_years), dtype=float)
    mmin = np.zeros((bins, n_years), dtype=float)
    leaching = np.zeros((bins, n_years), dtype=float)
    lu = _assign_land_use(float(landuse.loc[0, "CroplandRatio"]), float(landuse.loc[0, "PastureRatio"]), bins)
    ma[:, 0] = initial_ma[lu]
    mp[:, 0] = initial_mp[lu]
    mmin[:, 0] = initial_mmin[lu]
    subsurface = np.zeros(n_years, dtype=float)
    wastewater = np.zeros(n_years, dtype=float)
    total = np.zeros(n_years, dtype=float)
    leaching_history = np.zeros(n_years, dtype=float)
    kernel_tau = np.arange(max_travel, dtype=float) * dt
    kernel = gamma.pdf(kernel_tau, a=gamma_shape, scale=gamma_scale) * dt * np.exp(-den_sub * kernel_tau)

    for t in range(n_years):
        if t > 0:
            next_lu = _assign_land_use(float(landuse.loc[t, "CroplandRatio"]), float(landuse.loc[t, "PastureRatio"]), bins)
        else:
            next_lu = lu.copy()
        previous_lu = lu if t > 0 else next_lu
        q_unit_area = float(discharge.loc[t, "TotalDischarge_m3_per_year"]) / (watershed_area * 10000.0)
        ns_by_lu = np.array(
            [
                float(surplus.loc[t, "NS_other_kg_per_ha_per_year"]),
                float(surplus.loc[t, "NS_pasture_kg_per_ha_per_year"]),
                float(surplus.loc[t, "NS_cropland_kg_per_ha_per_year"]),
            ]
        )
        previous_ma = ma[:, t - 1] if t > 0 else ma[:, 0]
        previous_mp = mp[:, t - 1] if t > 0 else mp[:, 0]
        previous_mmin = mmin[:, t - 1] if t > 0 else mmin[:, 0]
        active_ma = previous_ma.copy()
        active_mp = previous_mp.copy()
        changed_to_crop = (next_lu == lu_crop) & (previous_lu <= lu_pasture)
        transfer = np.minimum(0.7 * mp_pristine[next_lu], active_mp)
        transfer = np.where(changed_to_crop, transfer, 0.0)
        active_mp -= transfer
        active_ma += transfer
        n_surface = ns_by_lu[next_lu]  # sf_t_current is hard-coded to zero in main.m.
        mineral_ma = ka[next_lu] * active_ma
        mineral_mp = kp[next_lu] * active_mp
        ma[:, t] = np.maximum(0.0, active_ma + ((1.0 - h[next_lu]) * n_surface - mineral_ma) * dt)
        mp[:, t] = np.maximum(0.0, active_mp + (h[next_lu] * n_surface - mineral_mp) * dt)
        water_volume = porosity[next_lu] * soil_water[next_lu] * soil_depth[next_lu]
        if not np.all(water_volume > 0):
            raise ValueError("Non-positive source-zone water volume")
        # ``mineraln_dynamics.m`` applies this branch separately to every
        # land-use bin.  The reference inputs happen to have one common water
        # volume, but retaining the vector condition is necessary for an exact
        # implementation if the local parameter workbook is later changed.
        fully_flushed = q_unit_area * dt >= water_volume
        js = np.where(
            previous_mmin > 0.0,
            np.where(fully_flushed, previous_mmin / dt, previous_mmin * q_unit_area / water_volume),
            0.0,
        )
        leaching[:, t] = np.maximum(js, 0.0)
        mineral_input = mineral_ma + mineral_mp
        mmin[:, t] = np.maximum(
            0.0,
            previous_mmin + (mineral_input - den_soil[next_lu] * previous_mmin - leaching[:, t]) * dt,
        )
        leaching_history[t] = float(leaching[:, t].mean())
        k = min(max_travel, t + 1)
        subsurface[t] = float(np.dot(leaching_history[t - np.arange(k)], kernel[:k]))
        population_total = float(population.loc[t, "Population_density_year"]) * watershed_area
        wastewater[t] = (1.0 - lambda_pop) * population_total * per_capita_n / watershed_area
        total[t] = subsurface[t] + wastewater[t]
        lu = next_lu
    return ElementNResult(
        years=np.arange(n_years, dtype=int),
        ma=ma,
        mp=mp,
        mmin=mmin,
        leaching_rate=leaching,
        subsurface_load=subsurface,
        wastewater_load=wastewater,
        total_load=total,
    )
