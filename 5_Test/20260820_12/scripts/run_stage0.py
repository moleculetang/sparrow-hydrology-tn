from __future__ import annotations

import json

import numpy as np
import pandas as pd

from a0_shared import (
    CACHE, OUT, PARENT_LOCAL_CACHE, PARENT_OBSERVATION_PATHS, REPORTS,
    authoritative_inputs, dump_json, formal_specs, harmonic_weights, hash_manifest,
    parent_shared, require_runtime, route_frame, seasonal_positive_arrays, simulate_seasonal,
)


def direct_parent_routed(model_id: str) -> pd.DataFrame:
    path = PARENT_LOCAL_CACHE / f"{model_id}.parquet"
    frame = pd.read_parquet(path)
    return route_frame(frame.loc[frame.year.between(2016, 2022)])


def max_differences(reference: pd.DataFrame, candidate: pd.DataFrame) -> dict[str, float]:
    keys = ["reach_id", "year", "month"]
    joined = reference.merge(candidate, on=keys, suffixes=("_reference", "_candidate"), validate="one_to_one")
    result: dict[str, float] = {}
    for column in ("routed_quick_tn_kg_n", "routed_gw_tn_kg_n", "routed_water_volume_m3"):
        delta = np.abs(joined[f"{column}_candidate"].to_numpy(float) - joined[f"{column}_reference"].to_numpy(float))
        scale = max(float(np.max(np.abs(joined[f"{column}_reference"].to_numpy(float)))), 1.0)
        result[f"{column}_max_abs"] = float(delta.max())
        result[f"{column}_max_relative_scale"] = float(delta.max() / scale)
    return result


def weighted_basis(basis: pd.DataFrame, weights: np.ndarray) -> pd.DataFrame:
    work = basis.copy()
    work["weight"] = work.basis_month.map({month: float(weights[month - 1]) for month in range(1, 13)})
    for column in ("routed_quick_tn_kg_n", "routed_gw_tn_kg_n"):
        work[column] *= work.weight
    return work.groupby(["reach_id", "year", "month"], as_index=False).agg(
        routed_quick_tn_kg_n=("routed_quick_tn_kg_n", "sum"),
        routed_gw_tn_kg_n=("routed_gw_tn_kg_n", "sum"),
        routed_water_volume_m3=("routed_water_volume_m3", "first"),
        terminal_tree_id=("terminal_tree_id", "first"),
    )


def main() -> None:
    require_runtime()
    for path in (CACHE / "basis_routed", CACHE / "direct_parent_routed", OUT, REPORTS):
        path.mkdir(parents=True, exist_ok=True)
    dump_json(REPORTS / "input_hashes_start.json", hash_manifest(authoritative_inputs()))

    shared = parent_shared()
    reach_ids, times, arrays, _ = shared.prepare_arrays()
    if np.any(arrays["negative_legacy_eligible_n_surplus_kg_n_month"] > 0):
        raise RuntimeError("STOP_UNREGISTERED_NEGATIVE_REACH_YEAR")

    spin_rows: list[dict[str, object]] = []
    engineering_rows: list[dict[str, object]] = []
    reproduction_rows: list[dict[str, object]] = []
    closure_rows: list[dict[str, object]] = []
    specs = formal_specs()
    for model_index, spec in enumerate(specs, start=1):
        model_id = str(spec["model_id"])
        direct = direct_parent_routed(model_id)
        direct.to_parquet(CACHE / "direct_parent_routed" / f"{model_id}.parquet", index=False)
        basis_parts: list[pd.DataFrame] = []
        for basis_month in range(1, 13):
            weights = np.zeros(12, dtype=float)
            weights[basis_month - 1] = 1.0
            local, audit, spin = simulate_seasonal(spec, weights)
            routed = route_frame(local)
            routed["basis_month"] = basis_month
            basis_parts.append(routed)
            spin_rows.append({"model_id": model_id, "basis_month": basis_month, **spin})
            engineering_rows.append({"model_id": model_id, "basis_month": basis_month, **audit})
        basis = pd.concat(basis_parts, ignore_index=True)
        basis.to_parquet(CACHE / "basis_routed" / f"{model_id}.parquet", index=False)

        uniform = weighted_basis(basis, np.full(12, 1.0 / 12.0))
        differences = max_differences(direct, uniform)
        reproduction_rows.append({
            "model_id": model_id, "comparison": "uniform_basis_vs_direct_parent", **differences,
            "pass": bool(max(value for key, value in differences.items() if key.endswith("relative_scale")) <= 1e-10),
        })

        positive = arrays["positive_legacy_eligible_n_surplus_kg_n_month"]
        for label, weights in (("uniform", np.full(12, 1.0 / 12.0)), ("registered_example", harmonic_weights(0.3, -0.2))):
            seasonal = seasonal_positive_arrays(times, positive, weights)
            years = sorted(set(year for year, _ in times))
            errors = []
            relative_errors = []
            scales = []
            for year in years:
                idx = [i for i, value in enumerate(times) if value[0] == year]
                expected = positive[idx].sum(axis=0)
                error = np.abs(seasonal[idx].sum(axis=0) - expected)
                errors.append(float(np.max(error)))
                scales.append(float(np.max(np.abs(expected))))
                relative_errors.append(float(np.max(error / np.maximum(np.abs(expected), 1.0))))
            max_abs = max(errors)
            max_scale = max(scales)
            closure_rows.append({
                "model_id": model_id, "weight_case": label,
                "annual_max_abs_error_kg_n": max_abs,
                "annual_max_relative_error": max(relative_errors),
                "combined_tolerance_kg_n": 1e-9 + 1e-12 * max_scale,
                "pass": bool(max_abs <= 1e-9 + 1e-12 * max_scale),
            })
        print(f"stage0 basis {model_index:02d}/12 {model_id}", flush=True)

    pd.DataFrame(spin_rows).to_parquet(OUT / "basis_spinup_audit.parquet", index=False)
    pd.DataFrame(engineering_rows).to_parquet(OUT / "basis_engineering_audit.parquet", index=False)
    reproduction = pd.DataFrame(reproduction_rows)
    reproduction.to_parquet(OUT / "parent_reproduction_audit.parquet", index=False)
    pd.DataFrame(closure_rows).to_parquet(OUT / "annual_mass_closure_audit.parquet", index=False)

    tests = [
        ("S0_mu_036m", 0.30, -0.20),
        ("S1_tau_012m_mu_240m", -0.40, 0.25),
    ]
    superposition_rows = []
    lookup = {str(spec["model_id"]): spec for spec in specs}
    for model_id, a, b in tests:
        weights = harmonic_weights(a, b)
        direct_local, audit, spin = simulate_seasonal(lookup[model_id], weights)
        direct = route_frame(direct_local)
        basis = pd.read_parquet(CACHE / "basis_routed" / f"{model_id}.parquet")
        combined = weighted_basis(basis, weights)
        differences = max_differences(direct, combined)
        superposition_rows.append({
            "model_id": model_id, "a": a, "b": b, **differences,
            "direct_spinup_converged": bool(spin["converged"]),
            "direct_mass_balance_max_abs_kg_n": float(audit["max_abs_mass_balance_error_kg_n"]),
            "pass": bool(max(value for key, value in differences.items() if key.endswith("relative_scale")) <= 1e-10),
        })
    superposition = pd.DataFrame(superposition_rows)
    superposition.to_parquet(OUT / "linear_superposition_audit.parquet", index=False)

    frozen_parent_paths = pd.read_parquet(PARENT_OBSERVATION_PATHS, filters=[("mechanism", "==", "PARENT")])
    path_rows = []
    for spec in specs:
        model_id = str(spec["model_id"])
        direct = pd.read_parquet(CACHE / "direct_parent_routed" / f"{model_id}.parquet")
        expected = frozen_parent_paths.loc[frozen_parent_paths.model_id.eq(model_id)]
        keys = ["reach_id", "year", "month"]
        joined = expected.merge(direct, on=keys, suffixes=("_expected", "_direct"), validate="many_to_one")
        diffs = {}
        for column in ("routed_quick_tn_kg_n", "routed_gw_tn_kg_n", "routed_water_volume_m3"):
            diffs[f"{column}_max_abs"] = float(np.max(np.abs(joined[f"{column}_expected"] - joined[f"{column}_direct"])))
        path_rows.append({"model_id": model_id, **diffs, "pass": bool(max(diffs.values()) <= 1e-6)})
    pd.DataFrame(path_rows).to_parquet(OUT / "frozen_parent_path_reproduction_audit.parquet", index=False)

    stage_pass = bool(
        reproduction["pass"].all()
        and superposition["pass"].all()
        and pd.DataFrame(closure_rows)["pass"].all()
        and pd.DataFrame(engineering_rows).max_relative_mass_balance_error.max() <= 1e-12
        and pd.DataFrame(spin_rows).converged.all()
        and pd.DataFrame(path_rows)["pass"].all()
    )
    result = {
        "scenario_id": "20260820_12", "stage": 0, "status": "PASS" if stage_pass else "FAIL",
        "negative_surplus_rows": 0, "basis_cases": len(spin_rows),
        "parent_reproduction_pass": bool(reproduction["pass"].all()),
        "linear_superposition_pass": bool(superposition["pass"].all()),
        "annual_mass_closure_pass": bool(pd.DataFrame(closure_rows)["pass"].all()),
        "basis_spinup_converged": bool(pd.DataFrame(spin_rows).converged.all()),
    }
    dump_json(REPORTS / "stage0_preflight.json", result)
    if not stage_pass:
        raise RuntimeError("STOP_A0_STAGE0_PREFLIGHT_FAILED")


if __name__ == "__main__":
    main()
