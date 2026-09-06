from __future__ import annotations

import json

import numpy as np
import pandas as pd

from hleg_shared import (
    BETA_GRID,
    CACHE,
    FORMAL_MUS,
    OUT,
    REPORTS,
    candidate_path_frame,
    cohort_state_end_2005,
    development_observations,
    dump_json,
    formal_specs,
    periodic_initial_cohorts,
    require_runtime,
    route_batch,
    simulate_age,
    simulate_bulk,
    synthetic_semantics_tests,
)


def main() -> None:
    require_runtime()
    preflight = json.loads((REPORTS / "stage0_preflight.json").read_text(encoding="utf-8"))
    if preflight["status"] != "PASS":
        raise RuntimeError("STOP_STAGE0_PREFLIGHT_NOT_PASS")
    observations = development_observations()
    state_registry = pd.read_parquet(OUT / "q72_hydrologic_state_registry_1961_2022.parquet")
    state_registry = state_registry.sort_values(["year", "month", "reach_id"])
    routed_dir = CACHE / "candidate_routed"
    routed_dir.mkdir(parents=True, exist_ok=True)

    synthetic = synthetic_semantics_tests()
    synthetic.to_parquet(OUT / "synthetic_cohort_semantics_tests.parquet", index=False)
    if not synthetic["pass"].all():
        raise RuntimeError("STOP_SYNTHETIC_COHORT_SEMANTICS_FAILED")

    path_rows: list[pd.DataFrame] = []
    age_summaries: list[pd.DataFrame] = []
    engineering_rows: list[dict[str, object]] = []
    init_rows: list[dict[str, object]] = []

    for index, spec in enumerate(formal_specs(), start=1):
        model_id = str(spec["model_id"])
        mu = int(spec["delivery_mu_month"])
        base = pd.read_parquet(CACHE / "parent_local" / f"{model_id}.parquet").sort_values(["year", "month", "reach_id"])
        spinup = pd.read_parquet(CACHE / "parent_local" / f"{model_id}__spinup_states.parquet").sort_values("reach_id")
        reach_ids = np.sort(base.reach_id.unique().astype(int))
        all_times = list(base[["year", "month"]].drop_duplicates().sort_values(["year", "month"]).itertuples(index=False, name=None))
        n_r = len(reach_ids)
        n_t = len(all_times)
        ordered = base.sort_values(["year", "month", "reach_id"])
        arrays = {
            column: ordered[column].to_numpy(float).reshape(n_t, n_r)
            for column in (
                "quick_tn_release_kg_n", "gw_tn_release_kg_n", "gw_path_n_input_kg_n",
                "gw_state_end_kg_n", "q_local_total_mm", "catchment_area_km2", "gw_discharge_mm",
            )
        }
        h_all = state_registry.H_anom.to_numpy(float).reshape(n_t, n_r)
        if not np.array_equal(state_registry.reach_id.unique(), reach_ids):
            raise RuntimeError("reach ordering mismatch")
        start_2006 = all_times.index((2006, 1))
        end_2005 = start_2006 - 1
        times = all_times[start_2006:]
        gw_input = arrays["gw_path_n_input_kg_n"][start_2006:]
        parent_gw = arrays["gw_tn_release_kg_n"][start_2006:]
        quick = arrays["quick_tn_release_kg_n"][start_2006:]
        water = (
            arrays["q_local_total_mm"][start_2006:]
            * arrays["catchment_area_km2"][start_2006:]
            * 1000.0
        )
        h = h_all[start_2006:]
        gate = (arrays["gw_discharge_mm"][start_2006:] > 1e-12).astype(float)
        parent_end_2005 = arrays["gw_state_end_kg_n"][end_2005]

        rho = mu / (1.0 + mu)
        initial_exact, initial_tail_m, initial_tail_j, initial_tail_k, initial_audit = periodic_initial_cohorts(
            arrays["gw_path_n_input_kg_n"][:12],
            spinup.gw_end_kg_n.to_numpy(float),
            rho,
        )
        end_exact, end_tail_m, end_tail_j, end_tail_k = cohort_state_end_2005(
            arrays["gw_path_n_input_kg_n"][:start_2006],
            initial_exact, initial_tail_m, initial_tail_j, initial_tail_k, rho,
        )
        end_total = end_exact.sum(axis=1) + end_tail_m
        end_diff = float(np.max(np.abs(end_total - parent_end_2005)))
        init_rows.append({"model_id": model_id, **initial_audit, "end_2005_total_max_abs_kg_n": end_diff})
        if not np.allclose(end_total, parent_end_2005, rtol=1e-12, atol=1e-6):
            raise RuntimeError(f"STOP_COHORT_PARENT_END2005_NOT_REPRODUCED {model_id}: {end_diff}")

        bulk_release, bulk_diag, bulk_audit = simulate_bulk(gw_input, parent_end_2005, h, gate, mu)
        age_release, age_diag, age_audit = simulate_age(
            gw_input, end_exact, end_tail_m, end_tail_j, end_tail_k, h, gate, mu, times,
        )
        zero_index = int(np.flatnonzero(BETA_GRID == 0.0)[0])
        bulk_zero = float(np.max(np.abs(bulk_release[:, zero_index, :] - parent_gw)))
        age_zero = float(np.max(np.abs(age_release[:, zero_index, :] - parent_gw)))
        if not np.allclose(bulk_release[:, zero_index, :], parent_gw, rtol=1e-12, atol=1e-6):
            raise RuntimeError(f"STOP_BULK_BETA0_NOT_PARENT {model_id}: {bulk_zero}")
        if not np.allclose(age_release[:, zero_index, :], parent_gw, rtol=1e-12, atol=1e-6):
            raise RuntimeError(f"STOP_AGE_BETA0_NOT_PARENT {model_id}: {age_zero}")
        # The registered nested point is the frozen Parent itself. After the
        # numerical equivalence audit, replace the computed beta=0 slice with
        # the authoritative parent array so machine-rounding cannot create a
        # spurious positive/negative OOF result for an identical structure.
        bulk_release[:, zero_index, :] = parent_gw
        age_release[:, zero_index, :] = parent_gw

        parent_paths = candidate_path_frame(
            model_id, "PARENT", np.asarray([0.0]), quick, parent_gw, water, reach_ids, times, observations,
        )
        bulk_paths = candidate_path_frame(
            model_id, "HLEG_BULK", BETA_GRID, quick, bulk_release, water, reach_ids, times, observations,
        )
        age_paths = candidate_path_frame(
            model_id, "HLEG_AGE", BETA_GRID, quick, age_release, water, reach_ids, times, observations,
        )
        path_rows.extend([parent_paths, bulk_paths, age_paths])

        rq, terminal = route_batch(quick, reach_ids)
        rw, _ = route_batch(water, reach_ids)
        rg_parent, _ = route_batch(parent_gw, reach_ids)
        rg_bulk, _ = route_batch(bulk_release, reach_ids)
        rg_age, _ = route_batch(age_release, reach_ids)
        np.savez_compressed(
            routed_dir / f"{model_id}.npz",
            reach_ids=reach_ids,
            years=np.asarray([year for year, _ in times], dtype=np.int16),
            months=np.asarray([month for _, month in times], dtype=np.int8),
            routed_quick=rq,
            routed_water=rw,
            routed_gw_parent=rg_parent,
            routed_gw_bulk=rg_bulk,
            routed_gw_age=rg_age,
            terminal_tree=np.asarray([terminal[int(rid)] for rid in reach_ids], dtype=np.int32),
            beta_grid=BETA_GRID,
        )

        bulk_diag["model_id"] = model_id
        bulk_diag["mechanism"] = "HLEG_BULK"
        age_diag["model_id"] = model_id
        age_diag["mechanism"] = "HLEG_AGE"
        age_summaries.extend([bulk_diag, age_diag])
        engineering_rows.extend([
            {"model_id": model_id, "mechanism": "HLEG_BULK", "beta0_parent_max_abs_kg_n": bulk_zero, **bulk_audit},
            {"model_id": model_id, "mechanism": "HLEG_AGE", "beta0_parent_max_abs_kg_n": age_zero, **age_audit},
        ])
        print(f"stage1 mechanism {index:02d}/12 {model_id}", flush=True)

    candidate_paths = pd.concat(path_rows, ignore_index=True)
    candidate_paths.to_parquet(OUT / "candidate_development_observation_paths.parquet", index=False)
    pd.concat(age_summaries, ignore_index=True).to_parquet(OUT / "candidate_cohort_diagnostic_summary.parquet", index=False)
    pd.DataFrame(engineering_rows).to_parquet(OUT / "candidate_engineering_audit.parquet", index=False)
    pd.DataFrame(init_rows).to_parquet(OUT / "cohort_initialization_audit.parquet", index=False)

    age_summary = pd.concat(age_summaries, ignore_index=True)
    age_nonzero = age_summary.loc[age_summary.mechanism.eq("HLEG_AGE") & age_summary.beta_h.ne(0)]
    semantics_pass = bool((age_nonzero.semantics_direction_fraction >= 0.99).all())
    engineering = pd.DataFrame(engineering_rows)
    audit = {
        "status": "PASS" if semantics_pass and (engineering.minimum_state_kg_n >= -1e-6).all() else "FAIL",
        "candidate_path_rows": int(len(candidate_paths)),
        "formal_models": 12,
        "mechanisms": ["PARENT", "HLEG_BULK", "HLEG_AGE"],
        "beta_values_per_dynamic_mechanism": 9,
        "synthetic_semantics_all_pass": bool(synthetic["pass"].all()),
        "real_state_semantics_fraction_ge_0_99": semantics_pass,
        "beta0_exact_parent_all_models": bool((engineering.beta0_parent_max_abs_kg_n <= 1e-6).all()),
        "minimum_state_kg_n": float(engineering.minimum_state_kg_n.min()),
        "TN_2022_rows_materialized": 0,
        "storage_policy": "all candidates retain routed totals and compact diagnostics; full cohort arrays are replayable and not permanently stored",
    }
    dump_json(REPORTS / "stage1_mechanism_audit.json", audit)
    if audit["status"] != "PASS":
        raise RuntimeError("STOP_STAGE1_MECHANISM_AUDIT_FAILED")


if __name__ == "__main__":
    main()
