from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(r"E:\SPARROW\5_Test\20260817_2")
OUT = ROOT / "outputs"
REPORTS = ROOT / "reports"
S17_1 = Path(r"E:\SPARROW\5_Test\20260817_1")
S16_4 = Path(r"E:\SPARROW\5_Test\20260816_4")
S16_7 = Path(r"E:\SPARROW\5_Test\20260816_7")
S15_2 = Path(r"E:\SPARROW\5_Test\20260815_2")
sys.path.insert(0, str(ROOT / "scripts"))
import legacy17_core as core  # noqa: E402

HORIZON = 2400


def require_runtime() -> None:
    if Path(sys.prefix).name.lower() != "sparrow":
        raise RuntimeError(sys.prefix)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        if os.environ.get(key) != "1":
            raise RuntimeError(f"{key}=1 required")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def compare_frames(actual: pd.DataFrame, expected: pd.DataFrame, columns: list[str]) -> dict[str, object]:
    keys = ["reach_id", "year", "month"]
    a = actual.sort_values(keys).reset_index(drop=True)
    b = expected.sort_values(keys).reset_index(drop=True)
    result = {"rows_actual": len(a), "rows_expected": len(b), "keys_equal": bool(a[keys].equals(b[keys])), "columns": {}}
    passed = len(a) == len(b) and result["keys_equal"]
    if passed:
        for column in columns:
            av, bv = a[column].to_numpy(float), b[column].to_numpy(float)
            diff = float(np.max(np.abs(av - bv)))
            close = bool(np.allclose(av, bv, rtol=1e-12, atol=1e-9, equal_nan=True))
            result["columns"][column] = {"max_abs_difference": diff, "pass": close}
            passed = passed and close
    result["pass"] = bool(passed)
    return result


def preflight() -> dict[str, object]:
    parent = core.parent_core()
    reach_ids, times, arrays, early = parent.prepare_model_arrays()
    columns = [
        "son_state_end_kg_n", "mobile_state_end_kg_n", "quick_state_end_kg_n", "gw_state_end_kg_n",
        "quick_tn_release_kg_n", "gw_tn_release_kg_n", "local_tn_release_kg_n",
        "negative_removed_kg_n", "negative_unmet_kg_n", "mass_balance_error_kg_n",
    ]
    details = {}
    replays = {}
    for model_id in ("S0_mu_036m", "S1_tau_012m_mu_036m", "S1_tau_012m_mu_240m"):
        structure, tau, mu = core.model_spec(model_id)
        new = core.simulate_history_replay(model_id).loc[lambda x: x.year <= 2021]
        old, _ = parent.simulate_candidate_totals(model_id, structure, tau, mu, reach_ids, times, arrays, early)
        old = old.loc[old.year <= 2021]
        details[f"{model_id}_parent_replay"] = compare_frames(new, old, columns)
        replays[model_id] = new

    # Persisted-output cross-checks prevent two implementations drifting together.
    frozen_recent = pd.read_parquet(S16_4 / "outputs" / "candidate_routed_paths_2016_2021.parquet")
    for model_id in ("S0_mu_036m", "S1_tau_012m_mu_036m", "S1_tau_012m_mu_240m"):
        local = replays[model_id].loc[replays[model_id].year.between(2016, 2021)]
        n_t = local[["year", "month"]].drop_duplicates().shape[0]
        quick = local.pivot(index=["year", "month"], columns="reach_id", values="quick_tn_release_kg_n")
        gw = local.pivot(index=["year", "month"], columns="reach_id", values="gw_tn_release_kg_n")
        reach = quick.columns.to_numpy(int)
        rq, _, terminal = core.route_matrix(quick.to_numpy(float), reach)
        rg, _, _ = core.route_matrix(gw.to_numpy(float), reach)
        rows = pd.DataFrame({
            "reach_id": np.tile(reach, n_t),
            "year": np.repeat([x[0] for x in quick.index], len(reach)),
            "month": np.repeat([x[1] for x in quick.index], len(reach)),
            "routed_quick_tn_kg_n": rq.reshape(-1),
            "routed_gw_tn_kg_n": rg.reshape(-1),
        })
        frozen = frozen_recent.loc[frozen_recent.model_id.eq(model_id), rows.columns]
        details[f"{model_id}_persisted_routed"] = compare_frames(rows, frozen, ["routed_quick_tn_kg_n", "routed_gw_tn_kg_n"])
    if not all(item["pass"] for item in details.values()):
        dump(REPORTS / "legacy17_core_parent_reproduction_audit.json", {"pass": False, "stop_code": "STOP_LEGACY17_CORE_NOT_REPRODUCED", "details": details})
        raise RuntimeError("STOP_LEGACY17_CORE_NOT_REPRODUCED")
    result = {"pass": True, "rtol": 1e-12, "atol_kg_n": 1e-9, "details": details}
    dump(REPORTS / "legacy17_core_parent_reproduction_audit.json", result)
    return result


def weights(reach_ids: np.ndarray) -> pd.DataFrame:
    monthly = pd.read_parquet(S15_2 / "outputs" / "reach_month_n_inputs_hydrology_1961_2022.parquet")
    dev = monthly.loc[monthly.year.between(2016, 2021)]
    positive = dev.groupby("reach_id").positive_legacy_eligible_n_surplus_kg_n_month.sum().reindex(reach_ids).to_numpy(float)
    area = dev.groupby("reach_id").catchment_area_km2.first().reindex(reach_ids).to_numpy(float)
    table = pd.DataFrame({
        "reach_id": reach_ids,
        "positive_diffuse_n_2016_2021_kg_n": positive,
        "n_source_weight": positive / positive.sum(),
        "catchment_area_km2": area,
        "area_weight": area / area.sum(),
    })
    table["positive_input_semantics"] = "sum of positive_legacy_eligible_n_surplus_kg_n_month after annual-to-monthly ledger construction"
    table["pulse_b_unit_semantics"] = "1 kg N in one monthly post-ledger timestep per reach"
    if abs(table.n_source_weight.sum() - 1) > 1e-12 or abs(table.area_weight.sum() - 1) > 1e-12:
        raise RuntimeError("weight normalization")
    table.to_parquet(OUT / "pulse_spatial_weight_registry.parquet", index=False)
    return table


class WideWriter:
    def __init__(self, path: Path):
        self.path = path
        self.writer = None

    def write(self, frame: pd.DataFrame) -> None:
        table = pa.Table.from_pandas(frame, preserve_index=False)
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.path, table.schema, compression="zstd")
        self.writer.write_table(table)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()


def curve_frame(model_id: str, start_month: int, release: np.ndarray, reach_ids: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame(release, columns=[f"reach_{rid}" for rid in reach_ids])
    frame.insert(0, "lag_month", np.arange(len(frame), dtype=int))
    frame.insert(0, "start_month", start_month)
    frame.insert(0, "model_id", model_id)
    frame["pulse_input_semantics"] = "1 kg N monthly post-ledger mass per reach"
    return frame


def aggregate_curves(local: np.ndarray, reach_ids: np.ndarray, weight: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weighted_local = local * weight[None, :]
    routed, terminals, _ = core.route_matrix(weighted_local, reach_ids)
    lookup = {int(rid): i for i, rid in enumerate(reach_ids)}
    terminal_curves = np.column_stack([routed[:, lookup[int(rid)]] for rid in terminals])
    basin = terminal_curves.sum(axis=1)
    return basin, terminal_curves, terminals


def main() -> None:
    require_runtime()
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    if not json.loads((S17_1 / "reports" / "completion_audit.json").read_text(encoding="utf-8"))["pass"]:
        raise RuntimeError("stage1 incomplete")
    protected = [
        S17_1 / "outputs" / "formal_12model_structural_ensemble.parquet",
        S17_1 / "outputs" / "analysis_model_registry_16.parquet",
        S15_2 / "outputs" / "reach_month_n_inputs_hydrology_1961_2022.parquet",
        core.PARENT_CORE,
    ]
    start_hash = {str(p): sha256(p) for p in protected}
    dump(ROOT / "upstream_manifest.json", start_hash)
    preflight()

    formal = pd.read_parquet(S17_1 / "outputs" / "formal_12model_structural_ensemble.parquet")
    analysis = pd.read_parquet(S17_1 / "outputs" / "analysis_model_registry_16.parquet")
    formal_ids = formal.model_id.tolist()
    analysis_ids = analysis.model_id.tolist()
    reach_ids, climate = core.climate_arrays()
    weight_table = weights(reach_ids)
    schemes = {"n_source_weighted": weight_table.n_source_weight.to_numpy(float), "area_weighted": weight_table.area_weight.to_numpy(float)}

    # Delivery-only curves and reach metrics.
    delivery_curves = []
    delivery_metrics = []
    for mu in (12, 36, 60, 96, 144, 240):
        intrinsic = core.simulate_delivery_pulse(mu, 1, False, HORIZON)["release"][:, 0]
        for lag, value in enumerate(intrinsic):
            delivery_curves.append({"delivery_mu_month": mu, "start_month": 1, "gate": "ungated", "weight_scheme": "intrinsic", "lag_month": lag, "release_fraction": value})
        for start_month in range(1, 13):
            result = core.simulate_delivery_pulse(mu, start_month, True, HORIZON)
            for rr, rid in enumerate(reach_ids):
                delivery_metrics.append({"delivery_mu_month": mu, "start_month": start_month, "reach_id": int(rid), "gate": "q72_binary_water_gate", **core.tail_metrics(result["release"][:, rr], 1.0)})
            for scheme, w in schemes.items():
                curve = result["release"] @ w
                for lag, value in enumerate(curve):
                    delivery_curves.append({"delivery_mu_month": mu, "start_month": start_month, "gate": "q72_binary_water_gate", "weight_scheme": scheme, "lag_month": lag, "release_fraction": value})
    pd.DataFrame(delivery_curves).to_parquet(OUT / "delivery_impulse_survival_curves.parquet", index=False)
    pd.DataFrame(delivery_metrics).to_parquet(OUT / "delivery_impulse_metrics_by_reach.parquet", index=False)

    local_writer = WideWriter(OUT / "local_source_release_kernels.parquet")
    terminal_writer = WideWriter(OUT / "terminal_tree_source_to_stream_curves.parquet")
    basin_rows = []
    metric_rows = []
    superposition = []
    local_t95_rows = []
    _, _, terminal_map = core.topology_operators(reach_ids)
    terminal_ids = np.array(sorted(set(terminal_map.values())), dtype=int)
    terminal_masks = {
        int(terminal): np.array([terminal_map[int(r)] == int(terminal) for r in reach_ids])
        for terminal in terminal_ids
    }
    for model_index, model_id in enumerate(analysis_ids, start=1):
        structure, tau, mu = core.model_spec(model_id)
        for start_month in range(1, 13):
            unit = core.simulate_n_pulse(structure, tau, mu, start_month, np.ones(len(reach_ids)), HORIZON)
            local = unit["local_release"]
            if model_id in formal_ids:
                local_writer.write(curve_frame(model_id, start_month, local, reach_ids))
            for rr, rid in enumerate(reach_ids):
                tm = core.tail_metrics(local[:, rr], 1.0)
                local_t95_rows.append({"model_id": model_id, "start_month": start_month, "reach_id": int(rid), "scope": "local_source_release", **tm})
                metric_rows.append({"model_id": model_id, "start_month": start_month, "scope": "local_source_release", "scope_id": int(rid), "weight_scheme": "unit_reach", **tm})
            for scheme, w in schemes.items():
                basin, trees, terminals = aggregate_curves(local, reach_ids, w)
                for lag, value in enumerate(basin):
                    basin_rows.append({"model_id": model_id, "start_month": start_month, "weight_scheme": scheme, "lag_month": lag, "release_mass_fraction": value})
                metric_rows.append({"model_id": model_id, "start_month": start_month, "scope": "basin_source_to_stream", "scope_id": "PRB", "weight_scheme": scheme, **core.tail_metrics(basin, 1.0)})
                if model_id in formal_ids:
                    injected_by_tree = np.array([float(w[terminal_masks[int(terminal)]].sum()) for terminal in terminals])
                    terminal_writer.write(pd.DataFrame({
                        "model_id": model_id,
                        "start_month": start_month,
                        "terminal_tree_id": np.tile(terminals.astype(int), len(trees)),
                        "weight_scheme": scheme,
                        "lag_month": np.repeat(np.arange(len(trees), dtype=int), len(terminals)),
                        "release_mass": trees.reshape(-1),
                        "injected_mass": np.tile(injected_by_tree, len(trees)),
                    }))
                    for col, terminal in enumerate(terminals):
                        injected = injected_by_tree[col]
                        metric_rows.append({"model_id": model_id, "start_month": start_month, "scope": "terminal_tree_source_to_stream", "scope_id": int(terminal), "weight_scheme": scheme, **core.tail_metrics(trees[:, col], injected)})

            if model_id in ("S0_mu_036m", "S1_tau_012m_mu_240m") and start_month in (1, 7):
                w = schemes["n_source_weighted"]
                direct = core.simulate_n_pulse(structure, tau, mu, start_month, w, HORIZON)["local_release"]
                direct_basin, direct_trees, terminals = aggregate_curves(direct, reach_ids, np.ones(len(reach_ids)))
                super_basin, super_trees, _ = aggregate_curves(local, reach_ids, w)
                scale = 1.0
                max_basin = float(np.max(np.abs(direct_basin - super_basin)) / scale)
                max_tree = float(np.max(np.abs(direct_trees - super_trees)) / scale)
                superposition.append({"model_id": model_id, "start_month": start_month, "max_basin_relative_difference": max_basin, "max_terminal_relative_difference": max_tree, "pass": bool(max(max_basin, max_tree) <= 1e-12)})
        print(f"N pulse {model_index}/{len(analysis_ids)} {model_id}", flush=True)
    local_writer.close()
    terminal_writer.close()
    pd.DataFrame(basin_rows).to_parquet(OUT / "basin_source_to_stream_curves.parquet", index=False)
    metrics = pd.DataFrame(metric_rows)
    metrics["scope_id"] = metrics["scope_id"].astype(str)
    metrics.to_parquet(OUT / "source_to_stream_tail_metrics.parquet", index=False)
    super_df = pd.DataFrame(superposition)
    super_pass = len(super_df) == 4 and super_df["pass"].all()
    dump(REPORTS / "linear_superposition_audit.json", {"pass": bool(super_pass), "stop_code_on_failure": "STOP_KERNEL_SUPERPOSITION_NOT_VALID", "cases": superposition})
    if not super_pass:
        raise RuntimeError("STOP_KERNEL_SUPERPOSITION_NOT_VALID")

    # Matched Q72 hydraulic-response kernels.
    q_writer = WideWriter(OUT / "q72_hydraulic_response_kernels_by_reach.parquet")
    q_metric_rows = []
    q_basin_rows = []
    q_local_t95 = []
    for start_month in range(1, 13):
        q = core.simulate_q72_hydraulic_pulse(start_month, HORIZON)
        release = q["release"]
        q_writer.write(curve_frame("Q72_main_hydraulic_response", start_month, release, reach_ids).drop(columns="pulse_input_semantics"))
        for rr, rid in enumerate(reach_ids):
            tm = core.tail_metrics(release[:, rr], 1.0)
            q_local_t95.append({"start_month": start_month, "reach_id": int(rid), **tm})
            q_metric_rows.append({"start_month": start_month, "scope": "local_hydraulic_response", "scope_id": int(rid), "weight_scheme": "unit_reach", **tm})
        for scheme, w in schemes.items():
            basin, trees, terminals = aggregate_curves(release, reach_ids, w)
            for lag, value in enumerate(basin):
                q_basin_rows.append({"start_month": start_month, "weight_scheme": scheme, "lag_month": lag, "release_fraction": value})
            q_metric_rows.append({"start_month": start_month, "scope": "basin_hydraulic_response", "scope_id": "PRB", "weight_scheme": scheme, **core.tail_metrics(basin, 1.0)})
    q_writer.close()
    q_metrics = pd.DataFrame(q_metric_rows)
    q_metrics["scope_id"] = q_metrics["scope_id"].astype(str)
    q_metrics.to_parquet(OUT / "q72_hydraulic_response_metrics.parquet", index=False)
    pd.DataFrame(q_basin_rows).to_parquet(OUT / "q72_hydraulic_response_basin_curves.parquet", index=False)

    nlocal = pd.DataFrame(local_t95_rows)
    qlocal = pd.DataFrame(q_local_t95)
    matched = nlocal.loc[nlocal.model_id.isin(formal_ids), ["model_id", "start_month", "reach_id", "T95_year", "T95_status"]].merge(
        qlocal[["start_month", "reach_id", "T95_year", "T95_status"]], on=["start_month", "reach_id"], suffixes=("_n", "_q"), validate="many_to_one"
    )
    matched["n_to_q_T95_ratio"] = matched.T95_year_n / matched.T95_year_q
    matched["n_T95_gt_q_T95"] = matched.T95_year_n > matched.T95_year_q
    matched.to_parquet(OUT / "matched_n_vs_hydraulic_metrics.parquet", index=False)

    mass_error = max(
        abs(metrics.loc[metrics.scope.eq("local_source_release"), "released_at_horizon"].max() - 1.0),
        abs(q_metrics.loc[q_metrics.scope.eq("local_hydraulic_response"), "released_at_horizon"].max() - 1.0),
    )
    dump(REPORTS / "pulse_mass_balance_audit.json", {
        "pulse_b_unit": "1 kg N monthly post-ledger mass per reach",
        "cdf_denominator": "initial injected mass or water; never horizon-released mass",
        "minimum_n_cdf_horizon": float(metrics.loc[metrics.scope.eq("local_source_release"), "cdf_at_horizon"].min()),
        "minimum_q_cdf_horizon": float(q_metrics.loc[q_metrics.scope.eq("local_hydraulic_response"), "cdf_at_horizon"].min()),
        "right_censor_supported": True,
    })

    end_hash = {str(p): sha256(p) for p in protected}
    if start_hash != end_hash:
        raise RuntimeError("protected input changed")
    completion = {
        "scenario_id": "20260817_2", "pass": True, "formal_models": len(formal_ids), "analysis_models": len(analysis_ids),
        "start_months": 12, "horizon_months": HORIZON, "reach_count": len(reach_ids),
        "terminal_tree_count": int(pd.read_parquet(OUT / "terminal_tree_source_to_stream_curves.parquet").terminal_tree_id.nunique()),
        "parent_reproduction_pass": True, "linear_superposition_pass": bool(super_pass),
        "pulse_b_post_ledger_monthly_mass_explicit": True, "eta_used": False, "locked_2022_used": False,
    }
    dump(REPORTS / "completion_audit.json", completion)
    dump(ROOT / "stage_lock.json", {"status": "complete", "scenario_id": "20260817_2", "completion_sha256": sha256(REPORTS / "completion_audit.json")})
    print(json.dumps(completion, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
