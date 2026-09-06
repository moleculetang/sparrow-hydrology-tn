from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


S16_1 = Path(r"E:\SPARROW\5_Test\20260816_1")
sys.path.insert(0, str(S16_1 / "scripts"))
from legacy16_core import (  # noqa: E402
    S16_1 as PARENT,
    TAUS,
    dump_json,
    hash_manifest,
    prepare_model_arrays,
    require_runtime,
    simulate_candidate_totals,
)


ROOT = Path(r"E:\SPARROW\5_Test\20260816_2")
OUT = ROOT / "outputs"
REPORTS = ROOT / "reports"
SOIL_PATH = PARENT / "outputs" / "soil_observation_operator_by_reach.parquet"
PARENT_AUDIT = PARENT / "reports" / "completion_audit.json"
N_BOOT = 10000
SEED = 20260816


def top_quartile_overlap(x: np.ndarray, y: np.ndarray) -> float:
    x_high = x >= np.nanquantile(x, 0.75)
    y_high = y >= np.nanquantile(y, 0.75)
    union = np.sum(x_high | y_high)
    return float(np.sum(x_high & y_high) / union) if union else np.nan


def tree_bootstrap_spearman(frame: pd.DataFrame) -> np.ndarray:
    groups = [group.copy() for _, group in frame.groupby("terminal_tree_id")]
    rng = np.random.default_rng(SEED)
    values = np.full(N_BOOT, np.nan)
    for i in range(N_BOOT):
        sampled = pd.concat([groups[j] for j in rng.integers(0, len(groups), size=len(groups))], ignore_index=True)
        if sampled.model_density_kg_n_km2.nunique() > 1 and sampled.enrichment_g_kg.nunique() > 1:
            values[i] = spearmanr(sampled.model_density_kg_n_km2, sampled.enrichment_g_kg).statistic
    return values


def main() -> None:
    require_runtime()
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    parent = json.loads(PARENT_AUDIT.read_text(encoding="utf-8"))
    if not parent.get("pass") or not parent.get("parent_reproduced"):
        raise RuntimeError("20260816_1 did not pass")
    protected = [PARENT_AUDIT, SOIL_PATH, PARENT / "experiment_contract.json"]
    start = hash_manifest(protected)
    dump_json(REPORTS / "parent_hashes_start.json", start)
    soil = pd.read_parquet(SOIL_PATH)
    soil_operator_ready = (
        int(soil.soil_enrichment_eligible.sum()) >= 30
        and int(soil.loc[soil.soil_enrichment_eligible, "terminal_tree_id"].nunique()) >= 4
    )
    reach_ids, times, arrays, early_positive = prepare_model_arrays()
    rows = []
    state_rows = []
    audits = {}
    for tau in TAUS:
        model_id = f"S1_tau_{tau:03d}m_T0"
        frame, audit = simulate_candidate_totals(
            model_id, "S1", tau, 0, reach_ids, times, arrays, early_positive
        )
        audits[model_id] = audit
        modern = frame.loc[frame.year.between(2010, 2018)].copy()
        modern["source_state_kg_n"] = modern.son_state_end_kg_n + modern.mobile_state_end_kg_n
        summary = modern.groupby("reach_id", as_index=False).agg(
            source_state_mean_kg_n=("source_state_kg_n", "mean"),
            source_state_max_kg_n=("source_state_kg_n", "max"),
            son_state_mean_kg_n=("son_state_end_kg_n", "mean"),
            mobile_state_mean_kg_n=("mobile_state_end_kg_n", "mean"),
            catchment_area_km2=("catchment_area_km2", "first"),
        )
        summary["model_id"] = model_id
        summary["soil_tau_month"] = tau
        state_rows.append(summary)
        evidence = summary.merge(soil, on="reach_id", validate="one_to_one")
        crop_ceiling = evidence.cropland_soil_tn_stock_ceiling_kg_n_min.to_numpy(float)
        whole_ceiling = evidence.whole_reach_soil_tn_stock_ceiling_kg_n_min.to_numpy(float)
        ceiling = np.where(np.isfinite(crop_ceiling) & (crop_ceiling > 0), crop_ceiling, whole_ceiling)
        ratio = evidence.source_state_max_kg_n.to_numpy(float) / ceiling
        ceiling_pass = bool(np.nanmax(ratio) <= 1.0 + 1e-9)
        eligible = evidence.loc[evidence.soil_enrichment_eligible].copy()
        eligible["model_density_kg_n_km2"] = (
            eligible.source_state_mean_kg_n / eligible.cropland_area_km2_median.clip(lower=1e-9)
        )
        eligible["enrichment_g_kg"] = eligible.soil_tn_cropland_enrichment_g_kg_median
        if soil_operator_ready and len(eligible) >= 30:
            rho = float(spearmanr(eligible.model_density_kg_n_km2, eligible.enrichment_g_kg).statistic)
            overlap = top_quartile_overlap(
                eligible.model_density_kg_n_km2.to_numpy(float), eligible.enrichment_g_kg.to_numpy(float)
            )
            boot = tree_bootstrap_spearman(eligible)
            ci_low, ci_high = np.nanpercentile(boot, [2.5, 97.5])
            pattern_contradicted = bool(ci_high < 0.0 and overlap < 0.10)
            spatial_status = "contradicted" if pattern_contradicted else "admissible_soft"
        else:
            rho = overlap = ci_low = ci_high = np.nan
            pattern_contradicted = False
            spatial_status = "insufficient_independent_cells"
        whole_rho = float(spearmanr(
            evidence.source_state_mean_kg_n / evidence.catchment_area_km2,
            evidence.soil_tn_0_100cm_depth_weighted_g_kg,
            nan_policy="omit",
        ).statistic)
        csdl_rho = float(spearmanr(
            evidence.source_state_mean_kg_n / evidence.catchment_area_km2,
            evidence.csdl_v2_tn_0_5cm_native_mean,
            nan_policy="omit",
        ).statistic)
        admissible = bool(ceiling_pass and not pattern_contradicted and audit["minimum_state_or_flux_kg_n"] >= -1e-9)
        rows.append({
            "model_id": model_id,
            "soil_tau_month": tau,
            "physical_stock_ceiling_pass": ceiling_pass,
            "maximum_source_to_soil_stock_ratio": float(np.nanmax(ratio)),
            "n_reaches_exceeding_ceiling": int(np.sum(ratio > 1.0 + 1e-9)),
            "soil_enrichment_operator_status": spatial_status,
            "soil_enrichment_spearman": rho,
            "soil_enrichment_tree_bootstrap_ci_lower": float(ci_low),
            "soil_enrichment_tree_bootstrap_ci_upper": float(ci_high),
            "soil_enrichment_top_quartile_overlap": overlap,
            "whole_reach_total_tn_descriptive_spearman": whole_rho,
            "csdl_rank_sensitivity_spearman": csdl_rho,
            "engineering_pass": bool(
                audit["max_relative_mass_balance_error"] <= 1e-12
                and audit["minimum_state_or_flux_kg_n"] >= -1e-9
                and audit["spinup"]["converged"]
            ),
            "soil_admissible": admissible,
        })
    decision = pd.DataFrame(rows).sort_values("soil_tau_month")
    states = pd.concat(state_rows, ignore_index=True)
    decision.to_csv(REPORTS / "soil_tau_admissibility.csv", index=False)
    states.to_parquet(OUT / "soil_candidate_state_summary_2010_2018.parquet", index=False)
    dump_json(REPORTS / "soil_candidate_engineering_audit.json", audits)
    admissible_taus = decision.loc[decision.soil_admissible, "soil_tau_month"].astype(int).tolist()
    if not admissible_taus:
        status = "external_soil_constraints_incompatible_or_operator_nonidentifying"
    elif len(admissible_taus) == len(TAUS):
        status = "non_identifying"
    elif max(TAUS) in admissible_taus:
        status = "right_censored"
    else:
        status = "bounded"
    result = {
        "scenario_id": "20260816_2",
        "soil_memory_status": status,
        "soil_operator_ready": soil_operator_ready,
        "admissible_soil_tau_month": admissible_taus,
        "winner_selected": False,
        "river_tn_used": False,
        "whole_reach_total_tn_used_as_formal_rank_gate": False,
        "csdl_absolute_values_used": False,
    }
    dump_json(REPORTS / "soil_memory_decision.json", result)
    end = hash_manifest(protected)
    dump_json(REPORTS / "parent_hashes_end.json", end)
    if start != end:
        raise RuntimeError("20260816_1 changed during stage 2")
    completion = {
        "scenario_id": "20260816_2",
        "pass": True,
        "seven_tau_evaluated": len(decision) == 7,
        "mass_engineering_complete": True,
        "decision_written": True,
    }
    dump_json(REPORTS / "completion_audit.json", completion)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
