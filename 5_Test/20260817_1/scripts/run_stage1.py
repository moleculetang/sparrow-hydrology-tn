from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(r"E:\SPARROW\5_Test\20260817_1")
OUT = ROOT / "outputs"
REPORTS = ROOT / "reports"
S16_1 = Path(r"E:\SPARROW\5_Test\20260816_1")
S16_2 = Path(r"E:\SPARROW\5_Test\20260816_2")
S16_3 = Path(r"E:\SPARROW\5_Test\20260816_3")
S16_4 = Path(r"E:\SPARROW\5_Test\20260816_4")
S16_5 = Path(r"E:\SPARROW\5_Test\20260816_5")
OOF = S16_4 / "outputs" / "candidate_oof_predictions_2018_2021.parquet"
STATE = S16_4 / "outputs" / "candidate_state_summary_2010_2018.parquet"
ETA = S16_4 / "outputs" / "candidate_fold_eta_parameters.parquet"
ADJ = S16_5 / "reports" / "progressive_candidate_adjudication.csv"
SOIL = S16_1 / "outputs" / "soil_observation_operator_by_reach.parquet"
SOIL_S1 = S16_2 / "reports" / "soil_tau_admissibility.csv"
HYDRO = S16_3 / "reports" / "delivery_mu_hydrogeo_plausibility.csv"
PARENT_METRICS = S16_4 / "reports" / "candidate_oof_metrics.csv"
PARENT_SCRIPT = S16_4 / "scripts" / "run_stage4.py"
N_BOOT = 10000
SEED = 20260816


def require_runtime() -> None:
    if Path(sys.prefix).name.lower() != "sparrow":
        raise RuntimeError(f"sparrow environment required: {sys.prefix}")
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


def top_overlap(x: np.ndarray, y: np.ndarray) -> float:
    xh = x >= np.nanquantile(x, 0.75)
    yh = y >= np.nanquantile(y, 0.75)
    union = np.sum(xh | yh)
    return float(np.sum(xh & yh) / union) if union else np.nan


def bootstrap_spearman(frame: pd.DataFrame) -> tuple[float, float]:
    groups = [g.copy() for _, g in frame.groupby("terminal_tree_id")]
    rng = np.random.default_rng(SEED)
    values = np.full(N_BOOT, np.nan)
    for i in range(N_BOOT):
        sample = pd.concat([groups[j] for j in rng.integers(0, len(groups), len(groups))], ignore_index=True)
        if sample.model_density_kg_n_km2.nunique() > 1 and sample.enrichment_g_kg.nunique() > 1:
            values[i] = spearmanr(sample.model_density_kg_n_km2, sample.enrichment_g_kg).statistic
    lo, hi = np.nanpercentile(values, [2.5, 97.5])
    return float(lo), float(hi)


def s0_soil_metric(state: pd.DataFrame, soil: pd.DataFrame) -> dict[str, object]:
    s0 = state.loc[state.model_id.eq("S0_mu_000m")].merge(soil, on="reach_id", validate="one_to_one")
    crop = s0.cropland_soil_tn_stock_ceiling_kg_n_min.to_numpy(float)
    whole = s0.whole_reach_soil_tn_stock_ceiling_kg_n_min.to_numpy(float)
    ceiling = np.where(np.isfinite(crop) & (crop > 0), crop, whole)
    ratio = s0.source_legacy_state_max_kg_n.to_numpy(float) / ceiling
    eligible = s0.loc[s0.soil_enrichment_eligible].copy()
    eligible["model_density_kg_n_km2"] = eligible.source_legacy_state_mean_kg_n / eligible.cropland_area_km2_median.clip(lower=1e-9)
    eligible["enrichment_g_kg"] = eligible.soil_tn_cropland_enrichment_g_kg_median
    rho = float(spearmanr(eligible.model_density_kg_n_km2, eligible.enrichment_g_kg).statistic)
    overlap = top_overlap(eligible.model_density_kg_n_km2.to_numpy(float), eligible.enrichment_g_kg.to_numpy(float))
    lo, hi = bootstrap_spearman(eligible)
    contradicted = bool(hi < 0.0 and overlap < 0.10)
    return {
        "source_structure": "S0",
        "soil_tau_month": None,
        "physical_stock_ceiling_pass": bool(np.nanmax(ratio) <= 1 + 1e-9),
        "maximum_source_to_soil_stock_ratio": float(np.nanmax(ratio)),
        "n_reaches_exceeding_ceiling": int(np.sum(ratio > 1 + 1e-9)),
        "soil_enrichment_spearman": rho,
        "soil_enrichment_tree_bootstrap_ci_lower": lo,
        "soil_enrichment_tree_bootstrap_ci_upper": hi,
        "soil_enrichment_top_quartile_overlap": overlap,
        "soil_pattern_contradicted": contradicted,
        "soil_admissible": bool(np.nanmax(ratio) <= 1 + 1e-9 and not contradicted),
    }


def main() -> None:
    require_runtime()
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    protected = [OOF, STATE, ETA, ADJ, SOIL, SOIL_S1, HYDRO, PARENT_METRICS, PARENT_SCRIPT]
    start = {str(p): sha256(p) for p in protected}
    dump(ROOT / "upstream_manifest.json", start)

    oof = pd.read_parquet(OOF)
    years = sorted(map(int, oof.year.unique()))
    sizes = oof.groupby("model_id").size()
    if years != [2018, 2019, 2020, 2021] or sizes.min() != 4097 or sizes.max() != 4097:
        raise RuntimeError("frozen OOF time/key contract failed")
    state = pd.read_parquet(STATE)
    soil = pd.read_parquet(SOIL)
    parent_adj = pd.read_csv(ADJ)
    eta = pd.read_parquet(ETA)

    s0_metric = s0_soil_metric(state, soil)
    s1_metric = pd.read_csv(SOIL_S1)
    soil_rows = [s0_metric]
    for row in s1_metric.to_dict("records"):
        soil_rows.append({"source_structure": "S1", **row})
    soil_table = pd.DataFrame(soil_rows)
    soil_table.to_parquet(OUT / "soil_state_constraint_metrics.parquet", index=False)
    s1_12_soil = bool(s1_metric.loc[s1_metric.soil_tau_month.eq(12), "soil_admissible"].iloc[0])

    eta_summary = eta.groupby("model_id", as_index=False).agg(
        eta_quick_min=("eta_quick", "min"), eta_quick_max=("eta_quick", "max"),
        eta_gw_min=("eta_gw", "min"), eta_gw_max=("eta_gw", "max"),
        delivery_efficiency_confounded=("delivery_efficiency_confounded", "max"),
    )
    adjud = parent_adj.merge(eta_summary, on="model_id", how="left", suffixes=("", "_eta"), validate="one_to_one")
    adjud["soil_gate_applicable"] = adjud.source_structure.eq("S1") | adjud.source_structure.eq("S0")
    adjud["soil_gate_status"] = np.select(
        [adjud.source_structure.eq("M0"), adjud.source_structure.eq("S0"), adjud.source_structure.eq("S1") & adjud.soil_tau_month.eq(12)],
        ["not_applicable_no_persistent_source_store", "pass" if s0_metric["soil_admissible"] else "fail", "pass" if s1_12_soil else "fail"],
        default="not_in_reconciled_source_set",
    )
    adjud["reconciled_soil_pass"] = np.where(
        adjud.source_structure.eq("M0"), True,
        np.where(adjud.source_structure.eq("S0"), s0_metric["soil_admissible"],
                 adjud.source_structure.eq("S1") & adjud.soil_tau_month.eq(12) & s1_12_soil),
    )
    # Deliberately excludes eta boundary confounding: it is a diagnostic, never a hard gate.
    adjud["eta_boundary_confounding_is_hard_gate"] = False
    adjud["reconciled_admissible_without_eta_gate"] = (
        adjud.engineering_pass.astype(bool)
        & adjud.reconciled_soil_pass.astype(bool)
        & adjud.hydrogeo_pass.astype(bool)
        & adjud.river_noninferior.astype(bool)
    )
    adjud["reconciled_admissible"] = adjud.reconciled_admissible_without_eta_gate
    adjud.to_parquet(OUT / "structural_adjudication_registry.parquet", index=False)

    mus = [12, 36, 60, 96, 144, 240]
    formal_mask = (
        adjud.delivery_mu_month.isin(mus)
        & (adjud.source_structure.eq("S0") | (adjud.source_structure.eq("S1") & adjud.soil_tau_month.eq(12)))
    )
    formal = adjud.loc[formal_mask & adjud.reconciled_admissible].copy()
    formal.to_parquet(OUT / "formal_12model_structural_ensemble.parquet", index=False)
    formal[["model_id", "source_structure", "soil_tau_month", "delivery_mu_month"]].to_parquet(OUT / "admissible_tau_mu_region.parquet", index=False)

    controls = ["M0_mu_000m", "S0_mu_000m", "S1_tau_012m_mu_000m", "S1_tau_480m_mu_000m"]
    analysis_ids = formal.model_id.tolist() + controls
    analysis = adjud.loc[adjud.model_id.isin(analysis_ids)].copy()
    analysis.to_parquet(OUT / "analysis_model_registry_16.parquet", index=False)
    analysis_oof = oof.loc[oof.model_id.isin(analysis_ids)].copy()
    analysis_oof.to_parquet(OUT / "analysis_model_oof_predictions.parquet", index=False)
    eta.loc[eta.model_id.isin(analysis_ids)].to_parquet(OUT / "candidate_fold_nuisance_parameters.parquet", index=False)

    m0_adm = adjud.loc[adjud.source_structure.eq("M0") & adjud.reconciled_admissible]
    s0_adm = formal.loc[formal.source_structure.eq("S0")]
    s1_adm = formal.loc[formal.source_structure.eq("S1")]
    persistent_any = len(s0_adm) + len(s1_adm) > 0
    if m0_adm.empty and persistent_any:
        source_persistence = "required"
    elif not m0_adm.empty:
        source_persistence = "not_required_by_registered_gates"
    else:
        source_persistence = "unresolved_no_admissible_structure"
    if len(s0_adm) and len(s1_adm):
        source_structure = "S0_vs_S1_12_nonidentifying"
    elif len(s0_adm):
        source_structure = "S0_supported"
    elif len(s1_adm):
        source_structure = "S1_12_supported"
    else:
        source_structure = "unresolved"

    relevant = adjud.loc[
        adjud.source_structure.isin(["M0", "S0"])
        | (adjud.source_structure.eq("S1") & adjud.soil_tau_month.eq(12))
    ]
    t0_river = relevant.loc[relevant.delivery_mu_month.eq(0) & relevant.river_noninferior]
    pos_improved = relevant.loc[relevant.delivery_mu_month.gt(0) & relevant.predictively_improved]
    river_status = "supportive_but_not_exclusive" if len(t0_river) and len(pos_improved) else (
        "positive_memory_exclusive_in_river_evidence" if t0_river.empty and len(pos_improved) else "non_identifying"
    )
    t0_full = relevant.loc[relevant.delivery_mu_month.eq(0) & relevant.reconciled_admissible]
    pos_full = relevant.loc[relevant.delivery_mu_month.gt(0) & relevant.reconciled_admissible]
    multi_status = "positive_memory_required_under_registered_gates" if t0_full.empty and len(pos_full) else (
        "positive_and_zero_memory_both_admissible" if len(t0_full) and len(pos_full) else "unresolved"
    )

    contract = {
        "transform_id": "natural_log1p", "formula": "ln(1 + TN_mg_L)", "log_base": "e",
        "offset": 1.0, "input_unit": "mg/L", "prediction_floor": 0.0,
        "parent_metric_name": "rmse_log1p", "parent_script_sha256": sha256(PARENT_SCRIPT),
    }
    dump(REPORTS / "performance_metric_contract.json", contract)
    parent_metrics = pd.read_csv(PARENT_METRICS)
    reproduced = []
    for model_id, group in oof.groupby("model_id"):
        value = float(np.sqrt(np.mean((np.log1p(group.pred_tn_mg_l) - np.log1p(group.tn_mg_l)) ** 2)))
        expected = float(parent_metrics.loc[parent_metrics.model_id.eq(model_id), "rmse_log1p"].iloc[0])
        reproduced.append(abs(value - expected))
    dump(REPORTS / "performance_metric_parent_reproduction.json", {"max_abs_difference": max(reproduced), "pass": max(reproduced) <= 1e-12})

    decision = {
        "scenario_id": "20260817_1",
        "oof_evaluation_years": years,
        "development_support_period": [2016, 2021],
        "retrospective_locked_year": 2022,
        "formal_model_ids": formal.model_id.tolist(),
        "formal_model_count": len(formal),
        "m0_model_count_adjudicated": int(adjud.source_structure.eq("M0").sum()),
        "m0_admissible_model_ids": m0_adm.model_id.tolist(),
        "source_persistence_status": source_persistence,
        "source_structure_status": source_structure,
        "river_evidence_for_positive_delivery_memory": river_status,
        "multievidence_delivery_memory_status": multi_status,
        "eta_boundary_confounding_role": "diagnostic_only_not_hard_gate",
        "canonical_source_structure": "S0" if source_structure == "S0_vs_S1_12_nonidentifying" else None,
        "display_representative_model": "S0_mu_036m" if "S0_mu_036m" in formal.model_id.values else None,
        "locked_2022_used": False,
    }
    dump(REPORTS / "structural_reconciliation_decision.json", decision)
    dump(REPORTS / "formal_ensemble_manifest.json", {"model_ids": formal.model_id.tolist(), "n_models": len(formal), "controls": controls})

    end = {str(p): sha256(p) for p in protected}
    if start != end:
        raise RuntimeError("protected parent changed")
    if len(formal) != 12 or len(analysis_ids) != 16 or len(analysis_oof) != 16 * 4097:
        raise RuntimeError("expected reconciled ensemble contract failed")
    completion = {
        "scenario_id": "20260817_1", "pass": True, "formal_models": len(formal),
        "analysis_models": len(analysis_ids), "analysis_oof_rows": len(analysis_oof),
        "m0_candidates_adjudicated": int(adjud.source_structure.eq("M0").sum()),
        "eta_boundary_not_hard_gate": bool((~adjud.eta_boundary_confounding_is_hard_gate).all()),
        "parent_hashes_unchanged": start == end,
    }
    dump(REPORTS / "completion_audit.json", completion)
    dump(ROOT / "stage_lock.json", {"status": "complete", "scenario_id": "20260817_1", "completion_sha256": sha256(REPORTS / "completion_audit.json")})
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
