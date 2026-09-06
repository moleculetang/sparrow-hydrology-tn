from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW\5_Test\20260817_5")
OUT = ROOT / "outputs"
REPORTS = ROOT / "reports"
S17_1 = Path(r"E:\SPARROW\5_Test\20260817_1")
S17_2 = Path(r"E:\SPARROW\5_Test\20260817_2")
S17_3 = Path(r"E:\SPARROW\5_Test\20260817_3")
S17_4 = Path(r"E:\SPARROW\5_Test\20260817_4")


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


def weighted_age_periods(age: pd.DataFrame) -> pd.DataFrame:
    fraction_columns = [name for name in age.columns if name.endswith("_fraction")]
    rows = []
    for (model_id, semantics), block0 in age.groupby(["model_id", "age_semantics"], sort=True):
        for period, block in (
            ("development_2016_2021", block0.loc[block0.year.between(2016, 2021)]),
            ("retrospective_2022", block0.loc[block0.year.eq(2022)]),
        ):
            total = float(block.total_mass_kg_n.sum())
            item: dict[str, object] = {"model_id": model_id, "age_semantics": semantics, "period": period, "total_mass_kg_n": total}
            for column in fraction_columns:
                item[column] = float((block[column] * block.total_mass_kg_n).sum() / total) if total > 0 else np.nan
            post_mass = block.total_mass_kg_n * (1.0 - block.pre1961_fraction)
            item["post1961_mean_age_month"] = float((block.post1961_mean_age_month * post_mass).sum() / post_mass.sum()) if post_mass.sum() > 0 else np.nan
            rows.append(item)
    return pd.DataFrame(rows)


def main() -> None:
    require_runtime()
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    for stage in range(1, 5):
        completion = Path(rf"E:\SPARROW\5_Test\20260817_{stage}\reports\completion_audit.json")
        if not json.loads(completion.read_text(encoding="utf-8"))["pass"]:
            raise RuntimeError(f"stage {stage} incomplete")
    protected = [
        S17_1 / "reports" / "structural_reconciliation_decision.json",
        S17_1 / "outputs" / "formal_12model_structural_ensemble.parquet",
        S17_2 / "outputs" / "source_to_stream_tail_metrics.parquet",
        S17_2 / "outputs" / "q72_hydraulic_response_metrics.parquet",
        S17_2 / "outputs" / "matched_n_vs_hydraulic_metrics.parquet",
        S17_3 / "outputs" / "terminal_historical_n_age_summary_2016_2022.parquet",
        S17_4 / "outputs" / "tn_performance_metrics_16models.parquet",
        S17_4 / "outputs" / "primary_metric_differences_vs_old_incumbent.parquet",
    ]
    start_hash = {str(path): sha256(path) for path in protected}
    dump(ROOT / "upstream_manifest.json", start_hash)
    decision1 = json.loads((S17_1 / "reports" / "structural_reconciliation_decision.json").read_text(encoding="utf-8"))
    formal = pd.read_parquet(S17_1 / "outputs" / "formal_12model_structural_ensemble.parquet")
    formal_ids = formal.model_id.astype(str).tolist()
    n_tail = pd.read_parquet(S17_2 / "outputs" / "source_to_stream_tail_metrics.parquet")
    q_tail = pd.read_parquet(S17_2 / "outputs" / "q72_hydraulic_response_metrics.parquet")
    matched = pd.read_parquet(S17_2 / "outputs" / "matched_n_vs_hydraulic_metrics.parquet")
    n_basin = n_tail.loc[n_tail.model_id.isin(formal_ids) & n_tail.scope.eq("basin_source_to_stream")].copy()
    q_basin = q_tail.loc[q_tail.scope.eq("basin_hydraulic_response")].copy()
    comparison = n_basin.merge(
        q_basin[["start_month", "weight_scheme", "T50_year", "T90_year", "T95_year", "T99_year"]],
        on=["start_month", "weight_scheme"],
        suffixes=("_n", "_q"),
        validate="many_to_one",
    )
    comparison["T95_n_to_q_ratio"] = comparison.T95_year_n / comparison.T95_year_q
    comparison["N_T95_gt_Q_T95"] = comparison.T95_year_n > comparison.T95_year_q
    comparison["spatial_weight_match"] = True
    comparison.to_parquet(OUT / "matched_basin_N_vs_Q_tail_metrics.parquet", index=False)
    basin_summary = comparison.groupby(["model_id", "weight_scheme"], as_index=False).agg(
        n_start_months=("start_month", "size"),
        N_T95_year_median=("T95_year_n", "median"),
        N_T95_year_p10=("T95_year_n", lambda x: x.quantile(0.10)),
        N_T95_year_p90=("T95_year_n", lambda x: x.quantile(0.90)),
        Q_T95_year_median=("T95_year_q", "median"),
        median_N_to_Q_T95_ratio=("T95_n_to_q_ratio", "median"),
        fraction_start_months_N_T95_gt_Q_T95=("N_T95_gt_Q_T95", "mean"),
    )
    reach_summary = matched.loc[matched.model_id.isin(formal_ids)].groupby("model_id", as_index=False).agg(
        n_reach_month_cases=("n_T95_gt_q_T95", "size"),
        fraction_reaches_startmonths_N_T95_gt_Q_T95=("n_T95_gt_q_T95", "mean"),
        median_N_to_Q_T95_ratio=("n_to_q_T95_ratio", "median"),
        p10_N_to_Q_T95_ratio=("n_to_q_T95_ratio", lambda x: x.quantile(0.10)),
        p90_N_to_Q_T95_ratio=("n_to_q_T95_ratio", lambda x: x.quantile(0.90)),
    )
    basin_summary.to_parquet(OUT / "basin_tail_summary_formal_ensemble.parquet", index=False)
    reach_summary.to_parquet(OUT / "reach_matched_tail_summary_formal_ensemble.parquet", index=False)

    age = pd.read_parquet(S17_3 / "outputs" / "terminal_historical_n_age_summary_2016_2022.parquet")
    age_period = weighted_age_periods(age)
    age_period.to_parquet(OUT / "historical_output_age_period_summary.parquet", index=False)
    performance = pd.read_parquet(S17_4 / "outputs" / "tn_performance_metrics_16models.parquet")
    primary = pd.read_parquet(S17_4 / "outputs" / "primary_metric_differences_vs_old_incumbent.parquet")
    performance.loc[performance.evaluation_period.isin(["OOF_2018_2021", "locked_2022"])].to_parquet(OUT / "integrated_TN_performance_summary.parquet", index=False)

    display_model = decision1["display_representative_model"]
    display_tail = basin_summary.loc[(basin_summary.model_id.eq(display_model)) & basin_summary.weight_scheme.eq("n_source_weighted")].iloc[0]
    formal_source = basin_summary.loc[basin_summary.weight_scheme.eq("n_source_weighted")]
    q95 = float(formal_source.Q_T95_year_median.iloc[0])
    display_age = age_period.loc[(age_period.model_id.eq(display_model)) & age_period.age_semantics.eq("eta_weighted_predicted_n_age") & age_period.period.eq("development_2016_2021")].iloc[0]
    display_perf = primary.loc[(primary.model_id.eq(display_model)) & primary.evaluation_period.eq("OOF_2018_2021")].iloc[0]
    locked_display_perf = primary.loc[(primary.model_id.eq(display_model)) & primary.evaluation_period.eq("locked_2022")].iloc[0]
    scientific = {
        "scenario_id": "20260817_5",
        "source_persistence_status": decision1["source_persistence_status"],
        "source_persistence_basis": "all seven frozen M0 candidates failed the registered admissibility algorithm while S0/S1 candidates survived",
        "source_structure_status": decision1["source_structure_status"],
        "river_evidence_for_positive_delivery_memory": decision1["river_evidence_for_positive_delivery_memory"],
        "multievidence_delivery_memory_status": decision1["multievidence_delivery_memory_status"],
        "formal_structural_ensemble_model_count": len(formal_ids),
        "formal_delivery_mu_month": sorted(formal.delivery_mu_month.astype(int).unique().tolist()),
        "effective_delivery_time_point_identified": False,
        "long_tail_uncertainty_statement": "retain the full registered formal ensemble; do not select a unique tau or mu from these data",
        "N_source_weighted_basin_T95_year_median_range_across_formal_models": [float(formal_source.N_T95_year_median.min()), float(formal_source.N_T95_year_median.max())],
        "Q72_hydraulic_response_same_N_source_weight_T95_year_median": q95,
        "display_representative_model": display_model,
        "display_representative_is_point_identification": False,
        "display_representative_N_source_weighted_T95_year_median": float(display_tail.N_T95_year_median),
        "display_representative_Q_same_weight_T95_year_median": float(display_tail.Q_T95_year_median),
        "display_representative_development_eta_weighted_post1961_mean_age_month": float(display_age.post1961_mean_age_month),
        "display_representative_development_fraction_age_gt10yr": float(display_age.age_gt10_to20yr_fraction + display_age.age_gt20_to50yr_fraction + display_age.age_gt50yr_fraction),
        "display_representative_OOF_delta_station_macro_log_RMSE_vs_old_incumbent": float(display_perf.delta_station_macro_rmse_log1p_vs_old_incumbent),
        "display_representative_locked_2022_delta_station_macro_log_RMSE_vs_old_incumbent": float(locked_display_perf.delta_station_macro_rmse_log1p_vs_old_incumbent),
        "locked_2022_changed_selection": False,
        "locked_2022_role": "retrospective_only",
        "eta_boundary_confounding_role": "diagnostic_only_not_hard_gate",
        "groundwater_age_claim": "not identified; all reported ages are N input-month mass cohort ages or model pulse tails, not water ages",
        "topology_time_semantics": "R0 is same-month mass routing, so downstream topology aggregates source kernels but adds no reach-by-reach temporal residence time",
    }
    dump(REPORTS / "integrated_scientific_decision.json", scientific)
    dump(REPORTS / "plain_language_interpretation.json", {
        "hydrology": "Q72 supplies monthly water partition and a comparatively short hydraulic response; its topology does not add repeated temporal memory under R0.",
        "soil_source": "At least one cross-month source N store is required under the registered gates, but S0 versus 12-month S1 is unresolved.",
        "subsurface_delivery": "Positive multi-month delivery memory is required only after combining river TN with the registered hydrogeologic gate; river TN alone is supportive but not exclusive.",
        "long_tail": "The basin N T95 is not a single identified constant. Across the formal ensemble it spans a broad range and is consistently evaluated against Q72 using identical spatial weights.",
        "historical_age": "The age composition of actual output differs from pulse T95 because recent inputs can dominate mass even when the response kernel has a long tail.",
    })
    end_hash = {str(path): sha256(path) for path in protected}
    if start_hash != end_hash:
        raise RuntimeError("protected input changed")
    completion = {
        "scenario_id": "20260817_5",
        "pass": True,
        "formal_models": len(formal_ids),
        "N_Q_same_weight_comparisons": len(comparison),
        "age_semantics": sorted(age_period.age_semantics.unique().tolist()),
        "locked_2022_changed_selection": False,
        "point_identification_claim": False,
    }
    dump(REPORTS / "completion_audit.json", completion)
    dump(ROOT / "stage_lock.json", {"status": "complete", "scenario_id": "20260817_5", "completion_sha256": sha256(REPORTS / "completion_audit.json")})
    print(json.dumps({"completion": completion, "scientific_decision": scientific}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
