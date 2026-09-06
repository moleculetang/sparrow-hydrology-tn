from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


S16_1 = Path(r"E:\SPARROW\5_Test\20260816_1")
sys.path.insert(0, str(S16_1 / "scripts"))
from legacy16_core import dump_json, hash_manifest, require_runtime  # noqa: E402


ROOT = Path(r"E:\SPARROW\5_Test\20260816_5")
OUT = ROOT / "outputs"
REPORTS = ROOT / "reports"
S16_2 = Path(r"E:\SPARROW\5_Test\20260816_2")
S16_3 = Path(r"E:\SPARROW\5_Test\20260816_3")
S16_4 = Path(r"E:\SPARROW\5_Test\20260816_4")
OOF_PATH = S16_4 / "outputs" / "candidate_oof_predictions_2018_2021.parquet"
METRIC_PATH = S16_4 / "reports" / "candidate_oof_metrics.csv"
ETA_PATH = S16_4 / "outputs" / "candidate_fold_eta_parameters.parquet"
ENGINEERING_PATH = S16_4 / "reports" / "candidate_engineering_and_mass_audit.json"
SOIL_DECISION = S16_2 / "reports" / "soil_memory_decision.json"
SOIL_TABLE = S16_2 / "reports" / "soil_tau_admissibility.csv"
HYDROGEO_TABLE = S16_3 / "reports" / "delivery_mu_hydrogeo_plausibility.csv"
HYDROGEO_DECISION = S16_3 / "reports" / "hydrogeo_ttd_decision.json"
GW_PANEL = S16_4 / "outputs" / "groundwater_aquifer_decade_panel.parquet"
GW_POINTS = S16_4 / "outputs" / "groundwater_point_context_diagnostics.parquet"
REFERENCE_MODEL = "S1_tau_480m_mu_000m"
N_BOOT = 10000
BOOT_SEED = 20260816
MARGIN = 0.01


def bootstrap_difference(candidate: pd.DataFrame, reference: pd.DataFrame, block: str) -> np.ndarray:
    keys = ["station_key", "year", "month"]
    candidate_columns = list(dict.fromkeys(keys + [block, "tn_mg_l", "pred_tn_mg_l"]))
    left = candidate[candidate_columns].rename(columns={"pred_tn_mg_l": "pred_candidate"})
    right = reference[keys + ["pred_tn_mg_l"]].rename(columns={"pred_tn_mg_l": "pred_reference"})
    paired = left.merge(right, on=keys, validate="one_to_one")
    paired["se_candidate"] = (np.log1p(paired.pred_candidate) - np.log1p(paired.tn_mg_l)) ** 2
    paired["se_reference"] = (np.log1p(paired.pred_reference) - np.log1p(paired.tn_mg_l)) ** 2
    summary = paired.groupby(block, as_index=False).agg(
        n=("se_candidate", "size"),
        se_candidate=("se_candidate", "sum"),
        se_reference=("se_reference", "sum"),
    )
    rng = np.random.default_rng(BOOT_SEED)
    draw = rng.integers(0, len(summary), size=(N_BOOT, len(summary)))
    n = summary.n.to_numpy(float)[draw].sum(axis=1)
    candidate_rmse = np.sqrt(summary.se_candidate.to_numpy(float)[draw].sum(axis=1) / n)
    reference_rmse = np.sqrt(summary.se_reference.to_numpy(float)[draw].sum(axis=1) / n)
    return candidate_rmse - reference_rmse


def gw_context_by_model(panel: pd.DataFrame, point: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_id, group in panel.groupby("model_id", sort=False):
        valid = group.dropna(subset=["observed_no3_mg_l", "model_gw_pattern_mg_l_as_tn"])
        spatial_values = []
        spatial_counts = []
        for _, decade_group in valid.groupby("decade"):
            spatial_counts.append(len(decade_group))
            if len(decade_group) >= 3:
                spatial_values.append(float(decade_group.observed_no3_mg_l.corr(decade_group.model_gw_pattern_mg_l_as_tn, method="spearman")))
        direction_pairs = 0
        direction_agreements = 0
        for _, aquifer_group in valid.groupby("aquifer_id"):
            aquifer_group = aquifer_group.assign(decade_start=aquifer_group.decade.str[:4].astype(int)).sort_values("decade_start")
            if len(aquifer_group) < 2:
                continue
            obs_delta = np.diff(aquifer_group.observed_no3_mg_l.to_numpy(float))
            model_delta = np.diff(aquifer_group.model_gw_pattern_mg_l_as_tn.to_numpy(float))
            direction_pairs += len(obs_delta)
            direction_agreements += int(np.sum(np.sign(obs_delta) == np.sign(model_delta)))
        p = point.loc[point.model_id.eq(model_id)]
        gross_fraction = float(p.gross_order_of_magnitude_contradiction.mean()) if len(p) else np.nan
        if not spatial_values and direction_pairs == 0:
            status = "non_identifying"
        elif spatial_values and all(value < 0 for value in spatial_values) and direction_pairs and direction_agreements == 0:
            status = "contextually_contradictory_not_hard_gate"
        elif spatial_values and np.nanmedian(spatial_values) > 0 and (direction_pairs == 0 or direction_agreements / direction_pairs >= 0.5):
            status = "contextually_consistent_not_hard_gate"
        else:
            status = "non_identifying_mixed"
        rows.append({
            "model_id": model_id,
            "gw_context_status": status,
            "decades_with_spatial_rank": len(spatial_values),
            "median_spatial_spearman": float(np.nanmedian(spatial_values)) if spatial_values else np.nan,
            "matched_same_aquifer_direction_pairs": direction_pairs,
            "matched_same_aquifer_direction_agreement_fraction": float(direction_agreements / direction_pairs) if direction_pairs else np.nan,
            "local_point_gross_contradiction_fraction": gross_fraction,
            "formal_pattern_excludes_eta_gw": True,
        })
    return pd.DataFrame(rows)


def main() -> None:
    require_runtime()
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    for stage in (2, 3, 4):
        audit = Path(rf"E:\SPARROW\5_Test\20260816_{stage}\reports\completion_audit.json")
        if not json.loads(audit.read_text(encoding="utf-8")).get("pass"):
            raise RuntimeError(f"20260816_{stage} did not pass")
    protected = [
        OOF_PATH, METRIC_PATH, ETA_PATH, ENGINEERING_PATH,
        SOIL_DECISION, SOIL_TABLE, HYDROGEO_TABLE, HYDROGEO_DECISION,
        GW_PANEL, GW_POINTS,
    ]
    start_hashes = hash_manifest(protected)
    dump_json(REPORTS / "parent_hashes_start.json", start_hashes)
    oof = pd.read_parquet(OOF_PATH)
    metrics = pd.read_csv(METRIC_PATH)
    eta = pd.read_parquet(ETA_PATH)
    engineering = json.loads(ENGINEERING_PATH.read_text(encoding="utf-8"))
    soil_decision = json.loads(SOIL_DECISION.read_text(encoding="utf-8"))
    admissible_tau = set(map(int, soil_decision["admissible_soil_tau_month"]))
    hydrogeo = pd.read_csv(HYDROGEO_TABLE)
    hydrogeo_lookup = hydrogeo.set_index("delivery_mu_month")
    panel = pd.read_parquet(GW_PANEL)
    point = pd.read_parquet(GW_POINTS)
    gw_context = gw_context_by_model(panel, point)
    gw_context.to_csv(REPORTS / "groundwater_context_by_candidate.csv", index=False)

    reference = oof.loc[oof.model_id.eq(REFERENCE_MODEL)].copy()
    if len(reference) != 4097:
        raise RuntimeError("reference candidate OOF is incomplete")
    bootstrap_rows = []
    ci_rows = []
    for model_id, candidate in oof.groupby("model_id", sort=True):
        ci_record: dict[str, object] = {"model_id": model_id}
        for block in ("station_key", "terminal_tree_id"):
            values = bootstrap_difference(candidate, reference, block)
            lower, upper = np.percentile(values, [2.5, 97.5])
            label = "station" if block == "station_key" else "terminal_tree"
            ci_record[f"{label}_ci_lower"] = float(lower)
            ci_record[f"{label}_ci_upper"] = float(upper)
            ci_record[f"{label}_bootstrap_mean"] = float(values.mean())
            bootstrap_rows.append(pd.DataFrame({
                "model_id": model_id,
                "block": block,
                "replicate": np.arange(N_BOOT, dtype=int),
                "delta_log_rmse": values,
            }))
        ci_record["river_noninferior"] = bool(ci_record["station_ci_upper"] < MARGIN and ci_record["terminal_tree_ci_upper"] < MARGIN)
        ci_record["predictively_improved"] = bool(ci_record["station_ci_upper"] < 0.0 and ci_record["terminal_tree_ci_upper"] < 0.0)
        ci_rows.append(ci_record)
    ci = pd.DataFrame(ci_rows)
    pd.concat(bootstrap_rows, ignore_index=True).to_parquet(OUT / "candidate_bootstrap_distributions.parquet", index=False)
    ci.to_csv(REPORTS / "candidate_bootstrap_ci.csv", index=False)

    records = []
    for metric in metrics.itertuples(index=False):
        model_id = str(metric.model_id)
        audit = engineering[model_id]
        engineering_pass = bool(
            audit["spinup"]["converged"]
            and audit["minimum_state_or_flux_kg_n"] >= -1e-9
            and audit["max_relative_mass_balance_error"] <= 1e-12
            and audit["T0_T1_mutually_exclusive"]
        )
        structure = str(metric.source_structure)
        tau = None if pd.isna(metric.soil_tau_month) else int(metric.soil_tau_month)
        mu = int(metric.delivery_mu_month)
        soil_pass = bool(structure == "S1" and tau in admissible_tau)
        hydrogeo_row = hydrogeo_lookup.loc[mu]
        hydrogeo_pass = not bool(hydrogeo_row.hydrogeo_hard_contradiction)
        ci_row = ci.loc[ci.model_id.eq(model_id)].iloc[0]
        gw_row = gw_context.loc[gw_context.model_id.eq(model_id)].iloc[0]
        final_admissible = bool(engineering_pass and soil_pass and hydrogeo_pass and ci_row.river_noninferior)
        records.append({
            **metric._asdict(),
            "engineering_pass": engineering_pass,
            "soil_structural_admissible": soil_pass,
            "soil_role": "formal_S1_candidate" if structure == "S1" else "delivery_control_only",
            "hydrogeo_hard_contradiction": bool(hydrogeo_row.hydrogeo_hard_contradiction),
            "hydrogeo_pass": hydrogeo_pass,
            "hydrogeo_gradient_sensitive": bool(hydrogeo_row.local_gradient_gross_contradiction != hydrogeo_row.path_gradient_gross_contradiction),
            "gw_context_status": gw_row.gw_context_status,
            "gw_context_is_hard_gate": False,
            "station_ci_lower": ci_row.station_ci_lower,
            "station_ci_upper": ci_row.station_ci_upper,
            "terminal_tree_ci_lower": ci_row.terminal_tree_ci_lower,
            "terminal_tree_ci_upper": ci_row.terminal_tree_ci_upper,
            "river_noninferior": bool(ci_row.river_noninferior),
            "predictively_improved": bool(ci_row.predictively_improved),
            "final_structural_admissible": final_admissible,
        })
    adjudication = pd.DataFrame(records).sort_values(["source_structure", "soil_tau_month", "delivery_mu_month", "model_id"])
    adjudication.to_csv(REPORTS / "progressive_candidate_adjudication.csv", index=False)

    admissible = adjudication.loc[adjudication.final_structural_admissible].copy()
    if admissible.empty:
        attribution_status = "external_constraints_incompatible_or_operator_nonidentifying"
        representative = None
        delivery_status = "no_admissible_delivery_parameter"
        predictive_reference = "20260815_7/S1_tau_480m+T0"
    else:
        attribution_status = "externally_constrained_admissible_region"
        superior = admissible.loc[admissible.predictively_improved]
        if not superior.empty:
            representative = str(superior.sort_values(["rmse_log1p", "model_id"]).iloc[0].model_id)
            representative_rule = "best_oof_within_jointly_bootstrap_superior_admissible_set"
        else:
            ordered_mu = sorted(admissible.delivery_mu_month.astype(int).unique())
            median_mu = float(np.median(ordered_mu))
            candidate_rep = admissible.assign(distance_to_mu_medoid=np.abs(admissible.delivery_mu_month - median_mu))
            representative = str(candidate_rep.sort_values(["distance_to_mu_medoid", "rmse_log1p", "model_id"]).iloc[0].model_id)
            representative_rule = "mu_medoid_then_oof_tie_break_within_noninferior_admissible_region"
        predictive_reference = REFERENCE_MODEL
        admissible_mu = sorted(admissible.delivery_mu_month.astype(int).unique())
        if len(admissible_mu) == 1:
            delivery_status = "bounded_to_one_registered_level"
        elif max(admissible_mu) == int(hydrogeo.delivery_mu_month.max()):
            delivery_status = "partially_bounded_right_censored_at_registered_20yr_limit"
        else:
            delivery_status = "partially_bounded_within_registered_grid"

    admissible_pairs = admissible[["model_id", "soil_tau_month", "delivery_mu_month"]].to_dict("records") if len(admissible) else []
    eta_summary = eta.groupby("model_id", as_index=False).agg(
        eta_quick_min=("eta_quick", "min"), eta_quick_max=("eta_quick", "max"),
        eta_gw_min=("eta_gw", "min"), eta_gw_max=("eta_gw", "max"),
        delivery_efficiency_confounded=("delivery_efficiency_confounded", "max"),
    )
    admissible_eta = eta_summary.loc[eta_summary.model_id.isin(admissible.model_id)] if len(admissible) else eta_summary.iloc[0:0]
    partition_status = (
        "delivery_efficiency_confounded" if len(admissible_eta) and admissible_eta.delivery_efficiency_confounded.any()
        else "not_boundary_confounded_in_admissible_region" if len(admissible_eta)
        else "not_evaluated_empty_admissible_region"
    )
    decision = {
        "scenario_id": "20260816_5",
        "attribution_status": attribution_status,
        "predictive_reference": predictive_reference,
        "incumbent_comparison_model_id": REFERENCE_MODEL,
        "soil_memory_status": soil_decision["soil_memory_status"],
        "admissible_soil_tau_month": sorted(admissible_tau),
        "hydrogeo_evidence_status": json.loads(HYDROGEO_DECISION.read_text(encoding="utf-8"))["hydrogeo_evidence_status"],
        "delivery_memory_status": delivery_status,
        "partition_status": partition_status,
        "admissible_parameter_pairs": admissible_pairs,
        "n_admissible": len(admissible_pairs),
        "representative_model_id": representative,
        "representative_rule": representative_rule if len(admissible) else None,
        "n_predictively_improved_within_admissible": int(admissible.predictively_improved.sum()) if len(admissible) else 0,
        "gw_n_role": "contextual_not_hard_gate",
        "empty_set_gate_relaxation_applied": False,
        "grid_expanded": False,
        "locked_2022_used": False,
    }
    dump_json(REPORTS / "legacy_attribution_decision.json", decision)
    eta_summary.to_csv(REPORTS / "eta_confounding_by_candidate.csv", index=False)
    end_hashes = hash_manifest(protected)
    dump_json(REPORTS / "parent_hashes_end.json", end_hashes)
    if start_hashes != end_hashes:
        raise RuntimeError("protected parent changed during stage 5")
    completion = {
        "scenario_id": "20260816_5",
        "pass": True,
        "candidate_count": len(adjudication),
        "bootstrap_rows": 63 * 2 * N_BOOT,
        "n_admissible": len(admissible_pairs),
        "attribution_status": attribution_status,
        "locked_2022_read": False,
    }
    dump_json(REPORTS / "completion_audit.json", completion)
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
