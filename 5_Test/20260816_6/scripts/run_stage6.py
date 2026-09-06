from __future__ import annotations

import json
import sys
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


S16_1 = Path(r"E:\SPARROW\5_Test\20260816_1")
sys.path.insert(0, str(S16_1 / "scripts"))
from legacy16_core import FOLD_PATH, OBS_PATH, dump_json, hash_manifest, load_module, require_runtime  # noqa: E402


ROOT = Path(r"E:\SPARROW\5_Test\20260816_6")
OUT = ROOT / "outputs"
REPORTS = ROOT / "reports"
S16_3 = Path(r"E:\SPARROW\5_Test\20260816_3")
S16_4 = Path(r"E:\SPARROW\5_Test\20260816_4")
S16_5 = Path(r"E:\SPARROW\5_Test\20260816_5")
DECISION5 = S16_5 / "reports" / "legacy_attribution_decision.json"
ADJUDICATION = S16_5 / "reports" / "progressive_candidate_adjudication.csv"
OOF = S16_4 / "outputs" / "candidate_oof_predictions_2018_2021.parquet"
ROUTED = S16_4 / "outputs" / "candidate_routed_paths_2016_2021.parquet"
GIS = S16_3 / "outputs" / "gis_ttd_by_reach.parquet"
TREE_MAPPING = Path(r"E:\SPARROW\5_Test\20260815_6\outputs\reach_terminal_tree_mapping.csv")
STAGE4_SCRIPT = S16_4 / "scripts" / "run_stage4.py"
SPEARMAN_MIN = 0.3
GROUP_AGREEMENT_MIN = 0.75


def weighted_log_mean(values: pd.Series, weights: pd.Series) -> float:
    valid = values.notna() & weights.notna() & weights.gt(0)
    if not valid.any():
        return np.nan
    return float(np.expm1(np.average(np.log1p(values[valid].to_numpy(float)), weights=weights[valid].to_numpy(float))))


def tree_hydrogeo_table() -> pd.DataFrame:
    gis = pd.read_parquet(GIS)
    mapping = pd.read_csv(TREE_MAPPING)
    merged = gis.merge(mapping, on="reach_id", validate="one_to_one")
    rows = []
    for tree_id, group in merged.groupby("terminal_tree_id"):
        rows.append({
            "terminal_tree_id": int(tree_id),
            "local_gradient_tree_ttd_month": weighted_log_mean(group.local_gradient_median_month, group.effective_area_km2),
            "path_gradient_tree_ttd_month": weighted_log_mean(group.path_gradient_median_month, group.effective_area_km2),
            "effective_area_km2": float(group.effective_area_km2.sum()),
            "low_quality_reach_fraction": float(np.average(group.ttd_prior_quality.eq("low"), weights=group.effective_area_km2)),
        })
    table = pd.DataFrame(rows)
    for method in ("local_gradient", "path_gradient"):
        rank = table[f"{method}_tree_ttd_month"].rank(method="first")
        table[f"{method}_group"] = pd.qcut(rank, 3, labels=["fast", "medium", "slow"]).astype(str)
    return table


def tree_best_mu(oof: pd.DataFrame, tau: int, admissible_mu: list[int]) -> pd.DataFrame:
    model_ids = {f"S1_tau_{tau:03d}m_mu_{mu:03d}m": mu for mu in admissible_mu}
    subset = oof.loc[oof.model_id.isin(model_ids)].copy()
    subset["se_log"] = (np.log1p(subset.pred_tn_mg_l) - np.log1p(subset.tn_mg_l)) ** 2
    score = subset.groupby(["model_id", "terminal_tree_id"], as_index=False).agg(n=("se_log", "size"), mse=("se_log", "mean"))
    score["rmse_log"] = np.sqrt(score.mse)
    score["delivery_mu_month"] = score.model_id.map(model_ids).astype(int)
    best = score.sort_values(["terminal_tree_id", "rmse_log", "delivery_mu_month"]).drop_duplicates("terminal_tree_id")
    return best[["terminal_tree_id", "delivery_mu_month", "rmse_log", "n"]]


def signal_diagnostics(best: pd.DataFrame, hydrogeo: pd.DataFrame, tau: int) -> list[dict[str, object]]:
    joined = best.merge(hydrogeo, on="terminal_tree_id", validate="one_to_one")
    rows = []
    for method in ("local_gradient", "path_gradient"):
        metric = joined[f"{method}_tree_ttd_month"]
        rho = float(spearmanr(metric, joined.delivery_mu_month).statistic)
        medians = joined.groupby(f"{method}_group").delivery_mu_month.median().reindex(["fast", "medium", "slow"])
        monotonic = bool(medians.fast <= medians.medium <= medians.slow)
        rows.append({
            "soil_tau_month": tau,
            "gradient_method": method,
            "n_terminal_trees": len(joined),
            "spearman_tree_ttd_vs_best_mu": rho,
            "fast_best_mu_median": float(medians.fast),
            "medium_best_mu_median": float(medians.medium),
            "slow_best_mu_median": float(medians.slow),
            "fast_le_medium_le_slow": monotonic,
            "signal_pass": bool(np.isfinite(rho) and rho >= SPEARMAN_MIN and monotonic),
        })
    return rows


def main() -> None:
    require_runtime()
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    if not json.loads((S16_5 / "reports" / "completion_audit.json").read_text(encoding="utf-8")).get("pass"):
        raise RuntimeError("20260816_5 did not pass")
    protected = [DECISION5, ADJUDICATION, OOF, ROUTED, GIS, TREE_MAPPING, STAGE4_SCRIPT, OBS_PATH, FOLD_PATH]
    start_hashes = hash_manifest(protected)
    dump_json(REPORTS / "parent_hashes_start.json", start_hashes)
    decision5 = json.loads(DECISION5.read_text(encoding="utf-8"))
    oof = pd.read_parquet(OOF)
    hydrogeo = tree_hydrogeo_table()
    hydrogeo.to_csv(REPORTS / "terminal_tree_hydrogeo_groups.csv", index=False)
    admissible_pairs = pd.DataFrame(decision5["admissible_parameter_pairs"])
    soil_taus = [int(value) for value in sorted(admissible_pairs.soil_tau_month.astype(int).unique())] if len(admissible_pairs) else []
    admissible_mu = [int(value) for value in sorted(admissible_pairs.delivery_mu_month.astype(int).unique())] if len(admissible_pairs) else []
    best_frames = []
    diagnostic_rows = []
    for tau in soil_taus:
        best = tree_best_mu(oof, tau, admissible_mu)
        best["soil_tau_month"] = tau
        best_frames.append(best)
        diagnostic_rows.extend(signal_diagnostics(best, hydrogeo, tau))
    best_all = pd.concat(best_frames, ignore_index=True) if best_frames else pd.DataFrame()
    best_all.to_csv(REPORTS / "terminal_tree_best_global_mu.csv", index=False)
    diagnostics = pd.DataFrame(diagnostic_rows)
    diagnostics.to_csv(REPORTS / "regionalization_signal_diagnostics.csv", index=False)
    group_agreement = float((hydrogeo.local_gradient_group == hydrogeo.path_gradient_group).mean())
    soil_precondition = decision5["soil_memory_status"] == "bounded" and bool(soil_taus)
    all_tau_stable = bool(len(diagnostics) and diagnostics.signal_pass.all())
    gradient_group_stable = bool(group_agreement >= GROUP_AGREEMENT_MIN)
    authorized = bool(soil_precondition and all_tau_stable and gradient_group_stable)

    if not soil_precondition:
        status = "soil_uncertainty_confounded_not_run"
    elif not authorized:
        status = "hydrogeo_or_tree_signal_nonidentifying_not_run"
    else:
        status = "authorized_and_run"

    regional_results = []
    regional_predictions = []
    selected_by_method = {}
    if authorized:
        stage4 = load_module(STAGE4_SCRIPT, "legacy16_stage4_regional")
        routed = pd.read_parquet(ROUTED)
        observations = pd.read_parquet(OBS_PATH).loc[lambda x: x.year.between(2016, 2021)].copy()
        folds = pd.read_parquet(FOLD_PATH)
        for tau in soil_taus:
            for method in ("local_gradient", "path_gradient"):
                group_map = hydrogeo.set_index("terminal_tree_id")[f"{method}_group"].to_dict()
                for fast_mu, medium_mu, slow_mu in combinations_with_replacement(admissible_mu, 3):
                    assignment = {"fast": fast_mu, "medium": medium_mu, "slow": slow_mu}
                    pieces = []
                    for group_name, mu in assignment.items():
                        model_id = f"S1_tau_{tau:03d}m_mu_{mu:03d}m"
                        trees = [tree for tree, label in group_map.items() if label == group_name]
                        pieces.append(routed.loc[routed.model_id.eq(model_id) & routed.terminal_tree_id.isin(trees)].copy())
                    combined = pd.concat(pieces, ignore_index=True)
                    regional_id = f"regional_{method}_tau{tau:03d}_mu{fast_mu:03d}_{medium_mu:03d}_{slow_mu:03d}"
                    combined["model_id"] = regional_id
                    prediction, parameters = stage4.oof_predictions(combined, observations, folds)
                    metric = stage4.metrics(prediction)
                    confounded = any(bool(row["delivery_efficiency_confounded"]) for row in parameters)
                    regional_results.append({
                        "model_id": regional_id,
                        "gradient_method": method,
                        "soil_tau_month": tau,
                        "fast_mu_month": fast_mu,
                        "medium_mu_month": medium_mu,
                        "slow_mu_month": slow_mu,
                        **metric,
                        "delivery_efficiency_confounded": confounded,
                    })
                    regional_predictions.append(prediction)
        result_table = pd.DataFrame(regional_results)
        result_table.to_csv(REPORTS / "regional_candidate_metrics.csv", index=False)
        pd.concat(regional_predictions, ignore_index=True).to_parquet(OUT / "regional_candidate_oof_predictions.parquet", index=False)
        for method, group in result_table.groupby("gradient_method"):
            selected_by_method[method] = str(group.sort_values(["rmse_log1p", "model_id"]).iloc[0].model_id)
        if len(set(selected_by_method.values())) > 1:
            status = "run_but_gradient_method_selection_unstable"
        else:
            status = "regional_structure_selected_exploratory"

    decision = {
        "scenario_id": "20260816_6",
        "regionalization_authorized": authorized,
        "regional_delivery_status": status,
        "soil_memory_status": decision5["soil_memory_status"],
        "soil_admissible_tau_month": soil_taus,
        "global_admissible_mu_month": admissible_mu,
        "all_soil_tau_signal_stable": all_tau_stable,
        "local_path_tree_group_agreement_fraction": group_agreement,
        "minimum_required_group_agreement_fraction": GROUP_AGREEMENT_MIN,
        "selected_regional_model_by_gradient_method": selected_by_method,
        "fast_le_medium_le_slow_enforced": True,
        "registered_mu_only": True,
        "locked_2022_used": False,
    }
    dump_json(REPORTS / "regionalization_decision.json", decision)
    end_hashes = hash_manifest(protected)
    dump_json(REPORTS / "parent_hashes_end.json", end_hashes)
    if start_hashes != end_hashes:
        raise RuntimeError("protected parent changed during stage 6")
    completion = {
        "scenario_id": "20260816_6",
        "pass": True,
        "regionalization_authorized": authorized,
        "regional_candidate_count": len(regional_results),
        "regional_delivery_status": status,
        "locked_2022_read": False,
    }
    dump_json(REPORTS / "completion_audit.json", completion)
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
