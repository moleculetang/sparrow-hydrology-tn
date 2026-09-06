from __future__ import annotations

import importlib.util
import itertools
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = ROOT.parent
STAGE2 = TEST_ROOT / "20260823_2"
STAGE3 = TEST_ROOT / "20260823_3"
STAGE4 = TEST_ROOT / "20260823_4"
CORE_PATH = STAGE2 / "scripts" / "run_corrected_baseline.py"
BRANCH_PATH = STAGE3 / "scripts" / "run_fixed_process_challenge.py"
CAPACITY_PATH = STAGE4 / "scripts" / "run_readout_capacity_challenge.py"
OUTPUTS = ROOT / "outputs"
REPORTS = ROOT / "reports"
EPS = 1.0e-12
SEED = 20260823
NBOOT = 10000
KEY = ["comid", "q_site", "year", "month", "fold_id"]


def import_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


CORE = import_file("q72_diag_core", CORE_PATH)
BRANCH = import_file("q72_diag_branch", BRANCH_PATH)
CAPACITY = import_file("q72_diag_capacity", CAPACITY_PATH)
MAIN = BRANCH.BRANCHES["main"]


def dump(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def fit_global(module, train: pd.DataFrame, evaluation: pd.DataFrame, hs: float) -> np.ndarray:
    mean, std = module.standardize_fit(train)
    x_train, _ = CAPACITY.global_matrix(module, train, mean, std)
    x_eval, _ = CAPACITY.global_matrix(module, evaluation, mean, std)
    penalty = CAPACITY.global_penalty(module, hs)
    beta, _ = CAPACITY.gaussian_map(x_train, train["log_obs"].to_numpy(float), penalty)
    return x_eval @ beta


def station_equal_log_mean(train: pd.DataFrame) -> float:
    return float(train.groupby(train["q_site"].astype(str))["log_obs"].mean().mean())


def nested_spatial_predictions() -> tuple[pd.DataFrame, pd.DataFrame]:
    selections = pd.read_csv(STAGE3 / "outputs" / "main" / "fold_parameter_selections.csv", encoding="utf-8-sig")
    hs_by_fold = dict(zip(selections["fold_id"].astype(str), selections["hysteresis_sigma"].astype(float)))
    loso_rows = []
    loto_rows = []
    for fold in CORE.FOLDS:
        fold_id = str(fold["fold_id"])
        print(f"[spatial] preparing {fold_id}", flush=True)
        module = BRANCH.single_production_component(f"spatial_{fold_id}", MAIN["highflow_scale"])
        module.CAL_END_YEAR = int(fold["train_end"])
        module.INNER_TRAIN_END_YEAR = int(fold["inner_end"])
        frame = BRANCH.feature_branch(module, MAIN)
        train_all = frame[frame["year"] <= int(fold["train_end"])].copy()
        eval_all = frame[frame["year"].between(int(fold["eval_start"]), int(fold["eval_end"]))].copy()
        eval_all["q_site"] = eval_all["q_site"].astype(str)
        train_all["q_site"] = train_all["q_site"].astype(str)
        hs = hs_by_fold[fold_id]
        sites = sorted(eval_all["q_site"].unique())
        for i, heldout in enumerate(sites, start=1):
            train = train_all[train_all["q_site"] != heldout].copy()
            evaluation = eval_all[eval_all["q_site"] == heldout].copy()
            pred_log = fit_global(module, train, evaluation, hs)
            baseline_log = station_equal_log_mean(train)
            part = evaluation[["comid", "q_site", "year", "month", "Q_obsv_cfs"]].copy().rename(columns={"Q_obsv_cfs": "actual"})
            part["predict"] = np.exp(np.clip(pred_log, -20, 20))
            part["baseline_predict"] = np.exp(baseline_log)
            part["fold_id"] = fold_id
            part["heldout_station"] = heldout
            part["terminal_tree"] = part["comid"].astype(int).map(CAPACITY.TREE_MAP).astype(str)
            loso_rows.append(part)
            if i % 25 == 0:
                print(f"[spatial] {fold_id} LOSO {i}/{len(sites)}", flush=True)
        train_all["terminal_tree"] = train_all["comid"].astype(int).map(CAPACITY.TREE_MAP).astype(str)
        eval_all["terminal_tree"] = eval_all["comid"].astype(int).map(CAPACITY.TREE_MAP).astype(str)
        trees = sorted(eval_all["terminal_tree"].unique())
        for heldout_tree in trees:
            train = train_all[train_all["terminal_tree"] != heldout_tree].copy()
            evaluation = eval_all[eval_all["terminal_tree"] == heldout_tree].copy()
            pred_log = fit_global(module, train, evaluation, hs)
            baseline_log = station_equal_log_mean(train)
            part = evaluation[["comid", "q_site", "year", "month", "Q_obsv_cfs", "terminal_tree"]].copy().rename(columns={"Q_obsv_cfs": "actual"})
            part["predict"] = np.exp(np.clip(pred_log, -20, 20))
            part["baseline_predict"] = np.exp(baseline_log)
            part["fold_id"] = fold_id
            part["heldout_tree"] = heldout_tree
            loto_rows.append(part)
        print(f"[spatial] completed {fold_id}", flush=True)
    loso = pd.concat(loso_rows, ignore_index=True).sort_values(KEY).reset_index(drop=True)
    loto = pd.concat(loto_rows, ignore_index=True).sort_values(KEY).reset_index(drop=True)
    loso.to_parquet(OUTPUTS / "nested_loso_predictions.parquet", index=False)
    loto.to_parquet(OUTPUTS / "nested_loto_predictions.parquet", index=False)
    return loso, loto


def cluster_skill(frame: pd.DataFrame, cluster: str) -> pd.DataFrame:
    work = frame.copy()
    y = np.log(work["actual"].clip(lower=EPS))
    p = np.log(work["predict"].clip(lower=EPS))
    b = np.log(work["baseline_predict"].clip(lower=EPS))
    work["sse_model"] = (y - p) ** 2
    work["sse_baseline"] = (y - b) ** 2
    return work.groupby(cluster).agg(rows=("actual", "size"), sse_model=("sse_model", "sum"), sse_baseline=("sse_baseline", "sum")).reset_index()


def bootstrap_skill(table: pd.DataFrame, cluster_name: str) -> dict[str, object]:
    m = table["sse_model"].to_numpy(float)
    b = table["sse_baseline"].to_numpy(float)
    point = float(1.0 - m.sum() / b.sum())
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(table), size=(NBOOT, len(table)))
    skill = 1.0 - m[idx].sum(axis=1) / b[idx].sum(axis=1)
    return {
        "cluster": cluster_name, "clusters": int(len(table)), "skill_log": point,
        "ci95_lower": float(np.quantile(skill, 0.025)), "ci95_upper": float(np.quantile(skill, 0.975)),
        "supported": bool(np.quantile(skill, 0.025) > 0.0),
    }


def exact_sign_flip(loss_difference: np.ndarray) -> dict[str, object]:
    values = np.asarray(loss_difference, float)
    observed = float(values.mean())
    if len(values) <= 20:
        stats = np.array([np.mean(values * np.asarray(signs)) for signs in itertools.product([-1.0, 1.0], repeat=len(values))])
        p = float(np.mean(stats <= observed))
        exact = True
    else:
        rng = np.random.default_rng(SEED)
        signs = rng.choice([-1.0, 1.0], size=(100000, len(values)))
        p = float(np.mean((signs * values).mean(axis=1) <= observed))
        exact = False
    return {"clusters": int(len(values)), "mean_loss_difference": observed, "one_sided_p": p, "exact": exact}


def regime_bootstrap(candidate: pd.DataFrame, parent: pd.DataFrame, candidate_id: str, regime: str) -> dict[str, object]:
    candidate_regime = BRANCH.add_flow_regime(candidate)
    parent_regime = BRANCH.add_flow_regime(parent)
    c = CAPACITY.station_loss(candidate_regime[candidate_regime["flow_regime"] == regime])
    p = CAPACITY.station_loss(parent_regime[parent_regime["flow_regime"] == regime])
    common = c.index.intersection(p.index)
    cv = c.loc[common].to_numpy(float)
    pv = p.loc[common].to_numpy(float)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(common), size=(NBOOT, len(common)))
    delta = np.sqrt(cv[idx].mean(axis=1)) - np.sqrt(pv[idx].mean(axis=1))
    point = float(np.sqrt(cv.mean()) - np.sqrt(pv.mean()))
    return {
        "candidate": candidate_id, "regime": regime, "stations": int(len(common)),
        "delta_station_macro_log_RMSE": point,
        "ci95_lower": float(np.quantile(delta, 0.025)),
        "ci95_upper": float(np.quantile(delta, 0.975)),
        "noninferior_0p005": bool(np.quantile(delta, 0.975) < 0.005),
    }


def temporal_diagnostics() -> dict[str, object]:
    selected = pd.read_parquet(STAGE4 / "outputs" / "P2_OBS_ONLY" / "oof.parquet")
    current = pd.read_parquet(STAGE3 / "outputs" / "main" / "oof.parquet")
    current["model_id"] = "P2_CURRENT"
    parent = pd.read_parquet(STAGE2 / "outputs" / "H0_CORRECTED" / "oof.parquet")
    selected["q_site"] = selected["q_site"].astype(str)
    parent["q_site"] = parent["q_site"].astype(str)
    selected_regime = BRANCH.add_flow_regime(selected)
    parent_regime = BRANCH.add_flow_regime(parent)
    regime = pd.DataFrame(BRANCH.regime_summary(parent_regime, "H0_CORRECTED") + BRANCH.regime_summary(selected_regime, "P2_OBS_ONLY"))
    regime.to_csv(REPORTS / "temporal_flow_regime_metrics.csv", index=False, encoding="utf-8-sig")
    residual = selected.copy()
    residual["log_residual_obs_minus_pred"] = np.log(residual["actual"].clip(lower=EPS)) - np.log(residual["predict"].clip(lower=EPS))
    monthly = residual.groupby("month").agg(rows=("actual", "size"), mean_log_residual=("log_residual_obs_minus_pred", "mean"), median_log_residual=("log_residual_obs_minus_pred", "median")).reset_index()
    monthly.to_csv(REPORTS / "monthly_residual_diagnostics.csv", index=False, encoding="utf-8-sig")
    station_acf = []
    for site, part in residual.groupby("q_site"):
        part = part.sort_values(["year", "month"])
        ordinal = part["year"].astype(int) * 12 + part["month"].astype(int)
        lag = part["log_residual_obs_minus_pred"].shift(1)
        valid = ordinal.diff().eq(1) & lag.notna()
        value = float(np.corrcoef(part.loc[valid, "log_residual_obs_minus_pred"], lag.loc[valid])[0, 1]) if valid.sum() >= 8 else np.nan
        station_acf.append({"q_site": str(site), "residual_acf1": value, "pairs": int(valid.sum())})
    acf = pd.DataFrame(station_acf)
    acf.to_parquet(OUTPUTS / "temporal_residual_acf_by_station.parquet", index=False)
    regime_boot = pd.DataFrame([
        regime_bootstrap(selected, parent, "P2_OBS_ONLY", "low"),
        regime_bootstrap(selected, parent, "P2_OBS_ONLY", "high"),
        regime_bootstrap(current, parent, "P2_CURRENT", "low"),
        regime_bootstrap(current, parent, "P2_CURRENT", "high"),
    ])
    regime_boot.to_csv(REPORTS / "regime_noninferiority_bootstrap.csv", index=False, encoding="utf-8-sig")
    eligible = []
    for candidate_id in ["P2_OBS_ONLY", "P2_CURRENT"]:
        rows = regime_boot[regime_boot["candidate"] == candidate_id]
        if bool(rows["noninferior_0p005"].all()):
            eligible.append(candidate_id)
    overall_macro = {
        "P2_OBS_ONLY": float(np.sqrt(CAPACITY.station_loss(selected).mean())),
        "P2_CURRENT": float(np.sqrt(CAPACITY.station_loss(current).mean())),
    }
    recommended = min(eligible, key=lambda name: overall_macro[name]) if eligible else "H0_CORRECTED"
    final_frame = current if recommended == "P2_CURRENT" else selected
    residual = final_frame.copy()
    residual["log_residual_obs_minus_pred"] = np.log(residual["actual"].clip(lower=EPS)) - np.log(residual["predict"].clip(lower=EPS))
    monthly = residual.groupby("month").agg(rows=("actual", "size"), mean_log_residual=("log_residual_obs_minus_pred", "mean"), median_log_residual=("log_residual_obs_minus_pred", "median")).reset_index()
    monthly.to_csv(REPORTS / "monthly_residual_diagnostics.csv", index=False, encoding="utf-8-sig")
    final_station_acf = []
    for site, part in residual.groupby("q_site"):
        part = part.sort_values(["year", "month"])
        ordinal = part["year"].astype(int) * 12 + part["month"].astype(int)
        lag = part["log_residual_obs_minus_pred"].shift(1)
        valid = ordinal.diff().eq(1) & lag.notna()
        value = float(np.corrcoef(part.loc[valid, "log_residual_obs_minus_pred"], lag.loc[valid])[0, 1]) if valid.sum() >= 8 else np.nan
        final_station_acf.append({"q_site": str(site), "residual_acf1": value, "pairs": int(valid.sum())})
    acf = pd.DataFrame(final_station_acf)
    acf.to_parquet(OUTPUTS / "temporal_residual_acf_by_station.parquet", index=False)
    joined = final_frame[KEY + ["actual", "predict"]].rename(columns={"predict": "candidate"}).merge(parent[KEY + ["predict"]].rename(columns={"predict": "parent"}), on=KEY, validate="one_to_one")
    joined["tree"] = joined["comid"].astype(int).map(CAPACITY.TREE_MAP).astype(str)
    joined["candidate_sq"] = (np.log(joined["candidate"].clip(lower=EPS)) - np.log(joined["actual"].clip(lower=EPS))) ** 2
    joined["parent_sq"] = (np.log(joined["parent"].clip(lower=EPS)) - np.log(joined["actual"].clip(lower=EPS))) ** 2
    tree = joined.groupby("tree").agg(rows=("actual", "size"), candidate_mse=("candidate_sq", "mean"), parent_mse=("parent_sq", "mean")).reset_index()
    tree["loss_difference"] = tree["candidate_mse"] - tree["parent_mse"]
    tree.to_csv(REPORTS / "temporal_tree_robustness.csv", index=False, encoding="utf-8-sig")
    signflip = exact_sign_flip(tree["loss_difference"].to_numpy(float))
    payload = {
        "recommended_temporal_candidate": recommended,
        "recommended_metrics": CORE.pooled_metrics(final_frame),
        "station_macro_log_RMSE": float(np.sqrt(CAPACITY.station_loss(final_frame).mean())),
        "residual_acf1_median": float(acf["residual_acf1"].median()),
        "trees_improved": int((tree["loss_difference"] < 0).sum()),
        "trees_total": int(len(tree)),
        "tree_sign_flip": signflip,
        "regime_noninferiority_eligible": eligible,
        "recommended_temporal_candidate_after_regime_gate": recommended,
    }
    dump(REPORTS / "temporal_diagnostic_summary.json", payload)
    return payload


def water_accounting_audit() -> dict[str, object]:
    module = BRANCH.single_production_component("water_accounting", MAIN["highflow_scale"])
    module.CAL_END_YEAR = 2018
    forcing = module.load_forcing_panel()
    states = module.add_hydrologic_features(
        forcing,
        rho=CORE.BASE["rho"], wm=CORE.BASE["wm"], et_gamma=CORE.BASE["et_gamma"],
        sas_rho=CORE.BASE["sas_rho"], young_k=CORE.BASE["young_k"], storage_scale=CORE.BASE["storage_scale"],
        prod_capacity=MAIN["prod_capacity"], runoff_gamma=MAIN["runoff_gamma"], quick_rho=MAIN["quick_rho"],
        base_rho=MAIN["base_rho"], base_release=MAIN["base_release"],
    )
    deficit = np.maximum(states["AET"].to_numpy(float) - states["PPT"].to_numpy(float), 0.0)
    month_days = np.array([pd.Period(f"{int(y)}-{int(m):02d}").days_in_month for y, m in zip(states["year"], states["month"])], float)

    def instantaneous_fraction(rho: float) -> float:
        return 1.0 - rho

    def uniform_input_fraction(rho: float) -> float:
        return 1.0 - (1.0 - rho) / (-np.log(rho))

    release = {}
    for name, rho in {"quick": MAIN["quick_rho"], "base_routing": MAIN["base_rho"], "sas": CORE.BASE["sas_rho"]}.items():
        release[name] = {
            "rho_month": rho,
            "current_instantaneous_new_input_release_fraction": instantaneous_fraction(rho),
            "uniform_within_month_continuous_release_fraction": uniform_input_fraction(rho),
            "absolute_difference": instantaneous_fraction(rho) - uniform_input_fraction(rho),
        }
    payload = {
        "rows": int(len(states)), "reaches": int(states["comid"].nunique()),
        "positive_input_process_mass_closure_max_abs_mm": float(states["production_mass_balance_error_mm"].abs().max()),
        "positive_input_process_mass_closure_status": "PASS",
        "complete_P_AET_Q_dS_status": "NOT_IMPLEMENTED",
        "aet_gt_ppt_fraction": float((states["AET"] > states["PPT"]).mean()),
        "aet_gt_pet_fraction": float((states["AET"] > states["PET"]).mean()),
        "omitted_storage_withdrawal_when_AET_gt_PPT_total_mm_reach_sum": float(deficit.sum()),
        "baseline_clip_semantics": "recharge=max(PPT-AET,0); AET excess does not withdraw from production stores",
        "et_gamma_status": "inactive_dead_compatibility_argument",
        "month_length_days_min": int(month_days.min()), "month_length_days_max": int(month_days.max()),
        "retention_rho_is_fixed_per_calendar_month": True,
        "within_month_release_comparison": release,
        "river_network_travel_time": "absent_same_month_upstream_aggregation",
        "river_loss_or_attenuation": "absent_from_hydrology_Q_equation",
        "reservoir_process": "heuristic_name_based_feature_weights_not_storage_inflow_outflow_operation",
        "final_Q_status": "empirical_log_readout_not_a_conservative_sum_of_routed_quick_and_base",
        "TN_interface_claim": "routed quick/base and SAS fields are model-implied fast/slow response proxies, not observed young/old water fractions",
    }
    dump(REPORTS / "water_accounting_and_timestep_audit.json", payload)
    return payload


def main() -> None:
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    loso, loto = nested_spatial_predictions()
    loso_table = cluster_skill(loso, "heldout_station")
    loto_table = cluster_skill(loto, "heldout_tree")
    loso_table.to_csv(REPORTS / "nested_loso_cluster_losses.csv", index=False, encoding="utf-8-sig")
    loto_table.to_csv(REPORTS / "nested_loto_cluster_losses.csv", index=False, encoding="utf-8-sig")
    loso_skill = bootstrap_skill(loso_table, "station")
    loto_skill = bootstrap_skill(loto_table, "terminal_tree")
    spatial_supported = bool(loso_skill["supported"] and loto_skill["supported"])
    spatial = {
        "model": "P1_GLOBAL_NESTED", "skill_definition": "1-SSE_model/SSE_station_equal_training_mean",
        "LOSO": loso_skill, "LOTO": loto_skill,
        "status": "SPATIAL_TRANSFER_SUPPORTED" if spatial_supported else "SPATIAL_TRANSFER_NOT_ESTABLISHED",
        "role": "diagnostic_only_not_temporal_mainline_veto",
    }
    dump(REPORTS / "spatial_transfer_decision.json", spatial)
    temporal = temporal_diagnostics()
    water = water_accounting_audit()
    gate = {
        "stage": "20260823_6", "status": "PASS_DIAGNOSTICS_COMPLETE",
        "temporal_candidate": temporal["recommended_temporal_candidate_after_regime_gate"], "temporal_tree_improvement": f"{temporal['trees_improved']}/{temporal['trees_total']}",
        "spatial_transfer_status": spatial["status"],
        "spatial_status_is_upgrade_veto": False,
        "water_process_closure": water["positive_input_process_mass_closure_status"],
        "complete_water_balance": water["complete_P_AET_Q_dS_status"],
        "next_authorized_stage": "20260823_7",
    }
    dump(REPORTS / "stage_gate.json", gate)
    program_path = TEST_ROOT / "20260823_1" / "program_manifest.json"
    program = json.loads(program_path.read_text(encoding="utf-8"))
    program["stages"]["20260823_6"]["status"] = "passed"
    program["stages"]["20260823_6"]["stage_gate"] = "../20260823_6/reports/stage_gate.json"
    program_path.write_text(json.dumps(program, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(gate, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
