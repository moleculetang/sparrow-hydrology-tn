from __future__ import annotations

import importlib.util
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = ROOT.parent
STAGE2 = TEST_ROOT / "20260823_2"
CORE_PATH = STAGE2 / "scripts" / "run_corrected_baseline.py"
OUTPUTS = ROOT / "outputs"
REPORTS = ROOT / "reports"
NBOOT = 10000
SEED = 20260823
EPS = 1.0e-12
KEY = ["comid", "q_site", "year", "month", "fold_id"]
BRANCHES = {
    "main": {"prod_capacity": 240.0, "runoff_gamma": 2.5, "quick_rho": 0.25, "base_release": 0.10, "base_rho": 0.85, "highflow_scale": 1.00},
    "flash": {"prod_capacity": 132.0, "runoff_gamma": 1.7, "quick_rho": 0.10, "base_release": 0.06, "base_rho": 0.76, "highflow_scale": 1.20},
    "slow": {"prod_capacity": 432.0, "runoff_gamma": 3.1, "quick_rho": 0.48, "base_release": 0.07, "base_rho": 0.94, "highflow_scale": 0.85},
    "buffer": {"prod_capacity": 348.0, "runoff_gamma": 3.3, "quick_rho": 0.66, "base_release": 0.12, "base_rho": 0.96, "highflow_scale": 0.65},
    "wet": {"prod_capacity": 288.0, "runoff_gamma": 2.2, "quick_rho": 0.34, "base_release": 0.08, "base_rho": 0.90, "highflow_scale": 1.10},
}


def load_core():
    spec = importlib.util.spec_from_file_location("q72_controlled_core", CORE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


CORE = load_core()


def dump(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def single_production_component(label: str, highflow_scale: float):
    module = CORE.load_component(label, highflow_scale=highflow_scale)
    removed = set(module.MULTISTORE_FEATURES)
    module.FIXED_FEATURES = [feature for feature in module.FIXED_FEATURES if feature not in removed]
    module.MULTISTORE_FEATURES = []
    return module


def feature_branch(module, branch: dict[str, float]) -> pd.DataFrame:
    forcing = module.load_forcing_panel()
    return module.build_featured_observation_panel(
        forcing,
        rho=CORE.BASE["rho"], wm=CORE.BASE["wm"], et_gamma=CORE.BASE["et_gamma"],
        sas_rho=CORE.BASE["sas_rho"], young_k=CORE.BASE["young_k"],
        storage_scale=CORE.BASE["storage_scale"],
        prod_capacity=branch["prod_capacity"], runoff_gamma=branch["runoff_gamma"],
        quick_rho=branch["quick_rho"], base_rho=branch["base_rho"],
        base_release=branch["base_release"], state_calendar_mode="full_forcing",
    )


def run_branch(branch_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    branch = BRANCHES[branch_id]
    parts = []
    selections = []
    for fold in CORE.FOLDS:
        fold_id = str(fold["fold_id"])
        print(f"[{branch_id}] starting {fold_id}", flush=True)
        module = single_production_component(f"{branch_id}_{fold_id}", branch["highflow_scale"])
        module.CAL_END_YEAR = int(fold["train_end"])
        module.INNER_TRAIN_END_YEAR = int(fold["inner_end"])
        frame = feature_branch(module, branch)
        stations = sorted(frame["q_site"].astype(str).unique())
        fold_dir = OUTPUTS / branch_id / fold_id
        fold_dir.mkdir(parents=True, exist_ok=True)
        hs, grid = CORE.choose_hysteresis(module, frame, stations, fold, fold_dir / "inner_artifacts")
        grid.to_csv(fold_dir / "hysteresis_grid_correct_kge2012.csv", index=False, encoding="utf-8-sig")
        beta, mean, std = CORE.fit(module, frame, stations, int(fold["train_end"]), hs, fold_dir / "final_fit")
        evaluation = frame[frame["year"].between(int(fold["eval_start"]), int(fold["eval_end"]))].copy()
        prediction = np.exp(np.clip(module.predict_log(evaluation, beta, stations, mean, std), -20, 20))
        out = evaluation[["comid", "q_site", "year", "month", "Q_obsv_cfs", "Q_calc_cfs", "routed_quick_cfs", "routed_base_cfs", "sas_old_release_cfs", "production_mass_balance_error_mm"]].copy()
        out = out.rename(columns={"Q_obsv_cfs": "actual"})
        out["predict"] = prediction
        out["fold_id"] = fold_id
        out["branch_id"] = branch_id
        out["hysteresis_sigma"] = hs
        for name, value in branch.items():
            out[name] = value
        out["network_input_scale"] = 1.0
        out.to_parquet(fold_dir / "evaluation_predictions.parquet", index=False)
        parts.append(out)
        selections.append({
            "branch_id": branch_id, "fold_id": fold_id, "hysteresis_sigma": hs,
            "design_columns": int(len(module.design_column_metadata(stations))),
            "evaluation_rows": int(len(out)),
            "network_input_scale": 1.0, **branch,
            "sum_primary_quick_cfs": float(evaluation["routed_quick_cfs"].sum()),
            "sum_primary_base_cfs": float(evaluation["routed_base_cfs"].sum()),
            "max_abs_mass_balance_error_mm": float(evaluation["production_mass_balance_error_mm"].abs().max()),
        })
        print(f"[{branch_id}] completed {fold_id}: hs={hs:g}, rows={len(out)}", flush=True)
    oof = pd.concat(parts, ignore_index=True).sort_values(KEY).reset_index(drop=True)
    if len(oof) != 7755 or oof[KEY].duplicated().any():
        raise RuntimeError(f"OOF population failed for {branch_id}: {len(oof)}")
    oof.to_parquet(OUTPUTS / branch_id / "oof.parquet", index=False)
    selection = pd.DataFrame(selections)
    selection.to_csv(OUTPUTS / branch_id / "fold_parameter_selections.csv", index=False, encoding="utf-8-sig")
    return oof, selection


def station_losses(frame: pd.DataFrame, mask: pd.Series | None = None) -> pd.Series:
    selected = frame if mask is None else frame.loc[mask].copy()
    selected["sq_log_error"] = (np.log(selected["predict"].clip(lower=EPS)) - np.log(selected["actual"].clip(lower=EPS))) ** 2
    return selected.groupby(selected["q_site"].astype(str))["sq_log_error"].mean()


def paired_bootstrap(candidate: pd.DataFrame, parent: pd.DataFrame, label: str, candidate_id: str) -> dict[str, object]:
    c = station_losses(candidate)
    p = station_losses(parent)
    common = c.index.intersection(p.index)
    c, p = c.loc[common].to_numpy(float), p.loc[common].to_numpy(float)
    point = float(np.sqrt(c.mean()) - np.sqrt(p.mean()))
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(common), size=(NBOOT, len(common)))
    delta = np.sqrt(c[idx].mean(axis=1)) - np.sqrt(p[idx].mean(axis=1))
    return {
        "candidate": candidate_id, "comparison": label, "stations": int(len(common)),
        "delta_station_macro_log_RMSE": point,
        "ci95_lower": float(np.quantile(delta, 0.025)),
        "ci95_upper": float(np.quantile(delta, 0.975)),
        "improved": bool(np.quantile(delta, 0.975) < 0.0),
        "noninferior_0p005": bool(np.quantile(delta, 0.975) < 0.005),
    }


def add_flow_regime(frame: pd.DataFrame) -> pd.DataFrame:
    source = pd.read_parquet(CORE.INPUT, columns=["q_site", "year", "Q_obsv_cfs"])
    source = source[source["Q_obsv_cfs"].notna() & source["Q_obsv_cfs"].gt(0)].copy()
    source["q_site"] = source["q_site"].astype(str)
    thresholds = []
    for fold in CORE.FOLDS:
        train = source[source["year"] <= int(fold["train_end"])]
        q = train.groupby("q_site")["Q_obsv_cfs"].quantile([0.2, 0.8]).unstack()
        q.columns = ["q20", "q80"]
        q["fold_id"] = str(fold["fold_id"])
        thresholds.append(q.reset_index())
    table = pd.concat(thresholds, ignore_index=True)
    out = frame.copy()
    out["q_site"] = out["q_site"].astype(str)
    out = out.merge(table, on=["q_site", "fold_id"], validate="many_to_one")
    out["flow_regime"] = np.where(out["actual"] <= out["q20"], "low", np.where(out["actual"] >= out["q80"], "high", "middle"))
    return out


def regime_summary(frame: pd.DataFrame, branch_id: str) -> list[dict[str, object]]:
    rows = []
    for regime in ["low", "middle", "high"]:
        part = frame[frame["flow_regime"] == regime]
        station = station_losses(part)
        rows.append({
            "branch_id": branch_id, "flow_regime": regime, "rows": int(len(part)),
            "stations": int(len(station)), "station_macro_log_RMSE": float(np.sqrt(station.mean())),
            "PBIAS_pct": float(100.0 * np.sum(part["predict"] - part["actual"]) / np.sum(part["actual"])),
        })
    return rows


def main() -> None:
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    parent = pd.read_parquet(STAGE2 / "outputs" / "H0_CORRECTED" / "oof.parquet")
    parent["q_site"] = parent["q_site"].astype(str)
    parent_regime = add_flow_regime(parent)
    candidate_frames = {}
    selections = []
    for branch_id in BRANCHES:
        frame, selection = run_branch(branch_id)
        frame["q_site"] = frame["q_site"].astype(str)
        candidate_frames[branch_id] = frame
        selections.append(selection)
    pd.concat(selections, ignore_index=True).to_csv(REPORTS / "all_fold_parameter_selections.csv", index=False, encoding="utf-8-sig")
    metric_rows = [{"branch_id": "H0_CORRECTED", **CORE.pooled_metrics(parent)}]
    bootstrap_rows = []
    regime_rows = regime_summary(parent_regime, "H0_CORRECTED")
    parent_station = CORE.station_metrics(parent)
    for branch_id, frame in candidate_frames.items():
        metric_rows.append({"branch_id": branch_id, **CORE.pooled_metrics(frame)})
        bootstrap_rows.append(paired_bootstrap(frame, parent, "vs_H0_CORRECTED", branch_id))
        regime_rows.extend(regime_summary(add_flow_regime(frame), branch_id))
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(REPORTS / "fixed_branch_pooled_metrics.csv", index=False, encoding="utf-8-sig")
    boot = pd.DataFrame(bootstrap_rows).sort_values("delta_station_macro_log_RMSE")
    boot.to_csv(REPORTS / "paired_station_bootstrap.csv", index=False, encoding="utf-8-sig")
    regimes = pd.DataFrame(regime_rows)
    regimes.to_csv(REPORTS / "flow_regime_metrics.csv", index=False, encoding="utf-8-sig")
    decisions = []
    parent_metrics = metrics.set_index("branch_id").loc["H0_CORRECTED"]
    parent_reg = regimes.set_index(["branch_id", "flow_regime"])
    for row in boot.itertuples(index=False):
        candidate_id = str(row.candidate)
        cm = metrics.set_index("branch_id").loc[candidate_id]
        cr = regimes.set_index(["branch_id", "flow_regime"])
        low_noninferior = float(cr.loc[(candidate_id, "low"), "station_macro_log_RMSE"] - parent_reg.loc[("H0_CORRECTED", "low"), "station_macro_log_RMSE"]) <= 0.005
        high_noninferior = float(cr.loc[(candidate_id, "high"), "station_macro_log_RMSE"] - parent_reg.loc[("H0_CORRECTED", "high"), "station_macro_log_RMSE"]) <= 0.005
        raw_guard = float(cm["raw_NSE"]) >= float(parent_metrics["raw_NSE"]) - 0.005
        kge_guard = float(cm["KGE_2012"]) >= float(parent_metrics["KGE_2012"]) - 0.005
        pbias_guard = abs(float(cm["PBIAS_pct"])) <= abs(float(parent_metrics["PBIAS_pct"])) + 3.0
        mass = pd.concat([candidate_frames[candidate_id][["production_mass_balance_error_mm"]]])["production_mass_balance_error_mm"].abs().max()
        gates = bool(row.improved and raw_guard and kge_guard and pbias_guard and low_noninferior and high_noninferior and mass <= 1e-8)
        decisions.append({
            "branch_id": candidate_id, "temporal_improved": bool(row.improved),
            "raw_NSE_guard": raw_guard, "KGE2012_guard": kge_guard, "PBIAS_guard": pbias_guard,
            "low_flow_guard": low_noninferior, "high_flow_guard": high_noninferior,
            "mass_closure_guard": bool(mass <= 1e-8), "all_upgrade_gates": gates,
        })
    decision = pd.DataFrame(decisions)
    decision.to_csv(REPORTS / "fixed_branch_decisions.csv", index=False, encoding="utf-8-sig")
    eligible = decision[decision["all_upgrade_gates"]]
    if len(eligible):
        ranked = boot[boot["candidate"].isin(eligible["branch_id"])].sort_values("delta_station_macro_log_RMSE")
        selected = str(ranked.iloc[0]["candidate"])
        state = "FIXED_PROCESS_BRANCH_SUPPORTED"
    else:
        selected = "H0_CORRECTED"
        state = "NO_FIXED_PROCESS_BRANCH_PASSED"
    gate = {
        "stage": "20260823_3", "status": "PASS_EXPERIMENT_COMPLETE",
        "scientific_decision": state, "selected_for_next_stage": selected,
        "eligible_branches": eligible["branch_id"].tolist(),
        "mainline_not_yet_upgraded": True,
        "next_authorized_stage": "20260823_4",
    }
    dump(REPORTS / "stage_gate.json", gate)
    program_path = TEST_ROOT / "20260823_1" / "program_manifest.json"
    program = json.loads(program_path.read_text(encoding="utf-8"))
    program["stages"]["20260823_3"]["status"] = "passed" if state == "FIXED_PROCESS_BRANCH_SUPPORTED" else "failed"
    program["stages"]["20260823_3"]["stage_gate"] = "../20260823_3/reports/stage_gate.json"
    program_path.write_text(json.dumps(program, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(gate, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

