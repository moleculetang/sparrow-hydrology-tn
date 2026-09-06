from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd


RUN = Path(__file__).resolve().parents[1]
TEST_ROOT = RUN.parent
Q72_OOF_RUN = TEST_ROOT / "20260728_32"
Q72_SOURCE = TEST_ROOT / "20260728_30"
Q78_SOURCE = TEST_ROOT / "20260728_18"
Q78_CLIMATE = TEST_ROOT / "20260608_1" / "reports" / "input_preprocessing" / "climate_monthly_used.csv"
OUT = RUN / "reports" / "f1_q72_q78_strict_oof"
EPS = 1.0e-6
SEED = 20260729
N_BOOT = 1000
EXCLUSIONS = {"劳村站", "富罗（二）站", "隆安站", "灵渠（三）站", "马口站"}
SHIJIAO = "石角站"
FOLDS = [
    ("fit_2006_2011_eval_2012_2013", 2011, 2012, 2013),
    ("fit_2006_2013_eval_2014_2015", 2013, 2014, 2015),
    ("fit_2006_2015_eval_2016_2018", 2015, 2016, 2018),
]
ALPHA = {
    "headwater": 0.100,
    "large_nonheadwater": 0.075,
    "mid_nonheadwater": 0.200,
    "reservoir_reach": 0.050,
    "small_nonheadwater": 0.200,
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def log_rmse(obs: np.ndarray, pred: np.ndarray) -> float:
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)
    ok = np.isfinite(obs) & np.isfinite(pred) & (obs > 0) & (pred > 0)
    if not ok.any():
        return float("nan")
    return float(np.sqrt(np.mean((np.log(pred[ok] + EPS) - np.log(obs[ok] + EPS)) ** 2)))


def pbias(obs: np.ndarray, pred: np.ndarray) -> float:
    denom = float(np.nansum(obs))
    return float(100.0 * np.nansum(pred - obs) / denom) if denom > EPS else float("nan")


def selected_oracle(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit one convex alpha per fold/class on the passed frame: theoretical upper bound only."""
    pieces: list[pd.DataFrame] = []
    rows: list[dict[str, object]] = []
    for (fold, cls), part in frame.groupby(["fold_id", "reach_class"], sort=True):
        work = part.copy()
        q72 = np.log(work["q72_interaction_lite_cfs"].to_numpy(dtype=float) + EPS)
        q78 = np.log(work["q78_mass_cfs"].to_numpy(dtype=float) + EPS)
        y = np.log(work["Q_obsv_cfs"].to_numpy(dtype=float) + EPS)
        d = q78 - q72
        denom = float(np.dot(d, d))
        alpha = 0.0 if denom <= EPS else float(np.clip(np.dot(d, y - q72) / denom, 0.0, 1.0))
        work["oracle_alpha"] = alpha
        work["oracle_cfs"] = np.exp(np.clip(q72 + alpha * d, -20.0, 20.0))
        rows.append({"fold_id": fold, "reach_class": cls, "rows": int(len(work)), "oracle_alpha": alpha})
        pieces.append(work)
    return pd.concat(pieces, ignore_index=True), pd.DataFrame(rows)


def bootstrap_gain(frame: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, object]] = []
    for cls, part in frame.groupby("reach_class", sort=True):
        sites = np.array(sorted(part["q_site"].astype(str).unique()))
        by_site = {site: part[part["q_site"].astype(str).eq(site)] for site in sites}
        if len(sites) == 0:
            continue
        gains = []
        for _ in range(N_BOOT):
            take = rng.choice(sites, size=len(sites), replace=True)
            sample = pd.concat([by_site[site] for site in take], ignore_index=True)
            a = log_rmse(sample["Q_obsv_cfs"].to_numpy(), sample["q72_interaction_lite_cfs"].to_numpy())
            b = log_rmse(sample["Q_obsv_cfs"].to_numpy(), sample["oracle_cfs"].to_numpy())
            gains.append(100.0 * (a - b) / max(a, EPS))
        rows.append({
            "reach_class": cls,
            "bootstrap_replicates": N_BOOT,
            "gain_pct_median": float(np.quantile(gains, 0.50)),
            "gain_pct_ci95_low": float(np.quantile(gains, 0.025)),
            "gain_pct_ci95_high": float(np.quantile(gains, 0.975)),
        })
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    q72_path = Q72_OOF_RUN / "reports" / "lowflow_oof_canonical_baseline" / "q72_interaction_lite_oof_predictions.csv"
    target_path = Q72_OOF_RUN / "reports" / "lowflow_oof_canonical_baseline" / "low_flow_signature_summary_canonical.csv"
    input_path = Q72_SOURCE / "inputs" / "indata.parquet"
    if not q72_path.exists() or not target_path.exists() or not input_path.exists():
        raise RuntimeError("Missing frozen Q72 OOF or development input")

    # Q72 canonical output contains only strict development OOF rows.
    q72 = pd.read_csv(q72_path, encoding="utf-8-sig")
    if int(q72["year"].max()) > 2018:
        raise RuntimeError("Canonical Q72 OOF contains protected confirmation years")
    targets = pd.read_csv(target_path, encoding="utf-8-sig")
    target_sites = set(targets.loc[targets["stable_nonreservoir_target"], "q_site"].astype(str))
    if len(target_sites) != 28:
        raise RuntimeError(f"Frozen target count changed: {len(target_sites)}")
    if not target_sites.isdisjoint(EXCLUSIONS):
        raise RuntimeError("An excluded station appears in frozen target sites")
    if SHIJIAO not in set(q72["q_site"].astype(str)):
        raise RuntimeError("石角站 is missing from Q72 OOF")

    # Pull only columns and development records from the Parquet reader. There is no broad post-2018 DataFrame.
    obs_cols = ["comid", "year", "month", "q_site", "Q_obsv_cfs"]
    observed = pd.read_parquet(input_path, columns=obs_cols, filters=[("year", "<=", 2018)])
    observed = observed[
        observed["year"].between(2006, 2018)
        & observed["month"].between(1, 12)
        & observed["Q_obsv_cfs"].gt(0)
    ].copy()
    observed["q_site"] = observed["q_site"].astype(str)
    observed["reach_id"] = pd.to_numeric(observed["comid"], errors="coerce").round().astype("Int64")
    if observed.empty or int(observed["year"].max()) != 2018:
        raise RuntimeError("Development-only observation filter failed")
    if observed["year"].gt(2018).any():
        raise RuntimeError("Protected observations entered Q78 fitting data")

    # Import frozen Q78 structure; only coefficients are refitted per outer fold.
    if not Q78_CLIMATE.exists():
        raise RuntimeError("Frozen Q78 reach-month climate source is unavailable")
    m77 = load_module("q78_class", Q78_SOURCE / "scripts" / "fit_reach_class_mass_model.py")
    map78 = load_module("q78_map", Q78_SOURCE / "scripts" / "fit_map_shrunk_mass_model.py")
    base = m77.BASE
    # The Q78 copy retained the old relative path but not its immutable reach-month climate input.
    # This is the exact input carried by the source 20260608_1 mass experiment, not a new forcing product.
    base.CLIMATE = Q78_CLIMATE
    topo = base.load_topology()
    order, warnings = base.topological_order(topo)
    panel = base.build_forcing_panel(topo)
    panel = panel[panel["year"].between(2006, 2018)].copy()
    if int(panel["year"].max()) != 2018:
        raise RuntimeError("Q78 forcing panel did not stop at development end")
    panel, class_counts = m77.assign_reach_classes(panel)
    basis_names = ["surplus", "threshold", "extreme", "memory", "slow_threshold_memory", "wet_surplus", "dry_buffer", "base_ppt"]
    class_names = sorted(panel["reach_class"].dropna().unique().tolist())
    feature_names = m77.build_class_basis(panel, basis_names, class_names)
    routed_basis = m77.build_routed_feature_basis(panel, topo, order, feature_names)
    obs_basis = observed.merge(routed_basis, on=["reach_id", "year", "month"], how="left", validate="many_to_one")
    if obs_basis["Q_obsv_cfs"].isna().any() or obs_basis["year"].gt(2018).any():
        raise RuntimeError("Q78 training observation basis audit failed")
    priors = map78.load_global_priors()
    lambda_prior = 0.1

    q78_rows: list[pd.DataFrame] = []
    fold_audit: list[dict[str, object]] = []
    coef_rows: list[pd.DataFrame] = []
    for fold_id, train_end, eval_start, eval_end in FOLDS:
        train_rows = obs_basis[obs_basis["year"].between(2010, train_end)].copy()
        if train_rows.empty:
            raise RuntimeError(f"Q78 has no usable training rows for {fold_id}")
        coef, coef_df = map78.fit_map_coefficients(obs_basis, feature_names, priors, (2010, train_end), lambda_prior)
        coef_df["fold_id"] = fold_id
        coef_rows.append(coef_df)
        routed = m77.route_with_class_coefficients(panel, topo, order, coef)
        part = routed[routed["year"].between(eval_start, eval_end)][["reach_id", "year", "month", "Q_out_cfs"]].copy()
        part["fold_id"] = fold_id
        part = part.rename(columns={"Q_out_cfs": "q78_mass_cfs"})
        q78_rows.append(part)
        fold_audit.append({
            "fold_id": fold_id,
            "q78_train_year_min": int(train_rows["year"].min()),
            "q78_train_year_max": int(train_rows["year"].max()),
            "q78_train_rows": int(len(train_rows)),
            "outer_eval_year_min": eval_start,
            "outer_eval_year_max": eval_end,
            "lambda_prior": lambda_prior,
            "q78_routed_eval_rows": int(len(part)),
        })
    q78 = pd.concat(q78_rows, ignore_index=True)
    coefficient_table = pd.concat(coef_rows, ignore_index=True)

    keys = ["fold_id", "reach_id", "year", "month"]
    joined = q72.merge(q78, on=keys, how="left", validate="many_to_one")
    joined["q_site"] = joined["q_site"].astype(str)
    joined = joined[joined["q_site"].isin(target_sites)].copy()
    joined = joined[~joined["q_site"].isin(EXCLUSIONS)].copy()
    joined = joined[joined["is_low_flow"].astype(bool)].copy()
    joined = joined.dropna(subset=["q72_interaction_lite_cfs", "q78_mass_cfs", "Q_obsv_cfs", "reach_class"])
    joined = joined[(joined["q72_interaction_lite_cfs"] > 0) & (joined["q78_mass_cfs"] > 0) & (joined["Q_obsv_cfs"] > 0)].copy()
    if joined.empty:
        raise RuntimeError("No comparable target low-flow rows remain")
    joined["alpha_static"] = joined["reach_class"].map(ALPHA)
    if joined["alpha_static"].isna().any():
        raise RuntimeError("Unknown canonical reach class in F1 joined data")
    joined["qmain_static_cfs"] = np.exp(
        (1.0 - joined["alpha_static"].to_numpy()) * np.log(joined["q72_interaction_lite_cfs"].to_numpy() + EPS)
        + joined["alpha_static"].to_numpy() * np.log(joined["q78_mass_cfs"].to_numpy() + EPS)
    )

    coverage = (
        joined.groupby(["fold_id", "reach_class"], as_index=False)
        .agg(rows=("q_site", "size"), target_stations=("q_site", "nunique"))
    )
    all_target_fold = (
        q72[q72["q_site"].astype(str).isin(target_sites) & q72["is_low_flow"].astype(bool)]
        .groupby("fold_id", as_index=False)
        .agg(expected_target_low_rows=("q_site", "size"))
    )
    matched_target_fold = joined.groupby("fold_id", as_index=False).agg(matched_target_low_rows=("q_site", "size"))
    total_coverage = all_target_fold.merge(matched_target_fold, on="fold_id", how="left")
    total_coverage["matched_target_low_rows"] = total_coverage["matched_target_low_rows"].fillna(0).astype(int)
    total_coverage["coverage_fraction"] = total_coverage["matched_target_low_rows"] / total_coverage["expected_target_low_rows"].clip(lower=1)
    coverage = coverage.merge(
        total_coverage[["fold_id", "coverage_fraction"]], on="fold_id", how="left"
    )
    coverage["class_authorized_for_f1"] = coverage["target_stations"].ge(5) & coverage["rows"].ge(24) & coverage["coverage_fraction"].ge(0.80)
    authorized_classes = sorted(coverage.loc[coverage["class_authorized_for_f1"], "reach_class"].unique().tolist())
    if not authorized_classes:
        authorized = joined.iloc[0:0].copy()
    else:
        # A class needs coverage in every outer fold before it can shape an F3 candidate.
        valid_classes = (
            coverage.groupby("reach_class")["class_authorized_for_f1"].all().loc[lambda s: s].index.tolist()
        )
        authorized = joined[joined["reach_class"].isin(valid_classes)].copy()

    oracle_all, alpha_all = selected_oracle(joined)
    oracle_auth, alpha_auth = selected_oracle(authorized) if not authorized.empty else (authorized.copy(), pd.DataFrame())
    oracle_all["evaluation_scope"] = "all_target_classes_theoretical"
    oracle_auth["evaluation_scope"] = "coverage_authorized_classes_gate"
    scored = pd.concat([oracle_all, oracle_auth], ignore_index=True)
    alpha_table = pd.concat([
        alpha_all.assign(evaluation_scope="all_target_classes_theoretical"),
        alpha_auth.assign(evaluation_scope="coverage_authorized_classes_gate"),
    ], ignore_index=True)

    metric_rows: list[dict[str, object]] = []
    station_rows: list[dict[str, object]] = []
    for scope, scope_frame in scored.groupby("evaluation_scope", sort=True):
        for fold, part in scope_frame.groupby("fold_id", sort=True):
            q72_rmse = log_rmse(part["Q_obsv_cfs"].to_numpy(), part["q72_interaction_lite_cfs"].to_numpy())
            q78_rmse = log_rmse(part["Q_obsv_cfs"].to_numpy(), part["q78_mass_cfs"].to_numpy())
            main_rmse = log_rmse(part["Q_obsv_cfs"].to_numpy(), part["qmain_static_cfs"].to_numpy())
            oracle_rmse = log_rmse(part["Q_obsv_cfs"].to_numpy(), part["oracle_cfs"].to_numpy())
            e72 = np.log(part["q72_interaction_lite_cfs"].to_numpy() + EPS) - np.log(part["Q_obsv_cfs"].to_numpy() + EPS)
            e78 = np.log(part["q78_mass_cfs"].to_numpy() + EPS) - np.log(part["Q_obsv_cfs"].to_numpy() + EPS)
            metric_rows.append({
                "evaluation_scope": scope, "fold_id": fold, "rows": int(len(part)), "stations": int(part["q_site"].nunique()),
                "q72_log_RMSE": q72_rmse, "q78_log_RMSE": q78_rmse, "qmain_static_log_RMSE": main_rmse,
                "oracle_log_RMSE": oracle_rmse,
                "oracle_gain_vs_q72_pct": 100.0 * (q72_rmse - oracle_rmse) / max(q72_rmse, EPS),
                "current_static_gain_vs_q72_pct": 100.0 * (q72_rmse - main_rmse) / max(q72_rmse, EPS),
                "W78": float(np.mean(np.abs(e78) < np.abs(e72))),
                "C_sign": float(np.mean(e72 * e78 < 0.0)),
                "q72_low_PBIAS_pct": pbias(part["Q_obsv_cfs"].to_numpy(), part["q72_interaction_lite_cfs"].to_numpy()),
                "oracle_low_PBIAS_pct": pbias(part["Q_obsv_cfs"].to_numpy(), part["oracle_cfs"].to_numpy()),
            })
        for (fold, site), part in scope_frame.groupby(["fold_id", "q_site"], sort=True):
            q72_rmse = log_rmse(part["Q_obsv_cfs"].to_numpy(), part["q72_interaction_lite_cfs"].to_numpy())
            oracle_rmse = log_rmse(part["Q_obsv_cfs"].to_numpy(), part["oracle_cfs"].to_numpy())
            station_rows.append({
                "evaluation_scope": scope, "fold_id": fold, "q_site": site, "reach_class": str(part["reach_class"].iloc[0]),
                "rows": int(len(part)), "q72_log_RMSE": q72_rmse, "oracle_log_RMSE": oracle_rmse,
                "oracle_gain_vs_q72_pct": 100.0 * (q72_rmse - oracle_rmse) / max(q72_rmse, EPS),
                "q72_low_PBIAS_pct": pbias(part["Q_obsv_cfs"].to_numpy(), part["q72_interaction_lite_cfs"].to_numpy()),
                "oracle_low_PBIAS_pct": pbias(part["Q_obsv_cfs"].to_numpy(), part["oracle_cfs"].to_numpy()),
            })
    metrics = pd.DataFrame(metric_rows)
    stations = pd.DataFrame(station_rows)
    pooled_rows: list[dict[str, object]] = []
    for scope, part in scored.groupby("evaluation_scope", sort=True):
        q72_rmse = log_rmse(part["Q_obsv_cfs"].to_numpy(), part["q72_interaction_lite_cfs"].to_numpy())
        oracle_rmse = log_rmse(part["Q_obsv_cfs"].to_numpy(), part["oracle_cfs"].to_numpy())
        station_part = stations[stations["evaluation_scope"].eq(scope)]
        pooled_rows.append({
            "evaluation_scope": scope, "rows": int(len(part)), "stations": int(part["q_site"].nunique()),
            "q72_log_RMSE": q72_rmse, "oracle_log_RMSE": oracle_rmse,
            "oracle_gain_vs_q72_pct": 100.0 * (q72_rmse - oracle_rmse) / max(q72_rmse, EPS),
            "station_fold_improvement_fraction": float(np.mean(station_part["oracle_gain_vs_q72_pct"].to_numpy() > 0.0)),
            "W78": float(np.mean(np.abs(np.log(part["q78_mass_cfs"].to_numpy() + EPS) - np.log(part["Q_obsv_cfs"].to_numpy() + EPS)) < np.abs(np.log(part["q72_interaction_lite_cfs"].to_numpy() + EPS) - np.log(part["Q_obsv_cfs"].to_numpy() + EPS)))),
            "C_sign": float(np.mean((np.log(part["q72_interaction_lite_cfs"].to_numpy() + EPS) - np.log(part["Q_obsv_cfs"].to_numpy() + EPS)) * (np.log(part["q78_mass_cfs"].to_numpy() + EPS) - np.log(part["Q_obsv_cfs"].to_numpy() + EPS)) < 0.0)),
        })
    pooled = pd.DataFrame(pooled_rows)
    bootstrap = bootstrap_gain(oracle_auth) if not oracle_auth.empty else pd.DataFrame(columns=["reach_class", "bootstrap_replicates", "gain_pct_median", "gain_pct_ci95_low", "gain_pct_ci95_high"])
    if not oracle_auth.empty:
        pooled_boot = bootstrap_gain(oracle_auth.assign(reach_class="_pooled_authorized_"))
        bootstrap = pd.concat([bootstrap, pooled_boot], ignore_index=True)

    gate_scope = pooled[pooled["evaluation_scope"].eq("coverage_authorized_classes_gate")]
    fold_gate = metrics[metrics["evaluation_scope"].eq("coverage_authorized_classes_gate")]
    station_gate = stations[stations["evaluation_scope"].eq("coverage_authorized_classes_gate")]
    pooled_gain = float(gate_scope["oracle_gain_vs_q72_pct"].iloc[0]) if len(gate_scope) == 1 else float("nan")
    folds_gain_ge_5 = int((fold_gate["oracle_gain_vs_q72_pct"] >= 5.0).sum())
    station_fraction = float(np.mean(station_gate["oracle_gain_vs_q72_pct"].to_numpy() > 0.0)) if not station_gate.empty else float("nan")
    w78 = float(gate_scope["W78"].iloc[0]) if len(gate_scope) == 1 else float("nan")
    csign = float(gate_scope["C_sign"].iloc[0]) if len(gate_scope) == 1 else float("nan")
    pooled_ci = bootstrap.loc[bootstrap["reach_class"].eq("_pooled_authorized_"), "gain_pct_ci95_low"]
    ci_low = float(pooled_ci.iloc[0]) if len(pooled_ci) == 1 else float("nan")
    coverage_pass = bool(total_coverage["coverage_fraction"].ge(0.90).all() and not authorized.empty)
    f1_pass = bool(
        coverage_pass
        and pooled_gain >= 5.0
        and folds_gain_ge_5 >= 2
        and station_fraction >= 0.60
        and (csign >= 0.20 or w78 >= 0.60)
        and ci_low >= 0.0
    )

    input_audit = pd.DataFrame([{
        "q72_oof_sha256": sha256(q72_path), "q72_target_summary_sha256": sha256(target_path), "development_input_sha256": sha256(input_path), "q78_climate_sha256": sha256(Q78_CLIMATE),
        "q72_oof_max_year": int(q72["year"].max()), "q78_observed_max_year": int(observed["year"].max()),
        "q78_panel_max_year": int(panel["year"].max()), "target_station_count": int(len(target_sites)),
        "q78_lambda_prior_frozen": lambda_prior, "topological_warnings": "|".join(warnings),
        "python": sys.version.replace("\n", " "), "platform": platform.platform(),
    }])
    q78_coefficients = coefficient_table
    coverage.to_csv(OUT / "coverage_by_fold_class.csv", index=False, encoding="utf-8-sig")
    total_coverage.to_csv(OUT / "coverage_total_by_fold.csv", index=False, encoding="utf-8-sig")
    q78_coefficients.to_csv(OUT / "q78_fold_coefficients.csv", index=False, encoding="utf-8-sig")
    class_counts.to_csv(OUT / "q78_reach_class_counts.csv", index=False, encoding="utf-8-sig")
    scored.to_csv(OUT / "q72_q78_lowflow_oof_scored.csv", index=False, encoding="utf-8-sig")
    alpha_table.to_csv(OUT / "oracle_alpha_by_fold_class.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(OUT / "f1_metrics_by_fold.csv", index=False, encoding="utf-8-sig")
    stations.to_csv(OUT / "f1_station_metrics.csv", index=False, encoding="utf-8-sig")
    pooled.to_csv(OUT / "f1_pooled_metrics.csv", index=False, encoding="utf-8-sig")
    bootstrap.to_csv(OUT / "f1_station_block_bootstrap.csv", index=False, encoding="utf-8-sig")
    input_audit.to_csv(OUT / "input_and_environment_audit.csv", index=False, encoding="utf-8-sig")

    payload = {
        "run_id": RUN.name,
        "stage": "F1_q72_q78_lowflow_complementarity",
        "used_year_max": int(max(q72["year"].max(), observed["year"].max(), panel["year"].max())),
        "confirmation_years_used": False,
        "target_station_count": int(len(target_sites)),
        "excluded_station_rows_in_primary": {name: int((joined["q_site"] == name).sum()) for name in sorted(EXCLUSIONS)},
        "shijiao_present_in_q72_oof": bool(SHIJIAO in set(q72["q_site"].astype(str))),
        "q78_lambda_prior": lambda_prior,
        "coverage_pass": coverage_pass,
        "coverage_authorized_classes": sorted(authorized["reach_class"].unique().tolist()) if not authorized.empty else [],
        "pooled_oracle_gain_pct": pooled_gain,
        "folds_oracle_gain_ge_5pct": folds_gain_ge_5,
        "station_fold_improvement_fraction": station_fraction,
        "W78": w78,
        "C_sign": csign,
        "bootstrap_pooled_gain_ci95_low_pct": ci_low,
        "f1_pass": f1_pass,
        "next_required_stage": "F3_conditional_fusion" if f1_pass else "F2_map_objective_dilution_diagnostic",
    }
    engineering_pass = bool(
        payload["used_year_max"] == 2018
        and not payload["confirmation_years_used"]
        and payload["target_station_count"] == 28
        and all(v == 0 for v in payload["excluded_station_rows_in_primary"].values())
        and payload["shijiao_present_in_q72_oof"]
    )
    payload["engineering_pass"] = engineering_pass
    payload["scientific_status"] = "PASS_F1" if engineering_pass and f1_pass else ("FAIL_F1" if engineering_pass else "FAIL_ENGINEERING")
    (OUT / "gate.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# F1 strict Q72/Q78 low-flow complementarity gate",
        "",
        f"- Engineering gate: {'PASS' if engineering_pass else 'FAIL'}",
        f"- Primary scientific gate: {'PASS → F3' if f1_pass else 'FAIL → F2'}",
        f"- Authorized reach classes: {payload['coverage_authorized_classes']}",
        f"- Pooled oracle gain: {pooled_gain:.2f}%",
        f"- Folds with ≥5% oracle gain: {folds_gain_ge_5}/3",
        f"- Improved target station-fold fraction: {station_fraction:.1%}",
        f"- W78: {w78:.3f}; C_sign: {csign:.3f}",
        f"- Station-block bootstrap 95% lower gain bound: {ci_low:.2f}%",
        "- Maximum used year: 2018; 2019-2022 used: False",
    ]
    (OUT / "gate.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    readme = "\n".join([
        "# 20260729_2 — F1 strict Q72/Q78 audit",
        "",
        "This folder runs the first permitted no-LSTM low-flow test: strict temporal Q72/Q78 OOF complementarity. The production model is unchanged.",
        "",
        "## Result",
        "",
        *lines[2:],
        "",
        "Oracle alpha is diagnostic only. The machine-readable decision is `reports/f1_q72_q78_strict_oof/gate.json`.",
    ]) + "\n"
    (RUN / "README.md").write_text(readme, encoding="utf-8")
    print("\n".join(lines))
    if not engineering_pass:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
