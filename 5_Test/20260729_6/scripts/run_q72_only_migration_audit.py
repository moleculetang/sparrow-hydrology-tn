from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import sys
from pathlib import Path

# Must be process-scoped and set before numerical libraries are imported.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd


RUN = Path(__file__).resolve().parents[1]
TEST_ROOT = RUN.parent
OUT = RUN / "reports" / "q72_only_migration_audit"
Q72_RUN = TEST_ROOT / "20260728_32"
Q72_SOURCE = TEST_ROOT / "20260728_30"
Q78_SOURCE = TEST_ROOT / "20260728_18"
Q78_COEFFICIENTS = (
    TEST_ROOT
    / "20260729_3"
    / "reports"
    / "f1_q72_q78_strict_oof"
    / "q78_fold_coefficients.csv"
)
Q78_CLIMATE = (
    TEST_ROOT
    / "20260608_1"
    / "reports"
    / "input_preprocessing"
    / "climate_monthly_used.csv"
)
Q72_OOF = (
    Q72_RUN
    / "reports"
    / "lowflow_oof_canonical_baseline"
    / "q72_interaction_lite_oof_predictions.csv"
)
LOWFLOW_TARGETS = (
    Q72_RUN
    / "reports"
    / "lowflow_oof_canonical_baseline"
    / "low_flow_signature_summary_canonical.csv"
)
DEVELOPMENT_INPUT = Q72_SOURCE / "inputs" / "indata.parquet"

EPS = 1.0e-6
EXCLUSIONS = {"劳村站", "富罗（二）站", "隆安站", "灵渠（三）站", "马口站"}
SHIJIAO = "石角站"
FOLDS = [
    ("fit_2006_2011_eval_2012_2013", 2012, 2013),
    ("fit_2006_2013_eval_2014_2015", 2014, 2015),
    ("fit_2006_2015_eval_2016_2018", 2016, 2018),
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
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def metric_dict(obs: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)
    ok = np.isfinite(obs) & np.isfinite(pred) & (obs > 0) & (pred > 0)
    obs = obs[ok]
    pred = pred[ok]
    if len(obs) < 3:
        return {
            "n": int(len(obs)),
            "NSE_raw": np.nan,
            "NSE_log": np.nan,
            "KGE_2012": np.nan,
            "KGE_r": np.nan,
            "KGE_beta": np.nan,
            "KGE_gamma": np.nan,
            "PBIAS_pct": np.nan,
        }
    denom = float(np.sum((obs - np.mean(obs)) ** 2))
    nse_raw = np.nan if denom <= 0 else float(
        1.0 - np.sum((obs - pred) ** 2) / denom
    )
    lo = np.log(obs + EPS)
    lp = np.log(pred + EPS)
    denom_log = float(np.sum((lo - np.mean(lo)) ** 2))
    nse_log = np.nan if denom_log <= 0 else float(
        1.0 - np.sum((lo - lp) ** 2) / denom_log
    )
    if np.std(obs) <= 0 or np.std(pred) <= 0:
        r = beta = gamma = kge = np.nan
    else:
        r = float(np.corrcoef(obs, pred)[0, 1])
        beta = float(np.mean(pred) / np.mean(obs))
        cv_obs = float(np.std(obs) / np.mean(obs))
        cv_pred = float(np.std(pred) / np.mean(pred))
        gamma = float(cv_pred / cv_obs) if cv_obs > 0 else np.nan
        kge = float(
            1.0
            - np.sqrt(
                (r - 1.0) ** 2
                + (beta - 1.0) ** 2
                + (gamma - 1.0) ** 2
            )
        )
    pbias = float(100.0 * np.sum(pred - obs) / np.sum(obs))
    return {
        "n": int(len(obs)),
        "NSE_raw": nse_raw,
        "NSE_log": nse_log,
        "KGE_2012": kge,
        "KGE_r": r,
        "KGE_beta": beta,
        "KGE_gamma": gamma,
        "PBIAS_pct": pbias,
    }


def log_rmse(obs: np.ndarray, pred: np.ndarray) -> float:
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)
    ok = np.isfinite(obs) & np.isfinite(pred) & (obs > 0) & (pred > 0)
    if not ok.any():
        return float("nan")
    return float(
        np.sqrt(
            np.mean(
                (
                    np.log(pred[ok] + EPS)
                    - np.log(obs[ok] + EPS)
                )
                ** 2
            )
        )
    )


def pbias(obs: np.ndarray, pred: np.ndarray) -> float:
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)
    denom = float(np.nansum(obs))
    return (
        float(100.0 * np.nansum(pred - obs) / denom)
        if denom > EPS
        else float("nan")
    )


def is_good(metrics: dict[str, float]) -> bool:
    return bool(
        metrics["n"] >= 24
        and metrics["NSE_log"] >= 0.65
        and metrics["KGE_2012"] >= 0.50
        and abs(metrics["PBIAS_pct"]) <= 25.0
    )


def build_q78_oof() -> tuple[pd.DataFrame, list[str]]:
    m77 = load_module(
        "q78_reach_class",
        Q78_SOURCE / "scripts" / "fit_reach_class_mass_model.py",
    )
    base = m77.BASE
    base.CLIMATE = Q78_CLIMATE
    topo = base.load_topology()
    order, warnings = base.topological_order(topo)
    panel = base.build_forcing_panel(topo)
    panel = panel[panel["year"].between(2006, 2018)].copy()
    panel, _ = m77.assign_reach_classes(panel)
    basis_names = [
        "surplus",
        "threshold",
        "extreme",
        "memory",
        "slow_threshold_memory",
        "wet_surplus",
        "dry_buffer",
        "base_ppt",
    ]
    class_names = sorted(panel["reach_class"].dropna().unique().tolist())
    feature_names = m77.build_class_basis(panel, basis_names, class_names)

    coefficient_table = pd.read_csv(Q78_COEFFICIENTS, encoding="utf-8-sig")
    coefficient_table = coefficient_table[
        coefficient_table["feature_name"].isin(feature_names)
    ].copy()
    pieces: list[pd.DataFrame] = []
    for fold_id, eval_start, eval_end in FOLDS:
        part_coef = coefficient_table[
            coefficient_table["fold_id"].eq(fold_id)
        ].copy()
        coef = {
            str(row.feature_name): float(row.coefficient)
            for row in part_coef.itertuples(index=False)
        }
        if set(coef) != set(feature_names):
            missing = sorted(set(feature_names) - set(coef))
            extra = sorted(set(coef) - set(feature_names))
            raise RuntimeError(
                f"Q78 coefficient mismatch for {fold_id}: "
                f"missing={missing}, extra={extra}"
            )
        routed = m77.route_with_class_coefficients(panel, topo, order, coef)
        part = routed[
            routed["year"].between(eval_start, eval_end)
        ][["reach_id", "year", "month", "Q_out_cfs"]].copy()
        part["fold_id"] = fold_id
        pieces.append(part.rename(columns={"Q_out_cfs": "q78_mass_cfs"}))
    q78 = pd.concat(pieces, ignore_index=True)
    if q78["year"].max() > 2018:
        raise RuntimeError("Protected confirmation year entered Q78 OOF")
    return q78, warnings


def station_variant_metrics(joined: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    variants = {
        "frozen_fusion": "qmain_frozen_cfs",
        "q72_only": "q72_interaction_lite_cfs",
    }
    for (fold_id, site), part in joined.groupby(
        ["fold_id", "q_site"], sort=True
    ):
        for variant, column in variants.items():
            metrics = metric_dict(
                part["Q_obsv_cfs"].to_numpy(),
                part[column].to_numpy(),
            )
            rows.append(
                {
                    "fold_id": fold_id,
                    "q_site": site,
                    "reach_id": int(part["reach_id"].iloc[0]),
                    "reach_class": str(part["reach_class"].iloc[0]),
                    "is_reservoir_related": bool(
                        float(part["is_reservoir_reach"].iloc[0]) > 0
                        or float(part["downstream_reservoir"].iloc[0]) > 0
                    ),
                    "variant": variant,
                    **metrics,
                    "abs_PBIAS": abs(metrics["PBIAS_pct"]),
                    "good": is_good(metrics),
                    "severe_PBIAS": abs(metrics["PBIAS_pct"]) > 50.0,
                }
            )
    return pd.DataFrame(rows)


def compare_station_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    index = [
        "fold_id",
        "q_site",
        "reach_id",
        "reach_class",
        "is_reservoir_related",
    ]
    value_cols = [
        "n",
        "NSE_raw",
        "NSE_log",
        "KGE_2012",
        "KGE_r",
        "KGE_beta",
        "KGE_gamma",
        "PBIAS_pct",
        "abs_PBIAS",
        "good",
        "severe_PBIAS",
    ]
    wide = metrics.pivot(
        index=index, columns="variant", values=value_cols
    ).reset_index()
    wide.columns = [
        "_".join(str(part) for part in col if str(part))
        if isinstance(col, tuple)
        else str(col)
        for col in wide.columns
    ]
    for metric in [
        "NSE_raw",
        "NSE_log",
        "KGE_2012",
        "KGE_r",
        "KGE_beta",
        "KGE_gamma",
    ]:
        wide[f"delta_{metric}_q72_minus_fusion"] = (
            wide[f"{metric}_q72_only"]
            - wide[f"{metric}_frozen_fusion"]
        )
    wide["delta_abs_PBIAS_q72_minus_fusion"] = (
        wide["abs_PBIAS_q72_only"]
        - wide["abs_PBIAS_frozen_fusion"]
    )
    wide["lost_good"] = (
        wide["good_frozen_fusion"].astype(bool)
        & ~wide["good_q72_only"].astype(bool)
    )
    wide["gained_good"] = (
        ~wide["good_frozen_fusion"].astype(bool)
        & wide["good_q72_only"].astype(bool)
    )
    wide["new_severe_PBIAS"] = (
        ~wide["severe_PBIAS_frozen_fusion"].astype(bool)
        & wide["severe_PBIAS_q72_only"].astype(bool)
    )
    return wide


def lowflow_comparison(
    joined: pd.DataFrame, target_sites: set[str]
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    target = joined[
        joined["q_site"].isin(target_sites)
        & joined["is_low_flow"].astype(bool)
    ].copy()
    for (fold_id, site), part in target.groupby(
        ["fold_id", "q_site"], sort=True
    ):
        obs = part["Q_obsv_cfs"].to_numpy()
        q72 = part["q72_interaction_lite_cfs"].to_numpy()
        fusion = part["qmain_frozen_cfs"].to_numpy()
        q72_rmse = log_rmse(obs, q72)
        fusion_rmse = log_rmse(obs, fusion)
        q72_pbias = pbias(obs, q72)
        fusion_pbias = pbias(obs, fusion)
        rows.append(
            {
                "fold_id": fold_id,
                "q_site": site,
                "reach_class": str(part["reach_class"].iloc[0]),
                "lowflow_rows": int(len(part)),
                "q72_low_log_RMSE": q72_rmse,
                "fusion_low_log_RMSE": fusion_rmse,
                "delta_low_log_RMSE_q72_minus_fusion": (
                    q72_rmse - fusion_rmse
                ),
                "q72_low_PBIAS_pct": q72_pbias,
                "fusion_low_PBIAS_pct": fusion_pbias,
                "delta_low_abs_PBIAS_q72_minus_fusion": (
                    abs(q72_pbias) - abs(fusion_pbias)
                ),
            }
        )
    return pd.DataFrame(rows)


def build_fold_gate(
    station_compare: pd.DataFrame,
    lowflow: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stone = station_compare[
        station_compare["q_site"].eq(SHIJIAO)
    ].copy()
    stone["shijiao_pass"] = (
        stone["delta_NSE_log_q72_minus_fusion"].ge(-0.020)
        & stone["delta_KGE_2012_q72_minus_fusion"].ge(-0.030)
        & stone["delta_abs_PBIAS_q72_minus_fusion"].le(5.0)
    )
    rows: list[dict[str, object]] = []
    for fold_id, _, _ in FOLDS:
        all_part = station_compare[
            station_compare["fold_id"].eq(fold_id)
        ].copy()
        low_part = lowflow[lowflow["fold_id"].eq(fold_id)].copy()
        stone_part = stone[stone["fold_id"].eq(fold_id)]
        row = {
            "fold_id": fold_id,
            "station_count": int(all_part["q_site"].nunique()),
            "median_delta_NSElog": float(
                all_part["delta_NSE_log_q72_minus_fusion"].median()
            ),
            "median_delta_KGE2012": float(
                all_part["delta_KGE_2012_q72_minus_fusion"].median()
            ),
            "median_delta_absPBIAS_pp": float(
                all_part["delta_abs_PBIAS_q72_minus_fusion"].median()
            ),
            "lost_good_count": int(all_part["lost_good"].sum()),
            "gained_good_count": int(all_part["gained_good"].sum()),
            "new_severe_PBIAS_count": int(
                all_part["new_severe_PBIAS"].sum()
            ),
            "lowflow_target_station_count": int(
                low_part["q_site"].nunique()
            ),
            "median_delta_low_log_RMSE": float(
                low_part[
                    "delta_low_log_RMSE_q72_minus_fusion"
                ].median()
            ),
            "median_delta_low_absPBIAS_pp": float(
                low_part[
                    "delta_low_abs_PBIAS_q72_minus_fusion"
                ].median()
            ),
            "shijiao_present": bool(len(stone_part) == 1),
            "shijiao_pass": bool(
                len(stone_part) == 1
                and stone_part["shijiao_pass"].iloc[0]
            ),
        }
        row["overall_skill_pass"] = bool(
            row["median_delta_NSElog"] >= -0.005
            and row["median_delta_KGE2012"] >= -0.010
            and row["median_delta_absPBIAS_pp"] <= 1.0
            and row["lost_good_count"] <= 1
            and row["new_severe_PBIAS_count"] == 0
        )
        row["lowflow_guardrail_pass"] = bool(
            row["lowflow_target_station_count"] > 0
            and row["median_delta_low_log_RMSE"] <= 0.020
            and row["median_delta_low_absPBIAS_pp"] <= 5.0
        )
        row["fold_pass"] = bool(
            row["overall_skill_pass"]
            and row["lowflow_guardrail_pass"]
            and row["shijiao_pass"]
        )
        rows.append(row)
    return pd.DataFrame(rows), stone


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    required = [
        Q72_OOF,
        LOWFLOW_TARGETS,
        DEVELOPMENT_INPUT,
        Q78_COEFFICIENTS,
        Q78_CLIMATE,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing frozen inputs: {missing}")

    q72 = pd.read_csv(Q72_OOF, encoding="utf-8-sig")
    q72["q_site"] = q72["q_site"].astype(str)
    q72["reach_id"] = (
        pd.to_numeric(q72["reach_id"], errors="raise")
        .round()
        .astype(int)
    )
    if int(q72["year"].max()) > 2018:
        raise RuntimeError("Protected years entered Q72 OOF")
    q72 = q72[~q72["q_site"].isin(EXCLUSIONS)].copy()
    if any((q72["q_site"] == site).any() for site in EXCLUSIONS):
        raise RuntimeError("Fixed exclusion failed")
    if not (q72["q_site"] == SHIJIAO).any():
        raise RuntimeError("石角站 is absent")
    if q72.duplicated(
        ["fold_id", "q_site", "reach_id", "year", "month"]
    ).any():
        raise RuntimeError("Q72 OOF key is not unique")

    target_table = pd.read_csv(
        LOWFLOW_TARGETS, encoding="utf-8-sig"
    )
    target_sites = set(
        target_table.loc[
            target_table["stable_nonreservoir_target"].astype(bool),
            "q_site",
        ].astype(str)
    )
    if len(target_sites) != 28:
        raise RuntimeError(
            f"Frozen low-flow target count changed: {len(target_sites)}"
        )
    if not target_sites.isdisjoint(EXCLUSIONS):
        raise RuntimeError("Excluded site entered low-flow target set")

    q78, topology_warnings = build_q78_oof()
    keys = ["fold_id", "reach_id", "year", "month"]
    joined = q72.merge(q78, on=keys, how="left", validate="many_to_one")
    joined = joined.dropna(
        subset=[
            "Q_obsv_cfs",
            "q72_interaction_lite_cfs",
            "q78_mass_cfs",
            "reach_class",
        ]
    ).copy()
    joined = joined[
        joined["Q_obsv_cfs"].gt(0)
        & joined["q72_interaction_lite_cfs"].gt(0)
        & joined["q78_mass_cfs"].gt(0)
    ].copy()
    joined["alpha_frozen"] = joined["reach_class"].map(ALPHA)
    if joined["alpha_frozen"].isna().any():
        classes = sorted(
            joined.loc[
                joined["alpha_frozen"].isna(), "reach_class"
            ].astype(str).unique()
        )
        raise RuntimeError(f"Unknown reach classes: {classes}")
    joined["qmain_frozen_cfs"] = np.exp(
        (1.0 - joined["alpha_frozen"].to_numpy())
        * np.log(joined["q72_interaction_lite_cfs"].to_numpy() + EPS)
        + joined["alpha_frozen"].to_numpy()
        * np.log(joined["q78_mass_cfs"].to_numpy() + EPS)
    )

    expected_rows = len(q72)
    coverage_fraction = len(joined) / max(expected_rows, 1)
    station_metrics = station_variant_metrics(joined)
    station_compare = compare_station_metrics(station_metrics)
    lowflow = lowflow_comparison(joined, target_sites)
    fold_gate, stone = build_fold_gate(station_compare, lowflow)

    migration_pass = bool(
        coverage_fraction >= 0.99
        and len(fold_gate) == 3
        and fold_gate["fold_pass"].all()
    )
    used_year_max = int(joined["year"].max())
    engineering_pass = bool(
        used_year_max == 2018
        and int(joined["year"].min()) == 2012
        and coverage_fraction >= 0.99
        and joined["q_site"].nunique() == q72["q_site"].nunique()
        and all(
            int((joined["q_site"] == name).sum()) == 0
            for name in EXCLUSIONS
        )
        and SHIJIAO in set(joined["q_site"])
        and len(target_sites) == 28
    )

    input_audit = pd.DataFrame(
        [
            {
                "q72_oof_sha256": sha256(Q72_OOF),
                "lowflow_targets_sha256": sha256(LOWFLOW_TARGETS),
                "q78_coefficients_sha256": sha256(
                    Q78_COEFFICIENTS
                ),
                "q78_climate_sha256": sha256(Q78_CLIMATE),
                "development_input_sha256": sha256(
                    DEVELOPMENT_INPUT
                ),
                "q72_expected_rows": expected_rows,
                "joined_rows": int(len(joined)),
                "coverage_fraction": coverage_fraction,
                "eligible_station_count": int(
                    joined["q_site"].nunique()
                ),
                "lowflow_target_count": len(target_sites),
                "used_year_min": int(joined["year"].min()),
                "used_year_max": used_year_max,
                "confirmation_years_used": False,
                "fixed_exclusions": "|".join(sorted(EXCLUSIONS)),
                "shijiao_present": SHIJIAO in set(joined["q_site"]),
                "topology_warnings": "|".join(topology_warnings),
                "python": sys.version.replace("\n", " "),
                "platform": platform.platform(),
                "conda_prefix": os.environ.get("CONDA_PREFIX", ""),
                "kmp_duplicate_lib_ok": os.environ.get(
                    "KMP_DUPLICATE_LIB_OK", ""
                ),
            }
        ]
    )

    joined.to_csv(
        OUT / "oof_predictions_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    station_metrics.to_csv(
        OUT / "station_metrics_long.csv",
        index=False,
        encoding="utf-8-sig",
    )
    station_compare.to_csv(
        OUT / "station_metrics_by_fold.csv",
        index=False,
        encoding="utf-8-sig",
    )
    lowflow.to_csv(
        OUT / "lowflow_target_metrics_by_fold.csv",
        index=False,
        encoding="utf-8-sig",
    )
    stone.to_csv(
        OUT / "shijiao_guardrail.csv",
        index=False,
        encoding="utf-8-sig",
    )
    fold_gate.to_csv(
        OUT / "fold_gate.csv", index=False, encoding="utf-8-sig"
    )
    input_audit.to_csv(
        OUT / "input_and_environment_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    payload = {
        "run_id": RUN.name,
        "stage": "H0_Q72_only_strict_OOF_migration_audit",
        "used_year_max": used_year_max,
        "confirmation_years_used": False,
        "eligible_station_count": int(joined["q_site"].nunique()),
        "lowflow_target_count": len(target_sites),
        "fixed_excluded_station_rows": {
            name: int((joined["q_site"] == name).sum())
            for name in sorted(EXCLUSIONS)
        },
        "shijiao_present": SHIJIAO in set(joined["q_site"]),
        "coverage_fraction": coverage_fraction,
        "fold_pass_count": int(fold_gate["fold_pass"].sum()),
        "fold_count": int(len(fold_gate)),
        "engineering_pass": engineering_pass,
        "migration_pass": migration_pass,
        "deployment_decision": (
            "MIGRATE_TO_Q72_ONLY"
            if engineering_pass and migration_pass
            else "RETAIN_FROZEN_FUSION_TEMPORARILY"
        ),
        "next_stage": (
            "H1_Q72_PROCESS_DIAGNOSIS"
            if engineering_pass
            else "STOP_ENGINEERING_REPAIR"
        ),
        "gate_thresholds": {
            "median_delta_NSElog_min": -0.005,
            "median_delta_KGE2012_min": -0.010,
            "median_delta_absPBIAS_pp_max": 1.0,
            "lost_good_count_max": 1,
            "new_severe_PBIAS_count_max": 0,
            "median_delta_low_log_RMSE_max": 0.020,
            "median_delta_low_absPBIAS_pp_max": 5.0,
            "shijiao_delta_NSElog_min": -0.020,
            "shijiao_delta_KGE2012_min": -0.030,
            "shijiao_delta_absPBIAS_pp_max": 5.0,
        },
        "fold_results": fold_gate.to_dict(orient="records"),
    }
    payload["scientific_status"] = (
        "PASS_MIGRATION"
        if engineering_pass and migration_pass
        else (
            "FAIL_MIGRATION"
            if engineering_pass
            else "FAIL_ENGINEERING"
        )
    )
    (OUT / "gate.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Q72-only strict OOF migration audit",
        "",
        f"- Engineering gate: {'PASS' if engineering_pass else 'FAIL'}",
        f"- Migration gate: {'PASS' if migration_pass else 'FAIL'}",
        f"- Eligible stations: {payload['eligible_station_count']}",
        f"- Frozen low-flow targets: {len(target_sites)}",
        f"- Comparable OOF coverage: {coverage_fraction:.3%}",
        f"- Fold gates passed: {payload['fold_pass_count']}/3",
        f"- Deployment decision: `{payload['deployment_decision']}`",
        "- Maximum used year: 2018; 2019–2022 used: False",
        "",
        "## Fold results",
        "",
        "| fold | ΔNSElog | ΔKGE2012 | Δ|PBIAS| pp | lost good | new severe | low Δlog-RMSE | low Δ|PBIAS| pp | 石角 | pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|",
    ]
    for row in fold_gate.itertuples(index=False):
        lines.append(
            f"| {row.fold_id} | {row.median_delta_NSElog:+.4f} | "
            f"{row.median_delta_KGE2012:+.4f} | "
            f"{row.median_delta_absPBIAS_pp:+.2f} | "
            f"{row.lost_good_count} | {row.new_severe_PBIAS_count} | "
            f"{row.median_delta_low_log_RMSE:+.4f} | "
            f"{row.median_delta_low_absPBIAS_pp:+.2f} | "
            f"{'PASS' if row.shijiao_pass else 'FAIL'} | "
            f"{'PASS' if row.fold_pass else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "A failed migration does not authorize alpha tuning. "
            "It retains the frozen fusion only as a temporary deployed "
            "version and proceeds to Q72 single-model process diagnosis.",
        ]
    )
    report = "\n".join(lines) + "\n"
    (OUT / "gate.md").write_text(report, encoding="utf-8")

    readme = "\n".join(
        [
            "# 20260729_6 — Q72-only strict OOF migration audit",
            "",
            "This folder tests whether the frozen interaction-lite Q72 "
            "can safely replace the frozen Q72/Q78 deployment fusion.",
            "No model or alpha is fitted here, and 2019–2022 are not used.",
            "",
            "## Result",
            "",
            *lines[2:9],
            "",
            "See `reports/q72_only_migration_audit/gate.md` and "
            "`gate.json` for the complete machine-readable decision.",
        ]
    ) + "\n"
    (RUN / "README.md").write_text(readme, encoding="utf-8")
    print(report)
    if not engineering_pass:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
