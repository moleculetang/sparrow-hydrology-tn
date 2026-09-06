from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


RUN = Path(__file__).resolve().parents[1]
SPARROW = RUN.parents[1]
SOURCE = SPARROW / "5_Test" / "20260728_32"
SOURCE_REPORT = SOURCE / "reports" / "lowflow_oof_canonical_baseline"
OOF_PATH = SOURCE_REPORT / "q72_interaction_lite_oof_predictions.csv"
LEGACY_CODE_PATH = SOURCE / "scripts" / "run_lowflow_oof_canonical_baseline.py"
EXPECTED_FOLD_PATH = SOURCE_REPORT / "low_flow_signature_by_fold.csv"
EXPECTED_SUMMARY_PATH = SOURCE_REPORT / "low_flow_signature_summary_canonical.csv"
OUT = RUN / "reports" / "signal_registry"

EPS = 1.0e-6
LOW_MONTH_MIN = 4
FOLD_PBIAS_THRESHOLD = 25.0
LATE_FOLD = "fit_2006_2015_eval_2016_2018"
EXCLUSIONS = {"劳村站", "富罗（二）站", "隆安站", "灵渠（三）站", "马口站"}

EXPECTED_HASHES = {
    "oof": "525d1520415c9c3250846f9169e656b51ecfb85a3c2aa96dbf48820f12bb1362",
    "legacy_code": "d0beb21b22d3e97dc15902922d678068c094970fad2a24fd9dbb93c664b1fba3",
    "expected_fold": "379343c5e3ef910bc1d32c50e2fa6888e26a12bfff53d271867cb604d82bf0bc",
    "expected_summary": "b2ffcbb11196e6d30055585c6a530504f2875c2225f4f2402d5f78cce7f87b03",
}

REQUIRED_COLUMNS = {
    "q_site",
    "reach_id",
    "year",
    "month",
    "Q_obsv_cfs",
    "is_reservoir_reach",
    "downstream_reservoir",
    "fold_id",
    "q72_interaction_lite_cfs",
    "train_q25_cfs",
    "is_low_flow",
    "reach_class",
    "CumAreaKm2",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )


def old_area_class(area: float) -> str:
    if area < 5000:
        return "headwater"
    if area < 25000:
        return "mid_nonheadwater"
    return "large_nonheadwater"


def load_oof() -> pd.DataFrame:
    frame = pd.read_csv(OOF_PATH, encoding="utf-8-sig")
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise RuntimeError(f"Frozen OOF schema missing columns: {missing}")
    frame["is_low_flow"] = bool_series(frame["is_low_flow"])
    if frame["is_low_flow"].isna().any():
        raise RuntimeError("Frozen OOF contains unparseable is_low_flow values")
    return frame


def reconstruct(oof: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_rows: list[dict[str, object]] = []
    for (fold_id, site, reach_id), part in oof.groupby(
        ["fold_id", "q_site", "reach_id"], sort=False, dropna=False
    ):
        low = part[part["is_low_flow"]].copy()
        if len(low) < LOW_MONTH_MIN:
            continue
        obs = low["Q_obsv_cfs"].to_numpy(dtype=float)
        pred = low["q72_interaction_lite_cfs"].to_numpy(dtype=float)
        first = part.iloc[0]
        pbias = 100.0 * (pred.sum() - obs.sum()) / max(obs.sum(), EPS)
        fold_rows.append(
            {
                "fold_id": str(fold_id),
                "q_site": str(site),
                "reach_id": int(float(reach_id)),
                "low_flow_months": int(len(low)),
                "low_flow_PBIAS_pct": float(pbias),
                "low_flow_log_RMSE": float(
                    np.sqrt(np.mean((np.log(pred + EPS) - np.log(obs + EPS)) ** 2))
                ),
                "low_flow_overprediction": bool(pbias > FOLD_PBIAS_THRESHOLD),
                "reservoir_related": bool(
                    float(first["is_reservoir_reach"]) > 0
                    or float(first["downstream_reservoir"]) > 0
                ),
                "area_audit_class": old_area_class(float(first["CumAreaKm2"])),
                "reach_class": str(first["reach_class"]),
            }
        )
    by_fold = pd.DataFrame(fold_rows)
    summary = (
        by_fold.groupby(
            ["q_site", "reach_id", "reservoir_related", "area_audit_class", "reach_class"],
            as_index=False,
        )
        .agg(
            eligible_folds=("fold_id", "nunique"),
            overprediction_folds=("low_flow_overprediction", "sum"),
            has_2016_2018=("fold_id", lambda values: LATE_FOLD in set(values)),
            median_low_flow_PBIAS=("low_flow_PBIAS_pct", "median"),
            median_low_flow_log_RMSE=("low_flow_log_RMSE", "median"),
        )
    )
    summary["stable_nonreservoir_target"] = (
        summary["overprediction_folds"].ge(2)
        & summary["has_2016_2018"].astype(bool)
        & ~summary["reservoir_related"].astype(bool)
        & ~summary["q_site"].isin(EXCLUSIONS)
    )
    return by_fold, summary


def compare_frames(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    keys: list[str],
    numeric_columns: list[str],
    boolean_columns: list[str],
    tolerance: float = 1e-10,
) -> dict[str, object]:
    left = actual.copy()
    right = expected.copy()
    for column in boolean_columns:
        left[column] = bool_series(left[column]) if left[column].dtype == object else left[column].astype(bool)
        right[column] = bool_series(right[column]) if right[column].dtype == object else right[column].astype(bool)
    merged = left.merge(right, on=keys, how="outer", suffixes=("_actual", "_expected"), indicator=True)
    only_actual = merged.loc[merged["_merge"] == "left_only", keys].to_dict("records")
    only_expected = merged.loc[merged["_merge"] == "right_only", keys].to_dict("records")
    shared = merged["_merge"].eq("both")
    max_numeric_delta: dict[str, float] = {}
    numeric_mismatch_count: dict[str, int] = {}
    for column in numeric_columns:
        delta = (
            pd.to_numeric(merged.loc[shared, f"{column}_actual"], errors="coerce")
            - pd.to_numeric(merged.loc[shared, f"{column}_expected"], errors="coerce")
        ).abs()
        max_numeric_delta[column] = float(delta.max()) if len(delta) else 0.0
        numeric_mismatch_count[column] = int((delta > tolerance).sum())
    boolean_mismatch_count: dict[str, int] = {}
    for column in boolean_columns:
        boolean_mismatch_count[column] = int(
            (
                merged.loc[shared, f"{column}_actual"].astype(bool).to_numpy()
                != merged.loc[shared, f"{column}_expected"].astype(bool).to_numpy()
            ).sum()
        )
    passed = (
        not only_actual
        and not only_expected
        and all(value <= tolerance for value in max_numeric_delta.values())
        and all(value == 0 for value in boolean_mismatch_count.values())
    )
    return {
        "passed": passed,
        "only_actual": only_actual,
        "only_expected": only_expected,
        "max_numeric_delta": max_numeric_delta,
        "numeric_mismatch_count": numeric_mismatch_count,
        "boolean_mismatch_count": boolean_mismatch_count,
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    hashes = {
        "oof": sha256(OOF_PATH),
        "legacy_code": sha256(LEGACY_CODE_PATH),
        "expected_fold": sha256(EXPECTED_FOLD_PATH),
        "expected_summary": sha256(EXPECTED_SUMMARY_PATH),
    }
    hash_checks = {key: hashes[key] == EXPECTED_HASHES[key] for key in EXPECTED_HASHES}

    oof = load_oof()
    key_columns = ["q_site", "reach_id", "year", "month", "fold_id"]
    duplicate_count = int(oof.duplicated(key_columns).sum())
    low_flag_expected = oof["Q_obsv_cfs"].le(oof["train_q25_cfs"])
    low_flag_mismatch_count = int((oof["is_low_flow"] != low_flag_expected).sum())
    min_year = int(oof["year"].min())
    max_year = int(oof["year"].max())
    station_count = int(oof["q_site"].nunique())
    folds = sorted(oof["fold_id"].astype(str).unique())

    by_fold, summary = reconstruct(oof)
    expected_fold = pd.read_csv(EXPECTED_FOLD_PATH, encoding="utf-8-sig")
    expected_summary = pd.read_csv(EXPECTED_SUMMARY_PATH, encoding="utf-8-sig")
    fold_comparison = compare_frames(
        by_fold,
        expected_fold,
        keys=["fold_id", "q_site", "reach_id"],
        numeric_columns=["low_flow_months", "low_flow_PBIAS_pct", "low_flow_log_RMSE"],
        boolean_columns=["low_flow_overprediction", "reservoir_related"],
    )
    summary_comparison = compare_frames(
        summary,
        expected_summary,
        keys=["q_site", "reach_id"],
        numeric_columns=[
            "eligible_folds",
            "overprediction_folds",
            "median_low_flow_PBIAS",
            "median_low_flow_log_RMSE",
        ],
        boolean_columns=["reservoir_related", "has_2016_2018", "stable_nonreservoir_target"],
    )

    actual_targets = summary.loc[
        summary["stable_nonreservoir_target"], ["q_site", "reach_id"]
    ].sort_values(["q_site", "reach_id"])
    expected_targets = expected_summary.loc[
        bool_series(expected_summary["stable_nonreservoir_target"]),
        ["q_site", "reach_id"],
    ].sort_values(["q_site", "reach_id"])
    target_comparison = actual_targets.merge(
        expected_targets,
        on=["q_site", "reach_id"],
        how="outer",
        indicator=True,
    )
    target_comparison["status"] = target_comparison["_merge"].map(
        {"both": "MATCH", "left_only": "RECONSTRUCTED_ONLY", "right_only": "EXPECTED_ONLY"}
    )
    target_set_equal = bool(target_comparison["_merge"].eq("both").all())

    excluded_rows = {name: int((oof["q_site"] == name).sum()) for name in sorted(EXCLUSIONS)}
    checks = {
        "all_frozen_hashes_match": all(hash_checks.values()),
        "row_count_9067": len(oof) == 9067,
        "station_count_114": station_count == 114,
        "year_range_2012_2018": min_year == 2012 and max_year == 2018,
        "three_expected_folds": folds
        == [
            "fit_2006_2011_eval_2012_2013",
            "fit_2006_2013_eval_2014_2015",
            "fit_2006_2015_eval_2016_2018",
        ],
        "oof_key_unique": duplicate_count == 0,
        "stored_low_flow_flag_reproduced": low_flag_mismatch_count == 0,
        "fold_metrics_reproduced": bool(fold_comparison["passed"]),
        "station_summary_reproduced": bool(summary_comparison["passed"]),
        "target_set_equal": target_set_equal,
        "target_count_28": len(actual_targets) == 28,
        "excluded_station_rows_zero": all(value == 0 for value in excluded_rows.values()),
        "shijiao_present": bool((oof["q_site"] == "石角站").any()),
        "no_2019_2022": max_year <= 2018,
    }
    passed = all(checks.values())
    terminal_status = (
        "CANONICAL_SIGNAL_REGISTRY_REPRODUCED"
        if passed
        else "SIGNAL_REGISTRY_NOT_REPRODUCIBLE"
    )

    by_fold.to_csv(
        OUT / "canonical_low_flow_station_fold_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        OUT / "canonical_low_flow_station_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    target_comparison.drop(columns=["_merge"]).to_csv(
        OUT / "canonical_signal_registry_expected_vs_reproduced.csv",
        index=False,
        encoding="utf-8-sig",
    )
    source_manifest = {
        "source_files": {
            "canonical_oof": {"path": str(OOF_PATH), "sha256": hashes["oof"]},
            "legacy_metric_code": {"path": str(LEGACY_CODE_PATH), "sha256": hashes["legacy_code"]},
            "expected_fold_metrics": {
                "path": str(EXPECTED_FOLD_PATH),
                "sha256": hashes["expected_fold"],
            },
            "expected_station_summary": {
                "path": str(EXPECTED_SUMMARY_PATH),
                "sha256": hashes["expected_summary"],
            },
        },
        "expected_hashes": EXPECTED_HASHES,
        "hash_checks": hash_checks,
        "row_count": len(oof),
        "station_count": station_count,
        "year_min": min_year,
        "year_max": max_year,
        "folds": folds,
        "reconstruction_code_path": str(Path(__file__).resolve()),
        "reconstruction_code_sha256": sha256(Path(__file__).resolve()),
    }
    (OUT / "canonical_oof_source_manifest.json").write_text(
        json.dumps(source_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    validation = {
        "gate": "LEGACY_CANONICAL_SIGNAL_REGISTRY_RECONSTRUCTION",
        "terminal_status": terminal_status,
        "passed": passed,
        "checks": checks,
        "hash_checks": hash_checks,
        "duplicate_key_count": duplicate_count,
        "low_flow_flag_mismatch_count": low_flag_mismatch_count,
        "excluded_station_rows": excluded_rows,
        "reconstructed_target_count": len(actual_targets),
        "expected_target_count": len(expected_targets),
        "fold_comparison": fold_comparison,
        "summary_comparison": summary_comparison,
    }
    (OUT / "signal_registry_validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"terminal_status": terminal_status, "checks": checks}, ensure_ascii=False, indent=2))
    return 0 if passed else 3


if __name__ == "__main__":
    sys.exit(main())

