from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


RUN = Path(__file__).resolve().parents[1]
SPARROW = RUN.parents[1]
SOURCE_REPORT = (
    SPARROW
    / "5_Test"
    / "20260728_32"
    / "reports"
    / "lowflow_oof_canonical_baseline"
)
OOF_PATH = SOURCE_REPORT / "q72_interaction_lite_oof_predictions.csv"
LEGACY_CODE_PATH = (
    SPARROW
    / "5_Test"
    / "20260728_32"
    / "scripts"
    / "run_lowflow_oof_canonical_baseline.py"
)
OUT = RUN / "reports" / "signal_registry"
LEGACY_SUMMARY = OUT / "canonical_low_flow_station_summary.csv"
LEGACY_VALIDATION = OUT / "signal_registry_validation.json"

EPS = 1.0e-6
BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_SEED = 20260731
FOLD_PBIAS_THRESHOLD = 25.0
EXCLUSIONS = {"劳村站", "富罗（二）站", "隆安站", "灵渠（三）站", "马口站"}
FOLD_ORDER = [
    "fit_2006_2011_eval_2012_2013",
    "fit_2006_2013_eval_2014_2015",
    "fit_2006_2015_eval_2016_2018",
]
FROZEN_SOURCE_HASHES = {
    "canonical_oof": "525d1520415c9c3250846f9169e656b51ecfb85a3c2aa96dbf48820f12bb1362",
    "legacy_metric_code": "d0beb21b22d3e97dc15902922d678068c094970fad2a24fd9dbb93c664b1fba3",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )


def station_rng(site: str, reach_id: int) -> np.random.Generator:
    """Return a reproducible stream that is invariant to station ordering.

    A shared generator would make a station's bootstrap results change if a
    preceding station were added, removed, or simply sorted differently.  The
    signal registry must instead be reproducible at the station grain.
    """
    payload = f"{BOOTSTRAP_SEED}|{site}|{int(reach_id)}".encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return np.random.default_rng(seed)


def validate_frozen_source(oof: pd.DataFrame) -> dict[str, object]:
    """Fail closed unless the enhanced audit is using the frozen canonical OOF."""
    actual_hashes = {
        "canonical_oof": sha256(OOF_PATH),
        "legacy_metric_code": sha256(LEGACY_CODE_PATH),
    }
    hash_checks = {
        key: actual_hashes[key] == expected
        for key, expected in FROZEN_SOURCE_HASHES.items()
    }
    required_columns = {
        "q_site",
        "reach_id",
        "year",
        "month",
        "fold_id",
        "Q_obsv_cfs",
        "q72_interaction_lite_cfs",
        "train_q25_cfs",
        "is_low_flow",
        "is_reservoir_reach",
        "downstream_reservoir",
        "reach_class",
    }
    missing_columns = sorted(required_columns - set(oof.columns))
    duplicate_keys = int(
        oof.duplicated(["q_site", "reach_id", "year", "month", "fold_id"]).sum()
    )
    expected_low = oof["Q_obsv_cfs"].le(oof["train_q25_cfs"])
    low_flag_mismatches = int((oof["is_low_flow"] != expected_low).sum())
    folds = sorted(oof["fold_id"].astype(str).unique())
    finite_prediction_and_observation = bool(
        np.isfinite(oof["Q_obsv_cfs"].to_numpy(float)).all()
        and np.isfinite(oof["q72_interaction_lite_cfs"].to_numpy(float)).all()
    )
    nonnegative_for_log = bool(
        oof["Q_obsv_cfs"].ge(0).all()
        and oof["q72_interaction_lite_cfs"].ge(0).all()
    )
    checks = {
        "frozen_hashes_match": all(hash_checks.values()),
        "required_columns_present": not missing_columns,
        "row_count_9067": len(oof) == 9067,
        "station_count_114": int(oof["q_site"].nunique()) == 114,
        "year_range_2012_2018": int(oof["year"].min()) == 2012
        and int(oof["year"].max()) == 2018,
        "three_expected_folds": folds == FOLD_ORDER,
        "oof_key_unique": duplicate_keys == 0,
        "stored_low_flow_flag_reproduced": low_flag_mismatches == 0,
        "finite_prediction_and_observation": finite_prediction_and_observation,
        "nonnegative_for_log": nonnegative_for_log,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "actual_hashes": actual_hashes,
        "hash_checks": hash_checks,
        "missing_columns": missing_columns,
        "duplicate_key_count": duplicate_keys,
        "low_flow_flag_mismatch_count": low_flag_mismatches,
    }


def bh_adjust(p_values: pd.Series) -> pd.Series:
    values = p_values.to_numpy(dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    n = len(values)
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    result = np.empty(n, dtype=float)
    result[order] = adjusted
    return pd.Series(result, index=p_values.index)


def fold_metrics(oof: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (fold_id, site, reach_id), part in oof.groupby(
        ["fold_id", "q_site", "reach_id"], sort=False
    ):
        low = part[part["is_low_flow"]].copy()
        if len(low) < 4:
            continue
        obs = low["Q_obsv_cfs"].to_numpy(dtype=float)
        pred = low["q72_interaction_lite_cfs"].to_numpy(dtype=float)
        diff = pred - obs
        log_ratio = np.log(pred + EPS) - np.log(obs + EPS)
        obs_volume = float(obs.sum())
        pbias = float(100.0 * diff.sum() / max(obs_volume, EPS))
        first = part.iloc[0]
        rows.append(
            {
                "fold_id": str(fold_id),
                "q_site": str(site),
                "reach_id": int(float(reach_id)),
                "low_flow_months": int(len(low)),
                "median_log_ratio": float(np.median(log_ratio)),
                "typical_prediction_observation_ratio": float(np.exp(np.median(log_ratio))),
                "low_flow_PBIAS_pct": pbias,
                "overprediction_month_fraction": float(np.mean(pred > obs)),
                "observed_low_flow_volume": obs_volume,
                "predicted_low_flow_volume": float(pred.sum()),
                "signed_low_flow_error": float(diff.sum()),
                "absolute_low_flow_error": float(np.abs(diff).sum()),
                "positive_low_flow_error": float(np.maximum(diff, 0.0).sum()),
                "minimum_positive_Q_obs": float(np.min(obs[obs > 0])) if np.any(obs > 0) else 0.0,
                "zero_or_near_zero_month_count": int((obs <= EPS).sum()),
                "denominator_guard_triggered": bool(obs_volume <= EPS),
                "PBIAS_floor_1x": pbias,
                "PBIAS_floor_10x": float(100.0 * diff.sum() / max(obs_volume, 10 * EPS)),
                "PBIAS_floor_100x": float(100.0 * diff.sum() / max(obs_volume, 100 * EPS)),
                "triple_direction_consistent": bool(
                    np.median(log_ratio) > 0
                    and pbias > 0
                    and np.mean(pred > obs) >= (2.0 / 3.0)
                ),
                "legacy_fold_overprediction": bool(pbias > FOLD_PBIAS_THRESHOLD),
                "reservoir_related": bool(
                    float(first["is_reservoir_reach"]) > 0
                    or float(first["downstream_reservoir"]) > 0
                ),
                "reach_class": str(first["reach_class"]),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_station(
    low: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    years = np.array(sorted(low["year"].astype(int).unique()), dtype=int)
    replicates = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    blocks = {year: low[low["year"].astype(int) == year] for year in years}
    for index in range(BOOTSTRAP_REPLICATES):
        sampled = rng.choice(years, size=len(years), replace=True)
        parts = [blocks[int(year)] for year in sampled]
        boot = pd.concat(parts, ignore_index=True)
        log_ratio = np.log(boot["q72_interaction_lite_cfs"].to_numpy(float) + EPS) - np.log(
            boot["Q_obsv_cfs"].to_numpy(float) + EPS
        )
        replicates[index] = float(np.median(log_ratio))
    p_value = float((1 + np.sum(replicates <= 0)) / (BOOTSTRAP_REPLICATES + 1))
    return replicates, p_value


def main() -> int:
    validation = json.loads(LEGACY_VALIDATION.read_text(encoding="utf-8"))
    if validation.get("terminal_status") != "CANONICAL_SIGNAL_REGISTRY_REPRODUCED":
        raise RuntimeError("Legacy canonical registry gate has not passed")

    oof = pd.read_csv(OOF_PATH, encoding="utf-8-sig")
    oof["is_low_flow"] = parse_bool(oof["is_low_flow"])
    source_guard = validate_frozen_source(oof)
    if not source_guard["passed"]:
        raise RuntimeError(
            "Enhanced audit source guard failed; frozen canonical OOF or metric "
            "code no longer matches the registered source."
        )
    legacy = pd.read_csv(LEGACY_SUMMARY, encoding="utf-8-sig")
    legacy["stable_nonreservoir_target"] = parse_bool(legacy["stable_nonreservoir_target"])
    folds = fold_metrics(oof)

    station_rows: list[dict[str, object]] = []
    bootstrap_rows: list[dict[str, object]] = []
    for (site, reach_id), part in oof.groupby(["q_site", "reach_id"], sort=True):
        low = part[part["is_low_flow"]].copy()
        eligible_fold_rows = folds[
            (folds["q_site"] == site) & (folds["reach_id"] == int(float(reach_id)))
        ].copy()
        if eligible_fold_rows.empty or low.empty:
            continue
        low = low[low["fold_id"].isin(eligible_fold_rows["fold_id"])].copy()
        obs = low["Q_obsv_cfs"].to_numpy(float)
        pred = low["q72_interaction_lite_cfs"].to_numpy(float)
        log_ratio = np.log(pred + EPS) - np.log(obs + EPS)
        replicates, p_value = bootstrap_station(
            low, station_rng(str(site), int(float(reach_id)))
        )
        for index, value in enumerate(replicates):
            bootstrap_rows.append(
                {
                    "q_site": site,
                    "reach_id": int(float(reach_id)),
                    "replicate": index,
                    "pooled_median_log_ratio": value,
                }
            )
        positive_by_fold = eligible_fold_rows.groupby("fold_id")["positive_low_flow_error"].sum()
        positive_total = float(positive_by_fold.sum())
        largest_fold_share = (
            float(positive_by_fold.max() / positive_total) if positive_total > 0 else 0.0
        )
        first = eligible_fold_rows.iloc[0]
        station_rows.append(
            {
                "q_site": str(site),
                "reach_id": int(float(reach_id)),
                "eligible_folds_enhanced": int(eligible_fold_rows["fold_id"].nunique()),
                "triple_direction_folds": int(eligible_fold_rows["triple_direction_consistent"].sum()),
                "legacy_overprediction_folds_recomputed": int(
                    eligible_fold_rows["legacy_fold_overprediction"].sum()
                ),
                "pooled_log_bias": float(np.median(log_ratio)),
                "pooled_typical_prediction_observation_ratio": float(np.exp(np.median(log_ratio))),
                "pooled_overprediction_month_fraction": float(np.mean(pred > obs)),
                "bootstrap_lower_one_sided_95": float(np.quantile(replicates, 0.05)),
                "bootstrap_lower_two_sided_95": float(np.quantile(replicates, 0.025)),
                "bootstrap_upper_two_sided_95": float(np.quantile(replicates, 0.975)),
                "bootstrap_p": p_value,
                "observed_low_flow_volume": float(obs.sum()),
                "predicted_low_flow_volume": float(pred.sum()),
                "absolute_low_flow_error": float(np.abs(pred - obs).sum()),
                "largest_fold_error_share": largest_fold_share,
                "fold_dominated_signal": bool(largest_fold_share > 0.60),
                "denominator_guard_triggered": bool(obs.sum() <= EPS),
                "zero_or_near_zero_month_count": int((obs <= EPS).sum()),
                "reservoir_related": bool(first["reservoir_related"]),
                "reach_class": str(first["reach_class"]),
            }
        )

    stations = pd.DataFrame(station_rows)
    eligible_for_bh = ~stations["reservoir_related"] & ~stations["q_site"].isin(EXCLUSIONS)
    stations["BH_q"] = np.nan
    stations.loc[eligible_for_bh, "BH_q"] = bh_adjust(
        stations.loc[eligible_for_bh, "bootstrap_p"]
    )
    stations = stations.merge(
        legacy[
            [
                "q_site",
                "reach_id",
                "eligible_folds",
                "overprediction_folds",
                "has_2016_2018",
                "stable_nonreservoir_target",
                "median_low_flow_PBIAS",
                "median_low_flow_log_RMSE",
            ]
        ],
        on=["q_site", "reach_id"],
        how="left",
        validate="one_to_one",
    )
    stations["legacy_canonical_membership"] = stations[
        "stable_nonreservoir_target"
    ].fillna(False).astype(bool)

    nonreservoir_abs_error = stations.loc[eligible_for_bh, "absolute_low_flow_error"]
    strong_abs_error_threshold = float(nonreservoir_abs_error.quantile(0.90))
    stations["confirmed_stable_signal"] = (
        stations["eligible_folds_enhanced"].eq(3)
        & stations["triple_direction_folds"].eq(3)
        & stations["bootstrap_lower_one_sided_95"].gt(0)
        & stations["BH_q"].le(0.05)
        & stations["largest_fold_error_share"].le(0.60)
        & ~stations["reservoir_related"]
        & ~stations["q_site"].isin(EXCLUSIONS)
        & ~stations["denominator_guard_triggered"]
    )
    stations["strong_signal"] = stations["confirmed_stable_signal"] & (
        stations["legacy_overprediction_folds_recomputed"].eq(3)
        | stations["absolute_low_flow_error"].ge(strong_abs_error_threshold)
    )
    stations["candidate_signal"] = (
        stations["legacy_canonical_membership"]
        | stations["triple_direction_folds"].ge(2)
        | (
            stations["pooled_log_bias"].gt(0)
            & stations["bootstrap_lower_one_sided_95"].le(0)
        )
    ) & ~stations["confirmed_stable_signal"]

    stations["robustness_class"] = "NO_STABLE_SIGNAL"
    stations.loc[stations["candidate_signal"], "robustness_class"] = "CANDIDATE_SIGNAL"
    stations.loc[
        stations["confirmed_stable_signal"], "robustness_class"
    ] = "CONFIRMED_STABLE_SIGNAL"
    stations.loc[stations["strong_signal"], "robustness_class"] = "STRONG_SIGNAL"
    stations.loc[
        stations["denominator_guard_triggered"], "robustness_class"
    ] = "NUMERICALLY_UNSTABLE_LOW_FLOW"
    stations["station"] = stations["q_site"]
    stations["signal_presence_class"] = np.where(
        stations["legacy_canonical_membership"],
        "LEGACY_CANONICAL_SIGNAL",
        "NOT_LEGACY_CANONICAL_SIGNAL",
    )
    stations["review_priority_evidence_class"] = stations["robustness_class"]
    stations["signal_class"] = stations["robustness_class"]
    stations["enhanced_evidence_status"] = "PROVISIONAL"
    stations["numeric_resolution_assessment"] = "NOT_YET_TESTABLE_WITH_OOF_ONLY"
    stations["cross_model_support"] = "NOT_ASSESSED_IN_THIS_GATE"
    stations["source_prediction_sha256"] = sha256(OOF_PATH)
    stations["metric_code_sha256"] = sha256(LEGACY_CODE_PATH)
    stations["legacy_metric_code_sha256"] = sha256(LEGACY_CODE_PATH)
    stations["enhanced_metric_code_sha256"] = sha256(Path(__file__).resolve())

    fold_lookup = {fold: f"fold{index + 1}" for index, fold in enumerate(FOLD_ORDER)}
    wide_columns = [
        "median_log_ratio",
        "low_flow_PBIAS_pct",
        "overprediction_month_fraction",
        "observed_low_flow_volume",
        "absolute_low_flow_error",
    ]
    wide = folds[["q_site", "reach_id", "fold_id", *wide_columns]].copy()
    wide["fold_short"] = wide["fold_id"].map(fold_lookup)
    wide = wide.pivot(index=["q_site", "reach_id"], columns="fold_short", values=wide_columns)
    wide.columns = [f"{fold}_{metric}" for metric, fold in wide.columns]
    wide = wide.reset_index()
    registry = stations.merge(wide, on=["q_site", "reach_id"], how="left")
    for fold_index in range(1, 4):
        registry[f"fold{fold_index}_log_bias"] = registry[
            f"fold{fold_index}_median_log_ratio"
        ]
        registry[f"fold{fold_index}_low_flow_PBIAS"] = registry[
            f"fold{fold_index}_low_flow_PBIAS_pct"
        ]
    registry["pooled_bootstrap_lower"] = registry["bootstrap_lower_one_sided_95"]
    registry["pooled_bootstrap_upper"] = registry["bootstrap_upper_two_sided_95"]
    registry["overprediction_month_fraction"] = registry[
        "pooled_overprediction_month_fraction"
    ]
    contract_columns = [
        "station",
        "q_site",
        "reach_id",
        "eligible_folds",
        "overprediction_folds",
        "fold1_log_bias",
        "fold2_log_bias",
        "fold3_log_bias",
        "fold1_low_flow_PBIAS",
        "fold2_low_flow_PBIAS",
        "fold3_low_flow_PBIAS",
        "pooled_log_bias",
        "pooled_bootstrap_lower",
        "pooled_bootstrap_upper",
        "BH_q",
        "overprediction_month_fraction",
        "observed_low_flow_volume",
        "absolute_low_flow_error",
        "largest_fold_error_share",
        "cross_model_support",
        "signal_presence_class",
        "review_priority_evidence_class",
        "signal_class",
        "enhanced_evidence_status",
        "numeric_resolution_assessment",
        "legacy_canonical_membership",
        "source_prediction_sha256",
        "metric_code_sha256",
        "legacy_metric_code_sha256",
        "enhanced_metric_code_sha256",
    ]
    remaining_columns = [
        column for column in registry.columns if column not in contract_columns
    ]
    registry = registry[contract_columns + remaining_columns]

    folds.to_csv(
        OUT / "canonical_low_flow_station_fold_metrics_enhanced.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(bootstrap_rows).to_csv(
        OUT / "canonical_signal_robustness_bootstrap.csv",
        index=False,
        encoding="utf-8-sig",
    )
    registry.to_csv(
        OUT / "canonical_signal_registry.csv",
        index=False,
        encoding="utf-8-sig",
    )
    class_counts = {
        str(key): int(value)
        for key, value in registry["robustness_class"].value_counts().sort_index().items()
    }
    payload = {
        "gate": "ENHANCED_SIGNAL_EVIDENCE_AUDIT",
        "passed": True,
        "legacy_registry_reproduced": True,
        "legacy_member_count": int(registry["legacy_canonical_membership"].sum()),
        "enhanced_class_counts": class_counts,
        "enhanced_evidence_status": "PROVISIONAL",
        "bootstrap_scheme": "POOLED_NON_OVERLAPPING_YEAR_CLUSTER_BOOTSTRAP",
        "bootstrap_p_interpretation": "EXPLORATORY_TAIL_PROBABILITY",
        "multiple_testing_dependence_sensitivity": "NOT_ASSESSED_IN_THIS_GATE",
        "numeric_resolution_assessment": "NOT_YET_TESTABLE_WITH_OOF_ONLY",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_random_stream": "STATION_ID_DERIVED_DETERMINISTIC",
        "frozen_source_guard": source_guard,
        "bh_population_count": int(eligible_for_bh.sum()),
        "strong_absolute_error_threshold_p90": strong_abs_error_threshold,
        "registry_required_columns_present": all(
            column in registry.columns for column in contract_columns
        ),
        "important_interpretation": (
            "Enhanced classes are secondary evidence and do not redefine "
            "the 28 legacy canonical members."
        ),
    }
    (OUT / "enhanced_signal_validation.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
