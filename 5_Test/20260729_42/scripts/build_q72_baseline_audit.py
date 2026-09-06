from __future__ import annotations

import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import sys
from typing import Any

import numpy as np
import pandas as pd

from runtime_guard import assert_sparrow_runtime


RUNTIME = assert_sparrow_runtime()
RUN = Path(__file__).resolve().parents[1]
MODEL_ROOT = RUN / "reports" / "q72_baseline"
REPORT = RUN / "reports" / "baseline_audit"
OUTPUT = RUN / "outputs"
MANIFEST = RUN / "inputs_manifest"
REFERENCE = RUN / "reference" / "blocked_folds"
COMMON_REFERENCE = RUN / "reference" / "q72_common_sample_reference.parquet"
FOLDS = [
    "fit_2006_2011_eval_2012_2013",
    "fit_2006_2013_eval_2014_2015",
    "fit_2006_2015_eval_2016_2018",
]
FIXED_EXCLUSIONS = {
    "劳村站", "富罗（二）站", "隆安站", "灵渠（三）站", "马口站"
}
PROTECTED = "石角站"
EPS = 1.0e-12


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def nse(observed: np.ndarray, predicted: np.ndarray) -> float:
    denominator = np.sum((observed - observed.mean()) ** 2)
    if denominator <= EPS:
        return np.nan
    return float(
        1.0 - np.sum((predicted - observed) ** 2) / denominator
    )


def kge(observed: np.ndarray, predicted: np.ndarray) -> float:
    if len(observed) < 4:
        return np.nan
    obs_std = observed.std(ddof=0)
    pred_std = predicted.std(ddof=0)
    obs_mean = observed.mean()
    if obs_std <= EPS or obs_mean <= EPS:
        return np.nan
    correlation = np.corrcoef(observed, predicted)[0, 1]
    alpha = pred_std / obs_std
    beta = predicted.mean() / obs_mean
    return float(
        1.0
        - np.sqrt(
            (correlation - 1.0) ** 2
            + (alpha - 1.0) ** 2
            + (beta - 1.0) ** 2
        )
    )


def station_metrics(
    frame: pd.DataFrame,
    scenario: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for station, group in frame.groupby("station_name", sort=True):
        group = group.sort_values(["year", "month"])
        observed = group["observed_m3s"].to_numpy(float)
        predicted = np.maximum(group["predicted_m3s"].to_numpy(float), 0.0)
        if len(group) < 12:
            continue
        low_threshold = np.quantile(observed, 0.25)
        low = observed <= low_threshold
        log_observed = np.log1p(np.maximum(observed, 0.0))
        log_predicted = np.log1p(predicted)
        metric_log_nse = nse(log_observed, log_predicted)
        metric_kge = kge(observed, predicted)
        pbias = float(
            100.0 * (predicted.sum() - observed.sum())
            / max(observed.sum(), EPS)
        )
        low_error = float(
            np.median(np.abs(log_predicted[low] - log_observed[low]))
        )
        high = observed >= np.quantile(observed, 0.75)
        high_nrmse = float(
            np.sqrt(np.mean((predicted[high] - observed[high]) ** 2))
            / max(np.mean(observed[high]), EPS)
        )
        rows.append(
            {
                "scenario": scenario,
                "station_name": station,
                "reach_id": int(group["reach_id"].iloc[0]),
                "months": int(len(group)),
                "nse": nse(observed, predicted),
                "log_nse": metric_log_nse,
                "kge": metric_kge,
                "pbias_pct": pbias,
                "absolute_pbias_pct": abs(pbias),
                "lowflow_log_error": low_error,
                "highflow_nrmse": high_nrmse,
                "good": bool(
                    metric_log_nse >= 0.5
                    and metric_kge >= 0.5
                    and abs(pbias) <= 25.0
                ),
                "severe_pbias": bool(abs(pbias) > 50.0),
            }
        )
    return pd.DataFrame(rows)


def summarize(stations: pd.DataFrame) -> dict[str, float | int]:
    return {
        "station_count": int(len(stations)),
        "median_nse": float(stations["nse"].median()),
        "median_log_nse": float(stations["log_nse"].median()),
        "median_kge": float(stations["kge"].median()),
        "median_absolute_pbias_pct": float(
            stations["absolute_pbias_pct"].median()
        ),
        "median_lowflow_log_error": float(
            stations["lowflow_log_error"].median()
        ),
        "median_highflow_nrmse": float(
            stations["highflow_nrmse"].median()
        ),
        "good_count": int(stations["good"].sum()),
        "severe_pbias_count": int(stations["severe_pbias"].sum()),
    }


def main() -> None:
    for directory in [REPORT, OUTPUT, MANIFEST]:
        directory.mkdir(parents=True, exist_ok=True)
    cfs_per_m3s = 35.3146667215
    local_parts = []
    reproduction_rows = []
    parameter_inventory = []
    design_inventory = []
    for fold_id in FOLDS:
        local_path = MODEL_ROOT / "blocked_folds" / fold_id / "evaluation_predictions.csv"
        reference_path = REFERENCE / fold_id / "evaluation_predictions.csv"
        local = pd.read_csv(local_path, encoding="utf-8-sig")
        reference = pd.read_csv(reference_path, encoding="utf-8-sig")
        keys = ["q_site", "comid", "year", "month"]
        comparison = local.merge(
            reference[keys + ["actual", "predict"]],
            on=keys,
            how="outer",
            suffixes=("_local", "_reference"),
            indicator=True,
            validate="one_to_one",
        )
        both = comparison["_merge"].eq("both")
        prediction_log_difference = np.abs(
            np.log(np.maximum(comparison.loc[both, "predict_local"], EPS))
            - np.log(
                np.maximum(comparison.loc[both, "predict_reference"], EPS)
            )
        )
        actual_difference = np.abs(
            comparison.loc[both, "actual_local"]
            - comparison.loc[both, "actual_reference"]
        )
        reproduction_rows.append(
            {
                "fold_id": fold_id,
                "local_rows": int(len(local)),
                "reference_rows": int(len(reference)),
                "matched_rows": int(both.sum()),
                "left_only_rows": int(
                    comparison["_merge"].eq("left_only").sum()
                ),
                "right_only_rows": int(
                    comparison["_merge"].eq("right_only").sum()
                ),
                "max_abs_log_prediction_difference": float(
                    prediction_log_difference.max()
                ),
                "max_abs_actual_difference_cfs": float(
                    actual_difference.max()
                ),
            }
        )
        part = local.rename(
            columns={
                "q_site": "station_name",
                "comid": "reach_id",
                "actual": "observed_cfs",
                "predict": "predicted_cfs",
            }
        )
        part["fold_id"] = fold_id
        part["observed_m3s"] = part["observed_cfs"] / cfs_per_m3s
        part["predicted_m3s"] = part["predicted_cfs"] / cfs_per_m3s
        local_parts.append(part)

        fold_report = MODEL_ROOT / "blocked_folds" / fold_id / "reports"
        for path in sorted(fold_report.glob("*parameters*.csv")):
            parameter_inventory.append(
                {
                    "fold_id": fold_id,
                    "path": str(path.relative_to(RUN)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
        design_manifest_path = (
            fold_report / "design_matrix" / "design_matrix_manifest.json"
        )
        design = json.loads(
            design_manifest_path.read_text(encoding="utf-8")
        )
        design_inventory.append(
            {
                "fold_id": fold_id,
                **design,
                "manifest_path": str(design_manifest_path.relative_to(RUN)),
            }
        )

    oof = pd.concat(local_parts, ignore_index=True)
    oof = oof.sort_values(
        ["fold_id", "station_name", "year", "month"]
    ).reset_index(drop=True)
    oof.to_parquet(OUTPUT / "q72_three_fold_oof_predictions.parquet", index=False)
    oof.to_csv(
        OUTPUT / "q72_three_fold_oof_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    reproduction = pd.DataFrame(reproduction_rows)
    reproduction.to_csv(
        REPORT / "exact_reproduction_by_fold.csv",
        index=False,
        encoding="utf-8-sig",
    )

    fold_station_frames = []
    fold_summary_rows = []
    for fold_id in FOLDS:
        fold = oof.loc[oof["fold_id"].eq(fold_id)]
        stations = station_metrics(fold, fold_id)
        fold_station_frames.append(stations.assign(fold_id=fold_id))
        fold_summary_rows.append({"fold_id": fold_id, **summarize(stations)})
    pd.concat(fold_station_frames, ignore_index=True).to_csv(
        REPORT / "station_metrics_by_fold.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(fold_summary_rows).to_csv(
        REPORT / "fold_summary_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    combined_stations = station_metrics(oof, "three_fold_oof")
    combined_stations.to_csv(
        REPORT / "three_fold_oof_station_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    combined_summary = summarize(combined_stations)

    common_reference = pd.read_parquet(COMMON_REFERENCE)
    confirmation_reference = common_reference.loc[
        common_reference["role"].eq("confirmation")
    ].copy()
    confirmation = oof.loc[
        oof["fold_id"].eq(FOLDS[-1])
    ].merge(
        confirmation_reference[
            ["station_name", "reach_id", "year", "month"]
        ],
        on=["station_name", "reach_id", "year", "month"],
        how="inner",
        validate="one_to_one",
    )
    confirmation_stations = station_metrics(
        confirmation, "confirmation_common_sample"
    )
    confirmation_summary = summarize(confirmation_stations)
    confirmation_stations.to_csv(
        REPORT / "confirmation_common_sample_station_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(
        [{"scope": "confirmation_common_sample", **confirmation_summary}]
    ).to_csv(
        REPORT / "confirmation_common_sample_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    confirmation.to_parquet(
        OUTPUT / "q72_confirmation_common_sample_predictions.parquet",
        index=False,
    )
    shijiao = combined_stations.loc[
        combined_stations["station_name"].eq(PROTECTED)
    ].copy()
    shijiao.to_csv(
        REPORT / "protected_shijiao_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(parameter_inventory).to_csv(
        REPORT / "parameter_file_inventory.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(design_inventory).to_csv(
        REPORT / "design_matrix_inventory.csv",
        index=False,
        encoding="utf-8-sig",
    )

    package_versions = {
        name: metadata.version(name)
        for name in ["numpy", "pandas", "scipy", "matplotlib", "pyarrow"]
    }
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": package_versions,
        "runtime": RUNTIME,
        "random_seeds": [],
        "deterministic_solver": "numpy.linalg.lstsq",
    }
    (MANIFEST / "environment.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    provenance_rows = []
    for root_name in ["inputs", "scripts", "reference"]:
        for path in sorted((RUN / root_name).rglob("*")):
            if path.is_file():
                provenance_rows.append(
                    {
                        "role": root_name,
                        "path": str(path.relative_to(RUN)),
                        "bytes": path.stat().st_size,
                        "sha256": sha256(path),
                    }
                )
    pd.DataFrame(provenance_rows).to_csv(
        MANIFEST / "provenance_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (MANIFEST / "provenance_manifest.json").write_text(
        json.dumps(provenance_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    input_frame = pd.read_parquet(RUN / "inputs" / "indata.parquet")
    active_stations = set(input_frame["q_site"].dropna().astype(str))
    model_scripts = [
        RUN / "scripts" / "run_q72_baseline.py",
        RUN / "scripts" / "components"
        / "fit_monthly_bayes_seasonal_hysteresis.py",
    ]
    forbidden_tokens = [
        "Q" + "78_mass",
        "station_month_" + "corrected_m3s",
        "q72_" + "mechanism_fusion",
        "fusion_" + "grid",
    ]
    model_code = "\n".join(
        path.read_text(encoding="utf-8") for path in model_scripts
    ).casefold()
    no_forbidden_branch = not any(
        token.casefold() in model_code for token in forbidden_tokens
    )
    key_columns = ["fold_id", "station_name", "reach_id", "year", "month"]
    target = {
        "median_log_nse": 0.7725354050175778,
        "median_kge": 0.7695754842481164,
        "median_lowflow_log_error": 0.24079193257880904,
        "good_count": 80,
    }
    checks = {
        "runtime_is_exact_sparrow": (
            RUNTIME["sys_prefix"].casefold()
            == RUNTIME["expected_prefix"].casefold()
        ),
        "authoritative_input_snapshot_local": (
            RUN.joinpath(
                "inputs", "authoritative_20260728_17", "indata.parquet"
            ).exists()
        ),
        "active_input_built_and_audited": (
            RUN.joinpath("inputs", "indata.parquet").exists()
            and RUN.joinpath(
                "inputs", "source_metadata", "q72_input_build_audit.json"
            ).exists()
        ),
        "topology_is_local": (
            RUN.joinpath(
                "inputs", "topology", "topology_edges.csv"
            ).exists()
        ),
        "three_blocked_folds_complete": len(reproduction) == 3,
        "three_fold_oof_keys_unique": (
            not oof.duplicated(key_columns).any()
        ),
        "reference_keys_exact": bool(
            (
                reproduction["left_only_rows"].eq(0)
                & reproduction["right_only_rows"].eq(0)
            ).all()
        ),
        "prediction_reproduction_below_1e_10_logq": bool(
            reproduction[
                "max_abs_log_prediction_difference"
            ].max()
            < 1e-10
        ),
        "observations_reproduced_exactly": bool(
            reproduction["max_abs_actual_difference_cfs"].max() < 1e-10
        ),
        "confirmation_common_sample_shape_exact": (
            len(confirmation) == 3407
            and confirmation["station_name"].nunique() == 96
        ),
        "confirmation_metrics_reproduced": (
            abs(
                confirmation_summary["median_log_nse"]
                - target["median_log_nse"]
            ) < 1e-10
            and abs(
                confirmation_summary["median_kge"]
                - target["median_kge"]
            ) < 1e-10
            and abs(
                confirmation_summary["median_lowflow_log_error"]
                - target["median_lowflow_log_error"]
            ) < 1e-10
            and confirmation_summary["good_count"] == target["good_count"]
        ),
        "five_fixed_exclusions_absent": not bool(
            active_stations & FIXED_EXCLUSIONS
        ),
        "protected_shijiao_present": (
            PROTECTED in active_stations and len(shijiao) == 1
        ),
        "model_code_has_no_legacy_branch": no_forbidden_branch,
        "complete_design_matrix_saved_each_fold": (
            len(design_inventory) == 3
            and all(
                RUN.joinpath(
                    "reports", "q72_baseline", "blocked_folds",
                    fold_id, "reports", "design_matrix",
                    "augmented_map_design_matrix.npz",
                ).exists()
                for fold_id in FOLDS
            )
        ),
        "parameters_saved_each_fold": (
            len(pd.DataFrame(parameter_inventory)["fold_id"].unique()) == 3
        ),
        "provenance_and_environment_saved": (
            len(provenance_rows) > 0
            and MANIFEST.joinpath("environment.json").exists()
        ),
        "later_period_not_used_for_fit_or_score": (
            int(oof["year"].max()) == 2018
            and all(
                int(fold_id.split("_eval_")[0].split("_")[-1])
                in {2011, 2013, 2015}
                for fold_id in FOLDS
            )
        ),
    }
    checks = {key: bool(value) for key, value in checks.items()}
    gate = {
        "run_id": RUN.name,
        "phase": "A0_Q72_ONLY_BASELINE_REPRODUCTION",
        "checks": checks,
        "passed": all(checks.values()),
        "passed_count": int(sum(checks.values())),
        "check_count": int(len(checks)),
        "decision": (
            "FREEZE_Q72_B0_BASELINE"
            if all(checks.values())
            else "DO_NOT_FREEZE_Q72_B0"
        ),
        "confirmation_common_sample": confirmation_summary,
        "three_fold_oof": combined_summary,
        "maximum_abs_logq_reproduction_error": float(
            reproduction["max_abs_log_prediction_difference"].max()
        ),
        "q72_only": True,
        "later_improvement_stages_executed": False,
    }
    (REPORT / "gate.json").write_text(
        json.dumps(gate, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report_lines = [
        "# Q72-B0 独立阻塞折基线复刻报告",
        "",
        f"- 决策：`{gate['decision']}`",
        f"- 独立检查：{gate['passed_count']}/{gate['check_count']}",
        f"- 最大预测复刻误差：{gate['maximum_abs_logq_reproduction_error']:.3e} logQ",
        "- 三折：fit≤2011→2012–2013；fit≤2013→2014–2015；"
        "fit≤2015→2016–2018",
        "- 后续拟合方法和水文方程实验：未执行",
        "",
        "## 2016–2018共同样本",
        "",
        f"- stations：{confirmation_summary['station_count']}",
        f"- median log-NSE：{confirmation_summary['median_log_nse']:.12f}",
        f"- median KGE：{confirmation_summary['median_kge']:.12f}",
        f"- median |PBIAS|：{confirmation_summary['median_absolute_pbias_pct']:.6f}%",
        f"- low-flow log error：{confirmation_summary['median_lowflow_log_error']:.12f}",
        f"- good：{confirmation_summary['good_count']}",
        "",
        "该目录的模型运行仅依赖本地 `inputs/`，不调用历史预测分支。",
    ]
    (REPORT / "technical_report.md").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(gate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
