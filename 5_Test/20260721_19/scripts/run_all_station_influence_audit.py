from __future__ import annotations

import gc
import hashlib
import importlib.util
import io
import json
import shutil
import time
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd


RUN = Path(__file__).resolve().parents[1]
TEST_ROOT = RUN.parent
CONTROL = TEST_ROOT / "20260721_1" / "reports" / "dynamic_station_screening"
EXPERIMENT = json.loads((RUN / "inputs" / "source_metadata" / "run_experiment.json").read_text(encoding="utf-8"))
PARENT = TEST_ROOT / str(EXPERIMENT["parent_run"])
REFERENCE = TEST_ROOT / str(EXPERIMENT["metric_reference_run"])
COMPONENT = RUN / "scripts" / "components" / "fit_monthly_bayes_seasonal_hysteresis.py"
OUT = RUN / "reports" / "station_influence_audit"
TEMP = OUT / "_temporary_model_outputs"
RESULTS_PATH = OUT / "audit_results.csv"
EPS = 1.0e-6
FOLDS = (
    ("fit_2006_2011_eval_2012_2013", 2011, 2012, 2013),
    ("fit_2006_2013_eval_2014_2015", 2013, 2014, 2015),
    ("fit_2006_2015_eval_2016_2018", 2015, 2016, 2018),
)
MANUAL_RESERVOIR_NAME_TOKENS = ("水库", "坝上", "坝下", "渠道", "大坝")
FIXED = {
    "rho": 0.70,
    "wm": 480.0,
    "et_gamma": 0.75,
    "sas_rho": 0.93,
    "young_k": 1.5,
    "storage_scale": 720.0,
    "prod_capacity": 240.0,
    "runoff_gamma": 2.5,
    "quick_rho": 0.25,
    "base_rho": 0.85,
    "base_release": 0.10,
    "fixed_sigma": 3.0,
    "production_sigma": 1.5,
    "group_sigma": 1.5,
    "multistore_sigma": 0.30,
    "hysteresis_sigma": 3.0,
    "station_sigma": 1.0,
    "slope_sigma": 0.15,
    "regime_slope_sigma": 0.25,
    "anomaly_weight": 0.0,
    "flow_contrast_weight": 1.0,
}


def load_component(tag: str):
    name = f"influence_{hashlib.sha1(tag.encode('utf-8')).hexdigest()[:12]}"
    spec = importlib.util.spec_from_file_location(name, COMPONENT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {COMPONENT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metric(frame: pd.DataFrame) -> dict[str, float | int | bool]:
    obs = pd.to_numeric(frame["actual"], errors="coerce").to_numpy(float)
    pred = pd.to_numeric(frame["predict"], errors="coerce").to_numpy(float)
    use = np.isfinite(obs) & np.isfinite(pred) & (obs > 0) & (pred > 0)
    obs, pred = obs[use], pred[use]
    n = int(len(obs))
    if n < 3:
        return {"n": n, "NSElog": np.nan, "KGE": np.nan, "PBIAS_pct": np.nan, "good": False}
    lo, lp = np.log(obs + EPS), np.log(pred + EPS)
    denom = float(np.sum((lo - lo.mean()) ** 2))
    nselog = np.nan if denom <= 0 else float(1.0 - np.sum((lp - lo) ** 2) / denom)
    if np.std(obs) <= 0 or np.std(pred) <= 0 or np.mean(obs) == 0:
        kge = np.nan
    else:
        corr = float(np.corrcoef(obs, pred)[0, 1])
        variability = float(np.std(pred) / np.std(obs))
        volume = float(np.mean(pred) / np.mean(obs))
        kge = float(1.0 - np.sqrt((corr - 1.0) ** 2 + (variability - 1.0) ** 2 + (volume - 1.0) ** 2))
    pbias = float(100.0 * np.sum(pred - obs) / np.sum(obs))
    good = bool(n >= 24 and np.isfinite(nselog) and np.isfinite(kge) and nselog >= 0.65 and kge >= 0.50 and abs(pbias) <= 25.0)
    return {"n": n, "NSElog": nselog, "KGE": kge, "PBIAS_pct": pbias, "good": good}


def station_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([{"q_site": str(site), **metric(part)} for site, part in frame.groupby("q_site", sort=True)])


def fold_delta(base: pd.DataFrame, trial: pd.DataFrame) -> dict[str, float | int]:
    common = sorted(set(base["q_site"].astype(str)) & set(trial["q_site"].astype(str)))
    base_m = station_metrics(base[base["q_site"].astype(str).isin(common)]).set_index("q_site").loc[common]
    trial_m = station_metrics(trial[trial["q_site"].astype(str).isin(common)]).set_index("q_site").loc[common]
    return {
        "common_stations": int(len(common)),
        "delta_good": int(trial_m["good"].sum() - base_m["good"].sum()),
        "delta_median_NSElog": float(trial_m["NSElog"].median() - base_m["NSElog"].median()),
        "delta_median_KGE": float(trial_m["KGE"].median() - base_m["KGE"].median()),
        "delta_mean_absPBIAS": float(trial_m["PBIAS_pct"].abs().mean() - base_m["PBIAS_pct"].abs().mean()),
    }


def fixed_choice(_frame: pd.DataFrame) -> tuple[dict[str, object], pd.DataFrame]:
    row = {**FIXED, "selection": "fixed_from_20260620_44"}
    return row, pd.DataFrame([row])


def load_base_observed() -> pd.DataFrame:
    module = load_component("base_observed")
    module.INPUT_PATH = PARENT / "inputs" / "indata.parquet"
    return module.load_observed_panel()


def same_reach_alternative_ids() -> set[int]:
    path = PARENT / "reports" / "input_preprocessing" / "same_reach_station_reliability.csv"
    raw = pd.read_csv(path, encoding="utf-8-sig")
    counts = raw.groupby("reach_id").size()
    return set(pd.to_numeric(counts[counts > 1].index, errors="coerce").dropna().astype(int))


def audit_station(site: str, observed: pd.DataFrame, base_predictions: dict[str, pd.DataFrame]) -> tuple[dict[str, object], pd.DataFrame]:
    started = time.perf_counter()
    filtered = observed[~observed["q_site"].astype(str).eq(site)].copy().reset_index(drop=True)
    fold_rows = []
    minimal_parts = []
    site_dir = OUT / "station_predictions" / hashlib.sha1(site.encode("utf-8")).hexdigest()[:12]
    site_dir.mkdir(parents=True, exist_ok=True)
    for fold_id, train_end, eval_start, eval_end in FOLDS:
        module = load_component(f"{site}:{fold_id}")
        report_dir = TEMP / hashlib.sha1(f"{site}:{fold_id}".encode("utf-8")).hexdigest()[:16]
        figure_dir = report_dir / "figure"
        module.REPORT_DIR = report_dir
        module.FIG_DIR = figure_dir
        module.CAL_END_YEAR = train_end
        module.INNER_TRAIN_END_YEAR = train_end
        module.choose_hyperparameters = fixed_choice
        module.load_observed_panel = lambda panel=filtered: panel.copy()
        module.setup_style = lambda: None
        module.write_readme = lambda *_args, **_kwargs: None
        with redirect_stdout(io.StringIO()):
            module.main()
        prediction_path = report_dir / "monthly_bayes_seasonal_hysteresis_prediction_vs_observed_2006_2022.csv"
        prediction = pd.read_csv(prediction_path, encoding="utf-8-sig")
        evaluation = prediction[prediction["year"].between(eval_start, eval_end)].copy()
        evaluation["q_site"] = evaluation["q_site"].astype(str)
        delta = fold_delta(base_predictions[fold_id], evaluation)
        fold_rows.append({"fold_id": fold_id, **delta})
        minimal = evaluation[["q_site", "year", "month", "actual", "predict"]].copy()
        minimal["fold_id"] = fold_id
        minimal_parts.append(minimal)
        shutil.rmtree(report_dir)
        del module, prediction, evaluation
        gc.collect()
    pd.concat(minimal_parts, ignore_index=True).to_parquet(site_dir / "evaluation_predictions_minimal.parquet", index=False)
    fold_frame = pd.DataFrame(fold_rows)
    positive_good_folds = int((fold_frame["delta_good"] > 0).sum())
    cumulative_good_gain = int(fold_frame["delta_good"].sum())
    mean_nse = float(fold_frame["delta_median_NSElog"].mean())
    mean_kge = float(fold_frame["delta_median_KGE"].mean())
    mean_pbias = float(fold_frame["delta_mean_absPBIAS"].mean())
    result = {
        "q_site": site,
        "folds_completed": 3,
        "positive_good_folds": positive_good_folds,
        "cumulative_good_gain": cumulative_good_gain,
        "mean_delta_median_NSElog": mean_nse,
        "mean_delta_median_KGE": mean_kge,
        "mean_delta_mean_absPBIAS": mean_pbias,
        "max_fold_good_loss": int(-min(0, int(fold_frame["delta_good"].min()))),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "prediction_evidence": str((site_dir / "evaluation_predictions_minimal.parquet").relative_to(RUN)),
    }
    return result, fold_frame.assign(q_site=site)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    TEMP.mkdir(parents=True, exist_ok=True)
    evidence = pd.read_csv(REFERENCE / "reports" / "station_screening" / "station_evidence_matrix.csv", encoding="utf-8-sig")
    evidence["q_site"] = evidence["q_site"].astype(str)
    deferred_by_topology = evidence["reservoir_deferred"].astype(str).str.lower().isin({"true", "1", "yes"})
    deferred_by_name = evidence["q_site"].astype(str).apply(
        lambda name: any(token in name for token in MANUAL_RESERVOIR_NAME_TOKENS)
    )
    evidence = evidence[~(deferred_by_topology | deferred_by_name)].copy()
    decision_ledger = pd.read_csv(CONTROL / "decision_ledger.csv", encoding="utf-8-sig")
    tested = set(
        decision_ledger.loc[
            decision_ledger["policy_sha256"].astype(str).eq(str(EXPERIMENT["accepted_policy_sha256"])), "candidate_station"
        ].astype(str)
    )
    remaining = evidence[~evidence["q_site"].isin(tested)].sort_values("q_site").copy()
    expected_sites = remaining["q_site"].tolist()
    prior = pd.read_csv(RESULTS_PATH, encoding="utf-8-sig") if RESULTS_PATH.exists() else pd.DataFrame()
    completed = set(prior.loc[prior.get("status", pd.Series(dtype=str)).eq("completed"), "q_site"].astype(str)) if not prior.empty else set()
    observed = load_base_observed()
    base_predictions = {}
    for fold_id, _train_end, _eval_start, _eval_end in FOLDS:
        path = REFERENCE / "reports" / "station_screening" / "blocked_folds" / fold_id / "evaluation_predictions.csv"
        frame = pd.read_csv(path, encoding="utf-8-sig")
        frame["q_site"] = frame["q_site"].astype(str)
        base_predictions[fold_id] = frame
    alternative_reaches = same_reach_alternative_ids()
    fold_detail_parts = []
    total = len(expected_sites)
    for index, row in enumerate(remaining.itertuples(index=False), start=1):
        site = str(row.q_site)
        if site in completed:
            print(f"[{index}/{total}] resume-skip {site}", flush=True)
            continue
        try:
            result, fold_detail = audit_station(site, observed, base_predictions)
            same_reach = int(row.reach_id) in alternative_reaches
            broad_signal = bool(
                bool(row.persistent_model_failure)
                or
                same_reach
                or (result["positive_good_folds"] >= 2 and result["cumulative_good_gain"] >= 1)
                or (result["mean_delta_median_NSElog"] >= 0.002 and result["mean_delta_median_KGE"] >= 0.0)
                or (result["mean_delta_median_KGE"] >= 0.002 and result["mean_delta_median_NSElog"] >= 0.0)
            )
            result.update(
                {
                    "status": "completed",
                    "reach_id": int(row.reach_id),
                    "reach_class": str(row.reach_class),
                    "persistent_model_failure": bool(row.persistent_model_failure),
                    "severe_data_flag_count": int(row.severe_data_flag_count),
                    "same_reach_alternative": same_reach,
                    "escalate_full_ablation": broad_signal,
                    "quality_classification": "needs_full_ablation" if broad_signal else "no_blocked_influence_signal",
                    "error": "",
                }
            )
            fold_detail_parts.append(fold_detail)
            print(
                f"[{index}/{total}] {site}: good={result['cumulative_good_gain']:+d}, "
                f"dNSE={result['mean_delta_median_NSElog']:+.5f}, dKGE={result['mean_delta_median_KGE']:+.5f}, "
                f"escalate={broad_signal}",
                flush=True,
            )
        except Exception as exc:
            result = {
                "q_site": site,
                "status": "error",
                "reach_id": int(row.reach_id),
                "reach_class": str(row.reach_class),
                "persistent_model_failure": bool(row.persistent_model_failure),
                "severe_data_flag_count": int(row.severe_data_flag_count),
                "same_reach_alternative": int(row.reach_id) in alternative_reaches,
                "escalate_full_ablation": True,
                "quality_classification": "audit_error_requires_review",
                "error": repr(exc),
            }
            print(f"[{index}/{total}] ERROR {site}: {exc!r}", flush=True)
        prior = pd.concat([prior[~prior["q_site"].astype(str).eq(site)] if not prior.empty else prior, pd.DataFrame([result])], ignore_index=True)
        prior.to_csv(RESULTS_PATH, index=False, encoding="utf-8-sig")
        if fold_detail_parts:
            existing_detail_path = OUT / "fold_deltas.csv"
            detail = pd.read_csv(existing_detail_path, encoding="utf-8-sig") if existing_detail_path.exists() else pd.DataFrame()
            new_detail = pd.concat(fold_detail_parts, ignore_index=True)
            detail = pd.concat([detail[~detail["q_site"].astype(str).isin(new_detail["q_site"].astype(str))] if not detail.empty else detail, new_detail], ignore_index=True)
            detail.to_csv(existing_detail_path, index=False, encoding="utf-8-sig")
            fold_detail_parts.clear()
    results = pd.read_csv(RESULTS_PATH, encoding="utf-8-sig")
    results = results[results["q_site"].astype(str).isin(expected_sites)].copy()
    persistent_mask = results["persistent_model_failure"].astype(str).str.lower().isin({"true", "1", "yes"})
    old_escalation = results["escalate_full_ablation"].astype(str).str.lower().isin({"true", "1", "yes"})
    results["escalate_full_ablation"] = old_escalation | persistent_mask
    results.loc[results["escalate_full_ablation"], "quality_classification"] = "needs_full_ablation"
    results.to_csv(RESULTS_PATH, index=False, encoding="utf-8-sig")
    detail_path = OUT / "fold_deltas.csv"
    if detail_path.exists():
        detail = pd.read_csv(detail_path, encoding="utf-8-sig")
        detail[detail["q_site"].astype(str).isin(expected_sites)].to_csv(detail_path, index=False, encoding="utf-8-sig")
    errors = results[results["status"].astype(str).eq("error")].copy()
    candidates = results[results["escalate_full_ablation"].astype(str).str.lower().isin({"true", "1", "yes"})].copy()
    candidates["audit_priority"] = (
        100 * candidates["same_reach_alternative"].astype(str).str.lower().isin({"true", "1", "yes"}).astype(int)
        + 20 * pd.to_numeric(candidates["positive_good_folds"], errors="coerce").fillna(0)
        + 5 * pd.to_numeric(candidates["cumulative_good_gain"], errors="coerce").fillna(0)
        + 1000 * pd.to_numeric(candidates["mean_delta_median_NSElog"], errors="coerce").fillna(0)
        + 1000 * pd.to_numeric(candidates["mean_delta_median_KGE"], errors="coerce").fillna(0)
    )
    candidates.sort_values("audit_priority", ascending=False).to_csv(OUT / "full_ablation_candidates.csv", index=False, encoding="utf-8-sig")
    coverage = evidence[["q_site", "reach_id", "reach_class", "reservoir_deferred", "persistent_model_failure", "severe_data_flag_count"]].merge(
        results[["q_site", "status", "quality_classification", "escalate_full_ablation"]], on="q_site", how="left"
    )
    coverage["previous_full_ablation"] = coverage["q_site"].isin(tested)
    coverage["coverage_complete"] = coverage["previous_full_ablation"] | coverage["status"].eq("completed")
    coverage.to_csv(OUT / "nonreservoir_station_coverage.csv", index=False, encoding="utf-8-sig")
    summary = {
        "run_id": RUN.name,
        "accepted_parent_run": PARENT.name,
        "policy_sha256": str(EXPERIMENT["accepted_policy_sha256"]),
        "nonreservoir_stations_total": int(len(evidence)),
        "previous_full_ablation_stations": int(len(tested & set(evidence["q_site"]))),
        "stations_expected_in_influence_audit": int(len(expected_sites)),
        "stations_completed_in_influence_audit": int(results["status"].astype(str).eq("completed").sum()),
        "audit_errors": int(len(errors)),
        "full_ablation_candidates": int(len(candidates)),
        "all_nonreservoir_covered": bool(coverage["coverage_complete"].all() and errors.empty),
    }
    (OUT / "audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    shutil.rmtree(TEMP, ignore_errors=True)
    readme = (RUN / "README.md").read_text(encoding="utf-8")
    if "\n## Audit result\n" in readme:
        readme = readme.split("\n## Audit result\n", 1)[0].rstrip() + "\n"
    readme += f"""

## Audit result

- active non-reservoir stations: {summary['nonreservoir_stations_total']}
- already covered by exact full ablation: {summary['previous_full_ablation_stations']}
- stations covered by this three-fold influence audit: {summary['stations_completed_in_influence_audit']}
- audit errors: {summary['audit_errors']}
- escalated to exact full Q72+Q78 ablation: {summary['full_ablation_candidates']}
- all non-reservoir stations covered: {summary['all_nonreservoir_covered']}

Evidence is under `reports/station_influence_audit`. A blocked-fold signal is not an exclusion decision; it only creates a later full-ablation candidate. Raw-data flags, model influence, and final exclusion status remain separate fields.
"""
    (RUN / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if not summary["all_nonreservoir_covered"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
