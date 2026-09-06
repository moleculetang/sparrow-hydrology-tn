from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


EPS = 1.0e-6
TEST_ROOT = Path(r"E:\SPARROW\5_Test")
CONTROL_RUN = TEST_ROOT / "20260721_1"
STATE_PATH = CONTROL_RUN / "reports" / "dynamic_station_screening" / "chain_state.json"
DECISION_POLICY_PATH = CONTROL_RUN / "inputs" / "source_metadata" / "station_screening_decision_policy_v2.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the frozen station-set and confirmation-only final report.")
    parser.add_argument("--final-run", required=True)
    parser.add_argument("--baseline-run", default="20260721_1")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def metric_dict(frame: pd.DataFrame) -> dict[str, float | int | bool]:
    thresholds = read_json(DECISION_POLICY_PATH)["good_thresholds"]
    obs = pd.to_numeric(frame["Q_obsv_cfs"], errors="coerce").to_numpy(float)
    pred = pd.to_numeric(frame["Q_pred_cfs"], errors="coerce").to_numpy(float)
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
    good = bool(
        n >= int(thresholds["minimum_months"])
        and np.isfinite(nselog)
        and np.isfinite(kge)
        and nselog >= float(thresholds["NSElog_min"])
        and kge >= float(thresholds["KGE_min"])
        and abs(pbias) <= float(thresholds["abs_PBIAS_pct_max"])
    )
    return {"n": n, "NSElog": nselog, "KGE": kge, "PBIAS_pct": pbias, "good": good}


def station_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([{"q_site": str(site), **metric_dict(part)} for site, part in frame.groupby("q_site", sort=True)])


def summarize(metrics: pd.DataFrame, prefix: str) -> dict[str, float | int]:
    station_count = int(len(metrics))
    good_count = int(metrics["good"].astype(bool).sum()) if station_count else 0
    return {
        f"{prefix}_station_count": station_count,
        f"{prefix}_mean_NSElog": float(metrics["NSElog"].mean()) if station_count else np.nan,
        f"{prefix}_median_NSElog": float(metrics["NSElog"].median()) if station_count else np.nan,
        f"{prefix}_mean_KGE": float(metrics["KGE"].mean()) if station_count else np.nan,
        f"{prefix}_median_KGE": float(metrics["KGE"].median()) if station_count else np.nan,
        f"{prefix}_mean_absPBIAS": float(metrics["PBIAS_pct"].abs().mean()) if station_count else np.nan,
        f"{prefix}_median_absPBIAS": float(metrics["PBIAS_pct"].abs().median()) if station_count else np.nan,
        f"{prefix}_good_count": good_count,
        f"{prefix}_good_rate": float(good_count / station_count) if station_count else np.nan,
    }


def main() -> None:
    args = parse_args()
    final_run = TEST_ROOT / args.final_run
    baseline_run = TEST_ROOT / args.baseline_run
    output = final_run / "reports" / "final_station_screening"
    output.mkdir(parents=True, exist_ok=True)

    state = read_json(STATE_PATH)
    if int(state.get("stable_full_scan_count", 0)) < 2:
        raise RuntimeError("The final report requires two completed stable full scans")
    policy_path = final_run / "inputs" / "source_metadata" / "station_screening_policy.csv"
    policy = pd.read_csv(policy_path, encoding="utf-8-sig")
    excluded = policy[as_bool(policy["exclude_before_training"]) & ~as_bool(policy["reservoir_deferred"])].copy()
    excluded_names = sorted(excluded["station_name"].astype(str))
    evidence_path = (
        TEST_ROOT
        / str(state["accepted_parent_run"])
        / "reports"
        / "station_screening"
        / "station_evidence_matrix.csv"
    )
    evidence = pd.read_csv(evidence_path, encoding="utf-8-sig")
    reservoirs = evidence[as_bool(evidence["reservoir_deferred"])].copy()
    reservoirs["station_name"] = reservoirs["q_site"].astype(str)
    reservoirs["station_status"] = "defer_reservoir"
    reservoirs["decision"] = "deferred_outside_nonreservoir_screening_scope"

    final_predictions_path = final_run / "reports" / "main_model" / "reach_class_selected_predictions_long.csv"
    baseline_predictions_path = baseline_run / "reports" / "main_model" / "reach_class_selected_predictions_long.csv"
    final_predictions = pd.read_csv(final_predictions_path, encoding="utf-8-sig")
    baseline_predictions = pd.read_csv(baseline_predictions_path, encoding="utf-8-sig")
    final_predictions["q_site"] = final_predictions["q_site"].astype(str)
    baseline_predictions["q_site"] = baseline_predictions["q_site"].astype(str)

    input_path = final_run / "inputs" / "indata.parquet"
    input_panel = pd.read_parquet(input_path, columns=["q_site", "station_id"])
    input_panel["q_site"] = input_panel["q_site"].astype(str)
    input_panel["station_id"] = input_panel["station_id"].astype(str)
    zero_rows = []
    for station in excluded_names:
        input_rows = int(input_panel["q_site"].eq(station).sum() + input_panel["station_id"].eq(station).sum())
        prediction_rows = int(final_predictions["q_site"].eq(station).sum())
        zero_rows.append(
            {
                "station_name": station,
                "exclude_before_training": True,
                "input_rows": input_rows,
                "main_Q72_Q78_rows": prediction_rows,
                "zero_participation_passed": bool(input_rows == 0 and prediction_rows == 0),
            }
        )
    zero = pd.DataFrame(zero_rows)
    zero_passed = bool(not zero.empty and zero["zero_participation_passed"].all())
    if not zero_passed:
        raise RuntimeError("At least one frozen exclusion still participates in final inputs or predictions")

    scope_rows: list[dict[str, object]] = []
    station_rows: list[pd.DataFrame] = []
    for scope, start, end in (("selection_2016_2018", 2016, 2018), ("confirmation_2019_2022", 2019, 2022)):
        final_scope = final_predictions[final_predictions["year"].between(start, end)].copy()
        baseline_scope = baseline_predictions[baseline_predictions["year"].between(start, end)].copy()
        final_active = station_metrics(final_scope)
        common_sites = sorted(set(final_scope["q_site"]) & set(baseline_scope["q_site"]))
        final_common = station_metrics(final_scope[final_scope["q_site"].isin(common_sites)]).set_index("q_site")
        baseline_common = station_metrics(baseline_scope[baseline_scope["q_site"].isin(common_sites)]).set_index("q_site")
        common_sites = sorted(set(final_common.index) & set(baseline_common.index))
        final_common = final_common.loc[common_sites]
        baseline_common = baseline_common.loc[common_sites]
        row: dict[str, object] = {
            "scope": scope,
            "baseline_run": baseline_run.name,
            "final_run": final_run.name,
            **summarize(baseline_common.reset_index(), "baseline_common"),
            **summarize(final_common.reset_index(), "final_common"),
            **summarize(final_active, "final_active"),
        }
        for metric in (
            "mean_NSElog",
            "median_NSElog",
            "mean_KGE",
            "median_KGE",
            "mean_absPBIAS",
            "median_absPBIAS",
            "good_count",
            "good_rate",
        ):
            row[f"delta_common_{metric}"] = float(row[f"final_common_{metric}"]) - float(row[f"baseline_common_{metric}"])
        scope_rows.append(row)
        station_delta = pd.DataFrame(
            {
                "scope": scope,
                "q_site": common_sites,
                "baseline_NSElog": baseline_common["NSElog"].to_numpy(),
                "final_NSElog": final_common["NSElog"].to_numpy(),
                "baseline_KGE": baseline_common["KGE"].to_numpy(),
                "final_KGE": final_common["KGE"].to_numpy(),
                "baseline_PBIAS_pct": baseline_common["PBIAS_pct"].to_numpy(),
                "final_PBIAS_pct": final_common["PBIAS_pct"].to_numpy(),
                "baseline_good": baseline_common["good"].astype(bool).to_numpy(),
                "final_good": final_common["good"].astype(bool).to_numpy(),
            }
        )
        station_delta["delta_NSElog"] = station_delta["final_NSElog"] - station_delta["baseline_NSElog"]
        station_delta["delta_KGE"] = station_delta["final_KGE"] - station_delta["baseline_KGE"]
        station_delta["delta_absPBIAS"] = station_delta["final_PBIAS_pct"].abs() - station_delta["baseline_PBIAS_pct"].abs()
        station_delta["good_change"] = station_delta["final_good"].astype(int) - station_delta["baseline_good"].astype(int)
        station_rows.append(station_delta)

    summary = pd.DataFrame(scope_rows)
    station_delta = pd.concat(station_rows, ignore_index=True)
    decision_ledger = pd.read_csv(
        CONTROL_RUN / "reports" / "dynamic_station_screening" / "decision_ledger.csv",
        encoding="utf-8-sig",
    ).fillna("")
    current_decisions = decision_ledger[
        decision_ledger["policy_sha256"].astype(str).eq(str(state["policy_sha256"]))
    ].copy()
    current_rejected = current_decisions[current_decisions["decision"].astype(str).eq("reject_exclusion")].copy()
    current_rejected["run_number"] = current_rejected["trial_run"].astype(str).str.rsplit("_", n=1).str[-1].astype(int)
    current_rejected = current_rejected.sort_values("run_number").drop_duplicates("candidate_station", keep="last")
    rejected_trial = dict(
        zip(current_rejected["candidate_station"].astype(str), current_rejected["trial_run"].astype(str))
    )
    accepted_history = decision_ledger[
        decision_ledger["decision"].astype(str).eq("accept_exclusion")
        & decision_ledger["candidate_station"].astype(str).isin(excluded_names)
    ].copy()
    accepted_history["run_number"] = accepted_history["trial_run"].astype(str).str.rsplit("_", n=1).str[-1].astype(int)
    accepted_history = accepted_history.sort_values("run_number").drop_duplicates("candidate_station", keep="last")
    accepted_route = dict(
        zip(accepted_history["candidate_station"].astype(str), accepted_history["accepted_by"].astype(str))
    )
    accepted_trial = dict(
        zip(accepted_history["candidate_station"].astype(str), accepted_history["trial_run"].astype(str))
    )
    terminal_rows = []
    for row in evidence.to_dict(orient="records"):
        station = str(row["q_site"])
        reservoir_deferred = bool(str(row["reservoir_deferred"]).strip().lower() in {"true", "1", "yes"})
        persistent = bool(str(row.get("persistent_model_failure", "")).strip().lower() in {"true", "1", "yes"})
        severe = int(float(row.get("severe_data_flag_count", 0) or 0))
        if reservoir_deferred:
            final_status = "defer_reservoir"
            evidence_source = "accepted_parent_station_evidence"
        elif station in rejected_trial or persistent or severe > 0:
            final_status = "retain_model_limitation"
            evidence_source = rejected_trial.get(station, "accepted_parent_station_evidence")
        else:
            final_status = "retain_healthy"
            evidence_source = "stable_full_scan_no_signal"
        terminal_rows.append(
            {
                "station_name": station,
                "final_status": final_status,
                "exclude_before_training": False,
                "reservoir_deferred": reservoir_deferred,
                "accepted_by": "",
                "last_exact_trial": rejected_trial.get(station, ""),
                "evidence_source": evidence_source,
                "reason_codes": str(row.get("reason_codes", "")),
                "accepted_parent_run": state["accepted_parent_run"],
                "final_model_run": final_run.name,
                "policy_sha256": state["policy_sha256"],
            }
        )
    for station in excluded_names:
        route = accepted_route.get(station, "")
        final_status = "exclude_negative_contributor" if route == "remove_negative_contributor" else "exclude_model_harmful"
        matched = excluded[excluded["station_name"].astype(str).eq(station)]
        reason_codes = "" if matched.empty else str(matched.iloc[0].get("reason_codes", ""))
        terminal_rows.append(
            {
                "station_name": station,
                "final_status": final_status,
                "exclude_before_training": True,
                "reservoir_deferred": False,
                "accepted_by": route,
                "last_exact_trial": accepted_trial.get(station, ""),
                "evidence_source": accepted_trial.get(station, "locked_group_minimality"),
                "reason_codes": reason_codes,
                "accepted_parent_run": state["accepted_parent_run"],
                "final_model_run": final_run.name,
                "policy_sha256": state["policy_sha256"],
            }
        )
    terminal = pd.DataFrame(terminal_rows).sort_values("station_name").reset_index(drop=True)
    if terminal["station_name"].duplicated().any():
        raise RuntimeError("Final terminal station ledger contains duplicate station names")
    allowed_statuses = {
        "retain_healthy",
        "retain_model_limitation",
        "exclude_data_invalid",
        "exclude_model_harmful",
        "exclude_negative_contributor",
        "defer_reservoir",
    }
    if not set(terminal["final_status"]).issubset(allowed_statuses):
        raise RuntimeError("Final terminal station ledger contains an unsupported status")
    terminal.to_csv(output / "final_station_status_ledger.csv", index=False, encoding="utf-8-sig")
    terminal.to_csv(
        CONTROL_RUN / "reports" / "dynamic_station_screening" / "final_station_status_ledger.csv",
        index=False,
        encoding="utf-8-sig",
    )
    first_flagged = dict(
        zip(excluded["station_name"].astype(str), excluded["first_flagged_run"].astype(str))
    )
    legacy_status = pd.DataFrame(
        {
            "station_name": terminal["station_name"],
            "station_status": terminal["final_status"],
            "exclude_before_training": terminal["exclude_before_training"],
            "reservoir_deferred": terminal["reservoir_deferred"],
            "reason_codes": terminal["reason_codes"],
            "first_flagged_run": terminal["station_name"].map(first_flagged).fillna(""),
            "last_tested_run": terminal.apply(
                lambda row: row["last_exact_trial"]
                or ("20260721_204" if not bool(row["reservoir_deferred"]) else str(state["accepted_parent_run"])),
                axis=1,
            ),
            "evidence_run": terminal["evidence_source"],
            "decision": "final_terminal_status",
        }
    )
    legacy_status.to_csv(
        CONTROL_RUN / "reports" / "dynamic_station_screening" / "station_status_ledger.csv",
        index=False,
        encoding="utf-8-sig",
    )
    status_counts = terminal["final_status"].value_counts().sort_index().to_dict()
    frozen_columns = [
        "station_name",
        "station_status",
        "reason_codes",
        "first_flagged_run",
        "last_tested_run",
        "evidence_run",
        "decision",
    ]
    excluded[frozen_columns].sort_values("station_name").to_csv(
        output / "frozen_exclusion_set.csv", index=False, encoding="utf-8-sig"
    )
    reservoir_columns = [
        "station_name",
        "station_status",
        "reach_id",
        "reach_class",
        "reservoir_relation",
        "reason_codes",
        "decision",
    ]
    reservoirs[reservoir_columns].sort_values("station_name").to_csv(
        output / "deferred_reservoir_stations.csv", index=False, encoding="utf-8-sig"
    )
    zero.to_csv(output / "final_zero_participation_audit.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output / "final_confirmation_summary.csv", index=False, encoding="utf-8-sig")
    station_delta.to_csv(output / "final_common_station_deltas.csv", index=False, encoding="utf-8-sig")

    confirmation = summary[summary["scope"].eq("confirmation_2019_2022")].iloc[0]
    selection = summary[summary["scope"].eq("selection_2016_2018")].iloc[0]
    report = f"""# Frozen Station Set and Final Confirmation

- final run: `{final_run.name}`
- accepted screening parent: `{state['accepted_parent_run']}`
- starting reference: `{baseline_run.name}` (numerical reproduction of `20260620_44`)
- policy hash: `{state['policy_sha256']}`
- stable full scans: `{state['stable_full_scan_count']}`
- frozen non-reservoir exclusions: `{len(excluded_names)}`
- deferred reservoir-related stations: `{len(reservoirs)}`
- stations with a unique terminal status: `{len(terminal)}`
- final zero-participation audit: `{zero_passed}`

## Frozen non-reservoir exclusions

{chr(10).join(f"- {name}" for name in excluded_names)}

## 2016–2018 selection-period final check

- final active stations: {int(selection['final_active_station_count'])}
- final active median NSElog: {float(selection['final_active_median_NSElog']):.6f}
- final active median KGE: {float(selection['final_active_median_KGE']):.6f}
- final active good: {int(selection['final_active_good_count'])}/{int(selection['final_active_station_count'])}
- common-station delta vs `{baseline_run.name}` median NSElog: {float(selection['delta_common_median_NSElog']):+.6f}
- common-station delta vs `{baseline_run.name}` median KGE: {float(selection['delta_common_median_KGE']):+.6f}

## 2019–2022 confirmation only

These years were not used to choose or revise the exclusion set.

- final active stations: {int(confirmation['final_active_station_count'])}
- final active median NSElog: {float(confirmation['final_active_median_NSElog']):.6f}
- final active median KGE: {float(confirmation['final_active_median_KGE']):.6f}
- final active median |PBIAS|: {float(confirmation['final_active_median_absPBIAS']):.6f}
- final active good: {int(confirmation['final_active_good_count'])}/{int(confirmation['final_active_station_count'])}
- common-station delta vs `{baseline_run.name}` median NSElog: {float(confirmation['delta_common_median_NSElog']):+.6f}
- common-station delta vs `{baseline_run.name}` median KGE: {float(confirmation['delta_common_median_KGE']):+.6f}
- common-station delta vs `{baseline_run.name}` mean |PBIAS|: {float(confirmation['delta_common_mean_absPBIAS']):+.6f}
- common-station good-count delta vs `{baseline_run.name}`: {int(confirmation['delta_common_good_count']):+d}

Detailed station deltas and the frozen policy snapshot are in this directory.
"""
    (output / "final_confirmation_report.md").write_text(report, encoding="utf-8")
    result = {
        "final_run": final_run.name,
        "baseline_run": baseline_run.name,
        "accepted_parent_run": state["accepted_parent_run"],
        "policy_sha256": state["policy_sha256"],
        "stable_full_scan_count": int(state["stable_full_scan_count"]),
        "frozen_exclusion_stations": excluded_names,
        "deferred_reservoir_station_count": int(len(reservoirs)),
        "terminal_station_count": int(len(terminal)),
        "terminal_status_counts": {str(key): int(value) for key, value in status_counts.items()},
        "zero_participation_passed": zero_passed,
        "confirmation_scope": "2019-2022_only_not_used_for_selection",
        "confirmation_final_active_station_count": int(confirmation["final_active_station_count"]),
        "confirmation_final_active_good_count": int(confirmation["final_active_good_count"]),
        "confirmation_final_active_median_NSElog": float(confirmation["final_active_median_NSElog"]),
        "confirmation_final_active_median_KGE": float(confirmation["final_active_median_KGE"]),
        "confirmation_final_active_median_absPBIAS": float(confirmation["final_active_median_absPBIAS"]),
    }
    (output / "final_confirmation_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
