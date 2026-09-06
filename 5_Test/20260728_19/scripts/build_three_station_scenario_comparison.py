from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


EPS = 1.0e-6
RUN = Path(__file__).resolve().parents[1]
TEST_ROOT = RUN.parent
BASELINES = {
    "no_dynamic_exclusions": TEST_ROOT / "20260721_1",
    "five_station_parent": TEST_ROOT / "20260721_206",
}
EXCLUDED = ["劳村站", "富罗（二）站", "隆安站"]
RESTORED = ["迁江站", "都安（二）站"]
PREDICTION_REL = Path("reports") / "main_model" / "reach_class_selected_predictions_long.csv"


def metric(frame: pd.DataFrame) -> dict[str, float | int | bool]:
    obs = pd.to_numeric(frame["Q_obsv_cfs"], errors="coerce").to_numpy(float)
    pred = pd.to_numeric(frame["Q_pred_cfs"], errors="coerce").to_numpy(float)
    keep = np.isfinite(obs) & np.isfinite(pred) & (obs > 0) & (pred > 0)
    obs, pred = obs[keep], pred[keep]
    if len(obs) < 3:
        return {"NSElog": np.nan, "KGE": np.nan, "PBIAS_pct": np.nan, "good": False}
    lo, lp = np.log(obs + EPS), np.log(pred + EPS)
    denom = float(np.sum((lo - lo.mean()) ** 2))
    nselog = np.nan if denom <= 0 else float(1.0 - np.sum((lp - lo) ** 2) / denom)
    if np.std(obs) <= 0 or np.std(pred) <= 0 or np.mean(obs) == 0:
        kge = np.nan
    else:
        corr = float(np.corrcoef(obs, pred)[0, 1])
        alpha = float(np.std(pred) / np.std(obs))
        beta = float(np.mean(pred) / np.mean(obs))
        kge = float(1.0 - np.sqrt((corr - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))
    pbias = float(100.0 * np.sum(pred - obs) / np.sum(obs))
    good = bool(len(obs) >= 24 and nselog >= 0.65 and kge >= 0.5 and abs(pbias) <= 25.0)
    return {"NSElog": nselog, "KGE": kge, "PBIAS_pct": pbias, "good": good}


def station_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([{"q_site": str(site), **metric(part)} for site, part in frame.groupby("q_site", sort=True)])


def summarize(frame: pd.DataFrame, prefix: str) -> dict[str, float | int]:
    return {
        f"{prefix}_station_count": int(len(frame)),
        f"{prefix}_mean_NSElog": float(frame["NSElog"].mean()),
        f"{prefix}_median_NSElog": float(frame["NSElog"].median()),
        f"{prefix}_mean_KGE": float(frame["KGE"].mean()),
        f"{prefix}_median_KGE": float(frame["KGE"].median()),
        f"{prefix}_mean_absPBIAS": float(frame["PBIAS_pct"].abs().mean()),
        f"{prefix}_median_absPBIAS": float(frame["PBIAS_pct"].abs().median()),
        f"{prefix}_good_count": int(frame["good"].sum()),
        f"{prefix}_good_rate": float(frame["good"].mean()),
    }


def markdown_table(frame: pd.DataFrame, float_columns: set[str] | None = None) -> str:
    float_columns = float_columns or set()
    headers = [str(column) for column in frame.columns]
    rows = []
    for values in frame.itertuples(index=False, name=None):
        cells = []
        for header, value in zip(headers, values):
            if header in float_columns and isinstance(value, (float, np.floating)):
                cells.append(f"{value:.6f}")
            else:
                cells.append(str(value))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |", *rows])


def comparison_rows(label: str, baseline: pd.DataFrame, trial: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scope, start, end in (("selection_2016_2018", 2016, 2018), ("confirmation_2019_2022", 2019, 2022)):
        base_scope = baseline[baseline["year"].between(start, end)]
        trial_scope = trial[trial["year"].between(start, end)]
        common = sorted(set(base_scope["q_site"]) & set(trial_scope["q_site"]))
        base_metrics = station_metrics(base_scope[base_scope["q_site"].isin(common)])
        trial_metrics = station_metrics(trial_scope[trial_scope["q_site"].isin(common)])
        row: dict[str, object] = {"comparison": label, "scope": scope, **summarize(base_metrics, "baseline_common"), **summarize(trial_metrics, "three_station_common")}
        for name in ("mean_NSElog", "median_NSElog", "mean_KGE", "median_KGE", "mean_absPBIAS", "median_absPBIAS", "good_count", "good_rate"):
            row[f"delta_{name}"] = float(row[f"three_station_common_{name}"]) - float(row[f"baseline_common_{name}"])
        rows.append(row)
    return rows


def main() -> None:
    out = RUN / "reports" / "three_station_scenario_comparison"
    out.mkdir(parents=True, exist_ok=True)
    trial = pd.read_csv(RUN / PREDICTION_REL, encoding="utf-8-sig")
    trial["q_site"] = trial["q_site"].astype(str)
    rows: list[dict[str, object]] = []
    for label, folder in BASELINES.items():
        baseline = pd.read_csv(folder / PREDICTION_REL, encoding="utf-8-sig")
        baseline["q_site"] = baseline["q_site"].astype(str)
        rows.extend(comparison_rows(label, baseline, trial))
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "common_station_summary.csv", index=False, encoding="utf-8-sig")

    policy = pd.read_csv(RUN / "inputs" / "source_metadata" / "station_screening_policy.csv", encoding="utf-8-sig")
    inputs = pd.read_parquet(RUN / "inputs" / "indata.parquet", columns=["q_site", "station_id"])
    folds = RUN / "reports" / "station_screening" / "blocked_folds"
    fold_frames = [pd.read_csv(path, encoding="utf-8-sig") for path in folds.glob("*/evaluation_predictions.csv")]
    fold_manifest = pd.read_csv(RUN / "reports" / "station_screening" / "blocked_fold_manifest.csv", encoding="utf-8-sig")
    if len(fold_manifest) != 3 or not fold_manifest["returncode"].eq(0).all():
        raise RuntimeError("The three blocked Q72 folds did not all complete successfully")
    audit_rows = []
    for station in EXCLUDED + RESTORED:
        policy_row = policy[policy["station_name"].astype(str).eq(station)].iloc[0]
        input_rows = int(inputs["q_site"].astype(str).eq(station).sum() + inputs["station_id"].astype(str).eq(station).sum())
        main_rows = int(trial["q_site"].eq(station).sum())
        blocked_rows = int(sum(frame["q_site"].astype(str).eq(station).sum() for frame in fold_frames))
        audit_rows.append({
            "station_name": station,
            "role": "excluded" if station in EXCLUDED else "restored_control",
            "exclude_before_training": str(policy_row["exclude_before_training"]),
            "input_rows": input_rows,
            "main_prediction_rows": main_rows,
            "blocked_q72_rows": blocked_rows,
            "participation_expectation_passed": bool((station in EXCLUDED and input_rows == main_rows == blocked_rows == 0) or (station in RESTORED and input_rows > 0 and main_rows > 0 and blocked_rows > 0)),
        })
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(out / "participation_audit.csv", index=False, encoding="utf-8-sig")
    excluded_passed = bool(audit[audit["role"].eq("excluded")]["participation_expectation_passed"].all())
    restored_passed = bool(audit[audit["role"].eq("restored_control")]["participation_expectation_passed"].all())

    display = summary.copy()
    for column in ["baseline_common_station_count", "three_station_common_station_count", "delta_good_count"]:
        display[column] = display[column].astype(int)
    report = [
        "# 20260727_1 Three-Station Scenario Check",
        "",
        "## Scenario boundary",
        "",
        "- excluded before input construction: 劳村站、富罗（二）站、隆安站",
        "- deliberately retained: 石角站、迁江站、都安（二）站",
        f"- zero participation for all three exclusions: {excluded_passed}",
        f"- restored controls participate in input, main prediction and all blocked folds: {restored_passed}",
        "",
        "## Common-station comparison",
        "",
        "Positive NSElog/KGE/good deltas and negative absolute-PBIAS deltas are improvements. `20260721_1` measures the total three-station-only scenario effect; `20260721_206` isolates the effect of restoring 迁江站 and 都安（二）站 relative to the current five-station parent.",
        "",
        markdown_table(display, {column for column in display.columns if column.startswith(("baseline_", "three_", "delta_")) and column not in {"baseline_common_station_count", "three_station_common_station_count", "delta_good_count"}}),
        "",
        "## Participation audit",
        "",
        markdown_table(audit),
        "",
        "## Blocked Q72 workflow log",
        "",
        markdown_table(fold_manifest[["fold_id", "train_end", "eval_start", "eval_end", "returncode", "stations", "evaluation_rows", "elapsed_seconds"]], {"elapsed_seconds"}),
        "",
        "2019–2022 is confirmation-only and is not used to select the station set.",
    ]
    text = "\n".join(report) + "\n"
    (out / "comparison_report.md").write_text(text, encoding="utf-8")
    (RUN / "logs" / "three_station_scenario_check.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
