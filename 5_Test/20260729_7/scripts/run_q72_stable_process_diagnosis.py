from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd


RUN = Path(__file__).resolve().parents[1]
TEST_ROOT = RUN.parent
OUT = RUN / "reports" / "q72_stable_process_diagnosis"
SOURCE_COMPARISON = (
    TEST_ROOT
    / "20260729_6"
    / "reports"
    / "q72_only_migration_audit"
    / "oof_predictions_comparison.csv"
)
SOURCE_GATE = (
    TEST_ROOT
    / "20260729_6"
    / "reports"
    / "q72_only_migration_audit"
    / "gate.json"
)
DEVELOPMENT_INPUT = TEST_ROOT / "20260728_30" / "inputs" / "indata.parquet"
EPS = 1.0e-6
EXCLUSIONS = {"劳村站", "富罗（二）站", "隆安站", "灵渠（三）站", "马口站"}
SHIJIAO = "石角站"
FOLDS = [
    ("fit_2006_2011_eval_2012_2013", 2011, 2012, 2013),
    ("fit_2006_2013_eval_2014_2015", 2013, 2014, 2015),
    ("fit_2006_2015_eval_2016_2018", 2015, 2016, 2018),
]
FINAL_FOLD = FOLDS[-1][0]
PROCESS_CLASSES = ["water_balance", "low_flow", "timing_process"]


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
            "NSE_log": np.nan,
            "KGE_2012": np.nan,
            "KGE_r": np.nan,
            "KGE_beta": np.nan,
            "KGE_gamma": np.nan,
            "PBIAS_pct": np.nan,
            "log_RMSE": np.nan,
        }
    lo = np.log(obs + EPS)
    lp = np.log(pred + EPS)
    denom_log = float(np.sum((lo - np.mean(lo)) ** 2))
    nse_log = (
        np.nan
        if denom_log <= 0
        else float(1.0 - np.sum((lo - lp) ** 2) / denom_log)
    )
    log_rmse = float(np.sqrt(np.mean((lp - lo) ** 2)))
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
        "NSE_log": nse_log,
        "KGE_2012": kge,
        "KGE_r": r,
        "KGE_beta": beta,
        "KGE_gamma": gamma,
        "PBIAS_pct": pbias,
        "log_RMSE": log_rmse,
    }


def build_training_quantiles() -> pd.DataFrame:
    cols = ["q_site", "year", "month", "Q_obsv_cfs"]
    observed = pd.read_parquet(
        DEVELOPMENT_INPUT,
        columns=cols,
        filters=[("year", "<=", 2018)],
    )
    observed = observed[
        observed["year"].between(2006, 2018)
        & observed["month"].between(1, 12)
        & observed["Q_obsv_cfs"].gt(0)
    ].copy()
    observed["q_site"] = observed["q_site"].astype(str)
    pieces: list[pd.DataFrame] = []
    for fold_id, train_end, _, _ in FOLDS:
        train = observed[observed["year"].le(train_end)].copy()
        quantiles = (
            train.groupby("q_site")["Q_obsv_cfs"]
            .quantile([0.25, 0.75])
            .unstack()
            .reset_index()
            .rename(columns={0.25: "train_q25_check", 0.75: "train_q75_cfs"})
        )
        quantiles["fold_id"] = fold_id
        pieces.append(quantiles)
    return pd.concat(pieces, ignore_index=True)


def build_fold_station_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (fold_id, site), part in frame.groupby(["fold_id", "q_site"], sort=True):
        obs = part["Q_obsv_cfs"].to_numpy()
        pred = part["q72_interaction_lite_cfs"].to_numpy()
        overall = metric_dict(obs, pred)
        low = part[part["Q_obsv_cfs"].le(part["train_q25_cfs"])].copy()
        mid = part[
            part["Q_obsv_cfs"].gt(part["train_q25_cfs"])
            & part["Q_obsv_cfs"].le(part["train_q75_cfs"])
        ].copy()
        low_metrics = metric_dict(
            low["Q_obsv_cfs"].to_numpy(),
            low["q72_interaction_lite_cfs"].to_numpy(),
        )
        mid_metrics = metric_dict(
            mid["Q_obsv_cfs"].to_numpy(),
            mid["q72_interaction_lite_cfs"].to_numpy(),
        )
        water_flag = bool(abs(overall["PBIAS_pct"]) > 25.0)
        low_flag = bool(
            low_metrics["n"] >= 4
            and mid_metrics["n"] >= 4
            and low_metrics["PBIAS_pct"] > 25.0
            and low_metrics["log_RMSE"] - mid_metrics["log_RMSE"] >= 0.10
        )
        timing_flag = bool(
            overall["KGE_2012"] < 0.50
            and abs(overall["PBIAS_pct"]) <= 25.0
            and (
                overall["KGE_r"] < 0.70
                or abs(overall["KGE_gamma"] - 1.0) > 0.25
            )
        )
        first = part.iloc[0]
        rows.append(
            {
                "fold_id": fold_id,
                "q_site": site,
                "reach_id": int(first["reach_id"]),
                "reach_class": str(first["reach_class"]),
                "reservoir_related": bool(
                    float(first["is_reservoir_reach"]) > 0
                    or float(first["downstream_reservoir"]) > 0
                ),
                **{f"overall_{k}": v for k, v in overall.items()},
                "low_n": low_metrics["n"],
                "low_log_RMSE": low_metrics["log_RMSE"],
                "low_PBIAS_pct": low_metrics["PBIAS_pct"],
                "mid_n": mid_metrics["n"],
                "mid_log_RMSE": mid_metrics["log_RMSE"],
                "low_minus_mid_log_RMSE": (
                    low_metrics["log_RMSE"] - mid_metrics["log_RMSE"]
                ),
                "water_balance_flag": water_flag,
                "low_flow_flag": low_flag,
                "timing_process_flag": timing_flag,
                "fold_flag_count": int(water_flag + low_flag + timing_flag),
            }
        )
    return pd.DataFrame(rows)


def direction_label(process: str, group: pd.DataFrame) -> str:
    if process == "water_balance":
        return (
            "positive_overprediction"
            if float(group["median_overall_PBIAS_pct"].iloc[0]) > 0
            else "negative_underprediction"
        )
    if process == "low_flow":
        return "positive_lowflow_overprediction"
    gamma = float(group["median_KGE_gamma"].iloc[0])
    r = float(group["median_KGE_r"].iloc[0])
    if np.isfinite(gamma) and gamma > 1.25:
        return "gamma_overdispersed"
    if np.isfinite(gamma) and gamma < 0.75:
        return "gamma_underdispersed"
    if np.isfinite(r) and r < 0.70:
        return "correlation_or_timing"
    return "mixed_timing"


def build_stable_classification(fold_diag: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    flag_map = {
        "water_balance": "water_balance_flag",
        "low_flow": "low_flow_flag",
        "timing_process": "timing_process_flag",
    }
    for site, part in fold_diag.groupby("q_site", sort=True):
        first = part.iloc[0]
        record: dict[str, object] = {
            "q_site": site,
            "reach_id": int(first["reach_id"]),
            "reach_class": str(first["reach_class"]),
            "reservoir_related": bool(first["reservoir_related"]),
            "fold_count": int(part["fold_id"].nunique()),
            "has_final_fold": bool((part["fold_id"] == FINAL_FOLD).any()),
            "median_overall_PBIAS_pct": float(part["overall_PBIAS_pct"].median()),
            "median_low_PBIAS_pct": float(part["low_PBIAS_pct"].median()),
            "median_low_minus_mid_log_RMSE": float(
                part["low_minus_mid_log_RMSE"].median()
            ),
            "median_KGE_2012": float(part["overall_KGE_2012"].median()),
            "median_KGE_r": float(part["overall_KGE_r"].median()),
            "median_KGE_gamma": float(part["overall_KGE_gamma"].median()),
        }
        stable_processes: list[str] = []
        for process, column in flag_map.items():
            flag_count = int(part[column].sum())
            final_flag = bool(
                part.loc[part["fold_id"].eq(FINAL_FOLD), column].any()
            )
            stable = bool(flag_count >= 2 and final_flag)
            record[f"{process}_flag_folds"] = flag_count
            record[f"{process}_final_fold"] = final_flag
            record[f"stable_{process}"] = stable
            if stable:
                stable_processes.append(process)

        if record["reservoir_related"]:
            primary = "human_or_reservoir_context"
        elif len(stable_processes) > 1:
            primary = "combined"
        elif len(stable_processes) == 1:
            primary = stable_processes[0]
        else:
            primary = "acceptable_or_unresolved"
        record["stable_process_count"] = len(stable_processes)
        record["stable_processes"] = "|".join(stable_processes)
        record["primary_stable_class"] = primary
        record["direction"] = (
            direction_label(primary, pd.DataFrame([record]))
            if primary in PROCESS_CLASSES
            else "not_applicable"
        )
        rows.append(record)
    return pd.DataFrame(rows)


def build_authorization(stable: pd.DataFrame) -> pd.DataFrame:
    natural = stable[~stable["reservoir_related"]].copy()
    eligible_natural = int(natural["q_site"].nunique())
    min_fraction_count = int(math.ceil(0.20 * eligible_natural))
    counts = {
        process: int(
            (natural["primary_stable_class"] == process).sum()
        )
        for process in PROCESS_CLASSES
    }
    rows: list[dict[str, object]] = []
    for process in PROCESS_CLASSES:
        part = natural[
            natural["primary_stable_class"].eq(process)
        ].copy()
        direction_counts = part["direction"].value_counts()
        dominant_direction = (
            str(direction_counts.index[0])
            if not direction_counts.empty
            else "none"
        )
        directional_agreement = (
            float(direction_counts.iloc[0] / len(part))
            if len(part)
            else 0.0
        )
        next_largest = max(
            [count for name, count in counts.items() if name != process],
            default=0,
        )
        count = counts[process]
        rows.append(
            {
                "process": process,
                "eligible_nonreservoir_station_count": eligible_natural,
                "stable_station_count": count,
                "stable_station_fraction": (
                    count / eligible_natural if eligible_natural else 0.0
                ),
                "minimum_fraction_count": min_fraction_count,
                "next_largest_process_count": next_largest,
                "lead_over_next_count": count - next_largest,
                "dominant_direction": dominant_direction,
                "directional_agreement": directional_agreement,
                "count_ge_10": count >= 10,
                "fraction_ge_20pct": count >= min_fraction_count,
                "lead_ge_5": count - next_largest >= 5,
                "direction_ge_60pct": directional_agreement >= 0.60,
                "authorized": bool(
                    count >= 10
                    and count >= min_fraction_count
                    and count - next_largest >= 5
                    and directional_agreement >= 0.60
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for required in [SOURCE_COMPARISON, SOURCE_GATE, DEVELOPMENT_INPUT]:
        if not required.exists():
            raise RuntimeError(f"Missing frozen input: {required}")
    source_gate = json.loads(SOURCE_GATE.read_text(encoding="utf-8"))
    if not source_gate.get("engineering_pass", False):
        raise RuntimeError("Source migration audit did not pass engineering")

    frame = pd.read_csv(SOURCE_COMPARISON, encoding="utf-8-sig")
    frame["q_site"] = frame["q_site"].astype(str)
    frame = frame[
        ~frame["q_site"].isin(EXCLUSIONS)
        & frame["year"].between(2012, 2018)
    ].copy()
    if int(frame["year"].max()) > 2018:
        raise RuntimeError("Protected confirmation years entered diagnosis")
    if SHIJIAO not in set(frame["q_site"]):
        raise RuntimeError("石角站 is absent")

    quantiles = build_training_quantiles()
    frame = frame.merge(
        quantiles,
        on=["fold_id", "q_site"],
        how="left",
        validate="many_to_one",
    )
    if frame["train_q75_cfs"].isna().any():
        missing = sorted(
            frame.loc[frame["train_q75_cfs"].isna(), "q_site"]
            .astype(str)
            .unique()
        )
        raise RuntimeError(f"Missing training Q75 for stations: {missing}")
    q25_delta = (
        frame["train_q25_cfs"] - frame["train_q25_check"]
    ).abs()
    q25_max_delta = float(q25_delta.max())

    fold_diag = build_fold_station_diagnostics(frame)
    stable = build_stable_classification(fold_diag)
    authorization = build_authorization(stable)
    authorized_rows = authorization[authorization["authorized"]].copy()
    if len(authorized_rows) > 1:
        raise RuntimeError("More than one process family was authorized")
    authorized_process = (
        str(authorized_rows["process"].iloc[0])
        if len(authorized_rows) == 1
        else None
    )

    used_year_max = int(frame["year"].max())
    eligible_station_count = int(frame["q_site"].nunique())
    reservoir_count = int(stable["reservoir_related"].sum())
    nonreservoir_count = eligible_station_count - reservoir_count
    class_counts = {
        str(key): int(value)
        for key, value in stable["primary_stable_class"]
        .value_counts()
        .sort_index()
        .items()
    }
    engineering_pass = bool(
        used_year_max == 2018
        and eligible_station_count
        == int(source_gate["eligible_station_count"])
        and all((frame["q_site"] != name).all() for name in EXCLUSIONS)
        and SHIJIAO in set(frame["q_site"])
        and fold_diag["fold_id"].nunique() == 3
        and len(stable) == eligible_station_count
        and len(authorized_rows) <= 1
    )

    fold_diag.to_csv(
        OUT / "fold_station_diagnostics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    stable.to_csv(
        OUT / "stable_station_classification.csv",
        index=False,
        encoding="utf-8-sig",
    )
    authorization.to_csv(
        OUT / "process_authorization_table.csv",
        index=False,
        encoding="utf-8-sig",
    )
    input_audit = pd.DataFrame(
        [
            {
                "source_comparison_sha256": sha256(SOURCE_COMPARISON),
                "source_gate_sha256": sha256(SOURCE_GATE),
                "development_input_sha256": sha256(DEVELOPMENT_INPUT),
                "used_year_min": int(frame["year"].min()),
                "used_year_max": used_year_max,
                "confirmation_years_used": False,
                "eligible_station_count": eligible_station_count,
                "eligible_nonreservoir_station_count": nonreservoir_count,
                "reservoir_context_station_count": reservoir_count,
                "q25_recalculation_max_abs_delta": q25_max_delta,
                "fixed_exclusions": "|".join(sorted(EXCLUSIONS)),
                "shijiao_present": SHIJIAO in set(frame["q_site"]),
                "python": sys.version.replace("\n", " "),
                "platform": platform.platform(),
                "conda_prefix": os.environ.get("CONDA_PREFIX", ""),
            }
        ]
    )
    input_audit.to_csv(
        OUT / "input_and_environment_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if authorized_process == "low_flow":
        next_stage = "RUN_MASS_CONSERVING_BASE_RESERVOIR_CANDIDATE"
    elif authorized_process == "water_balance":
        next_stage = "RUN_WATER_BALANCE_INPUT_AUDIT"
    elif authorized_process == "timing_process":
        next_stage = "RUN_CONSERVATIVE_QUICKFLOW_RESERVOIR_CANDIDATE"
    else:
        next_stage = "NO_PROCESS_AUTHORIZED_TERMINAL_SUMMARY"
    payload = {
        "run_id": RUN.name,
        "stage": "H1_Q72_stable_process_diagnosis",
        "used_year_max": used_year_max,
        "confirmation_years_used": False,
        "eligible_station_count": eligible_station_count,
        "eligible_nonreservoir_station_count": nonreservoir_count,
        "reservoir_context_station_count": reservoir_count,
        "stable_primary_class_counts": class_counts,
        "authorized_process": authorized_process,
        "authorized_process_count": int(len(authorized_rows)),
        "next_stage": next_stage,
        "engineering_pass": engineering_pass,
        "scientific_status": (
            f"AUTHORIZED_{authorized_process.upper()}"
            if engineering_pass and authorized_process
            else (
                "NO_SINGLE_PROCESS_AUTHORIZED"
                if engineering_pass
                else "FAIL_ENGINEERING"
            )
        ),
        "authorization_table": authorization.to_dict(orient="records"),
    }
    (OUT / "gate.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Q72 stable process diagnosis",
        "",
        f"- Engineering gate: {'PASS' if engineering_pass else 'FAIL'}",
        f"- Eligible stations: {eligible_station_count}",
        f"- Eligible non-reservoir stations: {nonreservoir_count}",
        f"- Reservoir/human-context stations inventoried only: {reservoir_count}",
        f"- Stable primary classes: {class_counts}",
        f"- Authorized process: `{authorized_process or 'NONE'}`",
        f"- Next stage: `{next_stage}`",
        "- Maximum used year: 2018; 2019–2022 used: False",
        "",
        "## Authorization table",
        "",
        "| process | stable n | fraction | lead | direction | agreement | authorized |",
        "|---|---:|---:|---:|---|---:|:---:|",
    ]
    for row in authorization.itertuples(index=False):
        lines.append(
            f"| {row.process} | {row.stable_station_count} | "
            f"{row.stable_station_fraction:.1%} | "
            f"{row.lead_over_next_count:+d} | "
            f"{row.dominant_direction} | "
            f"{row.directional_agreement:.1%} | "
            f"{'YES' if row.authorized else 'NO'} |"
        )
    report = "\n".join(lines) + "\n"
    (OUT / "gate.md").write_text(report, encoding="utf-8")
    (RUN / "README.md").write_text(
        "\n".join(
            [
                "# 20260729_7 — Q72 stable process diagnosis",
                "",
                "This folder classifies strict Q72 OOF failures and applies "
                "the pre-registered single-process authorization gate.",
                "Reservoir-related stations are inventoried but cannot "
                "authorize a natural-process change.",
                "",
                "## Result",
                "",
                *lines[2:10],
                "",
                "See `reports/q72_stable_process_diagnosis/gate.md` and "
                "`gate.json` for detailed evidence.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(report)
    if not engineering_pass:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
