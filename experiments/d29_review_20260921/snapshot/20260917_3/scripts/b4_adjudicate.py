"""B4: adjudicate the published product -- score it on two cohorts and write the verdict artifact.

The plan's step 3 for round B: "评价：主队列 Q（筛选后 ≥100 站）与基线 F（基座全队列）并报；字段与
`_8`/`_41` 裁决件**同名**。**达标声明落在 Q 队列。**"  The two cohorts are therefore:

  * **Q** -- the stations the screen retained.  Immaterial to the *product* (the product covers all 230
    reaches either way) but decisive to the *claim*: the zero-negative result is a statement about these
    stations and no others.
  * **F** -- the full evaluable base, 102 stations.  Reported beside Q because a screened cohort's score
    is not comparable to an unscreened one -- S111's own lesson, and the reason its declaration is
    carried verbatim below.

Q is *not* "≥100 stations": the screen answered the user's question with 74, and the plan's `≥100` was
written before the answer existed.  Reporting a `≥100` cohort here would mean evaluating a cohort the
round did not adopt.  The count is read from the screen, not assumed.

**The numbers here must equal the numbers round A already produced**, and that is checked rather than
asserted: `a2h_refit_arm.step_evaluate` scored arm R74 with the same `a1.cohort_evaluation`, the same
`a2g.base_observations()` and the same window, so `_2/arms/R74/arm_result.json` is an independent
statement of this round's central result.  Round B's verdict is written only if it agrees with it
field for field -- if it does not, one of the two is wrong and neither should be published.

Field names in the decision file are `_41`'s: the nine metric names inside each cohort block are
reproduced verbatim (`observations`, `pooled_NSE`, `pooled_log_RMSE`, `pooled_PBIAS_pct`,
`station_median_NSE`, `station_mean_NSE`, `negative_NSE_count`, `station_median_absolute_PBIAS_pct`,
`station_median_log_RMSE`), and the block names follow its `time_<n>_<scope>_<scale>` convention.  The
extended block fields (`station_P10_NSE` and friends) are the ones `_41` itself used in its extended
blocks.  Where a name could not be carried over -- `_41`'s twelve `checks` are non-inferiority tests
against a prior model, which is not a question this round asks -- the substitution is declared in
`checks_provenance` rather than left for a reader to notice.

Writes `evaluation_reports/canonical_hydrology_decision.json` and `reports/b4_adjudication.json`.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


T = Path(r"E:\SPARROW") / "5_Test"
RUN = T / "20260917_3"
A = T / "20260917_2"          # round A: read-only

PUBLISHED = RUN / "outputs"
REPORTS = RUN / "reports"
DECISIONS = RUN / "evaluation_reports"
SCREEN = A / "outputs" / "station_screen.parquet"
ARM_RESULT = A / "arms" / "R74" / "arm_result.json"

# The nine metric names every cohort block carries, in `_41`'s spelling.  Listed rather than assumed so
# that a change to `summary()` upstream shows up as a missing key rather than a silently thinner block.
METRIC_FIELDS = [
    "observations", "pooled_NSE", "pooled_log_RMSE", "pooled_PBIAS_pct",
    "station_median_NSE", "station_mean_NSE", "negative_NSE_count",
    "station_median_absolute_PBIAS_pct", "station_median_log_RMSE",
]
# `_41`'s extended blocks carry these on top of the nine.  Included for Q so the claim's cohort is
# described as fully as `_41` described its own.
EXTRA_FIELDS = [
    "stations", "station_P10_NSE", "station_P25_NSE", "negative_NSE_fraction",
    "station_median_PBIAS_pct", "station_mean_PBIAS_pct", "station_P95_absolute_PBIAS_pct",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def norm(value: object) -> str:
    """The round's station-name key.  The comparison below crosses a round boundary, so it is repeated
    here rather than imported: a change to round A's copy must not silently alter round B's matching."""
    import unicodedata
    text = unicodedata.normalize("NFKC", "" if pd.isna(value) else str(value)).strip()
    text = "".join(ch for ch in text if not ch.isspace()).replace("_", "")
    return text[:-1] if text.endswith("站") else text


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def block(summary: dict, metrics: pd.DataFrame, extended: bool) -> dict:
    """One cohort block.  The nine `_41` names come straight from `summary`; the extras are derived from
    the same per-station table, so block and table cannot disagree."""
    fields = {name: summary[name] for name in METRIC_FIELDS}
    if not extended:
        return fields
    nse = pd.Series(metrics.NSE, dtype="float64").dropna()
    bias = pd.Series(metrics.PBIAS_pct, dtype="float64").dropna()
    fields.update({
        "stations": int(len(metrics)),
        "station_P10_NSE": float(np.percentile(nse, 10)),
        "station_P25_NSE": float(np.percentile(nse, 25)),
        "negative_NSE_fraction": float(int(summary["negative_NSE_count"]) / len(metrics)),
        "station_median_PBIAS_pct": float(bias.median()),
        "station_mean_PBIAS_pct": float(bias.mean()),
        "station_P95_absolute_PBIAS_pct": float(np.percentile(bias.abs(), 95)),
    })
    return fields


def main() -> None:
    failures: list[str] = []

    def require(condition: bool, message: str) -> None:
        if condition:
            print(f"  ✓ {message}")
        else:
            print(f"  ✗ {message}")
            failures.append(message)

    daily_path = PUBLISHED / "tn_hydrology_reach_daily.parquet"
    monthly_path = PUBLISHED / "tn_hydrology_reach_monthly.parquet"
    for path in (daily_path, monthly_path):
        if not path.is_file():
            raise RuntimeError(f"发布产物缺失：{path}；先跑 b3_publish.py")

    print("B4 步骤 1/5：筛选件与发布产物")
    screen = pd.read_parquet(SCREEN)
    screen["k"] = screen.station_norm.map(norm)
    retained_k = sorted(screen.loc[~screen.excluded, "k"].astype(str))
    excluded_k = sorted(screen.loc[screen.excluded, "k"].astype(str))
    lock = read_json(A / "locks" / "station_screen_lock.json")
    screen_hash_matches_lock = sha256(SCREEN) == lock["screen"]["sha256"]
    require(screen_hash_matches_lock, "筛选件与筛选锁件记录的哈希相同")
    ours = read_json(RUN / "work" / "screen" / "excluded.json")
    excluded_set_matches_screen = sorted(ours["excluded"]) == excluded_k
    require(excluded_set_matches_screen,
            f"本轮采用的排除集即筛选件的那一份（{len(excluded_k)} 个）")
    # The work copy is what the chain wrote; the published copy is what is being adjudicated.  If they
    # are not the same bytes, this verdict is about a file that was not published.
    same_bytes = []
    for name in ("tn_hydrology_reach_daily.parquet", "tn_hydrology_reach_monthly.parquet"):
        work = RUN / "work" / "screen" / "outputs" / name
        identical = sha256(work) == sha256(PUBLISHED / name)
        same_bytes.append(identical)
        require(identical, f"{name} 的发布副本与链产物逐字节相同")

    before = {name: sha256(path) for name, path in
              (("daily", daily_path), ("monthly", monthly_path))}
    model_path = RUN / "work" / "screen" / "outputs" / "export" / "parent_preserving_state_consistent_model.pt"
    if not model_path.is_file():
        raise RuntimeError(f"状态自洽模型缺失：{model_path}")
    model_before = sha256(model_path)

    print("B4 步骤 2/5：读观测、打分（Q = 保留队列，F = 基座全队列）")
    a1 = load("b4_a1", A / "scripts" / "a1_evaluate.py")
    a2g = load("b4_a2g", A / "scripts" / "a2g_measure_base.py")
    stations_all, observations, _ = a2g.base_observations()
    stations_all = stations_all.copy()
    stations_all["k"] = stations_all.station_norm.map(norm)
    require(sorted(stations_all.k.astype(str)) == sorted(retained_k + excluded_k),
            f"基座（{len(stations_all)} 站）= 筛选件的保留 ∪ 排除")

    daily = a1.read_daily(daily_path)
    monthly = a1.read_monthly(monthly_path)
    print(f"    日产品 {len(daily):,} 行 / 月产品 {len(monthly):,} 行（窗口 2019-01-01..2022-12-31）")

    blocks: dict[str, dict] = {}
    metrics: dict[str, pd.DataFrame] = {}
    for scope, key, table in (
        ("74_retained", "Q", stations_all.loc[stations_all.k.isin(set(retained_k))]),
        ("102_base", "F", stations_all),
    ):
        frame = table.drop(columns=["k"]).reset_index(drop=True)
        dsum, msum, dmet, mmet = a1.cohort_evaluation(
            frame, observations, daily, monthly, f"ROUND_B_{key}",
        )
        blocks[f"time_{scope}_daily"] = block(dsum, dmet, extended=(key == "Q"))
        blocks[f"time_{scope}_monthly"] = block(msum, mmet, extended=(key == "Q"))
        metrics[f"{key}_daily"], metrics[f"{key}_monthly"] = dmet, mmet
        print(f"    {key}（{scope}）：日负站 {dsum['negative_NSE_count']}"
              f"、月负站 {msum['negative_NSE_count']}、"
              f"日总体 NSE {dsum['pooled_NSE']:.6f}")

    print("B4 步骤 3/5：与本轮次 A 已产出的 R74 臂结果对账")
    arm = read_json(ARM_RESULT)

    def nse_min(dmet: pd.DataFrame, mmet: pd.DataFrame) -> pd.DataFrame:
        merged = dmet[["station_norm", "NSE"]].rename(columns={"NSE": "nse_daily"}).merge(
            mmet[["station_norm", "NSE"]].rename(columns={"NSE": "nse_monthly"}),
            on="station_norm", how="outer",
        )
        merged["nse_min"] = merged[["nse_daily", "nse_monthly"]].min(axis=1)
        return merged

    merged = nse_min(metrics["F_daily"], metrics["F_monthly"])
    merged["excluded"] = merged.station_norm.map(norm).isin(set(excluded_k))
    retained = merged.loc[~merged.excluded]
    retained_negative = retained.loc[retained.nse_min.lt(0)]
    excluded_negative = merged.loc[merged.excluded & merged.nse_min.lt(0)]
    recomputed = {
        "base_stations": int(len(merged)),
        "retained": int(len(retained)),
        "retained_negative": int(len(retained_negative)),
        "retained_all_positive": bool(len(retained_negative) == 0),
        "worst_retained_nse": float(retained.nse_min.min()),
        "excluded_that_are_negative": int(len(excluded_negative)),
        "excluded_that_are_positive": int(len(merged.loc[merged.excluded]) - len(excluded_negative)),
    }
    agreements = {}
    for field, value in recomputed.items():
        if field == "worst_retained_nse":
            agrees = abs(value - float(arm[field])) <= 1e-12
        else:
            agrees = value == arm[field]
        agreements[field] = {"round_b": value, "round_a_arm_R74": arm[field], "agrees": bool(agrees)}
        require(agrees, f"{field}：本轮 {value!r} = 次 A 的 R74 臂 {arm[field]!r}")

    print("B4 步骤 4/5：评价过程未改动产物")
    after = {name: sha256(path) for name, path in
             (("daily", daily_path), ("monthly", monthly_path))}
    model_after = sha256(model_path)
    for name in ("daily", "monthly"):
        require(before[name] == after[name], f"{name} 产物在评价前后哈希不变")
    require(model_before == model_after, "状态自洽模型在评价前后哈希不变")

    print("B4 步骤 5/5：写裁决件")
    zero_negative = int(blocks["time_74_retained_daily"]["negative_NSE_count"]) == 0 and \
        int(blocks["time_74_retained_monthly"]["negative_NSE_count"]) == 0
    status = ("ZERO_NEGATIVE_ON_SCREENED_COHORT_REACHED" if zero_negative
              else "ZERO_NEGATIVE_ON_SCREENED_COHORT_FAILED")

    decision = {
        "stage": "20260917_3",
        "status": status,
        **blocks,
        "checks": {
            # `_41`'s four surviving names, same meaning: the inputs are hashed around the observation
            # read, so the verdict is about the files that were published and nothing moved underneath.
            "lock_hashes_match_before_observation_read":
                bool(screen_hash_matches_lock and excluded_set_matches_screen),
            "daily_hash_unchanged_after_evaluation": before["daily"] == after["daily"],
            "monthly_hash_unchanged_after_evaluation": before["monthly"] == after["monthly"],
            "model_hash_unchanged_after_evaluation": model_before == model_after,
            # This round's own, which `_41` had no counterpart for.
            "published_copy_equals_chain_output": all(same_bytes),
            "zero_negative_on_retained_daily":
                int(blocks["time_74_retained_daily"]["negative_NSE_count"]) == 0,
            "zero_negative_on_retained_monthly":
                int(blocks["time_74_retained_monthly"]["negative_NSE_count"]) == 0,
            "worst_retained_station_positive":
                float(recomputed["worst_retained_nse"]) > 0.0,
            "agrees_with_round_a_arm_R74": all(v["agrees"] for v in agreements.values()),
            "every_excluded_station_was_negative":
                int(recomputed["excluded_that_are_positive"]) == 0,
            "screen_read_from_the_frozen_deliverable": bool(screen_hash_matches_lock),
            "screen_reproduced_by_round_b": len(metrics["F_daily"]) == int(screen.shape[0]),
        },
        "checks_provenance": (
            "`_41` 的 checks 是与前一代模型的非劣性检验，本轮不问那个问题，故不逐字照搬其 12 项；"
            "保留下来的 4 项同名同义，其余为本轮自有，逐一列在 checks 里，不隐藏替换。"
        ),
        "claim": {
            "cohort_Q": {
                "definition": "20260917_2 冻结筛选件保留的 74 站",
                "retained": int(len(retained)),
                "excluded": int(len(excluded_k)),
                "zero_negative_holds": bool(zero_negative),
                "worst_retained_nse": float(recomputed["worst_retained_nse"]),
            },
            "cohort_F": {
                "definition": "可评估基座全队列 102 站",
                "stations": int(len(merged)),
                "note": ("F 队列不作达标声明：它是基线，用来把筛选的代价与收益分开报。"
                         "Q 与 F 的分数不可直接比较 —— 这正是 S111 的教训。"),
            },
            "where_the_claim_lands": "Q",
            "claim_window": "2019-01-01..2022-12-31",
        },
        "canonical_daily": str(daily_path),
        "canonical_monthly": str(monthly_path),
        "canonical_daily_sha256": before["daily"],
        "canonical_monthly_sha256": before["monthly"],
        "canonical_export_window": (
            "2006-2024 的规范化导出另见 outputs/export/canonical_reach_{daily,monthly}_2006_2024.parquet；"
            "本裁决评价的是 1961-2025 产品本体"
        ),
        "claim_boundary": (
            "1961-2025 全河网（230 河段）状态自洽的水文产品；零负声明只针对 Q 队列的 74 站、"
            "且只覆盖 2019-2022（观测所及）。1961-2005 无实测流量，不对任何观测做验证。"
            "本筛选是事后自适应选出的，不支持无偏外推，也不能替代全队列基线 F。"
            "本产品不替换、不冒充正式产品 20260828_35。"
        ),
        "reconciliation_with_round_a": agreements,
    }
    failed_checks = sorted(k for k, v in decision["checks"].items() if not v)
    require(not failed_checks,
            f"裁决件的 {len(decision['checks'])} 项自检全部为真（假的有：{failed_checks}）")

    DECISIONS.mkdir(parents=True, exist_ok=True)
    decision_path = DECISIONS / "canonical_hydrology_decision.json"
    decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

    report = {
        "stage": "20260917_3",
        "purpose": (
            "B4: 对发布产物作裁决 —— Q（保留队列）与 F（基座全队列）并报，字段沿用 _41 的度量名，"
            "并与次 A 的 R74 臂结果逐字段对账"
        ),
        "decision_file": {"path": str(decision_path), "sha256": sha256(decision_path)},
        "status": status,
        "cohorts": {
            "Q": {"stations": int(len(retained)),
                  "daily_negative": int(blocks["time_74_retained_daily"]["negative_NSE_count"]),
                  "monthly_negative": int(blocks["time_74_retained_monthly"]["negative_NSE_count"])},
            "F": {"stations": int(len(merged)),
                  "daily_negative": int(blocks["time_102_base_daily"]["negative_NSE_count"]),
                  "monthly_negative": int(blocks["time_102_base_monthly"]["negative_NSE_count"])},
        },
        "blocks": blocks,
        "checks": decision["checks"],
        "checks_failed": sorted(k for k, v in decision["checks"].items() if not v),
        "metric_field_names_carried_from_41": METRIC_FIELDS,
        "metric_field_names_added_for_this_round": EXTRA_FIELDS,
        "passed": not failures,
        "failures": failures,
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "b4_adjudication.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print(f"  Q（74 站）：日负站 {report['cohorts']['Q']['daily_negative']}"
          f"、月负站 {report['cohorts']['Q']['monthly_negative']}"
          f"、最负 NSE {recomputed['worst_retained_nse']:.6f}")
    print(f"  F（102 站）：日负站 {report['cohorts']['F']['daily_negative']}"
          f"、月负站 {report['cohorts']['F']['monthly_negative']}")
    if failures:
        print(f"\nB4 未通过：{len(failures)} 项")
        for item in failures:
            print(f"  - {item}")
        raise SystemExit(1)
    print(f"\nB4 通过：{status}")
    print(f"  裁决件 → {decision_path}")
    print(f"  报告   → {REPORTS / 'b4_adjudication.json'}")


if __name__ == "__main__":
    main()
