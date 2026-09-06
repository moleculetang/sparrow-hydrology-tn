from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


RUN = Path(__file__).resolve().parents[1]
ROOT = RUN.parent
OUT = RUN / "reports"
SITE = "马口站"


def row(run: str) -> dict[str, object]:
    path = ROOT / run / "reports" / "input_preprocessing" / "station_reach_match.csv"
    data = pd.read_csv(path)
    found = data.loc[data["station_norm"].eq(SITE)]
    if len(found) != 1:
        raise ValueError(f"Expected exactly one {SITE} row in {path}")
    record = found.iloc[0]
    values = {}
    for key in ["x", "y", "reach_id", "match_method", "snap_distance_m", "best_catchment_distance_m", "best_line_distance_m", "used"]:
        value = record[key]
        values[key] = value.item() if hasattr(value, "item") else value
    return {"run": run, **values}


def main() -> None:
    baseline = row("20260727_6")
    fresh = row("20260728_17")
    report = {
        "station": SITE,
        "fixed_max_snap_distance_m": 5000.0,
        "baseline": baseline,
        "fresh": fresh,
        "current_daily_data_present": True,
        "current_policy_excluded": False,
        "root_cause": "Current station-to-reach distance exceeds the fixed 5000 m threshold, so the fresh input builder labels the station too_far and does not include it in the model panel.",
        "implication": "20260728_17 is not a pure Lingqu-only exclusion comparison; it also has this coordinate/matching-driven population change."
    }
    (OUT / "makou_station_inclusion_trace.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 马口站未纳入 20260728_17 的追踪说明",
        "",
        "- 当前日流量文件存在，且马口站不在本轮排除策略中。",
        "- 输入构建器的固定最大站点—河段匹配距离为 `5,000 m`。",
        "",
        "| 运行 | 坐标 x/y | 匹配方法 | 最近距离（m） | used |",
        "| --- | --- | --- | ---: | --- |",
        f"| 20260727_6 | {baseline['x']:.3f}, {baseline['y']:.3f} | {baseline['match_method']} | {baseline['snap_distance_m']:.3f} | {baseline['used']} |",
        f"| 20260728_17 | {fresh['x']:.3f}, {fresh['y']:.3f} | {fresh['match_method']} | {fresh['snap_distance_m']:.3f} | {fresh['used']} |",
        "",
        "结论：当前重建中马口站到候选河段的距离为 `5,277.837 m`，超过固定阈值，故被标记为 `too_far` 并在建模面板前移除；旧版距离为 `4,939.003 m`，因此仍可用。它不是流量文件缺失，也不是本轮站点筛选策略排除。",
        "",
        "这也意味着 `20260728_17` 同时改变了灵渠（三）与马口站的站点集合，不能作为纯粹的“仅排除灵渠（三）”效果估计。",
    ]
    (OUT / "makou_station_inclusion_trace.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
