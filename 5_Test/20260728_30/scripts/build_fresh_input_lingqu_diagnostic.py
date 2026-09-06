from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


RUN = Path(__file__).resolve().parents[1]
ROOT = RUN.parent
OUT = RUN / "reports"
BASELINE = ROOT / "20260727_6" / "reports"


def selected_metrics(root: Path) -> pd.DataFrame:
    path = root / "main_model" / "reach_class_station_metrics.csv"
    data = pd.read_csv(path)
    result = data.loc[
        data["variant"].eq("reach_class_alpha")
        & data["split"].eq("validation_2019_2022"),
        ["q_site", "NSE_log", "KGE_2012", "abs_PBIAS", "good"],
    ].copy()
    if result["q_site"].duplicated().any():
        raise ValueError(f"Selected strict metrics are not station-unique: {path}")
    return result


def main() -> None:
    status = json.loads((OUT / "experiment" / "experiment_status.json").read_text(encoding="utf-8"))
    gate = json.loads((OUT / "latest_input_lingqu_exclusion_gate.json").read_text(encoding="utf-8"))
    new_summary = pd.read_csv(OUT / "final_model_summary.csv").iloc[0].to_dict()
    base_summary = pd.read_csv(BASELINE / "final_model_summary.csv").iloc[0].to_dict()
    new = selected_metrics(OUT)
    old = selected_metrics(BASELINE)
    common = old.merge(new, on="q_site", suffixes=("_baseline", "_fresh"), validate="one_to_one")
    common["delta_NSE_log"] = common["NSE_log_fresh"] - common["NSE_log_baseline"]
    common["delta_KGE"] = common["KGE_2012_fresh"] - common["KGE_2012_baseline"]
    common["delta_abs_PBIAS"] = common["abs_PBIAS_fresh"] - common["abs_PBIAS_baseline"]
    common["good_changed"] = common["good_fresh"].astype(bool) != common["good_baseline"].astype(bool)
    baseline_only = sorted(set(old["q_site"]) - set(new["q_site"]))
    fresh_only = sorted(set(new["q_site"]) - set(old["q_site"]))
    nonfocus_baseline_only = [x for x in baseline_only if x != "灵渠（三）站"]
    source_files = pd.read_csv(OUT / "input_preprocessing" / "discharge_files_all.csv", encoding="utf-8-sig")
    excluded = pd.read_csv(OUT / "input_preprocessing" / "excluded_active_calibration_stations.csv", encoding="utf-8-sig")
    unexpected_population_rows = []
    for site in nonfocus_baseline_only:
        in_current_source_inventory = bool(source_files.astype(str).apply(lambda c: c.eq(site)).any(axis=1).any())
        in_current_exclusion_report = bool(excluded.astype(str).apply(lambda c: c.eq(site)).any(axis=1).any())
        unexpected_population_rows.append({"q_site": site, "present_in_current_discharge_file_inventory": in_current_source_inventory, "present_in_current_exclusion_report": in_current_exclusion_report})
    common_summary = {
        "common_station_count": int(len(common)),
        "median_NSElog_baseline": float(common["NSE_log_baseline"].median()),
        "median_NSElog_fresh": float(common["NSE_log_fresh"].median()),
        "median_NSElog_delta": float(common["delta_NSE_log"].median()),
        "median_KGE_baseline": float(common["KGE_2012_baseline"].median()),
        "median_KGE_fresh": float(common["KGE_2012_fresh"].median()),
        "median_KGE_delta": float(common["delta_KGE"].median()),
        "median_abs_PBIAS_baseline": float(common["abs_PBIAS_baseline"].median()),
        "median_abs_PBIAS_fresh": float(common["abs_PBIAS_fresh"].median()),
        "median_abs_PBIAS_delta": float(common["delta_abs_PBIAS"].median()),
        "good_baseline": int(common["good_baseline"].sum()),
        "good_fresh": int(common["good_fresh"].sum()),
        "good_delta": int(common["good_fresh"].sum() - common["good_baseline"].sum()),
        "good_changed_station_count": int(common["good_changed"].sum()),
    }
    report = {
        "run_id": RUN.name,
        "execution_passed": status["status"] == "passed",
        "latest_input_lingqu_gate": gate,
        "all_station_strict_summary": {"baseline": base_summary, "fresh": new_summary},
        "common_station_strict_summary": common_summary,
        "baseline_only_stations": baseline_only,
        "fresh_only_stations": fresh_only,
        "unexpected_nonfocus_population_change": unexpected_population_rows,
        "pure_lingqu_only_ablation_population_guardrail": len(nonfocus_baseline_only) == 0,
        "interpretation_boundary": "This is a descriptive before/after diagnostic, not a causal ablation: it jointly changes the raw DischargeData snapshot and excludes Lingqu（三）station. It must not be used alone to promote a model.",
    }
    common.sort_values("q_site").to_csv(OUT / "fresh_input_lingqu_common_station_comparison.csv", index=False, encoding="utf-8-sig")
    (OUT / "fresh_input_lingqu_diagnostic.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 20260728_17 最新输入重建与灵渠（三）排除：模型诊断",
        "",
        "## 运行与输入门禁",
        "",
        f"- 完整运行：{'PASS' if report['execution_passed'] else 'FAIL'}，7/7步骤成功。",
        f"- 所有日流量源路径来自当前 `E:/SPARROW/1_Inputs/DischargeData`：{gate['all_discharge_input_paths_under_current_DischargeData']}。",
        f"- 灵渠（三）站在模型输入面板中的行数：{gate['lingqu_san_rows_in_model_panel']}；排除清单已记录：{gate['lingqu_san_is_recorded_in_excluded_station_report']}。",
        "- 模型结构和超参数未改；活动排除为劳村、富罗（二）、隆安、灵渠（三）站。",
        "",
        "## 严格验证（2019--2022）",
        "",
        "| 口径 | 站数 | 中位NSElog | 中位KGE | 中位绝对PBIAS | good站 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        f"| 20260727_6 全站 | {int(base_summary['station_count'])} | {base_summary['median_NSElog']:.6f} | {base_summary['median_KGE']:.6f} | {base_summary['median_absPBIAS']:.6f} | {int(base_summary['good_count'])} |",
        f"| 20260728_17 全站 | {int(new_summary['station_count'])} | {new_summary['median_NSElog']:.6f} | {new_summary['median_KGE']:.6f} | {new_summary['median_absPBIAS']:.6f} | {int(new_summary['good_count'])} |",
        f"| 共同站（{common_summary['common_station_count']}）变化 | — | {common_summary['median_NSElog_delta']:+.6f} | {common_summary['median_KGE_delta']:+.6f} | {common_summary['median_abs_PBIAS_delta']:+.6f} | {common_summary['good_delta']:+d} |",
        "",
        "共同站中位指标：NSElog由{:.6f}到{:.6f}；KGE由{:.6f}到{:.6f}；绝对PBIAS由{:.6f}到{:.6f}。".format(
            common_summary['median_NSElog_baseline'], common_summary['median_NSElog_fresh'], common_summary['median_KGE_baseline'], common_summary['median_KGE_fresh'], common_summary['median_abs_PBIAS_baseline'], common_summary['median_abs_PBIAS_fresh']
        ),
        "",
        "## 比较边界",
        "",
        "本轮同时改变了原始 `DischargeData` 快照和站点集合，故上述变化不能归因于“仅删除灵渠（三）站”。它是新的输入版本诊断，不是对20260727_6或20260728_9的模型晋级结论。",
        "",
        f"- 基线独有站：{'、'.join(baseline_only) if baseline_only else '无'}。",
        f"- 新运行独有站：{'、'.join(fresh_only) if fresh_only else '无'}。",
        f"- 纯“仅灵渠”站点集合门禁：{'PASS' if len(nonfocus_baseline_only) == 0 else 'FAIL'}。",
        "- 逐共同站指标见 `fresh_input_lingqu_common_station_comparison.csv`。",
    ]
    if unexpected_population_rows:
        lines += ["", "### 非目标人口变化", ""]
        for item in unexpected_population_rows:
            lines.append(f"- `{item['q_site']}`：在当前原始日值文件清单中={item['present_in_current_discharge_file_inventory']}；在本轮排除清单中={item['present_in_current_exclusion_report']}；但未进入最终模型面板。该差异需单独追溯，不能被当作灵渠排除的效果。")
    (OUT / "fresh_input_lingqu_diagnostic.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
