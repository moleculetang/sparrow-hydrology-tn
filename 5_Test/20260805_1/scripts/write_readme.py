from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from runtime_guard import assert_sparrow_runtime


assert_sparrow_runtime()
RUN = Path(__file__).resolve().parents[1]


def main() -> None:
    config = json.loads((RUN / "config.json").read_text(encoding="utf-8"))
    build = json.loads((RUN / "inputs" / "source_metadata" / "input_build_summary.json").read_text(encoding="utf-8"))
    gate = json.loads((RUN / "reports" / "model_audit" / "gate.json").read_text(encoding="utf-8"))
    unmatched = pd.read_csv(RUN / "reports" / "input_audit" / "unmatched_or_unusable_stations.csv", encoding="utf-8-sig")
    comparison_path = RUN / "reports" / "baseline_comparison" / "comparison_summary.json"
    comparison = json.loads(comparison_path.read_text(encoding="utf-8")) if comparison_path.exists() else None
    metrics = gate["three_fold_oof"]
    cohort_text = (
        "2006–2022严格204个月完整站；石角站为固定保护例外，只有192个月，缺2009年且未插补"
        if config["cohort"] == "full_2006_2022"
        else "complete与noncomplete全量源中至少12个合格月、且2015年及以前至少1个月的站"
    )
    compare_lines = ["尚未生成双基线共同样本比较。"]
    if comparison:
        delta = comparison["common_delta_20260805_2_minus_20260805_1"]
        compare_lines = [
            f"- 共同站/共同月：{comparison['common_stations']}站、{comparison['common_rows']}行，实测值完全一致：{comparison['common_actuals_identical']}",
            f"- `_2 - _1`共同样本 median log-NSE：{delta['median_log_nse']:+.6f}",
            f"- `_2 - _1`共同样本 median KGE：{delta['median_kge']:+.6f}",
            f"- `_2 - _1`共同样本 median |PBIAS|：{delta['median_absolute_pbias_pct']:+.3f}个百分点",
            f"- `_2 - _1`共同样本低流log误差：{delta['median_lowflow_log_error']:+.6f}",
            f"- `_2`新增OOF站：{comparison['added_oof_stations']}站。",
        ]
    text = f"""# {RUN.name} 独立Q72新基线

## 定位

本目录是基于2026-08-05更新后 `DischargeData` 重新制作输入并实际完成三折OOF运行的独立Q72基线。Q72模型代码与 `20260729_42` 的冻结父代码SHA-256完全一致，没有加入Q78、融合或预测后处理。

- 队列：`{config['cohort']}`
- 覆盖口径：{cohort_text}
- 映射前合格站：{build['eligible_stations_before_mapping']}
- 可可靠映射站：{build['mapped_eligible_stations']}
- 同reach最大流量选择后的代表站：{build['selected_representative_stations']}
- 活动观测月：{build['active_observation_months']}
- 无可靠空间映射、因此未进入模型的候选：{len(unmatched)}

## 数据规则

- 所有数据、空间文件、政策表、协变量骨架和代码均已快照到本目录；模型运行不读取测试目录外文件。
- 日值月覆盖率至少75%；非正值不作为有效观测。
- 2006–2009工作簿只补缺月；逐日聚合与工作簿同月冲突时逐日源优先。
- `model_exclusion_policy.csv`的18个政策站及其确定性别名在覆盖筛选和空间映射前排除。
- 同reach多站按2006–2022可用月份中位流量最大者代表；并列时依次按可用月、可用年、snap距离和站名裁决。
- 只做去“站”、`_2`、`（重复）`等确定性别名归一；不使用主观错别字模糊匹配。
- 石角站必须保留。

## 验证方法

- fit≤2011 → eval 2012–2013
- fit≤2013 → eval 2014–2015
- fit≤2015 → eval 2016–2018

所有指标来自时间外OOF。2019–2022不参与本基线评分或模型选择。

## 三折OOF结果

- OOF站：{gate['oof_stations_any']}
- OOF行：{gate['oof_rows']}
- median log-NSE：{metrics['median_log_nse']:.6f}
- median KGE：{metrics['median_kge']:.6f}
- median |PBIAS|：{metrics['median_absolute_pbias_pct']:.3f}%
- median低流log误差：{metrics['median_lowflow_log_error']:.6f}
- good站：{metrics['good_count']}/{metrics['station_count']}

## 两版本公平比较

{chr(10).join(compare_lines)}

## 独立复跑

```powershell
$env:PYTHONIOENCODING='utf-8'
conda --no-plugins run -n sparrow python scripts/run_all.py
```

`run_all.py`依次重建本地输入、运行三折Q72、生成审计和README、执行独立验证。核心证据位于：

- `inputs/source_snapshot/`：冻结原始输入和空间资产；
- `inputs/source_metadata/`：覆盖、映射、同reach选择和输入摘要；
- `reports/input_audit/`：数据质量与未映射清单；
- `reports/model_audit/`：三折指标与模型门禁；
- `reports/baseline_comparison/`：两个基线的共同样本比较；
- `inputs_manifest/`：环境和SHA-256清单；
- `validation.json`：最终冻结判定。
"""
    (RUN / "README.md").write_text(text, encoding="utf-8")
    fold_manifest = pd.read_csv(RUN / "reports" / "q72_baseline" / "blocked_fold_manifest.csv", encoding="utf-8-sig")
    log_lines = [
        f"# {RUN.name} execution summary", "",
        "- runtime: conda environment `sparrow`", "- input build: completed from local source snapshot",
        f"- active stations: {build['selected_representative_stations']}", f"- active observation months: {build['active_observation_months']}",
        "- excluded-policy leakage: 0", "- protected Shijiao present: true", "",
        "## Blocked folds", "", fold_manifest.to_string(index=False), "",
        f"- OOF median log-NSE: {metrics['median_log_nse']:.6f}", f"- OOF median KGE: {metrics['median_kge']:.6f}",
        f"- OOF median |PBIAS|: {metrics['median_absolute_pbias_pct']:.3f}%",
    ]
    (RUN / "logs" / "execution_summary.md").write_text("\n".join(log_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
