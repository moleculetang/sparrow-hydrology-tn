from __future__ import annotations

import json
from pathlib import Path

from runtime_guard import assert_sparrow_runtime


assert_sparrow_runtime()
RUN = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    build = load(RUN / "inputs" / "source_metadata" / "input_build_summary.json")
    spatial = load(RUN / "reports" / "spatial_correction" / "corrected_topology_validation.json")
    climate = load(RUN / "reports" / "spatial_correction" / "climate_reaggregation_manifest.json")
    s0 = load(RUN / "reports" / "s0_reproduction" / "gate.json")
    model = load(RUN / "reports" / "model_audit" / "gate.json")
    compare = load(RUN / "reports" / "corrected_baseline_comparison" / "performance_guard.json")
    m0, m1, delta = compare["s0"], compare["s1"], compare["delta_s1_minus_s0"]
    decision = (
        "CORRECTED_Q72_BASELINE_FROZEN"
        if compare["performance_guard_passed"]
        else "SPATIAL_CORRECTION_VALID_BASELINE_FROZEN_PERFORMANCE_DEGRADED"
    )
    text = f"""# {RUN.name}：拓扑与Catchment矫正后的独立Q72基线

## 终端状态

`{decision}`

空间矫正、输入重建和三折时间外运行均已完成。本目录是独立、完整、可复跑的矫正后Q72基线；空间正确性与模型性能分别判定，性能变化没有用于反向调整拓扑。

## 备份与空间矫正

- 矫正前备份：36个关键文件，48,523,752 bytes，全部记录SHA-256。
- 拓扑：{spatial['edge_count']}条边、{spatial['weak_components']}个弱连通分量、{spatial['terminal_count']}个terminal、0个环。
- 新增边：`14→19`、`64→59`、`132→149`、`180→168`、`199→196`。
- Catchment：{spatial['catchments']['catchment_count']}个，零面积{spatial['catchments']['zero_area_count']}；闭合前覆盖率{spatial['catchments']['coverage_ratio_before_fill']:.6%}，闭合后约{spatial['catchments']['coverage_ratio']:.6%}。
- 岔江坐标在重建后落入Reach 199 Catchment；没有人为改站点Reach。

## 输入门禁

- 协变量：{climate['rows']:,}行、{climate['reaches']} Reach，键唯一、气象完整、面积为正、无观测泄漏。
- 代表站：{build['selected_representative_stations']}个；活动观测月：{build['active_observation_months']:,}。
- 同Reach碰撞：{build['same_reach_collision_reaches']}组；新碰撞候选：{build['new_collision_candidate_rows']}。
- 固定排除站缺席：{build['fixed_exclusions_absent']}；石角保留：{build['protected_shijiao_present']}。
- 协变量SHA-256：`{build['backbone_sha256']}`。
- 模型输入SHA-256：`{build['input_sha256']}`。

## S0复现门禁

- 判定：`{s0['decision']}`。
- 矫正前OOF：{s0['rows']}行、{s0['stations']}站、{s0['folds']}折。
- 行键一致：{s0['keys_identical']}；最大数值差：{s0['max_numeric_difference']:.3g}；Parquet SHA-256一致：{s0['sha256_identical']}。

## S1三折Q72

- OOF：{model['oof_rows']}行、{model['oof_stations_any']}站。
- Q72冻结代码哈希一致：{model['parent_q72_code_hashes_exact']}。
- 预测键唯一、正且有限：{model['oof_keys_unique'] and model['predictions_positive_finite']}。
- 石角OOF存在：{model['protected_shijiao_oof_present']}。

## 矫正前后公平比较

比较只使用共同的站—Reach—年—月，实测值完全一致：{compare['observations_identical']}。

| 指标 | S0矫正前 | S1矫正后 | S1-S0 |
|---|---:|---:|---:|
| median log-NSE | {m0['median_log_nse']:.6f} | {m1['median_log_nse']:.6f} | {delta['median_log_nse']:+.6f} |
| median KGE | {m0['median_kge']:.6f} | {m1['median_kge']:.6f} | {delta['median_kge']:+.6f} |
| median \|PBIAS\| | {m0['median_absolute_pbias_pct']:.3f}% | {m1['median_absolute_pbias_pct']:.3f}% | {delta['median_absolute_pbias_pct']:+.3f} pp |
| median低流log-RMSE | {m0['median_lowflow_log_rmse']:.6f} | {m1['median_lowflow_log_rmse']:.6f} | {delta['median_lowflow_log_rmse']:+.6f} |
| median高流log-RMSE | {m0['median_highflow_log_rmse']:.6f} | {m1['median_highflow_log_rmse']:.6f} | {delta['median_highflow_log_rmse']:+.6f} |
| good站 | {m0['good_count']} | {m1['good_count']} | {delta['good_count']:+d} |

- 27/28 legacy低流目标中本轮可评价：{compare['legacy_targets_evaluable']}；绝对低流偏差减小：{compare['legacy_lowflow_abs_bias_improved_count']}。
- 石角log-NSE变化：{compare['shijiao_log_nse_delta']:+.6f}。
- 性能保护门禁：`{compare['decision']}`。

## 独立方法审计

- 终判：`ACCEPT_WITH_CAVEAT`。
- 未发现实质性标签泄漏、结果后选边、删站或调参；空间正确性与性能代价保持分离。
- 限制1：D8初始未覆盖的约0.84%区域（132个组件）采用最近Catchment启发式闭合，因此当前空间包是测试级内部一致基线，不是最终权威Catchment真值。
- 限制2：同Reach代表站使用2006—2022全期资料选择，存在cohort层面的未来信息依赖；它不影响本轮S0/S1同队列配对比较，但限制“严格deployment-pure”表述。
- 完整审计见`subagent_method_audit.md`。

## 冻结边界与复跑

- 只修改本测试目录；未修改`0_reach_topology`、`1_Inputs`、其他测试目录或主线代码。
- 全程使用`conda sparrow`。
- Q72结构、固定水文参数、排除政策和三折划分不变。
- 同Reach多站仍按2006—2022可用月中位流量最大者代表；石角必须保留。

```powershell
$env:PYTHONIOENCODING='utf-8'
$env:PYTHONUTF8='1'
conda --no-plugins run -n sparrow python scripts/run_all.py
```

核心证据位于`reports/spatial_correction/`、`reports/s0_reproduction/`、`reports/model_audit/`、`reports/corrected_baseline_comparison/`、`terminal_gate.json`和`inputs_manifest/`。
"""
    (RUN / "README.md").write_text(text, encoding="utf-8")
    (RUN / "logs" / "execution_summary.md").write_text(
        f"# {RUN.name} execution summary\n\n- runtime: conda sparrow\n- decision: {decision}\n"
        f"- OOF: {model['oof_rows']} rows, {model['oof_stations_any']} stations\n"
        f"- median log-NSE: {m0['median_log_nse']:.6f} -> {m1['median_log_nse']:.6f}\n"
        f"- performance guard: {compare['decision']}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
