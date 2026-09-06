from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260608_1"
BASE = RUN / "reports" / "main_model"
FIG = RUN / "figure" / "main_model"
OUT = BASE / "main_model_full_method_and_station_performance.md"


def md_table(df: pd.DataFrame, cols: list[str] | None = None, floatfmt: int = 3) -> str:
    frame = df[cols].copy() if cols else df.copy()
    for col in frame.columns:
        if pd.api.types.is_float_dtype(frame[col]):
            frame[col] = frame[col].map(lambda x: "" if pd.isna(x) else f"{x:.{floatfmt}f}")
        elif pd.api.types.is_bool_dtype(frame[col]):
            frame[col] = frame[col].map(lambda x: "是" if x else "否")
        else:
            frame[col] = frame[col].fillna("").astype(str)
    headers = list(frame.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in frame.iterrows():
        values = [str(row[col]).replace("\n", " ").replace("|", "/") for col in headers]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def load_tables() -> dict[str, pd.DataFrame | pd.Series]:
    summary = pd.read_csv(BASE / "reach_class_light_constraint_summary.csv", encoding="utf-8-sig").iloc[0]
    class_alpha = pd.read_csv(BASE / "selected_reach_class_alpha.csv", encoding="utf-8-sig")
    class_summary = pd.read_csv(BASE / "reach_class_summary.csv", encoding="utf-8-sig")
    metrics = pd.read_csv(BASE / "reach_class_station_metrics.csv", encoding="utf-8-sig")
    station_diag = pd.read_csv(BASE / "station_diagnostic_labels.csv", encoding="utf-8-sig")
    reservoir_summary = pd.read_csv(BASE / "reservoir_related_good_bad_summary.csv", encoding="utf-8-sig")
    failure_summary = pd.read_csv(BASE / "failure_mode_summary.csv", encoding="utf-8-sig")
    hydrograph_index = pd.read_csv(BASE / "station_hydrograph_index.csv", encoding="utf-8-sig")

    val = metrics[
        metrics["variant"].eq("reach_class_alpha") & metrics["split"].eq("validation_2019_2022")
    ].copy()
    q72 = metrics[metrics["variant"].eq("alpha0_q72") & metrics["split"].eq("validation_2019_2022")].copy()
    q78 = metrics[metrics["variant"].eq("alpha1_mass") & metrics["split"].eq("validation_2019_2022")].copy()
    val = val.merge(
        q72[["q_site", "NSE_log", "KGE_2012", "PBIAS_pct", "good"]].rename(
            columns={
                "NSE_log": "Q72_NSElog",
                "KGE_2012": "Q72_KGE",
                "PBIAS_pct": "Q72_PBIAS_pct",
                "good": "Q72_good",
            }
        ),
        on="q_site",
        how="left",
    )
    val = val.merge(
        q78[["q_site", "NSE_log", "KGE_2012", "PBIAS_pct", "good"]].rename(
            columns={
                "NSE_log": "Q78_NSElog",
                "KGE_2012": "Q78_KGE",
                "PBIAS_pct": "Q78_PBIAS_pct",
                "good": "Q78_good",
            }
        ),
        on="q_site",
        how="left",
    )
    val = val.merge(
        station_diag[["q_site", "reservoir_relation", "failure_mode"]],
        on="q_site",
        how="left",
    )
    val["Delta_NSElog_vs_Q72"] = val["NSE_log"] - val["Q72_NSElog"]
    val["Delta_absPBIAS_vs_Q72"] = val["PBIAS_pct"].abs() - val["Q72_PBIAS_pct"].abs()

    return {
        "summary": summary,
        "class_alpha": class_alpha,
        "class_summary": class_summary,
        "station_metrics": val,
        "reservoir_summary": reservoir_summary,
        "failure_summary": failure_summary,
        "hydrograph_index": hydrograph_index,
    }


def build_document() -> str:
    data = load_tables()
    summary = data["summary"]
    class_alpha = data["class_alpha"].copy()
    class_summary = data["class_summary"].copy()
    station_metrics = data["station_metrics"].copy()
    reservoir_summary = data["reservoir_summary"].copy()
    failure_summary = data["failure_summary"].copy()
    hydrograph_index = data["hydrograph_index"].copy()

    class_alpha = class_alpha.rename(
        columns={
            "class_alpha": "alpha_c",
            "station_count": "station_count",
            "inner_base_NSElog": "inner_Q72_NSElog",
            "inner_selected_NSElog": "inner_main_NSElog",
            "inner_base_KGE": "inner_Q72_KGE",
            "inner_selected_KGE": "inner_main_KGE",
            "inner_base_absPBIAS": "inner_Q72_absPBIAS",
            "inner_selected_absPBIAS": "inner_main_absPBIAS",
            "inner_base_good_count": "inner_Q72_good",
            "inner_selected_good_count": "inner_main_good",
        }
    )

    by_class = (
        station_metrics.groupby("reach_class", as_index=False)
        .agg(
            station_count=("q_site", "nunique"),
            good_count=("good", "sum"),
            median_NSElog=("NSE_log", "median"),
            median_KGE=("KGE_2012", "median"),
            median_absPBIAS=("PBIAS_pct", lambda s: np.nanmedian(np.abs(s))),
            median_delta_NSElog_vs_Q72=("Delta_NSElog_vs_Q72", "median"),
        )
    )
    by_class["bad_count"] = by_class["station_count"] - by_class["good_count"]
    by_class = by_class[
        [
            "reach_class",
            "station_count",
            "good_count",
            "bad_count",
            "median_NSElog",
            "median_KGE",
            "median_absPBIAS",
            "median_delta_NSElog_vs_Q72",
        ]
    ]

    stations = station_metrics.rename(
        columns={
            "q_site": "station",
            "class_alpha": "alpha_c",
            "NSE_log": "NSElog",
            "KGE_2012": "KGE",
            "PBIAS_pct": "PBIAS_pct",
        }
    ).copy()
    station_cols = [
        "station",
        "reach_id",
        "reach_class",
        "alpha_c",
        "NSElog",
        "KGE",
        "PBIAS_pct",
        "good",
        "failure_mode",
        "reservoir_relation",
        "Q72_NSElog",
        "Delta_NSElog_vs_Q72",
        "Q78_NSElog",
    ]
    good_stations = stations[stations["good"]].sort_values("NSElog", ascending=False)
    bad_stations = stations[~stations["good"]].sort_values(["failure_mode", "NSElog"])
    all_stations = stations.sort_values(["good", "NSElog"], ascending=[False, False])

    validation_comp = class_summary[class_summary["split"].eq("validation_2019_2022")][
        [
            "variant",
            "station_count",
            "median_NSElog",
            "median_KGE",
            "median_absPBIAS",
            "good_count",
            "median_abs_log_distance_to_mass",
        ]
    ]

    lines: list[str] = []
    add = lines.append
    add("# 20260608_1 主线模型完整数学原理与水文站模拟效果报告")
    add("")
    add(f"生成位置：`{OUT}`")
    add("")
    add("## 1. 版本定位")
    add("")
    add("`20260608_1` 是当前流量模拟的 clean mainline case。它复现了此前效果最好的主线结果，并把主输出集中到 `reports/main_model` 和 `figure/main_model`。")
    add("")
    add("模型本质：")
    add("")
    add("```text")
    add("高技能经验/贝叶斯水文特征回归 Q72")
    add("  + 质量守恒骨架提供的水量结构参照 Q78_mass")
    add("  + reach_class 级别的轻质量约束 log-space 融合")
    add("= 当前主线 Q_main")
    add("```")
    add("")
    add("它不是严格质量守恒模型，而是一个以预测技能为主、同时向质量守恒骨架轻微靠拢的主线模型。")
    add("")
    add("## 2. 数据输入与时间划分")
    add("")
    add("主要输入包括：")
    add("")
    add("- 水文站月均流量观测 `Q_obsv_cfs`。")
    add("- reach/catchment 拓扑与属性：`reach_id/comid`、增量面积 `IncAreaKm2`、累计面积 `CumAreaKm2`、河段长度、坡度、水库 reach 标记、上下游拓扑。")
    add("- 气象和水文驱动：降水 `PPT`、蒸散发 `AET/PET`、前期降水/蒸散、计算流量 `Q_calc_cfs`、多年平均流量 `Q_ma_cfs`、boundary flow 等。")
    add("- 当前主模型评价表覆盖 104 个验证站点，严格验证期为 2019-2022。")
    add("")
    add("时间分段规则：")
    add("")
    add("```text")
    add("2010-2015: calibration / 主体拟合期")
    add("2016-2018: inner validation / 选择 alpha 和部分超参数")
    add("2019-2022: strict validation / 严格验证，只评价不调参")
    add("```")
    add("")
    add("## 3. 评价指标")
    add("")
    add("对每个站点 `i`，验证期内观测为 `Qobs_t`，预测为 `Qpred_t`。")
    add("")
    add("```text")
    add("NSE = 1 - sum_t (Qpred_t - Qobs_t)^2 / sum_t (Qobs_t - mean(Qobs))^2")
    add("")
    add("NSElog = 1 - sum_t [log(Qpred_t + eps) - log(Qobs_t + eps)]^2")
    add("             / sum_t [log(Qobs_t + eps) - mean(log(Qobs + eps))]^2")
    add("")
    add("KGE = 1 - sqrt((r - 1)^2 + (beta - 1)^2 + (gamma - 1)^2)")
    add("beta  = mean(Qpred) / mean(Qobs)")
    add("gamma = CV(Qpred) / CV(Qobs)")
    add("")
    add("PBIAS = 100 * sum_t(Qpred_t - Qobs_t) / sum_t(Qobs_t)")
    add("```")
    add("")
    add("good station 判定：")
    add("")
    add("```text")
    add("NSElog >= 0.65 且 KGE >= 0.50 且 |PBIAS| <= 25%")
    add("```")
    add("")
    add("## 4. 分支一：高技能水文特征回归 Q72")
    add("")
    add("Q72 是当前主线的主要预测能力来源。它使用月尺度水文状态特征、SAS/TTD 风格的蓄泄记忆、快慢流信息、蒸散发约束、前期湿润程度、空间属性和站点/水文区组效应构造 log-flow 预测。")
    add("")
    add("抽象写法为：")
    add("")
    add("```text")
    add("eta72_{i,t} = beta0")
    add("             + X_fixed(r,t) beta_fixed")
    add("             + X_prod(r,t) beta_prod")
    add("             + X_multistore(r,t) beta_multistore")
    add("             + X_hysteresis(r,t) beta_hysteresis")
    add("             + b_station(i)")
    add("             + b_group(r)")
    add("             + random/regime slope terms")
    add("")
    add("Q72_median_{i,t} = exp(eta72_{i,t})")
    add("```")
    add("")
    add("重要水文含义：")
    add("")
    add("- `PPT, AET, PET` 描述降水输入与蒸散发损失。")
    add("- `P_surplus` / effective precipitation 表示降水超过蒸散与初损后的有效供水。")
    add("- quick-flow / slow-flow / storage features 表示快响应径流、慢响应基流和蓄水释放。")
    add("- SAS/TTD 风格变量表示水龄、前期湿润度和滞后释放。")
    add("- hysteresis / gate features 表示同样降水在不同湿润状态或季节中的非对称响应。")
    add("- station/group effects 吸收监测断面、局部地理与未显式过程差异。")
    add("")
    add("Q72 是 log 模型。为了从 log-flow 转回原始流量，使用训练残差估计 Duan smearing / log retransformation correction。当前主线中 Q72 作为 alpha=0 的基底。")
    add("")
    add("## 5. 分支二：质量守恒参照 Q78_mass")
    add("")
    add("Q78_mass 是 reach-based 的质量守恒骨架。它不单独追求最高预测分数，而是提供一个物理结构参照：本地产流沿河网向下游递推。")
    add("")
    add("```text")
    add("Q_in(r,t) = sum_{u in U(r)} f_{u,r} Q_out(u,t) + Q_boundary(r,t)")
    add("Q_out(r,t) = Q_in(r,t) + Q_local(r,t) + Q_reservoir/storage_adjustment(r,t)")
    add("")
    add("Q_local(r,t) = sum_k theta_{class(r),k} B_k(r,t)")
    add("```")
    add("")
    add("主要基函数类型：")
    add("")
    add("```text")
    add("surplus, threshold, extreme, memory,")
    add("slow_threshold_memory, wet_surplus, dry_buffer, base_ppt")
    add("```")
    add("")
    add("参数按 `reach_class × basis` 估计，并有非负约束和先验收缩：")
    add("")
    add("```text")
    add("min_theta ||Qobs - Q78_mass(theta)||^2 + lambda ||theta - theta_prior||^2")
    add("theta >= 0")
    add("```")
    add("")
    add("Q78_mass 的严格验证中位 NSElog 为 0.438682，明显低于 Q72，但它提供了质量守恒方向。")
    add("")
    add("## 6. 最终主线：reach_class 轻质量约束融合")
    add("")
    add("最终预测在 log space 中把 Q72 和 Q78_mass 做几何融合：")
    add("")
    add("```text")
    add("log(Q_main_{i,t})")
    add("  = (1 - alpha_c) log(Q72_{i,t})")
    add("    + alpha_c log(Q78_mass_{r(i),t})")
    add("")
    add("Q_main_{i,t}")
    add("  = exp[(1 - alpha_c) log(Q72_{i,t}) + alpha_c log(Q78_mass_{r(i),t})]")
    add("```")
    add("")
    add("其中 `c = reach_class(r(i))`，`alpha_c` 是 reach_class 级别的质量骨架牵引强度。")
    add("")
    add("alpha 选择规则是在 2016-2018 inner validation 中，对每个 reach_class 选择满足技能底线的最大 alpha：")
    add("")
    add("```text")
    add("median NSElog >= alpha0 median NSElog - 0.015")
    add("median KGE    >= alpha0 median KGE    - 0.030")
    add("median |PBIAS| <= alpha0 median |PBIAS| + 4 percentage points")
    add("good_count >= alpha0 good_count - 1")
    add("```")
    add("")
    add("选择得到的 alpha：")
    add("")
    add(
        md_table(
            class_alpha[
                [
                    "reach_class",
                    "alpha_c",
                    "station_count",
                    "inner_Q72_NSElog",
                    "inner_main_NSElog",
                    "inner_Q72_KGE",
                    "inner_main_KGE",
                    "inner_Q72_absPBIAS",
                    "inner_main_absPBIAS",
                    "inner_Q72_good",
                    "inner_main_good",
                ]
            ],
            floatfmt=3,
        )
    )
    add("")
    add("## 7. 整体严格验证结果")
    add("")
    add(f"- 验证站点数：{int(summary['validation_stations'])}")
    add(f"- median NSElog：{summary['class_median_NSElog']:.6f}")
    add(f"- median KGE：{summary['class_median_KGE']:.6f}")
    add(f"- median |PBIAS|：{summary['class_median_absPBIAS']:.6f}%")
    add(f"- good stations：{int(summary['class_good_count'])}")
    add(f"- median alpha：{summary['class_median_alpha']:.6f}")
    add(f"- mean alpha：{summary['class_mean_alpha']:.6f}")
    add(f"- 相对 alpha=0 到 mass 分支距离缩短：{summary['distance_to_mass_reduction_pct_vs_alpha0']:.2f}%")
    add("")
    add("与对照版本相比：")
    add("")
    add(md_table(validation_comp, floatfmt=3))
    add("")
    add("解释：当前主线的 median NSElog 高于 Q72 和 global alpha=0.1，低于 station-adaptive 版本一点点，但比 station-adaptive 更适合向未监测 reach 外推，因为 alpha 不是逐站单独选择，而是按 reach_class 共享。")
    add("")
    add("## 8. 按 reach_class 的验证效果")
    add("")
    add(md_table(by_class, floatfmt=3))
    add("")
    add("## 9. 水库相关诊断")
    add("")
    add("水库相关站点只按 reach/topology 名称识别，不按 station 名称识别。直接水库 reach 及下游 1-2 级被视作水库影响范围。")
    add("")
    add(md_table(reservoir_summary, floatfmt=3))
    add("")
    add("结论：水库相关站点确实是局部弱点，但不是全局主导误差来源。直接水库 reach 站点只有 2 个，且都未达到 good；水库相关范围内共 12 个站，3 good / 9 bad。")
    add("")
    add("## 10. 未达标站点误差类型")
    add("")
    add(md_table(failure_summary, floatfmt=3))
    add("")
    add("误差类型解释：")
    add("")
    add("- `moderate_shape_skill_gap`：趋势有一定能力，但涨落形状还不够。")
    add("- `shape_timing_problem`：峰谷时序或过程形状错位，PBIAS 未必很大。")
    add("- `bias_amplitude_problem`：趋势尚可，但总量或峰值幅度偏差过大。")
    add("- `shape_and_bias_problem`：形状和系统偏差都明显不好。")
    add("")
    add("## 11. 模拟效果较好的站点")
    add("")
    add("以下为严格验证期 good station，按 NSElog 从高到低排列。")
    add("")
    add(md_table(good_stations, station_cols, floatfmt=3))
    add("")
    add("## 12. 未达 good 的站点")
    add("")
    add("以下为严格验证期未达 good 的站点，包含错误类型和相对 Q72 的变化。")
    add("")
    add(md_table(bad_stations, station_cols, floatfmt=3))
    add("")
    add("## 13. 全部 104 个站逐站结果")
    add("")
    add("字段说明：`Delta_NSElog_vs_Q72 > 0` 表示最终主线相对纯 Q72 改善了该站 NSElog；`Q78_NSElog` 是完全质量分支的单独效果，不代表最终模型。")
    add("")
    add(md_table(all_stations, station_cols, floatfmt=3))
    add("")
    add("## 14. 图件索引")
    add("")
    add(f"- 验证期代表站过程图总览：`{FIG / 'representative_station_validation_hydrographs.png'}`")
    add(f"- 逐站流量过程图目录：`{FIG / 'station_hydrographs'}`")
    add(f"- 技能对比图：`{FIG / 'reach_class_skill_comparison.png'}`")
    add(f"- alpha 分布图：`{FIG / 'class_alpha_by_reach_class.png'}`")
    add(f"- 站点 NSElog/KGE/PBIAS 空间图：`{FIG / 'reach_class_station_metric_space.png'}`")
    add("")
    add("已有逐站 hydrograph 的站点索引：")
    add("")
    hidx = hydrograph_index[
        ["q_site", "reach_id", "good", "failure_mode", "reservoir_relation", "NSE_log", "KGE_2012", "PBIAS_pct", "figure"]
    ].rename(
        columns={
            "q_site": "station",
            "NSE_log": "NSElog",
            "KGE_2012": "KGE",
        }
    )
    add(md_table(hidx, floatfmt=3))
    add("")
    add("## 15. 当前模型边界")
    add("")
    add("这个版本应作为当前最好的主结果/主线，但它的边界也很清楚：")
    add("")
    add("- 它不是严格质量守恒模型，只是轻质量牵引。")
    add("- 它对已监测站点的时间预测能力较强，但未监测 reach 外推仍需要更严格的过程骨架支撑。")
    add("- 水库调度、跨境入流、人为取退水、引调水和 station-reach mismatch 仍可能造成局部误差。")
    add("- 当前未达标站点中，更多问题是峰值幅度、过程形状和时序，而不只是总量偏差。")
    add("")
    add("因此，后续如果要继续改进，优先方向不是继续堆经验特征，而是让局地产流、边界流、水库调节、取退水和拓扑匹配在可审计的过程结构中逐步增强。")
    add("")
    return "\n".join(lines)


def main() -> None:
    OUT.write_text(build_document() + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
