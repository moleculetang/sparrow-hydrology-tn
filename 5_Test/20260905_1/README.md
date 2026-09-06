# 20260905_1：TN 改进实验的证据冻结与数值审计

本轮执行用户确认的 20260905 TN 改进计划。所有计算使用 conda `sparrow`。
本目录负责输入、历史失败、评价契约和数值问题的可追溯记录，不代表新模型已通过验证。

本轮已于2026-09-06完成全部注册实验和科学复核。结构与数值要求已完成，
TN预测精度未通过升级门槛；最终报告见
[final_report.md](E:/SPARROW/5_Test/20260905_6/final_report.md)。

## 不变量

- 水文冻结于 20260828_35（1961–2024）及 20260828_38（2025 敏感性延伸）。
- 保留矿质氮 lifetime 与水文慢水氮记忆；不增加 Active/Fresh 有机库。
- 新主预测由过程参数与守恒状态产生，不使用站点自由参数或浓度后处理偏移。
- 所有训练从 2016 年开始；最终分别拟合 F24、F25。
- 点源沿用 20260818_3/_6/_7 的排除结论。没有新数据或确定的实现问题，不重启旧 WWTP 分支。
- 不能以 pooled NSE 作为选模目标；正式标准见 `experiment_contract.json`。

## 运行

在 E:\SPARROW 下执行：

```powershell
& 'D:\ProgramData\anaconda3\Scripts\conda.exe' --no-plugins run --no-capture-output -n sparrow python -B '5_Test\20260905_1\scripts\audit_inputs.py'
```

`reports/input_audit.json` 是已执行检查的结果；`reports/input_manifest.json` 冻结父输入哈希。
`outputs/observations.parquet` 保留来源字段、主评价资格及敏感性标记。
`outputs/temporal_folds.parquet`、`outputs/reach_folds.parquet` 由观测元数据确定，不使用预测分数。

后续顺序：_2 数值修复与旧结构对照；_3 两目标×三档区域化；_4 时序结构分离实验；
_5 时间与空间验证；_6 F24/F25 全量拟合、导出与完成审计。
每阶段须记录实际运行、失败及未完成项，不能以文档或状态名代替验证。
