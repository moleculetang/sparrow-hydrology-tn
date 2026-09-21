# 20260920_5 有限机制家族筛查

已完成18个登记配置，零拟合、零求根。结果与限制请从以下报告阅读：

- [潜力审计与专家诊断](reports/潜力审计与专家诊断报告.md)
- [实际方法与偏离](reports/实际方法与偏离.md)
- [独立完成审计](reports/独立完成审计.md)
- [简短结论](reports/简短结论.md)

## 数据接口

data/protocol.json为固定矩阵；input_manifest、physical_arrays、C_arrays_freeze、science_code_freeze及prediction_freeze分别登记来源、物理数组、C水域、代码与评价前预测。原始来源保持只读，未使用2025。

各配置land_history.npy轴为通道×23376日×230河段；source_tags.npy为通道×日×2试点河段×4源。日期1961-01-01起，河段顺序为lineage_H1域topology的global_reach_ids；来源名沿用父模型。通道顺序见summary.json及scripts/family_kernel.py的CHANNELS，库存kg、通量kg/日，exchange为带符号的F向M氮转移。effective_*_concentration单位mg/L；A/B为参考体积有效浓度，C为分域浓度，不与站点TN混称。

daily_station.parquet包含2021—2024的116站质量kg、水量m³、p浓度mg/L；daily_HF_history为固定HF站全历史。event_scores/coverage/pairs、HF_daily_predictions和monthly_predictions分别登记事件、HF日与原月报支持。H1-D29复用已验证历史控制，不新增前向。

annual_reach_ledger为年通量与年末库存；reports/mass_redistribution中的库存取时期末年，非多年库存相加。原共享水库账本存river_history.npz。data/C_*保存mm水域体积、交换水量和无量纲抽取率；每个C配置通过omega索引这些共同数组。

reports/three_error_summary保留站等权及事件直接平均；comparisons和bootstrap_summary为两类控制的配对变化；monthly_group_metrics包括ALL及N/H/X/OTHER；event_curves可用于重绘；pulse_screen为无标签潜力诊断。资格和分母不可跨产品混用。

追溯限制：原共享追加日志存在少量并发交错，3条尝试时间记录缺失、部分资源采样不可恢复；原日志保留，完成块由独立检查点账本核验。锁修复仅保证修复后写入，不能补回历史。详见reports/log_integrity_audit.json。

## 重现

使用完整conda sparrow、Python float64、每worker一线程。在归档副本中依次运行scripts下setup、prepare、admit_water、extended_checks、algebra_audit、pulse_screen、worker MIX、controller、evaluate、audit、delivery_analysis、write_reports和seal。不要覆盖已封存目录；脚本默认新目录R由自身位置决定。检查点支持日递推恢复，旧文件和既有结果受身份检查保护。
