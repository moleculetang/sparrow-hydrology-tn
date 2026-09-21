# 20260920_4 新近动员N快路旁路结构筛查

**已完成。三个配置物理通过；HYDRO-SELECT与FULL-BYPASS均未获事件三误差联合改善。不进行拟合或升级主线。**

- [专家诊断报告](reports/专家诊断报告.md)
- [实际方法与偏离](reports/实际方法与偏离.md)
- [独立完成审计](reports/独立完成审计.md)
- [简短结论](reports/简短结论.md)

## 数据与输出导航

data/protocol.json登记科学协议；input_manifest.json登记旧目录来源哈希；lineage_H1.json及parent_lineage_H1.json登记H1身份；evaluation目录保存冻结观测/事件/抽样；science_code_freeze.json为预测前物理核身份；prediction_freeze.json为评价前预测身份；delivery_manifest.json为交付文件身份。

outputs每配置：land_history.npy轴为通道×23376日×230河段，日期1961-01-01至2024-12-31，河段顺序随lineage/运行域。通道依次fast、slow、legacy、mobile、lower、uptake、loss、transfer、mobile_pre、available、bypass、mixed_fast、to_lower；库存kg，通量kg/日。source_tags.npy为通道×日×试点河段×四源，具体通道/河段见summary.json。bypass_fraction无量纲，q为每日概率；bypass_implied_concentration单位mg/L。

daily_station.parquet为2021—2024的116站日质量kg、水量m³、浓度mg/L；daily_HF_history为固定HF站完整历史。annual_reach_ledger按年/河段登记累计通量和年末库存。mass_redistribution.csv中的库存是报告时期最后年末值，不是多年年末库存相加。源标签仅158/225继承范围。river_history保存河网库存、损失和出口。

event_scores、event_coverage、event_pairs为逐事件与资格账本；reports/event_curves.parquet可重画逐事件曲线。three_error_summary同时保留站等权和事件直接平均，两种分母不可混用。centered_errors为逐站月中心化MSE。monthly_predictions与monthly_station/group/overall是116站原始月报面板；HF_daily_predictions另存，不能混合标签。bootstrap_changes为1000次配对变化，抽样源在data/evaluation。

state_*.npz和checkpoint.json保存实际日递推断点；logs保存追加资源、累计尝试和退出。所有运行脚本在scripts；旧代码只作为已登记继承算法。复现应在归档副本中按实际方法执行，以免覆盖哈希固定交付。
