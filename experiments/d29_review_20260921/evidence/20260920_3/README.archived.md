# 20260920_3：湿润状态控制的氮动员

状态：有限矩阵完成，结论未通过预注册结构门槛；两个水平敏感性配置来源硬门不合格。没有拟合或替换主线。

## 阅读入口

- [简短结论](reports/简短结论.md)
- [专家诊断报告](reports/专家诊断报告.md)
- [实际方法与偏离](reports/实际方法与偏离.md)
- [独立完成审计](reports/独立完成审计.md)
- [证据图](reports/figures/structure_evidence.png)

## 数据接口

`data/protocol.json`是冻结配置；`lineage_H0/H1.json`记录配套水文、容量和数组身份。`observed_days.parquet`为UTC+8自然日、正常TN、唯一读数≥4且跨度≥12小时、每月≥10合格日的统计删除前视图，原记录ID可回溯canonical。

每个`outputs/<arm>/`包含：`daily_station.parquet`（2021—2024、116站、日质量kg/日水量m³/浓度mg/L）；`daily_HF_history.parquet`（1961—2024固定15站）；`land_history.npz`（day×230、kg）；`source_tags.npz`（channel×day×继承试点×4来源，通道fast/slow/legacy/mobile/lower/uptake/loss/transfer）；`river_history.npz`；`q.npy`（day×230概率）；年度账本、事件与月产品和硬门summary。D29没有新增q/legacy/mobile。

NPZ历史日轴固定1961-01-01至2024-12-31，河段轴依照lineage域的global_reach_ids，source_tags的reaches另存全域ID。不能将局地通量当成下游站点质量，也不能用算术浓度×月水量替代负荷。

## 复核

所有脚本使用conda sparrow下python -B运行。正式协议已有完成记录；不要直接再次运行setup.py或覆盖归档。`independent_audit.py`、`additional_checks.py`、`finalize.py`用于同目录复核并更新审计产物，旧实验只读。物理worker安装标签读取屏障；观测准备和评价另进程运行。

关键限制：回顾性资料；均值匹配两个配置不合格；资源峰值预留与长任务安全检查点有实现缺项。详见偏离报告。本轮完成后不自动启动下一机制。
