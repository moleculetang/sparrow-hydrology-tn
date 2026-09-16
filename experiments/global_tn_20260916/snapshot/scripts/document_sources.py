"""Register inspected hydrologic method/time support references, read-only."""
import native_runtime as rt
R=rt.RUN;W=R.parents[1]
sources={
 'hydrology_objective':W/'5_Test/20260827_2/scripts/run_stage2_temporal.py',
 'cmfd_daily_builder':W/'5_Test/20260825_2/scripts/build_daily_cmfd_pet.py',
 'chm_definition':W/'0_reach_topology/data/raw/atmosphere/precipitation/chm_pre_v2/metadata/essd-17-3987-2025.pdf',
}
rt.write(R/'evidence/hydrologic_reference_hashes.json',{k:dict(path=str(p),sha256=rt.sha(p)) for k,p in sources.items()})
(R/'evidence/hydrologic_time_support.md').write_text('''# 水文方法及日界依据

本地已核查的水文训练函数 `composite_parts_strict` 从同一日轨迹构造日流量、高低流量、连续日事件形状、过去七日水量和完整月水量误差。窗口不跨训练分区，站点按终端水系分组归一化。本轮借鉴多尺度轨迹约束，未照搬流量损失权重。

CHM论文第3页定义：“Daily precipitation is defined as the cumulative precipitation from 20:00 on the previous day to 20:00 on the current day (local time in Beijing).”

CMFD日PET/温度构建脚本使用UTC日聚合八个三小时步长，对应北京时间当日08:00至次日08:00。冻结驱动沿用原日期索引，尚未证明不同驱动具有共同的24小时物理窗口。

本轮以北京时间自然日作条件性主对应，并登记CHM与CMFD两种日界敏感性。只调整观测分组，不修改冻结水文，不能称完成日界修复。高频日浓度均值与日氮质量/日水量也并非严格等价；未登记相容高频实测Q时无法量化日内协方差。

来源位置及SHA256见同目录 hydrologic_reference_hashes.json。上述方法和论文已在计划阶段阅读；不以附件中的2020年起始说法替代本地2021年6月起始快照证据。
''',encoding='utf8')
print('REGISTERED_HYDROLOGIC_REFERENCES')
