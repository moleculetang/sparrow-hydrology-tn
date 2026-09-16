# 数据、支持与训练队列

## 来源及职责

|来源|实际使用|本轮不做的事|
|---|---|---|
|`20260915_2` v3归档|原始模型可用月报、冻结S1源/需求、水文、地形土壤属性、水文一致230河段及13水库拓扑|不重新率定水文，不使用全历史统计清洗表筛选留出|
|`auto_4h`归档|canonical统计删除前的 adopted_value，正常状态、时间明确、已解决冲突的TN唯一读数；空间登记与月报对应|氨氮只作质量诊断，不拟合，不用TN−氨氮推算硝氮|
|`20260915_5`及其冻结继承算法|空间支持、无TN预处理定义、代码及对照元数据|旧参数不初始化本轮，旧库存不预热本轮|

精确文件来源和哈希见 [输入血缘](../evidence/algorithm_and_input_provenance.json)。`bootstrap_global.py`通过v3的 `InputRelease(training_mode=True, allow_sensitivity=True)` 校验输入，并把混合至2025缓存切成FULL24/FULL25。FULL24与此前已验证公共河段逐元素核对。归档loader未包含于本包：必须取得该归档，不能自行用同名任意数组替换。

HF快照实际自2021年6月始，至2025年2月13日；本次2025没有合格HF站月。月报2021—2024每年116个模型候选站；2025为1—11月1051条原始候选。缺失不是零，不补完整年度。

## 空间和时间支持

空间归属来自v3水文一致河线/集水区和坐标，而非CSV的流域文本。现有16个几何通过HF站冻结为候选，15站产生合格训练站月。实现重新核对河段ID、普通站类型、沿程比例及一对一对应。八角电站未产生合格日约束。其余94个候选不放宽准入。

同名同坐标仍只是条件性站点身份，不证明站码、迁站及审核血缘完全一致。同河段多个站全部保留。表内 `reach_id` 为1起始；模型内部路由使用0起始索引，不能混用。

原始快照时间修复、去重已在auto_4h归档完成；本轮消费最终 `monitoring_time` 与 `selected_record_id`，不重新凭浓度相似度改年。保留UTC时间转换至Asia/Shanghai。自然日为主；CHM窗口为前日20时至当日20时（offset=-4），CMFD为当日08时至次日08时（offset=8）。先限定真实时间年份，再排除跨折窗口。水文不重新聚合，因此模型日期仍是冻结日界的近似。

## 清洗和唯一站月并集

实现：[prepare_global.py](../snapshot/scripts/prepare_global.py)、[prepare_hf.py](../snapshot/scripts/prepare_hf.py)。

1. HF读取canonical中 `adopted_value`，要求TN、非负有限、正常状态、时间可用、`unresolved_conflict=False`。删除既有全历史统计标记；去重键为站点＋真实监测时刻。
2. 每折先限定允许站及训练年份。HF按站点×季节分组，至少72唯一时刻和30日期；月报按站点分组，至少12条。月报这里**不是按季节分组**。
3. 对浓度a计算z=log1p(a)，m=median(z)，MAD=max(median(abs(z−m)),0.05)。一次性标记 abs(z−m)>4.5×1.4826×MAD，或 a>10×同组其余样本均值（均值必须>0）。样本不足不统计删值，无迭代收紧，无全年替代阈值。统计异常不等于证实测量错误。
4. HF阈值在日覆盖筛选前计算。合格日≥4唯一读数且首末跨度≥12小时；合格月≥10合格日。日均为算术平均；HF月均为Σ(n_d y_d)/Σn_d。名义每天6次不冒充实际有效次数，不插值。
5. 每站月合格HF优先，否则合法月报；两者都有时月报只作来源对照，不重复拟合月水平。至少2训练月份的站才参与目标。
6. 留出HF采用未统计删除的正常读数，月报采用v3原始模型可用观测；训练阈值不删除留出高值。

未知月报按整月等日浓度均值读出，是条件假设，不认证自动月均。HF月值是所覆盖采样时段的均值，不称官方完整月代表值。

## 字段和量纲

FULL24有23376日、768月、230河段、13水库；FULL25长度和全部字段形状见 [布局哈希](../evidence/input_layout)。浮点为float64。每个日质量项是该日kg总量，每个日水量项是该日m³总量。

|模型字段|支持、含义及用途|
|---|---|
|source / crop|月×河段，kg N/月；冻结S1输入总量/潜在作物需求，月首日注入，需求不是一定实现的摄取|
|source_tags|月×河段×4，化肥、粪肥、BNF、沉降，kg N；仅试点完整追踪被动库存标签|
|fast_water / slow_water|日×河段，m³/day；冻结局地产水，站界分母和河网水量回放|
|official_water|日×河段，冻结官方边界水量对照；不是TN校准参数|
|contact / fast_fraction / lower_release|日×河段，冻结接触驱动、快路分率、L释放分率；保留上游归档定义，不由TN重构|
|upper_water / percolation|日×河段，水深型冻结水文状态/通量；log1p标准化，percolation×area_ha×10转m³|
|temperature / soil_wetness|日×河段，温度用于T/10，冻结湿润度用于W及前30日差值|
|h_month / h_day|水力暴露数组；本轮用h_month逐日重复，h_day不替换月内暴露；v_f×h须无量纲，沿用原参数/暴露单位约定|
|area_ha / soil_water_mm|面积ha及土壤水深mm；不通过像元数量重新改写面积|
|release_fraction / released_water / enabled|日×水库，释放分率、冻结释放水量m³/day、启用标记|
|mid / starts / stops / dates / months|日到月映射及边界索引，日期序列；状态演化不随观测缺测中断|
|static_raw|河段×7，无TN静态属性，具体顺序如下|

静态7列顺序：`log_awc_0_200_mm`、`bulk_density_0_30_g_cm3`、`glhymps_log10_permeability_m2`、`glhymps_porosity`、`log_dem_slope`、`log_predev_annual_precipitation_mm`、`log_predev_annual_pet_mm`。这些是已经变换的字段，不再重复取log。上游气象/水文生成及S1空间交叉表算法不是本轮校准对象；精确上游产品以归档与血缘哈希为准，本包不声称从原始气象和土地利用重新生成它们。

标签表：`station_key, observation_id, year, month, reach_id, station_type, downstream_fraction_on_reach, reservoir_index, tn_mg_l, fit_weight, fit_variance, day_index`。TN单位mg/L；`day_index=-1`是月读出，非负为1961-01-01起的日索引。HF血缘还有 `selected_record_id, monitoring_time, adopted_value, station_status`。`registry.json`只存日权重/排除状态；预测元数据走字段白名单，TN及其别名不能传入物理接口。

## 方差、隔离与2025

每站尺度是本折唯一月标签总体方差；统一下限为原NH34参考站中正方差的10%分位。空间折先去除留出及下游缓冲站的TN统计量。L/G公共站完全相同；无正参考方差停止，不补epsilon。方差用于目标尺度，不是声称观测独立误差方差。

环境标准化固定26参考河段，静态1%/99%截尾后标准化；动态参考2021—2022；β门控慢水比参考1991—2020。均由无TN驱动构造，不按本折TN重新选择属性。拟合worker在文件校验前安装标签读取屏障，只接收自己的合法训练文件；哈希读取也不得越过隔离。

2025使用敏感性水文/PET延伸及2024年度源、需求重复日历。1961—2024正式前缀核对一致；闰年与非闰年沉降月分配可能不同，按年度合同核验，不谎称每个月都相等。它不是独立同等级的2025正式驱动。
