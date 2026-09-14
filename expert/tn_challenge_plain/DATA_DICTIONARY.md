# 数据定义、索引和边界（纯文本版）

所有数组均保存为 `data/arrays/*.csv` 纯文本；`array_layout.json` 给出每个数组的shape、dtype、C序展开列数、各分片行范围及SHA256。第一行为列名，后续为数值；float64用17位有效数字，已验证重读后每个数值字节与原NPZ一致。CSV只改变存储格式。每天覆盖1961-01-01至2024-12-31，共23,376天、768个月。读取入口为 `text_arrays.read_arrays`，不需要任何压缩包或二进制数据。

| 数组 | 形状/单位 | 含义 |
|---|---|---|
| dates, months | 日/月份数；自1970-01-01的整数天 | 公历日期、月首 |
| mid | 日数；零基月索引 | 每日所在月份 |
| starts, stops | 月数；日索引 | 月内半开切片[start,stop) |
| source | 月×39；kg N/月 | S1四类总源，模型月首注入 |
| source_tags | 月×39×4；kg N/月 | 顺序：肥料、粪肥、农田固氮、大气沉降 |
| crop | 月×39；kg N/月 | 潜在作物摄取需求，月首申请、按库存限幅 |
| contact | 日×39；无量纲 | 原冻结接触水量/(接触水量+上层储水)，精确0仍禁止动员 |
| fast_fraction | 日×39；[0,1] | 水文快路比例，经共享aq参数作氮分配调整 |
| lower_release | 日×39；[0,1] | 慢水释放比例，控制L库出流 |
| fast_water, slow_water | 日×39；m³/日 | 冻结局地快/慢水体积 |
| official_water | 日×39；m³/日 | 原水文正式河段输移体积，辅助水量核查 |
| h_day, h_month | 日/月×39；day/m | 原水力暴露时长除以河道水深；主模型使用h_month |
| temperature | 日×39；摄氏度 | 冻结水文温度，供诊断；当前闭合不使用 |
| upper_water | 日×39；mm | 上层响应储水量 |
| percolation | 日×39；mm/日 | 向下层渗漏量 |
| soil_wetness | 日×39；[0,1] | 按原冻结土壤容量构建的湿润度W |
| release_fraction | 日×4；[0,1] | 每个水库的释放/(释放+期末水储量)，决定氮库释放比例 |
| released_water | 日×4；m³/日 | 冻结水库水量出口；不能改成拟合TN状态 |
| enabled | 日×4；bool | 水库启用标记，参与原捕获/旁路规则 |
| static_raw | 39×7；见字段名 | 原环境参数映射的未标准化属性（其中log字段已取对数） |
| area_ha | 39；ha | 原局地汇水区面积 |

static_fields依序为 `log_awc_0_200_mm`、`bulk_density_0_30_g_cm3`、`glhymps_log10_permeability_m2`、`glhymps_porosity`、`log_dem_slope`、`log_predev_annual_precipitation_mm`、`log_predev_annual_pet_mm`。这是原模型的现成属性，不应再对log字段重复取对数。

`topology.json` 的数组列顺序由 `global_reach_ids` 指定；`order`、`downstream`、`terminal`、水库controls/target为**零基本地索引**。CSV的 `reach_id` 则为**一基本地编号**，`global_reach_id`保留原230河段编号。`global_reservoir_index`保留原水库索引；运行使用重映射后的 `reservoir_index`。`terminal_global_ids`区分北江56/东江22，不要与原T/Y组件c04/c05的局部索引混淆。所有选入分量上游完整，没有省略的外部氮边界；4座水库的控制和目标河段都在包内。

观测CSV的每行粒度为**站×年×月**，不是日样本，也未证明为流量加权月均。`tn_mg_l`为原TN，`observation_id`为原唯一ID，`station_key`为原站名键。`downstream_fraction_on_reach`决定河段内部观测位置；预测使用上游传来负荷衰减加本河段相应比例局地负荷，除以同一边界的冻结月水量，再乘1000得到mg/L。**不能直接用河段出口浓度代替站内位置算子。**

`primary_gate`和`primary_exclusion_reason`沿用原空间支持审计；白盆珠位置待解的108条只保留追溯。`sampling_date_status`、`source_provenance_tier`、`raw_workbook_record_link_status`、`prepared_source_file/row`、`quality_flags`保留原资料可信度信息，其中本地路径是追溯标识，不是下载链接。`coverage.csv`只列实际有记录的站年；整年无记录的站年需要与完整站年笛卡尔积比较，不能把“无行”当作全年完整。

预测输出为mg/L；物理账本的M、L、可用库存和水库存氮为kg N，fast/slow/loss/uptake为逐日kg N。`reference_only/*summary.csv` 的NSE/r按至少8条、正观测方差站计算，缺失指标留空；原完整数据始终用于RMSE/偏差，不删掉高值再称精度改善。

哈希清单验证本次发布文件身份；`provenance.json`记录上游缓存/观测输入/核心代码SHA256。这证明追溯和复制一致性，不会补全未获得的原始采样定义，也不是对观测真值的认证。
