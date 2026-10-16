# 当前修正 H1 水文运行包

本包恢复生成当前 LAND1/TN 冻结水文的 **20260917_3/work/screen → 20260917_5 FULL24C** 版本。参数、共享网络、区域化分数、缩放、静态属性、面积、河网、历史 P/PET、1961 初态及水库规则均有对应文件；无需本机旧实验目录。NH₄/DO和评价TN不被重放器读取。

## 重放

仓库外层的 `assemble.py` 先恢复资产。Python 3.11、CPU float64、每进程一线程；环境原记录见 `environment_versions.json` 和原 spin-up 回执。`requirements-matched.txt` 记录当前安装的包版本，PyTorch的CUDA构建后缀属于本机发行身份；CPU重放可安装同版CPU构建，再核验跨构建一致性。

```bash
python -m pip install -r requirements-matched.txt
python replay.py --end 1961-01-30 --output check --perturbation-control
python replay.py --end 2024-12-31 --output full_history
```

完整重放命令实际执行1961—2024历史，不读取旧水库日输出来驱动水库：恢复checkpoint计算陆地快慢出流，再执行原河网和R2库水量守恒规则。原日文件仅在前向结束后用于比较。需要重算陆地预热时加 `--recompute-spinup`；按实际1961—1970日期重复周期，容差1e-8 mm、最多500周期。原基线实际2周期收敛，并已重新复核。新参数或长期新驱动下，新丰江1961库存还需按原 `historical_driver/s33_spinup.py` 的1961年10—12月92日供体规则重建；不能把原库初态永久视作真值。

`run(end, precipitation_scale, recompute_spinup)`提供可重算上游水文和水库的前向接口。30日独立干预降雨增加1%，库释放发生变化，证明不沿用冻结整套水库流量。它是固定初态的软件因果控制，不是降雨情景的科学成绩。R2沿用tau=180日、入流响应0.25、季节drawdown=0；控制规则来自研究表达，并非实际逐日调度观测。死库容、目标库容、建设日期、控制臂、下游及不确定标记见两份水库静态表和原规则源码；共享水库只保留一个物理日记录。

## 输入与权重

- `data/model.pt`：当前物理8参数、原始8参数、共享gate权重及种子260827、lambda_S=0.1805437376850875；SHA256 `fab0f9ae7dab5636b352d13a4d47a77a642b148adc25f2cd09e628376c10e426`。源是当前screen导出，不能替换成早期同名模型。
- `data/parent_fit_checkpoint.pt`、`parent_training_trace.parquet`：当前掩蔽面板重拟合终点及轨迹；后续受守恒SIG2P作用生成本包导出模型。区域分数和BFI系数另存。当前checkpoint已含完整推理所需共享网络，并非缺少权重的源码样例。
- `data/static.parquet`、`dynamic_feature_scaling.json`：原静态属性和动态特征缩放。`area.parquet`只从原Q72抽取230个物理面积，没有导出该旧大表中的TN或其他列。
- `forcing/forcing_1961.parquet`至`forcing_2025.parquet`：原实际日驱动的日期、reach_id、降雨和PET四列，分别为mm/day；逐值保留，按公历年分文件。原全字段548,327,589字节源表SHA256 `573af64090bdd4c2f442d288a0ef684dc2f5038a1dd311334c6f3b142446fb45`；这些是实际模型需要的列，不重算PET、不补值、不缩放。P/PET合计5,460,430河段日，1961—2025。当前FULL24C只取到2024；2025 ERA5/PET桥接研究限制保留。
- `data/tn_hydrology_reservoir_daily.parquet`：用户指定原件，46,053,564字节、308,633行、1961—2025、13库，SHA256 `e38f3d412f02be20d40a379277c1b731c89a08170603e02d33f0a996814d8dd2`；完整原字段和原缺项保留。当前1961—2024前缀303,888行。m3为每日体积/库存，m3/s列为相应流率，水龄矩为m3·day；文件名含TN不表示TN浓度观测。

河段表与图层对应PR10已确认的 `spatial/H1_unified_20260915_v3`，一始reach_id=1…230，面积410866.1378872674 km²。本次不重聚合新S1来源，不将其另一套几何差异混入水文重放。较早39河段降雨案例保留当时旧水文身份；不能将其H0结果与本包H1结果混用。

## 流量资料和站点口径

`observations/daily_discharge_development_2010_2018.parquet`为请求的原文件，323,222行、103站、2,730个q缺值；不能把文件行覆盖称作65站训练面板。`daily_discharge_locked_retrospective_2019_2022.parquet`为独立回顾性留出源，145,006行、102站、830个缺值，其中2020—2021有73,831行、426个缺值。保留原文件字节，没有以本次TN残差清洗或填0。

q_m3_s单位m³/s；date是源文件公历日标签，无时刻/时区字段，不能自行宣称UTC或四小时瞬时。表内原有station_norm、reach_id、terminal_tree和downstream_fraction_on_reach保留。日文件没有逐读数仪器质控码，不伪造该字段；站级代表性、原河段、映射方法、坐标及资格见 `original_gauge_representativeness_audit.parquet`。其x/y是投影坐标，不能当作经纬度。原站名和坐标按本轮用户授权保留。

`observations/current_screen/`是生成当前checkpoint实际使用的掩蔽日/月流量和登记表；s5登记74站，2010—2018合并有效日流量后57站、187,359个有效站日。原锁也记载BFI门57站保留、17站不满足覆盖；s8登记100站，另有独立的区域化拟合支持。面板、非空支持、目标权重和BFI资格是不同概念，不以原65站描述代替。

## 版本与证据边界

原实验未记录可证明覆盖本组忽略目录文件的Git提交号，本轮不补造。`source_lineage.json`登记源字节哈希、历史驱动和checkpoint锁；本次PR10提交号只标识这份发布包。`sources/`是实际加载的原方程依赖，字节不改；便携装载器只重绑根路径和具体资产路径。`historical_driver/`保留历史主运行/拟合/预热/导出入口供方法审阅，完整历史重新拟合涉及原PML、GRACE、决策锁等额外资料，不把这些历史入口称为本包已执行过的新率定。

本次实测：完整64年重放13库储水/释放/快慢来源/水龄矩误差0；10个TN水文耦合数组的值与NPY SHA256完全重现；另3个水库数组全历史逐值一致。其他温度、河道水力暴露和氮源数组未被这项重放复核覆盖，相关旧源码/冻结合同保留原身份。软件物理/字节核验NSE不适用，未计算新水文预测成绩。

这套资料关闭了“缺当前checkpoint/缩放/完整P-PET/原流量/库规则”的已定位交付缺口，支持复现冻结H1及另行设计联合约束。**它不是已实现并验收的端到端水文—TN联合率定器**：当前库和河网部分仍有NumPy/Pandas路径，需贯通参数到Q、库存、TN读出及联合目标的完整导数；原库调度与初态仍是研究表达；新S1空间支持分叉仍需整改。流量加TN不能唯一识别真实水龄。地下水/土壤水分/库水位和示踪资料未在本次补充中认证，缺项继续留空。

第三方输入沿用原提供方权利和限制；本轮明确上传授权不重授源产品许可。现役44路径及其资料未修改，PR不自动合并。
