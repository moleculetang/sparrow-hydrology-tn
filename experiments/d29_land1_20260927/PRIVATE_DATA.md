# 私有数据与完整复现边界

本PR不包含原始TN/流量监测、气象/H1数组、LUH/CLCD/MOD17/SoilGrids格网与河段驱动、完整逐日预测、优化检查点、下载论文、密钥或环境目录。

公开代码是实际实验源码快照，不是已移植的独立产品。`d29_platform/legacy.py`及vendor还含原Windows实验路径；`conditional_inputs.py`读取邻接的20260926_1产品。完整运行需要本地相同哈希资料和目录结构，不能仅克隆本PR后声称重现真实TN成绩。私有运行也不能用当前源码向原实验目录写入；应在新副本修订路径并核对身份。缺资料不得自动下载、零填或借TN补齐。

关键依赖：230河段1961—2024 H1与拓扑/OU元数据；LUH逐年状态和土地转移；分地类四分量沉降；MOD17非耕地年度生产/QC；农业年源量与植物活动；SoilGrids初态参考；仅评价阶段读取TN资格/读数和冻结事件。来源/初态/过程/观测支持需保持完整身份。公开锚参数不等于新LAND1最优参数。

`scripts/prepare_matrix.py`是继承测试和旧正式路径门的快照，含旧资料缺项文字，不能当本轮最新状态。当前状态以README和专家报告为准。28条正式路径仍禁止派发。

本地报告引用但本包未公开的工件（路径仅供已有私有资料者定位）：

- `E:/SPARROW/5_Test/20260927_1/evidence/literature/CLM50_Tech_Note_CN_Allocation.rst`：CTSM器官
- `E:/SPARROW/5_Test/20260927_1/evidence/literature/CLM50_Tech_Note_Decomposition.rst`：CTSM分解
- `E:/SPARROW/5_Test/20260927_1/evidence/literature/china_residue_return.txt`：残体论文
- `E:/SPARROW/5_Test/20260927_1/evidence/literature/faostat_budget_supplement.txt`：FAOSTAT补充
- `E:/SPARROW/5_Test/20260927_1/evidence/literature/mhm_crops.txt`：mHM配置
- `E:/SPARROW/5_Test/20260927_1/evidence/literature/resorption_meta.txt`：再吸收证据
- `E:/SPARROW/5_Test/20260927_1/evidence/literature/swat_doc_plain.txt`：SWAT理论
- `E:/SPARROW/5_Test/20260927_1/outputs/agriculture_reference/annual_crop_source_reference.parquet`：农业逐行登记
- `E:/SPARROW/5_Test/20260927_1/outputs/agriculture_reference/national_commodity_mapping_and_missing.csv`：商品映射/缺项
- `E:/SPARROW/5_Test/20260927_1/outputs/agriculture_reference/receipt.json`：receipt.json
- `E:/SPARROW/5_Test/20260927_1/outputs/conditional_reference/failed_run.json`：原严格失败
- `E:/SPARROW/5_Test/20260927_1/outputs/deposition_reference/receipt.json`：沉降日历
- `E:/SPARROW/5_Test/20260927_1/outputs/final_audit/completion_receipt.json`：运行与37项审查封存
- `E:/SPARROW/5_Test/20260927_1/outputs/luh3_adapter/receipt.json`：土地转换
- `E:/SPARROW/5_Test/20260927_1/outputs/soil_reference/son_soc14_receipt.json`：SON对照
- `E:/SPARROW/5_Test/20260927_1/outputs/supply_limited_reference/audit/annual_balance_and_activity_deficits.csv`：annual_balance_and_activity_deficits.csv
- `E:/SPARROW/5_Test/20260927_1/outputs/supply_limited_reference/audit/annual_local_export_source_fractions.csv`：annual_local_export_source_fractions.csv
- `E:/SPARROW/5_Test/20260927_1/outputs/supply_limited_reference/audit/station_year_nse_decomposition.csv`：station_year_nse_decomposition.csv
- `E:/SPARROW/5_Test/20260927_1/outputs/supply_limited_reference/configuration_frozen.json`：冻结实际配置
- `E:/SPARROW/5_Test/20260927_1/outputs/supply_limited_reference/evaluation/frozen_events.csv`：frozen_events.csv
- `E:/SPARROW/5_Test/20260927_1/reports/../config/mineralization_reference.json`：矿化配置
- `E:/SPARROW/5_Test/20260927_1/reports/../d29_platform/mineralization.py`：mineralization.py
- `E:/SPARROW/5_Test/20260927_1/reports/../outputs/soil_temperature/decision.json`：文件哈希与选择回执
- `E:/SPARROW/5_Test/20260927_1/reports/../outputs/soil_temperature/mineralization_validation.json`：验收回执
- `E:/SPARROW/5_Test/20260927_1/reports/../outputs/soil_temperature/missing_full_history_months.csv`：缺失月份
- `E:/SPARROW/5_Test/20260927_1/reports/../outputs/soil_temperature/monthly_layer_inventory.csv`：逐月分层清单
