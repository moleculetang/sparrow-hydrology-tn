# 冻结方法后的空间与时空验证补充

这不是新的训练集。它补充原17站挑战包未覆盖的地域和浓度范围，用于检验专家方法在**不读取新站TN、不重估新站参数或预处理**的情况下能否推广。科学判断与判读协议见 [SCIENTIFIC_VALIDITY_AND_TRANSFER.md](../SCIENTIFIC_VALIDITY_AND_TRANSFER.md)。

新增29河段、24站、1,815条观测；与原包的河段、站点和ID无重合。完整日驱动从1961至2024，上游不截断。四个新增小水系、棉江及其上游、持续高TN源区分别标记，参见 [分组表](group_summary.csv)、[站年覆盖](station_year_coverage.csv) 和 [质量/血缘](quality.json)。2023/2024各20/24站具备年度NSE条件。

数据继续是CSV/JSON：[数组清单](data/arrays/array_layout.json)、[文件哈希](data/manifest.json)、[拓扑](data/topology.json)。新增区域没有水库；原包4水库保持不动，不能把本补充说成新水库边界检验。所有标签只有月份、原始行溯源仍不完整，极值未经原采样表核实，不纠正。

## 先冻结，再预测，最后读标签

原训练仍使用 `expert/tn_challenge_plain/data/train.csv` 的393条2021—2022观测。保存代码、参数、训练ID及设计。这里的脚本直接使用相邻纯文本包的M0/SC核心；若专家实现新共享闭合，需对相应预测接口作清楚记录，不能在看到新站结果后静默修改方法。

```powershell
conda --no-plugins run -n sparrow python expert/tn_transfer_validation/transfer.py predict --model expert/tn_challenge_plain/runs/sc_0/model.json --metadata expert/tn_transfer_validation/metadata_2024.csv --out expert/tn_transfer_validation/runs/sc_2024_prediction
conda --no-plugins run -n sparrow python expert/tn_transfer_validation/transfer.py score --predictions expert/tn_transfer_validation/runs/sc_2024_prediction/predictions.csv --labels expert/tn_transfer_validation/evaluation_2024.csv --out expert/tn_transfer_validation/runs/sc_2024_score
```

默认metadata/labels覆盖2016—2024全部新增记录。全部年预测时可不指定`--metadata`，评分也不指定`--labels`；必须先完成整个预测文件再评分，输出逐站年度和多年汇总，不能只报多年合并NSE。想区分纯空间与时空，分别读2021—2022和2024结果。旧站2024与早期回报仍使用原包评价文件。

预测器强制训练ID与原393条一致，因此不能把用过全116站标签的 `reference_only` 参数当新站独立验证。静态和动态尺度完全冻结；SC的新河段S只用旧训练年份、旧尺度、原未校准preset0和新河段冻结输入重新构建，无新TN校准。M/L从1961演化，不载入已拟合全域库存。

`TransferPredictor`另处理无水库网络的普通河段观测算子，避免原通用`where`在空水库数组上提前索引；普通河段的质量/水量公式不变。完整上游子域与原230河段前向结果已核对：M0最大差2.85e−14、SC 1.43e−14 mg/L，S规则及守恒通过。[实现审计](implementation_validation.json)只验证数值实现，没有拟合专家新模型，没有产生新科学成绩。

`static_support_audit.csv`显示新增区域大量属性超出原训练特征裁剪界。这正是推广挑战的一部分；不得用新站TN或事后重算尺度消除该挑战。所有数据曾被项目查看且现已公开，称算法隔离的回顾性留出；不是前瞻盲测。若专家用新增标签继续调参，它们就成为开发数据。

复核现有覆盖、重合与分组可运行 `audit_coverage.ipynb`。本轮不启动大训练，不合并主线，不增加未知氮源，也不将预测成功当作M/L机理被唯一识别。
