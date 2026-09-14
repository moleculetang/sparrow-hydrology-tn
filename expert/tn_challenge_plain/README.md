# TN专家拟合数据：不压缩的纯文本版本

**本目录不用ZIP、NPZ、NPY、pickle或Git LFS。** 数组存成普通CSV，观测仍为CSV，维度/索引/哈希为JSON，模型为Python源码。适合能够读取GitHub源码文件、无法获取外部压缩包的环境。

对应已确认的原模型与数据版本 `24fcf199b2d5e922de750905f011a774c899e012`。39河段、17主评价站、4水库，完整1961—2024驱动；原始1,504条TN中主评价1,396条。**2023、2024均只有12/17站满足年度NSE条件**。本次仅更换存储和读取方式，不重新训练或修改结论。

## 文件入口

- [输入清单与哈希](data/manifest.json)：本目录全部输入相对路径和SHA256。
- [数组形状、类型及分片顺序](data/arrays/array_layout.json)：25组数组、623个CSV分片；每个分片小于380 KB，可按普通源码文本读取。
- [全部数组CSV](data/arrays)：分片是实际数值，不是压缩内容的Base64或下载指针。
- [训练TN](data/train.csv)、[2023 TN](data/development.csv)、[2024 TN](data/evaluation.csv)、[早期回报TN](data/hindcast.csv)。
- [单位/维度/观测算子说明](DATA_DICTIONARY.md)、[水系拓扑](data/topology.json)、[原数据质量](data/quality.json)。
- [格式转换逐位一致性](conversion_validation.json)、[纯文本运行验证](runtime_validation.json)。

`text_arrays.py` 按array_layout的顺序读取CSV，去掉首行列名，再以C序恢复原shape。浮点采用17位有效数字，完整核验所有数组的shape、dtype及数值字节与原NPZ一致；不是把精度降成几位小数。布尔和整数也保留原类型。分片按行拆分，不改变时间间隔或删除早期预热。

纯文本总量约206 MB，不能通过只读README或只读取训练TN完成模型重演。若使用GitHub源码读取工具，需把输入清单内文件按原相对路径保存到运行环境；无需访问ZIP链接，也无需解压。**若该环境连CSV/源码文件也无法保存到执行环境，仍需人工传入这些普通文件；换格式不能解除环境本身的访问限制。**

## 运行

在仓库根目录执行，使用conda `sparrow`、CPU float64，每worker一线程：

```powershell
conda --no-plugins run -n sparrow python expert/tn_challenge_plain/verify.py
conda --no-plugins run -n sparrow python expert/tn_challenge_plain/run.py fit --variant M0 --start 0 --max-calls 1000 --seconds 3600 --out expert/tn_challenge_plain/runs/m0_0
conda --no-plugins run -n sparrow python expert/tn_challenge_plain/run.py fit --variant SC --start 0 --max-calls 1000 --seconds 3600 --out expert/tn_challenge_plain/runs/sc_0
conda --no-plugins run -n sparrow python expert/tn_challenge_plain/run.py evaluate --model expert/tn_challenge_plain/runs/sc_0/model.json --labels expert/tn_challenge_plain/data/development.csv --out expert/tn_challenge_plain/runs/sc_0_development
```

新环境可用本目录 `environment.yml` 创建。也可以单独保存此目录，进入目录后用 `python verify.py` 或 `python run.py fit ...`，同时把命令中的输出路径改为本目录下的 `runs/...`。读取器只依赖本目录文件；不导入旧压缩数据目录，不需要原作者的绝对路径。

M0/SC方程、解析梯度、参数界、先验、RAW目标和冷起点不变。仍从1961年演化M/L及河网水库；拟合只打开指定训练标签，预测不接受TN标签。唯一模型代码改动是 `load_data` 改读CSV；`run.py` 另把评价物理账本输出改为CSV，避免再次生成NPZ。拟合/评价结果仍写普通JSON/CSV。Numba可能生成自己的本地编译缓存，它不是所需下载数据。

本入口仍是原专家包中的缩放L-BFGS-B起步脚本，不冒称原注册八路径TRF＋精修。`reference_only`仍是原全域F23参数和预测，仅用于对照；fit不会读取它们初始化。改变窗口、公式、先验或算法时应另行记录，不能把历史条件反演0.95 NSE当作独立预测成绩。

## 原研究证据

详细 [多轮失败原因](../WHY_MANY_ROUNDS_FAILED.md)、[最新专家报告](../evidence/20260911_1/expert_diagnostic_report.md)、[实际方法偏离](../evidence/20260911_1/actual_methods_and_deviations.md) 沿用原PR内容。只下载本纯文本目录时，这些相对链接需在GitHub上查阅；它们不是运行依赖。

高浓度尾部、月采样聚合、原始记录溯源和白盆珠位置排除等限制都保留。格式转换不是新增观测核实，也不证明灰箱预测已改善。
