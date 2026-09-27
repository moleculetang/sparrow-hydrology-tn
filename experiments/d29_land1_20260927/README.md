# D29 / LAND1 共同平台与训练策略审阅包

本目录保存两轮连续工作的实际源码快照。`LAND0`是原统一来源校正 D29 的兼容实现；`LAND1`是植物—土壤—可用氮候选。2026-09-27 的新增工作将 2016—2024 月报、2021 起高频辅助与全域站点训练策略接入共同目标，并修正 LAND1 的数值表示。**正式训练已按用户要求停在检查点；本更新没有新的 2023/2024 TN 预测成绩。**此前条件前向的逐站 NSE 见[原专家报告](docs/专家新模型与全链条审计报告.md)，不能当成这次修正后的成绩。

## 当前验收状态

- 1961—2024、23,376 日、230 河段、13 土地槽的修订 LAND1 条件前向通过：局地最大质量误差 `1.215×10⁻⁷ kg`，来源标签差 `9.443×10⁻⁸ kg`，网络相对误差 `3.05×10⁻¹⁷`；库存非负及植物预算通过。门槛仍为局地／来源 `1e−6 kg`，没有放宽。参见[完整安全回执](audit/numerical_v2/authoritative_safety_history.json)。
- 外部来源、来源标签和计划去向的 64 年输入哈希逐年相同。随模拟库存反馈更新的年度植物目标可有舍入级变化；不能把它称为新的观测输入。见[身份对照](audit/numerical_v2/scientific_input_identity.json)。
- 独立目标重算一致；私有实验工程测试 104 项通过。公开副本禁读私有实验目录，83 项合成测试通过，见[公开测试回执](public_validation.json)。
- 完整 23 坐标导数验收及正式门以[数值报告](docs/专家数值与设备审计报告.md)和[正式门回执](audit/numerical_v2/formal_gate.json)为准。**前向验收不等于优化收敛，也不证明真实 TN 动态改善。**
- RTX 4060 Ti 的简化一年陆地前向试验为 CPU 0.450 秒、GPU 0.0867 秒；它缺土地转换、来源分账和伴随。真实 64 年目标只把响应映射移到 CUDA，CPU 两次 129.83／135.17 秒、CUDA 映射 140.20 秒，目标和梯度等价但无速度收益。当前不启用 GPU 训练后端。

## 阅读与源码入口

1. [专家数值与设备审计报告](docs/专家数值与设备审计报告.md)与[实际方法与偏离](docs/数值修正与设备实验_实际方法与偏离.md)给出本次修正、失败尝试、数值验收及设备实验的证据边界。
2. [原专家全链条报告](docs/专家新模型与全链条审计报告.md)、[原实际方法](docs/实际方法与偏离.md)及[37 项问题对照](docs/历史37项问题逐项修正对照.md)是修订前的条件性研究记录。其旧质量失败数值不是当前候选的状态。
3. [私有资料边界](PRIVATE_DATA.md)和[训练源码清单](source_manifest_training_v2.json)说明本包的可复现范围与字节身份。

|路径|职责|
|---|---|
|`d29_platform/land1.py`|LAND1 原参考核、状态与伴随；保留用于前后对照|
|`d29_platform/precision_candidate.py`、`precision_transfer.py`、`mixture_tags_candidate.py`|当前待审阅的高位／低位数值核、土地转换与来源标签实现|
|`d29_platform/coupling.py`|H1 响应映射、河网／水库和 OU 同支持读出|
|`d29_training/annual_chain.py`、`land1_adapter.py`|完整年度连续递推、梯度和 23 参数接口|
|`d29_training/observations.py`、`objective.py`、`metrics.py`|月报、HF 异常、三策略目标与 NSE 评价|
|`scripts/verify_land1_precision_gradient_v6.py`、`verify_authoritative_safety_history.py`|私有完整历史导数／质量验收入口|
|`scripts/benchmark_full_objective_gpu_mapping.py`|完整目标的 CPU／仅映射 CUDA 对照；不实现 GPU 陆地核|

`precision_candidate_pre_authoritative.py`、`mixture_tags_direct_failed.py` 等文件保留失败尝试身份，**不是正式运行入口**。`scripts/run_land1_numerical_v2_queue.py` 是已停止队列的可恢复代码，不代表本 PR 自动启动训练。源、植物活动、H1、河网、OU 及 TN 标签各自有身份；物理模型不读取评价 TN。

## 公开可运行检查

Python 3.11 和依赖见[requirements.txt](requirements.txt)。在本目录运行：

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
python verify_public.py
```

公开检查使用合成夹具并阻止读取原私有实验目录。完整历史与真实 TN 复现需要本地冻结的 H1、230 河段、LUH/CLCD/MOD17/农业/沉降适配、OU、站点资格与观测表；本 PR 不发布监测值、驱动数组、栅格、逐日预测、优化检查点、缓存、密钥或环境文件。缺失资料不得自动置零、复制年份或从 TN 反推。`requirements.txt` 的 CUDA 版 Torch 仅用于可选设备实验；正式候选为 CPU float64。

所有水质效果表按日、月、空间口径分别给出 NSE；零方差或覆盖不足标不可定义。质量、梯度与速度验收的 NSE 不适用。当前修正未重新评价真实留出，因此不宣称振幅、相位或 NSE 已改善，也不认证来源与水文准确。此 PR 供审阅，不自动合并或替换主线。
