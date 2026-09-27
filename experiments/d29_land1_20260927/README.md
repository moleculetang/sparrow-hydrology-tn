# D29新陆地模块 LAND1：主线候选代码与专家审阅包

本包对应2026-09-27实际运行的**共同平台/LAND1主线候选**。它是待审查的研究实现，尚未替换已验证的生产主线。代码、配置和必要旧模型依赖可逐式检查；公开合成测试可以独立运行。

**目前结论：**1961—2024、230河段条件前向已完成，但新模型未经真实TN联合校准。振幅增大而日NSE和月内NSE总体恶化。实际大库存局地/来源绝对质量误差仍超过既定门槛，因此正式校准保持阻塞。不能从运行完成推断源资料准确、结构有效或具有预测能力。

## 阅读顺序

1. [专家新模型与全链条审计报告](docs/专家新模型与全链条审计报告.md)：逐站NSE、月内动态、输入/初态贡献与数值限制。
2. [实际方法与偏离](docs/实际方法与偏离.md)：文献默认、真实运行方式、供氮受限情景及失败记录。
3. [37项历史缺口逐项对照](docs/历史37项问题逐项修正对照.md)和[输入矛盾处置](docs/输入矛盾核查与修正进展.md)。
4. [公开测试](verify_public.py)、[源码哈希](source_manifest.json)、[私有数据/复现边界](PRIVATE_DATA.md)。

## 模型与源码导航

`独立来源/植物活动 → LAND0或LAND1 → 局地快慢N → H1河网/水库 → OU质量和水量 → 同支持浓度/评价`

|文件|职责|
|---|---|
|[land1.py](d29_platform/land1.py)|P/Sa/Sp/可用N/L前向、完整历史伴随、被动来源标签、守恒土地转换、补偿求和/检查点|
|[land1_reference.py](d29_platform/land1_reference.py)|独立PyTorch小核，用于前向与梯度对照|
|[conditional_inputs.py](d29_platform/conditional_inputs.py)|LUH、农业、沉降、MOD17等到日质量/植物活动的实际条件性适配|
|[mineralization.py](d29_platform/mineralization.py)、[source_definitions.py](d29_platform/source_definitions.py)|矿化公式/导数和来源定义门|
|[legacy.py](d29_platform/legacy.py)、[vendor/legacy22](vendor/legacy22)|LAND0兼容与冻结H1/原D29依赖；不是第二套已校准LAND1|
|[coupling.py](d29_platform/coupling.py)|快慢出口、河网水库、OU同支持质量/水量；拒绝TN字段进入物理采样接口|
|[objective.py](d29_platform/objective.py)、[metrics.py](d29_platform/metrics.py)|训练折隔离目标；日/月内NSE、月报/事件与同步月块抽样|
|[numerics.py](d29_platform/numerics.py)、[contracts.py](d29_platform/contracts.py)|多步长/边界/折点验收、输入和身份门|
|[run_conditional_reference.py](scripts/run_conditional_reference.py)|本次全历史前向入口，默认严格植物计划；显式选择潜在计划情景|

LAND1先守恒搬移土地库存，然后矿化昨日Sa/Sp；当天新有机输入次日起矿化。外源分矿质、有机和植物入口；植物摄取/归还是内部转移。可用质量A按 `E=p_mob*A` 动员，`F_fast=f*E`、`J=(1-f)*E`，非出口损失为 `p_loss*(A-E)`，剩余可用库存 `(1-p_loss)*(A-E)`；慢池以 `F_slow=ell*(L_old+J)` 释放。慢释放、动员和快慢映射仍继承H1/D29，未自动修复旅行时间或水龄。

旧M寿命及七项寿命空间参数退出LAND1；没有把旧有效寿命解释为新真实生地化参数。来源标签说明质量来历，不等于NO3/NH4/颗粒氮。城市及未解析表面采用被动保留的边界情景，不是城市冲刷模型。

## 公开可运行验证

建议Python 3.11及CPU版PyTorch；本次验证环境版本见[requirements.txt](requirements.txt)。在此目录运行：

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
python verify_public.py
python scripts/validate_potential_activity.py
python scripts/validate_precision_revision.py
```

可在现有`conda sparrow`执行；合成测试不需要TN或H1驱动。`verify_public.py`只选择不依赖私有文件的测试，禁止读原实验目录。后两个脚本是合成分支/精度检查，输出到被忽略的`outputs/`。它们不是私有全历史TN复现。**不要直接以全部测试通过为预期执行整个tests目录**：`test_contracts.py`含明确依赖本地私有日沉降拒绝夹具的集成测试。

发布时的新验证结果见[public_validation.json](public_validation.json)。原实验72项测试、28项分支检查、24项输入检查和合成全历史检查的历史回执位于[audit](audit)，与本次公开测试分开。最小差分步长出现消差误差仍保留，判定至少两个相邻步长一致通过，不能取最后两个误差最小值。

## 完整私有运行需要什么

本PR不包含原始监测/驱动数组、格网、完整逐日预测或检查点。实际脚本保留原Windows路径与实验依赖，以保持源码身份；**并非换一台电脑即可运行的完整数据产品**。详见[PRIVATE_DATA.md](PRIVATE_DATA.md)。需要已有相同身份的H1/OU数据、LUH状态/转换、分地类沉降、MOD17、农业活动和SON参考；评价另需合法TN支持。公开锚参数保存在[anchors.json](vendor/legacy22/input_potential_v2/configs/anchors.json)。

冻结目录及资料齐备后的实际运行命令是 `python scripts/run_conditional_reference.py --potential-activity`；缺资料时不填零、不自动借TN。此命令会在当前副本写输出，应另建可写工作目录并保留封存结果。原严格植物计划模式会拒绝本次不相容的计划，不应为了跑完而关闭拒绝门。

## 文献默认与必须保留的限制

- [条件配置](config/conditional_reference.json)逐项记录出处、改编与排除边界。SON来自现代SOC/14，活性比例.02；这不是1961实测。MOD17只约束非耕地碳生产，不是外部氮。
- [矿化配置](config/mineralization_reference.json)采用固定一阶率：活性半衰期90天、保护参考270年。土温仅2006—2024月尺度，1961—2005缺失，故温度式未启用。文献原库存定义与本模型不同，参数不是可直接移植的物理真值。
- 严格湿沉降/H1日雨仍有93河段月冲突；本次统一月内均分只是独立研究情景。真实农业事件日期、直接入河来源、自然BNF等仍不齐。
- 严格植物计划1968年因供氮不足中止；另命名的潜在计划版本按可用植物N实现去向，并记录短缺。2024归还计划仍缺7.40%，不能称NPP/收获全部实现。
- 实际全历史局地误差最大1.1444e−5kg、来源状态差5.4479e−5kg，均超1e−6kg绝对门。补偿求和和显式诊断继续没有把`strict_mass_pass`改为真；相对误差小不能替代绝对验收。正式28路径未派发。
- 2023/2024均使用冻结F23 U水文/过程条件。2024不是F24重新拟合或未来预报；新旧多因素改变不作单因素因果解释。

水质效果的权威核心表见[core_nse.csv](audit/final_audit/core_nse.csv)，含日/月内NSE、共同站数、两种配对中位数口径和改善比例。RMSE、偏差、相关、振幅和逐站结果见[audit评价目录](audit/supply_limited_reference/evaluation)。2024厂房大桥只有106日/5个月，资格门下合格14站；零方差/不足覆盖不补epsilon。结果为条件性描述，不是新模型校准失败或成功的定论。

## 快照、许可与发布范围

Python源码及配置按原字节复制，文档仅适配公开链接；`source_manifest.json`同时记录来源及公开哈希。未公开的原始证据明确标记，不能点击缺项链接后误以为完整数据已附上。外部论文/大模型代码未随包重新下载发布；外部参数证据为引用和改编说明，不能推断获得其全部再许可。项目代码的使用权遵循仓库权利人及原文件声明，本包不另加未经授权的许可证。

这份PR供专家审阅，不自动合并、不替换生产主线。
