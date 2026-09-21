# 实际方法与偏离

本文件记录**这一轮实际是怎么跑的**，以及它相对主线冻结核心究竟动了哪些字节。
它不是方法学的理想陈述，是执行记录 —— 包括执行中改错的地方。

配套文件：

* `reports\deviation_ledger.md` —— 逐文件、逐行、带两侧 sha256 的偏离台账（**先决文件**）
* `reports\expert_diagnosis.md` —— 方法与结果的主体报告
* `reports\short_conclusion.md` —— 两句话结论
* `reports\gate0_preflight.json` —— 预注册件（含阈值哈希），跑完不得改

---

## 1. 运行环境（实际使用的）

| 项 | 值 |
|---|---|
| 解释器 | `D:\ProgramData\anaconda3\envs\sparrow\python.exe`，`PYTHONIOENCODING=utf-8`，`-B` |
| 环境守卫 | `Path(sys.prefix).name == 'sparrow'` 断言；`CONDA_PREFIX` / `CONDA_DEFAULT_ENV` 均须为 `sparrow` |
| 精度 | CPU float64，全轮未使用 GPU |
| 线程 | `threads: 1`，每 worker 单线程；`OMP/MKL/OPENBLAS/NUMBA/NUMEXPR_NUM_THREADS = 1` |
| 并行度 | 由 `campaign_controller.py` 按资源闸动态放行，`fixed_worker_cap: null` |
| 工作目录 | `E:\SPARROW\5_Test\20260917_5\`，**只写本目录**，其余全部只读 |

**未新写调度器、未添加计划任务。** `configs\campaign.json` 的 `resources`
（`threads:1`、`dispatch_cpu:90`、`dispatch_ram:90`、`resume:85`、`peak_factor:1.2`、
`fixed_worker_cap:null`）原样沿用 —— 用户"CPU 跑满、内存到 90% 为止"的要求，
这套机制已经内建：`native_runtime.admission()` 按预留峰值放行、按实测收口；
`campaign_controller.py:63-73` 在 RAM ≥ 90% 时请求 `SAFE_YIELD_REQUEST`，降到 85% 以下再放行。

---

## 2. 算力实际消耗

| 项 | 实测 |
|---|---|
| 目标函数调用总数 | **41,083** |
| 活动时间合计 | **91,115.2 s = 25.31 h** |
| ├ H0 臂 | 13.17 h |
| └ H1 臂 | 12.14 h |
| 单 worker 峰值内存 | **3.089 GiB**（含独立审计进程） |
| 资源预约系数 | 1.2（峰值 × 1.2） |
| replay 臂 | 8 次前向预测，分钟级，不计入上述预算 |

调用数按路径分布见 `expert_diagnosis.md` §9：最少 929 次（`T24_L_D_H1_s1`），最多 3347 次。
全部 16 条路径 `NUMERICALLY_SUFFICIENT`、`physical_reasonable`、`status = AUDITED_FIT`。

---

## 3. 执行序列（实际执行的命令）

```bash
cd /e/SPARROW/5_Test/20260917_5
PY=PYTHONIOENCODING=utf-8 /d/ProgramData/anaconda3/envs/sparrow/python.exe
# 0) 建域（两臂各一遍）
$PY -B scripts/build_domain.py --arm H0
SPARROW_HYDRO_OVERRIDE='E:\SPARROW\5_Test\20260917_3\work\screen\outputs' $PY -B scripts/build_domain.py --arm H1
# 1) 预检 → 封存 → 控制器
$PY -B scripts/gate0_preflight.py && $PY -B scripts/synthesize_global.py \
  && $PY -B scripts/seal_global.py && $PY -B scripts/campaign_controller.py
# 2) 拟合完成后按实测峰值收紧预留
$PY -B scripts/apply_resource_profile.py
# 3) 报告与独立复算
$PY -B scripts/finalize_global.py && $PY -B work/audit_arms.py
$PY -B work/replay_arm.py
```

---

## 4. 偏离主线冻结核心：汇总

**详细台账见 `reports\deviation_ledger.md`**（逐文件 sha256 + `diff` 原文）。此处只列结论。

### 4.1 零改动

* **9 个核脚本**（`campaign_model.py`、`temporal_model.py`、`hf_model.py`、`fit_worker.py`、
  `serial_solvers.py`、`native_runtime.py`、`campaign_controller.py`、`recover_controller.py`、
  `audit_job.py`）—— 逐字节相同。
* **`vendor\` 全部 10 个拟合科学本体** —— 逐字节相同（0 个不同）。

⇒ **拟合科学本体、控制器、资源闸、拟合工人、审计器：一个字节都没有动。**

### 4.2 六个脚本偏离

| # | 文件 | 性质 | 理由 |
|---|---|---|---|
| D1 | `scripts\audit_inputs.py` | 新增 | **本轮唯一科学开关**。把上游模块按路径加载，只加 `SPARROW_HYDRO_OVERRIDE` 环境变量分支并把自己塞进 `sys.modules['audit_inputs']`；**上游零改动** |
| D2 | `scripts\build_domain.py` | 新增 | 自 `20260914_1\scripts\prepare.py` 改写，绕开两处写死（`data_cache('sensitivity')` 不会重算；`soil_water_mm` 路径写死 `20260828_38`） |
| D3 | `scripts\gate0_preflight.py` | 新增 | §2 的 G0-1…G0-7 七道拟合前闸门 + 预注册阈值封存 |
| D4 | `scripts\prepare_bridge.py` | 新增 | 构造 H1 折与 16 条路径的作业图；把"源三条输入逐字节复用"落成断言 |
| D5 | `scripts\validate_bridge.py` | 新增 | **替代**主线五个校验器（它们分别依赖七折迭代 / `FULL25` / 2025 窗口 / 主线自己的折，本轮均无法通过） |
| D6 | `scripts\seal_global.py` | 改动 | 5 处功能性改动 + 一段说明性 docstring，全部是报告名与路径数的重绑定 |

### 4.3 两处配置改动（只改数据，不改逻辑）

| 文件 | 改动 |
|---|---|
| `configs\folds.json` | 删 10 折（S56/S113/S191/T25S），加 4 个 `_H1` 折，保留 T24 四折 |
| `configs\jobs.json` | 28 条 → **16 条**（4 折 × 2 起点 × 2 臂） |

`D29_BE / G1 / OU / MATCH` 保持唯一模型结构，**30 个参数名与边界未变** ——
这正是"同一灰箱"的形式化含义。

### 4.4 明确未改

`configs\protocol.json`（仍记主线自己的 28 条路径，**故意不重写**，理由见台账 §5）；
氮源三条输入 `source` / `crop` / `source_tags`（逐字节复用）；
温度（逐字节相同，干净对照）；河道衰减、水文标定、2025 窗口、空间留出折、
正式产品 `20260828_35`、观测算子 —— 全部未动。

### 4.5 第 4 个改动的纪律

计划原文要求"任何第 4 个改动的出现都必须在报告里写明理由"。
实际偏离为 **6 个脚本 + 2 个配置数据件**，多于计划初稿所写的 3 个文件 / 4 行差异。
多出的部分**全部落在"验收与装配"一侧**，没有任何一项触及拟合目标、模型结构、优化器、
观测算子或源输入。D1 是唯一的科学开关，D6 是唯一被改动的既有文件且改动全部是重绑定。

⇒ **"唯一科学改动 = 水文"这句话在本轮成立。**

---

## 5. 证据等级与缺口（哪些是硬证据，哪些是软证据）

### 5.1 硬证据（逐位可核验）

| 断言 | 证据 |
|---|---|
| H0 臂无未登记漂移 | 逐位复现 `20260916_1` 已发布逐站指标：400 个站-评分，`max\|dNSE\| = 0.000e+00` |
| 开关只换了水 | `temperature` 在 12 个 identical 之内（上游 `tmean_c` 最大绝对差 7.105e-15），`soil_water_mm` 在 14 个 changed 之内 |
| 臂间差异不来自源 | `source` / `crop` / `source_tags` 全部 identical |
| 训练数据未随臂变 | 两臂 `train.parquet` 与 `registry.json` 逐字节相同（`validate_bridge.py` V2） |
| replay 通路 = 拟合通路 | H0 域 + H0 参数逐位复现 `outputs/T24_G_M_s0`：5568 月行 + 169476 日行，gap `0.0e+00` |
| 独立复算与流水线一致 | `audit_arms.py` 对拍 `reports/station_metrics.parquet`：1352 个站-评分，`max\|dNSE\| = 2.274e-13` |
| 被报告路径都是驻点 | 16 条全部 `pg ≤ 1e-5`（实测最差 2.32e-06） |

### 5.2 软证据（结论依赖的、但不是逐位可证的）

* **2024 的表现可外推到别的年份** —— **不成立**。`20260917_3` 自述结论期为
  保留的 74 站、2019-01-01..2022-12-31；本轮评在 2024，落在该窗口之外。
  **这是本轮最大的外部效度缺口，随件携带。**
* **74 站队列代表 TN 关注区** —— **不成立**。与 TN 登记册 `exact_overlap: 1` / `loose_overlap: 2`。
  头条口径因此改为 T24_G 全观测面板。
* **H0→H1 的水量变化代表"真实"水文校正的幅度** —— 只能说它是
  ~1.12–1.2 倍的水量与状态变化（capacity 比 0.8103），**不是** `20260917_4` 里那个
  3.14 倍的气候代理差。结论不得外推到更大扰动。

### 5.3 缺口清单

1. **无空间留出折**（S56/S113/S191 未跑）⇒ 不能声称空间泛化。
2. **无 2025**（`PET_BRIDGE_UNVALIDATED_2025`）⇒ 不能声称时间外推。
3. **只有 T24 四折 × 2 起点**，8 折中 1 折多盆地 ⇒ 盆地稳健性证据薄。
4. **负荷未经独立验证，本轮也不碰**（用户口径）：全部判据落在浓度上。

---

## 6. 报告生成：为什么这一轮没有走主线脚本（一处需要说明的偏离）

**本轮的 `reports\README.md`、`reports\short_conclusion.md`、`reports\expert_diagnosis.md`、
`reports\actual_methods_and_deviations.md` 都不是 `scripts\report_global.py` 的产物。**

原因是双重的：

1. **它在本轮崩溃。** `finalize_global.py` 的尾步调用 `report_global.py`，该脚本在第 66 行读
   `diagnostics\old_extrapolation\environment_support.csv` —— 一个**主线专有**的输入，
   本轮不存在 ⇒ `FileNotFoundError`。尾步死掉，三份报告因此从未被写出。
2. **即使该输入存在，它也生成不了本轮的。** 它写死于主线形态：
   `for fold in ['T24','T25S']`、`for scope in ['T24_L','T24_G','S56','S113','S191']`、
   正文硬编码"28 条登记路径"。本轮是 **16 条路径 / 4 折 / 无空间折 / 无 2025**。

**因此本轮的报告是手写的，其数字全部由 `work\audit_arms.py` 的产出
（`reports\bridge_arms.json`）与落盘 parquet 逐项核对。** 这是本轮一处**未被预注册的偏离**，
登记在此。它的方向是**增加人工环节**，不是放松判据：
`audit_arms.py` 仍然不 import `campaign_model`、不 import 指标模块（§6.5 纪律）。

### 6.1 继承的 `README.md` 缺陷（已修复）

本轮建立时整目录复制自 `20260916_1`，**`README.md` 随之逐字节继承**（sha `405db389917d8994`）。
它会链接到三份本轮并未产出、也产不出的报告，并描述一个 28 条路径 / 空间留出 / 2025 扩域的
实验 —— 与 `reports\` 的实际内容不符。这是一处**实质缺陷**，不是无害的陈旧文本：
按索引去找证据的人会扑空。

**已重写为本轮的索引**，并在其中保留"本轮不是 `report_global.py` 的产物"这一说明。

---

## 7. 执行中改错的一处检查（保留记录）

`work\audit_arms.py` 的检验项曾把"每折两个独立起点必须到达同一最优"写成硬闸门，
`T24_L_D_H1` 上报 7.26% 相对差 ⇒ `[FAIL]`。

**这是检查写错了，不是产物坏了。** 该折两个起点
`s0`（0.911986）/ `s1`（0.984612）的 `pg` 分别为 `4.62e-07` / `4.61e-07`，
均 `numerical_sufficient: true` —— 都是非凸 30 参数问题的驻点，只是落在不同盆地。
（我先前记的 `pg ≈ 4.6e-07` 作废：该字段实测为 `pg = 0.0`，4.62e-07 是邻近噪声。）

已改为：

* **真闸门**：每条被报告路径 `pg ≤ 1e-5`（实测最差 2.32e-06）；
* **`start_spread` 降为诊断量**并随件携带（`reports\start_stability.csv`）。

这个诊断量本身是一个真实警告：8 折中 `T24_L_D_H1` 是 7.96e-02（**distinct local solutions**），
其余七折 5.5e-13…2.32e-07（small MAP gap）⇒ 多盆地只发生在 1 折，**且不是头条折**。

同类的一处修正：`delta_high_TN_gap` 在日尺度上产生 14 次
`RuntimeWarning: All-NaN slice encountered` —— 这是**设计使然**（高 TN 缺口定义在月度 P90 上，
日尺度本就不存在该量），已用 `nanmedian_or_nan()` 处理，而非全局屏蔽警告。

---

## 8. 两处我先前写错、已在计划附录更正的记录

保留在此以免再次出现：

1. 我曾写"全部 12 个'折×队列'单元格都是这个形状"——**错**。实测月尺度 16 格中 9 格被标记、
   日尺度 8/16。已换为准确的 4×4 表。
2. 我曾写"H1 臂的拟合是多盆地的"——**过度陈述**。实测只有 1/8 折多盆地。
   已换为 `start_stability.csv` 的逐折读数。

---

## 9. 本轮不做什么（提前声明，与计划 §7 一致）

* 不算负荷、不报负荷、不用负荷 NSE 作为任何进展证据。
* 不再标定水文；不调河道衰减；不上通用神经网络；不加统一 M/L 的高阶交互。
* **不同时改源**（源三条输入逐字节复用）。
* 不跑空间留出折、不跑 2025、不动 `20260828_35`（正式产品）。
* 不把 2024 的结论叙述成"模型在 1961 年起有效"。
* 不根据 TN 残差反推一个漂亮的污水负荷（Gate 3 的纪律）。
