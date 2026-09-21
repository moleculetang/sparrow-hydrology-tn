# 偏离台账 — 20260917_5（旧/新水文 × 同一 D29_BE 灰箱的桥接实验）

本轮要向读者交付的一句话是：**唯一的科学改动是水文**。
这句话只有把"相对 `20260916_1` 冻结核心到底动了哪些字节"逐条列清之后才成立，所以这份台账
是本轮的先决文件，不是附录。

比对基准：`E:\SPARROW\5_Test\20260916_1\`（主线冻结核心）。
以下每条都给出两侧 sha256 前 16 位，可独立复算。

---

## 0. 冻结核心：**零改动**

`seal_global.py:30` 定义的 9 个核脚本，逐字节相同：

| 脚本 | sha256[:16] | 状态 |
|---|---|---|
| `campaign_model.py` | `b8c4a86a7424e53c` | same |
| `temporal_model.py` | `1b3217cbec5260f0` | same |
| `hf_model.py` | `88a85da13541e612` | same |
| `fit_worker.py` | `9890da85aa8fa543` | same |
| `serial_solvers.py` | `fbc5311f6b9f1873` | same |
| `native_runtime.py` | `be469ca09bbbbd88` | same |
| `campaign_controller.py` | `0001b52f26c95cb1` | same |
| `recover_controller.py` | `c2a0fd252987401a` | same |
| `audit_job.py` | `b3907f158105accd` | same |

`vendor\` 全部 10 个拟合科学本体，逐字节相同（**0 个不同**）：

```
expert/tn_challenge/{mix_routing,model,routing,run,sc_kernel,support_integral,text_arrays}.py
research/{closures,tagged_transport}.py
transfer_research/scientific_models.py
```

**拟合科学本体、控制器、资源闸、拟合工人、审计器：一个字节都没有动。**
本轮的全部差异只发生在"域是怎么造出来的"和"验收用哪些报告"这两件事上。

---

## 1. 六个脚本偏离

`scripts\`：主线 56 个文件 → 本轮 61 个。差集恰好是下面 6 条。

### D1 — `scripts\audit_inputs.py`（**新增**，sha `07d2aefccb2ae348`）

**理由**：本轮的**唯一科学开关**。
上游 `20260905_1\scripts\audit_inputs.py` 里 `HYDRO` 是普通字典，`HYDRO['sensitivity']`
写死 `20260828_38`。本文件把上游模块按路径加载进来，只加一个环境变量分支：

```python
_ov = os.environ.get("SPARROW_HYDRO_OVERRIDE")
if _ov:
    HYDRO["sensitivity"] = Path(_ov)
```

并把自己塞进 `sys.modules["audit_inputs"]`，使下游 `tn_reference.py:12` /
`fc_legacy.py:12` 的 `from audit_inputs import HYDRO` 拿到本模块。**上游零改动**，
`product` 字符串保持 `'sensitivity'`（语义正确：校正水文本就是 sensitivity 线经冻结筛选重跑）。

### D2 — `scripts\build_domain.py`（**新增**，sha `dad413aab7444da6`）

**理由**：自 `20260914_1\scripts\prepare.py` 改写，绕开两处写死。
`prepare.py:30` 的 `data_cache('sensitivity')` 只是 `torch.load` 一个预建 `.pt`，**不会重算**，
所以 H1 必须直调派生函数；`prepare.py:40` 把 `20260828_38` 写死，必须改为随 `HYDRO` 走。
产出物不变：`data/domains/<域>/{26 个 .npy, topology.json, arrays.json}`。

### D3 — `scripts\gate0_preflight.py`（**新增**，sha `358648146af26b32`）

**理由**：计划 §2 的 G0-1…G0-7 七道拟合前闸门，产出 `reports\gate0_preflight.json`。
本轮的预注册阈值（§5.3 两层判据及其操作化定义）在这一步写死并连同哈希封存，跑完不得改。

### D4 — `scripts\prepare_bridge.py`（**新增**，sha `177239c7a2f9e870`）

**理由**：构造 H1 折的域覆盖与设计（`FULL24C`、四个 `_H1` 折、16 条路径的作业图），
并把 §3 的"源三条输入逐字节复用"落成断言。这是本轮的域/作业装配器，主线无对应物。

### D5 — `scripts\validate_bridge.py`（**新增**，sha `91e3762fe6c008f7`）

**理由**：**替代**主线的五个校验器。
`validate_global.py` 迭代七折（本轮只建四折 ⇒ `StopIteration`）且加载 `FULL25`；
`validate_common_outputs.py` / `verify_extension_contract.py` 同样依赖 `FULL25` 与 2025 窗口；
`validate_hf.py` 打分主线自己的折。这五个**在本轮无法通过**，且不是本轮的输入有问题。

按自设纪律：**替代者不得弱于被替代者**，否则等于对启动闸门做了一次静默降级。
`validate_bridge.py` 的 V1…V6 中，V3（温度逐字节相同而土壤水不同）与 V6（30 列中心差分
Jacobian）覆盖了主线校验器本会覆盖的部分；V2（臂分离）与 V4（源身份）是本轮自己的保证，
主线无对应物。报告同时写 `process.peak_gib` 与 `peak.peak_gib` 两种形状，使 `seal_global` 只需换一个文件名。

### D6 — `scripts\seal_global.py`（**改动**，sha `0a1af3ca3c3f1db9` → `1ffee69828e94729`）

5 处功能性改动 + 一段说明性 docstring。逐条如下（`diff` 原文）：

| # | 主线 | 本轮 | 理由 |
|---|---|---|---|
| 1 | 断言五个主线校验报告（`global_validation` … `common_domain_validation`） | 断言一个 `bridge_validation.json` | D5：那五个在本轮无法通过 |
| 2 | 准入折 `T25S_G_D_s1` | `T24_G_D_H1_s0` | 本轮不建 T25S 折（2025 携带未验证的 PET 桥） |
| 3 | 峰值取自 `global_validation.json` + `boundary_validation.json` | 取自 `bridge_validation.json`（两种形状） | D5 |
| 4 | `driver_identity='Full 230 reaches … 2025 sensitivity prefix tested'`，`paths=28` | `driver_identity='Corrected hydrology (20260917_3) …'`，`paths=len(jobs)` | 身份串须描述本轮；路径数**不再写死** |
| 5 | `print('SEALED_28_PATHS')` | `print(f'SEALED_{len(jobs)}_PATHS')` | 同上，避免把 28 印成常量 |

**未改动**：准入算术、隔离身份测试（`PASS_ISOLATED_IDENTITY`）、冻结哈希清单的构造逻辑。

---

## 2. 两处配置改动（**只改数据，不改逻辑**）

| 文件 | 主线 sha | 本轮 sha | 改动内容 |
|---|---|---|---|
| `configs\folds.json` | `de3790c14903b1a8` | `64ea3652fd7f43fc` | 删去 S56/S113/S191/T25S 共 10 折；新增 4 个 `_H1` 折（`domain: FULL24C`，加 `arm`、`h0_fold` 两键）；保留 T24 四折 |
| `configs\jobs.json` | `b3350e4724c48151` | `85951e13fe1ff9e5` | 28 条 → **16 条**（4 折 × 2 起点 × 2 臂） |

两文件都只被配置读取器消费，不含可执行逻辑。`D29_BE / G1 / OU / MATCH` 保持唯一模型结构，
30 个参数名与边界未变 —— 这正是"同一灰箱"的形式化含义。

---

## 3. `work\` 下的三个新文件（**不属于冻结核心**）

| 文件 | 角色 |
|---|---|
| `work\audit_arms.py` | 独立复算器。**不 import** `campaign_model`、不 import 指标模块，从 parquet 重建配对并自己写 NSE/r/RMSE/log-RMSE/PBIAS。母本为 `20260917_4\work\audit_metrics.py` 的纪律（它同样不调用被审方的函数） |
| `work\replay_arm.py` | §5.2 P2 的补偿对照与 §6.4 的通路负对照（H1 水文 × H0 参数的前向预测） |
| `work\experiment_clock.json` | 本轮自建计时件 |

三件都在 `work\` 之下，不参与 `seal_global.py` 的冻结哈希清单（该清单只覆盖
`scripts` 核 + `vendor` + `configs` + `data/{domains,designs,folds}` + 六个 `data\` 件）。

---

## 4. 明确**没有**改的东西（否则"只换了水"不成立）

| 项 | 状态 |
|---|---|
| `configs\protocol.json` | **未改**。仍记主线自己的 `paths=28`、`time_folds=['T24','T25S']`、`spatial_outlets=[56,113,191]`。见 §5 |
| `configs\campaign.json` 的 `resources` | **未改**（`threads:1`、`dispatch_cpu:90`、`dispatch_ram:90`、`resume:85`、`peak_factor:1.2`、`fixed_worker_cap:null`）。用户口径是"CPU 跑满、内存到 90% 为止"，而这套**已经内建**，无需另写调度器 |
| 源（氮）三条输入 `source` / `crop` / `source_tags` | 逐字节复用（`validate_bridge.py` V4 断言，`audit_arms.py` P3 复核）。**先不要同时改源** |
| 温度 | 逐字节相同（`temperature` 在 26 数组中列为 identical；上游 `tmean_c` 最大绝对差 7.105e-15）。这是一个干净的对照 |
| 河道衰减、水文标定、2025 窗口、空间留出折、正式产品 `20260828_35` | 全部未动 |
| 观测算子 | 未动。仍用主线已解决的 `downstream_fraction_on_reach` / `station_type` / `reservoir_index` 与 pre-dam / dam_outlet / postdam_mixed 混合 |

---

## 5. `configs\protocol.json` 的陈旧字段：为什么不重写，以及谁在读

`protocol.json` 是**主线的注册件**，记录主线的 28 条路径、两个时间折、三个空间出流点。
本轮既没有构建也没有资格静默改写它。重写它反而会让"主线注册"这件事失去可追溯性。

**它是否影响本轮判据？没有。** 全仓检索结果：

* **没有任何共享脚本读取** `paths` / `time_folds` / `spatial_outlets`。
  `bootstrap_global.py:55` 只**写**该文件（且第 14 行断言它此前不存在）；
* `work\audit_arms.py:279` 只读 `bootstrap`（1000）与 `seed`（1729）两个标量，
  即计划 §5.3.2 要求"直接复用、不另设随机源"的那两个；
* `gate0_preflight.py:66` 只在报告里引用同样的 1000 / 1729。

**本轮权威的路径数是 `len(configs/jobs.json) == 16`**，由两处落盘产物一致地给出：

* `reports\launch_validation.json` → `paths: 16`、`status: PASS_LAUNCH_VALIDATION`；
* `reports\bridge_arms.json` → `paths: 16`，并附
  `paths_note = "from configs/jobs.json; configs/protocol.json still records the
  mainline's 28 and is deliberately not rewritten"`。

（`seal_global.py` 的 `SEALED_{len(jobs)}_PATHS` 只打到控制台，未落盘，故不作为可核验来源。）

**预注册件**：`reports\gate0_preflight.json`，`status: PASS`，七道闸门
G0-1…G0-7 全部 `true`、`blocking_failures: []`；§5.3 两层判据及其操作化定义的原文
连同 `preregistration_sha256 = 7cdbc44e8d8cc4630f38917f1ca2af1e7cd926c86c3de10d60254567594474df`
一同封存于该文件。跑完不得改。

---

## 6. 第 4 个改动的纪律

计划原文：**"任何第 4 个改动的出现都必须在报告里写明理由"** —— 否则"唯一科学改动 = 水文"
这句话不成立。

本轮实际偏离共 **6 个脚本 + 2 个配置数据件**（多于计划初稿所写的 3 个文件 / 4 行差异）。
多出的部分全部落在"验收与装配"一侧，没有任何一项触及拟合目标、模型结构、优化器、
观测算子或源输入；D1 是唯一的科学开关，D6 是唯一被改动的既有文件且改动全部是报告名与
路径数的重绑定。逐条理由见上。**"唯一科学改动 = 水文"这句话在本轮成立。**

---

## 7. 两处晚发现项（跑完之后才查清，必须补登）

上表 D1–D6 是**建轮时就知道**的偏离。以下两条是结果已出、写报告时才查清的，
方向都是"继承来的东西在本轮不成立"。补登在此，以免读者按继承的索引去找证据而扑空。

### 7.1 `scripts\report_global.py` 在本轮**不适用**（未被列入 D1–D6）

整目录复制时 `report_global.py` 随 `scripts\` 一起进来，与主线**逐字节相同**，
因此它不构成对冻结核心的偏离，前面几节也就没有它。但它在本轮**跑不通**，有两重原因：

1. **它崩了。** `finalize_global.py` 的尾步调用它，第 66 行读
   `diagnostics\old_extrapolation\environment_support.csv` —— 一个**主线专有**的输入，
   本轮不存在 ⇒ `FileNotFoundError`。尾步死掉，主线形态的三份报告因此从未落盘。
2. **就算那个输入存在，它也生不成这一轮的。** 写死于主线形态：
   第 25 行 `for fold in ['T24','T25S']`；第 37 行
   `for scope in ['T24_L','T24_G','S56','S113','S191']`；第 60/73 行正文硬编码
   "28 条登记路径"。本轮是 **16 条路径 / 4 折 / 无空间折 / 无 2025**，逐条都对不上。

**处置**：本轮的四份 `.md`（`README.md`、`short_conclusion.md`、`expert_diagnosis.md`、
`actual_methods_and_deviations.md`）为**手写**，数字逐项核对自 `reports\bridge_arms.json`
与落盘 parquet。这是一处**未被预注册的偏离**，但方向是**增加人工环节**而非放松判据：
`work\audit_arms.py` 仍不 import `campaign_model`、不 import 指标模块。

**仍需修而未修**：`scripts\report_global.py` 本身**未改动**（它属于主线脚本，按"冻结核心不动"
的纪律保留原样）。本轮的事实写在
`reports\actual_methods_and_deviations.md` §6 与 `README.md` 的引用块里。
若下一轮仍要手写报告，应改为新增一个 `scripts\report_bridge.py`，而不是改这一个。

### 7.2 继承的 `README.md` 是一个**实质缺陷**（已修复）

`20260917_5\README.md` 建轮时逐字节继承自 `20260916_1\README.md`（sha `405db389917d8994`）。
它的问题不是"文本陈旧"：

* 它链接三份本轮**并未产出、也产不出**的报告（见 7.1）；
* 它描述的是一个 **28 条路径 / 空间留出含下游禁用闭包 / 2024 正式与 2025 敏感性分别报告**
  的实验 —— 与 `reports\` 的实际内容不符。

⇒ 按该索引去找证据的人会**扑空**。这是路径级缺陷，不是文体问题。

**已修复**：重写为本轮的索引，链接到本轮实际存在的四份 `.md` + 预注册件 + 独立复算件，
并在其中显式保留"本轮不是 `report_global.py` 的产物"这一说明。

### 7.3 对"唯一科学改动 = 水文"的影响：**无**

两条都属于**报告与装配**一侧（`reports\` 与顶层 `README.md`），
不触及拟合目标、模型结构、优化器、观测算子或源输入；也不改变任何已落盘的判据数字。
**"唯一科学改动 = 水文"这句话在本轮仍然成立。**
