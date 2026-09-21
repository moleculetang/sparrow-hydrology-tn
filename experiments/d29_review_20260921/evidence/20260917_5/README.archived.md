# 旧/新水文 × 同一 D29_BE 灰箱的严格桥接

[简短结论](reports/short_conclusion.md) · [专家诊断](reports/expert_diagnosis.md) · [实际方法与偏离](reports/actual_methods_and_deviations.md) · [偏离台账](reports/deviation_ledger.md) · [预注册件](reports/gate0_preflight.json) · [独立复算](reports/bridge_arms.json)

**问题**：把已经校正的水文（`20260917_3`）接到**同一个**成熟的 D29_BE / G1 / OU 灰箱上重跑拟合，
TN **浓度**能不能改善？改进的水文是不是主要瓶颈？

**答案**：第一层 **SMALL**（`ΔNSE = +0.0512`）；第二层 **NO**（五项条件全部不成立）。
两句话同时为真。

## 本轮是什么

桥接实验：**换水文、不动模型**。H0 = 冻结旧水文（`20260828_38`），H1 = 校正水文
（`20260917_3\work\screen\outputs`），两臂**训练标签逐字节相同**，所以目标函数值可直接相减。
外加一个 replay 臂（H1 水文 × H0 参数，只前向、不拟合）分离"水文的直接影响"与"参数补偿"。

* 230 河段、D29_BE / G1 / OU / MATCH、**30 参数**（参数名与边界未变）
* **16 次拟合**（T24 四折 × 2 起点 × 2 臂）+ 8 次前向重放
* 训练 2021–2023 → 评价 **2024**
* **唯一科学改动 = 水文。** 温度 / 源 / 作物逐字节未动
* **全程不计算、不报告、不使用负荷** —— 所有判据落在浓度上
* 没有新增模型结构、四小时物理模型或定时任务

**不在本轮**：空间留出折（S56/S113/S191）、2025（携带 `PET_BRIDGE_UNVALIDATED_2025`）、
正式产品 `20260828_35`。

## 随件事实（读结论前必看）

1. `20260917_3` 自述结论期为**保留的 74 站、2019-01-01..2022-12-31**；本轮评在 **2024**，
   **落在该自述窗口之外**。这是本轮最大的外部效度缺口。
2. 74 站筛选队列与 TN 登记册**几乎不相交**（1 个精确、2 个宽松），且自述为
   `post_hoc_route2_extension`。**头条口径因此改为 T24_G 全观测面板**，
   74 站只作河段级代理注记（读数 `+0.0138`，VERY_SMALL）。
3. 本结论只对 **D29_BE × OU × G1** 这一个结构成立，**不得**外推成"灰箱结构全部无效"。

## 复现与报告入口

```bash
cd /e/SPARROW/5_Test/20260917_5
PY=PYTHONIOENCODING=utf-8 /d/ProgramData/anaconda3/envs/sparrow/python.exe
$PY -B scripts/gate0_preflight.py      # 预检 + 阈值封存
$PY -B scripts/seal_global.py          # 冻结哈希清单
$PY -B scripts/campaign_controller.py  # 拟合（16 条路径）
$PY -B work/audit_arms.py              # 独立复算（不 import 模型与指标模块）
$PY -B work/replay_arm.py              # replay 臂 + 通路负对照
```

所有运行须使用 conda `sparrow`，重复启动与累计预算保护生效。
`scripts` 中带 `hf` 前缀的继承报告不是本轮交付入口。

> **本轮的 `reports\` 不是 `scripts\report_global.py` 的产物。** 该脚本在本轮崩溃（尾步读一个
> 主线专有的 `diagnostics\old_extrapolation\environment_support.csv`），且写死于 28 条路径 /
> 空间留出 / 2025 扩域，无法生成本轮的报告。四份 `.md` 为手写，数字逐项核对自
> `reports\bridge_arms.json` 与落盘 parquet。详见 `reports\actual_methods_and_deviations.md` §6。
