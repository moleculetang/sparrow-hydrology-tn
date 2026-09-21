# FCT8 嵌套与可辨识性：数值证据

本文是 **20260918_1** 轮特有的技术证据件。它只回答一个问题：**"把释放多少与什么时候释放
从同一个 hazard 函数中分离"这件事，在实现层面是否真的被做到了，以及做到之后新增的自由度
是否可辨识。**

全部数字来自 **零拟合** 检查。文件在**任何 FCT8 拟合开始之前**由
`scripts/gate0_preflight.py` 哈希封存（`reports/gate0_preflight.json`，
`preregistration_sha256 = 91de0c9f0bb3098271b87158cac0e827137cb458c992b00ccefdeda64698bdb1`，
`frozen_before_any_fct8_fit = true`，`blocking_failures = []`）。因此本文不允许在跑完之后改口径。

**构造的完整推导与 30→38 参数的逐项定义见 `reports\模型与目标逐项说明.md`**；本文只登记
"构造 → 可验证的数值后果"。

---

## 0. 被验证的构造（唯一实现口径）

$$u_{r,t}(v)=\frac{1}{\sqrt8}\,v^{\top}\phi_{r,t},\qquad
\mu_{r,m}(v)=\frac{\sum_{t\in m}w_{r,t}\,u_{r,t}(v)}{\sum_{t\in m}w_{r,t}}$$

$$b_L=b+\tfrac{\delta}{2},\qquad b_T=b-\tfrac{\delta}{2}$$

$$u^{FCT}_{r,t}=\mu_{r,m}(b_L)+\big[u_{r,t}(b_T)-\mu_{r,m}(b_T)\big]$$

$$A^{FCT}_{r,t}=B\tanh\!\Big(\frac{u^{FCT}_{r,t}}{B}\Big),\qquad
h^{FCT}_{r,t}=h^{preD29}_{r,t}\cdot\exp\!\big(A^{FCT}_{r,t}\big),\qquad B=\ln 10$$

分解发生在 **soft-saturation 之前的 latent score** `u` 上，不在最终乘子上；只有 **一次**
非线性（与 `D29_BE` 同一个 `tanh` 通道）；乘子仍界在 ×[0.1, 10]。
`scripts\fct8_model.py` 实现之，`scripts\campaign_model.py` 的 `make_model()` 增加一条
**纯插入**的 `FCT8` 分派，`D29_BE` 分支源码逐字符不动。

---

## 1. V1 — δ=0 时的代数精确嵌套

判据（计划 §6 F1-1）：预测／数据项／先验项相对差 ≤ 1e-13。基准是**已发布的**
`20260917_5\outputs\T24_G_{D,M}_H1_s0`，本轮 0 次新拟合。

| 量 | `T24_G_D_H1_FCT8` | `T24_G_M_H1_FCT8` | 基准 |
|---|---|---|---|
| `max_abs_latent_gap` | **4.440892098500626e-16** | **4.440892098500626e-16** | — |
| `max_rel_prediction_gap` | **5.916539268525601e-16** | **3.644714465877226e-16** | ≤1e-13 ✔ |
| `max_abs_prediction_gap` | 1.7763568394002505e-15 | 8.881784197001252e-16 | — |
| `abs_data_gap` | **0.0** | **0.0** | 逐位相同 |
| `abs_prior_gap` | **0.0** | **0.0** | 逐位相同 |
| `rel_objective_gap` | **0.0** | **0.0** | 逐位相同 |
| `prior_zero_at_delta0` | true | true | 先验在 δ=0 处为 0 |
| 数据项 `data_d29` = `data_fct8` | 1.1549795517454142 | 1.1344890113941515 | 同一 float64 |
| 总目标 `objective_d29` = `objective_fct8` | 1.2084290708193024 | 1.187270258921654 | 同一 float64 |
| 先验项 `prior_d29` = `prior_fct8` | 0.05344951907388808 | 0.052781247527502406 | 同一 float64 |

三条读数：

1. **嵌套是代数恒等式，不是近似。** latent 差 4.44e-16 就是 float64 的一个 ulp 量级。
2. **嵌套对完整目标成立，不只是数据项。** δ 块的先验是二次、中心 0、形状 `λ‖δ‖²/2`，
   因此在 δ=0 处**先验项本身为 0**（`prior_zero_at_delta0 = true`），于是
   `J_FCT8(b, δ=0) ≡ J_D29_BE(b)` 对 **数据项 + 先验项** 同时成立。这正是"严格嵌套"的
   形式化含义，也是**旧 H1 基线可以合法复用、0 次新拟合**的依据。
3. 两个折独立复现同一结论（D 与 M 的 `data_d29` 不同，是各自的标签不同所致，不是缺陷）。

---

## 2. V2 — 偏差严格一阶，无二次项

判据（F1-2）：`gap_u/|δ|` 在 δ = 1e-2…1e-4 上为常数（三位有效数字）；**不设 δ==0 短路分支**。

| δ | `gap_over_delta`（D 折） |
|---|---|
| 1.0 | 8.754131515778148 |
| 0.1 | 8.754131515778152 |

`relative_drift` = **6.556228836359883e-13**（D）／**3.4468478487661536e-13**（M），
即常数性保持到 13 位有效数字。

**这条为什么重要**：`gap/|δ|` 为常数意味着偏差是 δ 的**严格一次项**。
若有二次项，`gap/|δ|` 会随 δ 线性下降。因此 δ 在 **一阶可辨识** —— 这正是 §7 的
`∂J/∂δ|₀` 诊断能干净成立、且 F2 值得跑的前提。

---

## 3. V3 — 闭式恒等式

判据（F1-3）：`u_FCT(δ) − u(b)` 与 `(1/√8)δᵀ(φ̄_m − φ_t/2)` 的最大相对差 ≤ 1e-13。

| 折 | `max_rel_gap` |
|---|---|
| `T24_G_D_H1_FCT8` | **4.586222378829066e-16** |
| `T24_G_M_H1_FCT8` | **4.646602440326246e-16** |

`fct8_model.py` 的 latent 计算与 `scripts\validate_fct8.py` 中 **独立转写**的闭式字面式
（`latent_literal` / `latent_closed`，不经模型对象）一致到 1e-16。`verify_fct8_weight.py`
的独立复核另见 §8。

---

## 4. V4 — 步长收敛的梯度检查（含一处**必须登记的归一化差异**）

判据（F1-4）：**38 列**中心差分 Jacobian 全通过；含 `delta_0..7` 八列。

| 步长 h | `max_abs_gap` | `max_rel_gap`（FCT8 38 参） | `max_rel_gap`（D29 30 参） |
|---|---|---|---|
| 1e-3 | 1.0321021431050773e-05 | 2.3092270081869915e-03 | 1.0090 |
| 3e-4 | 2.911785714491063e-06 | 6.5148e-04 | 1.1055 |
| 1e-4 | 1.0355639438762103e-06 | 2.3170e-04 | 1.3668 |
| 3e-5 | 9.8814e-09 | 2.2109e-06 | 3.4649e-02 |
| 1e-5 | 3.183043984325286e-09 | 7.121749835771801e-07 | 1.1420e-02 |

`gap_shrinks_with_step = true`，两个折均通过。

### 4.1 一处归一化差异（**随件登记，不得当成物理发现**）

`work\fct8_identifiability.py:87` 把相对差定义为

```python
max_rel_gap = d.max() / max(float(np.abs(fd).max()), 1e-300)
```

即分子是"解析减中心差分的最大绝对差"，**分母是该模型自己所有参数上 |FD 梯度| 的最大值**。
这是一个**每模型自归一化**，不是逐参数归一化。后果：

* 两臂的 `max_abs_gap` 序列**逐位相同**（1e-3 起：1.0321e-05、2.9118e-06、1.0356e-06、
  3.1830e-09），因为左右两臂共享那 30 个参数，差值由同一批量级参数决定；
* 两臂的 `max_rel_gap` 差约 3 个数量级，**完全由分母不同造成**：
  D29 的最大 |FD 梯度| ≈ 1.023e-05，而 FCT8 的最大 |FD 梯度| ≈ 4.4696e-03
  （恰等于 §7 的 `max_abs_dJ_ddelta_at_zero_scaled`）。

因此 `gradient_verdict.same_magnitude_class = false`（`d29_gap_min` 0.011420159891007912、
`fct8_gap_min` 7.121749835771801e-07）**不是**"FCT8 的梯度检查更准"的证据 ——
绝对精度两臂相同。它真实说明的是另一件事：**在扩展后的参数化里，最大的梯度方向是 δ 方向**
（4.47e-03），比 D29 最大的方向（1.02e-05）大 437 倍。这与 §7 一致：`b` 已收敛到 1e-7 量级，
于是"最未收敛的方向"必然是 δ。

`scripts\gate0_preflight.py` 的 G0-1 已把这一项**并列登记**而非覆盖，原话为
"the single-fixed-step V4 reading is recorded alongside so the two procedures can be compared,
not so one can override the other"。本轮不改这个归一化，因为它不改变任何晋级判据
（F1-4 的判据是"38 列全通过"，`pass = true`），而改它需要重算两次 38 列 Jacobian 并
使 gate0 已封存的哈希失效。

---

## 5. V5 — δ 块的可辨识性

判据（F1-5）：38 参数 Jacobian 最小奇异值；`corr(δ_j, b_k)` 矩阵；与 30 参数基线的条件数对比。

| 量 | FCT8（38 参） | D29_BE（30 参） | 变化 |
|---|---|---|---|
| `min_singular_value` | **0.26230487469227104** | 0.2649611059095577 | **−1.00%** |
| `condition_number` | 4034.7482433907066 | 3993.8543988549995 | +1.02% |
| 最大奇异值 | 1058.33413239746 | 1058.2160783623724 | +0.011% |
| `rank_deficient` | false | false | — |
| `n_observations` | 9599 | 9599 | 同 |

**增加 8 个参数只把最小奇异值压低 1.0%**，条件数几乎不动。δ 块不是在一个已经病态的
方向上加噪声，而是**加进了真正的新方向**：δ 行与 8 个 dynamic 行的主夹角
`largest_principal_angle_deg` = **12.917028926668218**（其余到 82.77°），即至少有 12.9° 的
非共线分量。

### 5.1 但**单个** δ_j 无法与它的 b_j 分离（必须这样写）

| 量 | 读数 |
|---|---|
| `max_abs_pairwise_corr_delta_to_dynamic`（`fct8_identifiability.json`） | **0.9372421233700994** |
| `max_abs_corr_delta_to_dynamic`（V4V5 自身读数） | **1.0000000000000004** |
| `max_abs_corr_dynamic_within` | 0.9748047423986391 |
| `max_abs_corr_beta_b_to_dynamic` | 0.634137752775291 |

这不是 bug，是构造的**结构性症状**。由 §0 的线性性可解析地写出

$$\frac{\partial u^{fct}}{\partial b_j}=\frac{1}{\sqrt8}\phi_j,\qquad
\frac{\partial u^{fct}}{\partial \delta_j}=\frac{1}{2\sqrt8}\big(2\bar\phi_j-\phi_j\big)$$

而 `phi_bar_std / basis_std` = **0.9011437121139146**（D）／**0.9002937957855877**（M），
即 `φ̄_j ≈ φ_j`。于是两行几乎平行 ⇒ `corr(δ_j, b_j) → 1`。

**因此本轮只做块水平断言，不做逐参数断言。** 任何"δ_3 单独解释了什么"的说法都不成立。
两个读数（0.9372 与 1.0000）来自不同步长／不同坐标缩放的 Jacobian，**并列登记，不取其一**。

---

## 6. 必须随件登记的 2:1 杠杆不对称

把 §0 的公式完全展开，无交叉项、无近似：

$$u^{FCT}_{r,t}=u_{r,t}(b)+\underbrace{\frac{1}{\sqrt8}\delta^{\top}\bar\phi_{r,m}}_{\text{月水平：月内常数}}
+\underbrace{\Big(-\frac{1}{2\sqrt8}\Big)\delta^{\top}\big(\phi_{r,t}-\bar\phi_{r,m}\big)}_{\text{月内时序：}w\text{-零均值}}$$

水平方向系数 `+(1/√8)δᵀφ̄`，时序方向系数 `−(1/(2√8))δᵀ(φ−φ̄)`。因 `μ` 在 `u^FCT` 的
定义里出现**两次**（`b_L` 一次、`b_T` 一次），**水平的有效杠杆是时序的 2 倍**。

后果：拟合出的 `|δ|` **不能**按 1:1 解读为"水平/时序冲突有多大"，时序那一侧必须除以 2。
本轮**保留**该不对称 —— 它是 ±δ/2 对称位移的必然推论；换成对称效应需要 `b_L = b`，
那会退化成纯时序模型，不是本轮要拆的东西。

---

## 7. δ=0 处的可复用梯度预判（**跑之前**的读数）

在 δ=0 处 `u_FCT` 对 δ 仿射，可解析给出第一步梯度。
记 `G_{r,t} := ∂J/∂u_{r,t}`，`L := (1/√8)Σ_r φ̄_r·(Σ_{t∈m}G_{r,t})`（月水平梯度），则

$$\frac{\partial J}{\partial \delta}\Big|_{\delta=0}=L-\tfrac12\frac{\partial J}{\partial b}\Big|_{\delta=0}$$

| 量（**求解器缩放坐标** `z = x/scale`） | 读数 |
|---|---|
| `norm_dJ_ddelta_at_zero_scaled` | **6.713710956395511e-03** |
| `norm_dJ_db_at_zero_scaled` | 1.2834333050575615e-07 |
| `ratio_norm_delta_over_norm_b` | **52310.55583441014** |
| `max_abs_dJ_ddelta_at_zero_scaled` | **4.469468979620806e-03** |
| `max_abs_dJ_db_at_zero_scaled` | 9.162802213692235e-08 |
| `solver_gradient_noise_floor` | 1e-05 |
| `delta_has_first_order_room` | **true** |
| `precheck_verdict` | **OUTSIDE_FLOOR_DELTA_HAS_FIRST_ORDER_ROOM** |

`max|∂J/∂δ|₀| = 4.469e-03` 是求解器噪声底 1e-5 的 **447 倍**，故 δ 有真实一阶空间。

**关键读数（写进 G0-5 并随件登记）**：因为 `∂J/∂δ|₀ = L − ½∂J/∂b|₀` 而
`∂J/∂b|₀ ≈ 1.28e-07`，第二项可忽略，**这个残余梯度就是月水平梯度 `L` 本身**。
所以可用的**一阶**增益是 **月水平**方向的增益，不是月内方向的增益。

已发布的 H1-D29_BE 最优点上 `reference_pg = 3.900272409484984e-07`
（`work\fct8_prepare.json`；`g0_5` 用的噪声底声明为 1e-5），即旧模型在**已有**参数方向
上已收敛。两者并置的含义是：**F2 检验的是"δ 能否吃到月水平方向的残余梯度"，
而月内时序方向的一阶空间并未被这条读数证明存在。** 这是 F2 结论必须携带的限定。

---

## 8. V6 / V7 / V8 — 下游未动、改动可加、φ̄ 兜底

### 8.1 V6 — M/L 递推、河网路由、水库混合在 δ=0 时逐位相同

| 量 | `T24_G_D_H1_FCT8` | `T24_G_M_H1_FCT8` |
|---|---|---|
| `worst_rel_gap` | **3.689894673226582e-16** | **2.918261561770023e-16** |
| `abs_network_balance_gap` | **0.0** | **0.0** |
| `network_balance_d29` = `network_balance_fct8` | 8.493661880493164e-06 | 3.3974647521972656e-06 |
| `local_balance_d29` = `local_balance_fct8` | 1.3224780559539795e-07 | 1.2578675523400307e-07 |

**这条决定了本轮判读的边界**：FCT8 **不碰** M/L 储库与路由。§9 的 F0 证据表明振幅损失
发生在该通道**下游**，因此 FCT8 在结构上**不可能**修复 `sd_ratio_within`。

### 8.2 V7 — `campaign_model.py` 的改动是可加性的（逐位 replay）

`work\fct8_replay_gate.json`，gate `F1-7`，claim "editing campaign_model.py is additive for D29_BE"：

| 折 | 参数 | 月行 | 日行 | `max_abs_monthly_concentration_gap` | `max_abs_daily_concentration_gap` | `bit_identical` |
|---|---|---|---|---|---|---|
| T24_G_D_H1 | 30 | 5568 | 169476 | **0.0** | **0.0** | **true** |
| T24_G_M_H1 | 30 | 5568 | 169476 | **0.0** | **0.0** | **true** |

耗时 14.46 s。用 **H1 域 + H1 参数**重放，与 `20260917_5` 已发布的 H1 预测**逐位相同**
（4 项共 344 kB 浓度值，gap 0.0e+00）。这一条通过 ⇒ `D29_BE` 通路未被改动
⇒ **旧 H1 基线合法复用，0 次新拟合**。

### 8.3 V8 — `w` 的兜底与 φ̄ 在 δ≠0 时的真实作用

| 量 | `T24_G_D_H1_FCT8` | `T24_G_M_H1_FCT8` |
|---|---|---|
| `weight_all_finite` | true | true |
| `phi_bar_all_finite` | true | true |
| `finite_at_delta_gt0` | true | true |
| `max_abs_gap_when_phi_bar_zeroed_at_delta0` | 1.7763568394002505e-15 | 8.881784197001252e-16 |
| `rel_gap_when_phi_bar_zeroed_at_delta0` | 2.021840119459278e-16 | 1.0936792960907815e-16 |
| `max_abs_prediction_change_at_delta_gt0` | **12.440722401476757** | **6.968240104154591** |

读法：δ=0 时把 φ̄ 置零**不改变任何预测**（1.8e-15，float64 噪声），因为
`u_FCT(δ=0) = μ(b) + u(b) − μ(b) ≡ u(b)` 对**任意** `w` 恒成立 ⇒ `w` 的选择
**不可能破坏嵌套**。而 δ>0 时置零 φ̄ 会使预测改变 12.44 —— 证明 φ̄ 在 δ≠0 处**确实承载**
月水平方向。这两条并置，说明 §8.4 的 `w` 争议只影响 δ≠0 的行为。

### 8.4 `w` 的裁定与独立复核

用户裁定：`w_{r,t} = h^{preD29}_{r,t}(θ̂_ref)`，即**进入 8 项动态 soft-saturation 之前的
base hazard**，在 H1-D29_BE 最优点上取值后整轮冻结。**不得**使用 `h^{D29} = h^{preD29}e^{A_t}`
—— 正在被分解的就是 `A_t`，权重里再含 `e^{A_t}` 等于用旧动态形状给自己二次加权。

`work\verify_fct8_weight.py`（**独立路线**：逐月布尔掩码，不用 `reduceat`）复核：

| 量 | `T24_G_D_H1_FCT8` | `T24_G_M_H1_FCT8` |
|---|---|---|
| 重算 φ̄ 的最大相对差 | **4.7396e-16** | **4.7528e-16** |
| 定义性质 `Σ_t w(φ̄−φ)` 最大残余 | **7.4261e-15** | **1.3524e-14** |
| 均匀兜底单元最大差 | 6.0375e-17 | 6.0375e-17 |
| 零权重质量单元数 | 58 | 58 |
| `mid` 递减游程数 | 0 | 0 |
| 游程边界 == `starts` | true | true |
| 游程 id == `arange` | true | true |
| 状态 | **PASS_WEIGHT_VERIFICATION** | 同 |

**为什么需要这条独立复核**：V1/V2/V3/V6/V8 全部从模型对象读 `phi_bar`/`mid`/`starts`，
因此**没有一个能看见这三者自身的缺陷**；且 δ=0 时 φ̄ 根本不进入任何预测。
`build_phi_bar` 用 `np.add.reduceat(weight, starts)`，在月轴非连续或未按 `mid` 排序时
会**静默给出错误答案**（无异常、无形状错误），而 `phi_bar[mid]` 仍自洽 ⇒ 代数恒等式查不出。
上表证明该前提成立。

兜底数值（`work\fct8_prepare.json`，D 折）：`cells_with_zero_weight_mass = 58`、
`reaches_with_a_zero_mass_month = 33`、`min_positive_monthly_weight_mass = 3.001251710630605e-05`、
`orthogonality_residual = 9.997076899785562e-16`、`max_abs_phi_bar = 2.3424449294574448`。
另需分清两个不同量：**单元级**零质量单元 58 / 176640 = **0.033%**，而
**条目级** `weight_fraction_exactly_zero` = **0.149607549921138（14.96%）**。
前者是"整月整河段无 contact"，后者是"该月内部分日 contact 为 0"，两者不可混用。

---

## 9. 汇总：本文每个数字的出处

| 数值 | 出处文件 |
|---|---|
| V1/V2/V3/V6/V8 全部、V4V5 | `reports\fct8_validation.json`（`status = PASS_F1`，`n_pass = 11/11`） |
| V4 两臂步长扫描、V5 主夹角、δ–dynamic 相关 | `reports\fct8_identifiability.json` |
| δ=0 梯度预判 | `reports\fct8_identifiability.json` →`zero_fit_precheck`；G0-5 |
| V7 逐位 replay | `work\fct8_replay_gate.json` |
| `w`/φ̄ 独立复核 | `work\fct8_weight_verification.json` |
| 零权重质量、`train_sha256`、`reference_pg` | `work\fct8_prepare.json` |
| 封存与预注册哈希 | `reports\gate0_preflight.json` |
| 门禁逐块判定 | `reports\gate0_preflight.json` → `G0-1`…`G0-10` |

**`n_gates = 11` 是两个折之和**（D 折 6 块：V1/V2/V3/V4V5/V6/V8；M 折 5 块：V1/V2/V3/V6/V8，
M 折不单列 V4V5）。V7 与 `w` 复核另行登记于各自文件，不在这 11 之内。

---

## 10. 本文**不**主张什么

* 不主张 δ 的**逐参数**含义可辨识（§5.1：`corr(δ_j,b_j) → 1`）。
* 不主张 FCT8 能修复月内振幅（§8.1：V6 证明下游逐位未动；F0 证明损失在下游）。
* 不主张 `same_magnitude_class = false` 说明 FCT8 的梯度检查更准（§4.1：归一化差异）。
* 不主张 F2 的收益是月内方向的（§7：可用的**一阶**空间是月水平方向）。
* 不主张任何**负荷**量。全程不计算、不报告、不使用负荷；本文全部判据落在浓度与目标函数上。
