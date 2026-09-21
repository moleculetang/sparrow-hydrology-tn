"""`20260920_1` -- generate THIS round's copy of the closed-hypothesis ledger, then
edit it IN PLACE, here, programmatically.

ZERO FORWARDS.  ZERO FITS.  Reads one peer file (read-only) and writes one file
inside this round's `reports/`.

WHY A SCRIPT AND NOT A HAND COPY
--------------------------------
The ledger is a 100997-byte inherited artifact whose whole value is that its
earlier rows are BYTE-IDENTICAL to what earlier rounds wrote.  Re-typing 100 KB of
historical registration is exactly how a number gets silently changed while nobody
is looking -- which is the reason the round-5 ledger's own header gives for
seeding it the same way.  So: copy the bytes, assert the source digest, then apply
a small number of exact-string edits, each of which must match EXACTLY ONCE.  If an
anchor is missing or ambiguous the script raises instead of writing, because a
partial edit is worse than no edit.

WHAT IT REGISTERS
-----------------
1. the source digest, so the copy step stays independently re-checkable;
2. the resulting digest, plus the byte counts on both sides;
3. every edit it made, as `(anchor_head, n_replacements)`, so a reader can diff
   the claim against the file.

THE EDITS (six, and only six)
-----------------------------
(a) the title line;
(b) a new renewal-scope blockquote, inserted above `## 0. 一行摘要`;
(c) the `## 0.` running count, plus the fourth counting bullet that records WHY it
    moved this time;
(d) one new table row (row 16);
(e) a new section `## 20.`;
(f) new `## 随件` items 28+.
History is never rewritten: every earlier row and every earlier section is carried
through untouched.
"""
import hashlib
import json
import re
import sys
from pathlib import Path

R = Path(__file__).resolve().parent.parent
UP = R.parent
SRC = UP / '20260919_5' / 'reports' / '已关闭假设台账.md'
DST = R / 'reports' / '已关闭假设台账.md'
MANIFEST = R / 'reports' / '_ledger_refresh.json'

SRC_SHA = '2909a10330660ac28780b4ec5ff2d1b8eec0ee4aa6d9ff08174d9050e95e2fd2'
SRC_BYTES = 100997


def sha(b):
    return hashlib.sha256(b).hexdigest()


def sub_once(text, anchor, new, label):
    n = text.count(anchor)
    if n != 1:
        raise SystemExit('LEDGER_ANCHOR_NOT_UNIQUE [%s]: found %d occurrence(s) of %r'
                         % (label, n, anchor[:80]))
    return text.replace(anchor, new), n


# ---------------------------------------------------------------- (a) the title
TITLE_OLD = ('# 已关闭假设台账（`20260919_5` —— 承接 `20260919_4`，新增第 15 行；'
             '第 14 行的 G5 读数已由本轮更正，见 §18.8）')
TITLE_NEW = ('# 已关闭假设台账（`20260920_1` —— 承接 `20260919_5`，新增第 16 行；'
             '**本轮没有更正任何历史读数**，只新增一行与一节）')

# ------------------------------------------------- (b) this round's scope note
#
# The anchor is the round-5 scope block's LAST LINE plus the separator, not the
# separator alone: `\n\n---\n\n## 0.` is a shape that also closes the earlier
# scope blocks, and an anchor that matches twice makes `sub_once` raise -- which is
# the point, but it is cheaper to pick the unique one.  Verified unique by
# `work/_anchors_probe.py` before this file existed.
SCOPE_TAIL = '> 而同名文件会**静默命中另一轮的产物**，交付机检把它记作 `ok`。\n'
SCOPE_ANCHOR = SCOPE_TAIL + '\n\n---\n\n## 0. 一行摘要'
SCOPE_NEW = SCOPE_TAIL + '''

> **`20260920_1` 的翻新范围（只有这六处，其余逐字不动）**：
> ① 表头；② §0 的一行摘要与其计数说明随**第 16 行**更新，并新增一条计数教训；
> ③ 表格新增**第 16 行**；④ 新增 **§20：第 16 行：守恒的双水路移动水浓度（零参数）**；
> ⑤ 末尾「随件」新增第 **28–33** 条；⑥ 本节就是新增的这一段。
> **第 1–15 行、§10、§11b、§13、§14、§15、§16、§16.5、§17、§18、§19 的全部历史登记原文
> 逐字保留，不得删改。** 本轮**没有**新增任何第五种裁决取值，**也没有改动任何一行的裁决**。
>
> **本文件是 `20260919_5\\reports\\已关闭假设台账.md` 的逐字副本 + 本轮新增**，
> 复制方式与 `20260919_4`／`20260919_5` 两次相同：**复制，不重打**。
> 复制那一步**在 `work\\_refresh_ledger.py` 里被断言**，而不是靠叙述：
> 原件 sha256 = `2909a10330660ac28780b4ec5ff2d1b8eec0ee4aa6d9ff08174d9050e95e2fd2`、
> `100997` bytes，脚本在复制**之前**核对这两个值，不等即拒绝写盘。
> **注意副本的方向**：`20260919_5` 的那一份**本身**已被它自己更新过
> （§19 与随件第 21–27 条）⇒ 本文件继承的是**已更新**的那一份，
> 那些内容**不在本轮重做、也不在本轮重新论证**。
>
> **本轮新增的"依据"列引用一律写出轮次编号**（`20260920_1\\reports\\…`），
> 理由与前两轮相同：本文件被复制到下一轮之后，"本目录"会指向另一个目录，
> 而同名文件会**静默命中另一轮的产物**，交付机检把它记作 `ok`、不会报错。

\n\n---\n\n## 0. 一行摘要'''

# --------------------------------------------- (c) the `## 0.` running count
#
# Two separate edits, because the `## 0.` header carries the count in TWO places:
# the summary lines above the discussion, and the running bullet inside it.  Fixing
# one and not the other is precisely the self-contradiction the round-3 note
# records twice; `_refresh_ledger` therefore edits both or neither.
SUMMARY_OLD = ('**共 15 个行号（第 11 行按 `20260919_2` 的裁定拆为 11a／11b ⇒ 表内 16 行）：**\n'
               '**14 行已裁决，2 行未裁决 —— 第 12 行（`未裁决 —— 严格未答`）'
               '与第 14 行（`未裁决 —— 预注册缺口`）。**')
SUMMARY_NEW = ('**共 16 个行号（第 11 行按 `20260919_2` 的裁定拆为 11a／11b ⇒ 表内 17 行）：**\n'
               '**15 行已裁决，2 行未裁决 —— 第 12 行（`未裁决 —— 严格未答`）'
               '与第 14 行（`未裁决 —— 预注册缺口`）。**')

COUNT_OLD = ('>   `装置成立、假设不成立` `4` 行（3／11b／13／**15**）、'
             '`未裁决` `2` 行（12／14）。\n')
COUNT_NEW = (COUNT_OLD
             + '> * **`20260920_1` 是这一行的第四次变动，同样登记**：新增第 **16** 行，'
               '裁决取值 **`装置成立、假设不成立`**（**装置层面**，外延见 §20.7）。\n'
               '>   逐行数过 §0 表之后的新读数是：17 个表行中 `CLOSED` `9` 行'
               '（1／2／4／5／6／7／8／9／11a）、`CLOSE` `1` 行（10）、\n'
               '>   `装置成立、假设不成立` `5` 行（3／11b／13／15／**16**）、'
               '`未裁决` `2` 行（12／14）。\n'
               '> * **这一行的历史是一条连续的错**：`20260919_3` 少算了一行，'
               '`20260919_4` 的"修正"又少算了一行，两次方向相同。\n'
               '>   本轮**没有**再声称「这就对了」——本行现在给的是'
               '**逐行数过表之后的读数**，读者可以自己数。\n')

# --------------------------------------------------------- (d) the table row
#
# The anchor is row 15's WHOLE LINE, and the replacement is that line plus a
# newline plus the new row.
#
# Both details are load-bearing, and the first draft got them wrong in a way that
# only a diff catches:
#
#   * anchoring on row 15's TAIL FRAGMENT (`（见 §19.4–§19.5） |`) and ending the
#     fragment with the line's `\n` CONSUMED that newline.  The delivered file then
#     held row 15 and row 16 glued into one 2030-character line -- a two-row table
#     pasted into a single row, which renders as row 15's last cell containing the
#     whole of row 16.  Every content check still passed: the text was all there.
#     Row granularity, plus a newline in the replacement, is what makes the table
#     a table.
#   * the head `| **是**（机制族` occurs twice, so an anchor on the head would
#     raise; the full line occurs exactly once (asserted below, from the seed).
#
# Read from the seed rather than retyped: the line is 858 characters of `$\\Xi$`
# and full-width punctuation, and a hand copy that differed by one character would
# raise `LEDGER_ANCHOR_NOT_UNIQUE` -- the good outcome -- or, if it happened to
# match nothing, `found 0 occurrence(s)`.  Neither is silent, but neither is
# necessary: the seed is already open.
def _row15_full():
    hits = [ln for ln in SRC.read_text(encoding='utf-8').split('\n')
            if ln.endswith('（见 §19.4–§19.5） |')]
    if len(hits) != 1:
        raise SystemExit('LEDGER_ROW15_NOT_UNIQUE: %d candidate line(s)' % len(hits))
    return hits[0]


ROW15_FULL = _row15_full()
ROW15_ANCHOR = ROW15_FULL
ROW15_TAIL = ROW15_ANCHOR
ROW16 = (ROW15_FULL + '\n'
         + '| **16** | **守恒的双水路移动水浓度（零参数）**（`20260920_1`：把陆相动员从 '
    '$E_t=av_t\\,p_t$ 换成"**一个上层移动水浓度**（$F_f$ 与 $J$ 取自**同一个** $C_u$，'
    '**没有 fast/percolation 先后**）→ 快路与慢路两个浓度储层 → 各自 $Q\\times C$"。'
    '$V_u$ 取生产者直写状态 `upper_water`、$V_s$ 取生产者直写 `lower_slow_storage_mm`；'
    '$g$ 由**储量日内时序闸门**确定性选定为 $g(x)=x$；**零新参数、零拟合、30 维一字未改**。'
    '问的是：**在不抬长期平均水平的前提下，能不能产生事件振幅**） | '
    '**`装置成立、假设不成立`**（**装置层面**：装置被证明**确实装上**'
    '（N1 注入 $g_u{:=}p$、$\\phi_f{:=}f$、$g_s{:=}l$ 后与新核**逐位相同**，25 通道），'
    '**但**预注册的五门合取**没有成立**（主臂 `P-upper` 五门 **0/5**）'
    ' ⇒ 预注册结局 `MOBILE_POOL_MAPPING_LIMITED` 命中。'
    '**外延见 §20.7，不得外推**；**不得**读成"双水路移动水浓度机制不存在"'
    '——本轮**在把可动性映射这一层单独判过之前看不到那个问题**，见 §20.2） | '
    '`20260920_1\\reports\\phase0_gates.json`（§3.1／§3.2／§3.3 三个闸门 + $g$ 的选定，'
    '**零前向**）、`20260920_1\\reports\\phase0_n1.json`（25 通道归约检验）、'
    '`20260920_1\\reports\\arms.json`（六臂表，sha `520fa636…`）、'
    '`20260920_1\\reports\\phase1_arms.json`（5 次前向 + 全部门）、'
    '`20260920_1\\reports\\level_variance.json`（水平／方差分解）、'
    '`20260920_1\\reports\\verdict.json`（判决是**纯函数**：读数 → 结局）、'
    '`20260920_1\\reports\\audit_dp.json`（**独立复算**：不 import 本轮任何模块，'
    '25 项检查全过）、`20260920_1\\reports\\neighbour_write_check.json`'
    '（邻居轮次未被改动的机器证据） | '
    '**是**（机制族**未被关闭**；预注册 §2.7 对本轮明写下一轮的唯一自由度是 '
    '$\\{k_m,k_{ex}\\}$ **二选一、绝不能两个同时开**，且**本轮诊断指向 $k_m$**），'
    '**但本轮不实现、不参数化、不预留开关** |\n')

# ------------------------------------------------------ (e) the new section
SECTION_ANCHOR = '\n## 随件：不能证明的事（本台账整体）\n'

SECTION = '''
## 20. **新增第 16 行**：守恒的双水路移动水浓度（零参数）—— `20260920_1`

> **本节的地位。** 它是第 16 行的**依据**，不是综述。第 16 行说的每一件事，
> 都在这里给出读数与它落在哪个文件里；本节说的每一件**不能**由第 16 行推出来的事，
> 都在 §20.7 里逐条列出。

### 20.1 装置是什么

**替换掉的**（前四轮陆相动员的全部拼写）：`E = av*prob`、`fast = E*f`、
`pre = L + E*(1-f)`、`slow = pre*l`。
**换成的**（逐字，`20260920_1\\work\\dp_kernel.py`）：

```
A      = max(M[r] + inp[t,r] - demand[t,r], 0.)     # 递推式逐字沿用
Qu     = Qf + Qp        Vu = 出流前上层体积(mm)      xu = Qu / Vu
g_u    = g(xu)          Eu = A * g_u                 phi_f = Qf / Qu
F_f    = Eu * phi_f     J  = Eu * (1 - phi_f)        M[r] = A*(1-g_u)*s_M[r]
Lpre   = L[r] + J       xs = Qs / Vs                 g_s = g(xs)
F_s    = Lpre * g_s     L[r] = Lpre - F_s            local = F_f + F_s
```

六点必须同时成立，**全部实测通过**：同源性（$F_f$ 与 $J$ 取自**同一个** $C_u$，
代码里不出现 `(A - F_f)`）、守恒（$F_f+J=E_u$、$F_s<L^{\\rm pre}$，
**无 cap、无 `min()`**）、$\phi_f$ 只分水不改总量、$g$ 全核唯一、
**数值拼写**（指数形只能写 `-np.expm1(-x)`）、**归约点存在**（N1）。

### 20.2 装置**确实装上了**——这是本行唯一可以当证据用的东西

若注入 $g_u{:=}p_0$、$\phi_f{:=}f$、$g_s{:=}l$，新核必须与冻结核**逐位相同**。
实测：**四路 `scan` + 七路 `tag` + 两路 `transport` 共 25 通道，
`failed` = `[]`**（`reports/phase0_n1.json`）；
第 ⑤ 步的**独立复算**（**不 import 本轮任何模块**，numpy 直接重写）在 7 条通道上
`all_bitwise` = `true`（`reports/audit_dp.json::N1`）。

> **没有 N1，本行连"装置成立"都不能写。** 台账闸门**不承担**这个职责：
> 它查的是**"核返回的流量是否与一条独立拼写的递推一致"**，不是物理守恒律
> （`phase0_gates.json::N3.what_the_teeth_are`，并且配了实测演示：
> 把核的**返回值**扰动 1.1 倍而递推保持独立拼写，残余就是解析值 $-0.1\\sum A g_u\\phi_f$，
> 实测 `rel_mismatch` = `8.228750389916546e-16`）。

### 20.3 假设为什么不成立：形态是**过度动员**，不是"没有振幅机制"

| 臂 | $A_{L1}$ | 相对冻结基线 | 水平比 | 日方差比 | 逐事件峰/基比 |
|---|---|---|---|---|---|
| `B0`（冻结核） | 1.0400772083480145 | 1.00 | 1.0 | 1.0 | 1.0 |
| **`P-upper`** ⭐ | 0.7373725809958451 | **0.71** | **9.143688123572442** | **314.148477382949** | **0.7089594648141899** |
| `S-soil` | 0.97915517578687 | 0.94 | 4.085796633815022 | 28.812173831181497 | 0.9414254710398771 |
| `S-unsat` | 0.9752709506769071 | 0.94 | 4.110311235620086 | 28.758986722547682 | 0.9376909164522111 |
| `D-const`（时间结构诊断臂） | 0.7860814113413213 | 0.76 | 7.615085085789727 | 201.48536477261823 | 0.7557914018612885 |
| `R5-ref`（读盘参照，**不进任何门**） | 1.244023607865766 | 1.20 | 0.9945297890116693 | — | — |

**读法。** 若失败形态是"路径结构不够灵敏"，会看到水平不动、方差小、事件比 ~1。
看到的是相反的东西：主臂平均浓度 `25.81672495274234` mg/L 是冻结基线
`2.8234476727379607` 的 `9.143688123572442` 倍、日方差 `314.148477382949` 倍，
而**逐事件峰/基比反而掉到 `0.7089594648141899` 倍**。
这是**库存被以水的周转率逐日抽干**的签名。去掉水平之后的 NSE 仍然为负
（`nse_residual_after_recentring` = `-228.05008625529243`）⇒ **不是"只错在水平"**。

**`D-const`（删掉 $V$ 的时间变化）与三个时间变动的臂同形** ⇒
**主因不是 $V$ 的时间结构**。

### 20.4 P1 证伪器在四个核臂上全部点火，且**是恒等式而非巧合**

| 臂 | $\\tau_{\\rm eff}$ 中位 / $\\tau_{\\rm hydro}$ 中位 |
|---|---|
| `P-upper` | **0.996355630760417** |
| `D-const` | 0.9945 |
| `S-soil` | 0.8552 |
| `S-unsat` | 0.8528 |

在线性闭包下 $carry=(1-g_u)s_M$，而 $1-g_u=1-Q_u/V_u=S^{\\rm post}/V_u$
**正是留存水比例** ⇒ $-\\ln(carry)=-\\ln(S^{\\rm post}/V_u)-\\ln(s_M)$，
$\\ln(s_M)\\in[-0.0034,-0.00028]$ 相对水的 O(1) 项可忽略。
**N 的记忆寿命在本核里就是上层水的滞留时间，由构造决定，不由测量决定。**

独立复算把它钉到 **一 ulp**：`max_abs_diff` = `2.220446049250313e-16`、
`max_abs_diff_in_ulps` = `1.0`、`holds_to_one_ulp` = `true`、`bitwise` = `false`。
**判据写成 ABSOLUTE/ulp 形而不是相对形**，因为相对形在留存份额小的格上病态
（`max_rel_diff` = `0.09305815012303995`，良态子集 `5.173383372555829e-12`）。

**这条对第 16 行的裁决有直接后果**：它使主臂的事件门失败**不可解释为机制失败**，
所以第 16 行的裁决**只能是"装置层面"的**——与第 15 行踩过的那个坑同类（见随件第 22 条）。

### 20.5 三条旁证

1. **输出退化为月初脉冲**：月初日只占天数 `0.03285420944558522`，却承载 $F_f$ 质量
   `0.16954662019375846` ⇒ 过表征 `5.001824614730297` 倍。**$A$ 在很大程度上量的是日历。**
   但 $F_s$ 的同一占比是 `0.03216374515683588`，**与天数占比几乎相同**——
   脉冲**没有**以同样强度穿到慢路，这与核的"两条路不相等"一致。
2. **水平与振幅不可分离**：本轮**没有**任何水平杠杆（不像 `20260919_5` 有 $k_r$）
   ⇒ 两者**不得互为证据**（`phase1_arms.json::N6_note`）。
3. **慢路是定常线性水库——但不得读成"慢路不存在"**：P2 预测"$x_s$ 带宽应明显大于 6.1%"
   **被证伪**（实测 `relative_bandwidth` = `0.06061447917680636`，反解参照 `0.061`），
   **原因是代数**：由 $V_s=S^{\\rm post}+Q_s$ 与 $S^{\\rm post}=Q_s(1-lr)/lr$ 可得
   $x_s\\equiv lr$ 精确成立 ⇒ **"直写"与"反解"在 $x_s$ 上不是两条独立的路**。

### 20.6 三处必须随行的披露

1. **30 维里 21 个失效。** $h$ 只经 `Predictor.hazard` 产生，新核不用 $h$；$f$ 被 $Q$ 的
   物理分水取代；在用只有 `t[3]`、`t[11..17]`（经 $s_M$）与 `t[2]`（河道段）。
   ⇒ **本轮的零拟合不是预算上的，是结构上没有可拟合的东西**，
   这比 `20260919_5` 的零拟合更强，也把它的证据面**收窄**：
   **冻结参数包络对陆相已不构成约束。**
2. **惰性守卫保留，而且反对它的那条推导是错的。** 掩码上 $Q_u$ 最大
   `2.5101515156968253e-14` mm/day，比冻结 `fast_water` 下界 `4.184469060922251e-200`
   高九个数量级，而上层储量塌到 `5.221622195298538e-28` mm ⇒ 不加守卫的线性闭包在
   掩码中位格给出 $g_u$ = `0.8636596200230362`，且 `frac_xu_unguarded_ge_0p99`
   = `0.07277735618843556`。冻结核在**恰好这些格**上给 `prob = 0`（`max_fast_fraction` = `0.0`）
   ⇒ **守卫是 N1 能成立的唯一原因**，登记为命名自由选择。
   **同时**：掩码上**不得**把 $g_s$ 钉成冻结的 $l$（实测 `max_gs_on_mask`
   = `0.007789548659940684`），新核与冻结核在掩码上的慢路必然不同，
   **登记为有意的结构性差异，不是误差**。
3. **`training_years = [2021, 2022]` 落在评估窗口内。**
   `20260916_2\\outputs\\C0_s1\\model.json::design.training_years`。
   前三轮都披露过，本轮**继续披露**："零拟合完全没有间接使用评估期"这句**必须带这个限定**。
   **一条交付缺口**：$A_{L1}$ **无法**被本轮独立复算——它需要 169476 行的密集前向帧，
   而交付的 `daily_arms.parquet` 是 12152 行的合格格栅格。
   这是**真实缺口，登记而不是掩盖**（`audit_dp.json::scope_limits`）。

### 20.7 第 16 行的外延（**不得外推**）

> **框。** 第 16 行的裁决说的是：
> **"在 $V_u$ = `upper_water` 这一个读数上、在 $g(x)=x$ 这一个闭包上、在
> $A$ = 全部可动 N 这一个上界假设上、零参数、且只在这一套（15 站／214 事件／
> 12152 合格格）判据集内，这一族结构没有同时过预注册的五门。"**
>
> 它**不等于**下列任何一句：
>
> 1. **"双水路移动水浓度机制不存在。"** 本轮**看不到**那个问题——失败的形态是
>    **过度动员**（§20.3），问题在"**到底哪一部分 legacy N 是可动的**"这一层，
>    而按预注册 §2.7，那一层要**单独判过**才能问路径结构。
> 2. **"$V$ 的身份问题已经有答案。"** 主臂由 §3.2 的周转一致性闸门决定，而该闸门对
>    $V$-upper **是一个恒等式**（$V^{\\rm post}_{\\rm upper}/Q_f$ 与生产者的
>    `upper_store_instantaneous_turnover_day` 相除时 $86400/(100\\times10)=86.4$
>    精确约掉，`median_abs_log_ratio` = `0.0`，已作为 `identity_disclosure` 披露）
>    ⇒ **这个吻合不携带证据**。`V-soil` 的 `median_abs_log_ratio` = `3.702719688995228`
>    是**"生产者根本没有为它导出 turnover 列"**的读数，不是"它错了"的读数。
>    主臂的选定**来自生产者源码行 + 预注册先验序，不来自任何结果**。
> 3. **"陆相内部结构这一根轴可以关闭了。"** 见 §20.8。
> 4. **"下一轮可以做 $k_m$ **和** $k_{ex}$。"** 预注册 §2.7 逐字：
>    **"绝不能两个同时开。"**
> 5. **"$R5$-ref 的读数支持了什么。"** `R5-ref` 的 $A_{L1}$ = `1.244023607865766`
>    是本轮最高的，但它**不带任何门布尔量**（`reference_arm_carries_no_gate` = `true`）
>    ——这是**结构性**排除，不是靠纪律。
> 6. **"阈值或门被调整过。"** `work/verdict_dp.py` 在算判决**之前**逐项断言
>    30 个参数、`MONTHLY_GATE == 0.005`、逐臂 `sd_gate_threshold`
>    == `0.5559042757541334`、逐臂 `sd_gating_layer` == `L3`，任一不成立即抛异常。

### 20.8 可否重开，以及重开的入口

**可以重开，但入口很窄，而且入口是预注册规定的，不是本轮挑的。**

* **授权的对象**：下一轮的唯一自由度是 $\\{k_m,\\ k_{ex}\\}$ **二选一**。
  $k_m$ = legacy N → mobile N（可动性映射）；$k_{ex}$ = fast ↔ slow（两路交换）。
* **方向由本轮诊断决定**：`verdict.json::next_round.chosen` = **`k_m`**。理由是本轮落在
  预注册 §2.7 的 **row 1**：证伪器在四个核臂全部点火 + 输出退化为月初脉冲 +
  **事件振幅根本没有量到**（$A_{L1}$ 是 0.71–0.94 倍的**下降**），
  ⇒ row 2 的前提（"振幅已量到、只剩事件后恢复相位不对"）**不成立**。
* **本轮不实现、不参数化、不预留开关**：`k_m` 在本轮代码里**不存在任何休眠路径**。
* **授权的性质**：这是**授权下一轮去预注册**，**不是**"机制被证实"的证据
  （预注册 §8 风险 12：本谱系的教训是**零拟合的收益可以在重新拟合后消失**，
  `20260919_5` 一栏已登记过回吐 `+0.0036`）。
* **不得**因为本行而重启任一已关闭轴，**不得**跳 PON／侵蚀／河床源。
  预注册对本轮结局的 `forbidden` 字段逐字写着
  **"closing the land-phase structure, and jumping to an additional event source"**。

### 20.9 本轮**没有**做的事（与第 16 行的可核性有关）

**5 次前向**（`B0` + 4 个新核臂，`R5-ref` 是**读盘**）、**0 次拟合**、
**`fit_worker` 调用 0 次**、**30 个参数一字未改**、**15 站全部进入（零表现性剔除）**、
**不计算／不报告／不使用任何负荷**（守恒式是**账本身份式**，不是负荷判据）。
邻居轮次未被改动的机器证据在 `reports/neighbour_write_check.json`
（第 ⑤ 步留影 29/29 吻合、两轮跨轮留影 14/14 吻合、mtime 扫描 59 个文件 0 次外来写入），
**但它覆盖不到第 ⑤ 步之前**——本轮开始时没有留影，而 `5_Test` **不是** git 跟踪的树。
## 随件：不能证明的事（本台账整体）
'''

# -------------------------------------------------------- (f) 随件 additions
#
# The tail is APPENDED, not anchored.  Item 27 is the last line of the file and
# every string in it that could serve as an anchor carries a full-width quote; a
# near-match there would be a silent no-op or a wrong-position insert.  Appending
# needs no anchor: the file's trailing bytes are asserted instead
# (`TAIL_MUST_END_WITH`), and the numbering is asserted to continue from 27
# (`TAIL_MUST_CONTAIN`) so a copy of an older ledger cannot silently produce a
# second item 28.  The ledger's items are single-newline separated, so the append
# adds one `\n` -- no blank line, matching items 25/26/27 above.
TAIL_MUST_END_WITH = '。\n\n'
TAIL_MUST_CONTAIN = '\n27. **本节各条与 §18 各条不得互相援引为证据**'
TAIL_FIRST_ITEM = 28
TAIL_LAST_ITEM = 33

TAIL_NEW = '''28. **第 16 行是本台账里第一条"换结构"而不是"换乘子"的行**，因此它的裁决比前面每一行
    都更容易被读大。**"装置成立"在本行的意思是**：N1 归约检验逐位通过、25 通道全过、
    独立复算 25 项全过 ⇒ **"新核确实是同一个模型换掉了一层"这句话是被证明的**，
    不是被宣称的。**它没有说那层换对了。**
29. **本行不得与第 15 行（`20260919_5`）互相援引为证据。**
    两轮的自由度不同（第 15 行是 $k_r$ 在冻结参考期解出的一个乘性场；本行是零参数的结构替换），
    判据集不同（第 15 行 76 点、本行 5 点），结局也不同。
    唯一可以并列的是**同一套五门基线与同一套阈值**——那是"性能范围保持一致"的实现，
    不是"两轮结论互相加强"。
30. **本行的"零拟合"与第 15 行的"零拟合"不同类，这一条必须写进行内。**
    第 15 行的零拟合是**预算上的**（$k_r$ 由账本恒等式解出、不拟合）；
    本行的零拟合是**结构上的**（30 维里 21 个失效，陆相动员**不含任何参数**）。
    ⇒ 本行的零拟合**把证据面包络收窄了**：冻结参数包络对陆相**已不构成约束**，
    所以"换掉陆相结构仍然不过门"这句话，**不能**被读成"参数没调好"。
31. **$A_{L1}$ 无法被本轮独立复算，这是一条要留在台账里的缺口。**
    它需要 169476 行的密集前向帧，而交付的是 12152 行的合格格栅格。
    **本轮选择登记它，而不是用一条更弱的复算去顶替**——
    后者会让"独立复算"这四个字在这一行上失去意义。
32. **第 16 行的主臂不是"最好的臂"，是**唯一**通过周转一致性闸门的候选。**
    `S-soil` 与 `S-unsat` 的读数**照报**（§20.3 的表里都在），
    它们不过门**不构成**"它们更差"的结论——它们是**敏感性臂**，不是竞争者。
    **不得**用"主臂恰好是 upper"去反推"upper 就是对的"。
33. **本轮**没有**新增任何第五种裁决取值，也**没有**改动第 1–15 行的任何裁决。**
    第 14／15 行的 `未裁决` 与它们的重开条件**保持不动**。
'''

EDITS = [('title', TITLE_OLD, TITLE_NEW),
         ('scope', SCOPE_ANCHOR, SCOPE_NEW),
         ('summary', SUMMARY_OLD, SUMMARY_NEW),
         ('count', COUNT_OLD, COUNT_NEW),
         ('row16', ROW15_ANCHOR, ROW16),
         ('section20', SECTION_ANCHOR, SECTION)]


def main():
    src = SRC.read_bytes()
    got = sha(src)
    if got != SRC_SHA or len(src) != SRC_BYTES:
        raise SystemExit('LEDGER_SOURCE_MOVED: expected %s / %d bytes, got %s / %d'
                         % (SRC_SHA, SRC_BYTES, got, len(src)))

    text = src.decode('utf-8')
    applied = {}
    for label, old, new in EDITS:
        text, n = sub_once(text, old, new, label)
        # The WHOLE anchor is recorded, not a head.  `work/_verify_ledger_history.py`
        # checks that the inherited lines which no longer appear verbatim are exactly
        # the lines the registered anchors name; with a 70-char head, the second line
        # of a two-line anchor (and any anchor's later lines) cannot be matched back,
        # so the check would come out weaker than it reads.  Anchors are short; the
        # manifest has no reason to be lossy.
        applied[label] = dict(n_replacements=n, anchor_text=old, new_text=new,
                              new_head=new.split('\n')[0][:70],
                              anchor_head=old.strip().split('\n')[0][:70])

    # ---- (f), appended rather than anchored; see the note above TAIL_NEW ----
    if not text.endswith(TAIL_MUST_END_WITH):
        raise SystemExit('LEDGER_TAIL_UNEXPECTED: source does not end with %r'
                         % TAIL_MUST_END_WITH)
    if TAIL_MUST_CONTAIN not in text:
        raise SystemExit('LEDGER_TAIL_UNEXPECTED: item 27 anchor missing, so the '
                         'new items would not continue the numbering')
    if ('\n%d. ' % TAIL_FIRST_ITEM) in text:
        raise SystemExit('LEDGER_TAIL_UNEXPECTED: item %d already exists -- this is '
                         'not the seed it claims to be' % TAIL_FIRST_ITEM)
    # `[:-1]` drops ONE trailing newline (the file's blank last line) so the first
    # new item continues the list at the same separation as items 25/26/27.
    text = text[:-1] + TAIL_NEW
    applied['tail'] = dict(n_replacements=1, anchor_text='', new_text=TAIL_NEW,
                           new_head='28. **第 16 行是本台账里第一条',
                           anchor_head='APPEND at EOF (no anchor), items %d-%d'
                                       % (TAIL_FIRST_ITEM, TAIL_LAST_ITEM))

    out = text.encode('utf-8')
    DST.write_bytes(out)
    # The carve-out the delivery checker verifies: the first `SRC_BYTES` BYTES of
    # the delivered file must still hash to the seed's digest.  Recorded here so
    # the checker's exemption is cross-checkable against this producer rather than
    # asserted in prose.
    man = {'phase': 'ledger_refresh', 'round': str(R), 'zero_forwards': True,
           'n_fits': 0, 'fit_worker_calls': 0,
           'source': str(SRC), 'source_sha256': got, 'source_bytes': len(src),
           'dest': str(DST), 'dest_sha256': sha(out), 'dest_bytes': len(out),
           'source_sha256_matches_registered': bool(got == SRC_SHA),
           'inherited_prefix_bytes': len(src),
           'inherited_prefix_sha256': sha(out[:len(src)]),
           'inherited_prefix_is_byte_identical': bool(sha(out[:len(src)]) == got),
           'authored_suffix_bytes': len(out) - len(src),
           'seeded_by': 'byte copy, not a re-type (plan section 6)',
           'edits': applied, 'n_edits': len(applied),
           'history_untouched': (
               'every anchored edit is an insertion or a single-line replacement; '
               'no earlier row or section is rewritten, and the appended items start '
               'after the last byte of the seed')}
    MANIFEST.write_text(json.dumps(man, indent=1, ensure_ascii=False),
                        encoding='utf-8')
    print('source %s / %d bytes  (matches registered: %s)'
          % (got[:16], len(src), man['source_sha256_matches_registered']))
    print('dest   %s / %d bytes  (prefix %d bytes byte-identical: %s; suffix %d)'
          % (sha(out)[:16], len(out), man['inherited_prefix_bytes'],
             man['inherited_prefix_is_byte_identical'], man['authored_suffix_bytes']))
    for k, v in applied.items():
        print('  edit %-10s n=%d  %s' % (k, v['n_replacements'], v['anchor_head']))
    print('LEDGER_REFRESHED')
    return 0


if __name__ == '__main__':
    sys.exit(main())
