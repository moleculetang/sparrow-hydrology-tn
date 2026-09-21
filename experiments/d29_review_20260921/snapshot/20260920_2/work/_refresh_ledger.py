"""`20260920_2` -- generate THIS round's copy of the closed-hypothesis ledger, then
edit it IN PLACE, here, programmatically.

ZERO FORWARDS.  ZERO FITS.  Reads one peer file (read-only) and writes two files
inside this round's `reports/`.

WHY A SCRIPT AND NOT A HAND COPY
--------------------------------
The ledger is a 119821-byte inherited artifact whose whole value is that its
earlier rows are BYTE-IDENTICAL to what earlier rounds wrote.  Re-typing 117 KB of
historical registration is exactly how a number gets silently changed while nobody
is looking -- which is the reason the round-1 ledger's own header gives for
seeding it the same way.  So: copy the bytes, assert the source digest, then apply
a small number of exact-string edits, each of which must match EXACTLY ONCE.  If an
anchor is missing or ambiguous the script raises instead of writing, because a
partial edit is worse than no edit.

THE ANCHORS ARE DERIVED FROM THE SEED, NOT RETYPED
--------------------------------------------------
`20260920_1`'s copy retyped `SCOPE_TAIL` by hand and paid for it with a comment
apologising for the fact that `\\n\\n---\\n\\n## 0.` is a shape that closes EARLIER
scope blocks too -- i.e. the retyped anchor was ambiguous in principle and only
unique by luck of that round's text.  This copy does not retype: every anchor is
LOCATED in the seed by a predicate and the surrounding whitespace is READ OUT of
the seed (`gap`, `sep`), so the splice cannot drift from the file it is splicing.
The two hand-written pieces of the round-1 script that remain hand-written here are
the ones that are genuinely new prose: the new row, the new section, the new tail
items, the new scope block, and the new count bullet.

WHAT IT REGISTERS
-----------------
1. the source digest, so the copy step stays independently re-checkable;
2. the resulting digest, plus the byte counts on both sides;
3. every edit it made, as `(label, anchor_head, n_replacements)`, so a reader can
   diff the claim against the file.

THE EDITS (six, and only six)
-----------------------------
(a) the title line;
(b) a new renewal-scope blockquote, inserted above `## 0. 一行摘要`;
(c) the `## 0.` running count, plus the fifth counting bullet that records WHY it
    moved this time;
(d) one new table row (row 17);
(e) a new section `## 21.`;
(f) new `## 随件` items 34+.
History is never rewritten: every earlier row and every earlier section is carried
through untouched.
"""
import hashlib
import json
import sys
from pathlib import Path

R = Path(__file__).resolve().parent.parent
UP = R.parent
SRC = UP / '20260920_1' / 'reports' / '已关闭假设台账.md'
DST = R / 'reports' / '已关闭假设台账.md'
MANIFEST = R / 'reports' / '_ledger_refresh.json'

SRC_SHA = '401277a5ee056b44c38193fc179dd71a31bf812292510d3b15cbe38d9225d176'
SRC_BYTES = 119821


def sha(b):
    return hashlib.sha256(b).hexdigest()


def sub_once(text, anchor, new, label):
    n = text.count(anchor)
    if n != 1:
        raise SystemExit('LEDGER_ANCHOR_NOT_UNIQUE [%s]: found %d occurrence(s) of %r'
                         % (label, n, anchor[:80]))
    return text.replace(anchor, new), n


def loc(lines, pred, label):
    """The unique 0-based index of the one line satisfying `pred`."""
    hits = [i for i, ln in enumerate(lines) if pred(ln)]
    if len(hits) != 1:
        raise SystemExit('LEDGER_ANCHOR_NOT_UNIQUE [%s]: %d line(s)' % (label, len(hits)))
    return hits[0]


SEED = SRC.read_text(encoding='utf-8')
if sha(SEED.encode('utf-8')) != SRC_SHA or len(SEED.encode('utf-8')) != SRC_BYTES:
    raise SystemExit('LEDGER_SEED_DIGEST_MISMATCH: refusing to write a ledger whose '
                     'history cannot be proved to be the round-1 bytes')
LINES = SEED.split('\n')
EDITS = []

# ---------------------------------------------------------------- (a) the title
#
# Read line 0 and rewrite only the three parts of it that describe THIS round.
# Retyping the title would be a second place for the round tag to be wrong.
TITLE_OLD = LINES[0].rstrip('\n')
if not TITLE_OLD.startswith('# 已关闭假设台账（`20260920_1`'):
    raise SystemExit('LEDGER_TITLE_UNEXPECTED: %r' % TITLE_OLD[:60])
TITLE_NEW = (TITLE_OLD.replace('`20260920_1`', '`20260920_2`')
             .replace('承接 `20260919_5`', '承接 `20260920_1`')
             .replace('新增第 16 行', '新增第 17 行'))
if TITLE_NEW == TITLE_OLD:
    raise SystemExit('LEDGER_TITLE_UNCHANGED')
SEED, n = sub_once(SEED, TITLE_OLD + '\n', TITLE_NEW + '\n', 'title')
EDITS.append(('title', TITLE_OLD + '\n', TITLE_NEW + '\n', n))

# ------------------------------------------------- (b) this round's scope note
#
# BOTH scope blocks end with a line that begins `> 而同名文件会`.  They are told
# apart by their LAST CHARACTERS, which round 5 and round 1 genuinely differ on:
# round 5's ends `。`, round 1's ends `、不会报错。`.  `loc` asserts uniqueness.
L5 = loc(LINES, lambda l: l.startswith('> 而同名文件会') and l.endswith('`ok`。'),
         'scope-tail/round5')
L1 = loc(LINES, lambda l: l.startswith('> 而同名文件会') and l.endswith('、不会报错。'),
         'scope-tail/round1')
POS5 = SEED.index(LINES[L5])
POS1 = SEED.index(LINES[L1])
# The whitespace between round 5's tail and round 1's block start, read out of the
# seed so this round reproduces the same spacing rather than guessing it.
BLOCK1_HEAD = '> **`20260920_1` 的翻新范围'
GAP = SEED[POS5 + len(LINES[L5]):SEED.index(BLOCK1_HEAD, POS5)]
if GAP not in ('\n\n', '\n\n\n'):
    raise SystemExit('LEDGER_GAP_UNEXPECTED: %r' % GAP)
# The separator between round 1's tail and the `## 0.` heading, likewise read out.
HEAD0 = '## 0. 一行摘要'
SEP = SEED[POS1 + len(LINES[L1]):SEED.index(HEAD0, POS1)]
if not SEP.startswith('\n') or not SEP.endswith('---\n\n'):
    raise SystemExit('LEDGER_SEP_UNEXPECTED: %r' % SEP)
SCOPE_TAIL = LINES[L1]
SCOPE_NEW = SCOPE_TAIL + GAP + '''> **`20260920_2` 的翻新范围（只有这六处，其余逐字不动）**：
> ① 表头；② §0 的一行摘要与其计数说明随**第 17 行**更新；
> ③ 表格新增**第 17 行**；④ 新增 **§21：第 17 行：legacy/mobile 可动性映射（单参数、零拟合）**；
> ⑤ 末尾「随件」新增第 **34–39** 条；⑥ 本节就是新增的这一段。
> **第 1–16 行、§10、§11b、§13、§14、§15、§16、§16.5、§17、§18、§19、§20 的全部历史登记原文
> 逐字保留，不得删改。** 本轮**没有**新增任何第五种裁决取值，**也没有改动任何一行的裁决**。
> **第 17 行是本台账里第一条判"状态之间的可动性"的行**：第 16 行判的是"某个池的动员律"
> （换结构），本行判的是**什么 N 能碰到水**这一层，且只判了**一阶、单参数**这一种拼写。
>
> **本文件是 `20260920_1\\reports\\已关闭假设台账.md` 的逐字副本 + 本轮新增**，
> 复制方式与前两轮相同：**复制，不重打**。复制那一步**在 `work\\_refresh_ledger.py` 里被断言**，
> 而不是靠叙述：原件 sha256 = `401277a5ee056b44c38193fc179dd71a31bf812292510d3b15cbe38d9225d176`、
> `119821` bytes，脚本在复制**之前**核对这两个值，不等即拒绝写盘。
> **注意副本的方向**：`20260920_1` 的那一份**本身**已被它自己更新过
> （§20 与随件第 28–33 条）⇒ 本文件继承的是**已更新**的那一份，
> 那些内容**不在本轮重做、也不在本轮重新论证**。
>
> **本轮新增的"依据"列引用一律写出轮次编号**（`20260920_2\\reports\\…`），
> 理由与前几轮相同：本文件被复制到下一轮之后，"本目录"会指向另一个目录，
> 而同名文件会**静默命中另一轮的产物**，交付机检把它记作 `ok`、不会报错。''' + SEP + HEAD0
SEED, n = sub_once(SEED, SCOPE_TAIL + SEP + HEAD0, SCOPE_NEW, 'scope')
EDITS.append(('scope-block', SCOPE_TAIL + SEP + HEAD0, SCOPE_NEW, n))

# --------------------------------------------- (c) the `## 0.` running count
#
# Two separate edits, because the `## 0.` header carries the count in TWO places:
# the summary lines above the discussion, and the running bullet inside it.  Fixing
# one and not the other is precisely the self-contradiction the round-3 note
# records twice; `_refresh_ledger` therefore edits both or neither.
J = loc(LINES, lambda l: '个行号（第 11 行' in l, 'summary')
SUMMARY_OLD = '\n'.join(LINES[J:J + 2])
SUMMARY_NEW = ('**共 17 个行号（第 11 行按 `20260919_2` 的裁定拆为 11a／11b ⇒ 表内 18 行）：**\n'
               '**16 行已裁决，2 行未裁决 —— 第 12 行（`未裁决 —— 严格未答`）'
               '与第 14 行（`未裁决 —— 预注册缺口`）。**')
SEED, n = sub_once(SEED, SUMMARY_OLD, SUMMARY_NEW, 'summary')
EDITS.append(('summary-count', SUMMARY_OLD, SUMMARY_NEW, n))

I = loc(LINES, lambda l: '`20260920_1` 是这一行的第四次变动' in l, 'count-bullet')
COUNT_OLD = '\n'.join(LINES[I:I + 3])
COUNT_NEW = (COUNT_OLD + '\n'
             + '> * **`20260920_2` 是这一行的第五次变动，同样登记**：新增第 **17** 行，'
               '裁决取值 **`装置成立、假设不成立`**（**装置层面**，外延见 §21.7）。\n'
               '>   逐行数过 §0 表之后的新读数是：18 个表行中 `CLOSED` `9` 行'
               '（1／2／4／5／6／7／8／9／11a）、`CLOSE` `1` 行（10）、\n'
               '>   `装置成立、假设不成立` `6` 行（3／11b／13／15／16／**17**）、'
               '`未裁决` `2` 行（12／14）。')
SEED, n = sub_once(SEED, COUNT_OLD, COUNT_NEW, 'count-bullet')
EDITS.append(('count-bullet', COUNT_OLD, COUNT_NEW, n))

# --------------------------------------------------------- (d) the table row
#
# The anchor is row 16's WHOLE LINE, and the replacement is that line plus a
# newline plus the new row.  Row granularity, plus a newline in the replacement,
# is what keeps the table a table: anchoring on a TAIL FRAGMENT and consuming the
# line's `\n` glued two rows into one 2000-character line in round 1's first draft
# -- every content check still passed, because the text was all there.
ROW16_FULL = LINES[loc(LINES, lambda l: l.endswith('不预留开关** |'), 'row16')]
ROW17 = (
    '| **17** | **legacy/mobile 可动性映射（单参数、零拟合）**（`20260920_2`：只增加'
    '**一个**化学无关的状态对（`N^L` 弱可动 legacy／`N^M` 可达 mobile）与**一个**共享时间尺度 '
    '`τ_m`（`k_m = 1/τ_m`，**单标量、与水文无关**：`T^{mob}_t = q_m·Ñ^L_t`，'
    '`q_m = -expm1(-k_m)`，**不得写成 `f(Q)·N^L`**）；输入只进 legacy 侧、demand 按比例'
    '参数-free 分抽（`U_L = U·NL/P`，`P>0`；`P==0 ⇒ U==0`）、两个池**共用**冻结的 `s_M`'
    '（`Loss_t = (1-s_M)(N^{L,*}+N^{M,rem})` —— 这就是"不引入第二个损失参数"）；'
    '`contact <= 0` **只禁水输出、不禁 `N^L → N^M`**；`k_ex = 0`，'
    '**不实现、不参数化、不预留**；30 维一字未改、`n_fits = 0`、前向 `10` 次 + 求根 `18` 次。'
    '问的是：**同一个 `τ_m` 上能不能同时买到事件振幅、又不把长期平均水平推出冻结基线**） | '
    '**`装置成立、假设不成立`**（**装置层面**：'
    '装置被证明确实装上 —— N1′ 的 `q_m = 1` 归约七通道与父控制 `P-upper` **逐位相同**；'
    '**但**主臂 `P-1e2`（`τ_m = 200 d`，由盲规则 `√(τ_water·τ_frozen)` 指定、**不看性能**）'
    '五门 **0/5**、九个核臂**全部 0/5** ⇒ 预注册层 1 结局 **`NO_AMPLITUDE_MECHANISM`**；'
    '层 2（`enters_no_gate = True`、不授权任何东西）结局 **`KM_MAPPING_NO_SHAPE_CAPABILITY`**：'
    'F1 最好只补上门差距的 `0.22988112794551832`（`A_L1`）／`0.27174197423503527`（`A_L3`），'
    '远低于 `0.5` ⇒ **关闭这一阶 `k_m` 映射**。'
    '**最强的一环是 `τ_m^★ = 21104.573561789428 d`**：把水平**由构造**压回冻结基线之后，'
    '`A_L1 = 0.8348555927188647` 与网格平台 `0.834557` **在小数点后第三位上仍然相同**，'
    '**仍远低于冻结基线 `1.0400772083480145`** ⇒ '
    '**振幅缺口是机制的性质，不是水平的性质**，'
    '而"去均值"论证永远不会重跑前向，所以看不到这一点。'
    '**外延见 §21.7，不得外推**；**不得**读成"legacy/mobile 可动性映射这个机制族不存在"'
    '——本轮只判了**一阶、单参数**这一种拼写） | '
    '`20260920_2\\reports\\phase0_gates.json`（§3.1 来源对齐／§3.2 闭式水平曲线／'
    '§3.3 时序与掩码语义 + N1′–N17，**零前向**）、'
    '`20260920_2\\reports\\arms.json`（十臂表 + 主臂盲规则 + 求根协议，sha `ca3807db…`）、'
    '`20260920_2\\reports\\phase1_arms.json`（`10` 次前向 + 层 1 全部门 + 全部诊断）、'
    '`20260920_2\\reports\\level_variance.json`（水平／方差分解）、'
    '`20260920_2\\reports\\shape_diagnostics.json`（层 2 shape 量）、'
    '`20260920_2\\reports\\level_matched_point.json`（`τ_m^★` 二分求根：**不是拟合**）、'
    '`20260920_2\\reports\\verdict.json`（**双层判决**：`main_verdict_gates` 与 `shape_layer` '
    '两把钥匙）、`20260920_2\\reports\\audit_dp2.json`（**独立复算**：不 import 本轮任何模块；'
    '并由密集帧独立重建 `A_L1`，**20/22 逐位相同**）、'
    '`20260920_2\\reports\\neighbour_write_check.json`（三个邻居轮次未被改动的机器证据）、'
    '`20260920_2\\reports\\预注册_冻结.json`（先冻结后前向：`forward_runs_so_far = 0`） | '
    '**否（这一阶已关闭）**：层 2 的 `KM_MAPPING_NO_SHAPE_CAPABILITY` 明写'
    '**不应继续扩成两参数模型**。重开入口**很高且必须另行预注册**：'
    '要么把"平衡 mobile availability"与"transfer timescale"**拆开**'
    '（§21.1 的 SWAT+／HYPE 多池出处），要么换掉"单一 `k_m` 承担两件事"这一形制本身；'
    '**不得**在同一轮里加第二个率常数来救第二个 |\n')
SEED, n = sub_once(SEED, ROW16_FULL, ROW16_FULL + '\n' + ROW17, 'row17')
EDITS.append(('table-row-17', ROW16_FULL, ROW16_FULL + '\n' + ROW17, n))

# ------------------------------------------------------ (e) the new section
SECTION_ANCHOR = '\n## 随件：不能证明的事（本台账整体）\n'

SECTION = '''
## 21. **新增第 17 行**：legacy/mobile 可动性映射（单参数、零拟合）—— `20260920_2`

> **本节的地位。** 它是第 17 行的**依据**，不是综述。第 17 行说的每一件事，
> 都在这里给出读数与它落在哪个文件里；本节说的每一件**不能**由第 17 行推出来的事，
> 都在 §21.7 里逐条列出。

### 21.0 被检验的命题（用户原话，不得改写）

> **"下一阶段唯一应该做的方向：增加 legacy → mobile 的可动性映射"**；
> **"只增加一个'legacy N → mobile N'的可动化状态和一个时间尺度；先解决'什么N能碰到水'，
> 再谈'碰到水以后走哪条路'"**；
> **"如果 k_m 在宽时间尺度包络下仍无法产生事件振幅，不要再开 k_ex"**。

⇒ 本轮问的是**一件事**：**在同一个 `τ_m` 上，能不能同时买到事件振幅、
且不把长期平均水平推出冻结基线？** 派生第二问：**水平统计量是"通量型"还是"比值均值型"？**

**方位由两个冻结读数给出，本轮的臂一个都还没跑**：水给 N 的寿命 `5.411787491476153 d`、
冻结模型给 N 的寿命 `1/1.442017619452899e-04`（≈ 6934 d）—— 相差约 1170 倍。
`k_m` 就是横跨这 1170 倍的那**一个**旋钮。

### 21.1 干预形式与装置（**零拟合、一个自由度**）

保留下来的：source ledger、crop demand、水文学、河道路由、观测算子、**冻结的 `s_M`**、
`g` 的两种拼写、`g_u = Qu/Vu`、`φ`、`V_u = upper_water + Qu`、`V_s = lower_slow_storage_mm + Qs`。
**替换掉的只有一处**：`A` 单状态 → **两个陆相状态** `N^L`／`N^M`。

六点同时成立（缺一即 `BLOCKED`），全部写成断言落在 `phase0_gates.json`：

1. **输入只进 legacy 侧**：`N^{L,+} = N^L_{t-1} + I_t`、`N^{M,+} = N^M_{t-1}`；
2. **demand 的分抽是参数-free 的比例式**（`U_L = U·NL/P`），并断言 `U_L + U_M == U`、
   `U_L <= NL_+`、`P == 0 ⇒ U == 0`（全域，含掩码，不得出现 `0/0`）；
3. **`T^{mob}_t = q_m·Ñ^L_t` 与水文无关**：`q_m = -expm1(-k_m)`、`k_m = 1/τ_m`，
   **单个标量、整轮只此一个**，所有臂共用同一拼写；
4. **`contact <= 0` 只禁止水输出**：掩码上 `g_u = 0 ⇒ E_M = F_f = J = 0`，
   **而 `T^{mob}` 照常发生**、两个池照常演化 —— 这条是红线，写成断言；
5. **数值拼写**：`q_m` 必须写 `-expm1(-k_m)`，**不得**写 `1 - exp(-k_m)`（N12）；
6. **归约点存在**（N1′，见 §21.2 第一段）。

⇒ **本轮的唯一自由度是 `τ_m` 一个标量**，臂表 = `τ_m` 网格；30 维参数一字未改。

### 21.2 裁决依据（可追溯，逐条落盘）

**（一）装置确实装上了 —— 这是本行唯一可以当证据用的"装置"读数。**
`q_m = 1`（父控制）时 `N^{L,*} = 0`、`N^{M,pre} = max(N^L + N^M + I − D, 0)`，
逐项回到上一轮：`E_M = A·g_u`、`F_f = E_M·φ_f`、`J = E_M(1−φ_f)`、
`N^M_t = A(1−g_u)s_M`、`F_s = L^{pre}·g_s` —— **七条通道逐位相同**，
且读数**就是** `20260920_1\\reports\\phase1_arms.json::arms/P-upper`（**零前向**）。
`K-inf` 是**门**不是臂，其读数只用于 N1′。
⇒ **"新核确实是同一个模型换掉了一层"这句话是被证明的，不是被宣称的。它没有说那层换对了。**

**（二）主臂由盲规则指定，不看性能。**
`τ_m^primary = √(τ_water·τ_frozen) ≈ 200 d`（臂 `P-1e2`），两个输入都是冻结读数。
**臂表落盘并哈希（sha `ca3807db…`）在任何前向之前**；
`预注册_冻结.json` 记 `forward_runs_so_far = 0`、`n_forwards = 10`、`primary_arm = P-1e2`。
**只有主臂可以签发层 1 的 capability。**

**（三）层 1（硬判决层，唯一能签发 capability 的一层）：主臂五门 0/5，九个核臂全部 0/5。**
`main_verdict_gates = ["G1","G2","G3","G5","G5b"]` 一个数都没改、也没放松。
主臂 `P-1e2` 的五门读数连同全部未达标臂**照报**在 `phase1_arms.json`。
结局 **`NO_AMPLITUDE_MECHANISM`**：本轮的问题正是要消掉上一轮的三个 Γ 形态，
而它们**没有**点火（`n_gamma_conditions_firing_on_the_primary = 0`）。

**（四）动态确实质变了 —— 这是本轮唯一"变好了"的地方，但它不买振幅。**
- 月初脉冲的过表征从上一轮父控制的 `5.001824614730297` 倍**塌到 `0.9246350263990235` 倍**
  （**低于 1**，即月初由过表征变成**欠表征**）⇒ **脉冲形态消失了，不只是被削弱**；
  掩码上新核与冻结核在慢路上的差异**登记为有意的结构性差异**。
- 新的 N 寿命是水周转时间的 `36.956366138731504` 倍。
⇒ **上一轮唯一未被击破的恒等式 Γ 形态被打破了**（`τ_eff/τ_hydro = 0.996355630760417`
这个恒等式本身在**水侧**仍在，因为 `g_u`／`s_M`／`x_u` 都不含 `q_m`；
被打破的是**N 的时间尺度不再被水锁死**），**而 `A_L1` 反而略降**（见（五））。

**（五）振幅：峰值/基比落在一个很窄的平台上，且平台远低于冻结基线。**
候选网格上 `A_L1` 只在 `0.8196184997235119`–`0.8340612498511533` 之间（跨度
`0.014442750127641402`），`A_L3` 只在 `0.827955825334509`–`0.8543579632495699` 之间；
**每一个候选都低于冻结基线**（`A_L1` `1.0400772083480145`、`A_L3` `1.023217381861253`），
更低于已注册门阈值（`A_L1` `1.1579754800655908`、`A_L3` `1.1495455668222099`）。
主臂 `A_L1 = 0.8328411654810111`、`A_L3 = 0.8422771602949068`。
⇒ **`k_m` 不但没有买到振幅，还把它略微拉低了。**

**（六）层 2（`enters_no_gate = True`、不授权任何东西）：F1 不成立。**
F1 要求至少一个**候选**把"父控制 → 已注册门阈值"的差距补上 ≥ 50%。
两个锚点都是冻结读数（`A_L1(P-upper) = 0.7373725809958451`、
`A_L3(P-upper) = 0.7442117522548058`）：`A_L1` 最好 `0.22988112794551832`（`K-365`）、
`A_L3` 最好 `0.27174197423503527`（`K-3p6e3`）⇒ 最好也只有约 **23%**。
结局 **`KM_MAPPING_NO_SHAPE_CAPABILITY`**。
（F2 读数为 `0.23176970947804762`／`0.27398785662708963`，**本次判决没有咨询它**：
注册表对 `KM_MAPPING_NO_SHAPE_CAPABILITY` 只要求 F1，F2 照报。）
**层 2 的结局只写进 `verdict.json::shape_layer`**，不进入 `main_verdict_gates`、
不改变任何门的 `passed`、不改写层 1 的判决（N17 断言）。

**（七）最强的一环：`τ_m^★` 上"把水平拉回去之后还剩多少"。**
`τ_m^★` 由**二分**求出（`ROOT_TOL = 1e-4` 相对、`MAX_ROOT_FORWARDS = 20`、
区间 `[3.6e3, 1.0e6] d`，实用 `18` 次前向）：
`τ_m^★ = 21104.573561789428 d`，`mean_concentration = 2.8235113396138667`
对目标 `2.8234476727379607`（相对残差 `2.2549338002867052e-05`）。
它**只用冻结基线**这一个目标、**不读任何观测 TN／NH₄**、**不是拟合参数**
（没有优化库、没有最小二乘、只有二分）；
**先决条件 N14 成立**：`C̄(τ_m)` 在注册区间上**单调不增**（0 次回升），
故**不落** `LEVEL_CURVE_NOT_MONOTONE`、求根被允许。
**且它明令不得承载 G5b** —— 在那里 G5b 是**由构造满足的**；`N16` 断言它不出现于任何门的
通过条件，层 1 的 `FULL_CAPABILITY` 在代码路径上不可能由它产生。
读数：`A_L1 = 0.8348555927188647`、`A_L3 = 0.8552682953266111`。
⇒ **与网格平台 `0.834557` 在小数点后第三位上仍然相同**：
**把水平由构造压回冻结基线几乎不花任何振幅代价，而振幅仍然远低于冻结基线。**
**这就是比"去均值"更强的证伪** —— 去均值是加法平移，而核心事件判据是峰/基比，
**不是平移不变量**；而且去均值论证永远不会重跑前向，所以看不到这一点。
（同一诊断点上 `nse = -1.3159861791622283`、`median_station_nse = -7.744967993025499`：
**水平匹配了，月尺度结构反而远差于冻结基线**。）

**（八）闭式的命运：作为预测它**部分**被确认、**部分**被证伪 —— 两者都必须照实写。**
- **水平统计量**：实测穿越 `τ_m^★ = 21104.573561789428 d` **远大于** `3.6e3 d`，
  故 P-A 的证伪器**没有点火** ⇒ **水平统计量是通量型**。
  一条独立旁证：闭式的 `E/I` 曲线与实测 `C̄` 曲线的**形状**在整条网格上只差约 5%–9%
  （两者都从快端到慢端下降约 2.4 倍），而**绝对水平**上闭式系统性低估约 2.2–2.4 倍 ——
  那个倍数正是它看不到的 `Cov(M_t, 1/Q_t)` 成分。
- **P-E①（"`C̄` 在候选网格上极差 ≤1%"）被证伪**：实测极差是 **2.4 倍**
  （`24.9330477317` → `10.388353086369117`）。**这条不需要前向就能看出**：
  注册表自己的 §1.2 里"整条注册网格上水平不动"与三行之下的"穿越点在 `6.9e3 d`"
  **互相矛盾** —— 若水平在网格上真不动，就不会有穿越；闭式在**实测 `s_M` 场**上的逐格计算
  同样给出 2.4 倍的跨度。**登记为对计划文本的一处更正**（落盘在 `phase0_gates.json::3.2`，
  与"按 `s_M` 带展开的穿越带 `2.4e3–2.9e4 d` 不可复现"一并登记）。
- **P-E②（"`A_L1` 朝观测方向改善 ≥ 门差距的一半"）被证伪**（见（六））。
- **P-C**：`A_L1` 从父控制 `0.7373725809958451` **升**到平台（预测的"先升"成立），
  但**没有**回落（预测的"后降"不成立）—— 它一直平到 `1e6 d`。
  ⇒ 计划里"若 `A_L1` 单调下降则'给 N 一个比水慢的寿命'本身不能买振幅"那条证伪器
  **没有点火**，但**实质结论相同**：那条上升只买到门差距的约 23%，然后就饱和了。

**（九）本行登记的一处**未由本轮裁决**的注册文本冲突。**
计划 §2.5 的 `BLOCKED` 行原文是"N1′–N16 任一失败 ⇒ 立即停止并登记；
**不得**当作机制被否的证据"。`N10` 在**极慢端元** `K-slowend` 上有一项**字面失败**：
`source_label_sum_errors` 的 `M` 通道 `1.043081283569336e-06` > 容差 `1e-6`（超出 4.3%）；
其余通道、以及 `local_balance_max_kg = 1.1210795491933823e-07`、
`network_balance_kg = 4.3213367462158203e-07` 全部通过。
**该臂的注册角色是"耦合证明"，明令不得签发任何结局。**
台账的处置：**照实登记为"字面已点火"，不重新归类、不改容差、不放宽 `1e-6`**，
并在 `verdict.json::layer1.registered_BLOCKED_clause` 里记 `needs_the_user = true`。
**这一条必须由用户裁定，本行不替它裁定。**（机理读数：该残差随池量级线性增长，
相对残差在六个数量级上恒为 `1e-15`，即纯 float64 舍入 —— 但**登记的是绝对容差**，
这一张力本身就是要交给用户的那个问题。）

### 21.3 三处**必须随行**的限定（否则本行会被读宽）

1. **本行的裁决是"装置层面"，说的是"这一种拼写"，不是机制族。**
   它说的是：**一个**与水文无关的时间尺度、**一个**共享 `s_M`、**一阶**转移、
   输入只进 legacy 侧、demand 参数-free 比例分抽 —— 这一种拼写**没有 shape 能力**。
   **不是**"legacy/mobile 可动性映射这个机制族不存在"。
2. **"关闭"关闭的是"一阶、单参数"这一阶。**
   层 2 的注册表原文：**不应继续扩成两参数模型**。想重开必须**另行预注册**并把
   "平衡 mobile availability"与"transfer timescale"**拆开** ——
   而那正是"下一步能否有依据"的前提（§1.1 的 SWAT+／HYPE 多池出处只提供出处，
   **本轮不加任何这些结构**）。
3. **`τ_m^★` 不得承载 G5b，层 2 不得承载 capability。** 见 §21.2（六）（七）。

### 21.4 本行的裁决：**`装置成立、假设不成立`（装置层面）**

装置被**证明**确实装上（N1′ 逐位、与 `P-upper` 读数逐位一致）；
假设（"一个 `τ_m` 能同时买到振幅且不推动水平"）**不成立**，
而且**不成立的方式比预注册的更强**：不是"水平与振幅被一个参数死锁"
（层 2 的 `SHAPE_RESPONSIVE_BUT_LEVEL_COUPLED`），而是
**振幅在任何 `τ_m` 上都没有被买到**（层 2 的 `KM_MAPPING_NO_SHAPE_CAPABILITY`）。
**这两者必须分开读**：前者会授权"下一步拆开两个职责"，后者**不授权**。

### 21.5 本行的重开条件（**入口很高，必须另行预注册**）

1. 把"平衡 mobile availability"与"transfer timescale"两个职责**拆开** ——
   **且必须在同一轮里一次预注册两个参数、并说明各自的识别来源**；
   **不得**在同一轮里加第二个率常数来救第二个。
2. 或者换掉"单一 `k_m` 承担两件事"这一形制本身（多池：
   active／stable organic-N + 独立矿化率）。
3. **不得**用"换判据／换聚合／换噪声带／换簇方案"把本行的任一层读成
   `DUAL_PATH_CAPABILITY_DEMONSTRATED`。

### 21.6 与第 16 行的关系（**两行不得互相援引为证据**）

第 16 行（`20260920_1`）判的是"**守恒的双水路移动水浓度**"这一层（零参数）；
本行判的是"**legacy→mobile 的可动性**"这一层（单参数）。
两轮的**自由度不同、干预位置不同、可签发的结局不同**。
唯一可以并列的是**同一套五门基线与同一套阈值** —— 那是"性能范围保持一致"的实现，
**不是**"两轮结论互相加强"。**第 16 行不得用来支撑本行的"关闭"，反之亦然。**
本行能成立，恰恰是因为第 16 行**已经被判过**：它把结构失败定位到"可动性"这一层。

### 21.7 本行的**外延界线**（与 §21.3 同等重要，缺一不可）

- **不得**读成"双路径移动水浓度结构失败"（那一层是第 16 行判的，且第 16 行明确禁止外推）。
- **不得**读成"`k_m` 只是没调好"：`τ_m` 走遍了 `7 d` 到 `1e6 d`（跨 5 个数量级），
  平台宽度只有 `0.014442750127641402`（`A_L1`）—— **没有"更好的点"这回事**。
- **不得**把 `τ_m^★` 的读数当成 G5b 通过（那里 G5b 由构造满足）。
- **不得**把层 2 的任何结局读成 capability（层 2 不使任何门通过）。
- **不得**读成"零拟合的收益已被排除"——本谱系的教训是**拟合会回吐**；
  本行只说**这一种零拟合拼写**没有 shape 能力。
- **不得**把本行读成"`k_ex` 的入口已打开"。用户裁定是"**如果 `k_m` 在宽时间尺度包络下
  仍无法产生事件振幅，不要再开 `k_ex`**" —— 本轮正是那个条件成立的情形，
  **且 `k_ex` 本轮不实现、不参数化、不预留**。
'''

SECTION_BLOCK = '\n' + SECTION.strip('\n') + '\n' + SECTION_ANCHOR.lstrip('\n')
SEED, n = sub_once(SEED, SECTION_ANCHOR, SECTION_BLOCK, 'section')
EDITS.append(('section-21', SECTION_ANCHOR, SECTION_BLOCK, n))

# -------------------------------------------------------- (f) the tail items
TAIL_NEW = '''
34. **第 17 行是本台账里第一条判"状态之间的可动性"的行**，因此它的裁决比前面每一行
    都更容易被读大。它判的是**什么 N 能碰到水**这一层，且只判了**一阶、单参数**这一种拼写。
35. **本行的"零拟合"是结构上的（与第 16 行同类），不是预算上的（与第 15 行不同类）。**
    ⇒ 本行的零拟合**把证据面包络收窄了**：30 维参数包络对"可动性映射"这一层
    **不构成约束**，所以"换掉可动性映射仍然不过门"这句话**不能**被读成"参数没调好"。
36. **`τ_m^★` 是本台账里第一个"由构造满足某个门"的诊断点，因此它绝不能承载那个门。**
    它在 `level_matched_point.json` 里明写 `enters_no_gate`，
    `verdict.json::N16` 断言它不进入任何门的通过条件、且层 1 的 `FULL_CAPABILITY`
    在代码路径上不可能由它产生。
37. **第 16 行登记的"`A_L1` 无法独立复算"这一缺口，本行已关闭。**
    本行落了 `reports\\daily_dense_pL.parquet` —— **每臂 169,476 个站日 × 10 臂 =
    1,694,760 行**（计划 §9.2 ⑰ 只写"`169476`"，而 §四 又列了 `arm` 列，
    **两种读法都报，该歧义已登记为偏离，未静默裁定**）、
    **只含浓度列、不含任何质量列**。独立复算器（`work\\audit_dp2.py`，
    **不 import 本轮任何模块**）据此重建全部 `A_L1`／`A_L3`：
    `n_statistics_recomputed = 22`、`n_agreeing_bitwise = 20`，
    **余 2 项全部是 `R5-ref`**（`L1` 差 `0.03697844074487833`、
    `L3` 差 `0.007011773820216671`），且**全部归因于**其稀疏格栅 ——
    3274 个必需日槽缺 465、214 个事件中 117 个窗口偏短（最差 16 d），
    而"短窗口只能降低 `C_peak`"所预测的方向与实测一致；
    `dense_frame_disagreements` **为空**，即**密集帧上没有一处分歧**。
    （`Q0-zero` 的 `L1`／`L3` 两侧**同为未定义**（`0/0`），记为 `both_undefined`，
    **不计入分歧** —— 本项**不得**被写成"有 4 处不一致"。）
    **上一轮的那个缺口无法追溯修复，本轮的是新落的。**
38. **本行登记了一处必须由用户裁定的注册文本冲突，且**不替用户裁定**。**
    见 §21.2（九）：`N10` 在 `K-slowend` 上的 `M` 通道字面超出 `1e-6` 容差 4.3%，
    而该臂的注册角色是"耦合证明"、不得签发任何结局；
    计划 §2.5 的 `BLOCKED` 行与"一次通过、不做早期停止"互相拉扯。
    **照实登记，不重新归类，不改容差，不放宽 `1e-6`。**
39. **本轮**没有**新增任何第五种裁决取值，也**没有**改动第 1–16 行的任何裁决。**
    第 12／14 行的 `未裁决` 与它们的重开条件**保持不动**。
'''

# The tail is an append, so its anchor is the seed's OWN last non-empty line --
# NOT a sentinel such as `'EOF'`.  A sentinel would be a line that appears nowhere
# in either file, so `_verify_ledger_history.py` would count it as a REGISTERED
# anchor line that was replaced, and the delivered file would not contain the
# matching alteration: the verifier's exact-count check (2) would fail on a ledger
# that is in fact correct.  The anchor has to be a real line for the count to mean
# anything.
TAIL_ANCHOR = [l for l in SEED.split('\n') if l.strip()][-1]
TAIL_NEW_TEXT = TAIL_ANCHOR + '\n' + TAIL_NEW
body = SEED.rstrip('\n') + '\n' + TAIL_NEW
if body.count(TAIL_NEW_TEXT) != 1:
    raise SystemExit('LEDGER_TAIL_ANCHOR_NOT_UNIQUE')
EDITS.append(('tail-items-34-39', TAIL_ANCHOR, TAIL_NEW_TEXT, 1))
SEED = body

# ------------------------------------------------------------------- write out
b = SEED.encode('utf-8')
DST.write_bytes(b)
SRC_NOW = sha(SRC.read_bytes())
# The manifest is keyed BY LABEL and carries each edit's FULL anchor and new text,
# because that is the schema `_verify_ledger_history.py` consumes: it re-derives,
# from the manifest alone, which inherited lines a registered anchor is allowed to
# have changed and which delivered lines a registered insertion is allowed to have
# added.  A manifest that recorded only `(label, head, count)` could not support
# that check, and the check is the whole point -- without it "the history is
# untouched" is an assertion rather than a measurement.
MANIFEST.write_text(json.dumps(dict(
    round='20260920_2',
    source=str(SRC.relative_to(UP)).replace('\\', '\\\\'),
    source_sha256=SRC_SHA, source_bytes=SRC_BYTES,
    source_sha256_recomputed=SRC_NOW,
    source_sha256_matches_registered=(SRC_NOW == SRC_SHA),
    destination=str(DST.relative_to(UP)).replace('\\', '\\\\'),
    destination_sha256=sha(b), destination_bytes=len(b),
    n_edits=len(EDITS), n_lines_seed=len(SRC.read_text(encoding='utf-8').split('\n')),
    n_lines_destination=len(SEED.split('\n')),
    edits={l: dict(anchor_text=a, new_text=nt, n_replacements=k)
           for l, a, nt, k in EDITS},
    scope_note=('the earlier rows are carried through byte-identically; the only '
                'differences from the source are the edits listed above'),
    zero_forwards=True, n_fits=0, fit_worker_calls=0),
    indent=1, ensure_ascii=False), encoding='utf-8')

print('LEDGER_WRITTEN  %d -> %d bytes  (%+d)' % (SRC_BYTES, len(b), len(b) - SRC_BYTES))
print('  source sha256  %s' % SRC_SHA)
print('  output sha256  %s' % sha(b))
for l, a, nt, k in EDITS:
    print('  edit %-20s n=%d  anchor_lines=%d new_lines=%d  anchor_head=%r'
          % (l, k, len([x for x in a.split('\n') if x.strip()]),
             len([x for x in nt.split('\n') if x.strip()]), a[:34]))
