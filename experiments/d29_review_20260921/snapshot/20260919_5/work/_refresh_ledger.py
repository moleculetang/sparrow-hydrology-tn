# -*- coding: utf-8 -*-
"""Seed this round's 已关闭假设台账.md from 20260919_4's copy, then add row 15.

The ledger is the CUMULATIVE decision record: rows 1-14 and the sections keyed to
them must survive verbatim, so the file is seeded by copying the peer's copy and
then edited in place -- never retyped.  Retyping a 42 KB historical register is
how digits change without anyone noticing.

RETARGETED this round: round 4's copy had `PEER = 20260919_2`, because THAT was
the round it was copied from.  A new round copies from its immediate parent, so
the seed source moves with it -- leaving `20260919_2` here would silently seed
this round from a ledger that is two revisions stale and would drop rows 13 and
14, i.e. exactly the two rows this lineage most recently argued about.  The
`台账` name filter is kept: it is what makes the assertion below meaningful (a
pattern that could match two files is not a check), and it is why the failure
mode is a loud `assert` rather than a silent pick.

NOTE ON THE PARENT'S FILE.  The parent's ledger is NOT read-only in this round:
section 5 of the plan authorises ONE class of edit to `20260919_4`'s PROSE --
the G5 correction annotations and the §18.8 / 随件-20 registration produced by
`work\round4_g5_correction.py`.  Those edits were made BEFORE this script runs,
so the seed carries them.  The seed is therefore the CORRECTED ledger, and this
round's ledger inherits the correction rather than re-deriving it.

WHY THE EDIT IS IN THE SCRIPT AND NOT DONE BY HAND.  A file that only an ad-hoc
edit writes cannot be rebuilt and cannot be checked; re-running this script is
the check.  Every insertion below is anchored on an exact string taken from the
seed with `assert old in text`, so a seed that has drifted fails LOUDLY instead
of silently producing a ledger that is missing a row.  The transform is a pure
function of the seed: seed -> text -> write is deterministic, and re-running
reproduces the file byte for byte.

Run:  python -B work/_refresh_ledger.py
"""
import hashlib
import pathlib

ROUND = pathlib.Path(__file__).resolve().parent.parent
SOURCE_ROUND = '20260919_4'
PEER = ROUND.parent / SOURCE_ROUND / 'reports'
dst = ROUND / 'reports'

# ---------------------------------------------------------------- anchors ---
ANCHOR_TITLE = ('# 已关闭假设台账（`20260919_4` —— 承接 `20260919_3`，'
                '新增第 14 行，并收窄第 13 行的行动行）')

ANCHOR_R4_BLOCK_END = '> **所以现在 sha 已经不同**——要复核复制这一步，核的是**原件**，不是本文件）。'

ANCHOR_S0_COUNT = ('**共 14 个行号（第 11 行按 `20260919_2` 的裁定拆为 11a／11b ⇒ 表内 15 行）：**\n'
                   '**13 行已裁决，2 行未裁决 —— 第 12 行（`未裁决 —— 严格未答`）'
                   '与第 14 行（`未裁决 —— 预注册缺口`）。**')

ANCHOR_S0_LESSON = ('> * **教训（与 §A2 同类）**：一个**自称已修正**的计数说明，其可信度取决于\n'
                    '>   **有没有逐行数过表**；两次错误的方向相同——都是**把未裁决的行数少算了一行**。')

ANCHOR_ROW14 = '| **14** | **动员前路径特异浓度**'

ANCHOR_APPENDIX = '## 随件：不能证明的事（本台账整体）'

ANCHOR_ITEM20_END = ('    这一条也是本台账"**该条轴没有被它自己的装置量到**"的一次实例：\n'
                     '    G5 在 round 4 里**报的是一个方向的数**，而它被当作另一个方向的证据读了进去。')

# ------------------------------------------------------------- insertions ---
R5_SCOPE_BLOCK = """

> **`20260919_5` 的翻新范围（只有这三处，其余逐字不动）**：
> ① 表头、§0 的一行摘要与其计数说明随**第 15 行**更新；
> ② 新增 **§19：第 15 行：动员前路径特异浓度（质量／浓度中性化）**；
> ③ 末尾「随件」新增第 **21–27** 条。
> **第 1–15 行、§10、§11b、§13、§14、§15、§16、§16.5、§17、§18 的全部历史登记原文逐字保留，
> 不得删改。** 本轮**没有**新增任何第五种裁决取值。
>
> **本文件是 `20260919_4\\reports\\已关闭假设台账.md` 的逐字副本 + 本轮新增。**
> **注意副本的方向**：`20260919_4` 的那一份**本身**已被本轮更新过
> （§18.8 与随件第 20 条，即计划 §五-4 授权的那一类**散文**更正）⇒
> 本文件继承的是**已更正**的台账，§18 的更正**不在本轮重做**，也**不在本轮重新论证**。
> 复制那一步用 `diff -q` 复核到**无输出**（逐字节相同）才做下面的插入。
>
> **本轮新增的"依据"列引用一律写出轮次编号**（`20260919_5\\reports\\…`），
> 理由与 `20260919_4` 那次相同：本文件被复制到下一轮之后，"本目录"会指向另一个目录，
> 而同名文件会**静默命中另一轮的产物**，交付机检把它记作 `ok`。
"""

NEW_TITLE = ('# 已关闭假设台账（`20260919_5` —— 承接 `20260919_4`，'
             '新增第 15 行；第 14 行的 G5 读数已由本轮更正，见 §18.8）')

NEW_S0_COUNT = (
    '**共 15 个行号（第 11 行按 `20260919_2` 的裁定拆为 11a／11b ⇒ 表内 16 行）：**\n'
    '**14 行已裁决，2 行未裁决 —— 第 12 行（`未裁决 —— 严格未答`）'
    '与第 14 行（`未裁决 —— 预注册缺口`）。**')

NEW_S0_LESSON = (
    '> * **`20260919_5` 是这一行的第三次变动，同样登记**：新增第 **15** 行，'
    '裁决取值 **`装置成立、假设不成立`**（**装置层面**，外延见 §19.7）。\n'
    '>   逐行数过 §0 表之后的新读数是：16 个表行中 `CLOSED` `9` 行（1／2／4／5／6／7／8／9／11a）、'
    '`CLOSE` `1` 行（10）、\n'
    '>   `装置成立、假设不成立` `4` 行（3／11b／13／**15**）、`未裁决` `2` 行（12／14）。\n'
    '> * **教训（与 §A2 同类）**：一个**自称已修正**的计数说明，其可信度取决于\n'
    '>   **有没有逐行数过表**；两次错误的方向相同——都是**把未裁决的行数少算了一行**。')

ROW15 = (
    '| **15** | **动员前路径特异浓度（质量／浓度中性化）**'
    '（`20260919_5`：把第 14 行的纯乘性 $\\Xi$ 换成 '
    '$\\Xi^*_{t,r}=k_r(\\beta)\\,\\Xi_{{\\rm raw},t,r}$，'
    '$k_r$ 在冻结 1961–2020 参考期由**模型内部账本恒等式解出**、**不拟合**；'
    '**删掉的是"整体抬高长期动员水平"这一部分，保留的是路径浓度的时间形状**） '
    '| **`装置成立、假设不成立`**（**装置层面**：水平与振幅**各自都有杠杆**，'
    '但**没有任何一个可用点同时过 G1 与 G5b** ⇒ 预注册结局 '
    '`LEVEL_AND_AMPLITUDE_DO_NOT_COINCIDE` 命中。'
    '**外延见 §19.7，不得外推**；**不得**读成"动员前路径特异浓度不存在"'
    '——实测方向**相反**，见 §19.2） '
    '| `20260919_5\\reports\\phase_minus1.json`（C1–C7，**零前向**）、'
    '`20260919_5\\reports\\k_field.json`（4 装置 × 19 β 的 $k$ 场，'
    '**先于读任何观测冻结并哈希**，sha `3e75a83e…`）、'
    '`20260919_5\\reports\\phase1_full.json`（76 点五门 + 月度指标）、'
    '`20260919_5\\reports\\level_variance.json`（§2.7 水平／方差分解 + '
    '§2.8 流域 $k$ 一致性 + C7 伴随）、`20260919_5\\reports\\verdict.json`、'
    '`20260919_5\\reports\\audit_beta.json`（**独立复算**：不 import 本轮模块，'
    '两条求解路线各自重实现） '
    '| **是**（机制族**仍未被关闭**；预注册 §7 对本格明写"'
    '**本格不单独关闭陆相内部结构这根轴**"），'
    '**但重新拟合未被授权**（见 §19.4–§19.5） |')

ROW14_XREF = '（4 候选 × 五门；**其中 G5 一列已被 `20260919_5` 更正，见 §18.8**）'


def main():
    srcs = [p for p in PEER.iterdir() if p.suffix == '.md' and '台账' in p.name]
    assert len(srcs) == 1, srcs
    src = srcs[0]
    target = dst / src.name
    seed = src.read_text(encoding='utf-8')

    print('seeded %s  <-  %s' % (target.name, src))
    print('seed: bytes = %d, lines = %d, sha256 = %s'
          % (len(seed.encode('utf-8')), seed.count('\n') + 1,
             hashlib.sha256(seed.encode('utf-8')).hexdigest()))

    # ---- the seed must already carry round 4's own §18.8 correction --------
    assert '### 18.8 **新增（`20260919_5`）**' in seed, \
        'the seed is NOT the corrected round-4 ledger: §18.8 is missing'
    assert ANCHOR_ITEM20_ALT in seed, 'the seed is missing 随件 item 20'

    text = seed

    def sub(old, new, why):
        nonlocal text
        assert old in text, 'anchor not found (%s): %r' % (why, old[:60])
        n = text.count(old)
        assert n == 1, 'anchor is not unique (%s): %d hits' % (why, n)
        text = text.replace(old, new, 1)

    # ① title
    sub(ANCHOR_TITLE, NEW_TITLE, 'title')

    # ①b the round-5 refresh-scope block, after round 4's block
    sub(ANCHOR_R4_BLOCK_END, ANCHOR_R4_BLOCK_END + R5_SCOPE_BLOCK, 'r5 scope block')

    # ①c §0 counts -- both the summary line and the change register
    sub(ANCHOR_S0_COUNT, NEW_S0_COUNT, 's0 count')
    sub(ANCHOR_S0_LESSON, NEW_S0_LESSON, 's0 lesson')

    # the row-14 cross-reference must ALREADY be in the seed (it came in with
    # round 4's own §18.8 correction); this round does NOT add it.
    sub(ROW14_XREF, ROW14_XREF, 'row14 xref -- must be inherited from the seed')

    # ② row 15 in the table, right after row 14's line
    lines = text.split('\n')
    idx = [i for i, ln in enumerate(lines) if ln.startswith(ANCHOR_ROW14)]
    assert len(idx) == 1, idx
    lines.insert(idx[0] + 1, ROW15)
    text = '\n'.join(lines)

    # ③ §19, immediately before the appendix
    assert ANCHOR_APPENDIX in text
    text = text.replace(ANCHOR_APPENDIX, SECTION19.lstrip('\n') + ANCHOR_APPENDIX, 1)

    # ③b the appendix items 21-27, appended after item 20
    sub(ANCHOR_ITEM20_END, ANCHOR_ITEM20_END + APPENDIX_TAIL, 'appendix tail')

    # `newline='\n'` is load-bearing, not cosmetic: `Path.write_text` opens in TEXT
    # mode, so on Windows the default translates every '\n' to '\r\n'.  The digest
    # printed below is taken over the LF string, and 1397 CR bytes made the file on
    # disk hash to something else entirely -- a producer whose own fingerprint did
    # not describe its own output.  Register as a deviation; the seed is read with
    # universal newlines either way, so this changes no line of the ledger.
    with open(target, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write(text)
    on_disk = hashlib.sha256(target.read_bytes()).hexdigest()
    assert on_disk == hashlib.sha256(text.encode('utf-8')).hexdigest(), \
        'the file on disk does not hash to the string that was written'
    print('wrote %s  bytes = %d, lines = %d, sha256 = %s'
          % (target.name, len(text.encode('utf-8')), text.count('\n') + 1, on_disk))


ANCHOR_ITEM20_ALT = ('G5 在 round 4 里**报的是一个方向的数**，'
                    '而它被当作另一个方向的证据读了进去。')

SECTION19 = r"""
---

## 19. **新增第 15 行**：动员前路径特异浓度（质量／浓度中性化）—— `20260919_5`

> **本节与 §18 是同一机制族的两个层级，两节不得互相援引为证据**（理由同 §19.6）。
> §18 用的是**纯乘性** $\Xi$（有水平杠杆，无水平约束）；本节用的是它的
> **质量中性变体** $\Xi^*=k_r\Xi$（水平被解出来的 $k_r$ 压住）。
> 两节的裁决取值**不同**，这不是矛盾：**装置换了，被检验的假设就换了**。

### 19.0 被检验的命题

计划 §一 的框内原句：

> round 4 那 `80.1%` 的事件振幅收益，是「路径浓度结构」买来的，
> 还是「整体抬高长期动员水平」买来的？

以及由它派生的第二问：**水平与振幅能不能被分开控制？**

用户对装置本身的裁定（逐字）：

> $\Xi^*_{t,r}=k_r(\beta)\Xi_{{\rm raw},t,r}$，其中 $k_r(\beta)$ **不是 TN 拟合参数**，
> 而是在 1961–2020 冻结参考期由模型内部账本求解，使该河段的长期基准动员质量保持：
> $\sum_t av^0_{t,r}[1-e^{-h_{t,r}\Xi^*_{t,r}}]=\sum_t av^0_{t,r}[1-e^{-h_{t,r}}]$。
> $\boxed{\text{只允许'什么时候浓'，不允许凭空把长期平均N抬高。}}$

**本行的裁决只回答这两问**，不回答"路径浓度结构存不存在"以外的任何问题
（外延界线见 §19.7）。

### 19.1 干预形式与装置（**零拟合**）

* 改动**一处**：冻结核 `closures.py:28` 的 `risk=h[t,r]` 换成
  `risk = h[t,r]·Xi_star[t,r]`，其中 $\Xi^*_{t,r}=k_r\cdot\Xi_{{\rm raw},t,r}$，
  **钉必须在乘之后**（`Xi_star = np.where(active, k[None,:]*Xi_raw, 1.0)`）。
* $k_r$ **解**出来，**不拟合**：`n_fits = 0`、`fit_worker_calls = 0`，无目标函数、无搜索，
  不读 TN／NH₄。求解窗口先冻结、后读观测：$k$ 场在**任何观测被读之前**落盘并哈希
  （`k_field_sha256 = 3e75a83e9c0886364905eecfea3b92cda212e14174a12c294dd32a7380bc33dc`，
  哈希**从磁盘读回的值重算**，不是记的）。
* $\beta=0$ 是**逐位** no-op：`np.array_equal(k, np.ones(230))` 与
  `np.array_equal(Xi_star, np.ones_like(Xi_star))` 精确成立（`array_equal`，不是 `allclose`）。
* **四个装置其实是两个**（§19.2 第 1 条），且**主装置由 C2 在任何前向之前改判**为
  `N1e`（评估窗口质量），承载装置为 $\{N1e, N3\}$。

### 19.2 裁决依据（可追溯，逐条落盘）

1. **装置退化，且这是实测不是设计选择。** 逐河段**常数**归一化子 $Q_r$
   在账本恒等式 $\sum_t av^0 Q_r[1-e^{-hk\Xi}]=\sum_t av^0 Q_r[1-e^{-h}]$ 两侧**约掉**
   ⇒ 同一窗口内的质量装置与浓度装置解出**同一个 $k$**。
   实到的是 $N1\equiv N3$、$N1e\equiv N2$（`level_variance.json` 与 `phase1_full.json`
   逐位一致）⇒ **两个装置，不是四个**。
   ⇒ 预注册结局 `INVARIANT_IS_CONCENTRATION_NOT_MASS` **不可能**从"靶量错配"触发，
   只能从**窗口对比**触发；本轮它**没有**命中（`matched` 全表见 `verdict.json`）。
2. **"收益不是水平买来的"——实测，且这是本行的正面读数。**
   round 4 的 $\Xi$-only 在 $\beta=+0.5$ 把平均浓度抬了 `0.22381499501517088`（+22.4%）；
   质量中性化把它压到 `0.005470210988330618`（`mean_concentration_relative_change`，`N1e|0.5`），
   **而 $A_{L1}$ 仍然过 G1**（`N1e|0.5` 的 $A_{L1}$ =
   `1.244023607865766` ≥ G1 目标 `1.1579754800655908`）。
   ⇒ **振幅收益不是整体抬升买来的**：把水平拿掉，振幅**留在原地**。
3. **"月尺度崩塌也不是水平效应"——实测，且这一条否决了本轮的前提。**
   §2.7 的去水平 NSE（候选月序列减其均值、加基线均值后重算）：
   `N1e|0.5` 由 `0.51321330283484945` 只恢复到 `0.5161160183412219`（**恢复崩塌幅度的 1.5%**）；
   `N1|0.5` 由 `0.54933385090562026` 恢复到 `0.5629747502609888`（**恢复崩塌幅度的 8.9%**）。
   ⇒ 崩塌是**方差**效应，不是水平效应。**中性化修复不了月尺度**，本行**不声称**它能。
4. **C7 的冻结线性路由伴随是精确的**：$\Delta\bar C_{\rm pred}$ 与实测 $\Delta\bar C$
   之比在四个点上分别为 `1.0000000000000013`（`N1|0.5`）、`1.0000000000000122`
   （`N1e|0.5`）、`1.0000000000000027`（`N3|0.5`）、`0.9999999999999778`（`N1e|-0.05`）
   ⇒ 失败可**完全归因**于地相质量再分配，线性化**没有被破坏**。
5. **两个杠杆都在，但没有单点同时达标。** G1 有 `6` 个可用点通过、
   G5b 有 `2` 个可用点通过，**两集合不相交** ⇒ `verdict.json::matched` 里
   `LEVEL_AND_AMPLITUDE_DO_NOT_COINCIDE` 是**唯一**为真的一格
   （`also_matched = []`）。承载装置点上最好的一格是 `N1e|0.5`：
   过 **3/5**（`G1 G2 G3`），G5 与 G5b 均不过。
6. **G4 的证伪器读数是"主张成立"**：$\lvert\Delta\alpha_{\rm hat}\rvert$ 在
   `N1|0.5` 为 `0.0015797186802644497`、`N1e|0.5` 为 `0.0018563800342498854`，
   **均小于翻门阈值 `0.004881`** ⇒ G4 在 $\beta=+0.5$ **仍失败**，
   "逐河段常数对 $\alpha$ 的一阶效应为零"这一论证**没有被证伪**。
   它**不进入主判决**（预注册 §4.2 与 §7 红线），但**必须报出**。
7. **C1 的 `PROCEED` 读不出"前提成立"。** 基线 NSE 已经 `0.7029748157444711`，
   **高于** `proceed_at_or_above = 0.65`，所以"停止规则触发否"这一问在基线一侧
   就已经被决定；纯水平标度只买到 `0.001930842454908266`。
   `phase_minus1.json::C1.cannot_falsify_because_base_above_high = true`。
   ⇒ **不得**把 `PROCEED` 读成"崩塌是水平效应"。
8. **12 个点按字面读会判 `BLOCKED`。** 全部落在 $\beta\in\{+4,-8,+8\}$、
   全部 `FIXED_POINT_NOT_CONVERGED`。预注册 §11.4 **在任何 Phase 1 前向之前**登记了
   §3 与 §7 两句话在这个网格上的冲突，并取**逐点**读法；
   `outcome_under_literal_reading = "BLOCKED"` 一并装运，**不隐瞒**。
   被划掉的 12 个点**本来就在 `usable()` 之外**，没有任何门槛被改动或移除。

### 19.3 三处**必须随行**的限定（否则本行会被读宽）

1. **$k_r$ 是"派生量"，不是参数**，但它**携带建模假设**
   （"长期可动员质量是该保持的不变量"）。动用 **1 dof／河段**；
   必须与 round 4 行动行的 `may_add_exchange_parameters = 1` 并列读，
   否则 230 个派生常数会被读成 230 个新参数。
2. **"零拟合"带一个限定**：$av^0$ 来自冻结的 30 维向量
   （`20260916_2\outputs\C0_s1\model.json`），其
   `design.training_years = [2021, 2022]`——**落在评估窗口内**。
   本轮**没有**跑任何拟合，但这句话不得被读成"与评估期无关"。
3. **$h\Xi$ 的近线性区是实测的，不是假定的**：
   `frac(h*Xi > 1)` 在全部日上为 `0.0013344089584964485`、
   在事件日上为 `0.0008263355368947651` ⇒ P1 的前置条件成立。
   若它不成立，P1 必须报成"无支撑"而不是被引用。

### 19.4 本行的裁决：**`装置成立、假设不成立`（装置层面）**

* **装置成立**：水平**有**杠杆、振幅**有**杠杆，两者都**实测到**了（§19.2 第 2、5 条），
  C7 伴随把归因**量化到 `ΔC̄` 绝对误差 `7.105427357601002e-15` 以内**（第 4 条）——
  这是 72 个非退化点上的**最坏**值，72 个点的 `linearisation_holds` **全为真**；
  $\beta=0$ 的比值为 `null`，因为变化量恰为零。这是本轮的**正面**结果。
* **假设不成立**：被检验的假设是"**把水平拿掉之后，存在一个设置同时满足 G1 与 G5b**"
  （等价地：round 4 的月尺度崩塌是水平效应、中性化能修复它）。
  实测**两者都不成立**：没有单点同时过（第 5 条），且去水平只恢复
  恢复份额 **1.5%**（第 3 条）。
* **取值仍是四值之一**（§19.4 表头那句"没有第五种"仍然有效）：
  `装置成立`（装置确实按设计工作）＋`假设不成立`（"中性化足以解耦到单点全过"不成立）。
  **本节没有新增第五种取值。**
* **不得**读成另外三个取值：不是 `CLOSED`（机制族**没有被关闭**，
  且预注册 §7 对本格明写"本格不单独关闭陆相内部结构这根轴"）；
  不是 `未裁决`（预注册 **有**这一格，它在跑任何前向之前写入，
  这正是它与第 14 行的区别）；**更不是**"机制被否定"（§19.2 第 2 条方向相反）。
* **独立复算**：`work\audit_beta.py` **不 import 本轮任何模块**，
  自己重写陆相递推，并把**两条**求解路线（闭式与自洽不动点）各自独立实现，
  在 12 个点上与注册 $k$ 场对拍（闭式子集 `3.638981156991526e-16`、不动点子集 `8.1e-11`，
  整网格数只报不用）。审计**自己**的两条缺陷（括号方向写反、拿一条方程对两条路线）
  **已登记**，**产物一个字节都没动**。

### 19.5 本行的重开条件

$$\boxed{\textbf{一次真双路径灰箱（}C_f,C_s\textbf{），且必须先解释"月尺度崩塌是方差效应"。}}$$

* **重开必须回答两问**（本轮明确不答）：
  (a) 月尺度崩塌既然是**方差**效应，它在**真双路径**结构里是被解释的、
      还是被搬到了另一个量上？
  (b) $k_r$ 的**跨河段变异**同时供给该装置全部的比值损伤与全部的水平控制力
      （§2.8 的根本张力）——真双路径结构能不能把这两件事**分开**？
* **升级形式**：fast/slow 各自带可达浓度的状态方程，**最多再加 1 个交换参数**。
* **本轮不授权重新拟合**：预注册 §7 只把重新拟合挂在 `DECOUPLING_DEMONSTRATED` 上。
* **不允许的重开理由**：把 `LEVEL_AND_AMPLITUDE_DO_NOT_COINCIDE` 读成
  `LEVEL_NOT_THE_CAUSE`（后者才关闭本轴，而本格**明写不关闭**）；
  把 §19.2 第 2 条读成"机制已被证实"；把 §19.2 第 3 条读成"月尺度问题已定位"。

### 19.6 与第 14 行、第 13 行的关系（**三行不得互相援引**）

* 第 13 行动 $f$（**后**分配），第 14 行动纯乘性 $p$（**动员前**），
  第 15 行动**质量中性**的 $p$（**动员前 + 水平被压住**）。三者是**三个装置**。
* **第 15 行不解除第 14 行的重开条件**，也**不满足**它：
  §18.5 要求"一次真双路径灰箱，且必须先过 G4"，本轮 G4 **仍失败**（§19.2 第 6 条）
  ⇒ §18.5 **原样保留**。
* **第 15 行也不解除 §16.3**：两条轴的装置不同（一个动 $f$，一个动 $p$）。
* 第 14 行的裁决**不变**（仍是 `未裁决 —— 预注册缺口`）；本轮只更正它的 **G5 读数**，
  登记在**已有的 §18.8**，本节**不重复**也不重新论证。

### 19.7 本行的**外延界线**（与 §19.3 同等重要，缺一不可）

1. **不得**读成"动员前路径特异浓度不存在"——**实测方向相反**：
   把长期水平压住之后事件振幅的收益**仍在**（§19.2 第 2 条）。
   这是本行**唯一**一条**加强**第 14 行的读数，而且它是**正面**的。
2. **不得**读成"陆相内部结构这根轴被关闭"。预注册 §7 对本格的原话是
   "**本格不单独关闭陆相内部结构这根轴**"，行动行**不得**写得比
   round 4 §0.3 的收窄版**更宽**。⇒ **本轮不写"转向额外事件 N 源"的行动行**，
   那一条只属于 `LEVEL_NOT_THE_CAUSE` 与 `INSUFFICIENT` 两格，**本格不是它们**。
3. **不得**读成"中性化失败"。第 3 条（§19.2）说的是
   **装置太弱还是靶不对**——这一问**本轮明确不答**，留 §19.5 的 (a)(b)。
4. **12 个点按字面读会判 `BLOCKED`**，但那 12 个点**本来就在 `usable()` 之外**；
   两句话的冲突写在预注册 §11.4，**取逐点读法是在任何前向之前就写死的**。
   ⇒ **不得**把 `outcome_under_literal_reading = "BLOCKED"` 当作本行的反证，
   也**不得**把它藏起来不讲。
5. **C2 触发 ⇒ 主装置改判为 `N1e`**，这是**在跑任何前向之前**写死的
   （预注册 §11.1–§11.2、§0.2 红线）。**不得**把它读成事后挑装置。
6. **量程限制**：`15` 个合格站点只覆盖 `13` 个河段（与第 14 行的 `4` 个河段不同口径），
   站点级判据的独立信息量**低于**"15 个独立样本"。
7. **G4 的失败不进主判决，但必须报出**（§19.2 第 6 条）：
   它的证伪器读数是"主张成立"，**不是**"G4 被解决"。
8. **`N2` 的读数没有进入任何门的通过判定**（预注册红线），
   `N1` 与 `N2` 被排除在每一门之外（`verdict.json::devices_excluded_from_every_gate`）。
"""

APPENDIX_TAIL = r"""
21. **第 15 行的"装置成立"与第 13 行的"装置成立"不是同一件事。**
    第 13 行说的是"这个干预形式是活的、可以被检验"；
    第 15 行说的是"**水平与振幅两把杠杆都被实测到**"。
    两者都**不**等于"机制被证实"，也都**不**等于"机制不存在"。
22. **第 15 行的裁决取值落在"装置层面"，不得被搬到"机制层面"。**
    本行的假设是"**中性化足以解耦到单点全过**"，这一条**不成立**；
    它**没有**检验"路径浓度结构存不存在"——而且它的读数（§19.2 第 2 条）
    **支持**后者存在。**裁决的层级写错，就是本台账最容易被读坏的一次。**
23. **第 15 行对第 14 行只做了一件加强、一件更正，没有做裁决。**
    加强：振幅收益不是水平效应（§19.2 第 2 条）；
    更正：第 14 行的 G5 读数（登记在已有的 §18.8，**不是本轮新写**）。
    第 14 行的取值**不变**，§18.5 的重开条件**不变**。
24. **第 15 行的两条预测 P1／P2 的判读必须分开写。**
    P1（$A_{L1}$ 落在 `1.2289037922001007 ± 0.025`）报出实测值由读者对区间；
    P2（`A_{L3}` 区间横跨 `G2_target`）在 `N1e|0.5` 上的实测是
    $A_{L3}$ = `1.1602894255651304` ≥ `1.1495455668222099` ⇒ G2 **过**，
    但那**不是**"硬币落地"的证明，因为承载装置已由 C2 改判为评估窗口。
25. **C1 的 `PROCEED` 不得被任何一行引用为证据。**
    它在基线一侧就被决定（`cannot_falsify_because_base_above_high = true`），
    对应的停止规则分支（`NSE(δ*) < 0.55`）在本轮**没有被触发**，
    但**也没有被否证**——它是一个**读不出"否"**的门，见预注册 §11.3。
26. **`k_r` 的冻结顺序是本行的可核性来源，不是一个流程描述。**
    $k$ 场在读任何观测之前落盘并哈希，且哈希**从磁盘读回的值重算**
    （`frozen_anchors.json::k_field_sha.matches = true`）。
    ⇒ "先冻结后读观测"这句话**可以被独立复核**，不必信任叙述。
27. **本节各条与 §18 各条不得互相援引为证据**；本节与 §19.2 也**不得**被用来支撑
    "陆相内部结构已被穷尽"——那一句只属于 `INSUFFICIENT` 一格，
    而本轮**不是** `INSUFFICIENT`（G1 有 `6` 个可用点通过）。
"""

if __name__ == '__main__':
    main()
