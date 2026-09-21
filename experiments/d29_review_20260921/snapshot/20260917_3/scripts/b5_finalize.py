"""B5: close round B -- reconcile the product against what was asked, and audit the deliverables.

Round B delivers a *product*, not an answer: the answer was round A's (74 stations retained, zero
negative).  So the reconciliation here is a product reconciliation, and the three things it has to
establish are:

  1. **The product is the screened product.**  Round A froze a station screen and this round built the
     1961-2025 network from it.  That claim can fail in a way that looks fine -- a chain that silently
     used a different panel produces a product of exactly the right shape -- so it is not asserted:
     `b2` reproduces every artifact against the very arm the screen was frozen on, and this report
     carries that result, including the artifacts that came out equal-but-not-byte-identical and why.

  2. **The claim lands on the right cohort.**  Zero-negative is a statement about the 74 retained
     stations over 2019-2022, and about nothing else.  The full 102-station base is reported *beside*
     it and is not a success claim; a screened cohort's score is not comparable to an unscreened one,
     which is S111's own lesson and the reason its declaration is carried verbatim.

  3. **The cost is carried.**  The screen was chosen after seeing the measurements, and it was not
     free: the same 74 stations score worse under the refit than under the pre-screen model by a small
     but nonzero margin.  That number travels with the product (`b3` puts it in the product lock, this
     report puts it in prose) rather than being left in the round that measured it.

The deliverable audit is `a5`'s: every promise this round made, its path, its sha256, whether it is on
disk, and whether it *checks out* -- a file can be present and still not delivered, so `missing` is
keyed on the check and not on existence.  The two files this script is in the middle of writing are
flagged rather than dropped.

Writes `reports/b5_reconciliation.json`, `reports/completion_audit.json` and
`reports/round_report_20260917_3.md`.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


T = Path(r"E:\SPARROW") / "5_Test"
RUN = T / "20260917_3"
A = T / "20260917_2"          # round A: read-only

OUT, REPORTS, LOCKS = RUN / "outputs", RUN / "reports", RUN / "locks"
DECISIONS = RUN / "evaluation_reports"

B0 = REPORTS / "b0_preflight.json"
B1 = REPORTS / "b1_run.json"
B2 = REPORTS / "b2_reproduction.json"
B4 = REPORTS / "b4_adjudication.json"
DEVIATIONS = REPORTS / "b6_deviations.json"
DECISION = DECISIONS / "canonical_hydrology_decision.json"
PRODUCT_LOCK = LOCKS / "product_lock.json"
SCREEN_LOCK = A / "locks" / "station_screen_lock.json"
# Published as chain evidence, so its own bytes are hash-verified before it is trusted here.
SPINUP_LOCK = LOCKS / "stage33" / "locks" / "frozen_reproduction_spinup_lock.json"

S111_CAVEAT = (
    "本筛选是事后的，两期都参与分组；不能支持无偏外推，不能替代全队列基线；"
    "同群体保护效应与零负目标必须分开报告。"
)

STANDING_FACTS = [
    "结论期是 2006–2022（观测档案所及）；1961–2005 无实测流量，"
    "“无站 NSE 为负”因此是 2006–2022 的陈述。",
    "2025 的 PET 桥接未过 0.98 门（r = 0.916165），2025 段携带 PET_BRIDGE_UNVALIDATED_2025。",
    "本产品不替换、不冒充正式产品 20260828_35。",
    "本轮的排除是按表现的（负 NSE），按用户指示作出；不得被叙述成 QC 排除。",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------- deliverable validators

def product_check(path: Path) -> tuple[bool, str]:
    """A network product checks out if it covers the whole network and the whole window.

    Shape alone would pass a product built on a truncated panel -- 230 reaches and 1961-2025 are the
    two facts that say the screen did not narrow the *product*, only the claim.
    """
    meta = pq.ParquetFile(path).metadata
    rows = meta.num_rows
    columns = meta.schema.to_arrow_schema().names
    if "reach_id" not in columns:
        # Reached by running it: the reservoir tables are keyed by `reservoir_entity_id`, so asking a
        # network question of them raises inside Arrow.  A validator that crashes on the wrong table
        # is not answering the question, so it says which table it cannot read instead.
        return False, f"{rows:,} 行，但表里没有 reach_id —— 这不是河网产品表"
    # `unique`/`min_max` in Arrow rather than Python sets and lists: the daily product is 5.4M rows and
    # materialising it as Python objects is a needless copy of the thing under audit.
    reaches = len(pc.unique(pq.read_table(path, columns=["reach_id"]).column("reach_id")))
    window = ""
    for key in ("date", "year", "month"):
        if key not in columns:
            continue
        values = pq.read_table(path, columns=[key]).column(key)
        if pa.types.is_timestamp(values.type) or pa.types.is_integer(values.type):
            lo, hi = pc.min_max(values).values()
            window = f"{key} {lo}..{hi}"
        break
    good = reaches == 230
    return good, f"{rows:,} 行、{reaches} 河段、{window}"


def reservoir_check(path: Path) -> tuple[bool, str]:
    """Reservoir tables, keyed by `reservoir_entity_id` rather than by reach.

    The static metadata must be exactly one row per reservoir: a duplicated entity would silently be
    counted twice by whatever joins it later, which is a defect that shows up as a plausible number.
    """
    meta = pq.ParquetFile(path).metadata
    rows = meta.num_rows
    columns = meta.schema.to_arrow_schema().names
    if "reservoir_entity_id" not in columns:
        return False, f"{rows:,} 行，但没有 reservoir_entity_id —— 这不是水库表"
    entities = len(pc.unique(
        pq.read_table(path, columns=["reservoir_entity_id"]).column("reservoir_entity_id")))
    static = not ({"date", "month"} & set(columns))
    if static:
        return rows == entities and entities > 0, f"{rows:,} 行 = {entities} 座水库（静态表一行一库）"
    return rows > 0 and entities > 0, f"{rows:,} 行、{entities} 座水库"


def decision_check(path: Path) -> tuple[bool, str]:
    data = read_json(path)
    failed = sorted(key for key, value in (data.get("checks") or {}).items() if not value)
    ok = (str(data.get("status", "")).startswith("ZERO_NEGATIVE") and not failed)
    return ok, f"status={data.get('status')}；自检 {len(data.get('checks') or {})} 项，假 {len(failed)}"


def passed_check(path: Path) -> tuple[bool, str]:
    """Validator for this round's own step reports: the report must say it passed."""
    data = read_json(path)
    return bool(data.get("passed")), ("通过" if data.get("passed") else "报告自称未通过")


def reproduction_check(path: Path) -> tuple[bool, str]:
    """A reproduction report only discharges its promise if it compared something and found no
    mismatches -- `passed` alone would be true of a comparison of nothing."""
    data = read_json(path)
    counts = data.get("counts") or {}
    compared = int(counts.get("artifacts_compared", 0))
    ok = bool(data.get("passed")) and compared > 0
    return ok, (f"比对 {compared} 件：逐字节 {counts.get('byte_identical')}、"
                f"内容相同 {counts.get('equal_but_not_byte_identical')}、"
                f"不一致 {counts.get('differs')}、无读取器 {counts.get('uncomparable')}、"
                f"单侧 {counts.get('present_in_one_tree_only')}")


def run_check(path: Path) -> tuple[bool, str]:
    data = read_json(path)
    products = data.get("products") or []
    corrections = data.get("label_corrections") or []
    ok = len(products) > 0 and len(corrections) == 1
    return ok, (f"产物 {len(products)} 件；标签订正 {len(corrections)} 处"
                + (f"（{corrections[0]['inherited']} → {corrections[0]['corrected']}）"
                   if corrections else ""))


def chain_evidence_check(path: Path) -> tuple[bool, str]:
    """Hold the product lock's own chain-evidence claims against the disk.

    `b3` publishes 13 files beyond the product -- the chain's locks and QA reports -- under
    `locks/<relative>` and `reports/chain/<relative>`, and the lock records a hash for each.  Presence
    of the lock proves nothing about them, so this walks the lock's own list and re-hashes every
    destination: a round that publishes its evidence and never looks at it again has an evidence chain
    only in the sense that it has a list.
    """
    data = read_json(path)
    chain = data.get("chain_evidence") or {}
    if not chain:
        return False, "产品锁件里没有链证据清单"
    absent, altered = [], []
    for relative, record in sorted(chain.items()):
        published = Path(record.get("published", ""))
        if not published.is_file():
            absent.append(relative)
        elif sha256(published) != record.get("sha256"):
            altered.append(relative)
    ok = not absent and not altered
    detail = f"链证据 {len(chain)} 件：缺失 {len(absent)}、哈希不符 {len(altered)}"
    if absent:
        detail += f"；缺失={absent}"
    if altered:
        detail += f"；哈希不符={altered}"
    return ok, detail


def initial_state_check(path: Path) -> tuple[bool, str]:
    """The two 1961-01-01 initial states, held against the digest the spin-up fork recorded for them.

    They were the audit's last present-only row, and present-only is the wrong answer for them: one is a
    compressed array blob and the other a JSON state, neither of which carries a shape a reader here
    could test, so "on disk" would report a truncated or superseded file as delivered.  The producer
    hashes both into its own lock at production time and that lock is itself published and
    hash-verified as chain evidence, so the claim these files make -- "this is the state the 1961-2025
    simulation started from" -- is checkable, and this is the check: that recorded digest, this file.
    """
    key = {
        "historical_initial_land_state_1961.npz": "land_initial_state",
        "historical_initial_reservoir_states_1961.json": "reservoir_initial_state",
    }.get(path.name)
    if key is None:
        return False, f"没有为 {path.name} 定义摘要键"
    if not SPINUP_LOCK.is_file():
        # Checked rather than left to raise: the audit's blanket `except` would absorb a
        # FileNotFoundError into a pass-shaped "校验失败", which is the right verdict but a worse
        # message than saying which lock went missing -- and a missing lock is a distinct failure from
        # a digest that disagrees.
        return False, f"自旋锁件不在盘上：{SPINUP_LOCK}"
    recorded = (read_json(SPINUP_LOCK).get("files") or {}).get(key)
    if not recorded:
        return False, f"自旋锁件里没有 {key} 的摘要"
    actual = sha256(path)
    if recorded == actual:
        return True, f"与自旋锁件所记 {key} 摘要一致（{actual[:16]}…）"
    return False, f"与自旋锁件所记摘要不符：锁件 {recorded[:16]}…、盘上 {actual[:16]}…"


def product_lock_check(path: Path) -> tuple[bool, str]:
    """The published lock's own claim, re-verified against the screen it cites.

    Its row was existence only, and the chain-evidence row checks the lock's *list of evidence* rather
    than the claim that list is evidence for.  What is checkable is the claim's provenance: the lock
    quotes a gate, a base and a screen file, and all three must be the ones round A froze.  A published
    lock that quoted different numbers than the screen it points at -- the one way a product can carry a
    correct-looking headline claim and not be the screened product -- would otherwise pass every check
    in this audit.
    """
    data = read_json(path)
    frozen = read_json(SCREEN_LOCK)
    cited = data.get("screen") or {}
    base = frozen.get("base") or {}
    problems = []
    if cited.get("gate") != frozen.get("gate"):
        problems.append("gate 与筛选锁件不符")
    for key, frozen_key in (("base", "evaluable"), ("retained", "retained"), ("excluded", "excluded")):
        if cited.get(key) != base.get(frozen_key):
            problems.append(f"{key}={cited.get(key)} 与筛选锁件的 {frozen_key}={base.get(frozen_key)} 不符")
    parquet = cited.get("screen_parquet") or {}
    if not parquet.get("path"):
        problems.append("没有引用筛选件")
    elif not Path(parquet["path"]).is_file():
        problems.append(f"所引筛选件不在盘上：{parquet['path']}")
    elif sha256(Path(parquet["path"])) != parquet.get("sha256"):
        problems.append("所引筛选件的摘要与锁件所记不符")
    if problems:
        return False, "；".join(problems)
    return True, (f"gate 与筛选锁件一致；基座 {cited.get('base')} → 保留 {cited.get('retained')}、"
                  f"排除 {cited.get('excluded')}；所引筛选件摘要相符")


def screen_check(path: Path) -> tuple[bool, str]:
    data = read_json(path)
    gate = data.get("gate") or {}
    ok = bool(gate.get("zero_negative_on_retained")) and int(gate.get("retained_negative", -1)) == 0
    return ok, (f"组合 {data.get('combination')}；保留 {data['base']['retained']}、"
                f"排除 {data['base']['excluded']}；零负门"
                + ("通过" if gate.get("zero_negative_on_retained") else "未通过"))


# --------------------------------------------------------------------------------------- reconciliation

def reconciliation(b4: dict) -> dict:
    lock = read_json(SCREEN_LOCK)
    protection = lock["same_population_protection"]
    b2 = read_json(B2)
    cohorts = b4["cohorts"]
    return {
        "stage": "20260917_3",
        "purpose": "B5: 把产品与它被要求回答的问题对起来，并审计本轮交付",
        "question": (
            "按 20260917_2 冻结的筛选件（基座 102 站中保留 74、排除 28）重跑 1961-2025 全河网产品，"
            "零额外调参；零负声明落在保留队列 Q 上，基座全队列 F 作为基线并报。"
        ),
        "screen": {
            "round": "20260917_2",
            "combination": lock["combination"],
            "origin": lock["origin"],
            "is_post_hoc": lock.get("post_hoc", {}).get("is_post_hoc"),
            "base": lock["base"]["evaluable"],
            "retained": lock["base"]["retained"],
            "excluded": lock["base"]["excluded"],
            "gate": lock["gate"],
            "lock_sha256": sha256(SCREEN_LOCK),
        },
        "product": {
            "reaches": 230,
            "window": "1961-01-01..2025-12-31",
            "scales": ["daily", "monthly"],
            "evaluated_daily": b4.get("canonical_daily"),
            "evaluated_daily_sha256": b4.get("canonical_daily_sha256"),
            "evaluated_monthly": b4.get("canonical_monthly"),
            "evaluated_monthly_sha256": b4.get("canonical_monthly_sha256"),
            "claim_window": b4.get("claim", {}).get("claim_window"),
        },
        "cohorts": {
            "Q": {
                "definition": "筛选件保留的 74 站 —— 达标声明落在这一队列",
                "stations": cohorts["Q"]["stations"],
                "daily_negative": cohorts["Q"]["daily_negative"],
                "monthly_negative": cohorts["Q"]["monthly_negative"],
                "worst_retained_nse": float(lock["gate"]["worst_retained_nse"]),
                "zero_negative": (cohorts["Q"]["daily_negative"] == 0
                                  and cohorts["Q"]["monthly_negative"] == 0),
            },
            "F": {
                "definition": "可评估基座全队列 102 站 —— 基线，不作达标声明",
                "stations": cohorts["F"]["stations"],
                "daily_negative": cohorts["F"]["daily_negative"],
                "monthly_negative": cohorts["F"]["monthly_negative"],
                "why_reported": (
                    "筛选队列的分数不与未筛选队列可比；F 并报是为了把筛选的代价与收益分开看，"
                    "不是为了给 F 争一个达标"
                ),
            },
        },
        "reproduction": {
            "counts": b2.get("counts"),
            "passed": b2.get("passed"),
            "why_it_matters": (
                "形状正确的产品也可能建在另一个面板上；复现是唯一能排斥这一点的检查 —— "
                "本轮产物逐件与筛选锁件所依据的那一臂比对"
            ),
        },
        "same_population_protection": protection,
        "s111_declaration": S111_CAVEAT,
        "standing_facts": STANDING_FACTS,
    }


# ------------------------------------------------------------------------------------------ the audit

def completion_audit(recon: dict) -> dict:
    promised = [
        ("筛选件（次 A 冻结，本轮只读）", SCREEN_LOCK, False, screen_check),
        ("筛选件与基座哈希预检", B0, False, passed_check),
        ("目标面板的产品构建（步骤 01–06）", B1, False, run_check),
        ("产品复现核对（本轮 vs 被冻结臂）", B2, False, reproduction_check),
        ("1961-2025 全河网日产品", OUT / "tn_hydrology_reach_daily.parquet", False, product_check),
        ("1961-2025 全河网月产品", OUT / "tn_hydrology_reach_monthly.parquet", False, product_check),
        ("水库日产品", OUT / "tn_hydrology_reservoir_daily.parquet", False, reservoir_check),
        ("水库月产品", OUT / "tn_hydrology_reservoir_monthly.parquet", False, reservoir_check),
        ("水库静态元数据", OUT / "tn_hydrology_reservoir_static_metadata.parquet", False, reservoir_check),
        ("状态自洽日产品", OUT / "state_consistent_reach_daily.parquet", False, product_check),
        ("状态自洽月产品", OUT / "state_consistent_reach_monthly.parquet", False, product_check),
        ("状态自洽水库日产品", OUT / "state_consistent_reservoir_daily.parquet", False, reservoir_check),
        ("状态自洽水库月产品", OUT / "state_consistent_reservoir_monthly.parquet", False, reservoir_check),
        ("状态自洽水库静态元数据", OUT / "state_consistent_reservoir_static_metadata.parquet", False, reservoir_check),
        ("规范化导出（日，2006-2024）", OUT / "canonical_reach_daily_2006_2024.parquet", False, product_check),
        ("规范化导出（月，2006-2024）", OUT / "canonical_reach_monthly_2006_2024.parquet", False, product_check),
        ("1961 初始陆面状态", OUT / "historical_initial_land_state_1961.npz", False, initial_state_check),
        ("1961 初始水库状态", OUT / "historical_initial_reservoir_states_1961.json", False, initial_state_check),
        ("裁决件（Q 与 F 并报）", DECISION, False, decision_check),
        ("产品锁件", PRODUCT_LOCK, False, product_lock_check),
        ("链证据（发布后逐件回核哈希）", PRODUCT_LOCK, False, chain_evidence_check),
        ("偏离登记", DEVIATIONS, False, passed_check),
        ("本轮对账", REPORTS / "b5_reconciliation.json", True, None),
        ("轮次报告", REPORTS / "round_report_20260917_3.md", True, None),
    ]
    entries, missing = [], []
    for name, path, self_produced, validator in promised:
        exists = path.is_file()
        if not exists:
            satisfied, check = False, "文件不在盘上"
        elif validator is None:
            satisfied, check = True, "在盘"
        else:
            try:
                satisfied, check = validator(path)
            except Exception as error:                       # noqa: BLE001 - any failure is a failed check
                satisfied, check = False, f"校验失败 {type(error).__name__}: {error}"
        entries.append({
            "deliverable": name, "path": str(path), "present": exists,
            "satisfied": satisfied, "check": check,
            "sha256": sha256(path) if exists else None,
            "bytes": path.stat().st_size if exists else None,
            "self_produced_by_this_script": self_produced,
        })
        if not satisfied and not self_produced:
            missing.append(name)
    # The screen says 74; if the product's own cohort fell short of it, the deliverable is present and
    # wrong, which the per-file checks above cannot see.
    if int(recon["cohorts"]["Q"]["stations"]) != int(recon["screen"]["retained"]):
        missing.append("保留队列的站数与筛选件不符")
    return {
        "stage": "20260917_3",
        "deliverables": entries,
        "delivered": sum(1 for entry in entries if entry["satisfied"]),
        "on_disk": sum(1 for entry in entries if entry["present"]),
        "promised": len(entries),
        "missing": missing,
        "unsatisfied": [{"deliverable": entry["deliverable"], "check": entry["check"]}
                        for entry in entries if entry["present"] and not entry["satisfied"]],
        "self_produced_now": [entry["deliverable"] for entry in entries
                              if entry["self_produced_by_this_script"]],
        "complete": not missing,
    }


# ------------------------------------------------------------------------------------- the round report

def round_report(recon: dict, audit: dict, b4: dict) -> str:
    lock = read_json(SCREEN_LOCK)
    protection = lock["same_population_protection"]
    b2_counts = recon["reproduction"]["counts"] or {}
    q, f = recon["cohorts"]["Q"], recon["cohorts"]["F"]
    lines = [
        "# 20260917_3 — 1961–2025 全河网水文产品：轮次报告",
        "",
        "## 交付了什么",
        "",
        "按 `20260917_2` 冻结的筛选件（基座 102 站中保留 74、排除 28）**重跑**的 "
        "1961–2025、230 河段、日与月水文产品。零额外调参：配置逐字继承自被冻结的那一臂，"
        "只有 stage 标签是本轮的。",
        "",
        "## 达标声明落在哪里",
        "",
        f"**Q 队列（保留的 {q['stations']} 站）**：2019–2022 日负站 "
        f"{q['daily_negative']}、月负站 {q['monthly_negative']}"
        + ("—— 零负成立。" if q["zero_negative"] else "—— **零负不成立**。"),
        f"保留站最负 NSE = {lock['gate']['worst_retained_nse']:.6f}（零负门通过）。",
        "",
        f"**F 队列（基座全 {f['stations']} 站）**：日负站 {f['daily_negative']}、"
        f"月负站 {f['monthly_negative']}。**这不是一个达标数字**，列在这里是因为筛选队列的分数"
        "不与未筛选队列可比；把两者并报才能把筛选的代价与收益分开看。",
        "",
        "## 为什么这个产品是筛选件的产品",
        "",
        "形状正确的产品也可能是建在另一个面板上的 —— 230 河段、1961–2025 这两个事实对面板"
        "一无所言。所以本轮把产物逐件与筛选锁件所依据的那一臂对了一遍：",
        "",
        f"- 比对 {b2_counts.get('artifacts_compared')} 件：逐字节相同 "
        f"{b2_counts.get('byte_identical')} 件、内容相同而字节不同 "
        f"{b2_counts.get('equal_but_not_byte_identical')} 件、不一致 "
        f"{b2_counts.get('differs')} 件、只在一棵树里有 {b2_counts.get('present_in_one_tree_only')} 件。",
        f"- 判定：`passed = {recon['reproduction']['passed']}`。",
        "",
        "`.pt` 检查点与 JSON 会记录自己写在哪个目录下，字节必然不同，故按张量键与逐元素相等比较，"
        "字符串条目（通常是路径）单独列出；没有把任何一件「跳过」地算作一致。",
        "",
        "## 筛选的来源（必须随件携带）",
        "",
        f"- 组合 `{lock['combination']}`，来源 `{lock['origin']}`，"
        f"**事后自适应**（`is_post_hoc = {lock.get('post_hoc', {}).get('is_post_hoc')}`）。",
        f"- 基座 {lock['base']['evaluable']} 站 → 保留 {lock['base']['retained']}、"
        f"排除 {lock['base']['excluded']}；被排除而原本非负的站："
        f"{lock['gate']['excluded_that_were_positive']} 个。",
        "",
        "### 同群体保护代价",
        "",
        f"同一批保留站上，只筛选 0.6253 → 筛选**加重拟合** 0.6199"
        f"（Δ {protection['delta_mean_nse_min']:+.4f}；变好 {protection['stations_improved']} 站、"
        f"变差 {protection['stations_worsened']} 站）。",
        "",
        "即：重拟合让这批站在平均意义上**变差了一点**，与 S111 当年写下的教训同形"
        "（它记的是 −0.01084），只是小得多。零负目标的达成不是免费的，这个代价必须与目标分开报。",
        "",
        "## 随件事实与边界",
        "",
        f"**S111 式声明（逐字照搬其边界）**：{S111_CAVEAT}",
        "",
    ]
    lines += [f"- {fact}" for fact in STANDING_FACTS]
    lines += [
        "",
        "## 与本轮次 A 的对账",
        "",
        "本轮的筛选件、基座与臂都不是重新选的，而是次 A 冻结的那一份；"
        "本轮的 Q 队列数字与次 A 的 `R74` 臂结果**逐字段相等**（见裁决件的 "
        "`reconciliation_with_round_a`）。两者不是同一份计算的两次复述：次 A 的臂是产生产品的那一次运行，"
        "本轮裁决是重新读产品再算一遍，两处一致才说明产品确实是那一臂的产品。",
        "",
        f"- 保留 {lock['base']['retained']} 站、零负、被排除而本为正 0 个 —— 三处一致。",
        "",
        "## 交付完整性",
        "",
        f"承诺 {audit['promised']} 件，在盘 {audit['on_disk']} 件，其中检验通过 {audit['delivered']} 件。",
        "",
        f"逐件哈希与逐件判定见 `{REPORTS / 'completion_audit.json'}`。",
        "",
        "## 偏离与只读声明",
        "",
        f"`{DEVIATIONS}`：本轮与冻结轮的每一处差异、理由与授权。",
        "",
        "本轮只读次 A 与全部冻结轮；未改动其中任何文件。产品的 stage 标签与继承值记在 "
        "`work/screen/experiment_contract.json` 与 `work/screen/work/locks/forcing_integrity_lock.json` 里。",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    for path, why in ((B2, "先跑 b2_verify_reproduction.py"), (B4, "先跑 b4_adjudicate.py"),
                      (B0, "先跑 b0_preflight.py"), (B1, "先跑 b1_run_product.py")):
        if not path.is_file():
            raise RuntimeError(f"{path} 不在盘上；{why}")

    b4 = read_json(B4)
    recon = reconciliation(b4)

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "b5_reconciliation.json").write_text(
        json.dumps(recon, ensure_ascii=False, indent=2), encoding="utf-8")

    audit = completion_audit(recon)
    (REPORTS / "completion_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    report = round_report(recon, audit, b4)
    (REPORTS / "round_report_20260917_3.md").write_text(report, encoding="utf-8")

    q, f = recon["cohorts"]["Q"], recon["cohorts"]["F"]
    print("B5 收尾")
    print(f"  Q（{q['stations']} 站）：日负站 {q['daily_negative']}、月负站 {q['monthly_negative']}")
    print(f"  F（{f['stations']} 站）：日负站 {f['daily_negative']}、月负站 {f['monthly_negative']}")
    print(f"  复现：{recon['reproduction']['counts']}")
    print(f"  交付：承诺 {audit['promised']}、在盘 {audit['on_disk']}、通过 {audit['delivered']}"
          f"{'（完整）' if audit['complete'] else ''}")
    if not audit["complete"]:
        print("  未交付：")
        for item in audit["missing"]:
            print(f"    - {item}")
        raise SystemExit(1)
    print(f"  轮次报告 → {REPORTS / 'round_report_20260917_3.md'}")


if __name__ == "__main__":
    main()
