"""B6: round B's deviation register -- every place this round does not do exactly what the frozen
rounds (and round A) do, why, and by whose authority.

Round B's deviations are a different species from round A's.  Round A changed *code*: it parameterised
forks, guarded entry points and rewrote a loop that had never been entered.  Round B changes nothing --
it re-runs round A's chain, byte for byte, by importing the driver as a module and rebinding its
module-level roots.  That is deliberate and it is the reason this register exists: a round that reports
"no deviations" while running someone else's code under a different set of paths has not reported
anything.  What round B must disclose is the *scaffolding*:

  * the driver is executed unmodified, so its sha256 still matches the one round A's screen lock
    records -- which is what makes the screen reproducible from its own lock;
  * the forks are round A's bytes, copied and hash-verified rather than re-derived;
  * the experiment contract is round A's R74 contract with only its `stage` label restated, written
    *before* step 01 because the frozen stage-5 fork `stat()`s it without an existence guard;
  * one label the driver writes is simply false for this round and is corrected, with both values kept;
  * the work tree was cleared after the first step-01 attempt died, so no pre-failure artifact can be
    read as the re-run's own;
  * and the round inherits round A's policy departure wholesale -- the exclusions here are
    performance-based, and that is not a QC exclusion no matter how many times it is repeated.

Each entry is one of: a user instruction, an engineering necessity discovered while building the round,
a known artifact (disclosed so it is not read as intended), or a disclosure that narrows a claim.
`verify()` re-checks everything checkable from the two rounds' own artifacts, so a stale entry fails
loudly instead of being quoted as fact.  Writes `reports/b6_deviations.json`.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


T = Path(r"E:\SPARROW") / "5_Test"
RUN = T / "20260917_3"
A = T / "20260917_2"          # round A: read-only

REPORTS, LOCKS = RUN / "reports", RUN / "locks"
WORK = RUN / "work" / "screen"
FORKS = RUN / "scripts" / "forks"
A_FORKS = A / "scripts" / "forks"

DRIVER = A / "scripts" / "a2h_refit_arm.py"
A_LOCK = A / "locks" / "station_screen_lock.json"
A_DEVIATIONS = A / "reports" / "deviations.json"
A_CONTRACT = A / "arms" / "R74" / "experiment_contract.json"
B_CONTRACT = WORK / "experiment_contract.json"
FORCING_LOCK = WORK / "work" / "locks" / "forcing_integrity_lock.json"
PRODUCT_LOCK = LOCKS / "product_lock.json"
B1_REPORT = REPORTS / "b1_run.json"

FORK_FILES = ["s5_parent_fit.py", "s8_regionalize.py", "s33_spinup.py"]
# The keys that carry round B's identity rather than its configuration.  Everything else in the
# inherited contract must be identical, which is what "configuration verbatim" means when checked.
CONTRACT_IDENTITY_KEYS = {
    "stage", "stage_label_inherited_value", "contract_kind", "derives_from",
    "not_a_replacement_for",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _lock() -> dict:
    """Round A's screen lock.  A function rather than a constant so a missing file raises inside
    `verify()`'s per-check try, where it becomes a failed check, instead of at import time, where it
    would stop the whole register from being written."""
    return read_json(A_LOCK)


# ------------------------------------------------------------------------------------------------ scope

DEVIATIONS = [
    {
        "id": "inherits_round_a_performance_based_exclusion_departure",
        "kind": "user_instruction",
        "what": (
            "`20260828_36` 的 `observation_policy.station_exclusion` 写着 \"No performance-based "
            "exclusions\"，而本产品的站点面板正是按实测负 NSE 排除 28 站得到的。"
        ),
        "why": (
            "这正是上一轮被要求做的事（\"我选择2路线 目标是不要有负站\"），本轮只是承接其结果。"
            "次 A 已就此登记过一次，本轮**继承**该偏离，并再次明写。"
        ),
        "authority": "user instruction（次 A 登记于 20260917_2，本轮继承）",
        "impact": (
            "本产品不是无排除基线。排除是按表现的，不是 QC 排除；任何叙述把它说成 QC 排除都是错的。"
        ),
    },
    {
        "id": "driver_executed_unmodified_with_rebound_roots",
        "kind": "engineering_necessity",
        "what": (
            "`b1_run_product.py` 以模块方式 import 次 A 的 `a2h_refit_arm.py`，重绑其五个模块级根"
            "（RUN/ARMS/REPORTS/FORKS/PRODUCT），而不改动该文件一个字节。"
        ),
        "why": (
            "该司机的 sha256 记在次 A 的筛选锁件 `provenance.driver_sha256` 里。改它就会让筛选件"
            "无法从自己的锁件复现 —— 而那正是本轮一切主张的地基。重绑之所以安全，是因为司机在调用时"
            "才读这五个根（`main()` 里解析 `ARMS / label`，各步骤函数在加载分叉时才读 `FORKS`）。"
        ),
        "authority": "engineering necessity（设计决定，非事后）",
        "impact": (
            "本轮跑的是与次 A 逐字节相同的司机代码，只是写往本轮的树。"
        ),
    },
    {
        "id": "forks_reused_by_byte_identical_copy",
        "kind": "engineering_necessity",
        "what": (
            "本轮的三个分叉不是重新推导的，而是从次 A `scripts/forks/` 逐字节复制过来的。"
        ),
        "why": (
            "分叉是次 A 已验证过的那一份（V1/V2/V3 保真检查、覆盖率门惰性证明都针对它）。"
            "重新推导一份「等价」的分叉，等于让本轮的产品建立在一个没有被验证过的代码上。"
            "复制是复用，但复用必须被证明，所以 `b0_preflight.py` 逐件核对复制前后哈希相同。"
        ),
        "authority": "engineering necessity（设计决定，非事后）",
        "impact": (
            "本轮不依赖次 A 的目录继续存在；分叉哈希同时与次 A 的偏离登记所记值相符。"
        ),
    },
    {
        "id": "experiment_contract_inherited_and_written_before_step01",
        "kind": "engineering_necessity",
        "what": (
            "`work/screen/experiment_contract.json` 由 `b0_preflight.py` 写出，配置逐字继承自次 A 的 "
            "`arms/R74/experiment_contract.json`，只有 `stage` 标签是本轮的；写出时点在步骤 01 **之前**。"
        ),
        "why": (
            "冻结的 stage-5 分叉把这文件哈希进 `input_code_hash_registry.json`，且对它调用 `stat()` "
            "**没有存在性守卫** —— 合约缺失会让父拟合在最后一个动作上死掉。这不是推测：第一次尝试"
            "正因此死在 `s5_parent_fit.py:456`，在约 13 分钟的拟合之后。"
        ),
        "authority": "engineering necessity（跑起来才暴露，已在代码注释与轮次报告中记明）",
        "impact": (
            "合约里的排除名单与保留数被 `b0` 拿筛选件复核过，故 stage-5 分叉哈希的不会是另一套面板。"
        ),
    },
    {
        "id": "stage_label_in_forcing_lock_corrected",
        "kind": "known_artifact",
        "what": (
            "司机的 `build_forcing_lock` 写出 `\"stage\": f\"20260917_2/arms/{label}\"`，对本轮而言"
            "这个字符串是**假的**（不存在 `20260917_2/arms/screen`）。包装函数在原函数返回后只改写"
            "这一个字段，并把继承值记进 `stage_label_inherited_value`。"
        ),
        "why": (
            "不改写就会在产品里留下一条指向不存在目录的自述；改写而不记录，则是把一处改动藏起来。"
        ),
        "authority": "engineering necessity（设计决定，非事后）",
        "impact": (
            "该锁的消费者是 `20260828_33` 与 `20260828_34`，它们取的是强迫路径与状态 token，"
            "不读 `stage`；故这是**标签**，不是数据。但正因如此才要写在这里，而不是悄悄改掉。"
        ),
    },
    {
        "id": "work_tree_cleared_after_failed_step01_attempt",
        "kind": "known_artifact",
        "what": (
            "第一次步骤 01 尝试死在合约缺失那一步之后，`work/screen/outputs/stage5/` 里留下了五个"
            "完整的产物却**没有** `steps/01_parent_fit.done` 标记。重跑前把 `outputs/`、`work/`、"
            "`steps/` 删空。"
            "`reports/b1_run.log` **没有被删**：它的创建时间是 19:19:40，即第一次启动留下的。但它"
            "**里面没有第一次尝试的内容** —— 重跑的 `>` 重定向把那份输出截掉了（截断不改创建时间，"
            "故 19:19:40 这个时间戳只是个残留，不代表内容归属）。实测该日志共两段："
            "① 从 19:42:07 起本轮真正的全链运行（首行即 `01_parent_fit 开始`），六步全部完成、打印"
            "「全链完成」，**末尾有一个 traceback**，属于 `b1_run_product.py` 写报告那一步的路径拼接"
            "缺陷（把「相对于 RUN 的路径」又拼到 `arm/outputs` 上，得到 "
            "`work/screen/outputs/work/screen/outputs/…`）—— 它**不是链的失败**：traceback 之前已打印"
            "「全链完成」、六枚标记俱在、22 件产物完好，受影响的只有那份报告；"
            "② `--- 重跑 b1（步骤应全跳过），19:58:12 ---` 起 —— 修好该缺陷后重跑，六步全跳过、报告写出。"
        ),
        "why": (
            "没有标记就不会跳过重跑，但若重跑再次中途失败，盘上那批「看起来完整」的旧产物就会被读成"
            "重跑的产物 —— 这正是次 A 记录过的第六类断链缺陷（陈旧标记造成假通过）的同形风险。"
            "删掉它们，使本轮产出的每一件都无歧义属于本轮。"
        ),
        "authority": "engineering necessity（跑起来才暴露）",
        "impact": (
            "清理只落在本轮目录，未触及任何冻结轮或次 A。残留的那一份日志是**日志**，不是产物："
            "它不被任何锁件哈希、不被任何步骤读取，故不影响产品。但**按 traceback 关键字扫这份日志会"
            "命中一次，且那一次不是链的失败**（是 b1 写报告的路径拼接），它之前已打印「全链完成」；"
            "故引用该日志的行必须按「全链完成」的位置与重跑标记分清段落，不得把那个 traceback 读成"
            "链断了 —— 这正是本条要写明的理由。"
        ),
    },
    {
        "id": "step07_scoring_not_part_of_the_product_chain",
        "kind": "design_decision",
        "what": (
            "`b1` 只跑步骤 01–06；步骤 07 是次 A 的**臂评分**步骤，本轮不跑它。"
        ),
        "why": (
            "步骤 07 为「哪个组合保留站全非负」这个问题打分，那个问题已在次 A 有了答案，本轮不再问。"
            "本轮的评分是对**已发布产品**做的，写在 `b4_adjudicate.py` 里。"
        ),
        "authority": "设计决定（计划 §五 规定）",
        "impact": "本轮的裁决与次 A 的臂评分是两次独立计算，两者一致才说明产品确实是那一臂的产品。",
    },
    {
        "id": "claim_window_is_narrower_than_the_product",
        "kind": "disclosure",
        "what": (
            "产品覆盖 1961–2025，但零负声明只覆盖保留的 74 站、且只覆盖 2019-01-01..2022-12-31。"
        ),
        "why": (
            "1961–2005 没有任何实测流量，「无站 NSE 为负」对它不是一句可验证的话；"
            "把它说成全期性质是把没有观测的年份当成了已证。"
        ),
        "authority": "disclosure（计划 §五 随件事实）",
        "impact": "本产品在 1961–2005 是模型输出，未经任何观测验证。",
    },
    {
        "id": "screen_is_post_hoc_and_adaptive",
        "kind": "disclosure",
        "what": (
            "本站点面板来自次 A 的**事后**扩展：八个预注册组合全部测完、无一全非负之后，尾巴由实测"
            "残余负站选出。"
        ),
        "why": (
            "该筛选与它被打分的同一批数据有关，故其零负性质更接近拟合量而非受检量。"
            "必须随件携带，否则读者会把一个事后选择读成预注册结果。"
        ),
        "authority": "disclosure（S111 式声明，逐字照搬其边界）",
        "impact": "不支持无偏外推，不能替代全队列基线；同群体保护效应与零负目标必须分开报告。",
    },
]


# ------------------------------------------------------------------------------------------ checkable

CHECKABLE = {
    "inherits_round_a_performance_based_exclusion_departure": lambda: (
        "policy_no_performance_exclusions_ignored"
        in [entry["id"] for entry in read_json(A_DEVIATIONS)["deviations"]]
    ),
    "driver_executed_unmodified_with_rebound_roots": lambda: (
        sha256(DRIVER) == _lock()["provenance"]["driver_sha256"]
    ),
    "forks_reused_by_byte_identical_copy": lambda: all(
        sha256(FORKS / name) == sha256(A_FORKS / name) for name in FORK_FILES
    ),
    "experiment_contract_inherited_and_written_before_step01": lambda: (
        read_json(B_CONTRACT)["derives_from"]["sha256"] == sha256(A_CONTRACT)
        and {
            key: value for key, value in read_json(B_CONTRACT).items()
            if key not in CONTRACT_IDENTITY_KEYS
        } == {
            key: value for key, value in read_json(A_CONTRACT).items()
            if key not in CONTRACT_IDENTITY_KEYS
        }
        # The act the first attempt died on: the contract hashed into the registry the frozen fork
        # writes, present and naming this round's contract.
        and (WORK / "reports" / "stage5" / "input_code_hash_registry.json").is_file()
    ),
    "stage_label_in_forcing_lock_corrected": lambda: (
        read_json(FORCING_LOCK)["stage"] == "20260917_3/work/screen"
        and read_json(FORCING_LOCK)["stage_label_inherited_value"] == "20260917_2/arms/screen"
        and read_json(FORCING_LOCK)["stage_label_corrected_by"] == "b1_run_product.py"
    ),
    "work_tree_cleared_after_failed_step01_attempt": lambda: (
        # Six markers present -- but presence alone is exactly what a surviving stale marker would also
        # produce, so that is not the check.  The discriminator is the forcing lock: step 01 writes it
        # at its start, so it is the earliest artifact of *this* run, and a marker left over from the
        # 19:32 attempt would necessarily predate it.  Every marker must be newer than it, which is
        # what makes "cleared and re-run" distinguishable from "skipped on a surviving marker".
        # (Windows: st_ctime is creation time, so this is when the lock was written, not when it was
        # last touched.)
        sorted(path.stem for path in (WORK / "steps").glob("*.done"))
        == ["01_parent_fit", "02_regionalization", "03_checkpoint", "04_spinup",
            "05_long_simulation", "06_tn_interface"]
        and all(
            path.stat().st_ctime >= FORCING_LOCK.stat().st_ctime
            for path in (WORK / "steps").glob("*.done")
        )
    ),
    "step07_scoring_not_part_of_the_product_chain": lambda: (
        read_json(B1_REPORT)["steps"].split(",")
        == ["01_parent_fit", "02_regionalization", "03_checkpoint", "04_spinup",
            "05_long_simulation", "06_tn_interface"]
    ),
    "claim_window_is_narrower_than_the_product": lambda: (
        read_json(PRODUCT_LOCK)["claim"]["zero_negative_applies_to"].endswith(
            "over 2019-01-01..2022-12-31")
        and "1961-2005" in read_json(PRODUCT_LOCK)["claim"]["zero_negative_is_not_a_statement_about"]
    ),
    "screen_is_post_hoc_and_adaptive": lambda: (
        bool(_lock()["post_hoc"]["is_post_hoc"])
        and _lock()["origin"] == "post_hoc_route2_extension"
        and read_json(PRODUCT_LOCK)["screen"]["is_post_hoc"] is True
    ),
}


def verify() -> dict:
    """Re-check what can be checked from the two rounds' own artifacts, so a stale entry fails loudly."""
    out = {}
    for name, check in CHECKABLE.items():
        try:
            out[name] = bool(check())
        except Exception as error:                            # noqa: BLE001 - a missing file is a failed check
            out[name] = f"CHECK_FAILED: {type(error).__name__}: {error}"
    return out


def main() -> None:
    report = {
        "stage": "20260917_3",
        "purpose": (
            "本轮与冻结轮、次 A 的每一处差异、理由与授权；本轮不改代码，故登记的是脚手架与声明"
        ),
        "fork_hashes_round_b": {name: sha256(FORKS / name) for name in FORK_FILES},
        "fork_hashes_round_a": {name: sha256(A_FORKS / name) for name in FORK_FILES},
        "driver_sha256": sha256(DRIVER),
        "deviations": DEVIATIONS,
        "checks": verify(),
    }
    failed = sorted(key for key, value in report["checks"].items() if value is not True)
    report["passed"] = not failed
    report["failed_checks"] = failed

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "b6_deviations.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"偏离登记：{len(DEVIATIONS)} 条")
    for entry in DEVIATIONS:
        mark = "" if entry["id"] not in CHECKABLE else (
            "  ✓" if report["checks"][entry["id"]] is True else "  *** 校验失败 ***")
        print(f"  [{entry['kind']}] {entry['id']}{mark}")
    print("\n分叉哈希（本轮 = 次 A）：")
    for name in FORK_FILES:
        same = report["fork_hashes_round_b"][name] == report["fork_hashes_round_a"][name]
        print(f"  {name:22s} {report['fork_hashes_round_b'][name][:16]}…"
              f"{'  一致' if same else '  *** 不一致 ***'}")
    if failed:
        print(f"\n校验失败：{failed}")
        raise SystemExit(1)
    print("\n全部可校验项通过")


if __name__ == "__main__":
    main()
