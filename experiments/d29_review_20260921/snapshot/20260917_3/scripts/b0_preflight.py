"""B0: gate round B on round A's frozen screen, then adopt it by *re-derivation*, not by copy.

Round B's only screening input is `<round A>/locks/station_screen_lock.json`, and the plan's entry
condition is that this file exists *and* its feasibility is true.  Both are checked here, and so is the
thing that actually matters and is not implied by either: that everything the lock recorded is still
what is on disk.  A lock whose own hashes no longer verify is a lock about a different base, and a
round B built on it would produce a product for a panel nobody chose.

The four checks, in the order they can fail:

  1. **The lock's self-hashes.**  `base_cohort_final.parquet`, `station_screen.parquet` and round A's
     rule/driver/combinations sources are re-hashed and compared to the values the lock recorded.
  2. **The screen is the screen.**  The excluded set is read from `station_screen.parquet` -- the frozen
     deliverable -- and required to agree with the lock's counts, with the arm result the lock froze on,
     and with `arms/<combination>/excluded.json`.  Three independent statements of the same 28 names.
  3. **The forks are round A's.**  Round B reuses round A's validated forks rather than re-deriving
     them; the copies are verified **byte-identical** to the originals and their hashes are pinned, so
     "the same code" is proven here rather than assumed downstream.
  4. **The masked inputs are the masked inputs.**  Round B's inputs are taken from the arm directory and
     then *checked by recomputing the masking's own signature* -- the row and finite-value counts that
     `a2h_arm_inputs.py` recorded for that arm before any arm ran.  A copy is not evidence; a recount is.

Nothing in round A is written.  Everything here is read except round B's own tree.

Writes `reports/b0_preflight.json`.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd


T = Path(r"E:\SPARROW") / "5_Test"
RUN = T / "20260917_3"
A = T / "20260917_2"          # round A: read-only, throughout

A_LOCK = A / "locks" / "station_screen_lock.json"
A_SCREEN = A / "outputs" / "station_screen.parquet"
A_INPUTS = A / "reports" / "a2h_arm_inputs.json"
A_FORKS = A / "scripts" / "forks"

WORK, REPORTS, FORKS = RUN / "work", RUN / "reports", RUN / "scripts" / "forks"
PRODUCT = RUN / "work" / "screen"
# The arm label round A's driver is invoked with.  `b1` sets `ARMS = work/` and passes this as `--arm`,
# so the chain's workspace lands at exactly this path -- the two must agree, hence one constant shared by
# both scripts' reading of `PRODUCT`.
LABEL = "screen"

FORK_FILES = ["s5_parent_fit.py", "s8_regionalize.py", "s33_spinup.py"]
# The five masked inputs the arm driver reads, and the column each one's finite count was recorded from
# in `a2h_arm_inputs.py:448-451`.  Restated here rather than imported: this file's job is to check that
# record, and a check whose expectations come from the thing it is checking is not a check.
INPUT_FILES = {
    "s5_discharge.parquet": "q_m3_s",
    "s5_registry.parquet": None,
    "s8_discharge.parquet": "q_m3_s",
    "s8_registry.parquet": None,
    "s8_monthly.parquet": "q_m3s",
}
COUNT_KEYS = {
    "s5_discharge.parquet": ("s5_daily_finite", "rows"),
    "s8_discharge.parquet": ("s8_daily_finite", None),
    "s8_monthly.parquet": ("s8_monthly_finite", None),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def norm(value: object) -> str:
    """The round's station-name key.  NFKC-symmetric, whitespace-stripped, trailing 站 dropped.

    Duplicated from round A's utilities on purpose: this is the rule that a *wrong* copy of would make
    every name comparison below silently match nothing, and an imported implementation would let a
    later edit to round A's copy change what round B adopted without either round noticing.
    """
    import unicodedata
    text = unicodedata.normalize("NFKC", "" if pd.isna(value) else str(value)).strip()
    text = "".join(ch for ch in text if not ch.isspace()).replace("_", "")
    return text[:-1] if text.endswith("站") else text


def main() -> None:
    failures: list[str] = []

    def require(condition: bool, message: str) -> None:
        if condition:
            print(f"  ✓ {message}")
        else:
            print(f"  ✗ {message}")
            failures.append(message)

    print("B0 步骤 1/4：筛选锁件的自哈希")
    lock = read_json(A_LOCK)
    for name, path, recorded in (
        ("基座 base_cohort_final.parquet", Path(lock["base"]["path"]), lock["base"]["sha256"]),
        ("筛选件 station_screen.parquet", Path(lock["screen"]["path"]), lock["screen"]["sha256"]),
        ("组合文件 combinations.parquet", Path(lock["provenance"]["combinations"]),
         lock["provenance"]["combinations_sha256"]),
        ("筛选规则 a2h_run_arms.py", Path(lock["provenance"]["rule_source"]),
         lock["provenance"]["rule_sha256"]),
        ("臂司机 a2h_refit_arm.py", Path(lock["provenance"]["driver_source"]),
         lock["provenance"]["driver_sha256"]),
    ):
        if not path.is_file():
            require(False, f"{name} 不在盘上：{path}")
            continue
        require(sha256(path) == recorded, f"{name} 哈希未变")

    print("B0 步骤 2/4：零负门与筛选件内容")
    require(bool(lock["gate"]["zero_negative_on_retained"]), "零负门在锁件中为真")
    require(int(lock["gate"]["retained_negative"]) == 0, "保留站中的负站数为 0")
    combination = lock["combination"]
    origin = lock["origin"]

    screen = pd.read_parquet(A_SCREEN)
    excluded_from_screen = sorted(screen.loc[screen.excluded, "station_norm"].astype(str))
    excluded_from_arm = sorted(
        read_json(A / "arms" / combination / "excluded.json")["excluded"]
    )
    result = read_json(A / "arms" / combination / "arm_result.json")
    excluded_from_result = sorted(
        result["stations"] and [
            row["station_norm"] for row in result["stations"] if row.get("excluded")
        ]
    )
    require(
        excluded_from_screen == excluded_from_arm == excluded_from_result,
        f"三处排除名单一致（{len(excluded_from_screen)} 个：筛选件／臂 excluded.json／臂结果）",
    )
    require(len(excluded_from_screen) == int(lock["base"]["excluded"]),
            f"排除数与锁件一致（{len(excluded_from_screen)}）")
    require(int(lock["base"]["retained"]) == int(len(screen)) - len(excluded_from_screen),
            f"保留数一致（{lock['base']['retained']}）")
    require(int(result["retained_negative"]) == 0,
            "被冻结的那一臂的残留负站数为 0（锁件所依据的实测）")

    print("B0 步骤 3/4：分叉逐字节复刻到下轮（复用即须证明）")
    FORKS.mkdir(parents=True, exist_ok=True)
    fork_hashes = {}
    for name in FORK_FILES:
        source, target = A_FORKS / name, FORKS / name
        if not source.is_file():
            require(False, f"分叉缺失：{source}")
            continue
        digest = sha256(source)
        fork_hashes[name] = {"sha256": digest, "source": str(source), "bytes": source.stat().st_size}
        shutil.copy2(source, target)
        require(sha256(target) == digest, f"{name} 已复制且逐字节相同（{digest[:16]}…）")
        # A stale `__pycache__` beside the copy would let an edited source keep being shadowed by an
        # older `.pyc`; the copy is new, so its cache must be new too.
        cached = FORKS / "__pycache__"
        if cached.is_dir():
            shutil.rmtree(cached)

    print("B0 步骤 4/4：被掩蔽输入的签名复算（不采信拷贝）")
    PRODUCT.mkdir(parents=True, exist_ok=True)
    (PRODUCT / "inputs").mkdir(parents=True, exist_ok=True)
    recorded = read_json(A_INPUTS)["arms"][combination]
    signature = {}
    for name, column in INPUT_FILES.items():
        source = A / "arms" / combination / "inputs" / name
        if not source.is_file():
            require(False, f"输入缺失：{source}")
            continue
        target = PRODUCT / "inputs" / name
        shutil.copy2(source, target)
        frame = pd.read_parquet(target)
        signature[name] = {"rows": int(len(frame)), "columns": list(frame.columns)}
        if column is None:
            continue
        finite = int(frame[column].notna().sum())
        signature[name]["finite_in_" + column] = finite
        key, rows_key = COUNT_KEYS.get(name, (None, None))
        if key:
            require(finite == int(recorded[key]),
                    f"{name} 有限值 {finite} = 记录值 {recorded[key]}")
        if name == "s5_discharge.parquet":
            require(int(len(frame)) == int(recorded["s5_daily_rows"]),
                    f"{name} 行数 {len(frame)} = 记录值 {recorded['s5_daily_rows']}"
                    "（掩蔽不删行，只有值变 NaN）")

    excluded_copy = PRODUCT / "excluded.json"
    excluded_copy.write_text(
        json.dumps({"combination": combination, "excluded": excluded_from_screen},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    require(
        sorted(read_json(excluded_copy)["excluded"]) == excluded_from_screen,
        "下一轮的 excluded.json 即筛选件的那一份",
    )

    # The experiment contract has to exist *before* step 01, not after: the stage-5 fork hashes it into
    # `input_code_hash_registry.json` and calls `stat()` on it without an existence guard, so a missing
    # contract kills the parent fit at its very last act -- after ~20 minutes of fitting.  (Found by
    # running it: the first attempt died at `s5_parent_fit.py:456` with FileNotFoundError.)
    #
    # Round B's contract is round A's R74 contract with its identity restated.  The *configuration* is
    # copied verbatim -- it is the same configuration, which is the entire point of the round -- and only
    # the stage label changes, with the inheritance recorded so nobody has to guess which round's
    # contract this is.
    source_contract = A / "arms" / combination / "experiment_contract.json"
    contract_record: dict = {}
    if not source_contract.is_file():
        require(False, f"被冻结臂的实验合约缺失：{source_contract}")
    else:
        inherited = read_json(source_contract)
        contract = dict(inherited)
        contract["stage"] = f"20260917_3/work/{LABEL}"
        contract["stage_label_inherited_value"] = inherited.get("stage")
        contract["contract_kind"] = (
            "round B 的产品构建合约：配置逐字继承自被冻结的那种掩蔽，只有 stage 标签是本轮的"
        )
        contract["derives_from"] = {
            "round": "20260917_2",
            "arm": combination,
            "path": str(source_contract),
            "sha256": sha256(source_contract),
        }
        contract["not_a_replacement_for"] = inherited.get("not_a_replacement_for", "20260828_35")
        contract_path = PRODUCT / "experiment_contract.json"
        contract_path.write_text(
            json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8")
        contract_record = {
            "path": str(contract_path), "sha256": sha256(contract_path),
            "inherited_from": str(source_contract), "inherited_sha256": sha256(source_contract),
            "stage_corrected_from": inherited.get("stage"),
            "configuration_verbatim": True,
        }
        # The exclusion list inside the contract is the one thing that must agree with the screen, or the
        # stage-5 fork would hash a contract describing a panel different from the masked inputs.
        require(
            sorted(contract.get("excluded_stations") or []) == excluded_from_screen,
            f"合约里的排除名单与筛选件一致（{len(excluded_from_screen)} 个）",
        )
        require(
            int(contract.get("excluded_count", -1)) == len(excluded_from_screen)
            and int(contract.get("retained_from_base", -1))
            == int(len(screen)) - len(excluded_from_screen),
            "合约的排除数／保留数与筛选件一致",
        )
        print(f"  ✓ 实验合约已写：{contract_path.name}（继承 {source_contract.name}，仅改 stage 标签）")

    report = {
        "stage": "20260917_3",
        "purpose": (
            "B0: gate round B on round A's frozen screen -- the lock verifies against disk, the screen "
            "is one 28-station exclusion set stated three ways, the forks are byte-identical, and the "
            "masked inputs recount to the numbers recorded before any arm ran"
        ),
        "entry_condition": {
            "lock": str(A_LOCK), "lock_exists": A_LOCK.is_file(),
            "feasibility_true": bool(lock["gate"]["zero_negative_on_retained"]),
        },
        "adopted": {
            "combination": combination,
            "origin": origin,
            "post_hoc": lock.get("post_hoc", {}).get("is_post_hoc"),
            "base_size": int(len(screen)),
            "retained": int(len(screen)) - len(excluded_from_screen),
            "excluded": excluded_from_screen,
            "excluded_count": len(excluded_from_screen),
            "worst_retained_nse": float(lock["gate"]["worst_retained_nse"]),
            "excluded_that_were_positive": int(lock["gate"]["excluded_that_were_positive"]),
        },
        "same_population_protection": lock.get("same_population_protection"),
        "forks": fork_hashes,
        "experiment_contract": contract_record,
        "input_signature": signature,
        "input_record_from_round_a": {
            key: recorded.get(key) for key in (
                "excluded", "masked_in_stage5_panel", "masked_in_stage8_panel", "inert_exclusions",
                "retained_on_base", "s5_daily_rows", "s5_daily_finite", "s8_daily_finite",
                "s8_monthly_finite",
            )
        },
        "written_by_this_round": {
            "product_inputs": str(PRODUCT / "inputs"),
            "excluded": str(excluded_copy),
        },
        "read_only_declaration": (
            "本轮只读 20260917_2 的筛选锁件与臂目录，未写入其中任何文件；"
            "20260917_2 的司机与分叉的 sha256 在锁件与偏离登记里各有记录，本轮不得改动它们。"
        ),
        "passed": not failures,
        "failures": failures,
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "b0_preflight.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    if failures:
        print(f"B0 未通过：{len(failures)} 项")
        for item in failures:
            print(f"  - {item}")
        raise SystemExit(1)
    print(f"B0 通过：采用组合 {combination}（{origin}），"
          f"基座 {report['adopted']['base_size']} 站中保留 {report['adopted']['retained']}、"
          f"排除 {report['adopted']['excluded_count']}；零负门为真")
    print(f"  写盘：{REPORTS / 'b0_preflight.json'}")


if __name__ == "__main__":
    main()
