"""B3: publish the product -- move the chain's files out of the working tree into the round's `outputs/`.

The chain writes into `work/screen/`, which is a *working* directory: it also holds step markers, the
copied core module and the per-step reports.  A product that a consumer is asked to read should not be
something you have to know the internals to find, and it should not be mixed in with the machinery that
produced it.  So the artifacts that constitute the product are copied to `<round>/outputs/` and the
chain's own locks and reports to `<round>/locks/` and `<round>/reports/chain/`, each with its hash
recorded on both sides.

Copy, not move: the chain's locks reference the files where they were written, and moving them would
leave those references pointing at nothing -- the round would then be shipping locks that describe a
directory which no longer exists.  Copy, and `publish()` re-hashes every destination against its source,
so "the published file is the produced file" is checked rather than asserted.

The product lock carries three things a reader needs in order to use this product honestly:

  * what the screen is and that it is **post-hoc** -- the 74 retained stations are the ones that came
    out non-negative under a mask chosen after seeing the measurements, which is a different kind of
    claim from a pre-registered one;
  * the **claim window**: the zero-negative result is a statement about 2019-2022, the only period with
    observations. 1961-2005 is model output with no station data behind it;
  * the **same-population protection cost**, so the screen is not read as free.

Writes `<round>/outputs/`, `<round>/locks/`, `<round>/reports/chain/` and
`locks/product_lock.json`.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


T = Path(r"E:\SPARROW") / "5_Test"
RUN = T / "20260917_3"
A = T / "20260917_2"

SRC = RUN / "work" / "screen"
OUT, LOCKS, REPORTS = RUN / "outputs", RUN / "locks", RUN / "reports"
CHAIN_REPORTS = REPORTS / "chain"

# The product: what a consumer of "1961-2025, 230 reaches, daily and monthly" actually reads.  Named
# explicitly rather than globbed -- a glob would silently publish whatever the chain happened to leave
# behind, and would silently stop publishing a file that a later change renamed.
PRODUCTS = [
    "outputs/tn_hydrology_reach_daily.parquet",
    "outputs/tn_hydrology_reach_monthly.parquet",
    "outputs/tn_hydrology_reservoir_daily.parquet",
    "outputs/tn_hydrology_reservoir_monthly.parquet",
    "outputs/tn_hydrology_reservoir_static_metadata.parquet",
    "outputs/state_consistent_reach_daily.parquet",
    "outputs/state_consistent_reach_monthly.parquet",
    "outputs/state_consistent_reservoir_daily.parquet",
    "outputs/state_consistent_reservoir_monthly.parquet",
    "outputs/state_consistent_reservoir_static_metadata.parquet",
    "outputs/export/canonical_reach_daily_2006_2024.parquet",
    "outputs/export/canonical_reach_monthly_2006_2024.parquet",
    "stage33/outputs/historical_initial_land_state_1961.npz",
    "stage33/outputs/historical_initial_reservoir_states_1961.json",
]
# The chain's own locks and reports.  Published because they are the evidence for the product, and an
# evidence file that only exists inside a working directory is one deletion away from being unrecoverable.
CHAIN_LOCKS = [
    "locks/tn_hydrology_interface_lock.json",
    "locks/long_simulation_lock.json",
    "reports/export/parent_preserving_product_lock.json",
    "stage33/locks/frozen_reproduction_spinup_lock.json",
    "work/locks/forcing_integrity_lock.json",
]
CHAIN_EVIDENCE = [
    "reports/tn_hydrology_interface_qa.json",
    "reports/long_simulation_qa.json",
    "reports/export/validation.json",
    "reports/stage5/validation.json",
    "reports/stage5/blind_prediction_lock.json",
    "reports/stage5/input_code_hash_registry.json",
    "reports/stage8/stage8_preparation_lock.json",
    "stage33/reports/frozen_reproduction_spinup_qa.json",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def publish(relative: str, target_dir: Path, flatten: bool) -> dict:
    """Copy one chain artifact to its published home and prove the copy is the source.

    `flatten` is for the product itself -- a consumer should find `outputs/tn_hydrology_reach_daily.
    parquet` without learning the chain's directory layout.  The chain's own locks and reports are
    published with their relative path preserved, because two of them share a basename
    (`reports/export/validation.json` and `reports/stage5/validation.json`): flattened, the second
    copy would silently overwrite the first and the round would ship seven evidence files while
    reporting eight.  `published_targets` in `main()` refuses that outcome outright.
    """
    source = SRC / relative
    if not source.is_file():
        raise RuntimeError(f"产物缺失，拒绝发布：{source}")
    target = (target_dir / Path(relative).name) if flatten else (target_dir / relative)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    source_hash, target_hash = sha256(source), sha256(target)
    if source_hash != target_hash:
        raise RuntimeError(f"{relative} 复制后哈希不同：{source_hash} != {target_hash}")
    return {
        "source": str(source), "published": str(target), "sha256": source_hash,
        "bytes": source.stat().st_size,
    }


def main() -> None:
    lock = read_json(A / "locks" / "station_screen_lock.json")
    preflight = read_json(REPORTS / "b0_preflight.json")
    if not preflight.get("passed"):
        raise RuntimeError("b0 预检未通过；不得据此发布产品")

    for directory in (OUT, LOCKS, CHAIN_REPORTS):
        directory.mkdir(parents=True, exist_ok=True)

    # Every source gets exactly one destination, and no two sources share one.  Checked before any
    # copy rather than discovered as a shorter list at the end.
    published_targets: dict[str, str] = {}
    for relative, target_dir, flatten in (
        *[(name, OUT, True) for name in PRODUCTS],
        *[(name, LOCKS, False) for name in CHAIN_LOCKS],
        *[(name, CHAIN_REPORTS, False) for name in CHAIN_EVIDENCE],
    ):
        target = str((target_dir / Path(relative).name) if flatten else (target_dir / relative))
        if target in published_targets:
            raise RuntimeError(
                f"发布目标重名：{relative} 与 {published_targets[target]} 都要写到 {target}"
            )
        published_targets[target] = relative

    published = {}
    for relative in PRODUCTS:
        published[relative] = publish(relative, OUT, flatten=True)
    chain = {}
    for relative in CHAIN_LOCKS:
        chain[relative] = publish(relative, LOCKS, flatten=False)
    for relative in CHAIN_EVIDENCE:
        chain[relative] = publish(relative, CHAIN_REPORTS, flatten=False)

    protection = lock["same_population_protection"]
    product_lock = {
        "stage": "20260917_3",
        "purpose": (
            "1961-2025 全河网（230 河段）日/月水文产品，站点面板取 20260917_2 冻结的筛选件；"
            "不替换、不冒充正式产品 20260828_35"
        ),
        "screen": {
            "round": "20260917_2",
            "combination": lock["combination"],
            "origin": lock["origin"],
            "is_post_hoc": lock.get("post_hoc", {}).get("is_post_hoc"),
            "how_the_configuration_was_chosen": lock.get("post_hoc", {}).get("how_the_configuration_was_chosen"),
            "declaration": lock.get("post_hoc", {}).get("declaration"),
            "base": lock["base"]["evaluable"],
            "retained": lock["base"]["retained"],
            "excluded": lock["base"]["excluded"],
            "gate": lock["gate"],
            "screen_parquet": lock["screen"],
        },
        "claim": {
            "zero_negative_applies_to": "the 74 retained stations, over 2019-01-01..2022-12-31",
            "zero_negative_is_not_a_statement_about": (
                "1961-2005, which has no station observations -- those years are model output and are "
                "not validated against any measurement here"
            ),
            "boundary": (
                "本产品的零负声明是**事后**筛选下的结果：掩蔽规则由实测残余负站选出，"
                "不支持无偏外推，也不能替代全队列基线报告"
            ),
        },
        "same_population_protection": protection,
        "not_a_replacement_for": "20260828_35",
        "carried_facts": [
            "结论期是 2006-2022（观测档案所及）；1961-2005 无实测流量，「无站 NSE 为负」因此是 "
            "2019-2022 的陈述。",
            "2025 的 PET 桥接未过 0.98 门（r = 0.916165），2025 段携带 PET_BRIDGE_UNVALIDATED_2025。",
            "本产品不替换、不冒充正式产品 20260828_35。",
        ],
        "products": published,
        "chain_evidence": chain,
        "provenance": {
            "round_a_lock": str(A / "locks" / "station_screen_lock.json"),
            "round_a_lock_sha256": sha256(A / "locks" / "station_screen_lock.json"),
            "b0_preflight": str(REPORTS / "b0_preflight.json"),
            "b1_run": str(REPORTS / "b1_run.json") if (REPORTS / "b1_run.json").is_file() else None,
            "b2_reproduction": (
                str(REPORTS / "b2_reproduction.json")
                if (REPORTS / "b2_reproduction.json").is_file() else None
            ),
        },
    }
    lock_path = LOCKS / "product_lock.json"
    lock_path.write_text(json.dumps(product_lock, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"发布完成：")
    print(f"  产物 {len(published)} 件 → {OUT}")
    print(f"  链证据 {len(chain)} 件 → {LOCKS} / {CHAIN_REPORTS}")
    print(f"  产品锁件 → {lock_path}")
    print(f"  声明窗口：零负针对保留的 {lock['base']['retained']} 站、2019-2022；"
          f"1961-2005 无观测")
    print(f"  保护代价（同群体）：Δ均值 NSE {protection['delta_mean_nse_min']:+.4f}，"
          f"变好 {protection['stations_improved']} 站、变差 {protection['stations_worsened']} 站")


if __name__ == "__main__":
    main()
