"""B1: build the 1961-2025 product for the frozen screen by re-running round A's chain in round B's tree.

The plan says round B *re-runs* the frozen combination rather than adopting round A's arm directory
("按 `_2` 冻结的组合**重跑** 1961-2025 全河网产品，两个面板都按筛选后重拟合，零额外调参").  Re-running
is also what makes round B's product self-contained: `outputs/` carries its own lineage instead of
pointing into another round's arm directory, and the reproduction can then be *checked* (see `b2`).

The chain itself is round A's, unmodified.  `20260917_2/scripts/a2h_refit_arm.py` is imported as a
module and its five module-level roots are rebound before `main()` is called, so the code that runs is
byte-for-byte the driver the screen lock records a `driver_sha256` for -- while every path it writes
resolves into `20260917_3/`.  Rebinding the roots is safe because the driver reads all five at call
time (it resolves `ARMS / label` inside `main()`, and each step function reads `FORKS` when it loads its
fork); it is also necessary, because editing that file to parameterise it would change the hash round A
froze and make the screen unreproducible from its own lock.

One label is corrected rather than inherited.  The driver's `build_forcing_lock` writes
`"stage": f"20260917_2/arms/{label}"`, a string that is simply false for this round -- there is no
`20260917_2/arms/screen`.  The wrapper below calls the original and then rewrites that one field,
recording both values.  Nothing reads it (the consumers of that lock are `_33` and `_34`, which take the
forcing path and the status token), so this is a label, not a datum -- but it is disclosed here and in
the round's deviation register rather than edited quietly.

Runs steps 01-06.  Step 07 is round A's *scoring* step and is deliberately not part of a product build;
round B scores its own product in `b3`.

Writes `reports/b1_run.json`.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path


T = Path(r"E:\SPARROW") / "5_Test"
RUN = T / "20260917_3"
A = T / "20260917_2"          # round A: read-only

DRIVER = A / "scripts" / "a2h_refit_arm.py"
REPORTS = RUN / "reports"
ARM_ROOT = RUN / "work"
LABEL = "screen"

STEPS = "01_parent_fit,02_regionalization,03_checkpoint,04_spinup,05_long_simulation,06_tn_interface"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", default=STEPS)
    args = parser.parse_args()

    arm = ARM_ROOT / LABEL
    if not (arm / "inputs" / "s5_registry.parquet").is_file():
        raise RuntimeError(f"{arm} carries no inputs; run b0_preflight.py first")
    if not (arm / "excluded.json").is_file():
        raise RuntimeError(f"{arm} carries no excluded.json; run b0_preflight.py first")

    driver = load("b1_round_a_driver", DRIVER)
    # The five roots, rebound before `main()`.  `FORKS` is round B's verified copy of round A's forks,
    # not round A's directory -- round B must not depend on round A staying where it is.
    driver.RUN = RUN
    driver.ARMS = ARM_ROOT
    driver.REPORTS = REPORTS
    driver.FORKS = RUN / "scripts" / "forks"

    # Keep the original and wrap it, rather than replacing it: if the original is later changed to write
    # something else, the correction below stops matching and the run fails instead of silently
    # overwriting a field it no longer recognises.
    original_build = driver.build_forcing_lock
    corrections: list[dict] = []

    def build_forcing_lock_with_correct_stage(arm_obj) -> None:
        original_build(arm_obj)
        path = arm_obj.work / "locks" / "forcing_integrity_lock.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        inherited = data.get("stage")
        expected = f"20260917_2/arms/{arm_obj.label}"
        if inherited != expected:
            raise RuntimeError(
                f"{path.name} 的 stage 既非继承值 {expected!r} 也非本轮应有的值（现为 {inherited!r}）；"
                "不改写未能识别的字段"
            )
        data["stage"] = f"20260917_3/work/{arm_obj.label}"
        data["stage_label_corrected_by"] = "b1_run_product.py"
        data["stage_label_inherited_value"] = inherited
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        corrections.append({"key": "stage", "inherited": inherited, "corrected": data["stage"]})

    driver.build_forcing_lock = build_forcing_lock_with_correct_stage

    sys.argv = [str(DRIVER), "--arm", LABEL, "--steps", args.steps]
    started = time.perf_counter()
    driver.main()
    elapsed = time.perf_counter() - started

    artifacts = sorted(path for path in (arm / "outputs").rglob("*") if path.is_file())
    products = [str(path.relative_to(RUN)) for path in artifacts]
    report = {
        "stage": "20260917_3",
        "purpose": "B1: re-run round A's chain on the frozen screen inside round B's tree",
        "driver": {
            "path": str(DRIVER),
            "sha256": sha256(DRIVER),
            "why_not_edited": (
                "a2h_refit_arm.py 的 sha256 记在 station_screen_lock.json 的 provenance 里；"
                "改它就会让筛选件无法从自己的锁件复现，故本轮只重绑根路径、不改字节"
            ),
        },
        "roots_rebound": {
            "RUN": str(RUN), "ARMS": str(ARM_ROOT), "REPORTS": str(REPORTS),
            "FORKS": str(RUN / "scripts" / "forks"),
        },
        "arm": str(arm),
        "steps": args.steps,
        "seconds": elapsed,
        "label_corrections": corrections,
        "products": products,
        # Hashed from the file objects themselves.  The earlier revision re-joined the relative string
        # onto `arm / "outputs"`, which doubled the prefix (`work/screen/outputs/work/screen/outputs/…`)
        # and killed the run at its last act -- after all six steps had finished.  Building the key from
        # the same path that was hashed is what makes that unrepresentable rather than merely fixed.
        "product_hashes": {
            str(path.relative_to(RUN)): sha256(path)
            for path in artifacts
            if path.suffix != ".log"
        },
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "b1_run.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nB1 完成，用时 {elapsed / 60:.1f} 分钟")
    print(f"  产物 {len(products)} 件，写在 {arm / 'outputs'}")
    for correction in corrections:
        print(f"  标签订正：{correction['key']} {correction['inherited']!r} → {correction['corrected']!r}")


if __name__ == "__main__":
    main()
