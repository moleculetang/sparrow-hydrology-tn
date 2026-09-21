"""A2h-2: run one refit arm of the station-screen experiment, end to end.

An arm is one combination's exclusion set applied by masking, followed by a *complete* model rerun --
the user's standing requirement ("你不能这样排除 要重跑 … 这需要你rerun the model的").  Nothing is
replayed from a frozen product anywhere in the chain.

The seven steps, and the round each is taken from:

    1  parent fit          fork of 20260827_5  ->  full_development_model_lock.pt
    2  regionalization     fork of 20260827_8  ->  regionalized_slow_score.parquet   (--prepare-only)
    3  checkpoint export   20260828_9          ->  parent_preserving_state_consistent_model.pt
    4  1961 spin-up        fork of 20260828_33 ->  historical_initial_land_state_1961.npz
    5  long simulation     20260828_34         ->  state_consistent_reach_{daily,monthly}.parquet
    6  TN interface        20260828_35         ->  tn_hydrology_reach_{daily,monthly}.parquet
    7  evaluation          a1_evaluate + a2g   ->  per-station NSE on the 102, retained set

Steps 5-6 mirror `20260828_43/scripts/run_hybrid_1961_2025.py` exactly, including its forcing-lock
layout, so an arm's product is laid out like the frozen one and the two are read by the same evaluator.
`20260828_35` is only ever *loaded as code* into an arm directory -- the formal product round is never
written to.

Two things are deliberately shared across every arm rather than varied:

  * **forcing** -- `20260828_40/outputs/daily_hbv_forcing_1961_2025_chmpre_cmfd_corrected_era5_2025.parquet`,
    the same file `20260828_43` ran on.  Forcing is a property of the reaches, not of the station panel,
    and holding it fixed is what makes an arm differ from `_43` only by the panel and the masking.
  * **lambda_S** and everything else inside `frozen_hydrology_core` except `CHECKPOINT` and `SCORE`.

Module identity is the one subtle part.  Both `_33` and `_34` do `sys.path[:0] = [... S33/"scripts" ...]`
**at import time**, so rebinding `RUN`/`S33` afterwards does not move the search path.  The arm's copy of
`frozen_hydrology_core.py` is therefore put at `sys.path[0]` and imported *before* the forks are loaded,
so `sys.modules` holds the arm's copy and the forks' `from frozen_hydrology_core import ...` binds to it.
`CHECKPOINT`/`SCORE` are then rebound on that module object, which is what `load_frozen_context()` reads
at call time.

Each step drops a marker under `<arm>/steps/`; re-running skips completed steps, so a long arm can be
resumed without redoing the parent fit.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


T = Path(r"E:\SPARROW") / "5_Test"
RUN = T / "20260917_2"
ARMS = RUN / "arms"
REPORTS = RUN / "reports"

FORKS = RUN / "scripts" / "forks"
FIREWALL = T / "20260828_1" / "inputs"
FW_FORCING = FIREWALL / "forcing_2006_2016.parquet"

# The frozen 1961-2025 forcing and the three rounds loaded as code.  Read-only, always.
FORCING_1961_2025 = (
    T / "20260828_40" / "outputs"
    / "daily_hbv_forcing_1961_2025_chmpre_cmfd_corrected_era5_2025.parquet"
)
SOURCE9 = T / "20260828_9" / "scripts" / "export_parent_preserving_state_product.py"
SOURCE34 = T / "20260828_34" / "scripts" / "run_locked_long_simulation.py"
SOURCE35 = T / "20260828_35" / "scripts" / "export_tn_hydrology_interface.py"
S33_CORE = T / "20260828_33" / "scripts" / "frozen_hydrology_core.py"

S5_PANEL, S8_PANEL = 74, 100
FROZEN_S5_PANEL, FROZEN_S8_PANEL = 65, 91


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class Arm:
    """One combination's workspace.  `done()` is the resume mechanism."""

    def __init__(self, label: str, core: object) -> None:
        self.label = label
        self.core = core
        self.root = ARMS / label
        self.steps = self.root / "steps"
        self.steps.mkdir(parents=True, exist_ok=True)
        self.outputs = self.root / "outputs"
        self.work = self.root / "work"
        self.timings: dict[str, float] = {}

    def done(self, step: str) -> bool:
        return (self.steps / f"{step}.done").exists()

    def mark(self, step: str, started: float) -> None:
        self.timings[step] = time.perf_counter() - started
        (self.steps / f"{step}.done").write_text(
            json.dumps({"step": step, "seconds": self.timings[step]}), encoding="utf-8",
        )
        print(f"  [{self.label}] {step} 完成，用时 {self.timings[step] / 60:.1f} 分钟", flush=True)

    def step(self, name: str, action) -> None:
        if self.done(name):
            print(f"  [{self.label}] {name} 已完成，跳过", flush=True)
            return
        started = time.perf_counter()
        print(f"  [{self.label}] {name} 开始 ...", flush=True)
        # The stage5 fork patches `pd.read_parquet` **on the shared pandas module** to blind itself to
        # the 2019-2022 formal test observations.  That guard is meaningful exactly while the parent fit
        # runs; left in place it would make step 07's `base_observations()` raise PermissionError when it
        # reads that same file for evaluation.  Save/restore so the guard keeps its scope and nothing
        # leaks downstream.
        original_read_parquet = pd.read_parquet
        original_argv = list(sys.argv)
        try:
            action()
        finally:
            pd.read_parquet = original_read_parquet
            sys.argv[:] = original_argv
        self.mark(name, started)


def build_forcing_lock(arm: Arm) -> None:
    """The `20260828_43` forcing-lock layout, so `_33` and `_34` find what they expect."""
    locks = arm.work / "locks"
    locks.mkdir(parents=True, exist_ok=True)
    (locks / "forcing_integrity_lock.json").write_text(json.dumps({
        "stage": f"20260917_2/arms/{arm.label}",
        "status": "PASS_FORCING_LOCK_1961_2025",
        "formal_period": "1961-01-01 through 2025-12-31",
        "formal_forcing_path": str(FORCING_1961_2025),
        "pet_bridge_status": "CORRECTED_ERA5_2025_ONLY",
        "note": "Shared verbatim with 20260828_43: forcing does not depend on the station panel.",
    }, ensure_ascii=False, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------------------------- steps

def step_parent_fit(arm: Arm) -> None:
    module = load(f"arm_{arm.label}_s5", FORKS / "s5_parent_fit.py")
    stage5 = arm.outputs / "stage5"
    module.RUN = arm.root
    module.OUT = stage5
    module.REPORTS = arm.root / "reports" / "stage5"
    module.PANEL_SIZE = S5_PANEL
    module.REGISTRY = arm.root / "inputs" / "s5_registry.parquet"
    module.DISCHARGE = arm.root / "inputs" / "s5_discharge.parquet"
    module.main()


def step_regionalization(arm: Arm) -> None:
    module = load(f"arm_{arm.label}_s8", FORKS / "s8_regionalize.py")
    stage8 = arm.outputs / "stage8"
    module.RUN = arm.root
    module.OUT = stage8
    module.REPORTS = arm.root / "reports" / "stage8"
    module.PANEL_SIZE = S8_PANEL
    module.GAUGES = arm.root / "inputs" / "s8_registry.parquet"
    module.DISCHARGE = arm.root / "inputs" / "s8_discharge.parquet"
    module.MONTHLY = arm.root / "inputs" / "s8_monthly.parquet"
    module.FORCING = FW_FORCING
    sys.argv = [str(FORKS / "s8_regionalize.py"), "--prepare-only"]
    module.main()


def step_checkpoint(arm: Arm) -> None:
    module = load(f"arm_{arm.label}_s9", SOURCE9)
    export = arm.outputs / "export"
    module.RUN = arm.root
    module.OUT = export
    module.REPORTS = arm.root / "reports" / "export"
    module.OLDMODEL = arm.outputs / "stage5" / "full_development_model_lock.pt"
    module.SCORE = arm.outputs / "stage8" / "regionalized_slow_score.parquet"
    # FORCING is deliberately left at its default: the exporter's canonical products are 2006-2024, so
    # it wants the 2006-2024 forcing, not the 1961-2025 one.  Those products are unused by the arm
    # (they only feed the `_33` parity proof, which a refit arm skips) but the exporter still builds
    # and hashes them, so feeding it the wrong period would be a real error.
    module.main()


def step_spinup(arm: Arm) -> None:
    module = load(f"arm_{arm.label}_s33", FORKS / "s33_spinup.py")
    stage33 = arm.root / "stage33"
    module.RUN = stage33
    module.OUT = stage33 / "outputs"
    module.REPORTS = stage33 / "reports"
    module.LOCKS = stage33 / "locks"
    module.S9 = arm.outputs / "export"
    module.S31 = arm.work
    module.PARITY = False
    # The fork bound CHECKPOINT by value at import; keep it in step with the module the core will read.
    module.CHECKPOINT = arm.core.CHECKPOINT
    module.main()


def step_long_simulation(arm: Arm) -> None:
    stage34 = load(f"arm_{arm.label}_s34", SOURCE34)
    stage34.RUN = arm.root
    stage34.OUT = arm.outputs
    stage34.REPORTS = arm.root / "reports"
    stage34.LOCKS = arm.root / "locks"
    stage34.S31 = arm.work
    stage34.S33 = arm.root / "stage33"
    stage34.main()


def step_tn_interface(arm: Arm) -> None:
    """`20260828_35` loaded as code into the arm directory; the formal round is never written."""
    stage35 = load(f"arm_{arm.label}_s35", SOURCE35)
    stage35.RUN = arm.root
    stage35.OUT = arm.outputs
    stage35.REPORTS = arm.root / "reports"
    stage35.LOCKS = arm.root / "locks"
    stage35.S31 = arm.work
    stage35.S34 = arm.root
    stage35.main()


def step_evaluate(arm: Arm) -> None:
    """Score the arm's own product on all 102 stations and count its negatives.

    The observations are `a2g`'s frame, unmasked: the arm's excluded stations are still evaluated, so
    the report can say what excluding them *achieved* as well as what it cost.
    """
    a1 = load(f"arm_{arm.label}_a1", RUN / "scripts" / "a1_evaluate.py")
    a2g = load(f"arm_{arm.label}_a2g", RUN / "scripts" / "a2g_measure_base.py")
    stations, observations, _ = a2g.base_observations()

    daily = a1.read_daily(arm.outputs / "tn_hydrology_reach_daily.parquet")
    monthly = a1.read_monthly(arm.outputs / "tn_hydrology_reach_monthly.parquet")
    _, _, daily_metrics, monthly_metrics = a1.cohort_evaluation(
        stations, observations, daily, monthly, f"ARM_{arm.label}",
    )

    excluded = set(json.loads((arm.root / "excluded.json").read_text(encoding="utf-8"))["excluded"])
    merged = daily_metrics[["station_norm", "NSE"]].rename(columns={"NSE": "nse_daily"}).merge(
        monthly_metrics[["station_norm", "NSE"]].rename(columns={"NSE": "nse_monthly"}),
        on="station_norm", how="outer",
    )
    merged["nse_min"] = merged[["nse_daily", "nse_monthly"]].min(axis=1)
    merged["excluded"] = merged.station_norm.isin(excluded)

    # `excluded.json` is keyed by the *normalized* station key, while `station_norm` here is the table's
    # own raw spelling.  A silent mismatch between the two has already happened once this round
    # (`a2h_arm_inputs.py:257`: masking by normalized key against a raw column, where only the few names
    # that happen to be spelled identically still matched).  Measured on this base the two agree exactly
    # -- all 102 stations satisfy `norm(station_norm) == station_norm`, and all 8 arms' excluded lists
    # match verbatim -- but if that ever stopped holding, an unmatched name would be counted as
    # *retained*, inflating the very number `a2h_run_arms.answer()` selects the combination by.  `a3`
    # would not catch it either, because `a3` normalizes both of its own sides and stays self-consistent.
    # So an unmatched name is a failure here, not a detail to be averaged over.
    unmatched = sorted(excluded - set(merged.station_norm.astype(str)))
    if unmatched:
        raise RuntimeError(
            f"{arm.label}: {len(unmatched)} excluded names matched no scored station and would be "
            f"counted as retained: {unmatched}"
        )

    retained = merged.loc[~merged.excluded]
    retained_negative = retained.loc[retained.nse_min.lt(0)]
    excluded_negative = merged.loc[merged.excluded & merged.nse_min.lt(0)]
    result = {
        "stage": "20260917_2",
        "purpose": f"A2h: refit arm {arm.label} -- masked observations, full model rerun",
        "combination": arm.label,
        "excluded_count": len(excluded),
        "base_stations": int(len(merged)),
        "retained": int(len(retained)),
        "retained_negative": int(len(retained_negative)),
        "retained_negative_stations": sorted(retained_negative.station_norm.astype(str)),
        "retained_all_positive": bool(len(retained_negative) == 0),
        "worst_retained_nse": float(retained.nse_min.min()),
        "excluded_that_are_negative": int(len(excluded_negative)),
        "excluded_that_are_positive": int(len(merged.loc[merged.excluded]) - len(excluded_negative)),
        "excluded_positive_stations": sorted(
            merged.loc[merged.excluded & merged.nse_min.ge(0), "station_norm"].astype(str)
        ),
        "timings_seconds": arm.timings,
        "stations": merged.sort_values("nse_min").to_dict(orient="records"),
    }
    (arm.root / "arm_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8",
    )
    merged.to_parquet(arm.root / "arm_station_metrics.parquet", index=False, compression="zstd")

    print(f"  [{arm.label}] 保留 {result['retained']} 站，其中负 {result['retained_negative']} 站，"
          f"最负 {result['worst_retained_nse']:.4f}；被排除而原本为正的 {result['excluded_that_are_positive']} 站")


def main() -> None:
    global S5_PANEL, S8_PANEL

    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    parser.add_argument("--steps", default="")
    # Defaults are the arms' panels.  The V1/V2 fork-fidelity checks pass the frozen sizes (65/91) with
    # the frozen inputs in the arm's `inputs/`, which is how the forks are shown to still reproduce
    # `20260827_5` and `20260828_2` after forking.
    parser.add_argument("--s5-panel", type=int, default=S5_PANEL)
    parser.add_argument("--s8-panel", type=int, default=S8_PANEL)
    args = parser.parse_args()
    S5_PANEL, S8_PANEL = args.s5_panel, args.s8_panel

    arm_root = ARMS / args.arm
    if not arm_root.is_dir():
        raise RuntimeError(f"No such arm: {arm_root}")

    # Module identity, before any fork is loaded (see the docstring).
    core_dir = arm_root / "stage33" / "scripts"
    core_dir.mkdir(parents=True, exist_ok=True)
    core_path = core_dir / "frozen_hydrology_core.py"
    if not core_path.exists() or core_path.read_bytes() != S33_CORE.read_bytes():
        shutil.copy2(S33_CORE, core_path)
    sys.path.insert(0, str(core_dir))
    core = importlib.import_module("frozen_hydrology_core")
    if Path(core.__file__).resolve() != core_path.resolve():
        raise RuntimeError(f"frozen_hydrology_core resolved to {core.__file__}, not the arm's copy")
    core.CHECKPOINT = arm_root / "outputs" / "export" / "parent_preserving_state_consistent_model.pt"
    core.SCORE = arm_root / "outputs" / "stage8" / "regionalized_slow_score.parquet"
    print(f"[{args.arm}] 核心模块 = {core.__file__}")
    print(f"[{args.arm}] CHECKPOINT = {core.CHECKPOINT}")

    arm = Arm(args.arm, core)
    build_forcing_lock(arm)

    order = [
        ("01_parent_fit", step_parent_fit),
        ("02_regionalization", step_regionalization),
        ("03_checkpoint", step_checkpoint),
        ("04_spinup", step_spinup),
        ("05_long_simulation", step_long_simulation),
        ("06_tn_interface", step_tn_interface),
        ("07_evaluate", step_evaluate),
    ]
    wanted = {name.strip() for name in args.steps.split(",") if name.strip()}
    for name, action in order:
        if wanted and name not in wanted:
            continue
        arm.step(name, lambda action=action: action(arm))

    print(f"[{args.arm}] 全链完成")


if __name__ == "__main__":
    main()
