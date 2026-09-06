"""Run one deterministic H7 start for the 2021--2025 sensitivity fit."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(r"E:\SPARROW")
HERE = ROOT / "5_Test/20260904_4/scripts"
sys.path.insert(0, str(HERE))
from f25_common import observations, save_trial  # noqa: E402
from unified_fit import FoldDesign, JointObjective, UnifiedTNModel, gamma_start, optimize, s41, starting_values  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=int, required=True, choices=range(5))
    args = parser.parse_args()
    if Path(sys.prefix).name.lower() != "sparrow":
        raise RuntimeError("sparrow environment required")
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(4)
    started = time.perf_counter()
    train = observations().loc[lambda frame: frame.year.between(2021, 2025)].copy()
    model = UnifiedTNModel("sensitivity")
    design = FoldDesign.build("H7", train)
    objective = JointObjective(model, design, train, [2021, 2022, 2023, 2024, 2025])
    processes, sites = starting_values(model, "T3", objective.stations)
    anchor_vf = float(processes[0][model.names().index("v_f")])
    result = optimize(
        objective, processes[args.variant], gamma_start(7, args.variant), sites[args.variant], anchor_vf,
        adam_steps=3, lbfgs_steps=6,
    )
    save_trial(args.variant, result)
    rss, peak = s41.s28.memory_gib()
    print(json.dumps({
        "status": "PASS_F25_START_CHECKPOINT", "variant": args.variant,
        "objective": result["objective"], "kkt": result["combined_kkt"],
        "runtime_seconds": time.perf_counter() - started, "rss_gib": rss, "peak_gib": peak,
    }), flush=True)


if __name__ == "__main__":
    main()
