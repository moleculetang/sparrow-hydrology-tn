"""One deterministic start for a Stage 4 temporal OOF fit."""

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
from temporal_common import FOLDS, observations, save_trial  # noqa: E402
from unified_fit import FoldDesign, JointObjective, UnifiedTNModel, gamma_start, optimize, s41, starting_values  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capacity", choices=["H7", "H14", "H22"], required=True)
    parser.add_argument("--fold", choices=list(FOLDS), required=True)
    parser.add_argument("--variant", type=int, choices=range(5), required=True)
    args = parser.parse_args()
    if Path(sys.prefix).name.lower() != "sparrow":
        raise RuntimeError("sparrow environment required")
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(4)
    started = time.perf_counter()
    training_years, _ = FOLDS[args.fold]
    train = observations().loc[lambda frame: frame.year.isin(training_years)].copy()
    model = UnifiedTNModel("formal")
    design = FoldDesign.build(args.capacity, train)
    objective = JointObjective(model, design, train, training_years)
    process, sites = starting_values(model, args.fold, objective.stations)
    anchor_vf = float(process[0][model.names().index("v_f")])
    result = optimize(
        objective, process[args.variant], gamma_start(len(design.fields), args.variant), sites[args.variant],
        anchor_vf, adam_steps=3, lbfgs_steps=6,
    )
    save_trial(args.capacity, args.fold, args.variant, result)
    rss, peak = s41.s28.memory_gib()
    print(json.dumps({
        "status": "PASS_TEMPORAL_START_CHECKPOINT", "capacity": args.capacity, "fold": args.fold,
        "variant": args.variant, "objective": result["objective"], "kkt": result["combined_kkt"],
        "runtime_seconds": time.perf_counter() - started, "rss_gib": rss, "peak_gib": peak,
    }), flush=True)


if __name__ == "__main__":
    main()
