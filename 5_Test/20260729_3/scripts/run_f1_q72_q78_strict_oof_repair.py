from __future__ import annotations

import importlib.util
from pathlib import Path


RUN = Path(__file__).resolve().parents[1]
SOURCE = RUN.parent / "20260729_2" / "scripts" / "run_f1_q72_q78_strict_oof.py"


def main() -> None:
    spec = importlib.util.spec_from_file_location("f1_repaired_source", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.RUN = RUN
    module.OUT = RUN / "reports" / "f1_q72_q78_strict_oof"
    module.main()


if __name__ == "__main__":
    main()
