from __future__ import annotations

import importlib.util
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
BASE_PATH = RUN_DIR / "prb_q_config.py"
SPEC = importlib.util.spec_from_file_location("prb_q_config_base", BASE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load base config: {BASE_PATH}")
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)

CONFIG = dict(BASE.CONFIG)
CONFIG["home_results"] = str(RUN_DIR / "outputs_pilot_equal_weight")
CONFIG["results_dir"] = str(RUN_DIR / "outputs_pilot_equal_weight")
