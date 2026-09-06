"""TP diagnostic override config for coefficient plus reservoir-decay validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from sparrow_py.config.puget_tp_override_config import get_config as _get_base_config


BASE_DIR = Path(__file__).resolve().parents[2]
HOME_RESULTS = BASE_DIR / "3_Validation" / "tp_py_override_decay_run"

CONFIG: Dict[str, Any] = _get_base_config()
CONFIG["home_results"] = str(HOME_RESULTS)
CONFIG["results_dir"] = str(HOME_RESULTS)
CONFIG["tp_predict_beta_overrides"] = dict(CONFIG["tp_predict_beta_overrides"])
CONFIG["tp_predict_beta_overrides"]["bresload1"] = 31.0


def get_config() -> Dict[str, Any]:
    """Return a copy of the config dict."""
    return dict(CONFIG)
