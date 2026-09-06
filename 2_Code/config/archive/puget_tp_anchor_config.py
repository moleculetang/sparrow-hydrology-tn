"""TP anchored-local-refinement validation config."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from sparrow_py.config.puget_tp_config import get_config as _get_base_config


BASE_DIR = Path(__file__).resolve().parents[2]
HOME_RESULTS = BASE_DIR / "3_Validation" / "tp_py_anchor2"

CONFIG: Dict[str, Any] = _get_base_config()
CONFIG["home_results"] = str(HOME_RESULTS)
CONFIG["results_dir"] = str(HOME_RESULTS)


def get_config() -> Dict[str, Any]:
    """Return a copy of the config dict."""
    return dict(CONFIG)
