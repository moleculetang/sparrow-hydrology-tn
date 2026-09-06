"""TP diagnostic override config for prediction-only validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from sparrow_py.config.puget_tp_config import get_config as _get_base_config


BASE_DIR = Path(__file__).resolve().parents[2]
HOME_RESULTS = BASE_DIR / "3_Validation" / "tp_py_override_run"

CONFIG: Dict[str, Any] = _get_base_config()
CONFIG["home_results"] = str(HOME_RESULTS)
CONFIG["results_dir"] = str(HOME_RESULTS)
CONFIG["if_make_input_data"] = "no"
CONFIG["if_estimate"] = "no"
CONFIG["if_tp_local_refine"] = "no"
CONFIG["if_tp_predict_beta_override"] = "yes"
CONFIG["tp_predict_beta_overrides"] = {
    "bnatp_mt": 0.00302819807683,
    "burb_km2": 78.6920859057,
    "blppt": 1.48609290875,
    "blden": 10.1209852249,
    "blkfact": 6.25621770319,
    "blclay": -1.74781871371,
    "bbslope": 0.875291571514,
}


def get_config() -> Dict[str, Any]:
    """Return a copy of the config dict."""
    return dict(CONFIG)
