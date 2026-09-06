"""TP root-fix calibration test config."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from sparrow_py.config.puget_tp_config import get_config as _get_base_config


BASE_DIR = Path(__file__).resolve().parents[2]
HOME_RESULTS = BASE_DIR / "3_Validation" / "tp_py_rootfix"

CONFIG: Dict[str, Any] = _get_base_config()
CONFIG["home_results"] = str(HOME_RESULTS)
CONFIG["results_dir"] = str(HOME_RESULTS)
CONFIG["betailst"] = (
    "bforeign 1.0 1.0:1.0 "
    "bpoint1_kg 1.0 0:1.0 "
    "bnatp_mt 0.00302819807683 0:. "
    "bcafos 2.20 0:. "
    "bseptic 0.07 0:. "
    "bag_kg 0.027 0.01:. "
    "burb_km2 78.6920859057 0:. "
    "bstorage 0.30 0:. "
    "blppt 1.48609290875 .:. "
    "blaet -0.55 .:. "
    "blden 10.1209852249 .:. "
    "blkfact 6.25621770319 .:. "
    "blclay -1.74781871371 .:. "
    "bbslope 0.875291571514 .:. "
    "bplppt -0.52 .:. "
    "bplaet -0.27 .:. "
    "bstrmload1 0.18 0:. "
    "bresload1 31.0 0:."
)


def get_config() -> Dict[str, Any]:
    """Return a copy of the config dict."""
    return dict(CONFIG)
