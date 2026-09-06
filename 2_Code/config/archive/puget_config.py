"""Config dict template for the Puget dataset run."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

BASE_DIR = Path(__file__).resolve().parents[2]
HOME_DATA = BASE_DIR / "1_Inputs"
HOME_RESULTS = BASE_DIR / "3_Validation"

# Fill the fields below with values from your SAS control file.
CONFIG: Dict[str, Any] = {
    "home_data": str(HOME_DATA),
    "home_results": str(HOME_RESULTS),
    "input_dir": str(HOME_DATA),
    "results_dir": str(HOME_RESULTS),
    "indata": "indata",
    "if_make_input_data": "yes",
    "optional_station_information": "",
    "optional_reach_information": "",
    "waterid": "",
    "arcid": "",
    "fnode": "",
    "tnode": "",
    "hydseq": "",
    "inc_area": "",
    "tot_area": "",
    "mean_flow": "",
    "frac": "",
    "iftran": "",
    "target": "",
    "ls_weight": "",
    "staid": "",
    "lat": "",
    "lon": "",
    "depvar": "",
    "srcvar": "",
    "dlvvar": "",
    "decvar": "",
    "resvar": "",
    "othvar": "",
    "betailst": "",
    "bsrcvar": "",
    "bdlvvar": "",
    "bdecvar": "",
    "bresvar": "",
    "bothvar": "",
    "reach_decay_specification": "",
    "reservoir_decay_specification": "",
    "incr_delivery_specification": "",
    "dlvdsgn": "",
    "data_modifications": "",
    "if_estimate": "yes",
    "if_predict": "yes",
    "if_parm_bootstrap": "no",
    "n_boot_iter": 0,
    "n_extra_jter": 0,
    "n_seeds": 0,
    "master_seed": 0,
    "if_output_to_tab": "no",
    "if_print_details": "no",
}


def get_config() -> Dict[str, Any]:
    """Return a copy of the config dict."""
    return dict(CONFIG)
