"""TN model config translated from sparrow_control_Puget_TN.sas."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from sparrow_py.preprocessing.puget_tn_mods import apply_puget_tn_modifications

BASE_DIR = Path(__file__).resolve().parents[2]
HOME_DATA = BASE_DIR / "1_Inputs"
HOME_RESULTS = BASE_DIR / "3_Validation" / "tn_py"

CONFIG: Dict[str, Any] = {
    "home_data": str(HOME_DATA),
    "home_results": str(HOME_RESULTS),
    "input_dir": str(HOME_DATA),
    "results_dir": str(HOME_RESULTS),
    "indata": "indata",
    "if_make_input_data": "yes",
    "if_estimate": "yes",
    "if_predict": "yes",
    "if_parm_bootstrap": "no",
    "n_boot_iter": 0,
    "start_iter": 0,
    "start_jter": 0,
    "end_iter": 0,
    "n_extra_jter": 0,
    "master_seed": 6727775,
    "n_seeds": 4,
    "cov_prob": 90,
    "if_init_beta_w_previous_est": "no",
    "if_adjust": "no",
    "if_exclude_inc_decay": "no",
    "n_periods": 64,
    "if_estimate_ic": "yes",
    "catchment_storage_source": 9,
    "catchment_storage_source_exclude": "1 2",
    "catchment_storage_source_exclud0": "1 2 9",
    "depvar": "load",
    "load_units": "kg/yr",
    "if_concentration_in_micrograms": "no",
    "ndt_per_year": 4,
    "srcvar": "foreign point1_kg cafos septic ag_kg atm_kg urb_km2 nfix_m2 storage",
    "bsrcvar": "bforeign bpoint1_kg bcafos bseptic bag_kg batm_kg burb_km2 bnfix_m2 bstorage",
    "dlvvar": "lppt lom lclay outfalls DeltaLogR",
    "bdlvvar": "blppt blom blclay boutfalls bDeltaLogR",
    "dlvdsgn": (
        "0 0 0 0 0, "
        "0 0 0 0 0, "
        "1 1 1 0 0, "
        "1 1 1 0 0, "
        "1 1 1 0 0, "
        "0 1 1 0 0, "
        "1 1 1 1 0, "
        "1 1 1 0 0, "
        "0 0 0 0 1"
    ),
    "if_mean_adjust_delivery_vars": "yes",
    "decvar": "strmload ltemp1",
    "bdecvar": "bstrmload bltemp1",
    "resvar": "resload",
    "bresvar": "bresload",
    "othvar": "",
    "bothvar": "",
    "betailst": (
        "bforeign 1.0 0:1.0 "
        "bpoint1_kg 1.0 0:1.0 "
        "bcafos 7.7 0:. "
        "bseptic 4.2 0:. "
        "bag_kg 0.08 0:. "
        "batm_kg 0.25 0:. "
        "burb_km2 791.0 0:. "
        "bnfix_m2 0.33 0:. "
        "bstorage 0.24 0:. "
        "blppt 1.47 .:. "
        "blom 0.84 .:. "
        "blclay -0.62 .:. "
        "boutfalls -0.47 .:. "
        "bDeltaLogR -0.39 .:. "
        "bstrmload 0.28 0:. "
        "bresload 4.0 0:. "
        "bltemp1 0.19 .:."
    ),
    "reach_decay_specification": "exp(-(data[:, jdecvar] @ beta[jbdecvar])><0)",
    "reservoir_decay_specification": "exp(-(data[:, jresvar] @ beta[jbresvar])><0)",
    "incr_delivery_specification": "exp((data[:, jdlvvar] * beta[jbdlvvar]) @ dlvdsgn.T)",
    "staid": "staid",
    "optional_station_information": (
        "comid cumarea flow_cfs year quarter period load station_id HUC12 WRIA "
        "TN_kfluxkg load_00600 sload_00600"
    ),
    "lat": "lat",
    "lon": "lon",
    "ls_weight": "ls_weight",
    "waterid": "time_comid",
    "optional_reach_information": "year quarter period station_id HUC12 WRIA",
    "inc_area": "incarea",
    "tot_area": "cumarea",
    "mean_flow": "flow_cfs",
    "if_flow_units_metric": "no",
    "arcid": "comid",
    "fnode": "time_cfromnode",
    "tnode": "time_ctonode",
    "hydseq": "time_hydroseq",
    "frac": "divfrac",
    "iftran": "iftran",
    "target": "termflag",
    "retrans_exclude_list": "del_frac",
    "calibrate_selection_criteria": "",
    "NLP_printing_option": 1,
    "calibration_max_nfev": 12000,
    "calibration_ftol": 1e-7,
    "calibration_xtol": 1e-7,
    "calibration_gtol": 1e-7,
    "calibration_solver": "least_squares",
    "calibration_max_iter": 20000,
    "if_test_calibrate": "no",
    "if_accumulate_with_dll": "no",
    "if_test_predict": "no",
    "test_obs": 2134,
    "if_print_boot_predictions": "no",
    "if_distribute_yield_by_land_use": "no",
    "land_class_list": "",
    "if_print_details": "yes",
    "if_output_to_tab": "yes",
    "predict_output_schema": "sas_tn",
    "sas_predict_template": str(BASE_DIR / "2_Code" / "config" / "templates" / "predict_puget_tn_header.csv"),
    "if_debug_predict_trace": "no",
    "debug_trace_time_comids": "",
    "debug_trace_periods": "3 7 11 15 19 23 27 31 35 39 43 47 51 55 59 63",
    "debug_trace_max_rows": 200,
    "if_gis": "no",
    "home_gis": "",
    "gis_file": "",
    "data_modifications": apply_puget_tn_modifications,
}


def get_config() -> Dict[str, Any]:
    """Return a copy of the config dict."""
    return dict(CONFIG)
