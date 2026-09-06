"""TP model config translated from sparrow_control_Puget_TP.sas."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from sparrow_py.preprocessing.puget_tp_mods import apply_puget_tp_modifications

BASE_DIR = Path(__file__).resolve().parents[2]
HOME_DATA = BASE_DIR / "1_Inputs"
HOME_RESULTS = BASE_DIR / "3_Validation" / "tp_py"

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
    "if_adjust": "no",
    "if_exclude_inc_decay": "no",
    "n_periods": 64,
    "if_estimate_ic": "yes",
    "catchment_storage_source": 8,
    "catchment_storage_source_exclude": "1 2",
    "catchment_storage_source_exclud0": "1 2 8",
    "depvar": "load",
    "load_units": "kg/yr",
    "if_concentration_in_micrograms": "no",
    "ndt_per_year": 4,
    "srcvar": "foreign point1_kg natp_mt cafos septic ag_kg urb_km2 storage",
    "bsrcvar": "bforeign bpoint1_kg bnatp_mt bcafos bseptic bag_kg burb_km2 bstorage",
    "dlvvar": "lppt laet lden lkfact lclay bslope plppt plaet",
    "bdlvvar": "blppt blaet blden blkfact blclay bbslope bplppt bplaet",
    "dlvdsgn": (
        "0 0 0 0 0 0 0 0, "
        "0 0 0 0 0 0 0 0, "
        "1 0 1 0 0 1 0 0, "
        "1 0 1 1 1 1 0 0, "
        "0 1 1 1 1 1 0 0, "
        "1 1 1 1 1 1 0 0, "
        "1 0 1 1 1 1 0 0, "
        "0 0 0 0 0 0 1 1"
    ),
    "if_mean_adjust_delivery_vars": "yes",
    "decvar": "strmload1",
    "bdecvar": "bstrmload1",
    "resvar": "resload1",
    "bresvar": "bresload1",
    "othvar": "",
    "bothvar": "",
    "betailst": (
        "bforeign 1.0 1.0:1.0 "
        "bpoint1_kg 1.0 0:1.0 "
        "bnatp_mt 0.0028 0:. "
        "bcafos 2.20 0:. "
        "bseptic 0.07 0:. "
        "bag_kg 0.027 0.01:. "
        "burb_km2 80.0 0:. "
        "bstorage 0.30 0:. "
        "blppt 1.57 .:. "
        "blaet -0.55 .:. "
        "blden 10.80 .:. "
        "blkfact 6.14 .:. "
        "blclay -1.62 .:. "
        "bbslope 0.86 .:. "
        "bplppt -0.52 .:. "
        "bplaet -0.27 .:. "
        "bstrmload1 0.18 0:. "
        "bresload1 20.0 0:."
    ),
    "reach_decay_specification": "exp(-(data[:, jdecvar] @ beta[jbdecvar])><0)",
    "reservoir_decay_specification": "exp(-(data[:, jresvar] @ beta[jbresvar])><0)",
    "incr_delivery_specification": "exp((data[:, jdlvvar] * beta[jbdlvvar]) @ dlvdsgn.T)",
    "staid": "staid",
    "optional_station_information": "comid cumarea flow_cfs year quarter period load station_id HUC12 WRIA",
    "lat": "lat",
    "lon": "lon",
    "ls_weight": "ls_weight",
    "waterid": "time_comid",
    "optional_reach_information": "year quarter period load station_id HUC12 WRIA",
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
    "calibration_ftol": 1e-8,
    "calibration_xtol": 1e-8,
    "calibration_gtol": 1e-8,
    "calibration_solver": "least_squares",
    "calibration_max_iter": 20000,
    "if_tp_local_refine": "yes",
    "tp_local_refine_betas": "bnatp_mt burb_km2 blppt blden blkfact blclay bbslope bresload1",
    "tp_local_refine_max_nfev": 8000,
    "tp_local_refine_ftol": 1e-10,
    "tp_local_refine_xtol": 1e-10,
    "tp_local_refine_gtol": 1e-10,
    "tp_local_refine_anchor_weight": 1.0,
    "tp_local_refine_anchor_targets": {
        "bnatp_mt": 0.00302819807683,
        "burb_km2": 78.6920859057,
        "blppt": 1.48609290875,
        "blden": 10.1209852249,
        "blkfact": 6.25621770319,
        "blclay": -1.74781871371,
        "bbslope": 0.875291571514,
        "bresload1": 31.0,
    },
    "tp_local_refine_max_obj_increase_frac": 0.001,
    "if_test_calibrate": "no",
    "if_accumulate_with_dll": "no",
    "if_test_predict": "no",
    "test_obs": 2134,
    "if_print_boot_predictions": "no",
    "if_distribute_yield_by_land_use": "no",
    "land_class_list": "",
    "if_print_details": "yes",
    "if_output_to_tab": "yes",
    "predict_output_schema": "sas_tp",
    "sas_predict_template": str(BASE_DIR / "2_Code" / "config" / "templates" / "predict_puget_tp_header.csv"),
    "if_tp_predict_beta_override": "no",
    "tp_predict_beta_overrides": {},
    "if_debug_predict_trace": "no",
    "debug_trace_time_comids": "",
    "debug_trace_periods": "3 7 11 15 19 23 27 31 35 39 43 47 51 55 59 63",
    "debug_trace_max_rows": 200,
    "debug_del_frac_chain_time_comids": "",
    "debug_del_frac_chain_max_rows": 300,
    "if_gis": "no",
    "home_gis": "",
    "gis_file": "",
    "data_modifications": apply_puget_tp_modifications,
}


def get_config() -> Dict[str, Any]:
    """Return a copy of the config dict."""
    return dict(CONFIG)
