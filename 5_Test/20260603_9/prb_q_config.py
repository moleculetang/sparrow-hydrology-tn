from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd


RUN_DIR = Path(__file__).resolve().parent
ROOT = Path(r"E:\SPARROW")


def _resolve_col(df: pd.DataFrame, name: str) -> str:
    if name in df.columns:
        return name
    lower_map = {col.lower(): col for col in df.columns}
    return lower_map.get(name.lower(), name)


def _series(df: pd.DataFrame, name: str, default: float = np.nan) -> pd.Series:
    col = _resolve_col(df, name)
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(default, index=df.index, dtype=float)


def _safe_log(values: pd.Series) -> pd.Series:
    logged = np.log(values)
    logged = pd.Series(logged, index=values.index, dtype=float)
    return logged.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def _station_weight_map() -> dict[str, float]:
    path = RUN_DIR / "reports" / "station_empirical_bayes_weights.csv"
    if not path.exists():
        return {}
    weights = pd.read_csv(path, encoding="utf-8-sig")
    if "q_site" not in weights.columns or "empirical_bayes_weight" not in weights.columns:
        return {}
    return {
        str(row.q_site): float(row.empirical_bayes_weight)
        for row in weights.itertuples(index=False)
        if pd.notna(row.q_site) and pd.notna(row.empirical_bayes_weight)
    }


def _reservoir_influence_reach_ids() -> set[int]:
    path = ROOT / "0_reach_topology" / "results" / "tables" / "topology_edges.csv"
    if not path.exists():
        return set()
    topo = pd.read_csv(path, encoding="utf-8-sig")
    if "reach_id" not in topo.columns or "src_id" not in topo.columns or "downstream_reach" not in topo.columns:
        return set()
    is_reservoir = topo["src_id"].astype(str).str.contains("\u6c34\u5e93", na=False, regex=False)
    reservoir_ids = set(pd.to_numeric(topo.loc[is_reservoir, "reach_id"], errors="coerce").dropna().astype(int))
    immediate: dict[int, list[int]] = {}
    for row in topo[["reach_id", "downstream_reach"]].itertuples(index=False):
        rid = int(row.reach_id)
        downstream: list[int] = []
        if pd.notna(row.downstream_reach) and str(row.downstream_reach).strip():
            for token in str(row.downstream_reach).replace(";", ",").split(","):
                token = token.strip()
                if not token:
                    continue
                try:
                    downstream.append(int(float(token)))
                except ValueError:
                    pass
        immediate[rid] = downstream
    influenced = set(reservoir_ids)
    frontier = list(reservoir_ids)
    while frontier:
        current = frontier.pop()
        for nxt in immediate.get(current, []):
            if nxt not in influenced:
                influenced.add(nxt)
                frontier.append(nxt)
    return influenced


def apply_prb_q_modifications(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    fnode = _series(out, "time_cfromnode")
    tnode = _series(out, "time_ctonode")
    out = out.loc[(fnode > 0) & (tnode > 0)].copy()

    out["termflag"] = (_series(out, "TermFlag", default=0.0) == 1).astype(float)
    out["iftran"] = (_series(out, "LENGTHKM", default=0.0) > 0).astype(float)
    out.loc[out["termflag"] == 1, "iftran"] = 0.0
    out["divfrac"] = _series(out, "DivFrac", default=1.0).fillna(1.0)

    out["load"] = _series(out, "Q_obsv_cfs")
    out.loc[out["load"] <= 0, "load"] = np.nan
    ifmon = pd.Series(0.0, index=out.index, dtype=float)
    ifmon.loc[out["load"] > 0] = 1.0
    out["ifmon"] = ifmon
    out["ls_weight"] = np.where(ifmon == 1, 1.0, np.nan)
    station_weights = _station_weight_map()
    if station_weights and "q_site" in out.columns:
        mapped = out["q_site"].astype(str).map(station_weights)
        out.loc[(ifmon == 1) & mapped.notna(), "ls_weight"] = mapped[(ifmon == 1) & mapped.notna()]
    out["station_count"] = ifmon.cumsum()
    out["staid"] = np.where(ifmon == 1, out["station_count"], np.nan)

    out["cumarea"] = _series(out, "CumAreaKm2", default=0.0).fillna(0.0)
    out["incarea"] = _series(out, "IncAreaKm2", default=0.0).fillna(0.0)
    out["maflow_cfs"] = _series(out, "MAFlowUcfs", default=0.0).clip(lower=0).fillna(0.0)
    flow = _series(out, "Q_calc_cfs")
    out["flow_cfs"] = flow.where(flow > 0, out["maflow_cfs"]).fillna(0.0)

    ppt_depth = (_series(out, "PPT", default=0.0) - _series(out, "AET", default=0.0)).clip(lower=0.0)
    pre_ppt_depth = (_series(out, "prePPT", default=0.0) - _series(out, "preAET", default=0.0)).clip(lower=0.0)
    out["ppt"] = (ppt_depth * out["incarea"] * 0.004541).clip(lower=0.01).fillna(0.01)
    out["pppt"] = (pre_ppt_depth * out["incarea"] * 0.004541).clip(lower=0.01).fillna(0.01)
    pet_minus_aet = (_series(out, "PET", default=0.0) - _series(out, "AET", default=0.0)).clip(lower=0.0)
    pre_pet_minus_aet = (_series(out, "prePET", default=0.0) - _series(out, "preAET", default=0.0)).clip(lower=0.0)
    out["aet"] = (pet_minus_aet * out["incarea"] * 0.004541).clip(lower=0.01).fillna(0.01)
    out["paet"] = (pre_pet_minus_aet * out["incarea"] * 0.004541).clip(lower=0.01).fillna(0.01)
    out["lppt"] = _safe_log(out["ppt"])
    out["plppt"] = _safe_log(out["pppt"])
    out["laet"] = _safe_log(out["aet"])
    out["plaet"] = _safe_log(out["paet"])

    out["wd"] = 0.0
    out["storage"] = 0.0
    reservoir_ids = _reservoir_influence_reach_ids()
    reach_id = _series(out, "comid").astype("Int64")
    out["res_decay"] = reach_id.isin(reservoir_ids).astype(float)
    out = out.sort_values(["comid", "period"]).copy()
    out["res_lag"] = out.groupby("comid")["ppt"].shift(1).fillna(0.0)
    out.loc[out["res_decay"] <= 0, "res_lag"] = 0.0
    out = out.sort_values(["period", "time_hydroseq"]).copy()
    out["boundary_cfs"] = _series(out, "boundary_cfs", default=0.0).fillna(0.0)
    out["foreign_cfs"] = out["boundary_cfs"]
    out["point1_cfs"] = 0.0
    return out


CONFIG: Dict[str, Any] = {
    "home_data": str(RUN_DIR / "inputs"),
    "home_results": str(RUN_DIR / "outputs"),
    "input_dir": str(RUN_DIR / "inputs"),
    "results_dir": str(RUN_DIR / "outputs"),
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
    "if_adjust": "yes",
    "if_exclude_inc_decay": "no",
    "n_periods": 68,
    "if_estimate_ic": "yes",
    "catchment_storage_source": 4,
    "catchment_storage_source_exclude": "1 2",
    "catchment_storage_source_exclud0": "1 2 4",
    "depvar": "load",
    "load_units": "cfs",
    "if_concentration_in_micrograms": "no",
    "adjust_units": 1,
    "ndt_per_year": 4,
    "srcvar": "foreign_cfs point1_cfs ppt storage res_lag",
    "bsrcvar": "bforeign_cfs bpoint1_cfs bppt bstorage bres_lag",
    "dlvvar": "plaet",
    "bdlvvar": "bplaet",
    "dlvdsgn": "0, 0, 1, 1, 1",
    "if_mean_adjust_delivery_vars": "yes",
    "decvar": "wd",
    "bdecvar": "bwd",
    "resvar": "res_decay",
    "bresvar": "bres_decay",
    "othvar": "",
    "bothvar": "",
    "betailst": (
        "bforeign_cfs 1.0 1.0:1.0 "
        "bpoint1_cfs 1.0 1.0:1.0 "
        "bppt 0.40 0:. "
        "bstorage 0.40 0:. "
        "bres_lag 0.20 0:. "
        "bplaet -0.1 .:. "
        "bwd 0.0 0:0 "
        "bres_decay 0.05 0:."
    ),
    "reach_decay_specification": "exp(-(data[:, jdecvar] @ beta[jbdecvar]))",
    "reservoir_decay_specification": "exp(-(data[:, jresvar] @ beta[jbresvar]))",
    "incr_delivery_specification": "exp((data[:, jdlvvar] * beta[jbdlvvar]) @ dlvdsgn.T)",
    "staid": "staid",
    "optional_station_information": "year quarter period q_site HUC12 WRIA Q_calc_cfs Q_ma_cfs",
    "lat": "",
    "lon": "",
    "ls_weight": "ls_weight",
    "waterid": "time_comid",
    "optional_reach_information": "year quarter period q_site HUC12 WRIA Q_calc_cfs Q_ma_cfs",
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
    "if_test_calibrate": "no",
    "if_accumulate_with_dll": "no",
    "if_test_predict": "no",
    "test_obs": 0,
    "if_print_boot_predictions": "no",
    "if_distribute_yield_by_land_use": "no",
    "land_class_list": "",
    "if_print_details": "yes",
    "if_output_to_tab": "yes",
    "predict_output_schema": "sas_q_adjust",
    "if_gis": "no",
    "home_gis": "",
    "gis_file": "",
    "data_modifications": apply_prb_q_modifications,
}


def get_config() -> Dict[str, Any]:
    return dict(CONFIG)
