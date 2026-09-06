"""Puget Q-specific preprocessing translated from SAS data_modifications."""

from __future__ import annotations

import numpy as np
import pandas as pd


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


def apply_puget_q_modifications(df: pd.DataFrame) -> pd.DataFrame:
    """Apply Q-model preprocessing logic equivalent to control-file SAS statements."""
    out = df.copy()

    fnode = _series(out, "time_cfromnode")
    tnode = _series(out, "time_ctonode")
    keep = (fnode > 0) & (tnode > 0)
    out = out.loc[keep].copy()

    comid = _series(out, "comid")
    termflag = _series(out, "TermFlag")
    lengthkm = _series(out, "LENGTHKM")
    fl_fcode = _series(out, "FL_FCode")

    out["termflag"] = (termflag == 1).astype(float)

    out["iftran1"] = 1.0
    out["iftran"] = (out["iftran1"] == 1).astype(float)
    out["iftran"] = (lengthkm > 0).astype(float)
    out["iftran"] = (termflag == 0).astype(float)
    out["iftran"] = (fl_fcode != 56600).astype(float)

    div_transfer = _series(out, "div_transfer", default=0.0).fillna(0.0)
    out["div_transfer"] = div_transfer
    out["divfrac"] = _series(out, "DivFrac")

    out.loc[comid == 23980809, "divfrac"] = out.loc[comid == 23980809, "div_transfer"]
    out.loc[comid == 971091491, "divfrac"] = out.loc[comid == 971091491, "div_transfer"]
    out.loc[comid == 24281992, "divfrac"] = 0.695
    out.loc[comid == 24287222, "divfrac"] = 0.378
    out.loc[comid == 24285514, "divfrac"] = 0.242
    out.loc[comid == 24537920, "divfrac"] = 0.785
    out.loc[comid == 23970777, "divfrac"] = 0.874
    out.loc[comid == 23977680, "divfrac"] = 0.956
    out.loc[comid == 23963709, "divfrac"] = 0.932
    out.loc[comid == 24534418, "divfrac"] = 0.946
    out.loc[comid == 24270338, "divfrac"] = 0.927
    out.loc[comid == 23955834, "divfrac"] = 1.0
    out.loc[comid == 23955772, "divfrac"] = 1.0
    out.loc[comid == 24534424, "divfrac"] = 0.984
    out.loc[comid == 24279130, "divfrac"] = 0.996
    out.loc[comid == 24286854, "divfrac"] = 0.976
    out.loc[comid == 24286882, "divfrac"] = 0.995
    out.loc[comid == 23997286, "divfrac"] = 0.995

    out["wd"] = 0.0

    out["load"] = np.nan
    q_obsv_cfs = _series(out, "Q_obsv_cfs")
    out["load"] = q_obsv_cfs
    out.loc[comid == 971091491, "load"] = np.nan
    out.loc[out["load"] <= 0, "load"] = np.nan

    ifmon = pd.Series(0.0, index=out.index, dtype=float)
    ifmon.loc[out["load"] > 0] = 1.0
    out["ifmon"] = ifmon
    out["ls_weight"] = np.where(ifmon == 1, 1.0, np.nan)
    out.loc[(out["load"] > 0) & (out["ls_weight"].isna()), "ls_weight"] = 1.0
    out["station_count"] = ifmon.cumsum()
    out["staid"] = np.where(ifmon == 1, out["station_count"], np.nan)

    out["cumarea"] = _series(out, "CumAreaKm2", default=0.0).fillna(0.0)
    out["incarea"] = _series(out, "IncAreaKm2", default=0.0).fillna(0.0)

    maflow_cfs = _series(out, "MAFlowUcfs")
    maflow_cfs = maflow_cfs.mask(maflow_cfs <= 0, 0.0).fillna(0.0)
    out["maflow_cfs"] = maflow_cfs

    flow_cfs = maflow_cfs.copy()
    flow_cfs = flow_cfs.mask(flow_cfs <= 0, 0.0).fillna(0.0)
    out["flow_cfs"] = flow_cfs

    out["resload"] = 0.0

    wwtp1 = _series(out, "Flow_mgd_4952", default=0.0).fillna(0.0)
    wwtp3 = _series(out, "Flow_mgd_INDU", default=0.0).fillna(0.0)
    out["wwtp1_kg"] = wwtp1
    out["wwtp3_kg"] = wwtp3
    out["point1_cfs"] = (wwtp1 + wwtp3) * 1.55093 / 3.0

    ppt = _series(out, "PPT", default=0.0) - _series(out, "AET", default=0.0)
    ppt = ppt.mask(ppt <= 0, 0.0)
    pppt = _series(out, "prePPT", default=0.0) - _series(out, "preAET", default=0.0)
    pppt = pppt.mask(pppt <= 0, 0.0)

    aet0 = _series(out, "PET", default=0.0) - _series(out, "AET", default=0.0)
    aet = aet0 * out["incarea"] * 0.004541
    aet = aet.fillna(0.0)
    aet = aet.mask(aet <= 0, 0.01)
    out["aet"] = aet
    out["laet"] = _safe_log(aet)

    paet0 = _series(out, "prePET", default=0.0) - _series(out, "preAET", default=0.0)
    paet = paet0 * out["incarea"] * 0.004541
    paet = paet.fillna(0.0)
    paet = paet.mask(paet <= 0, 0.01)
    out["paet"] = paet
    out["plaet"] = _safe_log(paet)

    ppt = (ppt * out["incarea"] * 0.004541).fillna(0.0)
    ppt = ppt.mask(ppt <= 0, 0.01)
    out["ppt"] = ppt
    out["lppt"] = _safe_log(ppt)

    pppt = (pppt * out["incarea"] * 0.004541).fillna(0.0)
    pppt = pppt.mask(pppt <= 0, 0.01)
    out["pppt"] = pppt
    out["plppt"] = _safe_log(pppt)

    out["storage"] = 0.0

    boundary_cfs = _series(out, "boundary_cfs", default=0.0).fillna(0.0)
    out["boundary_cfs"] = boundary_cfs
    out["foreign_cfs"] = boundary_cfs

    return out
