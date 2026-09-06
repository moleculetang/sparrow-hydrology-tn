"""Puget TN-specific preprocessing translated from sparrow_control_Puget_TN.sas."""

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


def _safe_log(values: pd.Series, floor: float | None = None) -> pd.Series:
    vals = pd.to_numeric(values, errors="coerce")
    if floor is not None:
        vals = vals.mask(vals <= 0, floor)
    with np.errstate(divide="ignore", invalid="ignore"):
        logged = np.log(vals)
    logged = pd.Series(logged, index=values.index, dtype=float)
    return logged.replace([np.inf, -np.inf], np.nan)


def apply_puget_tn_modifications(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    fnode = _series(out, "time_cfromnode")
    tnode = _series(out, "time_ctonode")
    out = out.loc[(fnode > 0) & (tnode > 0)].copy()

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

    out["div_transfer"] = _series(out, "div_transfer", default=0.0).fillna(0.0)
    out["divfrac"] = _series(out, "DivFrac", default=1.0)
    out.loc[comid == 23980809, "divfrac"] = out.loc[comid == 23980809, "div_transfer"]
    out.loc[comid == 971091491, "divfrac"] = out.loc[comid == 971091491, "div_transfer"]
    out.loc[comid == 24281992, "divfrac"] = 0.695
    out.loc[comid == 24285514, "divfrac"] = 0.242
    out.loc[comid == 24537920, "divfrac"] = 0.785
    out.loc[comid == 23970777, "divfrac"] = 0.874
    out.loc[comid == 23977680, "divfrac"] = 0.956
    out.loc[comid == 23963709, "divfrac"] = 0.932
    out.loc[comid == 24534418, "divfrac"] = 0.946
    out.loc[comid == 24270338, "divfrac"] = 0.927
    out.loc[comid == 23955834, "divfrac"] = 1.0
    out.loc[comid == 23955772, "divfrac"] = 1.0
    out.loc[comid == 24287222, "divfrac"] = 0.378
    out.loc[comid == 24534424, "divfrac"] = 0.984
    out.loc[comid == 24279130, "divfrac"] = 0.996
    out.loc[comid == 24286854, "divfrac"] = 0.976
    out.loc[comid == 24286882, "divfrac"] = 0.995
    out.loc[comid == 23997286, "divfrac"] = 0.995

    out["load"] = _series(out, "TN_kfluxkg")
    load_00600 = _series(out, "load_00600")
    sload_00600 = _series(out, "sload_00600")
    out.loc[load_00600.isna(), "load"] = np.nan
    out.loc[(load_00600 > 0) & (out["load"].isna()), "load"] = load_00600

    for cid in (
        24285494, 24287056, 23956550, 24274377, 24001453, 24001227, 23970845, 23970171, 23970271,
        23970349, 23970215, 23977636, 24537908, 24538374, 24538324, 24538338, 23990455, 23990503,
        23977824, 24538318, 23970727, 24538380, 24538082, 23971053, 23970867,
    ):
        out.loc[comid == cid, "load"] = load_00600
    out.loc[comid == 23977990, "load"] = np.nan
    out.loc[(load_00600 > 0) & (sload_00600 / load_00600 > 0.5), "load"] = np.nan

    out["cumarea"] = _series(out, "CumAreaKm2", default=0.0).fillna(0.0)
    out["incarea"] = _series(out, "IncAreaKm2", default=0.0).fillna(0.0)

    maflow_cfs = _series(out, "Q_ma_cfs")
    maflow_fallback = _series(out, "MAFlowUcfs")
    maflow_cfs = maflow_cfs.where(maflow_cfs.notna(), maflow_fallback)
    maflow_cfs = maflow_cfs.mask(maflow_cfs <= 0, 0.0).fillna(0.0)
    out["maflow_cfs"] = maflow_cfs

    flow_cfs = _series(out, "Q_calc_cfs")
    flow_cfs = flow_cfs.mask(flow_cfs <= 0, np.nan).where(flow_cfs.notna(), maflow_cfs)
    flow_cfs = flow_cfs.mask(flow_cfs <= 0, 0.0).fillna(0.0)
    out["flow_cfs"] = flow_cfs
    out["rQ"] = np.where(maflow_cfs > 0, flow_cfs / maflow_cfs, 0.0)

    length_m = _series(out, "LENGTHKM", default=0.0).fillna(0.0) * 1000.0
    slope = _series(out, "SLOPE", default=0.0).fillna(0.0).mask(lambda s: s < 0, 0.0)
    tarea = out["cumarea"] * (1000.0**2)
    flowi = flow_cfs / (3.2808**3)
    flowa = maflow_cfs / (3.2808**3)
    g = 9.8

    da = np.where(flowa > 0, ((tarea**1.25) * (g**0.5)) / flowa, 0.0)
    vel = np.where(
        (tarea > 0) & (out["rQ"] > 0),
        0.02 + (0.051 * (da**0.821) * (out["rQ"] ** (-0.465)) * (flowi / tarea)),
        0.0,
    )
    vel = vel * 3.2808
    velso = np.where(
        (tarea > 0) & (out["rQ"] > 0),
        0.094 + (0.0143 * (da**0.919) * (out["rQ"] ** (-0.469)) * (slope**0.159) * (flowi / tarea)),
        0.0,
    )
    velso = velso * 3.2808
    velocity = np.where(slope <= 0, vel, velso)
    velocity = np.nan_to_num(velocity, nan=0.0, posinf=0.0, neginf=0.0)
    out["velocity"] = velocity

    rchtot = np.where(velocity > 0, (length_m * 3.2808) / (velocity * 86400.0), 0.0)
    rchtot = np.nan_to_num(rchtot, nan=0.0, posinf=0.0, neginf=0.0)
    out["rchtot"] = np.where(rchtot < 0, 0.0, rchtot)

    depth = np.where(flow_cfs > 0, 0.0635 * (flow_cfs**0.3966), 0.0)
    strmload = np.where(depth > 0, out["rchtot"] / depth, 0.0)
    strmload = np.nan_to_num(strmload, nan=0.0, posinf=0.0, neginf=0.0)

    wb_areao = _series(out, "WB_AreaKm2_o", default=0.0).fillna(0.0) * (1000.0**2)
    wb_area = _series(out, "SurfAre", default=0.0).fillna(0.0).mask(lambda s: s < 0, 0.0)
    wb_area = wb_area.where(~((wb_area == 0) & (wb_areao > 0)), wb_areao)
    resload = np.where(flow_cfs > 0, wb_area / (flow_cfs * 893033.0), 0.0)
    resload = np.nan_to_num(resload, nan=0.0, posinf=0.0, neginf=0.0)
    wb_comid = _series(out, "WB_Comid")
    year = _series(out, "year")
    resload = np.where((wb_comid == 23999554) & (year > 2013), 0.0, resload)

    strmload = np.where((resload > 0) | (wb_area > 0), 0.0, strmload)
    out["strmload"] = strmload
    out["resload"] = resload

    out.loc[out["load"] <= 0, "load"] = np.nan
    out.loc[flow_cfs <= 0, "load"] = np.nan
    ifmon = (out["load"] > 0).astype(float)
    out["ifmon"] = ifmon
    out["ls_weight"] = np.where(ifmon == 1, 1.0, np.nan)
    quarter = _series(out, "quarter")
    out.loc[(ifmon == 1) & (quarter == 1), "ls_weight"] = 23.16
    out.loc[(ifmon == 1) & (quarter == 2), "ls_weight"] = 20.18
    out.loc[(ifmon == 1) & (quarter == 3), "ls_weight"] = 14.84
    out.loc[(ifmon == 1) & (quarter == 4), "ls_weight"] = 22.85
    out["station_count"] = ifmon.cumsum()
    out["staid"] = np.where(ifmon == 1, out["station_count"], np.nan)

    wwtp1 = _series(out, "TN_ps_kg_4952", default=0.0).fillna(0.0)
    wwtp2 = _series(out, "TN_ps_kg_0921", default=0.0).fillna(0.0)
    wwtp3 = _series(out, "TN_ps_kg_INDU", default=0.0).fillna(0.0)
    out["wwtp1_kg"] = wwtp1
    out["wwtp2_kg"] = wwtp2
    out["wwtp3_kg"] = wwtp3
    out["point2_kg"] = wwtp2 + wwtp3
    out["point3_kg"] = wwtp3
    out["point1_kg"] = wwtp1 + wwtp2 + wwtp3

    urb = _series(out, "urb", default=0.0).fillna(0.0).mask(lambda s: s < 0, 0.0)
    out["urb_km2"] = urb * out["incarea"] / 100.0 * 0.25

    ra = _series(out, "RedAlder_m2", default=0.0).fillna(0.0).mask(lambda s: s < 0, 0.0)
    out["nfix_m2"] = ra * 0.5

    out["ag_kg"] = _series(out, "wsda_tn_kg", default=0.0).fillna(0.0)
    atm_wet = _series(out, "TIN_atm", default=0.0).fillna(0.0).mask(lambda s: s <= 0, 0.0)
    out["atm_wet_kg"] = atm_wet
    out["atm_kg"] = atm_wet

    temp = _series(out, "TAV", default=0.0).fillna(0.0).mask(lambda s: s < 0, 0.0)
    out["temp"] = temp
    out["ltemp"] = _safe_log(temp).fillna(0.0)

    aet = _series(out, "AET", default=0.0).fillna(0.0).mask(lambda s: s <= 0, 0.01)
    paet = _series(out, "preAET", default=0.0).fillna(0.0).mask(lambda s: s <= 0, 0.01)
    ppt = _series(out, "PPT", default=0.0).fillna(0.0).mask(lambda s: s <= 0, 0.01)
    pppt = _series(out, "prePPT", default=0.0).fillna(0.0).mask(lambda s: s <= 0, 0.01)
    out["aet"] = aet
    out["laet"] = _safe_log(aet, floor=0.01).fillna(0.0)
    out["paet"] = paet
    out["plaet"] = _safe_log(paet, floor=0.01).fillna(0.0)
    out["ppt"] = ppt
    out["lppt"] = _safe_log(ppt, floor=0.01).fillna(0.0)
    out["pppt"] = pppt
    out["plppt"] = _safe_log(pppt, floor=0.01).fillna(0.0)

    period = _series(out, "period", default=0.0)
    delta_log_r = out["laet"] - out["plaet"]
    delta_log_r = np.where(period == 1, 0.0, delta_log_r)
    out["DeltaLogR"] = np.nan_to_num(delta_log_r, nan=0.0)

    tile = _series(out, "tile_hr", default=0.0).fillna(0.0)
    out["tile"] = tile
    out["ltile"] = np.sqrt(tile)

    clay = _series(out, "clay", default=0.0).fillna(0.0)
    out["lclay"] = np.log(clay + 1.0)
    om = _series(out, "OM", default=0.0).fillna(0.0)
    out["lom"] = _safe_log(om).fillna(0.0)

    wtemp = _series(out, "wtemp")
    lwtemp = _safe_log(wtemp)
    lmeantemp = 2.089
    ltemp0 = lwtemp.fillna(lmeantemp) - lmeantemp
    out["ltemp1"] = (ltemp0 * out["strmload"]).fillna(0.0)

    cafos = _series(out, "N_animals", default=0.0).fillna(0.0) * 0.25
    out["cafos"] = cafos.mask(cafos < 0, 0.0)
    out["septic"] = _series(out, "septic_n", default=0.0).fillna(0.0) * 0.25

    outfall_n = _series(out, "outfall_n", default=0.0).fillna(0.0)
    outfalls = np.where(out["incarea"] > 0, outfall_n / out["incarea"], 0.0)
    pptm = _series(out, "PPTm", default=np.nan)
    outfalls = outfalls * np.where(pptm > 0, out["ppt"] / pptm, 0.0)
    outfalls = pd.Series(outfalls, index=out.index, dtype=float)
    out["outfalls"] = _safe_log(outfalls).fillna(0.0)

    out["boundary"] = _series(out, "boundary_kg_tn", default=0.0).fillna(0.0)
    out["foreign"] = out["boundary"]
    out["storage"] = 0.0
    return out
