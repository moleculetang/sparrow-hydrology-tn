from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit, logit


RUN_DIR = Path(__file__).resolve().parents[2]
INPUT_PATH = RUN_DIR / "inputs" / "indata.parquet"
TOPOLOGY_PATH = RUN_DIR / "inputs" / "topology" / "topology_edges.csv"
REPORT_DIR = RUN_DIR / "reports" / "component_model30"
FIG_DIR = RUN_DIR / "figure" / "component_model30"
EPS = 1.0e-6

CAL_END_YEAR = 2018
INNER_TRAIN_END_YEAR = 2015
STATE_CALENDAR_MODES = {"observed_only", "full_forcing"}
# Retain the parent component's observed-only behavior unless a caller
# explicitly requests the repaired state calendar.  The 20260810_4 fold
# runner sets this to ``full_forcing``.
STATE_CALENDAR_MODE = "observed_only"
FORCING_SEMANTICS_MODES = {"legacy_mixed_et", "prescribed_aet_balance"}
MASS_ACCOUNTING_MODES = {"legacy_local_scaled", "explicit_upstream_volume"}
FORCING_SEMANTICS_MODE = "legacy_mixed_et"
MASS_ACCOUNTING_MODE = "legacy_local_scaled"
ET_STATE_OPERATOR_MODES = {"baseline_clip", "storage_withdrawal"}
# ``baseline_clip`` is the R=max(P-AET, 0) control.  ``storage_withdrawal``
# applies D=max(AET-P, 0) to each slow state store before that store releases.
ET_STATE_OPERATOR_MODE = "baseline_clip"
ET_FEATURE_BLOCK_MODES = {"full", "state_only_et"}
ET_FEATURE_BLOCK_MODE = "full"
AET_SOURCE = "ERA5"
SCENARIO_ID = "E00"

# 20260812_3 corrected-accounting experiment controls.  I0 and I1 share
# every setting except DYNAMIC_BETA_W.  beta=0 is the exact fixed-c=0.35
# corrected control; I1 estimates the single state-dependence coefficient.
ACCOUNTING_REPAIR_MODE = True
DYNAMIC_BETA_W = 0.0
QMA_TRAIN_THRESHOLD_CFS = np.nan

# The frozen five-scenario mechanism.  ERA5/PML select the forcing panel in
# the runner; the component owns only state and feature-block semantics.
ET_SCENARIOS = {
    "E00": {"forcing": "ERA5", "operator": "baseline_clip", "feature_block": "full"},
    "E10": {"forcing": "PML", "operator": "baseline_clip", "feature_block": "full"},
    "E01": {"forcing": "ERA5", "operator": "storage_withdrawal", "feature_block": "full"},
    "E11": {"forcing": "PML", "operator": "storage_withdrawal", "feature_block": "full"},
    "E00-F0": {"forcing": "ERA5", "operator": "baseline_clip", "feature_block": "state_only_et"},
}


def setup_style() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 160


def kge_2012(obs: np.ndarray, pred: np.ndarray) -> float:
    if len(obs) < 2 or np.mean(obs) == 0 or np.std(obs) == 0:
        return np.nan
    r = np.corrcoef(obs, pred)[0, 1] if np.std(pred) > 0 else np.nan
    alpha = np.std(pred) / np.std(obs) if np.std(obs) > 0 else np.nan
    beta = np.mean(pred) / np.mean(obs) if np.mean(obs) != 0 else np.nan
    if not np.isfinite(r) or not np.isfinite(alpha) or not np.isfinite(beta):
        return np.nan
    return float(1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def metric_dict(obs: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(obs) & np.isfinite(pred) & (obs > 0) & (pred > 0)
    obs = obs[mask]
    pred = pred[mask]
    if len(obs) == 0:
        return {
            "n": 0,
            "NSE_raw": np.nan,
            "NSE_log": np.nan,
            "KGE_2012": np.nan,
            "PBIAS_pct": np.nan,
            "trend_r": np.nan,
            "amplitude_ratio": np.nan,
            "peak_pct_error": np.nan,
        }
    err = pred - obs
    sst = np.sum((obs - np.mean(obs)) ** 2)
    log_obs = np.log(obs)
    log_pred = np.log(pred)
    log_sst = np.sum((log_obs - np.mean(log_obs)) ** 2)
    obs_peak_i = int(np.argmax(obs))
    trend_r = np.corrcoef(obs, pred)[0, 1] if len(obs) > 1 and np.std(pred) > 0 and np.std(obs) > 0 else np.nan
    return {
        "n": int(len(obs)),
        "NSE_raw": float(1 - np.sum(err**2) / sst) if sst > 0 else np.nan,
        "NSE_log": float(1 - np.sum((log_pred - log_obs) ** 2) / log_sst) if log_sst > 0 else np.nan,
        "KGE_2012": kge_2012(obs, pred),
        "PBIAS_pct": float(100 * np.sum(err) / np.sum(obs)) if np.sum(obs) != 0 else np.nan,
        "trend_r": float(trend_r) if np.isfinite(trend_r) else np.nan,
        "amplitude_ratio": float(np.std(pred) / np.std(obs)) if np.std(obs) > 0 else np.nan,
        "peak_pct_error": float(100 * (pred[obs_peak_i] - obs[obs_peak_i]) / max(obs[obs_peak_i], EPS)),
    }


def station_median_metrics(frame: pd.DataFrame, pred: np.ndarray) -> dict[str, float]:
    tmp = frame[["q_site", "Q_obsv_cfs"]].copy()
    tmp["predict"] = pred
    rows = []
    for _, part in tmp.groupby("q_site", sort=False):
        md = metric_dict(part["Q_obsv_cfs"].to_numpy(dtype=float), part["predict"].to_numpy(dtype=float))
        rows.append(md)
    metrics = pd.DataFrame(rows)
    metrics["abs_PBIAS"] = metrics["PBIAS_pct"].abs()
    metrics["good"] = (
        (metrics["n"] >= 24)
        & (metrics["NSE_log"] >= 0.65)
        & (metrics["KGE_2012"] >= 0.50)
        & (metrics["abs_PBIAS"] <= 25.0)
    )
    return {
        "median_NSE_log": float(metrics["NSE_log"].median()),
        "median_KGE": float(metrics["KGE_2012"].median()),
        "median_abs_PBIAS": float(metrics["abs_PBIAS"].median()),
        "median_alpha": float(metrics["amplitude_ratio"].median()),
        "good_count": int(metrics["good"].sum()),
    }


def observation_mask(frame: pd.DataFrame) -> pd.Series:
    """Return rows eligible for fitting/scoring, never rows eligible for state time."""
    return frame["Q_obsv_cfs"].notna() & frame["Q_obsv_cfs"].gt(0)


def load_forcing_panel() -> pd.DataFrame:
    """Load the complete reach-month forcing panel without filtering on Q.

    ``indata.parquet`` is the 230 x 204 climate/covariate backbone with sparse
    discharge observations merged onto it.  Hydrologic states must use this
    calendar; observations are selected only after state calculation.
    """
    cols = [
        "comid",
        "year",
        "month",
        "quarter",
        "period",
        "q_site",
        "Q_obsv_cfs",
        "CumAreaKm2",
        "IncAreaKm2",
        "Q_calc_cfs",
        "Q_ma_cfs",
        "connected_fraction",
        "connectivity_score_fold_z",
        "connected_local_cfs",
        "unconnected_local_cfs",
        "connected_upstream_cfs",
        "unconnected_upstream_cfs",
        "PPT",
        "AET",
        "PET",
        "explicit_upstream_net_cfs",
        "explicit_upstream_threshold_cfs",
    ]
    all_df = pd.read_parquet(INPUT_PATH)
    df = all_df[[c for c in cols if c in all_df.columns]].copy()
    for col in df.columns:
        if col != "q_site":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df["year"].between(2006, 2022) & df["month"].between(1, 12)].copy()
    df["date"] = pd.to_datetime({"year": df["year"].astype(int), "month": df["month"].astype(int), "day": 1})
    return df.sort_values(["comid", "year", "month"]).reset_index(drop=True)


def load_observed_panel() -> pd.DataFrame:
    """Legacy-compatible observation panel used by D0 reproduction only."""
    df = load_forcing_panel()
    df = df.loc[observation_mask(df)].copy()
    df["q_site"] = df["q_site"].astype(str)
    return df.sort_values(["comid", "year", "month"]).reset_index(drop=True)


def _state_calendar_mode(mode: str | None) -> str:
    selected = STATE_CALENDAR_MODE if mode is None else str(mode)
    if selected not in STATE_CALENDAR_MODES:
        raise ValueError(
            f"Unsupported state calendar mode {selected!r}; "
            f"expected one of {sorted(STATE_CALENDAR_MODES)}"
        )
    return selected


def _et_state_operator_mode(mode: str | None) -> str:
    selected = ET_STATE_OPERATOR_MODE if mode is None else str(mode)
    if selected not in ET_STATE_OPERATOR_MODES:
        raise ValueError(
            f"Unsupported ET state operator {selected!r}; "
            f"expected one of {sorted(ET_STATE_OPERATOR_MODES)}"
        )
    return selected


def build_featured_observation_panel(
    forcing: pd.DataFrame,
    rho: float,
    wm: float,
    et_gamma: float,
    sas_rho: float,
    young_k: float,
    storage_scale: float,
    prod_capacity: float,
    runoff_gamma: float,
    quick_rho: float,
    base_rho: float,
    base_release: float,
    state_calendar_mode: str | None = None,
    et_state_operator_mode: str | None = None,
) -> pd.DataFrame:
    """Generate states on a chosen calendar, then return Q-observed model rows.

    ``observed_only`` is retained solely as a D0 reproduction control.  The
    repaired ``full_forcing`` mode advances every available reach-month and
    subsequently masks Q-missing rows before design construction, fitting, and
    evaluation.  Design columns and their distributions are consequently still
    calculated on the same observed rows as the parent baseline.
    """
    mode = _state_calendar_mode(state_calendar_mode)
    full = forcing.sort_values(["comid", "year", "month"]).reset_index(drop=True).copy()
    state_input = full.loc[observation_mask(full)].copy() if mode == "observed_only" else full
    state_input = state_input.reset_index(drop=True)
    with_states = add_hydrologic_features(
        state_input,
        rho=rho,
        wm=wm,
        et_gamma=et_gamma,
        sas_rho=sas_rho,
        young_k=young_k,
        storage_scale=storage_scale,
        prod_capacity=prod_capacity,
        runoff_gamma=runoff_gamma,
        quick_rho=quick_rho,
        base_rho=base_rho,
        base_release=base_release,
        et_state_operator_mode=et_state_operator_mode,
    )
    model_rows = with_states.loc[observation_mask(with_states)].copy()
    model_rows["q_site"] = model_rows["q_site"].astype(str)
    model_rows = model_rows.sort_values(["comid", "year", "month"]).reset_index(drop=True)
    return prepare_design(model_rows)


def reservoir_influence_class_by_reach(max_order: int = 2) -> dict[int, tuple[float, float]]:
    path = TOPOLOGY_PATH
    if not path.exists():
        return {}
    topo = pd.read_csv(path, encoding="utf-8-sig")
    if "reach_id" not in topo.columns or "src_id" not in topo.columns or "downstream_reach" not in topo.columns:
        return {}
    src = topo["src_id"].astype(str)
    is_reservoir = src.str.contains("水库", na=False, regex=False) | src.str.contains("姘村簱", na=False, regex=False)
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
    strength: dict[int, tuple[float, float]] = {rid: (1.0, 0.0) for rid in reservoir_ids}
    frontier: list[tuple[int, int]] = [(rid, 0) for rid in reservoir_ids]
    while frontier:
        current, order = frontier.pop()
        if order >= max_order:
            continue
        for nxt in immediate.get(current, []):
            next_order = order + 1
            next_down = {1: 0.45, 2: 0.20}.get(next_order, 0.0)
            current_self, current_down = strength.get(nxt, (0.0, 0.0))
            if next_down > current_down:
                strength[nxt] = (current_self, next_down)
                frontier.append((nxt, next_order))
    return strength


def _downstream_ids(value: object) -> list[int]:
    if pd.isna(value) or not str(value).strip():
        return []
    return [int(float(token.strip())) for token in str(value).replace(";", ",").split(",") if token.strip()]


def _upstream_matrix(reaches: np.ndarray) -> np.ndarray:
    """Return U[target, source]=1 for the frozen, non-splitting topology."""
    topo = pd.read_csv(TOPOLOGY_PATH, encoding="utf-8-sig")
    if not np.allclose(pd.to_numeric(topo["frac"], errors="coerce"), 1.0):
        raise ValueError("Corrected accounting requires unit topology fractions")
    reach_list = [int(v) for v in reaches]
    index = {reach: i for i, reach in enumerate(reach_list)}
    reverse: dict[int, set[int]] = {reach: set() for reach in reach_list}
    for row in topo[["reach_id", "downstream_reach"]].itertuples(index=False):
        source = int(row.reach_id)
        for target in _downstream_ids(row.downstream_reach):
            if source in index and target in index:
                reverse[target].add(source)
    matrix = np.zeros((len(reach_list), len(reach_list)), dtype=float)
    for target in reach_list:
        seen = {target}
        stack = [target]
        while stack:
            current = stack.pop()
            for parent in reverse.get(current, set()):
                if parent not in seen:
                    seen.add(parent)
                    stack.append(parent)
        for source in seen:
            matrix[index[target], index[source]] = 1.0
    return matrix


def _panel_layout(out: pd.DataFrame) -> tuple[np.ndarray, int, np.ndarray]:
    reaches = out["comid"].astype(int).drop_duplicates().to_numpy()
    counts = out.groupby("comid", sort=False).size().to_numpy()
    if len(set(counts.tolist())) != 1:
        raise ValueError("Corrected accounting requires a complete reach-month panel")
    n_time = int(counts[0])
    expected = len(reaches) * n_time
    if len(out) != expected:
        raise ValueError(f"Panel layout mismatch: {len(out)} != {expected}")
    return reaches, n_time, _upstream_matrix(reaches)


def _aggregate_local_cfs(out: pd.DataFrame, local_cfs: np.ndarray) -> np.ndarray:
    reaches, n_time, upstream = _panel_layout(out)
    values = np.asarray(local_cfs, dtype=float).reshape(len(reaches), n_time)
    return (upstream @ values).reshape(-1)


def _aggregate_local_depth(out: pd.DataFrame, local_depth_mm: np.ndarray) -> np.ndarray:
    """Area-weight local depth/state over every upstream incremental catchment."""
    reaches, n_time, upstream = _panel_layout(out)
    depth = np.asarray(local_depth_mm, dtype=float).reshape(len(reaches), n_time)
    inc = out["IncAreaKm2"].to_numpy(dtype=float).reshape(len(reaches), n_time)
    cum = out["CumAreaKm2"].to_numpy(dtype=float).reshape(len(reaches), n_time)
    return np.divide(upstream @ (depth * inc), cum, out=np.zeros_like(depth), where=cum > 0).reshape(-1)


def _aggregate_variant(out: pd.DataFrame, values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    cfs_fields = {"quick_cfs", "base_cfs", "overflow_cfs", "routed_quick_cfs", "routed_base_cfs", "highflow_cfs"}
    depth_fields = {"storage_mm", "saturation", "et_withdrawn_mm", "et_unmet_mm", "routing_et_withdrawn_mm", "quick_routing_storage_mm", "base_routing_storage_mm"}
    for key, value in values.items():
        if key in cfs_fields:
            result[key] = _aggregate_local_cfs(out, value)
        elif key in depth_fields:
            result[key] = _aggregate_local_depth(out, value)
        else:
            result[key] = value
    return result


def retention_step(previous: float, input_flux: float, rho: float) -> tuple[float, float, float]:
    """Mass-conserving linear reservoir: available, named release, next store."""
    available = max(float(previous) + float(input_flux), 0.0)
    release = (1.0 - float(rho)) * available
    next_store = float(rho) * available
    return available, release, next_store


def fold_pure_qma_reference(out: pd.DataFrame, base_qcalc: np.ndarray, train_end: int) -> tuple[pd.Series, float]:
    reference = pd.DataFrame({"comid": out["comid"], "year": out["year"], "value": np.asarray(base_qcalc, dtype=float)})
    per_reach = reference.loc[reference["year"].le(int(train_end))].groupby("comid")["value"].mean()
    return out["comid"].map(per_reach).astype(float), float(per_reach.quantile(0.70))


def simulate_production_variant(
    out: pd.DataFrame,
    seconds: np.ndarray,
    cumarea: np.ndarray,
    prod_capacity: float,
    runoff_gamma: float,
    quick_rho: float,
    base_rho: float,
    base_release: float,
    highflow_scale: float = 1.0,
    et_state_operator_mode: str | None = None,
) -> dict[str, np.ndarray]:
    mode = _et_state_operator_mode(et_state_operator_mode)
    storage = np.zeros(len(out), dtype=float)
    saturation = np.zeros(len(out), dtype=float)
    quick_cfs = np.zeros(len(out), dtype=float)
    base_cfs = np.zeros(len(out), dtype=float)
    overflow_cfs = np.zeros(len(out), dtype=float)
    routed_quick_cfs = np.zeros(len(out), dtype=float)
    routed_base_cfs = np.zeros(len(out), dtype=float)
    quick_routing_storage_mm = np.zeros(len(out), dtype=float)
    base_routing_storage_mm = np.zeros(len(out), dtype=float)
    et_withdrawn_mm = np.zeros(len(out), dtype=float)
    et_unmet_mm = np.zeros(len(out), dtype=float)
    for _, idx in out.groupby("comid", sort=False).groups.items():
        soil_store = 0.50 * prod_capacity
        quick_store = 0.0
        base_store = 0.0
        for pos in idx:
            eff_mm = float(out.at[pos, "sas_effective_mm"])
            sat0 = float(np.clip(soil_store / max(prod_capacity, EPS), 0.0, 1.5))
            quick_generated_mm = min(eff_mm, eff_mm * (sat0**runoff_gamma) * highflow_scale)
            quick_mm = quick_generated_mm
            infiltrate_mm = max(eff_mm - quick_generated_mm, 0.0)
            soil_store = max(soil_store + infiltrate_mm, 0.0)
            demand_mm = float(out.at[pos, "aet_storage_demand_mm"])
            # Each production branch is an independent counterfactual state
            # process.  D is therefore applied separately rather than divided
            # across main/multistore branches.  Fast routing stores are not
            # eligible for this withdrawal.
            withdrawn_mm = min(demand_mm, soil_store) if mode == "storage_withdrawal" else 0.0
            soil_store = max(soil_store - withdrawn_mm, 0.0)
            unmet_mm = demand_mm - withdrawn_mm if mode == "storage_withdrawal" else 0.0
            overflow_mm = max(soil_store - prod_capacity, 0.0)
            if overflow_mm > 0:
                soil_store = prod_capacity
                quick_mm += overflow_mm * highflow_scale
            sat1 = float(np.clip(soil_store / max(prod_capacity, EPS), 0.0, 1.5))
            slow_mm = base_release * soil_store
            soil_store = max(soil_store - slow_mm, 0.0)
            quick_available, quick_release_mm, quick_store = retention_step(quick_store, quick_mm, quick_rho)
            base_available, base_release_mm, base_store = retention_step(base_store, slow_mm, base_rho)
            area = float(out.at[pos, "IncAreaKm2"])
            factor = area * 1_000_000.0 / 1000.0 / float(seconds[pos]) * 35.3146667
            storage[pos] = soil_store
            saturation[pos] = sat1
            quick_cfs[pos] = quick_generated_mm * factor
            base_cfs[pos] = slow_mm * factor
            overflow_cfs[pos] = overflow_mm * factor
            routed_quick_cfs[pos] = quick_release_mm * factor
            routed_base_cfs[pos] = base_release_mm * factor
            quick_routing_storage_mm[pos] = quick_store
            base_routing_storage_mm[pos] = base_store
            et_withdrawn_mm[pos] = withdrawn_mm
            et_unmet_mm[pos] = unmet_mm
    return {
        "storage_mm": storage,
        "saturation": saturation,
        "quick_cfs": quick_cfs,
        "base_cfs": base_cfs,
        "overflow_cfs": overflow_cfs,
        "routed_quick_cfs": routed_quick_cfs,
        "routed_base_cfs": routed_base_cfs,
        "quick_routing_storage_mm": quick_routing_storage_mm,
        "base_routing_storage_mm": base_routing_storage_mm,
        "highflow_cfs": routed_quick_cfs,
        "et_withdrawn_mm": et_withdrawn_mm,
        "et_unmet_mm": et_unmet_mm,
        "routing_et_withdrawn_mm": np.zeros(len(out), dtype=float),
    }


def add_hydrologic_features(
    df: pd.DataFrame,
    rho: float,
    wm: float,
    et_gamma: float,
    sas_rho: float,
    young_k: float,
    storage_scale: float,
    prod_capacity: float,
    runoff_gamma: float,
    quick_rho: float,
    base_rho: float,
    base_release: float,
    et_state_operator_mode: str | None = None,
) -> pd.DataFrame:
    # State arrays below are positional.  Establish one monotone, zero-based
    # reach calendar before any recurrence so a full forcing panel and an
    # observed-only D0 panel share unambiguous indexing semantics.
    out = df.sort_values(["comid", "year", "month"]).reset_index(drop=True).copy()
    days = np.array([pd.Period(f"{int(y)}-{int(m):02d}").days_in_month for y, m in zip(out["year"], out["month"])])
    seconds = days.astype(float) * 86400.0
    ppt = out["PPT"].fillna(0).clip(lower=0).to_numpy(dtype=float)
    aet = out["AET"].fillna(0).clip(lower=0).to_numpy(dtype=float)
    pet = out["PET"].fillna(0).clip(lower=0).to_numpy(dtype=float)
    cumarea = out["CumAreaKm2"].fillna(out["CumAreaKm2"].median()).clip(lower=1).to_numpy(dtype=float)
    incarea = out["IncAreaKm2"].fillna(out["IncAreaKm2"].median()).clip(lower=1).to_numpy(dtype=float)
    if FORCING_SEMANTICS_MODE not in FORCING_SEMANTICS_MODES:
        raise ValueError(f"Unsupported forcing semantics: {FORCING_SEMANTICS_MODE}")
    if MASS_ACCOUNTING_MODE not in MASS_ACCOUNTING_MODES:
        raise ValueError(f"Unsupported mass accounting: {MASS_ACCOUNTING_MODE}")
    et_operator = _et_state_operator_mode(et_state_operator_mode)
    et_deficit = np.maximum(pet - aet, 0.0)
    recharge = np.maximum(ppt - aet, 0.0)
    withdrawal_demand = np.maximum(aet - ppt, 0.0)
    net = recharge
    surplus_pet = np.maximum(ppt - 0.8 * pet, 0.0)
    out["local_net_cfs"] = net / 1000.0 * incarea * 1_000_000.0 / seconds * 35.3146667
    if MASS_ACCOUNTING_MODE == "explicit_upstream_volume":
        required = ["explicit_upstream_net_cfs", "explicit_upstream_threshold_cfs"]
        missing = [column for column in required if column not in out.columns]
        if missing:
            raise ValueError(f"Explicit upstream accounting columns missing: {missing}")
        out["basin_net_cfs"] = out["explicit_upstream_net_cfs"].to_numpy(dtype=float)
        out["basin_threshold_cfs"] = out["explicit_upstream_threshold_cfs"].to_numpy(dtype=float)
    else:
        out["basin_net_cfs"] = net / 1000.0 * cumarea * 1_000_000.0 / seconds * 35.3146667
        out["basin_threshold_cfs"] = surplus_pet / 1000.0 * cumarea * 1_000_000.0 / seconds * 35.3146667
    out["aridity"] = (pet - aet) / np.maximum(pet + 1.0, 1.0)
    out["aet_mm"] = aet
    out["pet_mm"] = pet
    out["et_deficit_mm"] = et_deficit
    out["aet_pet_ratio"] = aet / np.maximum(pet, 1.0)
    out["aet_ppt_ratio"] = aet / np.maximum(ppt, 1.0)
    out["pet_ppt_ratio"] = pet / np.maximum(ppt, 1.0)
    out["et_recharge_mm"] = recharge
    out["aet_rainfall_offset_mm"] = np.minimum(aet, ppt)
    # This is D=max(AET-PPT, 0), intentionally distinct from the legacy
    # PET-minus-AET diagnostic retained above as ``et_deficit_mm``.
    out["aet_storage_demand_mm"] = withdrawal_demand
    out["aet_source"] = AET_SOURCE
    out["et_operator"] = et_operator
    out["scenario_id"] = SCENARIO_ID
    out["sas_effective_mm"] = recharge
    # Preserve the frozen S1 antecedent-wetness definition exactly.  The
    # baseline clipping applies to recharge entering the slow stores, not to
    # this signed diagnostic recursion.  Both operators therefore use the
    # identical signed P-AET wetness input; E01/E11 differ only by the explicit
    # pre-release withdrawal from eligible slow stores below.
    out["wet_input"] = (ppt - aet) / wm
    # Retained as an audit-compatible column; production withdrawal is now
    # handled inside each independent state process rather than pre-subtracted.
    out["production_demand_mm"] = 0.0

    reservoir_class = reservoir_influence_class_by_reach(max_order=2)
    rid = out["comid"].astype("Int64")
    out["res_decay_self"] = rid.map(lambda x: reservoir_class.get(int(x), (0.0, 0.0))[0] if pd.notna(x) else 0.0).astype(float)
    out["res_decay_down"] = rid.map(lambda x: reservoir_class.get(int(x), (0.0, 0.0))[1] if pd.notna(x) else 0.0).astype(float)
    out = out.sort_values(["comid", "year", "month"]).reset_index(drop=True).copy()
    state = np.zeros(len(out), dtype=float)
    for _, idx in out.groupby("comid", sort=False).groups.items():
        last = 0.0
        for pos in idx:
            last = rho * last + float(out.at[pos, "wet_input"])
            last = float(np.clip(last, -3.0, 3.0))
            state[pos] = last
    out["antecedent_wetness"] = state
    # The empirical 0.35 is now a bounded, state-dependent connected fraction
    # applied to each incremental catchment before topology aggregation.
    # Standardization is outer-fold pure; validation forcing never affects it.
    lag_state = out.groupby("comid")["antecedent_wetness"].shift(1).fillna(0.0)
    train_state = lag_state[out["year"].le(CAL_END_YEAR)]
    state_mean = float(train_state.mean())
    state_sd = float(train_state.std(ddof=0))
    if not np.isfinite(state_sd) or state_sd <= EPS:
        state_sd = 1.0
    dynamic_score = (lag_state.to_numpy(dtype=float) - state_mean) / state_sd
    connected_fraction = expit(logit(0.35) + float(DYNAMIC_BETA_W) * dynamic_score)
    local_net_cfs = out["local_net_cfs"].to_numpy(dtype=float)
    out["connectivity_score_fold_z"] = dynamic_score
    out["connected_fraction"] = connected_fraction
    out["connected_local_cfs"] = connected_fraction * local_net_cfs
    out["unconnected_local_cfs"] = (1.0 - connected_fraction) * local_net_cfs
    out["Q_calc_cfs"] = _aggregate_local_cfs(out, out["connected_local_cfs"].to_numpy(dtype=float))
    out["connected_upstream_cfs"] = out["Q_calc_cfs"]
    out["unconnected_upstream_cfs"] = _aggregate_local_cfs(out, out["unconnected_local_cfs"].to_numpy(dtype=float))
    # Q_ma is a deployment-style training-forcing reference shared by I0/I1.
    base_qcalc = 0.35 * out["explicit_upstream_net_cfs"].to_numpy(dtype=float)
    out["Q_ma_cfs"], qma_threshold = fold_pure_qma_reference(out, base_qcalc, CAL_END_YEAR)
    out["MAFlowUcfs"] = out["Q_ma_cfs"]
    global QMA_TRAIN_THRESHOLD_CFS
    QMA_TRAIN_THRESHOLD_CFS = qma_threshold
    sas_storage = np.zeros(len(out), dtype=float)
    sas_young_frac = np.zeros(len(out), dtype=float)
    sas_young_cfs = np.zeros(len(out), dtype=float)
    sas_old_release_cfs = np.zeros(len(out), dtype=float)
    sas_et_withdrawn = np.zeros(len(out), dtype=float)
    sas_et_unmet = np.zeros(len(out), dtype=float)
    for _, idx in out.groupby("comid", sort=False).groups.items():
        last_storage = 0.0
        for pos in idx:
            eff_mm = float(out.at[pos, "sas_effective_mm"])
            wet = float(out.at[pos, "antecedent_wetness"])
            young_frac = 1.0 / (1.0 + np.exp(-young_k * wet))
            young_frac = float(np.clip(young_frac, 0.05, 0.95))
            available_storage = last_storage + (1.0 - young_frac) * eff_mm
            demand_mm = float(out.at[pos, "aet_storage_demand_mm"])
            withdrawn_mm = min(demand_mm, available_storage) if et_operator == "storage_withdrawal" else 0.0
            available_storage = max(available_storage - withdrawn_mm, 0.0)
            unmet_mm = demand_mm - withdrawn_mm if et_operator == "storage_withdrawal" else 0.0
            _available, old_release_mm, last_storage = retention_step(0.0, available_storage, sas_rho)
            area = float(out.at[pos, "IncAreaKm2"])
            month_seconds = float(seconds[pos])
            sas_storage[pos] = last_storage
            sas_young_frac[pos] = young_frac
            sas_young_cfs[pos] = young_frac * eff_mm / 1000.0 * area * 1_000_000.0 / month_seconds * 35.3146667
            sas_old_release_cfs[pos] = old_release_mm / 1000.0 * area * 1_000_000.0 / month_seconds * 35.3146667
            sas_et_withdrawn[pos] = withdrawn_mm
            sas_et_unmet[pos] = unmet_mm
    out["sas_storage_mm"] = _aggregate_local_depth(out, sas_storage)
    out["sas_young_fraction"] = _aggregate_local_depth(out, sas_young_frac)
    out["sas_young_cfs"] = _aggregate_local_cfs(out, sas_young_cfs)
    out["sas_old_release_cfs"] = _aggregate_local_cfs(out, sas_old_release_cfs)
    out["sas_et_withdrawn_mm"] = _aggregate_local_depth(out, sas_et_withdrawn)
    out["sas_et_unmet_mm"] = _aggregate_local_depth(out, sas_et_unmet)
    out["sas_routing_et_withdrawn_mm"] = 0.0
    out["sas_old_fraction"] = 1.0 - out["sas_young_fraction"]
    out["sas_storage_scaled"] = out["sas_storage_mm"] / max(storage_scale, EPS)
    primary = _aggregate_variant(out, simulate_production_variant(
        out, seconds, cumarea, prod_capacity, runoff_gamma, quick_rho, base_rho,
        base_release, et_state_operator_mode=et_operator,
    ))
    out["production_storage_mm"] = primary["storage_mm"]
    out["production_saturation"] = primary["saturation"]
    out["production_quick_cfs"] = primary["quick_cfs"]
    out["production_base_cfs"] = primary["base_cfs"]
    out["production_overflow_cfs"] = primary["overflow_cfs"]
    out["routed_quick_cfs"] = primary["routed_quick_cfs"]
    out["routed_base_cfs"] = primary["routed_base_cfs"]
    out["production_et_withdrawn_mm"] = primary["et_withdrawn_mm"]
    out["production_et_unmet_mm"] = primary["et_unmet_mm"]
    out["production_routing_et_withdrawn_mm"] = primary["routing_et_withdrawn_mm"]
    out["production_quick_routing_storage_mm"] = primary["quick_routing_storage_mm"]
    out["production_base_routing_storage_mm"] = primary["base_routing_storage_mm"]
    # This pair is deliberately the main production-state audit, not a sum of
    # counterfactual SAS/multistore branches.  Branch-specific columns below
    # retain independent accounting without double counting D.
    out["aet_storage_withdrawn_mm"] = out["production_et_withdrawn_mm"]
    out["aet_unmet_mm"] = out["production_et_unmet_mm"]
    out["production_saturation_wetness"] = out["production_saturation"] * np.maximum(state, 0.0)
    out["production_dry_base_release"] = np.log1p(out["routed_base_cfs"].clip(lower=0.0)) * np.maximum(-state, 0.0)
    state_delta = np.zeros(len(out), dtype=float)
    for _, idx in out.groupby("comid", sort=False).groups.items():
        last = 0.0
        for pos in idx:
            current = float(out.at[pos, "antecedent_wetness"])
            state_delta[pos] = current - last
            last = current
    out["wetness_delta"] = state_delta
    out["wetting_state"] = np.clip(state_delta, 0.0, 2.0)
    out["drying_state"] = np.clip(-state_delta, 0.0, 2.0)
    month_int = out["month"].astype(int)
    out["early_wet_gate"] = month_int.between(2, 5).astype(float)
    out["peak_rain_gate"] = month_int.between(5, 7).astype(float)
    out["late_recession_gate"] = month_int.between(8, 11).astype(float)
    out["dry_recharge_gate"] = ((month_int >= 12) | (month_int <= 2)).astype(float)

    variants = {
        "flash_headwater": _aggregate_variant(out, simulate_production_variant(
            out,
            seconds,
            cumarea,
            prod_capacity=0.55 * prod_capacity,
            runoff_gamma=max(1.3, runoff_gamma - 0.8),
            quick_rho=0.10,
            base_rho=0.76,
            base_release=0.06,
            highflow_scale=1.20,
            et_state_operator_mode=et_operator,
        )),
        "slow_large": _aggregate_variant(out, simulate_production_variant(
            out,
            seconds,
            cumarea,
            prod_capacity=1.80 * prod_capacity,
            runoff_gamma=runoff_gamma + 0.6,
            quick_rho=0.48,
            base_rho=0.94,
            base_release=0.07,
            highflow_scale=0.85,
            et_state_operator_mode=et_operator,
        )),
        "buffer_reservoir": _aggregate_variant(out, simulate_production_variant(
            out,
            seconds,
            cumarea,
            prod_capacity=1.45 * prod_capacity,
            runoff_gamma=runoff_gamma + 0.8,
            quick_rho=0.66,
            base_rho=0.96,
            base_release=0.12,
            highflow_scale=0.65,
            et_state_operator_mode=et_operator,
        )),
        "wet_large": _aggregate_variant(out, simulate_production_variant(
            out,
            seconds,
            cumarea,
            prod_capacity=1.20 * prod_capacity,
            runoff_gamma=max(1.5, runoff_gamma - 0.3),
            quick_rho=0.34,
            base_rho=0.90,
            base_release=0.08,
            highflow_scale=1.10,
            et_state_operator_mode=et_operator,
        )),
    }
    for name, values in variants.items():
        for key, arr in values.items():
            out[f"ms_{name}_{key}"] = arr

    out["lag1_aet_mm"] = out.groupby("comid")["aet_mm"].shift(1).fillna(out["aet_mm"])
    out["lag1_et_deficit_mm"] = out.groupby("comid")["et_deficit_mm"].shift(1).fillna(out["et_deficit_mm"])
    lagged_net = out.groupby("comid")["basin_net_cfs"].shift(1).fillna(0.0)
    out["res_lag_self"] = lagged_net * out["res_decay_self"]
    out["res_lag_down"] = lagged_net * out["res_decay_down"]
    out["wet_quickflow"] = np.log1p(out["basin_threshold_cfs"].clip(lower=0.0)) * np.maximum(state, 0.0)
    out["sas_young_wet_interaction"] = np.log1p(out["sas_young_cfs"].clip(lower=0.0)) * np.maximum(state, 0.0)
    out["sas_old_dry_release"] = np.log1p(out["sas_old_release_cfs"].clip(lower=0.0)) * np.maximum(-state, 0.0)
    out["et_deficit_wetness"] = out["et_deficit_mm"] * np.maximum(-state, 0.0)
    quick_gate = 1.0 / (1.0 + np.exp(-2.0 * state))
    young_gate = out["sas_young_fraction"].to_numpy(dtype=float)
    out["high_flow_regime_gate"] = np.clip(0.55 * quick_gate + 0.45 * young_gate, 0.05, 0.95)
    out["low_flow_regime_gate"] = 1.0 - out["high_flow_regime_gate"]
    out["wet_season_gate"] = out["month"].astype(int).between(4, 9).astype(float)
    out["dry_season_gate"] = 1.0 - out["wet_season_gate"]
    out["wet_high_regime_gate"] = out["wet_season_gate"] * out["high_flow_regime_gate"]
    out["wet_low_regime_gate"] = out["wet_season_gate"] * out["low_flow_regime_gate"]
    out["dry_high_regime_gate"] = out["dry_season_gate"] * out["high_flow_regime_gate"]
    out["dry_low_regime_gate"] = out["dry_season_gate"] * out["low_flow_regime_gate"]
    out["month_sin"] = np.sin(2 * np.pi * out["month"].astype(float) / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * out["month"].astype(float) / 12.0)
    out["is_reservoir_reach"] = (out["res_decay_self"].fillna(0.0) > 0).astype(float)
    out["downstream_reservoir"] = (out["res_decay_down"].fillna(0.0) > 0).astype(float)
    out["hys_early_wetting_quick"] = np.log1p(out["routed_quick_cfs"].clip(lower=0.0)) * out["early_wet_gate"] * out["wetting_state"]
    out["hys_early_threshold_flush"] = np.log1p(out["basin_threshold_cfs"].clip(lower=0.0)) * out["early_wet_gate"] * np.maximum(out["antecedent_wetness"], 0.0)
    out["hys_peak_highflow_pulse"] = np.log1p(out["routed_quick_cfs"].clip(lower=0.0)) * out["peak_rain_gate"] * out["high_flow_regime_gate"]
    out["hys_peak_saturation_flush"] = out["production_saturation_wetness"] * out["peak_rain_gate"]
    out["hys_late_storage_release"] = np.log1p((out["routed_base_cfs"] + out["sas_old_release_cfs"]).clip(lower=0.0)) * out["late_recession_gate"]
    out["hys_late_drying_attenuation"] = np.log1p(out["routed_quick_cfs"].clip(lower=0.0)) * out["late_recession_gate"] * out["drying_state"]
    out["hys_late_storage_excess"] = (
        out["late_recession_gate"]
        * (out["sas_storage_scaled"].fillna(0.0) + out["production_storage_mm"].fillna(0.0) / max(prod_capacity, EPS))
    )
    out["hys_dry_recharge_memory"] = np.log1p(out["routed_base_cfs"].clip(lower=0.0)) * out["dry_recharge_gate"] * np.maximum(-out["antecedent_wetness"], 0.0)
    return out


HYSTERESIS_FEATURES = [
    "wetness_delta",
    "wetting_state",
    "drying_state",
    "hys_early_wetting_quick",
    "hys_early_threshold_flush",
    "hys_peak_highflow_pulse",
    "hys_peak_saturation_flush",
    "hys_late_storage_release",
    "hys_late_drying_attenuation",
    "hys_late_storage_excess",
    "hys_dry_recharge_memory",
]


MULTISTORE_FEATURES = [
    "log_ms_headwater_flash_highflow",
    "log_ms_headwater_flash_quick",
    "log_ms_headwater_flash_base",
    "log_ms_large_slow_base",
    "log_ms_large_slow_storage",
    "log_ms_reservoir_buffer_base",
    "log_ms_reservoir_buffer_highflow",
    "log_ms_wet_large_highflow",
    "log_ms_wet_large_quick",
    "log_ms_dry_headwater_base",
]


FIXED_FEATURES = [
    "log_qcalc",
    "log_qma",
    "log_cumarea",
    "log_basin_threshold",
    "log_sas_young",
    "log_sas_old_release",
    "log_production_quick",
    "log_production_base",
    "log_production_overflow",
    "log_routed_quick",
    "log_routed_base",
    "production_saturation",
    "production_saturation_wetness",
    "production_dry_base_release",
    "sas_young_fraction",
    "sas_old_fraction",
    "sas_storage_scaled",
    "antecedent_wetness",
    "wet_quickflow",
    "sas_young_wet_interaction",
    "sas_old_dry_release",
    "aridity",
    "log_aet",
    "log_pet",
    "log_et_deficit",
    "aet_pet_ratio",
    "aet_ppt_ratio",
    "pet_ppt_ratio",
    "log_lag1_aet",
    "log_lag1_et_deficit",
    "et_deficit_wetness",
    "month_sin",
    "month_cos",
    "log_res_lag_self",
    "log_res_lag_down",
    "is_reservoir_reach",
    "downstream_reservoir",
    *MULTISTORE_FEATURES,
    *HYSTERESIS_FEATURES,
]

PRODUCTION_FEATURES = [
    "log_production_quick",
    "log_production_base",
    "log_production_overflow",
    "log_routed_quick",
    "log_routed_base",
    "production_saturation",
    "production_saturation_wetness",
    "production_dry_base_release",
]

RANDOM_SLOPE_FEATURES = [
    "log_basin_threshold",
    "log_sas_young",
    "log_sas_old_release",
    "antecedent_wetness",
    "wet_quickflow",
    "sas_young_wet_interaction",
    "sas_old_dry_release",
    "et_deficit_wetness",
    "log_routed_base",
    "production_saturation_wetness",
]

REGIME_SLOPE_FEATURES = [
    "log_basin_threshold",
    "log_sas_young",
    "log_sas_old_release",
    "wet_quickflow",
    "sas_young_wet_interaction",
]

REGIME_GATES = ["wet_high_regime_gate", "wet_low_regime_gate", "dry_high_regime_gate", "dry_low_regime_gate"]

SPATIAL_GROUP_GATES = [
    "group_headwater_area",
    "group_mid_area",
    "group_large_area",
    "group_major_flow",
    "group_reservoir_influenced",
    "group_wet_large_area",
    "group_dry_headwater",
]

SPATIAL_GROUP_FEATURES = [
    "log_routed_quick",
    "log_routed_base",
    "log_production_quick",
    "log_production_base",
    "log_sas_young",
    "log_sas_old_release",
    "production_saturation_wetness",
]

# Direct AET/PET contract columns are optional explanatory covariates.  In
# ``state_only_et`` they are removed from every design block while AET remains
# active inside R/D and the hydrologic state recurrences.
ET_CONTRACT_FEATURES = frozenset(
    {
        "aridity",
        "log_aet",
        "log_pet",
        "log_et_deficit",
        "aet_pet_ratio",
        "aet_ppt_ratio",
        "pet_ppt_ratio",
        "log_lag1_aet",
        "log_lag1_et_deficit",
        "et_deficit_wetness",
        "log_basin_threshold",
        "wet_quickflow",
        "hys_early_threshold_flush",
    }
)
_FULL_FIXED_FEATURES = tuple(FIXED_FEATURES)
_FULL_RANDOM_SLOPE_FEATURES = tuple(RANDOM_SLOPE_FEATURES)
_FULL_REGIME_SLOPE_FEATURES = tuple(REGIME_SLOPE_FEATURES)
_FULL_SPATIAL_GROUP_FEATURES = tuple(SPATIAL_GROUP_FEATURES)


def set_et_feature_block_mode(mode: str) -> None:
    """Apply the frozen ET covariate gate before model fitting.

    This updates all design dependencies together, preventing F0 from leaving
    an ET contract column as a fixed effect, station random slope, or
    regime-gated station slope.
    """
    global ET_FEATURE_BLOCK_MODE
    global FIXED_FEATURES, RANDOM_SLOPE_FEATURES, REGIME_SLOPE_FEATURES, SPATIAL_GROUP_FEATURES
    selected = str(mode)
    if selected not in ET_FEATURE_BLOCK_MODES:
        raise ValueError(f"Unsupported ET feature block mode {selected!r}; expected one of {sorted(ET_FEATURE_BLOCK_MODES)}")
    ET_FEATURE_BLOCK_MODE = selected
    blocked = ET_CONTRACT_FEATURES if selected == "state_only_et" else frozenset()
    FIXED_FEATURES = [feature for feature in _FULL_FIXED_FEATURES if feature not in blocked]
    RANDOM_SLOPE_FEATURES = [feature for feature in _FULL_RANDOM_SLOPE_FEATURES if feature not in blocked]
    REGIME_SLOPE_FEATURES = [feature for feature in _FULL_REGIME_SLOPE_FEATURES if feature not in blocked]
    SPATIAL_GROUP_FEATURES = [feature for feature in _FULL_SPATIAL_GROUP_FEATURES if feature not in blocked]


def configure_et_scenario(scenario_id: str) -> dict[str, str]:
    """Configure one of E00/E10/E01/E11/E00-F0 without selecting its input."""
    global AET_SOURCE, ET_STATE_OPERATOR_MODE, SCENARIO_ID
    try:
        scenario = ET_SCENARIOS[str(scenario_id)]
    except KeyError as exc:
        raise ValueError(f"Unknown ET scenario {scenario_id!r}; expected one of {sorted(ET_SCENARIOS)}") from exc
    ET_STATE_OPERATOR_MODE = str(scenario["operator"])
    AET_SOURCE = str(scenario["forcing"])
    SCENARIO_ID = str(scenario_id)
    set_et_feature_block_mode(str(scenario["feature_block"]))
    return dict(scenario)


def feature_group(feature: str) -> str:
    if feature.startswith("spatial_group_slope_"):
        return "spatial_group_hydrologic_response"
    if feature in HYSTERESIS_FEATURES:
        return "seasonal_hysteresis_response"
    if feature in MULTISTORE_FEATURES:
        return "spatial_group_multistore_response"
    if feature in PRODUCTION_FEATURES:
        return "nonlinear_production_routing"
    if feature.startswith("regime_random_slope_"):
        return "station_random_season_regime_slope"
    if feature in RANDOM_SLOPE_FEATURES:
        return "station_random_hydrologic_sas_slope"
    if "sas_" in feature:
        return "sas_young_old_storage"
    if feature.startswith("res_") or feature in ["is_reservoir_reach", "downstream_reservoir", "log_res_lag_self", "log_res_lag_down"]:
        return "reservoir_lag"
    if "aet" in feature or "pet" in feature or "et_" in feature or feature == "aridity":
        return "et_water_balance"
    if feature.startswith("month_"):
        return "seasonality"
    if feature in ["log_qcalc", "log_qma", "log_cumarea"]:
        return "sparrow_base"
    return "rainfall_runoff_state"


def prepare_design(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["log_qcalc"] = np.log1p(out["Q_calc_cfs"].fillna(0).clip(lower=0))
    out["log_qma"] = np.log1p(out["Q_ma_cfs"].fillna(0).clip(lower=0))
    out["log_cumarea"] = np.log1p(out["CumAreaKm2"].fillna(out["CumAreaKm2"].median()).clip(lower=1))
    out["log_basin_net"] = np.log1p(out["basin_net_cfs"].fillna(0).clip(lower=0))
    out["log_basin_threshold"] = np.log1p(out["basin_threshold_cfs"].fillna(0).clip(lower=0))
    out["log_sas_young"] = np.log1p(out["sas_young_cfs"].fillna(0).clip(lower=0))
    out["log_sas_old_release"] = np.log1p(out["sas_old_release_cfs"].fillna(0).clip(lower=0))
    out["log_production_quick"] = np.log1p(out["production_quick_cfs"].fillna(0).clip(lower=0))
    out["log_production_base"] = np.log1p(out["production_base_cfs"].fillna(0).clip(lower=0))
    out["log_production_overflow"] = np.log1p(out["production_overflow_cfs"].fillna(0).clip(lower=0))
    out["log_routed_quick"] = np.log1p(out["routed_quick_cfs"].fillna(0).clip(lower=0))
    out["log_routed_base"] = np.log1p(out["routed_base_cfs"].fillna(0).clip(lower=0))
    out["log_aet"] = np.log1p(out["aet_mm"].fillna(0).clip(lower=0))
    out["log_pet"] = np.log1p(out["pet_mm"].fillna(0).clip(lower=0))
    out["log_et_deficit"] = np.log1p(out["et_deficit_mm"].fillna(0).clip(lower=0))
    out["log_lag1_aet"] = np.log1p(out["lag1_aet_mm"].fillna(0).clip(lower=0))
    out["log_lag1_et_deficit"] = np.log1p(out["lag1_et_deficit_mm"].fillna(0).clip(lower=0))
    out["log_res_lag_self"] = np.log1p(out["res_lag_self"].fillna(0).clip(lower=0))
    out["log_res_lag_down"] = np.log1p(out["res_lag_down"].fillna(0).clip(lower=0))
    cumarea = out["CumAreaKm2"].fillna(out["CumAreaKm2"].median()).clip(lower=1)
    qma = out["Q_ma_cfs"].fillna(out["Q_ma_cfs"].median()).clip(lower=0)
    out["group_headwater_area"] = (cumarea < 5_000).astype(float)
    out["group_mid_area"] = ((cumarea >= 5_000) & (cumarea < 25_000)).astype(float)
    out["group_large_area"] = (cumarea >= 25_000).astype(float)
    threshold = float(QMA_TRAIN_THRESHOLD_CFS)
    if not np.isfinite(threshold):
        raise ValueError("Fold-pure Q_ma threshold was not initialized")
    out["group_major_flow"] = (qma >= threshold).astype(float)
    out["group_reservoir_influenced"] = ((out["is_reservoir_reach"] > 0) | (out["downstream_reservoir"] > 0)).astype(float)
    out["group_wet_large_area"] = out["group_large_area"] * out["month"].astype(int).between(4, 9).astype(float)
    out["group_dry_headwater"] = out["group_headwater_area"] * (~out["month"].astype(int).between(4, 9)).astype(float)
    out["log_ms_headwater_flash_highflow"] = np.log1p(out["ms_flash_headwater_highflow_cfs"].fillna(0).clip(lower=0)) * out["group_headwater_area"]
    out["log_ms_headwater_flash_quick"] = np.log1p(out["ms_flash_headwater_routed_quick_cfs"].fillna(0).clip(lower=0)) * out["group_headwater_area"]
    out["log_ms_headwater_flash_base"] = np.log1p(out["ms_flash_headwater_routed_base_cfs"].fillna(0).clip(lower=0)) * out["group_headwater_area"]
    out["log_ms_large_slow_base"] = np.log1p(out["ms_slow_large_routed_base_cfs"].fillna(0).clip(lower=0)) * out["group_large_area"]
    out["log_ms_large_slow_storage"] = (out["ms_slow_large_storage_mm"].fillna(0).clip(lower=0) / 480.0) * out["group_large_area"]
    out["log_ms_reservoir_buffer_base"] = np.log1p(out["ms_buffer_reservoir_routed_base_cfs"].fillna(0).clip(lower=0)) * out["group_reservoir_influenced"]
    out["log_ms_reservoir_buffer_highflow"] = np.log1p(out["ms_buffer_reservoir_highflow_cfs"].fillna(0).clip(lower=0)) * out["group_reservoir_influenced"]
    out["log_ms_wet_large_highflow"] = np.log1p(out["ms_wet_large_highflow_cfs"].fillna(0).clip(lower=0)) * out["group_wet_large_area"]
    out["log_ms_wet_large_quick"] = np.log1p(out["ms_wet_large_routed_quick_cfs"].fillna(0).clip(lower=0)) * out["group_wet_large_area"]
    out["log_ms_dry_headwater_base"] = np.log1p(out["ms_flash_headwater_routed_base_cfs"].fillna(0).clip(lower=0)) * out["group_dry_headwater"]
    out["high_flow_regime_gate"] = out["high_flow_regime_gate"].fillna(0.5).clip(0.05, 0.95)
    out["low_flow_regime_gate"] = out["low_flow_regime_gate"].fillna(0.5).clip(0.05, 0.95)
    for gate in REGIME_GATES:
        out[gate] = out[gate].fillna(0.0).clip(0.0, 1.0)
    for gate in SPATIAL_GROUP_GATES:
        out[gate] = out[gate].fillna(0.0).clip(0.0, 1.0)
    out["log_obs"] = np.log(out["Q_obsv_cfs"].clip(lower=EPS))
    return out


def standardize_fit(train: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    mean = train[FIXED_FEATURES].mean()
    std = train[FIXED_FEATURES].std(ddof=0).replace(0, 1.0)
    return mean, std


def build_matrix(df: pd.DataFrame, stations: list[str], mean: pd.Series, std: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    x_fixed_df = ((df[FIXED_FEATURES] - mean) / std).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    x_fixed = x_fixed_df.to_numpy(dtype=float)
    group_cols = []
    for gate in SPATIAL_GROUP_GATES:
        gate_values = df[gate].fillna(0.0).clip(0.0, 1.0).to_numpy(dtype=float)
        for feat in SPATIAL_GROUP_FEATURES:
            group_cols.append(x_fixed_df[feat].to_numpy(dtype=float) * gate_values)
    x_group = np.column_stack(group_cols) if group_cols else np.zeros((len(df), 0), dtype=float)
    intercept = np.ones((len(df), 1), dtype=float)
    station_index = {name: i for i, name in enumerate(stations)}
    unknown = sorted(set(df["q_site"].astype(str)) - set(station_index))
    if unknown:
        raise ValueError(
            "Prediction contains stations absent from the registered station "
            f"universe: {unknown[:10]}"
        )
    site_codes = np.array([station_index[name] for name in df["q_site"].astype(str)], dtype=int)

    z_intercept = np.zeros((len(df), len(stations)), dtype=float)
    z_intercept[np.arange(len(df)), site_codes] = 1.0

    slope_blocks = []
    for feat in RANDOM_SLOPE_FEATURES:
        block = np.zeros((len(df), len(stations)), dtype=float)
        block[np.arange(len(df)), site_codes] = x_fixed_df[feat].to_numpy(dtype=float)
        slope_blocks.append(block)
    for gate in REGIME_GATES:
        gate_values = df[gate].fillna(0.5).clip(0.05, 0.95).to_numpy(dtype=float)
        for feat in REGIME_SLOPE_FEATURES:
            block = np.zeros((len(df), len(stations)), dtype=float)
            block[np.arange(len(df)), site_codes] = x_fixed_df[feat].to_numpy(dtype=float) * gate_values
            slope_blocks.append(block)
    x = np.column_stack([intercept, x_fixed, x_group, z_intercept, *slope_blocks])
    y = df["log_obs"].to_numpy(dtype=float)
    return x, y


def design_column_metadata(stations: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = [
        {"parameter": "intercept", "parameter_block": "intercept"}
    ]
    rows.extend(
        {
            "parameter": feature,
            "parameter_block": feature_group(feature),
        }
        for feature in FIXED_FEATURES
    )
    rows.extend(
        {
            "parameter": f"spatial_group_slope_{gate}_{feature}",
            "parameter_block": "spatial_group_hydrologic_response",
        }
        for gate in SPATIAL_GROUP_GATES
        for feature in SPATIAL_GROUP_FEATURES
    )
    rows.extend(
        {
            "parameter": f"station_intercept::{station}",
            "parameter_block": "station_random_intercept",
        }
        for station in stations
    )
    rows.extend(
        {
            "parameter": f"random_slope_{feature}::{station}",
            "parameter_block": "station_random_hydrologic_sas_slope",
        }
        for feature in RANDOM_SLOPE_FEATURES
        for station in stations
    )
    rows.extend(
        {
            "parameter": (
                f"regime_random_slope_{gate}_{feature}::{station}"
            ),
            "parameter_block": "station_random_season_regime_slope",
        }
        for gate in REGIME_GATES
        for feature in REGIME_SLOPE_FEATURES
        for station in stations
    )
    metadata = pd.DataFrame(rows)
    metadata.insert(0, "column_index", np.arange(len(metadata), dtype=int))
    return metadata


def fit_map_ridge(
    train: pd.DataFrame,
    stations: list[str],
    mean: pd.Series,
    std: pd.Series,
    fixed_sigma: float,
    production_sigma: float,
    group_sigma: float,
    multistore_sigma: float,
    hysteresis_sigma: float,
    station_sigma: float,
    slope_sigma: float,
    regime_slope_sigma: float,
    anomaly_weight: float,
    flow_contrast_weight: float,
) -> np.ndarray:
    x, y = build_matrix(train, stations, mean, std)
    x_parts = [x]
    y_parts = [y]
    if anomaly_weight > 0:
        sqrt_w = float(np.sqrt(anomaly_weight))
        anomaly_x = []
        anomaly_y = []
        for _, pos in train.groupby("q_site", sort=False).indices.items():
            pos = np.array(pos, dtype=int)
            local_x = x[pos, :]
            local_y = y[pos]
            sd = float(np.std(local_y))
            if len(pos) < 12 or sd <= EPS:
                continue
            anomaly_x.append((local_x - local_x.mean(axis=0, keepdims=True)) / sd * sqrt_w)
            anomaly_y.append((local_y - local_y.mean()) / sd * sqrt_w)
        if anomaly_x:
            x_parts.append(np.vstack(anomaly_x))
            y_parts.append(np.concatenate(anomaly_y))
    if flow_contrast_weight > 0:
        sqrt_w = float(np.sqrt(flow_contrast_weight))
        contrast_x = []
        contrast_y = []
        for _, pos in train.groupby("q_site", sort=False).indices.items():
            pos = np.array(pos, dtype=int)
            if len(pos) < 36:
                continue
            local_x = x[pos, :]
            local_y = y[pos]
            local_q = train.iloc[pos]["Q_obsv_cfs"].to_numpy(dtype=float)
            q25 = np.nanquantile(local_q, 0.25)
            q50 = np.nanquantile(local_q, 0.50)
            q75 = np.nanquantile(local_q, 0.75)
            q90 = np.nanquantile(local_q, 0.90)
            low = local_q <= q25
            high = local_q >= q75
            mid = (local_q >= q25) & (local_q <= q75)
            peak = local_q >= q90
            if low.sum() >= 6 and high.sum() >= 6:
                contrast_x.append((local_x[high].mean(axis=0) - local_x[low].mean(axis=0)) * sqrt_w)
                contrast_y.append((local_y[high].mean() - local_y[low].mean()) * sqrt_w)
            if peak.sum() >= 3 and mid.sum() >= 12:
                contrast_x.append((local_x[peak].mean(axis=0) - local_x[mid].mean(axis=0)) * sqrt_w)
                contrast_y.append((local_y[peak].mean() - local_y[mid].mean()) * sqrt_w)
        if contrast_x:
            x_parts.append(np.vstack(contrast_x))
            y_parts.append(np.array(contrast_y, dtype=float))
    x = np.vstack(x_parts)
    y = np.concatenate(y_parts)
    n_fixed_base = 1 + len(FIXED_FEATURES)
    n_group = len(SPATIAL_GROUP_GATES) * len(SPATIAL_GROUP_FEATURES)
    n_fixed = n_fixed_base + n_group
    n_station = len(stations)
    n_slope = len(RANDOM_SLOPE_FEATURES) * n_station
    n_regime_slope = len(REGIME_GATES) * len(REGIME_SLOPE_FEATURES) * n_station
    penalty = np.zeros(n_fixed + n_station + n_slope + n_regime_slope, dtype=float)
    penalty[1:n_fixed_base] = 1.0 / max(fixed_sigma, EPS)
    for i, feat in enumerate(FIXED_FEATURES, start=1):
        if feat in PRODUCTION_FEATURES:
            penalty[i] = 1.0 / max(production_sigma, EPS)
        if feat in MULTISTORE_FEATURES:
            penalty[i] = 1.0 / max(multistore_sigma, EPS)
        if feat in HYSTERESIS_FEATURES:
            penalty[i] = 1.0 / max(hysteresis_sigma, EPS)
    penalty[n_fixed_base:n_fixed] = 1.0 / max(group_sigma, EPS)
    penalty[n_fixed : n_fixed + n_station] = 1.0 / max(station_sigma, EPS)
    penalty[n_fixed + n_station : n_fixed + n_station + n_slope] = 1.0 / max(slope_sigma, EPS)
    penalty[n_fixed + n_station + n_slope :] = 1.0 / max(regime_slope_sigma, EPS)
    observation_rows = int(len(train))
    contrast_or_anomaly_rows = int(len(y) - observation_rows)
    x_aug = np.vstack([x, np.diag(penalty)])
    y_aug = np.concatenate([y, np.zeros(len(penalty), dtype=float)])
    beta, residuals, rank, singular_values = np.linalg.lstsq(
        x_aug, y_aug, rcond=None
    )
    design_dir = REPORT_DIR / "design_matrix"
    design_dir.mkdir(parents=True, exist_ok=True)
    metadata = design_column_metadata(stations)
    if len(metadata) != x_aug.shape[1]:
        raise RuntimeError(
            f"Design metadata mismatch: {len(metadata)} != {x_aug.shape[1]}"
        )
    sparse.save_npz(
        design_dir / "observation_design_matrix.npz",
        sparse.csr_matrix(x[:observation_rows]),
        compressed=True,
    )
    sparse.save_npz(
        design_dir / "augmented_map_design_matrix.npz",
        sparse.csr_matrix(x_aug),
        compressed=True,
    )
    np.save(
        design_dir / "observation_log_flow_response.npy",
        y[:observation_rows],
        allow_pickle=False,
    )
    np.save(
        design_dir / "augmented_map_response.npy",
        y_aug,
        allow_pickle=False,
    )
    metadata.to_csv(
        design_dir / "design_matrix_columns.csv",
        index=False,
        encoding="utf-8-sig",
    )
    fitted_residual = y[:observation_rows] - x[:observation_rows] @ beta
    residual_variance = float(
        np.sum(fitted_residual**2)
        / max(observation_rows - int(rank), 1)
    )
    precision_diagonal = np.sum(x_aug * x_aug, axis=0)
    curvature = metadata.copy()
    curvature["ridge_precision_diagonal"] = precision_diagonal
    curvature["approx_covariance_diagonal"] = np.divide(
        residual_variance,
        precision_diagonal,
        out=np.full_like(precision_diagonal, np.nan, dtype=float),
        where=precision_diagonal > EPS,
    )
    curvature.to_csv(
        design_dir / "ridge_curvature_and_approx_covariance_diagonal.csv",
        index=False,
        encoding="utf-8-sig",
    )
    positive_singular = singular_values[singular_values > EPS]
    condition_number = (
        float(positive_singular.max() / positive_singular.min())
        if len(positive_singular)
        else np.nan
    )
    manifest = {
        "observation_rows": observation_rows,
        "contrast_or_anomaly_pseudo_rows": contrast_or_anomaly_rows,
        "ridge_prior_pseudo_rows": int(len(penalty)),
        "augmented_rows": int(x_aug.shape[0]),
        "columns": int(x_aug.shape[1]),
        "observation_nonzero": int(np.count_nonzero(x[:observation_rows])),
        "augmented_nonzero": int(np.count_nonzero(x_aug)),
        "rank": int(rank),
        "rank_fraction": float(rank / max(x_aug.shape[1], 1)),
        "condition_number": condition_number,
        "residual_variance": residual_variance,
        "lstsq_residual_sum": (
            float(residuals[0]) if len(residuals) else None
        ),
        "full_hessian_saved": False,
        "covariance_note": (
            "The full dense Hessian/covariance is intentionally not materialized; "
            "the complete sparse augmented design and diagonal curvature are saved "
            "and reproduce X'X exactly when needed."
        ),
    }
    (design_dir / "design_matrix_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return beta


def predict_log(df: pd.DataFrame, beta: np.ndarray, stations: list[str], mean: pd.Series, std: pd.Series) -> np.ndarray:
    x, _ = build_matrix(df, stations, mean, std)
    return x @ beta


def choose_hyperparameters(forcing: pd.DataFrame) -> tuple[dict[str, float], pd.DataFrame]:
    base = {
        "rho": 0.70,
        "wm": 480.0,
        "et_gamma": 0.75,
        "sas_rho": 0.93,
        "young_k": 1.5,
        "storage_scale": 720.0,
        "prod_capacity": 240.0,
        "runoff_gamma": 2.5,
        "quick_rho": 0.25,
        "base_rho": 0.85,
        "base_release": 0.10,
        "fixed_sigma": 3.0,
        "production_sigma": 1.5,
        "group_sigma": 1.5,
        "multistore_sigma": 0.30,
        "station_sigma": 1.0,
        "slope_sigma": 0.15,
        "regime_slope_sigma": 0.25,
        "anomaly_weight": 0.0,
        "flow_contrast_weight": 1.0,
    }
    featured = build_featured_observation_panel(
        forcing,
        rho=base["rho"],
        wm=base["wm"],
        et_gamma=base["et_gamma"],
        sas_rho=base["sas_rho"],
        young_k=base["young_k"],
        storage_scale=base["storage_scale"],
        prod_capacity=base["prod_capacity"],
        runoff_gamma=base["runoff_gamma"],
        quick_rho=base["quick_rho"],
        base_rho=base["base_rho"],
        base_release=base["base_release"],
        state_calendar_mode=STATE_CALENDAR_MODE,
    )
    tr = featured[featured["year"] <= INNER_TRAIN_END_YEAR].copy()
    iv = featured[
        (featured["year"] > INNER_TRAIN_END_YEAR)
        & (featured["year"] <= CAL_END_YEAR)
    ].copy()
    stations = sorted(featured["q_site"].unique())
    mean, std = standardize_fit(tr)
    rows = []
    for hysteresis_sigma in [0.30, 0.80, 1.50, 3.00]:
        beta = fit_map_ridge(
            tr,
            stations,
            mean,
            std,
            fixed_sigma=base["fixed_sigma"],
            production_sigma=base["production_sigma"],
            group_sigma=base["group_sigma"],
            multistore_sigma=base["multistore_sigma"],
            hysteresis_sigma=hysteresis_sigma,
            station_sigma=base["station_sigma"],
            slope_sigma=base["slope_sigma"],
            regime_slope_sigma=base["regime_slope_sigma"],
            anomaly_weight=base["anomaly_weight"],
            flow_contrast_weight=base["flow_contrast_weight"],
        )
        pred = np.exp(np.clip(predict_log(iv, beta, stations, mean, std), -20, 20))
        obs = iv["Q_obsv_cfs"].to_numpy(dtype=float)
        md = metric_dict(obs, pred)
        sm = station_median_metrics(iv, pred)
        inner_score = (
            (sm["median_NSE_log"] if np.isfinite(sm["median_NSE_log"]) else -99)
            + 1.5 * (sm["median_KGE"] if np.isfinite(sm["median_KGE"]) else -99)
            - 0.60 * min(abs(sm["median_alpha"] - 1.0), 2.0)
            - 0.005 * min(sm["median_abs_PBIAS"], 9999)
            + 0.01 * sm["good_count"]
        )
        rows.append(
            {
                **base,
                "hysteresis_sigma": hysteresis_sigma,
                "spatial_group_gates": "|".join(SPATIAL_GROUP_GATES),
                "spatial_group_features": "|".join(SPATIAL_GROUP_FEATURES),
                "multistore_features": "|".join(MULTISTORE_FEATURES),
                "hysteresis_features": "|".join(HYSTERESIS_FEATURES),
                "inner_pooled_NSE_log": md["NSE_log"],
                "inner_pooled_KGE": md["KGE_2012"],
                "inner_pooled_abs_PBIAS": abs(md["PBIAS_pct"]),
                "inner_median_NSE_log": sm["median_NSE_log"],
                "inner_median_KGE": sm["median_KGE"],
                "inner_median_alpha": sm["median_alpha"],
                "inner_median_abs_PBIAS": sm["median_abs_PBIAS"],
                "inner_good_count": sm["good_count"],
                "inner_score": inner_score,
            }
        )
    grid = pd.DataFrame(rows).sort_values("inner_score", ascending=False)
    return grid.iloc[0].to_dict(), grid


def station_metrics(pred_obs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for station, part in pred_obs.groupby("q_site"):
        row = {"q_site": station}
        for split, label in [("calibration", "cal"), ("validation", "val"), ("full", "full")]:
            sub = part if split == "full" else part[part["split"] == split]
            md = metric_dict(sub["actual"].to_numpy(dtype=float), sub["predict"].to_numpy(dtype=float))
            row.update({f"{label}_{k}": v for k, v in md.items()})
        rows.append(row)
    metrics = pd.DataFrame(rows)
    metrics["abs_val_PBIAS_pct"] = metrics["val_PBIAS_pct"].abs()
    metrics["good_validation"] = (
        (metrics["val_n"] >= 36)
        & (metrics["val_NSE_log"] >= 0.65)
        & (metrics["val_KGE_2012"] >= 0.50)
        & (metrics["abs_val_PBIAS_pct"] <= 25.0)
    )
    metrics["failure_reason"] = ""
    metrics.loc[metrics["val_n"] < 36, "failure_reason"] += "validation_n<36;"
    metrics.loc[metrics["val_NSE_log"] < 0.65, "failure_reason"] += "NSE_log<0.65;"
    metrics.loc[metrics["val_KGE_2012"] < 0.50, "failure_reason"] += "KGE<0.50;"
    metrics.loc[metrics["abs_val_PBIAS_pct"] > 25.0, "failure_reason"] += "|PBIAS|>25%;"
    metrics["failure_reason"] = metrics["failure_reason"].str.rstrip(";")
    metrics["validation_rank_score"] = (
        metrics["val_NSE_log"].clip(-2, 1)
        + metrics["val_KGE_2012"].clip(-2, 1)
        - 0.01 * metrics["abs_val_PBIAS_pct"].clip(upper=200)
    )
    return metrics.sort_values(["good_validation", "validation_rank_score"], ascending=[False, False])


def write_readme(best: dict[str, float], summary: pd.DataFrame, good: pd.DataFrame) -> None:
    lines = [
        "# 20260607_30 seasonal hysteresis Bayesian model",
        "",
        "## Mechanism",
        "",
        "This generation keeps the 20260607_29 spatial multistore production-routing Bayesian mainline, then adds explicit seasonal hysteresis states for wet-season rising limb, peak-rainfall pulse, late-season recession, and dry-season recharge memory.",
        "",
        "Base random slopes: " + ", ".join(RANDOM_SLOPE_FEATURES),
        "",
        "Production-routing features: " + ", ".join(PRODUCTION_FEATURES),
        "",
        "Spatial group gates: " + ", ".join(SPATIAL_GROUP_GATES),
        "",
        "Spatial group response features: " + ", ".join(SPATIAL_GROUP_FEATURES),
        "",
        "Spatial multistore features: " + ", ".join(MULTISTORE_FEATURES),
        "",
        "Seasonal hysteresis features: " + ", ".join(HYSTERESIS_FEATURES),
        "",
        "Season-regime gates: " + ", ".join(REGIME_GATES),
        "",
        "Season-regime gated random slopes: " + ", ".join(REGIME_SLOPE_FEATURES),
        "",
        "The fitted equation is:",
        "",
        "`logQ = fixed monthly hydrologic/SAS terms + nonlinear production-routing terms + spatial multistore states + seasonal hysteresis response states + spatial group response slopes + station random intercept + station random slopes + season x flow-regime gated station random slopes + Gaussian priors`",
        "",
        "The hysteresis states are computed only from month, PPT/AET/PET-derived wetness states, production/routing states, reach/basin attributes, and topology flags, never from validation-period performance. This is parameter-level Bayesian/MAP fitting, not prediction post-processing.",
        "",
        "## Best Hyperparameters",
        "",
        pd.DataFrame([best]).to_string(index=False),
        "",
        "## Strict Validation Summary",
        "",
        summary.to_string(index=False),
        "",
        "## Good Validation Stations",
        "",
        good[["q_site", "val_n", "val_NSE_log", "val_KGE_2012", "val_PBIAS_pct"]].head(40).to_string(index=False),
        "",
    ]
    (RUN_DIR / "README_20260607_30.md").write_text("\n".join(lines), encoding="utf-8")
    (RUN_DIR / "logs" / "run_log.md").write_text("\n".join(lines[:28]), encoding="utf-8")


def main() -> None:
    # Runners may set the mode globals directly; re-apply the feature gate so
    # every model-matrix block is synchronized before hyperparameter fitting.
    set_et_feature_block_mode(ET_FEATURE_BLOCK_MODE)
    setup_style()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "logs").mkdir(parents=True, exist_ok=True)
    forcing = load_forcing_panel()
    best, grid = choose_hyperparameters(forcing)
    grid.to_csv(REPORT_DIR / "monthly_bayes_seasonal_hysteresis_hyperparameter_grid.csv", index=False, encoding="utf-8-sig")

    featured = build_featured_observation_panel(
        forcing,
        rho=float(best["rho"]),
        wm=float(best["wm"]),
        et_gamma=float(best["et_gamma"]),
        sas_rho=float(best["sas_rho"]),
        young_k=float(best["young_k"]),
        storage_scale=float(best["storage_scale"]),
        prod_capacity=float(best["prod_capacity"]),
        runoff_gamma=float(best["runoff_gamma"]),
        quick_rho=float(best["quick_rho"]),
        base_rho=float(best["base_rho"]),
        base_release=float(best["base_release"]),
        state_calendar_mode=STATE_CALENDAR_MODE,
    )
    expected_observed_rows = int(observation_mask(forcing).sum())
    if len(featured) != expected_observed_rows:
        raise RuntimeError(
            "Hydrologic feature construction changed the observation population: "
            f"featured_rows={len(featured)}, expected_rows={expected_observed_rows}"
        )
    train = featured[featured["year"] <= CAL_END_YEAR].copy()
    stations = sorted(featured["q_site"].unique())
    mean, std = standardize_fit(train)
    pd.DataFrame(
        {
            "feature": FIXED_FEATURES,
            "training_mean": mean.reindex(FIXED_FEATURES).to_numpy(float),
            "training_std": std.reindex(FIXED_FEATURES).to_numpy(float),
        }
    ).to_csv(
        REPORT_DIR / "design_standardization.csv",
        index=False,
        encoding="utf-8-sig",
    )
    beta = fit_map_ridge(
        train,
        stations,
        mean,
        std,
        fixed_sigma=float(best["fixed_sigma"]),
        production_sigma=float(best["production_sigma"]),
        group_sigma=float(best["group_sigma"]),
        multistore_sigma=float(best["multistore_sigma"]),
        hysteresis_sigma=float(best["hysteresis_sigma"]),
        station_sigma=float(best["station_sigma"]),
        slope_sigma=float(best["slope_sigma"]),
        regime_slope_sigma=float(best["regime_slope_sigma"]),
        anomaly_weight=float(best["anomaly_weight"]),
        flow_contrast_weight=float(best["flow_contrast_weight"]),
    )
    featured["predict_log"] = predict_log(featured, beta, stations, mean, std)
    featured["predict"] = np.exp(np.clip(featured["predict_log"], -20, 20))
    featured["actual"] = featured["Q_obsv_cfs"]
    featured["split"] = np.where(featured["year"] <= CAL_END_YEAR, "calibration", "validation")

    pred_cols = [
        "comid",
        "q_site",
        "year",
        "month",
        "quarter",
        "period",
        "split",
        "actual",
        "predict",
        "Q_calc_cfs",
        "Q_ma_cfs",
        "connected_fraction",
        "connectivity_score_fold_z",
        "connected_local_cfs",
        "unconnected_local_cfs",
        "connected_upstream_cfs",
        "unconnected_upstream_cfs",
        "PPT",
        "AET",
        "PET",
        "aet_source",
        "et_operator",
        "scenario_id",
        "et_recharge_mm",
        "aet_rainfall_offset_mm",
        "aet_storage_demand_mm",
        "aet_storage_withdrawn_mm",
        "aet_unmet_mm",
        "antecedent_wetness",
        "basin_net_cfs",
        "basin_threshold_cfs",
        "sas_storage_mm",
        "sas_young_fraction",
        "sas_young_cfs",
        "sas_old_release_cfs",
        "sas_et_withdrawn_mm",
        "sas_et_unmet_mm",
        "sas_routing_et_withdrawn_mm",
        "production_storage_mm",
        "production_saturation",
        "production_quick_cfs",
        "production_base_cfs",
        "production_overflow_cfs",
        "routed_quick_cfs",
        "routed_base_cfs",
        "production_et_withdrawn_mm",
        "production_et_unmet_mm",
        "production_routing_et_withdrawn_mm",
        "ms_flash_headwater_highflow_cfs",
        "ms_flash_headwater_routed_quick_cfs",
        "ms_flash_headwater_routed_base_cfs",
        "ms_slow_large_storage_mm",
        "ms_slow_large_routed_base_cfs",
        "ms_buffer_reservoir_highflow_cfs",
        "ms_buffer_reservoir_routed_base_cfs",
        "ms_wet_large_highflow_cfs",
        "ms_wet_large_routed_quick_cfs",
        "ms_flash_headwater_et_withdrawn_mm",
        "ms_flash_headwater_et_unmet_mm",
        "ms_flash_headwater_routing_et_withdrawn_mm",
        "ms_slow_large_et_withdrawn_mm",
        "ms_slow_large_et_unmet_mm",
        "ms_slow_large_routing_et_withdrawn_mm",
        "ms_buffer_reservoir_et_withdrawn_mm",
        "ms_buffer_reservoir_et_unmet_mm",
        "ms_buffer_reservoir_routing_et_withdrawn_mm",
        "ms_wet_large_et_withdrawn_mm",
        "ms_wet_large_et_unmet_mm",
        "ms_wet_large_routing_et_withdrawn_mm",
        *HYSTERESIS_FEATURES,
        *SPATIAL_GROUP_GATES,
        "high_flow_regime_gate",
        "low_flow_regime_gate",
        "wet_high_regime_gate",
        "wet_low_regime_gate",
        "dry_high_regime_gate",
        "dry_low_regime_gate",
    ]
    pred_obs = featured[pred_cols].copy()
    pred_obs["et_feature_block"] = ET_FEATURE_BLOCK_MODE
    population_audit = {
        "forcing_rows": int(len(forcing)),
        "forcing_observed_rows": int(observation_mask(forcing).sum()),
        "featured_rows": int(len(featured)),
        "featured_station_count": int(featured["q_site"].nunique()),
        "training_rows": int(len(train)),
        "training_station_count": int(train["q_site"].nunique()),
        "registered_station_count": int(len(stations)),
        "prediction_rows_before_write": int(len(pred_obs)),
        "prediction_station_count_before_write": int(pred_obs["q_site"].nunique()),
        "calibration_end_year": int(CAL_END_YEAR),
    }
    (REPORT_DIR / "model_population_audit.json").write_text(
        json.dumps(population_audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    # Parquet is the authoritative machine-readable artifact.  CSV remains a
    # human-readable mirror and is verified by the fold runner before use.
    pred_obs.to_parquet(
        REPORT_DIR / "monthly_bayes_seasonal_hysteresis_prediction_vs_observed_2006_2022.parquet",
        index=False,
    )
    # Keep the CSV mirror compact.  The full audit-width frame exceeds the
    # reliable single-file text size of the synchronized Test volume; Parquet
    # above remains complete and authoritative.
    csv_columns = [
        "comid", "q_site", "year", "month", "split", "actual", "predict",
        "Q_calc_cfs", "Q_ma_cfs", "PPT", "AET", "PET", "scenario_id",
        "connected_fraction", "connected_upstream_cfs",
        "unconnected_upstream_cfs", "et_feature_block",
    ]
    pred_obs[csv_columns].to_csv(
        REPORT_DIR / "monthly_bayes_seasonal_hysteresis_prediction_vs_observed_2006_2022.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metrics = station_metrics(pred_obs)
    metrics.to_csv(REPORT_DIR / "monthly_bayes_seasonal_hysteresis_metrics_by_station.csv", index=False, encoding="utf-8-sig")
    good = metrics[metrics["good_validation"]].copy()
    not_good = metrics[~metrics["good_validation"]].copy()
    good.to_csv(REPORT_DIR / "monthly_bayes_seasonal_hysteresis_good_validation_stations.csv", index=False, encoding="utf-8-sig")
    not_good.to_csv(REPORT_DIR / "monthly_bayes_seasonal_hysteresis_not_good_validation_stations.csv", index=False, encoding="utf-8-sig")

    summary_rows = []
    for split, label in [("calibration", "cal"), ("validation", "val"), ("full", "full")]:
        rows = metrics if split == "full" else metrics[metrics[f"{label}_n"] > 0]
        summary_rows.append(
            {
                "split": split,
                "stations": int(len(rows)),
                "rows": int(len(pred_obs) if split == "full" else (pred_obs["split"] == split).sum()),
                "median_NSE_raw": float(rows[f"{label}_NSE_raw"].median()),
                "median_NSE_log": float(rows[f"{label}_NSE_log"].median()),
                "median_KGE": float(rows[f"{label}_KGE_2012"].median()),
                "median_abs_PBIAS_pct": float(rows[f"{label}_PBIAS_pct"].abs().median()),
                "good_validation_station_count": int(len(good)) if split == "validation" else np.nan,
                "not_good_validation_station_count": int(len(not_good)) if split == "validation" else np.nan,
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(REPORT_DIR / "monthly_bayes_seasonal_hysteresis_metric_summary.csv", index=False, encoding="utf-8-sig")

    param_rows = [{"parameter": "intercept", "coefficient": beta[0], "feature_group": "intercept", "prior_sigma": np.nan}]
    for i, feat in enumerate(FIXED_FEATURES, start=1):
        param_rows.append(
            {
                "parameter": feat,
                "coefficient_standardized": beta[i],
                "feature_group": feature_group(feat),
                    "prior_sigma": float(
                        best["multistore_sigma"]
                        if feat in MULTISTORE_FEATURES
                        else best["hysteresis_sigma"]
                        if feat in HYSTERESIS_FEATURES
                        else best["production_sigma"]
                        if feat in PRODUCTION_FEATURES
                        else best["fixed_sigma"]
                    ),
            }
        )
    n_fixed_base = 1 + len(FIXED_FEATURES)
    n_group = len(SPATIAL_GROUP_GATES) * len(SPATIAL_GROUP_FEATURES)
    group_rows = []
    offset = n_fixed_base
    for gate_i, gate in enumerate(SPATIAL_GROUP_GATES):
        for feat_i, feat in enumerate(SPATIAL_GROUP_FEATURES):
            value = beta[offset + gate_i * len(SPATIAL_GROUP_FEATURES) + feat_i]
            group_rows.append(
                {
                    "parameter": f"spatial_group_slope_{gate}_{feat}",
                    "coefficient_standardized": value,
                    "feature_group": "spatial_group_hydrologic_response",
                    "prior_sigma": float(best["group_sigma"]),
                    "spatial_group_gate": gate,
                    "slope_feature": feat,
                }
            )
    n_fixed = n_fixed_base + n_group
    n_station = len(stations)
    station_effect = pd.DataFrame({"q_site": stations, "station_random_intercept": beta[n_fixed : n_fixed + n_station]})
    offset = n_fixed + n_station
    slope_rows = []
    for feat_i, feat in enumerate(RANDOM_SLOPE_FEATURES):
        vals = beta[offset + feat_i * n_station : offset + (feat_i + 1) * n_station]
        station_effect[f"random_slope_{feat}"] = vals
        for station, value in zip(stations, vals):
            slope_rows.append(
                {
                    "parameter": f"random_slope_{feat}::{station}",
                    "coefficient_standardized": value,
                    "feature_group": "station_random_hydrologic_sas_slope",
                    "prior_sigma": float(best["slope_sigma"]),
                    "q_site": station,
                    "slope_feature": feat,
                }
            )
    offset += len(RANDOM_SLOPE_FEATURES) * n_station
    regime_rows = []
    for gate_i, gate in enumerate(REGIME_GATES):
        for feat_i, feat in enumerate(REGIME_SLOPE_FEATURES):
            block_i = gate_i * len(REGIME_SLOPE_FEATURES) + feat_i
            vals = beta[offset + block_i * n_station : offset + (block_i + 1) * n_station]
            station_effect[f"regime_random_slope_{gate}_{feat}"] = vals
            for station, value in zip(stations, vals):
                regime_rows.append(
                    {
                        "parameter": f"regime_random_slope_{gate}_{feat}::{station}",
                        "coefficient_standardized": value,
                        "feature_group": "station_random_season_regime_slope",
                        "prior_sigma": float(best["regime_slope_sigma"]),
                        "q_site": station,
                        "regime_gate": gate,
                        "slope_feature": feat,
                    }
                )
    fixed_params = pd.DataFrame(param_rows)
    spatial_group_params = pd.DataFrame(group_rows)
    random_slope_params = pd.DataFrame(slope_rows)
    regime_slope_params = pd.DataFrame(regime_rows)
    fixed_params.to_csv(REPORT_DIR / "monthly_bayes_seasonal_hysteresis_fixed_parameters.csv", index=False, encoding="utf-8-sig")
    spatial_group_params.to_csv(REPORT_DIR / "monthly_bayes_seasonal_hysteresis_spatial_group_parameters.csv", index=False, encoding="utf-8-sig")
    random_slope_params.to_csv(REPORT_DIR / "monthly_bayes_seasonal_hysteresis_station_slope_parameters.csv", index=False, encoding="utf-8-sig")
    regime_slope_params.to_csv(REPORT_DIR / "monthly_bayes_seasonal_hysteresis_regime_slope_parameters.csv", index=False, encoding="utf-8-sig")
    station_effect.to_csv(REPORT_DIR / "monthly_bayes_seasonal_hysteresis_station_random_effects.csv", index=False, encoding="utf-8-sig")
    pd.concat(
        [
            fixed_params[fixed_params["parameter"] != "intercept"][["parameter", "coefficient_standardized", "feature_group", "prior_sigma"]],
            spatial_group_params[["parameter", "coefficient_standardized", "feature_group", "prior_sigma"]],
            random_slope_params[["parameter", "coefficient_standardized", "feature_group", "prior_sigma"]],
            regime_slope_params[["parameter", "coefficient_standardized", "feature_group", "prior_sigma"]],
        ],
        ignore_index=True,
    ).assign(abs_coef=lambda x: x["coefficient_standardized"].abs()).groupby(["feature_group", "prior_sigma"], dropna=False).agg(
        parameters=("parameter", "count"),
        mean_abs_coef=("abs_coef", "mean"),
        max_abs_coef=("abs_coef", "max"),
        sum_abs_coef=("abs_coef", "sum"),
    ).reset_index().sort_values("sum_abs_coef", ascending=False).to_csv(
        REPORT_DIR / "monthly_bayes_seasonal_hysteresis_parameter_group_strength.csv", index=False, encoding="utf-8-sig"
    )

    write_readme(best, summary, good)
    print(summary.to_string(index=False))
    print(f"good_validation_stations={len(good)}")
    print(f"best_hyperparameters={best}")


if __name__ == "__main__":
    main()
