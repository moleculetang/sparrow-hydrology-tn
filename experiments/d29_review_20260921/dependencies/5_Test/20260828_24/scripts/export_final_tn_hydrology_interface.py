"""Export the locked, state-consistent R2 Reach/reservoir interface through 2024."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
STAGE = ROOT / "5_Test" / "20260828_24"
S13 = ROOT / "5_Test" / "20260828_13"
S14 = ROOT / "5_Test" / "20260828_14"
S15 = ROOT / "5_Test" / "20260828_15"
S20 = ROOT / "5_Test" / "20260828_20"
S21 = ROOT / "5_Test" / "20260828_21"
S23 = ROOT / "5_Test" / "20260828_23"

for folder in [S20 / "scripts", S15 / "scripts", S14 / "scripts"]:
    sys.path.insert(0, str(folder))

from reservoir_development_core import (  # noqa: E402
    PARENT,
    SECONDS_PER_DAY,
    build_runtimes,
    load_static_context,
    prepare_reservoir_metadata,
)
from reservoir_network_router import build_reach_graph, route_network_steps  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_fraction(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator > 0,
    )


def build_reach_products(
    parent: pd.DataFrame,
    routed_fast: np.ndarray,
    routed_slow: np.ndarray,
    routed_direct: np.ndarray,
    routed_age: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = parent.sort_values(["date", "reach_id"], kind="mergesort").reset_index(drop=True).copy()
    fast = (routed_fast / SECONDS_PER_DAY).reshape(-1)
    slow = (routed_slow / SECONDS_PER_DAY).reshape(-1)
    direct = (routed_direct / SECONDS_PER_DAY).reshape(-1)
    age_rate = (routed_age / SECONDS_PER_DAY).reshape(-1)
    total = fast + slow + direct

    frame = frame.rename(
        columns={
            "routed_fast_response_m3_s": "parent_routed_fast_response_m3_s",
            "routed_slow_response_m3_s": "parent_routed_slow_response_m3_s",
            "routed_total_m3_s": "parent_routed_total_m3_s",
            "state_consistent_fast_fraction": "parent_state_consistent_fast_fraction",
        }
    )
    frame["routed_fast_response_m3_s"] = fast
    frame["routed_slow_response_m3_s"] = slow
    frame["routed_direct_response_m3_s"] = direct
    frame["routed_total_m3_s"] = total
    frame["state_consistent_fast_fraction"] = safe_fraction(fast, total)
    frame["state_consistent_slow_fraction"] = safe_fraction(slow, total)
    frame["state_consistent_direct_fraction"] = safe_fraction(direct, total)
    frame["routed_water_age_moment_m3_s_day"] = age_rate
    frame["mean_routed_water_age_day"] = safe_fraction(age_rate, total)
    frame["reservoir_model"] = "R2_tau180_a0.25_d0.00"

    mean_columns = [
        "local_fast_response_m3_s",
        "local_slow_response_m3_s",
        "parent_routed_fast_response_m3_s",
        "parent_routed_slow_response_m3_s",
        "parent_routed_total_m3_s",
        "routed_fast_response_m3_s",
        "routed_slow_response_m3_s",
        "routed_direct_response_m3_s",
        "routed_total_m3_s",
        "percolation_to_lower_mm_day",
        "soil_storage_mm",
        "upper_response_storage_mm",
        "lower_slow_storage_mm",
        "actual_aet_mm_day",
        "routed_water_age_moment_m3_s_day",
    ]
    monthly = (
        frame.assign(month=pd.to_datetime(frame["date"]).dt.to_period("M").dt.to_timestamp())
        .groupby(["month", "reach_id"], as_index=False)[mean_columns]
        .mean()
    )
    mfast = monthly["routed_fast_response_m3_s"].to_numpy(float)
    mslow = monthly["routed_slow_response_m3_s"].to_numpy(float)
    mdirect = monthly["routed_direct_response_m3_s"].to_numpy(float)
    mtotal = monthly["routed_total_m3_s"].to_numpy(float)
    mage = monthly["routed_water_age_moment_m3_s_day"].to_numpy(float)
    monthly["state_consistent_fast_fraction"] = safe_fraction(mfast, mtotal)
    monthly["state_consistent_slow_fraction"] = safe_fraction(mslow, mtotal)
    monthly["state_consistent_direct_fraction"] = safe_fraction(mdirect, mtotal)
    monthly["mean_routed_water_age_day"] = safe_fraction(mage, mtotal)
    monthly["reservoir_model"] = "R2_tau180_a0.25_d0.00"
    return frame, monthly


def build_reservoir_products(
    diagnostics: tuple[dict[str, object], ...],
    metadata: dict[str, dict[str, object]],
    priority: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    daily = pd.DataFrame(diagnostics)
    daily["date"] = pd.to_datetime(daily["date"])
    diagnostic_capture = (
        daily.pop("local_capture_fraction")
        if "local_capture_fraction" in daily
        else pd.Series(np.nan, index=daily.index, dtype=float)
    )
    static = pd.DataFrame(
        [{"reservoir_entity_id": entity, **item} for entity, item in metadata.items()]
    )
    priority_columns = priority[
        ["reservoir_entity_id", "domain_storage_scale", "notes"]
    ].drop_duplicates("reservoir_entity_id")
    static = static.merge(priority_columns, on="reservoir_entity_id", how="left", validate="one_to_one")
    static["interior_partial_capture_flag"] = static["local_capture_fraction"].astype(float) < 1.0
    static["partial_domain_storage_flag"] = static["domain_storage_scale"].astype(float) < 1.0
    static["extension_capacity_prior_flag"] = static["capacity_source"].astype(str).str.contains(
        "extension-only", case=False, na=False
    )
    static["uncertainty_flag"] = (
        static["interior_partial_capture_flag"]
        | static["partial_domain_storage_flag"]
        | static["extension_capacity_prior_flag"]
    )
    merge_columns = [
        "reservoir_entity_id",
        "name",
        "capacity_m3",
        "capacity_source",
        "regulation_class",
        "active_start",
        "local_capture_fraction",
        "domain_storage_scale",
        "interior_partial_capture_flag",
        "partial_domain_storage_flag",
        "extension_capacity_prior_flag",
        "uncertainty_flag",
    ]
    daily = daily.merge(static[merge_columns], on="reservoir_entity_id", how="left", validate="many_to_one")
    capture_valid = diagnostic_capture.notna().to_numpy()
    if capture_valid.any():
        capture_error = np.abs(
            diagnostic_capture.to_numpy(float)[capture_valid]
            - daily["local_capture_fraction"].to_numpy(float)[capture_valid]
        )
        if float(capture_error.max()) > 1.0e-12:
            raise RuntimeError("Daily operator and static local-capture fractions disagree")
    daily["mean_storage_age_day"] = safe_fraction(
        daily["storage_age_moment_m3_day"].to_numpy(float), daily["storage_m3"].to_numpy(float)
    )
    for field in ["controlled_release", "spill", "total_release", "release_origin_fast", "release_origin_slow", "release_origin_direct"]:
        daily[f"{field}_m3_s"] = daily[field].astype(float) / SECONDS_PER_DAY
    total_release = daily["total_release"].to_numpy(float)
    daily["release_fast_fraction"] = safe_fraction(daily["release_origin_fast"].to_numpy(float), total_release)
    daily["release_slow_fraction"] = safe_fraction(daily["release_origin_slow"].to_numpy(float), total_release)
    daily["release_direct_fraction"] = safe_fraction(daily["release_origin_direct"].to_numpy(float), total_release)
    storage = daily["storage_m3"].to_numpy(float)
    daily["storage_fast_fraction"] = safe_fraction(daily["storage_fast_m3"].to_numpy(float), storage)
    daily["storage_slow_fraction"] = safe_fraction(daily["storage_slow_m3"].to_numpy(float), storage)
    daily["storage_direct_fraction"] = safe_fraction(daily["storage_direct_m3"].to_numpy(float), storage)

    daily = daily.sort_values(["date", "reservoir_entity_id"], kind="mergesort").reset_index(drop=True)
    daily["month"] = daily["date"].dt.to_period("M").dt.to_timestamp()
    sum_columns = [
        "controlled_release", "spill", "total_release", "release_origin_fast",
        "release_origin_slow", "release_origin_direct", "release_age_moment",
    ]
    mean_columns = ["storage_m3", "storage_fast_m3", "storage_slow_m3", "storage_direct_m3", "storage_age_moment_m3_day"]
    monthly_sum = daily.groupby(["month", "reservoir_entity_id"], as_index=False)[sum_columns].sum()
    monthly_mean = daily.groupby(["month", "reservoir_entity_id"], as_index=False)[mean_columns].mean()
    monthly_last = (
        daily.groupby(["month", "reservoir_entity_id"], as_index=False)
        .tail(1)[["month", "reservoir_entity_id", "storage_m3", "storage_fast_m3", "storage_slow_m3", "storage_direct_m3", "storage_age_moment_m3_day"]]
        .rename(columns={name: f"month_end_{name}" for name in mean_columns})
    )
    active_days = daily.groupby(["month", "reservoir_entity_id"], as_index=False)["enabled"].sum().rename(columns={"enabled": "active_days"})
    monthly = monthly_sum.merge(monthly_mean, on=["month", "reservoir_entity_id"], validate="one_to_one")
    monthly = monthly.merge(monthly_last, on=["month", "reservoir_entity_id"], validate="one_to_one")
    monthly = monthly.merge(active_days, on=["month", "reservoir_entity_id"], validate="one_to_one")
    monthly = monthly.merge(static[merge_columns], on="reservoir_entity_id", how="left", validate="many_to_one")
    monthly["mean_release_m3_s"] = monthly["total_release"] / (
        monthly["month"].dt.days_in_month.astype(float) * SECONDS_PER_DAY
    )
    monthly["mean_release_age_day"] = safe_fraction(
        monthly["release_age_moment"].to_numpy(float), monthly["total_release"].to_numpy(float)
    )
    monthly["month_end_mean_storage_age_day"] = safe_fraction(
        monthly["month_end_storage_age_moment_m3_day"].to_numpy(float),
        monthly["month_end_storage_m3"].to_numpy(float),
    )
    monthly["release_fast_fraction"] = safe_fraction(
        monthly["release_origin_fast"].to_numpy(float), monthly["total_release"].to_numpy(float)
    )
    monthly["release_slow_fraction"] = safe_fraction(
        monthly["release_origin_slow"].to_numpy(float), monthly["total_release"].to_numpy(float)
    )
    monthly["release_direct_fraction"] = safe_fraction(
        monthly["release_origin_direct"].to_numpy(float), monthly["total_release"].to_numpy(float)
    )
    daily = daily.rename(
        columns={
            "captured_inflow": "captured_inflow_m3",
            "bypass_inflow": "bypass_inflow_m3",
            "controlled_release": "controlled_release_m3",
            "spill": "spill_m3",
            "total_release": "total_release_m3",
            "release_origin_fast": "release_origin_fast_m3",
            "release_origin_slow": "release_origin_slow_m3",
            "release_origin_direct": "release_origin_direct_m3",
            "release_age_moment": "release_age_moment_m3_day",
            "mass_balance_error": "mass_balance_error_m3",
            "fast_balance_error": "fast_balance_error_m3",
            "slow_balance_error": "slow_balance_error_m3",
            "direct_balance_error": "direct_balance_error_m3",
            "age_moment_balance_error": "age_moment_balance_error_m3_day",
            "state_tracer_error": "state_tracer_error_m3",
            "release_tracer_error": "release_tracer_error_m3",
        }
    )
    monthly = monthly.rename(
        columns={
            "controlled_release": "controlled_release_volume_m3",
            "spill": "spill_volume_m3",
            "total_release": "total_release_volume_m3",
            "release_origin_fast": "release_origin_fast_volume_m3",
            "release_origin_slow": "release_origin_slow_volume_m3",
            "release_origin_direct": "release_origin_direct_volume_m3",
            "release_age_moment": "release_age_moment_m3_day",
        }
    )
    daily = daily.drop(columns="month")
    return daily, monthly, static


def main() -> None:
    outputs = STAGE / "outputs"
    reports = STAGE / "reports"
    locks = STAGE / "locks"
    for folder in [outputs, reports, locks]:
        folder.mkdir(parents=True, exist_ok=True)

    retrospective = json.loads(
        (S23 / "locks" / "retrospective_tn_interface_lock.json").read_text(encoding="utf-8")
    )
    if retrospective["status"] != "R2_RETROSPECTIVE_NONINFERIOR_TN_INTERFACE_AUTHORIZED":
        raise RuntimeError("Stage 23 did not authorize the R2 export")
    development = json.loads(
        (S21 / "locks" / "boundary_complete_development_lock.json").read_text(encoding="utf-8")
    )
    selected = development["selected_candidates"]["R2"]
    if selected != {
        "candidate_id": "R2_tau180_a0.25_d0.00",
        "tau_day": 180.0,
        "inflow_response": 0.25,
        "drawdown_fraction": 0.0,
    }:
        raise RuntimeError("Locked R2 candidate changed")

    context = load_static_context("2024-12-31")
    metadata = prepare_reservoir_metadata(context)
    graph = build_reach_graph(context["topology"], context["reach_ids"])
    runtimes = build_runtimes(
        metadata, "R2", selected["tau_day"], selected["inflow_response"], selected["drawdown_fraction"]
    )
    routed = route_network_steps(
        context["local_fast_rate"] * SECONDS_PER_DAY,
        context["local_slow_rate"] * SECONDS_PER_DAY,
        graph,
        runtimes,
        dates=context["dates"],
        values_are_volumes_per_step=True,
        keep_reservoir_diagnostics=True,
    )
    parent = pd.read_parquet(PARENT)
    reach_daily, reach_monthly = build_reach_products(
        parent, routed.routed_fast, routed.routed_slow, routed.routed_direct, routed.routed_age_moment
    )
    reservoir_daily, reservoir_monthly, static = build_reservoir_products(
        routed.reservoir_diagnostics, metadata, context["priority"]
    )

    reach_daily_path = outputs / "tn_hydrology_reach_daily_2006_2024.parquet"
    reach_monthly_path = outputs / "tn_hydrology_reach_monthly_2006_2024.parquet"
    reservoir_daily_path = outputs / "tn_hydrology_reservoir_daily_2006_2024.parquet"
    reservoir_monthly_path = outputs / "tn_hydrology_reservoir_monthly_2006_2024.parquet"
    static_path = outputs / "tn_hydrology_reservoir_static_metadata.parquet"
    reach_daily.to_parquet(reach_daily_path, index=False, compression="zstd")
    reach_monthly.to_parquet(reach_monthly_path, index=False, compression="zstd")
    reservoir_daily.to_parquet(reservoir_daily_path, index=False, compression="zstd")
    reservoir_monthly.to_parquet(reservoir_monthly_path, index=False, compression="zstd")
    static.to_parquet(static_path, index=False, compression="zstd")

    flow_error = np.abs(
        reach_daily["routed_total_m3_s"].to_numpy(float)
        - reach_daily[["routed_fast_response_m3_s", "routed_slow_response_m3_s", "routed_direct_response_m3_s"]].sum(axis=1).to_numpy(float)
    )
    positive = reach_daily["routed_total_m3_s"].to_numpy(float) > 0
    fraction_sum = reach_daily[["state_consistent_fast_fraction", "state_consistent_slow_fraction", "state_consistent_direct_fraction"]].sum(axis=1).to_numpy(float)
    qa = {
        "status": "PASS_TN_HYDROLOGY_INTERFACE_QA",
        "model": selected,
        "date_min": str(pd.to_datetime(reach_daily["date"]).min().date()),
        "date_max": str(pd.to_datetime(reach_daily["date"]).max().date()),
        "reach_daily_rows": int(len(reach_daily)),
        "reach_monthly_rows": int(len(reach_monthly)),
        "reservoir_daily_rows": int(len(reservoir_daily)),
        "reservoir_monthly_rows": int(len(reservoir_monthly)),
        "reach_count": int(reach_daily["reach_id"].nunique()),
        "reservoir_count": int(reservoir_daily["reservoir_entity_id"].nunique()),
        "maximum_flow_component_closure_m3_s": float(flow_error.max()),
        "maximum_positive_flow_fraction_sum_error": float(np.abs(fraction_sum[positive] - 1.0).max()),
        "minimum_reach_flow_m3_s": float(reach_daily["routed_total_m3_s"].min()),
        "maximum_reservoir_mass_balance_error_m3": float(reservoir_daily["mass_balance_error_m3"].abs().max()),
        "maximum_reservoir_fast_balance_error_m3": float(reservoir_daily["fast_balance_error_m3"].abs().max()),
        "maximum_reservoir_slow_balance_error_m3": float(reservoir_daily["slow_balance_error_m3"].abs().max()),
        "maximum_reservoir_direct_balance_error_m3": float(reservoir_daily["direct_balance_error_m3"].abs().max()),
        "maximum_reservoir_age_moment_balance_error_m3_day": float(reservoir_daily["age_moment_balance_error_m3_day"].abs().max()),
        "maximum_reservoir_state_tracer_error_m3": float(reservoir_daily["state_tracer_error_m3"].abs().max()),
        "maximum_reservoir_release_tracer_error_m3": float(reservoir_daily["release_tracer_error_m3"].abs().max()),
        "extension_capacity_prior_entities": static.loc[static["extension_capacity_prior_flag"], "reservoir_entity_id"].tolist(),
        "partial_domain_storage_entities": static.loc[static["partial_domain_storage_flag"], "reservoir_entity_id"].tolist(),
        "interior_partial_capture_entities": static.loc[static["interior_partial_capture_flag"], "reservoir_entity_id"].tolist(),
        "claim_boundary": "Source fractions are conservative model-state tracers; reservoir-specific release observations were unavailable.",
        "tn_observations_opened": False,
    }
    numeric_gates = [
        qa["date_min"] == "2006-01-01",
        qa["date_max"] == "2024-12-31",
        qa["reach_daily_rows"] == 6940 * 230,
        qa["reach_count"] == 230,
        qa["reservoir_count"] == 13,
        qa["minimum_reach_flow_m3_s"] >= 0,
        qa["maximum_flow_component_closure_m3_s"] <= 1.0e-10,
        qa["maximum_positive_flow_fraction_sum_error"] <= 1.0e-12,
        qa["maximum_reservoir_mass_balance_error_m3"] <= 1.0e-3,
        qa["maximum_reservoir_state_tracer_error_m3"] <= 1.0e-3,
        qa["maximum_reservoir_release_tracer_error_m3"] <= 1.0e-3,
    ]
    if not all(numeric_gates):
        qa["status"] = "FAIL_TN_HYDROLOGY_INTERFACE_QA"

    qa_text = json.dumps(qa, ensure_ascii=False, indent=2)
    (reports / "tn_hydrology_interface_qa.json").write_text(qa_text, encoding="utf-8")
    hashes = {
        "stage": "20260828_24",
        "status": qa["status"],
        "model": selected,
        "files": {
            "contract": sha256(STAGE / "experiment_contract.json"),
            "stage23_lock": sha256(S23 / "locks" / "retrospective_tn_interface_lock.json"),
            "parent": sha256(PARENT),
            "runner_code": sha256(Path(__file__)),
            "reach_daily": sha256(reach_daily_path),
            "reach_monthly": sha256(reach_monthly_path),
            "reservoir_daily": sha256(reservoir_daily_path),
            "reservoir_monthly": sha256(reservoir_monthly_path),
            "reservoir_static": sha256(static_path),
            "qa": sha256(reports / "tn_hydrology_interface_qa.json"),
        },
    }
    lock_text = json.dumps(hashes, ensure_ascii=False, indent=2)
    (locks / "final_tn_hydrology_interface_lock.json").write_text(lock_text, encoding="utf-8")
    print(qa_text)
    if qa["status"] != "PASS_TN_HYDROLOGY_INTERFACE_QA":
        raise SystemExit(qa["status"])


if __name__ == "__main__":
    main()
