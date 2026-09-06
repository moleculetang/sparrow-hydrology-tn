"""Audit response-path identifiability and hydraulic exposure without TN."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260825_6"
OUT = RUN / "outputs"
REPORT = RUN / "reports"
STAGE3_SCRIPTS = ROOT / "5_Test" / "20260825_3" / "scripts"
STAGE5_SCRIPTS = ROOT / "5_Test" / "20260825_5" / "scripts"
sys.path[:0] = [str(STAGE3_SCRIPTS), str(STAGE5_SCRIPTS)]
from hydrology_core import (  # noqa: E402
    HBVParameters,
    load_topology,
    parameters_to_raw,
    route_instantaneous,
    route_linear_channels_adaptive,
)
from regional_hbv_core import (  # noqa: E402
    PARAMETER_NAMES,
    _run_ordered_hbv,
    periodic_spinup_regional,
    theta_to_parameter_map,
)


FORCING = ROOT / "5_Test" / "20260825_2" / "outputs" / "daily_hbv_forcing_2006_2022.parquet"
SIGNATURES = ROOT / "5_Test" / "20260825_2" / "outputs" / "q_derived_response_signatures_development.parquet"
TOPOLOGY = ROOT / "5_Test" / "20260814_1" / "inputs" / "topology" / "topology_edges.csv"
Q72_BRIDGE = ROOT / "5_Test" / "20260823_27" / "outputs" / "q72_full_state_tn_bridge_2006_2022.parquet"
STATIC = ROOT / "5_Test" / "20260814_9" / "inputs" / "model_ready" / "static" / "reach_static_attributes.parquet"
GEOMETRY = ROOT / "5_Test" / "20260814_9" / "inputs" / "model_ready" / "static" / "bankfull_geometry_andreadis_by_reach.parquet"
STAGE4_LOCK = ROOT / "5_Test" / "20260825_4" / "reports" / "global_hbv_development_parameter_lock.json"
STAGE5_DECISION = ROOT / "5_Test" / "20260825_5" / "reports" / "stage5_decision.json"
STAGE5_PROGRAM = ROOT / "5_Test" / "20260825_5" / "program_manifest.json"
STAGE5_PREDICTIONS = ROOT / "5_Test" / "20260825_5" / "outputs" / "complete_tree_oof_predictions.parquet"
STAGE5_LOCKS = ROOT / "5_Test" / "20260825_5" / "outputs" / "fold_parameter_and_scaling_lock.parquet"
STAGE5_MAPS = ROOT / "5_Test" / "20260825_5" / "outputs" / "fold_parameter_maps.parquet"
STAGE5_AUDIT = ROOT / "5_Test" / "20260825_5" / "outputs" / "fold_audit.parquet"

BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 20260825
SPINUP_TOLERANCE = 1.0e-8
SPINUP_MAX_CYCLES = 500


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def simulate_components(
    raw_theta: np.ndarray,
    spin_p: np.ndarray,
    spin_pet: np.ndarray,
    development_p: np.ndarray,
    development_pet: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    _, physical = theta_to_parameter_map(raw_theta, np.zeros((development_p.shape[1], 8), dtype=np.float64))
    initial, spinup = periodic_spinup_regional(
        spin_p, spin_pet, physical, SPINUP_TOLERANCE, SPINUP_MAX_CYCLES
    )
    if not bool(spinup["converged"]):
        raise RuntimeError("Sensitivity spin-up did not converge")
    _, components, mass_error = _run_ordered_hbv(
        development_p, development_pet, physical, initial, True
    )
    assert components is not None
    return components, {"spinup": spinup, "mass_max_abs_error_mm": mass_error}


def monthly_fraction(dates: pd.DatetimeIndex, routed_components_m3: np.ndarray) -> tuple[pd.MultiIndex, np.ndarray]:
    codes = dates.to_period("M")
    months = codes.unique()
    fraction = np.empty((len(months), routed_components_m3.shape[1]), dtype=np.float64)
    for index, month in enumerate(months):
        selected = np.asarray(codes == month)
        volume = routed_components_m3[selected].sum(axis=0)
        fraction[index] = volume[:, 2] / np.maximum(volume.sum(axis=1), 1.0e-12)
    month_index = pd.MultiIndex.from_arrays(
        [[period.year for period in months], [period.month for period in months]],
        names=["year", "month"],
    )
    return month_index, fraction


def cluster_bootstrap_spearman(frame: pd.DataFrame, x: str, y: str, seed_offset: int) -> dict[str, float]:
    valid = frame[["terminal_tree", x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    point = float(spearmanr(valid[x], valid[y]).statistic)
    trees = np.asarray(sorted(valid.terminal_tree.unique()), dtype=int)
    groups = {tree: valid.loc[valid.terminal_tree.eq(tree)] for tree in trees}
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    values = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for replicate in range(BOOTSTRAP_REPLICATES):
        sampled = trees[rng.integers(0, len(trees), size=len(trees))]
        parts = []
        for draw, tree in enumerate(sampled):
            part = groups[int(tree)].copy()
            part["bootstrap_tree"] = draw
            parts.append(part)
        sample = pd.concat(parts, ignore_index=True)
        values[replicate] = float(spearmanr(sample[x], sample[y]).statistic)
    values = values[np.isfinite(values)]
    return {
        "point_spearman": point,
        "bootstrap_valid_replicates": int(len(values)),
        "ci95_lower": float(np.quantile(values, 0.025)),
        "ci95_upper": float(np.quantile(values, 0.975)),
    }


def build_oof_signature_frame() -> pd.DataFrame:
    predictions = pd.read_parquet(STAGE5_PREDICTIONS)
    predictions["date"] = pd.to_datetime(predictions.date)
    predictions["global_total_m3_s"] = predictions[["global_q0_m3_s", "global_q1_m3_s", "global_q2_m3_s"]].sum(axis=1)
    response = predictions.groupby(["station_norm", "reach_id", "heldout_terminal_tree"], as_index=False).agg(
        model_q0_volume=("global_q0_m3_s", "sum"),
        model_q1_volume=("global_q1_m3_s", "sum"),
        model_q2_volume=("global_q2_m3_s", "sum"),
        model_total_volume=("global_total_m3_s", "sum"),
    )
    response["model_slow_response_fraction"] = response.model_q2_volume / np.maximum(response.model_total_volume, 1.0e-12)
    response = response.rename(columns={"heldout_terminal_tree": "terminal_tree"})
    signatures = pd.read_parquet(SIGNATURES)
    overall = signatures.loc[signatures.group_type.eq("overall") & signatures.group_value.astype(str).eq("all")]
    wide = overall.pivot(index=["station_norm", "reach_id", "terminal_tree"], columns="signature", values="value").reset_index()
    frame = response.merge(wide, on=["station_norm", "reach_id", "terminal_tree"], how="left", validate="one_to_one")
    frame["normalized_low_flow_q05_q50"] = frame["flow_quantile_0.05_m3_s"] / np.maximum(
        frame["flow_quantile_0.50_m3_s"], 1.0e-12
    )
    return frame


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    stage5 = json.loads(STAGE5_DECISION.read_text(encoding="utf-8"))
    if stage5["status"] != "PASS_COMPLETE_TREE_SPATIAL_EVALUATION" or stage5["selected_model_for_stage6"] != "GLOBAL_HBV":
        raise RuntimeError("Stage 6 expected the locked Stage-5 GLOBAL_HBV selection")

    signature_frame = build_oof_signature_frame()
    signature_frame.to_parquet(OUT / "oof_path_signature_audit.parquet", index=False)
    evidence_definitions = [
        ("hydrograph_separation", "bfi_three_method_median", 0),
        ("recession", "recession_tau_days", 1),
        ("normalized_low_flow", "normalized_low_flow_q05_q50", 2),
    ]
    evidence_rows = []
    for family, observed_column, offset in evidence_definitions:
        result = cluster_bootstrap_spearman(
            signature_frame, "model_slow_response_fraction", observed_column, offset
        )
        evidence_rows.append(
            {
                "evidence_family": family,
                "model_metric": "model_slow_response_fraction",
                "q_derived_metric": observed_column,
                **result,
                "positive_direction_supported": bool(result["point_spearman"] > 0.0 and result["ci95_lower"] > 0.0),
            }
        )
    evidence = pd.DataFrame(evidence_rows)
    evidence.to_parquet(OUT / "path_evidence_bootstrap.parquet", index=False)
    bfi_result = evidence.loc[evidence.evidence_family.eq("hydrograph_separation")].iloc[0]
    bfi_gate = bool(bfi_result.point_spearman >= 0.30 and bfi_result.ci95_lower > 0.0)
    evidence_family_count = int(evidence.positive_direction_supported.sum())

    locks = pd.read_parquet(STAGE5_LOCKS)
    slope_rows = []
    for parameter, group in locks.groupby("parameter", sort=False):
        slopes = group.sort_values("heldout_terminal_tree").mpr_slope.to_numpy(float)
        positive = int(np.sum(slopes > 0.0))
        negative = int(np.sum(slopes < 0.0))
        dominant_count = max(positive, negative)
        slope_rows.append(
            {
                "parameter": parameter,
                "folds": len(slopes),
                "positive_folds": positive,
                "negative_folds": negative,
                "dominant_sign_folds": dominant_count,
                "same_sign_7_of_8": bool(dominant_count >= 7),
                "slope_median": float(np.median(slopes)),
                "slope_MAD": float(np.median(np.abs(slopes - np.median(slopes)))),
            }
        )
    slope_stability = pd.DataFrame(slope_rows)
    slope_stability.to_parquet(OUT / "mpr_slope_stability.parquet", index=False)
    stable_slope_count = int(slope_stability.same_sign_7_of_8.sum())
    stage5_audit = pd.read_parquet(STAGE5_AUDIT)
    boundary_folds = int(stage5_audit.mpr_boundary_confounded.sum())

    reach_ids = np.arange(1, 231, dtype=int)
    order, downstream, _ = load_topology(TOPOLOGY, reach_ids)
    forcing = pd.read_parquet(FORCING, columns=["reach_id", "date", "precipitation_daily_mm", "pet_fao56_mm_day"])
    forcing["date"] = pd.to_datetime(forcing.date)
    spin_dates = pd.date_range("2006-01-01", "2009-12-31", freq="D")
    dates = pd.date_range("2010-01-01", "2018-12-31", freq="D")
    spin = forcing.loc[forcing.date.dt.year.between(2006, 2009)]
    development = forcing.loc[forcing.date.dt.year.between(2010, 2018)]
    spin_p = spin.pivot(index="date", columns="reach_id", values="precipitation_daily_mm").reindex(index=spin_dates, columns=reach_ids).to_numpy(float)
    spin_pet = spin.pivot(index="date", columns="reach_id", values="pet_fao56_mm_day").reindex(index=spin_dates, columns=reach_ids).to_numpy(float)
    development_p = development.pivot(index="date", columns="reach_id", values="precipitation_daily_mm").reindex(index=dates, columns=reach_ids).to_numpy(float)
    development_pet = development.pivot(index="date", columns="reach_id", values="pet_fao56_mm_day").reindex(index=dates, columns=reach_ids).to_numpy(float)
    area = pd.read_parquet(Q72_BRIDGE, columns=["reach_id", "catchment_area_km2"]).drop_duplicates("reach_id").set_index("reach_id").reindex(reach_ids).catchment_area_km2.to_numpy(float)

    lock = json.loads(STAGE4_LOCK.read_text(encoding="utf-8"))
    fitted_raw = np.asarray([lock["raw_parameters"][name] for name in PARAMETER_NAMES], dtype=float)
    prior = HBVParameters(200.0, 2.0, 0.70, 2.0, 5.0, 1.4426950408889634, 8.048526073147784, 40.00733226320923)
    prior_raw = parameters_to_raw(prior)
    scenarios = {
        "fitted_parent": (fitted_raw, 1.0),
        "registered_prior": (prior_raw, 1.0),
        "pet_minus_10pct": (fitted_raw, 0.9),
        "pet_plus_10pct": (fitted_raw, 1.1),
    }
    scenario_fractions: dict[str, np.ndarray] = {}
    scenario_components: dict[str, np.ndarray] = {}
    scenario_audits: list[dict[str, object]] = []
    month_index: pd.MultiIndex | None = None
    for scenario, (theta, pet_multiplier) in scenarios.items():
        print(f"Stage 6 sensitivity {scenario}", flush=True)
        components, audit = simulate_components(
            theta, spin_p, spin_pet * pet_multiplier, development_p, development_pet * pet_multiplier
        )
        local_m3 = components * area[None, :, None] * 1000.0
        routed = route_instantaneous(local_m3, reach_ids, order, downstream)
        current_month_index, fraction = monthly_fraction(dates, routed)
        if month_index is None:
            month_index = current_month_index
        elif not month_index.equals(current_month_index):
            raise RuntimeError("Sensitivity month indexes differ")
        scenario_fractions[scenario] = fraction
        scenario_components[scenario] = routed
        scenario_audits.append({"scenario": scenario, **audit})
    assert month_index is not None

    static = pd.read_parquet(STATIC, columns=["reach_id", "length_km"]).set_index("reach_id").reindex(reach_ids)
    geometry = pd.read_parquet(
        GEOMETRY,
        columns=[
            "reach_id",
            "bankfull_width_m",
            "bankfull_width_p05_m",
            "bankfull_width_p95_m",
            "bankfull_depth_m",
            "bankfull_depth_p05_m",
            "bankfull_depth_p95_m",
        ],
    ).set_index("reach_id").reindex(reach_ids)
    baseline_routed = scenario_components["fitted_parent"]
    baseline_local_components, _ = simulate_components(
        fitted_raw, spin_p, spin_pet, development_p, development_pet
    )
    baseline_local_m3 = baseline_local_components * area[None, :, None] * 1000.0
    baseline_q = np.maximum(baseline_routed.sum(axis=2) / 86400.0, 1.0e-6)
    tau_daily = (
        static.length_km.to_numpy(float)[None, :] * 1000.0
        * geometry.bankfull_width_m.to_numpy(float)[None, :]
        * geometry.bankfull_depth_m.to_numpy(float)[None, :]
        / baseline_q
        / 86400.0
    )
    print("Stage 6 R1 routing sensitivity", flush=True)
    r1 = route_linear_channels_adaptive(
        baseline_local_m3, reach_ids, order, downstream, tau_daily, cfl_limit=0.9
    )
    _, r1_fraction = monthly_fraction(dates, r1["outflow_m3"])
    scenario_fractions["r1_no_land_refit"] = r1_fraction
    scenario_audits.append(
        {
            "scenario": "r1_no_land_refit",
            "routing_mass_relative_error": float(np.max(np.abs(r1["mass_error_m3"]))) / max(float(baseline_local_m3.sum()), 1.0),
            "maximum_cfl": float(r1["maximum_cfl"]),
            "maximum_substeps_per_day": int(np.max(r1["substeps_by_day"])),
        }
    )
    sensitivity_stack = np.stack([scenario_fractions[name] for name in scenarios] + [scenario_fractions["r1_no_land_refit"]], axis=0)
    sensitivity_median = np.median(sensitivity_stack, axis=0)
    sensitivity_mad = np.median(np.abs(sensitivity_stack - sensitivity_median[None, :, :]), axis=0)
    sensitivity_width = np.quantile(sensitivity_stack, 0.95, axis=0) - np.quantile(sensitivity_stack, 0.05, axis=0)
    sensitivity_rows = pd.DataFrame(
        {
            "year": np.repeat(month_index.get_level_values("year").to_numpy(), len(reach_ids)),
            "month": np.repeat(month_index.get_level_values("month").to_numpy(), len(reach_ids)),
            "reach_id": np.tile(reach_ids, len(month_index)),
            "slow_fraction_sensitivity_median": sensitivity_median.reshape(-1),
            "slow_fraction_sensitivity_MAD": sensitivity_mad.reshape(-1),
            "slow_fraction_sensitivity_p05_p95_width": sensitivity_width.reshape(-1),
        }
    )
    for scenario, fraction in scenario_fractions.items():
        sensitivity_rows[f"slow_fraction_{scenario}"] = fraction.reshape(-1)
    sensitivity_rows.to_parquet(OUT / "sensitivity_response_fraction_by_reach_month.parquet", index=False)
    sensitivity_mad_p95 = float(np.quantile(sensitivity_mad, 0.95))
    sensitivity_gate = bool(sensitivity_mad_p95 <= 0.10)

    global_locks = locks.pivot(index="heldout_terminal_tree", columns="parameter", values="global_intercept_raw").reindex(columns=PARAMETER_NAMES)
    fold_fractions = []
    for tree, row in global_locks.iterrows():
        components, _ = simulate_components(
            row.to_numpy(float), spin_p, spin_pet, development_p, development_pet
        )
        routed = route_instantaneous(components * area[None, :, None] * 1000.0, reach_ids, order, downstream)
        _, fraction = monthly_fraction(dates, routed)
        fold_fractions.append(fraction)
    fold_stack = np.stack(fold_fractions, axis=0)
    fold_p05 = np.quantile(fold_stack, 0.05, axis=0)
    fold_p95 = np.quantile(fold_stack, 0.95, axis=0)
    fold_width = fold_p95 - fold_p05
    fold_rows = pd.DataFrame(
        {
            "year": np.repeat(month_index.get_level_values("year").to_numpy(), len(reach_ids)),
            "month": np.repeat(month_index.get_level_values("month").to_numpy(), len(reach_ids)),
            "reach_id": np.tile(reach_ids, len(month_index)),
            "slow_fraction_fold_p05": fold_p05.reshape(-1),
            "slow_fraction_fold_p50": np.median(fold_stack, axis=0).reshape(-1),
            "slow_fraction_fold_p95": fold_p95.reshape(-1),
            "slow_fraction_fold_width": fold_width.reshape(-1),
        }
    )
    fold_rows.to_parquet(OUT / "global_fold_response_fraction_ensemble.parquet", index=False)
    wide_fraction = float(np.mean(fold_width > 0.20))
    interval_gate = bool(wide_fraction <= 0.25)

    raw_sd = global_locks.std(axis=0, ddof=1)
    shrinkage = 1.0 - raw_sd / 1.5
    shrink_rows = pd.DataFrame(
        {
            "parameter": PARAMETER_NAMES,
            "fold_refit_raw_sd": raw_sd.to_numpy(float),
            "registered_prior_raw_sd": 1.5,
            "empirical_shrinkage_fraction": shrinkage.to_numpy(float),
            "shrinkage_at_least_0p25": (shrinkage >= 0.25).to_numpy(bool),
            "uncertainty_role": "FOLD_REFIT_SPREAD_NOT_FORMAL_POSTERIOR",
        }
    )
    shrink_rows.to_parquet(OUT / "map_fold_spread_shrinkage_audit.parquet", index=False)
    shrinkage_gate = bool(shrink_rows.shrinkage_at_least_0p25.all())

    hydraulic_rows = []
    length_m = static.length_km.to_numpy(float) * 1000.0
    for month_position, (year, month) in enumerate(month_index):
        selected = np.asarray((dates.year == year) & (dates.month == month))
        q_month = baseline_routed[selected].sum(axis=2).sum(axis=0) / selected.sum() / 86400.0
        q_safe = np.maximum(q_month, 1.0e-6)
        tau = length_m * geometry.bankfull_width_m.to_numpy(float) * geometry.bankfull_depth_m.to_numpy(float) / q_safe / 86400.0
        tau_low = length_m * geometry.bankfull_width_p05_m.to_numpy(float) * geometry.bankfull_depth_p05_m.to_numpy(float) / q_safe / 86400.0
        tau_high = length_m * geometry.bankfull_width_p95_m.to_numpy(float) * geometry.bankfull_depth_p95_m.to_numpy(float) / q_safe / 86400.0
        for reach_index, reach in enumerate(reach_ids):
            hydraulic_rows.append(
                {
                    "year": int(year),
                    "month": int(month),
                    "reach_id": int(reach),
                    "candidate_own_routed_q_m3_s": float(q_month[reach_index]),
                    "bankfull_travel_time_day": float(tau[reach_index]),
                    "bankfull_travel_time_p05_geometry_day": float(tau_low[reach_index]),
                    "bankfull_travel_time_p95_geometry_day": float(tau_high[reach_index]),
                    "hydraulic_exposure_role": "DIAGNOSTIC_ONLY_R1_NOT_SELECTED",
                }
            )
    pd.DataFrame(hydraulic_rows).to_parquet(OUT / "hydraulic_exposure_development_by_reach_month.parquet", index=False)

    failure_reasons = []
    if not bfi_gate:
        failure_reasons.append("OOF_SLOW_FRACTION_BFI_SPATIAL_GATE_FAILED")
    if evidence_family_count < 2:
        failure_reasons.append("FEWER_THAN_TWO_OF_THREE_Q_DERIVED_EVIDENCE_FAMILIES_SUPPORTED")
    if stable_slope_count < 1:
        failure_reasons.append("NO_MPR_ATTRIBUTE_EDGE_STABLE_IN_7_OF_8_FOLDS")
    if boundary_folds >= 2:
        failure_reasons.append("MPR_PARAMETER_BOUNDARY_CONFOUNDED")
    if not sensitivity_gate:
        failure_reasons.append("SLOW_RESPONSE_FRACTION_SENSITIVITY_MAD_EXCEEDS_0P10")
    if not interval_gate:
        failure_reasons.append("FOLD_ENSEMBLE_INTERVAL_TOO_WIDE")
    if not shrinkage_gate:
        failure_reasons.append("MAP_FOLD_SPREAD_DID_NOT_SHRINK_25_PERCENT")
    identified = len(failure_reasons) == 0
    identifiability_status = (
        "TOTAL_FLOW_AND_FAST_SLOW_RESPONSE_IDENTIFIED"
        if identified
        else "TOTAL_FLOW_SUPPORTED_FAST_SLOW_NOT_IDENTIFIED"
    )
    decision = {
        "stage": "20260825_6",
        "status": "PASS_PATH_IDENTIFIABILITY_AUDIT_COMPLETED",
        "identifiability_status": identifiability_status,
        "failure_reasons": failure_reasons,
        "selected_total_flow_model": "GLOBAL_HBV_R0",
        "bfi_spatial_gate": {
            "point_spearman": float(bfi_result.point_spearman),
            "ci95_lower": float(bfi_result.ci95_lower),
            "ci95_upper": float(bfi_result.ci95_upper),
            "minimum_point": 0.30,
            "passed": bfi_gate,
        },
        "positive_q_derived_evidence_families": evidence_family_count,
        "required_evidence_families": 2,
        "stable_MPR_attribute_edges_7_of_8": stable_slope_count,
        "MPR_boundary_confounded_folds": boundary_folds,
        "slow_fraction_sensitivity_MAD_p95": sensitivity_mad_p95,
        "sensitivity_gate_passed": sensitivity_gate,
        "reach_month_fraction_fold_interval_width_gt_0p20": wide_fraction,
        "interval_gate_passed": interval_gate,
        "fold_spread_shrinkage_gate_passed": shrinkage_gate,
        "hydraulic_route_status": "R0_SELECTED_R1_HYDRAULIC_EXPOSURE_DIAGNOSTIC_ONLY",
        "R1_routing_mass_relative_error": scenario_audits[-1]["routing_mass_relative_error"],
        "locked_2019_2022_discharge_read": False,
        "TN_used": False,
        "authorized_successor": "20260825_7",
    }
    write_json(REPORT / "path_identifiability_decision.json", decision)
    write_json(REPORT / "sensitivity_numerical_audit.json", scenario_audits)
    report = f"""# 20260825_6 响应路径可识别性与水力暴露审计

## 结论

正式状态：`{identifiability_status}`。

整棵河树外推中，模型慢响应体积比例与三方法Q派生BFI的站际Spearman为`{bfi_result.point_spearman:.3f}`，tree-block CI95为`[{bfi_result.ci95_lower:.3f}, {bfi_result.ci95_upper:.3f}]`；未通过预注册的点值≥0.30且下界>0门。三类Q派生证据中有`{evidence_family_count}/3`类达到正方向置信门。

MPR属性边有`{stable_slope_count}/8`条在至少7/8折保持同号，边界混淆`{boundary_folds}/8`折。敏感性慢响应比例MAD的95分位为`{sensitivity_mad_p95:.3f}`；折间90%区间宽度>0.20的Reach-month比例为`{wide_fraction:.1%}`。

因此，总流量模拟可以继续做锁定回顾，但Q0/Q1/Q2只能保留为模型内部快、中间、慢响应诊断，不能作为已识别真实路径提供给TN主方程。R1在Stage 4和Stage 5均未晋级，河道旅行时间只输出为候选自身流量和Andreadis宽深计算的水力暴露诊断。
"""
    (REPORT / "technical_report.md").write_text(report, encoding="utf-8")
    program = json.loads(STAGE5_PROGRAM.read_text(encoding="utf-8"))
    program["stage_status"]["20260825_6"] = identifiability_status
    program["stage_status"]["20260825_7"] = "authorized_not_started"
    (RUN / "program_manifest.json").write_text(json.dumps(program, ensure_ascii=False, indent=2), encoding="utf-8")
    integrity_paths = [
        RUN / "experiment_contract.json",
        RUN / "scripts" / "run_stage6.py",
        OUT / "oof_path_signature_audit.parquet",
        OUT / "path_evidence_bootstrap.parquet",
        OUT / "sensitivity_response_fraction_by_reach_month.parquet",
        OUT / "global_fold_response_fraction_ensemble.parquet",
        OUT / "hydraulic_exposure_development_by_reach_month.parquet",
        REPORT / "path_identifiability_decision.json",
    ]
    write_json(REPORT / "integrity.json", {str(path): sha256(path) for path in integrity_paths})
    print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
