from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
TEST_ROOT = ROOT / "5_Test"
RUN = Path(__file__).resolve().parents[1]
OUT = RUN / "reports" / "structure_compensation"
REFERENCE_ID = "20260727_6"
CANDIDATES = {
    "20260727_13": "nonlinear_recession",
    "20260727_15": "disconnection_threshold",
}
DESIGN_TOLERANCE = 1.0e-12
REPRODUCTION_TOLERANCE = 1.0e-9
EPS = 1.0e-12


@dataclass
class Bundle:
    run_id: str
    model: object
    frame: pd.DataFrame
    train: pd.DataFrame
    stations: list[str]
    mean: pd.Series
    std: pd.Series
    beta: np.ndarray
    eta: np.ndarray
    x: np.ndarray
    hp: dict[str, float]
    lists: dict[str, list[str]]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_hyperparameters(run_dir: Path) -> dict[str, float]:
    paths = [
        run_dir / "reports" / "workflow" / "run_manifest.csv",
        run_dir / "reports" / "run_manifest.csv",
    ]
    manifest_path = next((path for path in paths if path.exists()), None)
    if manifest_path is None:
        raise FileNotFoundError(f"run_manifest.csv under {run_dir}")
    manifest = pd.read_csv(manifest_path, encoding="utf-8-sig")
    raw = str(manifest.loc[0, "selected_hyperparameters"])
    output: dict[str, float] = {}
    for part in raw.split(";"):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        try:
            output[key.strip()] = float(value)
        except ValueError:
            continue
    required = [
        "fixed_sigma",
        "production_sigma",
        "group_sigma",
        "multistore_sigma",
        "hysteresis_sigma",
        "station_sigma",
        "slope_sigma",
        "regime_slope_sigma",
        "flow_contrast_weight",
    ]
    missing = [key for key in required if key not in output]
    if missing:
        raise KeyError(f"Missing hyperparameters for {run_dir.name}: {missing}")
    return output


def capture_lists(model) -> dict[str, list[str]]:
    names = [
        "FIXED_FEATURES",
        "PRODUCTION_FEATURES",
        "MULTISTORE_FEATURES",
        "HYSTERESIS_FEATURES",
        "RANDOM_SLOPE_FEATURES",
        "REGIME_SLOPE_FEATURES",
        "REGIME_GATES",
        "SPATIAL_GROUP_FEATURES",
        "SPATIAL_GROUP_GATES",
    ]
    return {name: list(getattr(model, name)) for name in names}


def apply_lists(model, lists: dict[str, list[str]]) -> None:
    for name, values in lists.items():
        setattr(model, name, list(values))


def load_components(run_dir: Path, tag: str) -> tuple[object, ...]:
    scripts = run_dir / "scripts"
    components = scripts / "components"
    previous_settings = sys.modules.get("experiment_settings")
    settings_path = scripts / "experiment_settings.py"
    try:
        if settings_path.exists():
            settings = load_module(
                f"experiment_settings_{tag}",
                settings_path,
            )
            sys.modules["experiment_settings"] = settings
        else:
            sys.modules.pop("experiment_settings", None)
        model = load_module(
            f"model30_{tag}",
            components / "fit_monthly_bayes_seasonal_hysteresis.py",
        )
        et = load_module(
            f"et62_{tag}",
            components / "run_et_role_experiment.py",
        )
        slow = load_module(
            f"slow63_{tag}",
            components / "run_slowflow_redundancy_experiment.py",
        )
        storage = load_module(
            f"storage64_{tag}",
            components / "run_storage_timing_experiment.py",
        )
        gate = load_module(
            f"gate65_{tag}",
            components / "run_hysteresis_gate_experiment.py",
        )
    finally:
        if previous_settings is None:
            sys.modules.pop("experiment_settings", None)
        else:
            sys.modules["experiment_settings"] = previous_settings
    return model, et, slow, storage, gate


def build_bundle(run_id: str) -> Bundle:
    run_dir = TEST_ROOT / run_id
    model, et, slow, storage, gate = load_components(
        run_dir,
        run_id.replace("_", ""),
    )
    hp = parse_hyperparameters(run_dir)
    observed = model.load_observed_panel()
    original = capture_lists(model)
    reduced = storage.base_slow_reduced_lists(slow, original)
    apply_lists(model, reduced)
    et_config = {
        "variant": "et_surplus_water_stress_state",
        "water_for_sas": "surplus",
        "water_for_production": "surplus",
        "wetness": "stress_adjusted",
        "production_demand_fraction": 0.00,
        "stress_interaction": "dimensionless_dry_stress",
        "remove_stress_features": False,
    }
    frame = et.add_hydrologic_features_et_variant(
        model,
        observed,
        hp,
        et_config,
    )
    frame = storage.recompute_state(frame, hp)
    frame = storage.recompute_sas(
        frame,
        hp,
        "current_clipped",
        "current_pre_release",
    )
    frame = gate.apply_gate_form(frame, "current_overlap")
    frame = storage.recompute_dependent_features(
        frame,
        hp,
        "clipped_delta",
    )
    frame = model.prepare_design(frame)
    frame = et.add_depth_features(frame)
    frame = slow.add_slow_indices(frame)
    frame = frame.sort_values(["q_site", "year", "month"]).reset_index(
        drop=True
    )
    train = frame.loc[frame["year"].le(2018)].copy().reset_index(drop=True)
    stations = sorted(frame["q_site"].astype(str).unique())
    mean, std = model.standardize_fit(train)
    beta = model.fit_map_ridge(
        train,
        stations,
        mean,
        std,
        fixed_sigma=hp["fixed_sigma"],
        production_sigma=hp["production_sigma"],
        group_sigma=hp["group_sigma"],
        multistore_sigma=hp["multistore_sigma"],
        hysteresis_sigma=hp["hysteresis_sigma"],
        station_sigma=hp["station_sigma"],
        slope_sigma=hp["slope_sigma"],
        regime_slope_sigma=hp["regime_slope_sigma"],
        anomaly_weight=0.0,
        flow_contrast_weight=hp["flow_contrast_weight"],
    )
    x, _ = model.build_matrix(frame, stations, mean, std)
    eta = x @ beta
    return Bundle(
        run_id=run_id,
        model=model,
        frame=frame,
        train=train,
        stations=stations,
        mean=mean,
        std=std,
        beta=beta,
        eta=eta,
        x=x,
        hp=hp,
        lists=reduced,
    )


def matrix_column_names(model, stations: list[str]) -> list[str]:
    names = ["intercept", *model.FIXED_FEATURES]
    names.extend(
        f"spatial_group_slope::{gate}::{feature}"
        for gate in model.SPATIAL_GROUP_GATES
        for feature in model.SPATIAL_GROUP_FEATURES
    )
    names.extend(f"station_intercept::{station}" for station in stations)
    names.extend(
        f"station_slope::{feature}::{station}"
        for feature in model.RANDOM_SLOPE_FEATURES
        for station in stations
    )
    names.extend(
        f"regime_slope::{gate}::{feature}::{station}"
        for gate in model.REGIME_GATES
        for feature in model.REGIME_SLOPE_FEATURES
        for station in stations
    )
    return names


def objective_matrix(
    model,
    frame: pd.DataFrame,
    stations: list[str],
    mean: pd.Series,
    std: pd.Series,
    flow_contrast_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    frame = frame.reset_index(drop=True)
    x, y = model.build_matrix(frame, stations, mean, std)
    x_parts = [x]
    y_parts = [y]
    if flow_contrast_weight > 0:
        sqrt_weight = float(np.sqrt(flow_contrast_weight))
        contrast_x: list[np.ndarray] = []
        contrast_y: list[float] = []
        for positions in frame.groupby("q_site", sort=False).indices.values():
            positions = np.asarray(positions, dtype=int)
            if len(positions) < 36:
                continue
            local_x = x[positions]
            local_y = y[positions]
            local_q = frame.iloc[positions]["Q_obsv_cfs"].to_numpy(
                dtype=float
            )
            q25, q50, q75, q90 = np.nanquantile(
                local_q,
                [0.25, 0.50, 0.75, 0.90],
            )
            low = local_q <= q25
            high = local_q >= q75
            middle = (local_q >= q25) & (local_q <= q75)
            peak = local_q >= q90
            if low.sum() >= 6 and high.sum() >= 6:
                contrast_x.append(
                    (
                        local_x[high].mean(axis=0)
                        - local_x[low].mean(axis=0)
                    )
                    * sqrt_weight
                )
                contrast_y.append(
                    float(
                        (local_y[high].mean() - local_y[low].mean())
                        * sqrt_weight
                    )
                )
            if peak.sum() >= 3 and middle.sum() >= 12:
                contrast_x.append(
                    (
                        local_x[peak].mean(axis=0)
                        - local_x[middle].mean(axis=0)
                    )
                    * sqrt_weight
                )
                contrast_y.append(
                    float(
                        (local_y[peak].mean() - local_y[middle].mean())
                        * sqrt_weight
                    )
                )
        if contrast_x:
            x_parts.append(np.vstack(contrast_x))
            y_parts.append(np.asarray(contrast_y, dtype=float))
    return np.vstack(x_parts), np.concatenate(y_parts)


def penalty_vector(
    model,
    stations: list[str],
    hp: dict[str, float],
) -> np.ndarray:
    n_fixed_base = 1 + len(model.FIXED_FEATURES)
    n_group = len(model.SPATIAL_GROUP_GATES) * len(
        model.SPATIAL_GROUP_FEATURES
    )
    n_fixed = n_fixed_base + n_group
    n_station = len(stations)
    n_slope = len(model.RANDOM_SLOPE_FEATURES) * n_station
    n_regime = (
        len(model.REGIME_GATES)
        * len(model.REGIME_SLOPE_FEATURES)
        * n_station
    )
    penalty = np.zeros(
        n_fixed + n_station + n_slope + n_regime,
        dtype=float,
    )
    penalty[1:n_fixed_base] = 1.0 / max(hp["fixed_sigma"], EPS)
    for index, feature in enumerate(model.FIXED_FEATURES, start=1):
        if feature in model.PRODUCTION_FEATURES:
            penalty[index] = 1.0 / max(
                hp["production_sigma"],
                EPS,
            )
        if feature in model.MULTISTORE_FEATURES:
            penalty[index] = 1.0 / max(
                hp["multistore_sigma"],
                EPS,
            )
        if feature in model.HYSTERESIS_FEATURES:
            penalty[index] = 1.0 / max(
                hp["hysteresis_sigma"],
                EPS,
            )
    penalty[n_fixed_base:n_fixed] = 1.0 / max(
        hp["group_sigma"],
        EPS,
    )
    penalty[n_fixed : n_fixed + n_station] = 1.0 / max(
        hp["station_sigma"],
        EPS,
    )
    penalty[
        n_fixed + n_station : n_fixed + n_station + n_slope
    ] = 1.0 / max(hp["slope_sigma"], EPS)
    penalty[n_fixed + n_station + n_slope :] = 1.0 / max(
        hp["regime_slope_sigma"],
        EPS,
    )
    return penalty


def partial_refit(
    baseline: Bundle,
    candidate_frame: pd.DataFrame,
    changed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    train = candidate_frame.loc[
        candidate_frame["year"].le(2018)
    ].copy().reset_index(drop=True)
    x_objective, y_objective = objective_matrix(
        baseline.model,
        train,
        baseline.stations,
        baseline.mean,
        baseline.std,
        baseline.hp["flow_contrast_weight"],
    )
    unchanged = ~changed
    residual = y_objective - (
        x_objective[:, unchanged] @ baseline.beta[unchanged]
    )
    penalty = penalty_vector(
        baseline.model,
        baseline.stations,
        baseline.hp,
    )
    x_changed = x_objective[:, changed]
    p_changed = penalty[changed]
    x_augmented = np.vstack(
        [x_changed, np.diag(p_changed)]
    )
    y_augmented = np.concatenate(
        [residual, np.zeros(len(p_changed), dtype=float)]
    )
    fitted_changed, *_ = np.linalg.lstsq(
        x_augmented,
        y_augmented,
        rcond=None,
    )
    beta = baseline.beta.copy()
    beta[changed] = fitted_changed
    x_all, _ = baseline.model.build_matrix(
        candidate_frame,
        baseline.stations,
        baseline.mean,
        baseline.std,
    )
    return beta, x_all @ beta


def saved_eta(run_id: str) -> pd.DataFrame:
    path = (
        TEST_ROOT
        / run_id
        / "reports"
        / "intermediate"
        / "base_regression"
        / "smearing_predictions_long.csv"
    )
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame = frame.loc[
        frame["variant"].eq("median_exp_eta"),
        ["q_site", "year", "month", "eta_log"],
    ]
    return frame.sort_values(
        ["q_site", "year", "month"]
    ).reset_index(drop=True)


def check_saved_eta(bundle: Bundle) -> dict[str, object]:
    saved = saved_eta(bundle.run_id)
    keys = bundle.frame[["q_site", "year", "month"]].reset_index(drop=True)
    key_match = keys.astype(str).equals(
        saved[["q_site", "year", "month"]].astype(str)
    )
    difference = (
        float(
            np.max(
                np.abs(
                    bundle.eta
                    - saved["eta_log"].to_numpy(dtype=float)
                )
            )
        )
        if key_match
        else float("inf")
    )
    return {
        "run_id": bundle.run_id,
        "rows": int(len(saved)),
        "key_match": bool(key_match),
        "max_abs_eta_difference": difference,
        "passed": bool(
            key_match and difference <= REPRODUCTION_TOLERANCE
        ),
    }


def classify(leverage: float, compensation_index: float) -> str:
    if leverage < 0.01:
        return "weak_structural_leverage"
    if compensation_index >= 0.80:
        return "compensation_dominated"
    if compensation_index <= 0.20:
        return "structural_effect_retained"
    return "mixed_or_uncertain"


def period_masks(year: pd.Series) -> dict[str, np.ndarray]:
    values = year.to_numpy(dtype=int)
    return {
        "fit_2006_2015": (values >= 2006) & (values <= 2015),
        "selection_2016_2018": (values >= 2016) & (values <= 2018),
        "development_2006_2018": (values >= 2006) & (values <= 2018),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    baseline = build_bundle(REFERENCE_ID)
    verification_rows = [check_saved_eta(baseline)]
    column_names = matrix_column_names(
        baseline.model,
        baseline.stations,
    )
    if len(column_names) != baseline.x.shape[1]:
        raise RuntimeError(
            f"Column-name count {len(column_names)} != "
            f"matrix width {baseline.x.shape[1]}"
        )

    summary_rows: list[dict[str, object]] = []
    changed_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    integrity_rows: list[dict[str, object]] = []

    for candidate_id, candidate_name in CANDIDATES.items():
        candidate = build_bundle(candidate_id)
        verification_rows.append(check_saved_eta(candidate))
        key_columns = ["q_site", "year", "month"]
        key_match = baseline.frame[key_columns].astype(str).equals(
            candidate.frame[key_columns].astype(str)
        )
        lists_match = all(
            baseline.lists[name] == candidate.lists[name]
            for name in baseline.lists
        )
        stations_match = baseline.stations == candidate.stations
        if not key_match or not lists_match or not stations_match:
            raise RuntimeError(
                f"Incompatible candidate {candidate_id}: "
                f"keys={key_match}, lists={lists_match}, "
                f"stations={stations_match}"
            )

        x_candidate, _ = baseline.model.build_matrix(
            candidate.frame,
            baseline.stations,
            baseline.mean,
            baseline.std,
        )
        design_difference = x_candidate - baseline.x
        max_column_difference = np.max(
            np.abs(design_difference),
            axis=0,
        )
        mean_column_difference = np.mean(
            np.abs(design_difference),
            axis=0,
        )
        changed = max_column_difference > DESIGN_TOLERANCE
        if not changed.any():
            raise RuntimeError(
                f"No changed design columns for {candidate_id}"
            )

        eta_frozen = x_candidate @ baseline.beta
        beta_partial, eta_partial = partial_refit(
            baseline,
            candidate.frame,
            changed,
        )
        eta_full = candidate.eta
        masks = period_masks(baseline.frame["year"])
        for period, mask in masks.items():
            frozen_delta = eta_frozen[mask] - baseline.eta[mask]
            partial_delta = eta_partial[mask] - baseline.eta[mask]
            full_delta = eta_full[mask] - baseline.eta[mask]
            denominator = float(np.linalg.norm(frozen_delta))
            partial_ratio = float(
                np.linalg.norm(partial_delta) / (denominator + EPS)
            )
            full_ratio = float(
                np.linalg.norm(full_delta) / (denominator + EPS)
            )
            compensation_partial = 1.0 - partial_ratio
            compensation_full = 1.0 - full_ratio
            leverage = float(np.median(np.abs(frozen_delta)))
            summary_rows.append(
                {
                    "candidate_run": candidate_id,
                    "candidate": candidate_name,
                    "period": period,
                    "rows": int(mask.sum()),
                    "stations": int(
                        baseline.frame.loc[mask, "q_site"].nunique()
                    ),
                    "changed_design_columns": int(changed.sum()),
                    "total_design_columns": int(len(changed)),
                    "structure_leverage_median_abs_delta_logq": leverage,
                    "structure_leverage_p95_abs_delta_logq": float(
                        np.quantile(np.abs(frozen_delta), 0.95)
                    ),
                    "frozen_effect_l2": denominator,
                    "partial_effect_l2": float(
                        np.linalg.norm(partial_delta)
                    ),
                    "full_effect_l2": float(
                        np.linalg.norm(full_delta)
                    ),
                    "partial_retention_ratio": partial_ratio,
                    "full_retention_ratio": full_ratio,
                    "partial_compensation_index": compensation_partial,
                    "full_compensation_index": compensation_full,
                    "partial_median_abs_delta_logq": float(
                        np.median(np.abs(partial_delta))
                    ),
                    "full_median_abs_delta_logq": float(
                        np.median(np.abs(full_delta))
                    ),
                    "classification": classify(
                        leverage,
                        compensation_full,
                    ),
                }
            )

        for index, name in enumerate(column_names):
            changed_rows.append(
                {
                    "candidate_run": candidate_id,
                    "candidate": candidate_name,
                    "column_index": index,
                    "column_name": name,
                    "changed": bool(changed[index]),
                    "max_abs_design_delta": float(
                        max_column_difference[index]
                    ),
                    "mean_abs_design_delta": float(
                        mean_column_difference[index]
                    ),
                    "baseline_coefficient": float(
                        baseline.beta[index]
                    ),
                    "partial_coefficient": float(beta_partial[index]),
                    "partial_coefficient_changed": bool(
                        abs(
                            beta_partial[index] - baseline.beta[index]
                        )
                        > 1.0e-12
                    ),
                }
            )

        prediction = baseline.frame.loc[
            baseline.frame["year"].le(2018),
            ["q_site", "comid", "year", "month", "Q_obsv_cfs"],
        ].copy()
        evidence_mask = baseline.frame["year"].le(2018).to_numpy()
        prediction["candidate_run"] = candidate_id
        prediction["candidate"] = candidate_name
        prediction["eta_baseline"] = baseline.eta[evidence_mask]
        prediction["eta_frozen_coefficients"] = eta_frozen[evidence_mask]
        prediction["eta_partial_refit"] = eta_partial[evidence_mask]
        prediction["eta_full_refit"] = eta_full[evidence_mask]
        prediction["delta_frozen_vs_baseline"] = (
            prediction["eta_frozen_coefficients"]
            - prediction["eta_baseline"]
        )
        prediction["delta_partial_vs_baseline"] = (
            prediction["eta_partial_refit"]
            - prediction["eta_baseline"]
        )
        prediction["delta_full_vs_baseline"] = (
            prediction["eta_full_refit"]
            - prediction["eta_baseline"]
        )
        prediction_frames.append(prediction)
        integrity_rows.append(
            {
                "candidate_run": candidate_id,
                "candidate": candidate_name,
                "keys_match_baseline": bool(key_match),
                "feature_lists_match_baseline": bool(lists_match),
                "stations_match_baseline": bool(stations_match),
                "matrix_shape_match": bool(
                    x_candidate.shape == baseline.x.shape
                ),
                "changed_design_columns": int(changed.sum()),
                "unchanged_coefficients_exact_in_partial": bool(
                    np.array_equal(
                        beta_partial[~changed],
                        baseline.beta[~changed],
                    )
                ),
                "partial_coefficients_finite": bool(
                    np.isfinite(beta_partial).all()
                ),
                "predictions_finite": bool(
                    np.isfinite(
                        np.column_stack(
                            [
                                eta_frozen,
                                eta_partial,
                                eta_full,
                            ]
                        )
                    ).all()
                ),
                "maximum_year_used": 2018,
            }
        )

    summary = pd.DataFrame(summary_rows)
    changed_frame = pd.DataFrame(changed_rows)
    verification = pd.DataFrame(verification_rows)
    integrity = pd.DataFrame(integrity_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    summary.to_csv(
        OUT / "compensation_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    changed_frame.to_csv(
        OUT / "changed_design_columns.csv",
        index=False,
        encoding="utf-8-sig",
    )
    verification.to_csv(
        OUT / "fit_reconstruction_verification.csv",
        index=False,
        encoding="utf-8-sig",
    )
    integrity.to_csv(
        OUT / "audit_integrity.csv",
        index=False,
        encoding="utf-8-sig",
    )
    predictions.to_csv(
        OUT / "compensation_predictions_2006_2018.csv",
        index=False,
        encoding="utf-8-sig",
    )

    development = summary.loc[
        summary["period"].eq("development_2006_2018")
    ].copy()
    lines = [
        f"# {RUN.name} Q72 Structure-Compensation Audit",
        "",
        "This is a diagnostic-only comparison.  It cannot promote `_13` "
        "or `_15`.",
        "",
        "| candidate | changed / total columns | frozen median | "
        "partial CI | full CI | classification |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in development.to_dict("records"):
        lines.append(
            f"| {row['candidate']} | "
            f"{int(row['changed_design_columns'])} / "
            f"{int(row['total_design_columns'])} | "
            f"{row['structure_leverage_median_abs_delta_logq']:.6f} | "
            f"{row['partial_compensation_index']:.6f} | "
            f"{row['full_compensation_index']:.6f} | "
            f"{row['classification']} |"
        )
    lines.extend(
        [
            "",
            "Interpretation thresholds:",
            "",
            "- frozen median absolute delta logQ below 0.01: weak "
            "structural leverage;",
            "- leverage at least 0.01 and full CI at least 0.80: "
            "compensation dominated;",
            "- full CI at most 0.20: structural effect retained;",
            "- all other cases: mixed or uncertain.",
            "",
            "Only 2006--2018 was evaluated.  No 2019--2022 result was "
            "read by this audit.",
        ]
    )
    (OUT / "structure_compensation_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
