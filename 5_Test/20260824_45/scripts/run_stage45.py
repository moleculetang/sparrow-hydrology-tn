"""Build the U3-fitted canonical L0-v2 parent and an exact source-tagged mass ledger."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test/20260824_45"
OUT = RUN / "outputs"
REPORTS = RUN / "reports"
LOCKS = RUN / "locks"
CONTRACT = RUN / "experiment_contract.json"
MANIFEST = RUN / "program_manifest.json"
PARENT_LOCK = ROOT / "5_Test/20260824_44/objective_calibration_revision/locks/objective_calibration_lock.json"
U3_PARAMETERS = ROOT / "5_Test/20260824_44/objective_calibration_revision/outputs/u3_objective_calibration_fold_parameters.parquet"
OLD_OOF = ROOT / "5_Test/20260824_28/outputs/l0_old36_temporal_oof_predictions.parquet"
STAGE41_OOF = ROOT / "5_Test/20260824_41/outputs/l0_v2_temporal_oof_predictions.parquet"
STAGE41_SCRIPTS = ROOT / "5_Test/20260824_41/scripts"
STAGE44_SCRIPTS = ROOT / "5_Test/20260824_44/scripts"

sys.path.insert(0, str(STAGE41_SCRIPTS))
sys.path.insert(0, str(STAGE44_SCRIPTS))
import l0_v2_core as l0  # noqa: E402
import run_stage41 as s41  # noqa: E402
import run_stage44 as s44  # noqa: E402

s28 = l0.s28
NU = 4.0
SITE_RIDGE = 12.0
J_REF = 119.0
SEED = 260845
GRAD_TOL = 1.0e-5
SOURCE_COLUMNS = [
    "fertilizer_kg_n", "manure_kg_n", "cropland_bnf_kg_n", "atmospheric_deposition_kg_n"
]
KEYS = ["fold_id", "station_key", "reach_id", "terminal_tree_id", "year", "month", "tn_mg_l"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")
    os.replace(tmp, path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)


class U3Objective:
    def __init__(self, model: l0.TorchL0V2, train: pd.DataFrame) -> None:
        self.model = model
        self.train = train.reset_index(drop=True).copy()
        self.stations = sorted(self.train.station_key.astype(str).unique())
        station_index = {station: index for index, station in enumerate(self.stations)}
        self.row_station = torch.tensor([station_index[str(value)] for value in self.train.station_key])
        station_weight, tree_weight = s41.data_weights(self.train)
        self.station_weight = torch.tensor(station_weight)
        self.tree_weight = torch.tensor(tree_weight)
        self.center_weight = torch.tensor(s41.center_weights(self.train, self.stations))
        self.observed = torch.tensor(np.log1p(self.train.tn_mg_l.to_numpy(float)))
        self.train_start = int(self.train.year.min())
        self.train_end = int(self.train.year.max())

    def site_effects(self, raw_site: torch.Tensor) -> torch.Tensor:
        return raw_site - torch.sum(self.center_weight * raw_site)

    def layer_loss(self, prediction: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        error = prediction - self.observed
        point = torch.log(sigma) + 0.5 * (NU + 1.0) * torch.log1p(error.square() / (NU * sigma.square()))
        return 0.90 * torch.sum(self.station_weight * point) + 0.10 * torch.sum(self.tree_weight * point)

    def loss(self, raw_process: torch.Tensor, raw_site: torch.Tensor) -> torch.Tensor:
        physical = self.model.to_physical(raw_process)
        values = dict(zip(self.model.names(), physical))
        _, population = self.model.evaluate(self.train, physical, self.train_start, self.train_end)
        effects = self.site_effects(raw_site)
        conditional = population + effects[self.row_station]
        sigma = torch.exp(values["log_sigma"])
        data = 0.90 * self.layer_loss(population, sigma) + 0.10 * self.layer_loss(conditional, sigma)
        prior = 0.5 * (
            ((values["beta_contact"] - 1.0) / 0.35) ** 2
            + (values["delta_path"] / 0.5) ** 2
            + (values["beta_low"] / 0.35) ** 2
            + (values["beta_high"] / 0.35) ** 2
        )
        site_prior = 0.10 * 0.5 * SITE_RIDGE * torch.sum(effects.square())
        return data + (prior + site_prior) / J_REF


def starts(model: l0.TorchL0V2, fold_id: str) -> list[np.ndarray]:
    table = pd.read_parquet(U3_PARAMETERS)
    anchor_row = table.loc[table.candidate.eq("U3_P90_S90") & table.fold_id.eq(fold_id)].iloc[0]
    anchor = np.asarray([anchor_row[name] for name in model.names()], dtype=float)
    initial0 = model.initial(0)
    initial1 = model.initial(1)
    return [anchor, initial0, initial1, 0.5 * (anchor + initial0), 0.5 * (anchor + initial1)]


def fit_model(model: l0.TorchL0V2, train: pd.DataFrame, fold_id: str) -> dict:
    objective = U3Objective(model, train)
    trials = []
    for variant, start in enumerate(starts(model, fold_id)):
        torch.manual_seed(SEED + variant)
        raw = torch.nn.Parameter(model.to_raw(start))
        raw_site = torch.nn.Parameter(torch.tensor(s44.site_start(len(objective.stations), variant)))
        adam = torch.optim.AdamW([raw, raw_site], lr=0.03, weight_decay=0.0)
        for _ in range(100):
            adam.zero_grad()
            value = objective.loss(raw, raw_site)
            value.backward()
            torch.nn.utils.clip_grad_norm_([raw, raw_site], 10.0)
            adam.step()
        lbfgs = torch.optim.LBFGS(
            [raw, raw_site], lr=1.0, max_iter=120, tolerance_grad=1.0e-11,
            tolerance_change=1.0e-13, line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            lbfgs.zero_grad()
            result = objective.loss(raw, raw_site)
            result.backward()
            return result

        lbfgs.step(closure)
        raw.grad = None
        raw_site.grad = None
        final = objective.loss(raw, raw_site)
        final.backward()
        kkt = s44.projected_kkt(model, raw, raw.grad, raw_site.grad)
        with torch.no_grad():
            physical = model.to_physical(raw).detach().numpy()
            effects = objective.site_effects(raw_site).detach().numpy()
            score = float(final.detach())
        trials.append({
            "variant": variant,
            "objective": score,
            "finite": bool(np.isfinite(score) and np.isfinite(physical).all() and np.isfinite(effects).all()),
            "physical": physical,
            "effects": effects,
            "stations": objective.stations.copy(),
            "raw_site": raw_site.detach().numpy().copy(),
            "kkt": kkt,
        })
        del raw, raw_site, adam, lbfgs
        gc.collect()
    finite = [trial for trial in trials if trial["finite"]]
    if len(finite) != 5:
        raise RuntimeError(f"Only {len(finite)}/5 finite starts in {fold_id}")
    best = min(finite, key=lambda item: item["objective"])
    best["all_starts"] = [
        {"variant": item["variant"], "objective": item["objective"], "finite": item["finite"], "kkt": item["kkt"]}
        for item in trials
    ]
    best["objective_spread"] = float(max(item["objective"] for item in finite) - min(item["objective"] for item in finite))
    return best


def source_tagged_ledger(model: l0.TorchL0V2, physical: np.ndarray, fold_id: str):
    source = (
        pd.read_parquet(l0.SOURCE).loc[lambda x: x.calendar_scenario.eq("CENTRAL")]
        .sort_values(["year", "month", "reach_id"]).reset_index(drop=True)
    )
    tag_input = np.stack([
        source[column].to_numpy(np.float64).reshape(768, 230) for column in SOURCE_COLUMNS
    ])
    crop_demand = source.crop_demand_kg_n.to_numpy(np.float64).reshape(768, 230)
    with torch.no_grad():
        values = dict(zip(model.names(), torch.tensor(physical)))
        coefficient_t = model._daily_monthly_coefficients(values)
        aggregate_fast_t, aggregate_slow_t, aggregate_state = model.local_fluxes(values, diagnostics=True)
    coefficient = {name: value.detach().numpy() for name, value in coefficient_t.items() if name in {
        "fast", "slow_from_upper", "lower_from_upper", "other", "upper_carry", "lower_carry"
    }}
    mineral = np.zeros((len(SOURCE_COLUMNS), 230), dtype=np.float64)
    lower = np.zeros_like(mineral)
    cumulative = {name: np.zeros(len(SOURCE_COLUMNS), dtype=np.float64) for name in ["input", "crop", "fast", "slow", "other"]}
    formal_cumulative = {name: np.zeros(len(SOURCE_COLUMNS), dtype=np.float64) for name in cumulative}
    formal_initial_mineral = None
    formal_initial_lower = None
    monthly = {name: [] for name in ["input", "crop", "fast", "slow", "other", "mineral_end", "lower_end"]}
    for index in range(768):
        if index == model.formal_start:
            formal_initial_mineral = mineral.sum(axis=1).copy()
            formal_initial_lower = lower.sum(axis=1).copy()
        incoming = tag_input[:, index, :]
        pre = mineral + incoming
        total_pre = pre.sum(axis=0)
        crop_total = np.minimum(total_pre, crop_demand[index])
        crop_fraction = np.divide(crop_total, total_pre, out=np.zeros_like(total_pre), where=total_pre > l0.EPS)
        crop = pre * crop_fraction[None, :]
        after_crop = pre - crop
        fast = after_crop * coefficient["fast"][index][None, :]
        slow = (
            lower * (1.0 - coefficient["lower_carry"][index][None, :])
            + after_crop * coefficient["slow_from_upper"][index][None, :]
        )
        lower = (
            lower * coefficient["lower_carry"][index][None, :]
            + after_crop * coefficient["lower_from_upper"][index][None, :]
        )
        other = after_crop * coefficient["other"][index][None, :]
        mineral = after_crop * coefficient["upper_carry"][index][None, :]
        values_now = {"input": incoming, "crop": crop, "fast": fast, "slow": slow, "other": other}
        for name, value in values_now.items():
            cumulative[name] += value.sum(axis=1)
            if index >= model.formal_start:
                formal_cumulative[name] += value.sum(axis=1)
        if index >= model.formal_start:
            for name, value in {**values_now, "mineral_end": mineral, "lower_end": lower}.items():
                monthly[name].append(value.copy())
    assert formal_initial_mineral is not None and formal_initial_lower is not None
    monthly_arrays = {name: np.stack(values, axis=1) for name, values in monthly.items()}
    fast_reference = aggregate_fast_t.detach().numpy()
    slow_reference = aggregate_slow_t.detach().numpy()
    reference = {
        "input": model.input[model.formal_start :].detach().numpy(),
        "crop": aggregate_state["crop"].detach().numpy(),
        "fast": fast_reference,
        "slow": slow_reference,
        "other": aggregate_state["other_loss"].detach().numpy(),
        "mineral_end": aggregate_state["mineral_end"].detach().numpy(),
        "lower_end": aggregate_state["lower_end"].detach().numpy(),
    }
    mismatch = {}
    for name, aggregate in reference.items():
        reconstructed = monthly_arrays[name].sum(axis=0)
        scale = np.maximum(np.abs(aggregate), 1.0)
        mismatch[name] = float(np.max(np.abs(reconstructed - aggregate) / scale))

    periods = []
    for tag_index, tag in enumerate(SOURCE_COLUMNS):
        for period, initial_mineral, initial_lower, totals in [
            ("1961_2024", 0.0, 0.0, cumulative),
            ("2010_2024", formal_initial_mineral[tag_index], formal_initial_lower[tag_index], formal_cumulative),
        ]:
            terminal_mineral = float(mineral[tag_index].sum())
            terminal_lower = float(lower[tag_index].sum())
            input_mass = float(totals["input"][tag_index])
            outputs = sum(float(totals[name][tag_index]) for name in ["crop", "fast", "slow", "other"])
            available = float(initial_mineral + initial_lower + input_mass)
            closure = available - outputs - terminal_mineral - terminal_lower
            periods.append({
                "fold_id": fold_id, "source_tag": tag, "period": period,
                "initial_mineral_kg_n": float(initial_mineral), "initial_lower_kg_n": float(initial_lower),
                "input_kg_n": input_mass, "crop_kg_n": float(totals["crop"][tag_index]),
                "fast_export_kg_n": float(totals["fast"][tag_index]),
                "slow_export_kg_n": float(totals["slow"][tag_index]),
                "other_loss_kg_n": float(totals["other"][tag_index]),
                "terminal_mineral_kg_n": terminal_mineral, "terminal_lower_kg_n": terminal_lower,
                "closure_kg_n": closure,
                "closure_relative": abs(closure) / max(available, 1.0),
            })

    years = np.asarray([year for year, _ in model.formal_keys], dtype=np.int16)
    months = np.asarray([month for _, month in model.formal_keys], dtype=np.int8)
    frames = []
    for tag_index, tag in enumerate(SOURCE_COLUMNS):
        frame = pd.DataFrame({
            "fold_id": fold_id,
            "source_tag": tag,
            "year": np.repeat(years, 230),
            "month": np.repeat(months, 230),
            "reach_id": np.tile(np.arange(1, 231, dtype=np.int16), len(years)),
        })
        for name, array in monthly_arrays.items():
            frame[f"{name}_kg_n"] = array[tag_index].reshape(-1)
        frames.append(frame)
    checks = {
        "maximum_monthly_aggregate_relative_mismatch": float(max(mismatch.values())),
        "monthly_mismatch_by_term": mismatch,
        "maximum_tag_period_closure_relative": float(max(row["closure_relative"] for row in periods)),
        "input_aggregate_max_abs_kg_n": float(np.max(np.abs(tag_input.sum(axis=0) - model.input.detach().numpy()))),
    }
    return pd.concat(frames, ignore_index=True), pd.DataFrame(periods), checks


def macro_rmse(frame: pd.DataFrame, block: str) -> float:
    return float(np.mean([
        np.sqrt(np.mean((np.log1p(group.pred_tn_mg_l) - np.log1p(group.tn_mg_l)) ** 2))
        for _, group in frame.groupby(block)
    ]))


def main() -> None:
    s28.require_runtime()
    started = time.perf_counter()
    for path in [OUT, REPORTS, LOCKS]:
        path.mkdir(parents=True, exist_ok=True)
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    parent_lock = json.loads(PARENT_LOCK.read_text(encoding="utf-8"))
    if contract.get("status") != "REGISTERED_BEFORE_RESULTS" or manifest.get("status") != "REGISTERED_BEFORE_RESULTS":
        raise RuntimeError("Stage45 not registered before results")
    if parent_lock.get("status") != "PASS_OBJECTIVE_CALIBRATION_READY_FOR_STAGE45" or parent_lock.get("selected_objective") != "U3_P90_S90":
        raise RuntimeError("Stage44 revision did not authorize Stage45")

    observations = s28.s19.build_observations()
    folds = s28.s19.build_folds(observations, "temporal")
    model = l0.TorchL0V2(regionalized=False)
    predictions = []
    parameters = []
    sites = []
    structural_rows = []
    monthly_ledgers = []
    ledger_summaries = []
    ledger_checks = {}
    invariant_max = 0.0

    for _, fold in folds.iterrows():
        fold_id = str(fold.fold_id)
        model._obs_index_cache.clear()
        model._q_feature_cache.clear()
        train, test = s28.s19.fold_frames(observations, fold)
        fit = fit_model(model, train, fold_id)
        effects = dict(zip(fit["stations"], map(float, fit["effects"])))
        layers = s41.prediction_layers(model, test, fit["physical"], int(fold.train_start_year), int(fold.train_end_year), effects)
        with torch.no_grad():
            raw_a, pop_a = model.evaluate(test, torch.tensor(fit["physical"]), int(fold.train_start_year), int(fold.train_end_year))
            raw_b, pop_b = model.evaluate(test, torch.tensor(fit["physical"]), int(fold.train_start_year), int(fold.train_end_year))
        invariant_max = max(invariant_max, float(torch.max(torch.abs(raw_a - raw_b))), float(torch.max(torch.abs(pop_a - pop_b))))
        base = test[["station_key", "reach_id", "terminal_tree_id", "year", "month", "tn_mg_l"]].copy()
        for layer in ["raw_mass_process", "population_transferable", "gauged_conditional"]:
            frame = base.copy()
            log_prediction = layers[layer]
            frame["pred_tn_mg_l"] = np.where(np.isfinite(log_prediction), np.maximum(np.expm1(log_prediction), 0.0), np.nan)
            frame["conditional_available"] = layers["known"] if layer == "gauged_conditional" else True
            frame["fold_id"] = fold_id
            frame["layer"] = layer
            frame["candidate"] = "L0_v2_U3_R0"
            predictions.append(frame)
        parameter = {
            "candidate": "L0_v2_U3_R0", "fold_id": fold_id, "objective": fit["objective"],
            "objective_spread": fit["objective_spread"], "selected_start": fit["variant"],
            "projected_kkt_max": fit["kkt"]["combined_max"],
            "process_projected_kkt_max": fit["kkt"]["process_projected_kkt_max"],
            "site_gradient_max": fit["kkt"]["site_gradient_max"],
            "all_starts_json": json.dumps(fit["all_starts"], default=json_default),
            "train_rows": len(train), "train_stations": train.station_key.nunique(),
        }
        parameter.update(dict(zip(model.names(), map(float, fit["physical"]))))
        parameters.append(parameter)
        center = s41.center_weights(train, fit["stations"])
        station_tree = train[["station_key", "terminal_tree_id"]].drop_duplicates().set_index("station_key")
        counts = train.groupby("station_key").size()
        for station, effect, raw_value, weight in zip(fit["stations"], fit["effects"], fit["raw_site"], center):
            sites.append({
                "candidate": "L0_v2_U3_R0", "fold_id": fold_id, "station_key": station,
                "terminal_tree_id": station_tree.loc[station, "terminal_tree_id"],
                "training_observations": int(counts.loc[station]), "centering_weight": float(weight),
                "b_raw_log_unit": float(raw_value), "b_station_log_unit": float(effect),
            })
        with torch.no_grad():
            structural_rows.append({"candidate": "L0_v2_U3_R0", "fold_id": fold_id, **model.structural_diagnostics(torch.tensor(fit["physical"]))})
        monthly, summary, checks = source_tagged_ledger(model, fit["physical"], fold_id)
        monthly_ledgers.append(monthly)
        ledger_summaries.append(summary)
        ledger_checks[fold_id] = checks
        current, peak = s28.memory_gib()
        print(json.dumps({"fold": fold_id, "objective": fit["objective"], "kkt": fit["kkt"]["combined_max"], "ledger_mismatch": checks["maximum_monthly_aggregate_relative_mismatch"], "rss_gib": current, "peak_gib": peak}), flush=True)

    prediction = pd.concat(predictions, ignore_index=True)
    parameter_frame = pd.DataFrame(parameters)
    site_frame = pd.DataFrame(sites)
    structural = pd.DataFrame(structural_rows)
    monthly_ledger = pd.concat(monthly_ledgers, ignore_index=True)
    ledger_summary = pd.concat(ledger_summaries, ignore_index=True)
    metrics = []
    for layer, group in prediction.loc[prediction.pred_tn_mg_l.notna()].groupby("layer"):
        metrics.append({
            "candidate": "L0_v2_U3_R0", "layer": layer,
            "station_macro_log_rmse": macro_rmse(group, "station_key"),
            "tree_macro_log_rmse": macro_rmse(group, "terminal_tree_id"),
            "rows": len(group), "stations": group.station_key.nunique(), "trees": group.terminal_tree_id.nunique(),
        })
    metrics_frame = pd.DataFrame(metrics)

    old = pd.read_parquet(OLD_OOF).loc[lambda x: x.candidate.eq("L0")]
    old_p1 = s44.reference_layer(old, "P1")
    old_p2 = s44.reference_layer(old, "P2")
    pop = s44.reference_layer(prediction, "population_transferable")
    cond = s44.reference_layer(prediction, "gauged_conditional")
    comparisons = {}
    for label, candidate_frame, reference in [("population_vs_Stage32_P1", pop, old_p1), ("conditional_vs_Stage32_P2", cond, old_p2)]:
        for index, block in enumerate(["station_key", "terminal_tree_id"]):
            comparisons[f"{label}_{block}"] = s44.paired_block_ci(candidate_frame, reference, block, SEED + index)
    old_stage41 = pd.read_parquet(STAGE41_OOF).loc[lambda x: x.candidate.eq("L0_v2_R0")]
    comparisons["population_vs_Stage41_R0_station_key"] = s44.paired_block_ci(
        pop, s44.reference_layer(old_stage41, "population_transferable"), "station_key", SEED
    )

    centered = site_frame.assign(z=site_frame.centering_weight * site_frame.b_station_log_unit).groupby("fold_id").z.sum().abs().max()
    current, peak = s28.memory_gib()
    checks = {
        "three_temporal_folds": parameter_frame.fold_id.nunique() == 3,
        "five_finite_starts_per_fold": bool(parameter_frame.all_starts_json.map(lambda value: len(json.loads(value)) == 5).all()),
        "selected_kkt_le_1e_5": bool((parameter_frame.projected_kkt_max <= GRAD_TOL).all()),
        "zero_1961_state_exact": bool((structural.initial_mineral_1961_kg_n == 0).all() and (structural.initial_lower_1961_kg_n == 0).all()),
        "operator_closure_le_1e_10": float(structural.operator_closure_max_abs.max()) <= 1.0e-10,
        "probabilities_valid": bool(structural.daily_probability_min.min() >= -1.0e-12 and structural.daily_probability_max.max() <= 1.0 + 1.0e-12),
        "zero_contact_gate_exact": float(structural.zero_contact_probability_max.max()) <= 1.0e-12,
        "land_closure_le_1e_9": float(structural.land_mass_closure_relative.max()) <= 1.0e-9,
        "channel_closure_le_1e_9": float(structural.channel_closure_relative.max()) <= 1.0e-9,
        "source_tag_period_closure_le_1e_9": max(item["maximum_tag_period_closure_relative"] for item in ledger_checks.values()) <= 1.0e-9,
        "source_tag_monthly_aggregate_le_1e_9": max(item["maximum_monthly_aggregate_relative_mismatch"] for item in ledger_checks.values()) <= 1.0e-9,
        "source_inputs_sum_exact_le_1e_9_kg": max(item["input_aggregate_max_abs_kg_n"] for item in ledger_checks.values()) <= 1.0e-9,
        "site_effects_centered_le_1e_10": float(centered) <= 1.0e-10,
        "site_head_forward_invariant_le_1e_12": invariant_max <= 1.0e-12,
        "prediction_keys_unique": not prediction.duplicated(["candidate", "layer"] + KEYS[:-1]).any(),
        "rss_below_12_gib": peak < 12.0,
    }
    engineering_pass = bool(all(checks.values()))
    status = "PASS_CANONICAL_L0_V2_SOURCE_LEDGER_READY_FOR_STAGE46" if engineering_pass else "FAIL_STAGE45_ENGINEERING"

    output_paths = {
        "predictions": OUT / "l0_v2_u3_temporal_oof_predictions.parquet",
        "parameters": OUT / "l0_v2_u3_fold_parameters.parquet",
        "site_effects": OUT / "l0_v2_u3_site_effects.parquet",
        "structural": OUT / "l0_v2_u3_structural_audit.parquet",
        "source_monthly": OUT / "source_tagged_local_fluxes_2010_2024.parquet",
        "source_summary": OUT / "source_tagged_mass_ledger.parquet",
        "metrics": OUT / "l0_v2_u3_metrics.parquet",
    }
    for key, frame in {
        "predictions": prediction, "parameters": parameter_frame, "site_effects": site_frame,
        "structural": structural, "source_monthly": monthly_ledger, "source_summary": ledger_summary,
        "metrics": metrics_frame,
    }.items():
        atomic_parquet(frame, output_paths[key])
    part_files = [str(path) for path in RUN.rglob("*.part")]
    checks["no_part_files_after_write"] = len(part_files) == 0
    if part_files:
        engineering_pass = False
        status = "FAIL_STAGE45_ENGINEERING"

    decision = {
        "stage": "20260824_45", "status": status,
        "scientific_parent": "L0_v2_U3_R0" if engineering_pass else None,
        "production_reference": "20260824_32",
        "objective": "U3_P90_S90",
        "engineering_pass": engineering_pass, "checks": checks,
        "ledger_checks": ledger_checks, "metrics": metrics,
        "descriptive_comparisons": comparisons,
        "claim_boundary": "Source tags are exact modeled bookkeeping under proportional crop allocation, not observed molecular provenance.",
        "runtime": {"python": sys.executable, "torch": torch.__version__, "dtype": str(torch.get_default_dtype()), "threads": torch.get_num_threads(), "current_rss_gib": current, "peak_rss_gib": peak, "elapsed_seconds": time.perf_counter() - started},
        "input_hashes": {str(path): sha256(path) for path in [CONTRACT, MANIFEST, PARENT_LOCK, U3_PARAMETERS, l0.SOURCE, l0.CANONICAL_DAILY, STAGE41_SCRIPTS / "l0_v2_core.py"]},
        "output_hashes": {key: sha256(path) for key, path in output_paths.items()},
        "authorized_successor": "20260824_46" if engineering_pass else None,
    }
    atomic_json(decision, REPORTS / "stage45_decision.json")
    atomic_json({
        "stage": "20260824_45", "status": status,
        "decision_sha256": sha256(REPORTS / "stage45_decision.json"),
        "contract_sha256": sha256(CONTRACT), "manifest_sha256": sha256(MANIFEST),
        "authorized_successor": decision["authorized_successor"],
    }, LOCKS / "stage45_lock.json")
    lines = [
        "# `20260824_45` canonical L0-v2 parent", "", f"Status: `{status}`.", "",
        "L0-v2 now uses the locked U3_P90_S90 one-fit objective. Hydrology and all TN process equations are unchanged from the verified Stage41 core.", "",
        "| layer | station-macro log-RMSE | tree-macro log-RMSE |", "|---|---:|---:|",
    ]
    for row in metrics_frame.itertuples(index=False):
        lines.append(f"| {row.layer} | {row.station_macro_log_rmse:.5f} | {row.tree_macro_log_rmse:.5f} |")
    lines += [
        "", "## Source ledger", "",
        f"Maximum monthly tag-to-aggregate relative mismatch: `{max(item['maximum_monthly_aggregate_relative_mismatch'] for item in ledger_checks.values()):.3e}`.",
        f"Maximum source-period mass closure error: `{max(item['maximum_tag_period_closure_relative'] for item in ledger_checks.values()):.3e}`.",
        "", "Source tags are a deterministic proportional bookkeeping convention. They do not independently validate source identity or real molecular provenance.",
        "", "Stage32 remains production; this mass-exact L0-v2 product is the parent for Stage46 structure screening.",
    ]
    (REPORTS / "technical_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, indent=2, default=json_default), flush=True)
    if not engineering_pass:
        raise RuntimeError(status)


if __name__ == "__main__":
    main()
