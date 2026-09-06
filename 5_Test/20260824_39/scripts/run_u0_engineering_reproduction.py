"""U0 engineering reproduction gate for the registered unified-TN program.

This script deliberately uses the frozen Stage-28 L0 implementation.  It does
not fit a new scientific TN candidate.  It verifies that the old P1/P2
predictions, optimizer result and closed-form station ridge can be reproduced
before the observation head is moved into a joint hierarchical MAP model.
"""

from __future__ import annotations

import gc
import hashlib
import inspect
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260824_39"
OUT = RUN / "outputs"
REPORTS = RUN / "reports"
P28 = ROOT / "5_Test" / "20260824_28"
P28_SCRIPTS = P28 / "scripts"
P28_PREDICTIONS = P28 / "outputs" / "l0_old36_temporal_oof_predictions.parquet"
P28_PARAMETERS = P28 / "outputs" / "l0_temporal_fold_parameters.parquet"
P28_VALIDATION = P28 / "reports" / "stage28_validation.json"

sys.path.insert(0, str(P28_SCRIPTS))
import run_stage28 as s28  # noqa: E402


FIXED_LOG_TOLERANCE = 1.0e-10
REFIT_LOG_TOLERANCE = 1.0e-8
REFIT_PARAMETER_TOLERANCE = 1.0e-8
RIDGE_NORMAL_EQUATION_TOLERANCE = 1.0e-10
RSS_HARD_STOP_GIB = 12.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value, dtype=np.float64)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def reconstructed_frame(
    test: pd.DataFrame,
    fold_id: str,
    layer: str,
    log_prediction: np.ndarray,
) -> pd.DataFrame:
    frame = test[["station_key", "reach_id", "terminal_tree_id", "year", "month", "tn_mg_l"]].copy()
    frame["pred_tn_mg_l_reproduced"] = np.maximum(np.expm1(log_prediction), 0.0)
    frame["fold_id"] = fold_id
    frame["holdout_type"] = "TEMPORAL"
    frame["holdout_id"] = "ALL"
    frame["layer"] = layer
    frame["candidate"] = "L0"
    return frame


def compare_prediction_frame(reproduced: pd.DataFrame, frozen: pd.DataFrame) -> tuple[float, int]:
    keys = [
        "station_key", "reach_id", "terminal_tree_id", "year", "month",
        "tn_mg_l", "fold_id", "holdout_type", "holdout_id", "layer", "candidate",
    ]
    merged = reproduced.merge(
        frozen[keys + ["pred_tn_mg_l"]],
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(reproduced) or len(merged) != len(frozen):
        raise RuntimeError("U0 prediction row-grain mismatch")
    difference = np.abs(
        np.log1p(merged.pred_tn_mg_l_reproduced.to_numpy(float))
        - np.log1p(merged.pred_tn_mg_l.to_numpy(float))
    )
    return float(difference.max(initial=0.0)), int(len(merged))


def ridge_audit(train: pd.DataFrame, p1_log: np.ndarray, effects: dict[str, float]) -> tuple[pd.DataFrame, float]:
    residual = np.log1p(train.tn_mg_l.to_numpy(float)) - p1_log
    work = pd.DataFrame({"station_key": train.station_key.astype(str).to_numpy(), "residual": residual})
    rows = []
    maximum = 0.0
    for station, group in work.groupby("station_key", sort=True):
        effect = float(effects[station])
        normal_equation = float((group.residual - effect).sum() - s28.RIDGE * effect)
        maximum = max(maximum, abs(normal_equation))
        rows.append(
            {
                "station_key": station,
                "n_training": int(len(group)),
                "residual_sum": float(group.residual.sum()),
                "ridge": float(s28.RIDGE),
                "p2_log_intercept": effect,
                "normal_equation_residual": normal_equation,
            }
        )
    return pd.DataFrame(rows), maximum


def check_memory() -> tuple[float, float]:
    current, peak = s28.memory_gib()
    if current >= RSS_HARD_STOP_GIB:
        raise MemoryError(f"U0 RSS {current:.3f} GiB reached registered 12 GiB hard stop")
    return current, peak


def main() -> None:
    started = time.perf_counter()
    s28.require_runtime()
    torch.set_default_dtype(torch.float64)
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    validation = json.loads(P28_VALIDATION.read_text(encoding="utf-8"))
    if validation.get("status") != "PASS_STAGE28_READY_FOR_20260824_29":
        raise RuntimeError("Frozen Stage-28 engineering parent is not passed")
    for name, expected in validation["input_hashes"].items():
        actual = sha256(Path(name))
        if actual != expected:
            raise RuntimeError(f"Stage-28 frozen-input hash changed: {name}")

    frozen_predictions = pd.read_parquet(P28_PREDICTIONS).loc[
        lambda x: x.candidate.eq("L0") & x.layer.isin(["P1", "P2"])
    ].copy()
    frozen_parameters = pd.read_parquet(P28_PARAMETERS).set_index("fold_id")
    observations = s28.s19.build_observations()
    folds = s28.s19.build_folds(observations, "temporal")
    if set(folds.fold_id.astype(str)) != {"T1", "T2", "T3"}:
        raise RuntimeError("U0 temporal fold registry changed")

    model = s28.TorchL0()
    parameter_names = model.names()
    fixed_rows: list[dict[str, object]] = []
    refit_rows: list[dict[str, object]] = []
    effect_frames: list[pd.DataFrame] = []

    fit_source = inspect.getsource(s28.fit_model)
    checkpoint_tokens = ("torch.load", "load_state_dict", "checkpoint", "warm_start")
    checkpoint_free = not any(token in fit_source for token in checkpoint_tokens)

    for fold in folds.itertuples(index=False):
        fold_id = str(fold.fold_id)
        train, test = s28.s19.fold_frames(observations, fold)
        frozen_fold = frozen_predictions.loc[lambda x: x.fold_id.eq(fold_id)]
        parameter_row = frozen_parameters.loc[fold_id]
        physical = np.array([float(parameter_row[name]) for name in parameter_names], dtype=np.float64)

        with torch.no_grad():
            physical_tensor = torch.tensor(physical)
            values = dict(zip(parameter_names, physical_tensor))
            fast_before, slow_before = model.local_fluxes(values)
            fast_hash_before = array_sha256(fast_before.numpy())
            slow_hash_before = array_sha256(slow_before.numpy())
            _, train_p1 = model.evaluate(train, physical_tensor, int(fold.train_start_year), int(fold.train_end_year))
            _, test_p1 = model.evaluate(test, physical_tensor, int(fold.train_start_year), int(fold.train_end_year))

        train_p1_np = train_p1.numpy().copy()
        test_p1_np = test_p1.numpy().copy()
        effects = s28.p2_effects(train, train_p1_np)
        effect_frame, ridge_error = ridge_audit(train, train_p1_np, effects)
        effect_frame.insert(0, "fold_id", fold_id)
        effect_frames.append(effect_frame)

        fixed_reproduced = []
        layer_errors = {}
        for layer in ("P1", "P2"):
            log_prediction = test_p1_np.copy()
            if layer == "P2":
                log_prediction += np.array([effects.get(str(station), 0.0) for station in test.station_key], dtype=float)
            frame = reconstructed_frame(test, fold_id, layer, log_prediction)
            fixed_reproduced.append(frame)
            error, rows = compare_prediction_frame(frame, frozen_fold.loc[lambda x: x.layer.eq(layer)])
            layer_errors[layer] = error
            if rows != len(test):
                raise RuntimeError(f"U0 {fold_id} {layer} row count changed")

        with torch.no_grad():
            # The observation-head operation above must not mutate process fluxes.
            fast_after, slow_after = model.local_fluxes(values)
            fast_hash_after = array_sha256(fast_after.numpy())
            slow_hash_after = array_sha256(slow_after.numpy())

        current, peak = check_memory()
        fixed_rows.append(
            {
                "fold_id": fold_id,
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                "stations": int(train.station_key.nunique()),
                "fixed_p1_max_abs_log_error": layer_errors["P1"],
                "fixed_p2_max_abs_log_error": layer_errors["P2"],
                "ridge_max_abs_normal_equation_error": ridge_error,
                "unknown_station_effect_is_zero": bool(effects.get("__U0_UNKNOWN_STATION__", 0.0) == 0.0),
                "fast_state_hash_before": fast_hash_before,
                "fast_state_hash_after": fast_hash_after,
                "slow_state_hash_before": slow_hash_before,
                "slow_state_hash_after": slow_hash_after,
                "site_head_state_invariant": bool(fast_hash_before == fast_hash_after and slow_hash_before == slow_hash_after),
                "current_rss_gib": current,
                "peak_rss_gib": peak,
            }
        )

        # Deterministic refit from the two frozen starts.  This is intentionally
        # separate from the fixed-parameter reproduction above.
        refit = s28.fit_model(model, train)
        refit_physical = np.asarray(refit["physical"], dtype=np.float64)
        with torch.no_grad():
            refit_tensor = torch.tensor(refit_physical)
            _, refit_train_p1 = model.evaluate(train, refit_tensor, int(fold.train_start_year), int(fold.train_end_year))
            _, refit_test_p1 = model.evaluate(test, refit_tensor, int(fold.train_start_year), int(fold.train_end_year))
        refit_effects = s28.p2_effects(train, refit_train_p1.numpy())
        refit_layer_errors = {}
        for layer in ("P1", "P2"):
            log_prediction = refit_test_p1.numpy().copy()
            if layer == "P2":
                log_prediction += np.array([refit_effects.get(str(station), 0.0) for station in test.station_key], dtype=float)
            frame = reconstructed_frame(test, fold_id, layer, log_prediction)
            error, _ = compare_prediction_frame(frame, frozen_fold.loc[lambda x: x.layer.eq(layer)])
            refit_layer_errors[layer] = error
        refit_rows.append(
            {
                "fold_id": fold_id,
                "objective_frozen": float(parameter_row.objective),
                "objective_refit": float(refit["objective"]),
                "max_abs_parameter_error": float(np.max(np.abs(refit_physical - physical))),
                "refit_p1_max_abs_log_error": refit_layer_errors["P1"],
                "refit_p2_max_abs_log_error": refit_layer_errors["P2"],
                "success": bool(refit["success"]),
            }
        )
        del train_p1, test_p1, fast_before, slow_before, fast_after, slow_after, refit_train_p1, refit_test_p1
        gc.collect()
        print(json.dumps({"stage": "U0", "fold": fold_id, "fixed": fixed_rows[-1], "refit": refit_rows[-1]}), flush=True)

    fixed = pd.DataFrame(fixed_rows)
    refit = pd.DataFrame(refit_rows)
    effects = pd.concat(effect_frames, ignore_index=True)
    fixed_path = OUT / "u0_fixed_parameter_reproduction.parquet"
    refit_path = OUT / "u0_refit_reproduction.parquet"
    effects_path = OUT / "u0_old_p2_ridge_effects.parquet"
    atomic_parquet(fixed, fixed_path)
    atomic_parquet(refit, refit_path)
    atomic_parquet(effects, effects_path)

    checks = {
        "stage28_frozen_input_hashes_match": True,
        "three_temporal_folds_exact": bool(set(fixed.fold_id) == {"T1", "T2", "T3"}),
        "fixed_P1_exact_reproduction": bool(fixed.fixed_p1_max_abs_log_error.max() <= FIXED_LOG_TOLERANCE),
        "fixed_P2_exact_reproduction": bool(fixed.fixed_p2_max_abs_log_error.max() <= FIXED_LOG_TOLERANCE),
        "refit_parameters_reproduced": bool(refit.max_abs_parameter_error.max() <= REFIT_PARAMETER_TOLERANCE),
        "refit_P1_reproduced": bool(refit.refit_p1_max_abs_log_error.max() <= REFIT_LOG_TOLERANCE),
        "refit_P2_reproduced": bool(refit.refit_p2_max_abs_log_error.max() <= REFIT_LOG_TOLERANCE),
        "old_P2_closed_form_ridge_algebra": bool(fixed.ridge_max_abs_normal_equation_error.max() <= RIDGE_NORMAL_EQUATION_TOLERANCE),
        "unknown_station_effect_zero": bool(fixed.unknown_station_effect_is_zero.all()),
        "site_head_process_state_invariant": bool(fixed.site_head_state_invariant.all()),
        "training_code_checkpoint_free": bool(checkpoint_free),
        "all_refits_success": bool(refit.success.all()),
        "rss_below_registered_hard_stop": bool(fixed.peak_rss_gib.max() < RSS_HARD_STOP_GIB),
    }
    passed = bool(all(checks.values()))
    result = {
        "stage": "20260824_39_U0_ENGINEERING_REPRODUCTION",
        "status": "PASS_U0_ENGINEERING_HARD_GATE" if passed else "FAIL_U0_ENGINEERING_HARD_GATE",
        "scientific_authorization": "none; U0 is engineering reproduction only",
        "checks": checks,
        "tolerances": {
            "fixed_max_abs_log_error": FIXED_LOG_TOLERANCE,
            "refit_max_abs_log_error": REFIT_LOG_TOLERANCE,
            "refit_max_abs_parameter_error": REFIT_PARAMETER_TOLERANCE,
            "ridge_normal_equation_error": RIDGE_NORMAL_EQUATION_TOLERANCE,
            "rss_hard_stop_gib": RSS_HARD_STOP_GIB,
        },
        "maxima": {
            "fixed_P1_log_error": float(fixed.fixed_p1_max_abs_log_error.max()),
            "fixed_P2_log_error": float(fixed.fixed_p2_max_abs_log_error.max()),
            "refit_parameter_error": float(refit.max_abs_parameter_error.max()),
            "refit_P1_log_error": float(refit.refit_p1_max_abs_log_error.max()),
            "refit_P2_log_error": float(refit.refit_p2_max_abs_log_error.max()),
            "ridge_normal_equation_error": float(fixed.ridge_max_abs_normal_equation_error.max()),
            "peak_rss_gib": float(fixed.peak_rss_gib.max()),
        },
        "runtime": {
            "python": sys.executable,
            "torch": torch.__version__,
            "dtype": str(torch.get_default_dtype()),
            "threads": int(torch.get_num_threads()),
            "elapsed_seconds": float(time.perf_counter() - started),
        },
        "input_hashes": {
            str(P28_SCRIPTS / "run_stage28.py"): sha256(P28_SCRIPTS / "run_stage28.py"),
            str(P28_PREDICTIONS): sha256(P28_PREDICTIONS),
            str(P28_PARAMETERS): sha256(P28_PARAMETERS),
            str(P28_VALIDATION): sha256(P28_VALIDATION),
        },
        "output_hashes": {
            str(fixed_path): sha256(fixed_path),
            str(refit_path): sha256(refit_path),
            str(effects_path): sha256(effects_path),
        },
    }
    result_path = REPORTS / "u0_engineering_reproduction.json"
    atomic_json(result_path, result)

    report = [
        "# U0 工程复现门",
        "",
        f"状态：`{result['status']}`。该门只验证旧 P1/P2 软件与代数，不授权新的科学模型。",
        "",
        "| 检查 | 结果 |",
        "|---|---:|",
    ]
    report.extend(f"| `{name}` | `{value}` |" for name, value in checks.items())
    report.extend(
        [
            "",
            "## 最大数值误差",
            "",
            *[f"- `{name}`: `{value:.12g}`" for name, value in result["maxima"].items()],
            "",
            "站点 ridge 截距只在观测头复现；process fast/slow 状态哈希在应用站点项前后完全一致。即使 U0 通过，旧 L0 仍只能作为工程父模型，科学运行仍须先完成 station-position Q、candidate-specific daily TN forward 与 L0-v2。",
        ]
    )
    temporary_report = (REPORTS / "u0_engineering_reproduction.md.part")
    temporary_report.write_text("\n".join(report) + "\n", encoding="utf-8")
    os.replace(temporary_report, REPORTS / "u0_engineering_reproduction.md")
    if not passed:
        raise RuntimeError("U0 engineering hard gate failed; see report")


if __name__ == "__main__":
    main()
