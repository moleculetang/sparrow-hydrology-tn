"""SAS sparrow_main.sas replacement helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import shutil
import time
from typing import Callable, Mapping, MutableMapping

import numpy as np
import pandas as pd

from ..calibration import calibrate
from ..config import apply_testglobal_defaults, build_model_spec, write_model_summary
from ..plots import graph_resids
from ..postprocess import (
    CompilePredictState,
    compile_calibrate,
    compile_predict,
    summarize_calibrate,
    summarize_predict,
)
from ..prediction import predict
from ..preprocessing import (
    check_model_vars,
    check_network,
    check_station_vars,
    get_seeds,
    init_beta,
    make_nested_area,
    mean_adjust_delivery_vars,
    set_seeds,
    setdata,
    sort_data,
)
from ..tools.rng import rannor as default_rannor
from ..tools.rng import ranuni as default_ranuni


@dataclass
class RunResult:
    config: dict[str, object]
    model_spec: object | None
    comment_all: pd.DataFrame | None
    predict_state: CompilePredictState | None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class FileTableStore:
    """Minimal table store backed by disk with optional in-memory cache."""

    def __init__(
        self,
        base_dir: str | Path | None,
        table_paths: Mapping[str, str] | None = None,
        *,
        default_ext: str = ".csv",
    ) -> None:
        self.base_dir = Path(base_dir) if base_dir else None
        self.table_paths = dict(table_paths) if table_paths else {}
        self.default_ext = default_ext
        self._memory: dict[str, pd.DataFrame] = {}

    def exists(self, name: str) -> bool:
        if name in self._memory:
            return True
        path = self._resolve_path(name, for_write=False)
        return bool(path and path.exists())

    def read(self, name: str) -> pd.DataFrame:
        if name in self._memory:
            return self._memory[name].copy()
        path = self._resolve_path(name, for_write=False)
        if not path or not path.exists():
            raise FileNotFoundError(f"Table {name} not found.")
        df = _read_table(path)
        self._memory[name] = df.copy()
        return df

    def write(self, name: str, df: pd.DataFrame) -> None:
        self._memory[name] = df.copy()
        path = self._resolve_path(name, for_write=True)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_table(path, df)

    def delete(self, name: str) -> None:
        self._memory.pop(name, None)
        path = self._resolve_path(name, for_write=False)
        if path and path.exists():
            try:
                path.unlink()
            except PermissionError:
                pass

    def rename(self, src: str, dest: str) -> None:
        if src in self._memory:
            self._memory[dest] = self._memory.pop(src)
        src_path = self._resolve_path(src, for_write=False)
        dest_path = self._resolve_path(dest, for_write=True)
        if src_path and src_path.exists() and dest_path:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                src_path.rename(dest_path)
            except PermissionError:
                shutil.copy2(src_path, dest_path)
                try:
                    src_path.unlink()
                except Exception:
                    pass

    def _resolve_path(self, name: str, *, for_write: bool) -> Path | None:
        direct = Path(name)
        if direct.suffix or str(direct).find("\\") >= 0 or str(direct).find("/") >= 0:
            if direct.is_absolute() or direct.exists():
                return direct
            if self.base_dir:
                direct = self.base_dir / name
                if for_write or direct.exists():
                    return direct
        if name in self.table_paths:
            return Path(self.table_paths[name])
        if not self.base_dir:
            return None
        if for_write:
            return self.base_dir / f"{name}{self.default_ext}"
        return _find_existing(self.base_dir, name)


def run_main(
    config: MutableMapping[str, object],
    *,
    input_store: FileTableStore | None = None,
    results_store: FileTableStore | None = None,
    ranuni: Callable[[int, int], np.ndarray] | None = None,
    rannor: Callable[[int, int], np.ndarray] | None = None,
    basemap: pd.DataFrame | None = None,
) -> RunResult:
    """Run the SPARROW main orchestration."""
    warnings: list[str] = []
    errors: list[str] = []

    cfg = apply_testglobal_defaults(dict(config))
    if "if_error" not in cfg:
        cfg["if_error"] = "no"

    home_results = _first_path(cfg, "home_results", "results_dir")
    home_data = _first_path(cfg, "home_data", "input_dir")

    if results_store is None:
        results_store = FileTableStore(home_results, cfg.get("results_tables"))
    if input_store is None:
        input_store = FileTableStore(home_data, cfg.get("input_tables"))

    iter_val = _to_int(cfg.get("start_iter"), 0)
    jter_val = _to_int(cfg.get("start_jter"), 0)
    cfg["iter"] = iter_val
    cfg["jter"] = jter_val

    comment_all = (
        results_store.read("comment_all")
        if results_store.exists("comment_all")
        else None
    )

    def put_comment(message: str) -> None:
        nonlocal comment_all
        comment_all = _append_comment(
            results_store, comment_all, iter_val, jter_val, message
        )

    def warn(message: str) -> None:
        warnings.append(message)
        put_comment(_format_comment("WARNING", message))

    def error(message: str) -> None:
        errors.append(message)
        cfg["if_error"] = "yes"
        put_comment(_format_comment("ERROR", message))

    if ranuni is None:
        ranuni = default_ranuni
    if rannor is None:
        rannor = default_rannor

    if iter_val == 0:
        _backup_tables(results_store, ["comment_all"])

    if not home_results or not Path(home_results).is_dir():
        error(
            "Results directory not found - check directory specification - stop processing."
        )

    if _is_yes(cfg.get("if_make_input_data")):
        indata_name = _as_table_name(cfg.get("indata"))
        if not indata_name or not input_store.exists(indata_name):
            error("Data directory not found - check directory specification - stop processing.")

    home_gis = str(cfg.get("home_gis") or "")
    if home_gis and not Path(home_gis).is_dir():
        warn(
            "GIS directory is specified but was not found - "
            "analysis proceeding without GIS."
        )
        cfg["home_gis"] = ""
        cfg["if_gis"] = "no"

    if home_results and Path(home_results).is_dir():
        _initialize_run_outputs(Path(home_results))

    indata_df: pd.DataFrame | None = None
    n_obs = None
    if _is_yes(cfg.get("if_make_input_data")):
        indata_df = _read_indata(cfg, input_store)
        if indata_df is not None:
            n_obs = len(indata_df)
    elif results_store.exists("indata"):
        indata_df = results_store.read("indata")
        n_obs = len(indata_df)

    model_spec = None
    if cfg.get("if_error") != "yes":
        model_spec = build_model_spec(
            cfg, table_exists=results_store.exists, n_obs=n_obs
        )
        for msg in getattr(model_spec, "warnings", []):
            warn(msg)
        for msg in getattr(model_spec, "errors", []):
            error(msg)

    if cfg.get("if_error") != "yes":
        write_model_summary(cfg)

    seeds_df = None
    if cfg.get("if_error") != "yes":
        n_seeds = _to_int(cfg.get("n_seeds"), 0)
        if n_seeds > 0:
            def _ranuni_scalar(seed: int) -> float:
                values = ranuni(seed, 1)
                return float(values[0])

            seeds_df = set_seeds(
                _to_int(cfg.get("master_seed"), 0),
                _to_int(cfg.get("n_boot_iter"), 0),
                _to_int(cfg.get("n_extra_jter"), 0),
                n_seeds,
                ranuni=_ranuni_scalar,
            )

    indata = indata_df
    station_data = None
    ancillary = None

    if cfg.get("if_error") != "yes":
        if _is_yes(cfg.get("if_make_input_data")):
            if indata is None:
                error("Input data could not be loaded - stop processing.")
            else:
                setdata_result = setdata(indata, cfg)
                _log_messages(setdata_result.warnings, warn)
                _log_messages(setdata_result.errors, error)
                indata = setdata_result.indata
                station_data = setdata_result.station_data
                ancillary = setdata_result.ancillary

                if cfg.get("if_error") != "yes":
                    _, errs = check_model_vars(indata, cfg)
                    _log_messages(errs, error)
                    badfrac, badhydseq, net_warn = check_network(
                        setdata_result.fnodes, setdata_result.tnodes, cfg
                    )
                    _ = badfrac, badhydseq
                    _log_messages(net_warn, warn)
                    if _is_yes(cfg.get("if_estimate")):
                        _, errs = check_station_vars(station_data, cfg)
                        _log_messages(errs, error)

                if cfg.get("if_error") != "yes":
                    indata, station_data, ancillary = sort_data(
                        indata, station_data, ancillary, cfg
                    )
                    if _is_yes(cfg.get("if_estimate")):
                        station_data = make_nested_area(indata, station_data, cfg)
                    if _is_yes(cfg.get("if_mean_adjust_delivery_vars")):
                        indata, _, errs = mean_adjust_delivery_vars(
                            indata, cfg, store=results_store
                        )
                        _log_messages(errs, error)

                if cfg.get("if_error") != "yes":
                    results_store.write("indata", indata)
                    results_store.write("station_data", station_data)
                    results_store.write("ancillary", ancillary)
        else:
            missing = []
            if not results_store.exists("indata"):
                missing.append("indata")
            if not results_store.exists("ancillary"):
                missing.append("ancillary")
            if _is_yes(cfg.get("if_estimate")) and not results_store.exists("station_data"):
                missing.append("station_data")
            if missing:
                error(
                    "No input data available. Check that if_make_input_data is set to yes "
                    "- stop processing."
                )
            else:
                put_comment("Using indata from a previous run.")
                indata = results_store.read("indata")
                ancillary = results_store.read("ancillary")
                station_data = (
                    results_store.read("station_data")
                    if results_store.exists("station_data")
                    else None
                )

    if cfg.get("if_error") != "yes" and iter_val == 0:
        if _is_yes(cfg.get("if_mean_adjust_delivery_vars")) and results_store.exists(
            "mean_delivery_vars"
        ):
            mean_df = results_store.read("mean_delivery_vars")
            results_store.write("bak_mean_delivery_vars", mean_df)

        if _is_yes(cfg.get("if_estimate")):
            _backup_tables(
                results_store,
                [
                    "summary_betaest",
                    "cov_betaest",
                    "boot_betaest_all",
                    "resids",
                    "test_resids",
                    "error_report",
                ],
            )
            _backup_files(
                home_results,
                [
                    "PredictVsObserved.png",
                    "PredictVsResiduals.png",
                    "YieldVsResiduals.png",
                    "PredictVsWghtResids.png",
                    "YieldVsWghtResids.png",
                    "ProbabilityPlot.png",
                    "ResidualsMap.png",
                ],
            )
        if _is_yes(cfg.get("if_predict")):
            _backup_tables(
                results_store,
                [
                    "predict",
                    "predict_stats",
                    "test_data",
                    "test_predict",
                    "summary_predict",
                    "LU_yield_percentiles",
                    "boot_detail",
                    "upmonload",
                ],
            )

    predict_state: CompilePredictState | None = None
    if cfg.get("if_error") != "yes":
        n_boot_iter = _to_int(cfg.get("n_boot_iter"), 0)
        end_iter = _to_int(cfg.get("end_iter"), n_boot_iter)
        end_jter = _to_int(cfg.get("end_jter"), n_boot_iter)

        while iter_val <= end_iter and cfg.get("if_error") != "yes":
            cfg["iter"] = iter_val
            cfg["jter"] = jter_val
            _clear_data(iter_val, results_store)

            seed_map = (
                get_seeds(seeds_df, jter_val, _to_int(cfg.get("n_seeds"), 0))
                if seeds_df is not None
                else {}
            )
            cfg.update(seed_map)

            boot_betaest = None
            summary_betaest = None
            cov_betaest = None
            resids = None

            if _is_yes(cfg.get("if_estimate")):
                betahat0 = init_beta(
                    cfg,
                    iter_val=iter_val,
                    jter_val=jter_val,
                    temp_rc=_safe_read(results_store, "temp_rc"),
                    temp_beta=_safe_read(results_store, "temp_beta"),
                    bak_summary_betaest=_safe_read(results_store, "bak_summary_betaest"),
                    summary_betaest=_safe_read(results_store, "summary_betaest"),
                )

                cov_est = None
                if iter_val > 0 and _is_yes(cfg.get("if_parm_bootstrap")):
                    cov_df = _safe_read(results_store, "cov_betaest")
                    betalst = _split_tokens(cfg.get("betalst"))
                    if cov_df is not None and betalst:
                        cov_est = cov_df.loc[
                            :, [c for c in betalst if c in cov_df.columns]
                        ].to_numpy(dtype=float)

                calibration = calibrate(
                    indata,
                    cfg,
                    betahat0,
                    iter_val=iter_val,
                    jter_val=jter_val,
                    constraints=_load_constraints(cfg, results_store),
                    ranuni=ranuni,
                    rannor=rannor,
                    cov_estimate_input=cov_est,
                )
                _log_messages(calibration.warnings, warn)
                _log_messages(calibration.errors, error)

                boot_betaest = calibration.boot_betaest
                summary_betaest = calibration.summary_betaest
                cov_betaest = calibration.cov_betaest
                resids = calibration.resids
                if calibration.error_report is not None:
                    results_store.write("error_report", calibration.error_report)
                if calibration.test_resids is not None:
                    results_store.write("test_resids", calibration.test_resids)
                if calibration.temp_beta is not None:
                    results_store.write("temp_beta", calibration.temp_beta)
                if calibration.temp_rc is not None:
                    results_store.write("temp_rc", calibration.temp_rc)

                if boot_betaest is not None and cfg.get("if_error") != "yes":
                    results_store.write("boot_betaest", boot_betaest)
                    compile_result = compile_calibrate(
                        iter_val=iter_val,
                        boot_betaest=boot_betaest,
                        summary_betaest=summary_betaest,
                        cov_betaest=cov_betaest,
                        resids=resids,
                        station_data=station_data,
                        config=cfg,
                        boot_betaest_all=_safe_read(results_store, "boot_betaest_all"),
                    )
                    _log_messages(compile_result.warnings, warn)
                    _log_messages(compile_result.errors, error)
                    if compile_result.boot_betaest_all is not None:
                        results_store.write(
                            "boot_betaest_all", compile_result.boot_betaest_all
                        )
                    if iter_val == 0:
                        if compile_result.summary_betaest is not None:
                            results_store.write(
                                "summary_betaest", compile_result.summary_betaest
                            )
                            _append_model_results(
                                home_results,
                                "calibration",
                                compile_result.summary_betaest,
                            )
                        if compile_result.cov_betaest is not None:
                            results_store.write("cov_betaest", compile_result.cov_betaest)
                        if compile_result.resids is not None:
                            results_store.write("resids", compile_result.resids)
                            basemap_data = basemap if basemap is not None else _load_basemap(cfg)
                            graph_res = graph_resids(
                                compile_result.resids, cfg, basemap=basemap_data
                            )
                            _log_messages(graph_res.warnings, warn)
                            _log_messages(graph_res.errors, error)
                        if _is_yes(cfg.get("if_output_to_tab")) and home_results:
                            if compile_result.summary_betaest is not None:
                                _write_tab(
                                    compile_result.summary_betaest,
                                    Path(home_results) / "summary_betaest.txt",
                                )
                            if compile_result.cov_betaest is not None:
                                _write_tab(
                                    compile_result.cov_betaest,
                                    Path(home_results) / "cov_betaest.txt",
                                )
                            if compile_result.resids is not None:
                                _write_tab(
                                    compile_result.resids,
                                    Path(home_results) / "resids.txt",
                                )
                            if _is_yes(cfg.get("if_test_calibrate")) and results_store.exists(
                                "test_resids"
                            ):
                                _write_tab(
                                    results_store.read("test_resids"),
                                    Path(home_results) / "test_resids.txt",
                                )
                else:
                    warn(
                        f"Calibration for iteration {iter_val} and seed {jter_val} failed - going to next seed value"
                    )
                    cfg["if_error"] = "yes"
            else:
                if results_store.exists("boot_betaest_all"):
                    boot_all = results_store.read("boot_betaest_all")
                    match = boot_all.loc[boot_all["iter"] == iter_val]
                    if len(match) != 1:
                        error(
                            f"Calibration results for iteration {iter_val} not available - stop processing"
                        )
                        jter_val = end_jter
                    else:
                        boot_betaest = match.reset_index(drop=True)
                        jter_val = int(boot_betaest.loc[0, "jter"])
                        cfg["jter"] = jter_val
                else:
                    error(
                        "No calibration specified but previous calibration results not available - stop processing"
                    )
                    jter_val = end_jter

            if cfg.get("if_error") != "yes":
                if boot_betaest is not None:
                    if _is_yes(cfg.get("if_predict")):
                        predict_result = predict(indata, boot_betaest, cfg)
                        _log_messages(predict_result.warnings, warn)
                        _log_messages(predict_result.errors, error)
                        if predict_result.debug_trace is not None:
                            results_store.write(
                                "debug_predict_trace", predict_result.debug_trace
                            )
                        if predict_result.debug_del_frac_chain is not None:
                            results_store.write(
                                "debug_del_frac_chain", predict_result.debug_del_frac_chain
                            )
                        if (
                            predict_result.predict is not None
                            and cfg.get("if_error") != "yes"
                        ):
                            if iter_val == 0 or (
                                results_store.exists("resids")
                                and results_store.exists("summary_betaest")
                            ):
                                resids_use = (
                                    results_store.read("resids")
                                    if results_store.exists("resids")
                                    else pd.DataFrame()
                                )
                                summary_use = (
                                    results_store.read("summary_betaest")
                                    if results_store.exists("summary_betaest")
                                    else pd.DataFrame()
                                )
                                predict_state = predict_state or CompilePredictState()
                                predict_compile = compile_predict(
                                    iter_val=iter_val,
                                    jter_val=jter_val,
                                    n_boot_iter=n_boot_iter,
                                    predict=predict_result.predict,
                                    upmonload=predict_result.upmonload,
                                    test_data=predict_result.test_data,
                                    boot_betaest=boot_betaest,
                                    resids=resids_use,
                                    summary_betaest=summary_use,
                                    indata=indata,
                                    config=cfg,
                                    state=predict_state,
                                    ranuni=ranuni,
                                )
                                _log_messages(predict_compile.warnings, warn)
                                _log_messages(predict_compile.errors, error)
                                predict_state = predict_compile.state
                                if predict_compile.predict is not None:
                                    results_store.write("predict", predict_compile.predict)
                                if predict_state.predict_stats is not None:
                                    results_store.write(
                                        "predict_stats", predict_state.predict_stats
                                    )
                                if predict_state.test_data is not None:
                                    results_store.write("test_data", predict_state.test_data)
                                if predict_state.test_predict is not None:
                                    results_store.write(
                                        "test_predict", predict_state.test_predict
                                    )
                                if predict_state.model_parm_predict is not None:
                                    results_store.write(
                                        "model_parm_predict",
                                        predict_state.model_parm_predict,
                                    )
                                if predict_state.boot_detail is not None:
                                    results_store.write(
                                        "boot_detail", predict_state.boot_detail
                                    )
                                if iter_val == 0 and predict_result.upmonload is not None:
                                    results_store.write(
                                        "upmonload", predict_result.upmonload
                                    )
                            else:
                                error(
                                    "SAS files not found - no bootstrap predictions compiled, stop processing"
                                )
                                jter_val = end_jter
                        else:
                            warn(
                                f"Predictions for iteration {iter_val} and seed {jter_val} failed - going to next seed value"
                            )
                            cfg["if_error"] = "yes"
                else:
                    warn(
                        f"Calibration for iteration {iter_val} and seed {jter_val} failed - going to next seed value"
                    )
                    cfg["if_error"] = "yes"

            if cfg.get("if_error") != "yes":
                put_comment(f"Completed iteration {iter_val}")
                iter_val += 1

            jter_val += 1
            cfg["if_error"] = "no"

            if jter_val > end_jter and iter_val <= end_iter:
                error("Exhausted seeds - stop processing")

    if cfg.get("if_error") != "yes":
        n_boot_iter = _to_int(cfg.get("n_boot_iter"), 0)
        if _is_yes(cfg.get("if_estimate")) and n_boot_iter > 0 and not _is_yes(
            cfg.get("if_parm_bootstrap")
        ):
            boot_all = _safe_read(results_store, "boot_betaest_all")
            if boot_all is not None and len(boot_all) == n_boot_iter + 1:
                summary = _safe_read(results_store, "summary_betaest")
                if summary is not None:
                    summ = summarize_calibrate(
                        boot_betaest_all=boot_all, summary_betaest=summary, config=cfg
                    )
                    _log_messages(summ.warnings, warn)
                    _log_messages(summ.errors, error)
                    if summ.summary_betaest is not None:
                        results_store.write("summary_betaest", summ.summary_betaest)
                        if _is_yes(cfg.get("if_output_to_tab")) and home_results:
                            _write_tab(
                                summ.summary_betaest,
                                Path(home_results) / "summary_betaest.txt",
                            )
                    if summ.summary_boot_betaest is not None:
                        results_store.write(
                            "summary_boot_betaest", summ.summary_boot_betaest
                        )
                    if _is_yes(cfg.get("if_output_to_tab")) and home_results:
                        _write_tab(boot_all, Path(home_results) / "boot_betaest_all.txt")

        if _is_yes(cfg.get("if_predict")) and iter_val == n_boot_iter + 1:
            predict_df = _safe_read(results_store, "predict")
            ancillary_df = _safe_read(results_store, "ancillary")
            if predict_df is not None and ancillary_df is not None:
                summary_pred = summarize_predict(
                    predict=predict_df, ancillary=ancillary_df, config=cfg
                )
                _log_messages(summary_pred.warnings, warn)
                _log_messages(summary_pred.errors, error)
                if summary_pred.predict is not None:
                    if _is_yes(cfg.get("if_add_predict_placeholders")):
                        summary_pred.predict = _ensure_predict_placeholders(
                            summary_pred.predict, cfg
                        )
                    results_store.write("predict", summary_pred.predict)
                if summary_pred.summary_predict is not None:
                    results_store.write("summary_predict", summary_pred.summary_predict)
                    _append_model_results(
                        home_results,
                        "prediction",
                        summary_pred.summary_predict,
                    )
                if summary_pred.lu_yield_percentiles is not None:
                    results_store.write(
                        "LU_yield_percentiles", summary_pred.lu_yield_percentiles
                    )

                if _is_yes(cfg.get("if_output_to_tab")) and home_results:
                    if summary_pred.predict is not None:
                        _write_tab(
                            summary_pred.predict, Path(home_results) / "predict.txt"
                        )
                    if results_store.exists("test_data"):
                        _write_tab(
                            results_store.read("test_data"),
                            Path(home_results) / "test_data.txt",
                        )
                    if results_store.exists("test_predict"):
                        _write_tab(
                            results_store.read("test_predict"),
                            Path(home_results) / "test_predict.txt",
                        )
                    if summary_pred.summary_predict is not None:
                        _write_tab(
                            summary_pred.summary_predict,
                            Path(home_results) / "summary_predict.txt",
                        )
                    if summary_pred.lu_yield_percentiles is not None:
                        _write_tab(
                            summary_pred.lu_yield_percentiles,
                            Path(home_results) / "LU_yield_percentiles.txt",
                        )

    return RunResult(
        config=dict(cfg),
        model_spec=model_spec,
        comment_all=comment_all,
        predict_state=predict_state,
        warnings=warnings,
        errors=errors,
    )


def _split_tokens(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return value.split()
    if isinstance(value, (list, tuple)):
        tokens: list[str] = []
        for item in value:
            if item is None:
                continue
            if isinstance(item, str):
                tokens.extend(item.split())
            else:
                tokens.append(str(item))
        return tokens
    return str(value).split()


def _is_yes(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().upper() == "YES"


def _to_int(value: object, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    try:
        return int(float(str(value).strip()))
    except ValueError:
        return default


def _first_path(config: Mapping[str, object], *keys: str) -> str | None:
    for key in keys:
        val = config.get(key)
        if val:
            return str(val)
    return None


def _as_table_name(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _read_indata(cfg: Mapping[str, object], store: FileTableStore) -> pd.DataFrame | None:
    val = cfg.get("indata")
    if isinstance(val, pd.DataFrame):
        return val.copy()
    name = _as_table_name(val)
    if not name:
        return None
    if not store.exists(name):
        return None
    return store.read(name)


def _load_constraints(
    cfg: Mapping[str, object], store: FileTableStore
) -> pd.DataFrame | None:
    constraint_file = cfg.get("constraint_file")
    if constraint_file is None:
        return None
    if isinstance(constraint_file, pd.DataFrame):
        return constraint_file.copy()
    path = Path(str(constraint_file))
    if path.suffix:
        if path.exists():
            return _read_table(path)
        return None
    name = str(constraint_file).strip()
    if name and store.exists(name):
        return store.read(name)
    return None


def _safe_read(store: FileTableStore, name: str) -> pd.DataFrame | None:
    if store.exists(name):
        return store.read(name)
    return None


def _clear_data(iter_val: int, store: FileTableStore) -> None:
    if iter_val == 0:
        names = [
            "resids",
            "cov_betaest",
            "summary_betaest",
            "boot_betaest",
            "summary_boot_betaest",
            "plotdat",
            "probdat",
            "mapresids",
            "predict",
            "model_parm_predict",
            "comment",
        ]
    else:
        names = ["boot_betaest", "predict", "test"]
    for name in names:
        store.delete(name)


def _backup_tables(store: FileTableStore, names: list[str]) -> None:
    for name in names:
        bak = f"bak_{name}"
        if store.exists(bak):
            store.delete(bak)
        if store.exists(name):
            store.write(bak, store.read(name))


def _backup_files(base_dir: str | None, names: list[str]) -> None:
    if not base_dir:
        return
    base = Path(base_dir)
    for name in names:
        src = base / name
        if not src.exists():
            continue
        bak = base / f"bak_{name}"
        if bak.exists():
            bak.unlink()
        shutil.copy2(src, bak)


def _append_comment(
    store: FileTableStore,
    existing: pd.DataFrame | None,
    iter_val: int,
    jter_val: int,
    comment: str,
) -> pd.DataFrame:
    row = pd.DataFrame(
        [
            {
                "iter": iter_val,
                "jter": jter_val,
                "comment": str(comment)[:200],
            }
        ]
    )
    if existing is None or existing.empty:
        updated = row
    else:
        updated = pd.concat([existing, row], axis=0, ignore_index=True)
    store.write("comment_all", updated)
    return updated


def _format_comment(prefix: str, message: str) -> str:
    text = str(message)
    upper = text.strip().upper()
    if upper.startswith("WARNING:") or upper.startswith("ERROR:"):
        return text
    return f"{prefix}: {text}"


def _log_messages(messages: list[str], sink: Callable[[str], None]) -> None:
    for msg in messages:
        sink(msg)


def _append_model_results(
    home_results: str | None, section: str, table: pd.DataFrame
) -> None:
    if not home_results:
        return
    out = Path(home_results) / "summary_model_rslts.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as fh:
        fh.write(f"\n[{section}]\n")
        if table is None or table.empty:
            fh.write("no rows\n")
            return
        fh.write(table.to_string(index=False))
        fh.write("\n")


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        if path.suffix.lower() in {".htm", ".html"}:
            path.write_text("<html><body></body></html>\n", encoding="utf-8")
        else:
            path.write_text("", encoding="utf-8")


def _initialize_run_outputs(home_results: Path) -> None:
    """Reset run-level text summaries so reruns do not retain stale paths or sections."""
    home_results.mkdir(parents=True, exist_ok=True)
    (home_results / "summary_model_specs.txt").write_text("", encoding="utf-8")
    (home_results / "summary_model_rslts.txt").write_text("", encoding="utf-8")
    _touch(home_results / "SPARROW_Output.htm")


def _find_existing(base: Path, name: str) -> Path | None:
    for ext in (".parquet", ".feather", ".csv", ".tsv", ".txt", ".sas7bdat", ".xpt"):
        candidate = base / f"{name}{ext}"
        if candidate.exists():
            return candidate
    return None


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in (".csv", ".txt"):
        sep = "\t" if suffix == ".txt" else ","
        return pd.read_csv(path, sep=sep)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".feather":
        return pd.read_feather(path)
    if suffix in (".sas7bdat", ".xpt"):
        return pd.read_sas(path)
    raise ValueError(f"Unsupported table format: {path.suffix}")


def _write_table(path: Path, df: pd.DataFrame) -> None:
    suffix = path.suffix.lower()
    tmp_path = path.with_name(f"{path.stem}.tmp{path.suffix}")

    def _replace_with_retry() -> None:
        last_error: PermissionError | None = None
        for delay in (0.0, 0.05, 0.1, 0.25, 0.5, 1.0):
            if delay:
                time.sleep(delay)
            try:
                os.replace(tmp_path, path)
                return
            except PermissionError as exc:
                last_error = exc
        if last_error is not None:
            raise last_error

    def _rewrite_text_if_nuls(sep: str) -> None:
        try:
            if b"\x00" not in path.read_bytes():
                return
        except OSError:
            return
        with path.open("w", encoding="utf-8", newline="") as handle:
            df.to_csv(handle, sep=sep, index=False)
        if b"\x00" in path.read_bytes():
            raise OSError(f"NUL bytes remain after rewriting {path}.")

    if suffix in (".csv", ".txt"):
        sep = "\t" if suffix == ".txt" else ","
        df.to_csv(tmp_path, sep=sep, index=False)
        _replace_with_retry()
        _rewrite_text_if_nuls(sep)
        return
    if suffix == ".tsv":
        df.to_csv(tmp_path, sep="\t", index=False)
        _replace_with_retry()
        _rewrite_text_if_nuls("\t")
        return
    if suffix == ".parquet":
        df.to_parquet(tmp_path, index=False)
        _replace_with_retry()
        return
    if suffix == ".feather":
        df.to_feather(tmp_path)
        _replace_with_retry()
        return
    raise ValueError(f"Unsupported table format: {path.suffix}")


def _write_tab(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)


def _ensure_predict_placeholders(
    predict_df: pd.DataFrame, cfg: Mapping[str, object]
) -> pd.DataFrame:
    df = predict_df.copy()
    srcvar = _split_tokens(cfg.get("srcvar"))
    base_missing = ["pload_nd_total", "res_decay"]
    base_missing.extend([f"pload_nd_{src}" for src in srcvar])

    for col in base_missing:
        if col not in df.columns:
            df[col] = np.nan

    n_boot_iter = _to_int(cfg.get("n_boot_iter"), 0)
    if n_boot_iter > 0:
        for col in base_missing:
            for prefix in ("mean_", "se_", "ci_lo_", "ci_hi_"):
                name = f"{prefix}{col}"
                if name not in df.columns:
                    df[name] = np.nan

    return df


def _load_basemap(cfg: Mapping[str, object]) -> pd.DataFrame | None:
    if not _is_yes(cfg.get("if_gis")):
        return None
    gis_file = str(cfg.get("gis_file") or "").strip()
    if not gis_file:
        return None

    path = Path(gis_file)
    if not path.is_absolute():
        home_gis = _first_path(cfg, "home_gis", "gis_dir")
        if home_gis:
            path = Path(home_gis) / gis_file

    candidate = path if path.suffix else None
    if candidate is None or not candidate.exists():
        base = path.parent if path.parent != Path(".") else Path(".")
        name = path.stem if path.suffix else path.name
        found = _find_existing(base, name)
        if found is not None:
            candidate = found
        elif path.exists():
            candidate = path

    if candidate is None or not candidate.exists():
        return None

    try:
        return _read_table(candidate)
    except Exception:
        return None
