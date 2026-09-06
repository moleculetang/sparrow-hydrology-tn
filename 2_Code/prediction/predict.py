"""SAS sparrow_predict.sas replacement helpers."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import re
from statistics import NormalDist
from typing import Iterable, Mapping, MutableMapping

import numpy as np
import pandas as pd


@dataclass
class PredictResult:
    predict: pd.DataFrame | None
    predlst: list[str]
    predlst_str: str
    upmonload: pd.DataFrame | None
    test_data: pd.DataFrame | None
    debug_trace: pd.DataFrame | None = None
    debug_del_frac_chain: pd.DataFrame | None = None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


_NORMAL_DIST = NormalDist()


def _norm_ppf(x: float | np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    arr = np.clip(arr, np.finfo(float).tiny, 1 - np.finfo(float).eps)
    return np.vectorize(_NORMAL_DIST.inv_cdf)(arr).astype(float)


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


def _first_token(value: object) -> str:
    tokens = _split_tokens(value)
    return tokens[0] if tokens else ""


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


def _set_unique(tokens: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen_upper: set[str] = set()
    for item in tokens:
        if not item:
            continue
        upper = item.upper()
        if upper not in seen_upper:
            seen_upper.add(upper)
            out.append(item)
    return out


def _locin(search_list: Iterable[str], master_list: Iterable[str]) -> list[int]:
    master_upper = [item.upper() for item in master_list]
    out: list[int] = []
    for item in search_list:
        if not item:
            out.append(0)
            continue
        target = item.upper()
        try:
            out.append(master_upper.index(target) + 1)
        except ValueError:
            out.append(0)
    return out or [0]


def _translate_iml(expr: str) -> str:
    binary_term = r"(?:-?\([^()]+\)|-?[A-Za-z_][\w\[\],:]*|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
    pattern_max = re.compile(rf"(?P<a>{binary_term})\s*<>\s*(?P<b>{binary_term})")
    pattern_min = re.compile(rf"(?P<a>{binary_term})\s*><\s*(?P<b>{binary_term})")
    while pattern_max.search(expr):
        expr = pattern_max.sub(r"__sas_max(\g<a>, \g<b>)", expr)
    while pattern_min.search(expr):
        expr = pattern_min.sub(r"__sas_min(\g<a>, \g<b>)", expr)
    expr = expr.replace("##", "**")
    expr = expr.replace("#", "*")
    expr = re.sub(r"\bdata\s*\[,\s*([A-Za-z_]\w*)\s*\]", r"data[:, \1]", expr)
    expr = re.sub(r"\bbeta\s*\[,\s*([A-Za-z_]\w*)\s*\]", r"beta[\1]", expr)
    return expr


def _eval_spec(spec: object, context: Mapping[str, object], default: np.ndarray) -> np.ndarray:
    if spec is None:
        return default
    if isinstance(spec, np.ndarray):
        return spec
    if callable(spec):
        return np.asarray(spec(context))
    text = str(spec).strip()
    if not text:
        return default
    expr = _translate_iml(text)
    local_dict = dict(context)
    local_dict.update(
        {
            "np": np,
            "exp": np.exp,
            "log": np.log,
            "sqrt": np.sqrt,
            "floor": np.floor,
            "ceil": np.ceil,
            "repeat": lambda x, n: np.tile(x, (int(n), 1)),
            "j": lambda r, c, v: np.full((int(r), int(c)), v),
            "choose": lambda cond, if_true, if_false: np.where(cond, if_true, if_false),
            "probit": _norm_ppf,
            "__sas_max": np.maximum,
            "__sas_min": np.minimum,
        }
    )
    return np.asarray(eval(expr, {"__builtins__": {}}, local_dict))


def _parse_dlvdsgn(
    value: object, nsrc: int, ndlv: int
) -> np.ndarray:
    if not value or nsrc <= 0 or ndlv <= 0:
        return np.empty((0, 0), dtype=float)
    rows = [row.strip() for row in str(value).split(",") if row.strip()]
    out = np.zeros((nsrc, ndlv), dtype=float)
    for i, row in enumerate(rows[:nsrc]):
        parts = row.split()
        for j, part in enumerate(parts[:ndlv]):
            try:
                out[i, j] = float(part)
            except ValueError:
                out[i, j] = 0.0
    return out


def _parse_float_tokens(value: object) -> set[float]:
    out: set[float] = set()
    for token in _split_tokens(value):
        try:
            out.add(float(token))
        except ValueError:
            continue
    return out


def _parse_int_tokens(value: object) -> set[int]:
    out: set[int] = set()
    for token in _split_tokens(value):
        try:
            out.add(int(float(token)))
        except ValueError:
            continue
    return out


def _parse_override_mapping(value: object) -> dict[str, float]:
    overrides: dict[str, float] = {}
    if value is None:
        return overrides
    if isinstance(value, Mapping):
        for key, raw_value in value.items():
            if key is None or raw_value is None:
                continue
            try:
                overrides[str(key).strip()] = float(raw_value)
            except (TypeError, ValueError):
                continue
        return overrides
    if isinstance(value, (list, tuple)):
        for item in value:
            overrides.update(_parse_override_mapping(item))
        return overrides
    text = str(value).strip()
    if not text:
        return overrides
    for token in re.split(r"[\s,;]+", text):
        if not token or "=" not in token:
            continue
        key, raw_value = token.split("=", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        if not key or not raw_value:
            continue
        try:
            overrides[key] = float(raw_value)
        except ValueError:
            continue
    return overrides


def predict(
    indata: pd.DataFrame,
    boot_betaest: pd.DataFrame | Mapping[str, object],
    config: MutableMapping[str, object],
) -> PredictResult:
    warnings: list[str] = []
    errors: list[str] = []

    def warn(msg: str) -> None:
        warnings.append(msg)

    def error(msg: str) -> None:
        errors.append(msg)
        config["if_error"] = "yes"

    if "if_error" not in config:
        config["if_error"] = "no"

    if_exclude_inc_decay = _is_yes(config.get("if_exclude_inc_decay"))
    if_adjust = _is_yes(config.get("if_adjust"))
    if_estimate_ic = _is_yes(config.get("if_estimate_ic"))
    if_test_predict = _is_yes(config.get("if_test_predict"))

    datalst = _split_tokens(config.get("datalst"))
    if not datalst:
        error("datalst is missing in config; run makemacros first.")
        return PredictResult(None, [], "", None, None, None, None, warnings, errors)

    missing_cols = [col for col in datalst if col and col not in indata.columns]
    if missing_cols:
        error(
            "The following datalst variables were not found in indata: "
            + " ".join(missing_cols)
            + "."
        )
        return PredictResult(None, [], "", None, None, None, None, warnings, errors)

    data_df = indata.loc[:, [col for col in datalst if col]].copy()
    data = data_df.to_numpy()

    makecol = config.get("makecol") or {}

    def _idx(name: str) -> int:
        vals = makecol.get(name) or []
        return int(vals[0] - 1) if vals and vals[0] > 0 else -1

    def _idx_list(name: str) -> np.ndarray:
        vals = makecol.get(name) or []
        return np.array([v - 1 for v in vals if v and v > 0], dtype=int)

    jwaterid = _idx("jwaterid")
    jstaid = _idx("jstaid")
    jfnode = _idx("jfnode")
    jtnode = _idx("jtnode")
    jfrac = _idx("jfrac")
    jtarget = _idx("jtarget")
    jiftran = _idx("jiftran")
    jdepvar = _idx("jdepvar")
    jsrcvar = _idx_list("jsrcvar")
    jdlvvar = _idx_list("jdlvvar")
    jdecvar = _idx_list("jdecvar")
    jresvar = _idx_list("jresvar")
    jbsrcvar = _idx_list("jbsrcvar")
    jbdlvvar = _idx_list("jbdlvvar")
    jbdecvar = _idx_list("jbdecvar")
    jbresvar = _idx_list("jbresvar")

    jsrcvar_raw = makecol.get("jsrcvar") or []
    has_sources = bool(jsrcvar_raw) and all(val > 0 for val in jsrcvar_raw)

    nreach = data.shape[0]
    if jfnode >= 0 and jtnode >= 0:
        nnode = int(np.nanmax(data[:, [jfnode, jtnode]]))
    else:
        nnode = 0

    n_periods = _to_int(config.get("n_periods"), 0)
    nrch = int(nreach / n_periods) if n_periods else nreach

    if isinstance(boot_betaest, Mapping):
        boot_betaest_df = pd.DataFrame([boot_betaest])
    else:
        boot_betaest_df = boot_betaest

    if boot_betaest_df is None or boot_betaest_df.empty:
        error("boot_betaest is empty; prediction requires a row for the current iteration.")
        return PredictResult(None, [], "", None, None, None, None, warnings, errors)

    betalst = _split_tokens(config.get("betalst"))
    if not betalst:
        error("betalst is missing in config; run makemacros first.")
        return PredictResult(None, [], "", None, None, None, None, warnings, errors)

    for col in betalst:
        if col not in boot_betaest_df.columns:
            error(f"boot_betaest is missing coefficient column {col}.")
            return PredictResult(None, [], "", None, None, None, None, warnings, errors)

    if "mean_exp_weighted_error" not in boot_betaest_df.columns:
        error("boot_betaest is missing mean_exp_weighted_error.")
        return PredictResult(None, [], "", None, None, None, None, warnings, errors)

    beta = boot_betaest_df.loc[boot_betaest_df.index[0], betalst].to_numpy(dtype=float).copy()
    mean_exp_weighted_error = float(boot_betaest_df.loc[boot_betaest_df.index[0], "mean_exp_weighted_error"])
    if _is_yes(config.get("if_tp_predict_beta_override")):
        override_map = _parse_override_mapping(config.get("tp_predict_beta_overrides"))
        if override_map:
            name_to_idx = {name.upper(): idx for idx, name in enumerate(betalst)}
            applied: list[str] = []
            unknown: list[str] = []
            for name, value in override_map.items():
                idx = name_to_idx.get(name.upper())
                if idx is None:
                    unknown.append(name)
                    continue
                beta[idx] = float(value)
                applied.append(f"{betalst[idx]}={float(value):.12g}")
            if applied:
                warn("Predict diagnostic beta overrides applied: " + ", ".join(applied))
            if unknown:
                warn(
                    "Predict diagnostic beta overrides ignored because coefficients were not found: "
                    + ", ".join(sorted(unknown))
                )

    srcvar = _split_tokens(config.get("srcvar"))
    varnames = ["total", *srcvar, "total_nd"]
    predlst = [f"pload_{name}" for name in varnames]
    predlst.extend([f"pload_inc_{name}" for name in varnames])
    predlst.append("del_frac")
    predlst_str = " ".join(predlst)
    config["predlst"] = predlst_str

    retrans_exclude = _split_tokens(config.get("retrans_exclude_list"))
    exclude_list = _locin(retrans_exclude, predlst)
    exclude_idx = [idx - 1 for idx in exclude_list if idx and idx > 0]

    reach_spec = config.get("reach_decay_specification")
    res_spec = config.get("reservoir_decay_specification")
    incr_spec = config.get("incr_delivery_specification")
    dlvdsgn = _parse_dlvdsgn(config.get("dlvdsgn"), int(jsrcvar.size), int(jdlvvar.size))

    waterid_name = _first_token(config.get("waterid"))
    period_values: np.ndarray | None = None
    if "period" in data_df.columns:
        period_values = pd.to_numeric(data_df["period"], errors="coerce").to_numpy(dtype=float)

    debug_trace_df: pd.DataFrame | None = None
    debug_del_frac_chain_df: pd.DataFrame | None = None
    debug_indices = np.array([], dtype=int)
    debug_rows: dict[int, dict[str, object]] = {}
    if _is_yes(config.get("if_debug_predict_trace")):
        if not waterid_name or waterid_name not in data_df.columns:
            warn(
                "Predict trace requested but waterid is missing from indata; debug_predict_trace not generated."
            )
        else:
            trace_time_comids = _parse_float_tokens(config.get("debug_trace_time_comids"))
            trace_periods = _parse_int_tokens(config.get("debug_trace_periods"))
            if not trace_time_comids and not trace_periods:
                warn(
                    "Predict trace requested but no debug_trace_time_comids or debug_trace_periods were supplied; debug_predict_trace not generated."
                )
            else:
                mask = np.ones((nreach,), dtype=bool)
                if trace_time_comids:
                    water_vals = pd.to_numeric(data_df[waterid_name], errors="coerce")
                    mask &= water_vals.isin(trace_time_comids).to_numpy()
                if trace_periods:
                    if "period" in data_df.columns:
                        period_vals = pd.to_numeric(data_df["period"], errors="coerce")
                        mask &= period_vals.isin(trace_periods).to_numpy()
                    else:
                        warn(
                            "Predict trace period filter requested but period is missing from indata; period filter ignored."
                        )
                debug_indices = np.flatnonzero(mask)
                debug_trace_max_rows = _to_int(config.get("debug_trace_max_rows"), 0)
                if debug_trace_max_rows > 0 and debug_indices.size > debug_trace_max_rows:
                    warn(
                        f"Predict trace matched {debug_indices.size} rows; truncating to the first {debug_trace_max_rows}."
                    )
                    debug_indices = debug_indices[:debug_trace_max_rows]
                base_cols = [
                    waterid_name,
                    "comid",
                    "year",
                    "quarter",
                    "period",
                ]
                for idx in debug_indices.tolist():
                    row: dict[str, object] = {"row_index": int(idx)}
                    for col in base_cols:
                        if col in data_df.columns:
                            row[col] = data_df.iloc[idx][col]
                    if jfnode >= 0:
                        row["fnode"] = float(data[idx, jfnode])
                    if jtnode >= 0:
                        row["tnode"] = float(data[idx, jtnode])
                    if jfrac >= 0:
                        row["frac"] = float(data[idx, jfrac])
                    if jiftran >= 0:
                        row["iftran"] = float(data[idx, jiftran])
                    if jtarget >= 0:
                        row["target"] = float(data[idx, jtarget])
                    if jdepvar >= 0:
                        row["depvar"] = float(data[idx, jdepvar]) if not np.isnan(data[idx, jdepvar]) else np.nan
                    debug_rows[idx] = row

    def _trace_scalar(idx: int, key: str, value: object) -> None:
        row = debug_rows.get(idx)
        if row is None:
            return
        if isinstance(value, (np.generic,)):
            row[key] = value.item()
        else:
            row[key] = value

    def _trace_vector(idx: int, prefix: str, names: list[str], values: np.ndarray) -> None:
        row = debug_rows.get(idx)
        if row is None:
            return
        for pos, name in enumerate(names):
            if pos >= values.shape[0]:
                break
            val = values[pos]
            row[f"{prefix}_{name}"] = val.item() if isinstance(val, np.generic) else val

    def _get_decay(spec: object, default_val: float = 1.0) -> np.ndarray:
        arr = _eval_spec(
            spec,
            {
                "data": data,
                "beta": beta,
                "jdecvar": jdecvar,
                "jbdecvar": jbdecvar,
                "jresvar": jresvar,
                "jbresvar": jbresvar,
                "nreach": nreach,
            },
            np.full((nreach,), default_val),
        )
        arr = np.asarray(arr, dtype=float).reshape(-1)
        if arr.size == 1 and nreach > 1:
            arr = np.full((nreach,), float(arr[0]))
        return arr

    def _get_del2strm() -> np.ndarray:
        if jbsrcvar.size == 0:
            return np.zeros((nreach, 0))
        base = beta[jbsrcvar]
        if incr_spec:
            factors = _eval_spec(
                incr_spec,
                {
                    "data": data,
                    "beta": beta,
                    "jbsrcvar": jbsrcvar,
                    "jsrcvar": jsrcvar,
                    "jdlvvar": jdlvvar,
                    "jbdlvvar": jbdlvvar,
                    "dlvdsgn": dlvdsgn,
                    "nreach": nreach,
                },
                np.tile(base, (nreach, 1)),
            )
            factors = np.asarray(factors, dtype=float)
            if factors.ndim == 1:
                factors = factors.reshape(-1, 1)
            if factors.shape[0] != nreach:
                factors = np.tile(factors.reshape(1, -1), (nreach, 1))
            return factors * base
        return np.tile(base, (nreach, 1))

    rchdcayf = _get_decay(reach_spec, 1.0)
    resdcayf = _get_decay(res_spec, 1.0)

    incddsrc = np.zeros((nreach, 1))
    storage_src_idx = -1
    if has_sources and jsrcvar.size > 0:
        del2strm_factors = _get_del2strm()
        data_src = data[:, jsrcvar].copy()

        catchment_storage_source = _to_int(config.get("catchment_storage_source"), 0)
        storage_src_idx = catchment_storage_source - 1 if catchment_storage_source > 0 else -1
        exclude_tokens = _split_tokens(config.get("catchment_storage_source_exclude"))
        exclude0_tokens = _split_tokens(config.get("catchment_storage_source_exclud0"))
        exclude_vals: list[int] = []
        for token in exclude_tokens:
            try:
                exclude_vals.append(int(float(token)) - 1)
            except ValueError:
                continue
        jstoreincldsrcs = np.array(
            [i for i in range(jsrcvar.size) if i not in exclude_vals], dtype=int
        )
        exclude0_vals: list[int] = []
        for token in exclude0_tokens:
            try:
                exclude0_vals.append(int(float(token)) - 1)
            except ValueError:
                continue
        jstoreincldsrcs0 = np.array(
            [i for i in range(jsrcvar.size) if i not in exclude0_vals], dtype=int
        )

        if catchment_storage_source > 0 and n_periods > 0:
            if if_estimate_ic:
                if jstoreincldsrcs0.size > 0:
                    incdelbysrc0 = data_src[:, jstoreincldsrcs0] * del2strm_factors[:, jstoreincldsrcs0]
                    incload0 = incdelbysrc0.sum(axis=1)
                    src_idx = catchment_storage_source - 1
                    if 0 <= src_idx < del2strm_factors.shape[1]:
                        bt = del2strm_factors[:, src_idx]
                        K = 0.97
                        start0 = 0
                        end0 = nrch
                        slices: list[tuple[np.ndarray, np.ndarray]] = []
                        for _ in range(4):
                            b = np.minimum(bt[start0:end0], K)
                            x = incload0[start0:end0]
                            slices.append((b, x))
                            start0 += nrch
                            end0 += nrch
                        if len(slices) == 4:
                            b1, X1 = slices[0]
                            b2, X2 = slices[1]
                            b3, X3 = slices[2]
                            b4, X4 = slices[3]
                            pb = np.minimum(b1 * b2 * b3 * b4, K)
                            denom = np.where((1 - pb) == 0, np.nan, (1 - pb))
                            L1 = X1 + (b4 * b3 * b1 * X2 + b4 * b1 * X3 + b1 * X4) / denom
                            L1 = np.nan_to_num(L1, nan=0.0, posinf=0.0, neginf=0.0)
                            data_src[:nrch, src_idx] = L1
                            for idx in debug_indices.tolist():
                                if idx < nrch:
                                    _trace_scalar(idx, "storage_ic_bt", bt[idx])
                                    _trace_scalar(idx, "storage_ic_pb", pb[idx])
                                    _trace_scalar(idx, "storage_ic_incload0", incload0[idx])
                                    _trace_scalar(idx, "storage_ic_l1", L1[idx])

            incdelbysrc = np.zeros_like(data_src)
            start0 = 0
            end0 = nrch
            for t in range(1, n_periods + 1):
                incdelbysrc[start0:end0, :] = (
                    data_src[start0:end0, :] * del2strm_factors[start0:end0, :]
                )
                if t < n_periods:
                    start1 = start0 + nrch
                    end1 = end0 + nrch
                    src_idx = catchment_storage_source - 1
                    if 0 <= src_idx < data_src.shape[1] and jstoreincldsrcs.size > 0:
                        data_src[start1:end1, src_idx] = incdelbysrc[
                            start0:end0, jstoreincldsrcs
                        ].sum(axis=1)
                    start0 = start1
                    end0 = end1
            incddsrc = incdelbysrc
        else:
            incddsrc = data_src * del2strm_factors

        for idx in debug_indices.tolist():
            _trace_scalar(idx, "rchdcayf", rchdcayf[idx])
            _trace_scalar(idx, "resdcayf", resdcayf[idx])
            _trace_vector(idx, "src_input", srcvar, data[idx, jsrcvar])
            _trace_vector(idx, "src_used", srcvar, data_src[idx, :])
            _trace_vector(idx, "del2strm", srcvar, del2strm_factors[idx, :])
            _trace_vector(idx, "incdel", srcvar, incddsrc[idx, :])
            _trace_scalar(idx, "incdel_total_pre_decay", float(np.sum(incddsrc[idx, :])))
            if 0 <= storage_src_idx < data_src.shape[1]:
                _trace_scalar(idx, "storage_src_input", data[idx, jsrcvar[storage_src_idx]])
                _trace_scalar(idx, "storage_src_used", data_src[idx, storage_src_idx])
                _trace_scalar(
                    idx,
                    "storage_replaced_delta",
                    data_src[idx, storage_src_idx] - data[idx, jsrcvar[storage_src_idx]],
                )

    inctot = incddsrc.sum(axis=1)
    incddsrc = np.column_stack([inctot, incddsrc])
    inc_decay = (rchdcayf ** 0.5) * resdcayf
    incddsrc = np.column_stack([inc_decay.reshape(-1, 1) * incddsrc, inctot])

    n_vars = incddsrc.shape[1]
    n_src = n_vars

    node = np.zeros((nnode + 1, n_vars))
    rchld = np.zeros((nreach, n_vars))

    adjust_load = np.full((n_src,), 1.0 / mean_exp_weighted_error)
    carryf = np.tile((data[:, jfrac] * rchdcayf * resdcayf).reshape(-1, 1), (1, n_src))
    iftran_mat = np.tile(data[:, jiftran].reshape(-1, 1), (1, n_src))

    for idx in debug_indices.tolist():
        _trace_scalar(idx, "inc_decay", inc_decay[idx])
        _trace_scalar(idx, "carryf_total", carryf[idx, 0])
        _trace_scalar(idx, "incddsrc_total", incddsrc[idx, 0])
        _trace_vector(idx, "incddsrc", srcvar, incddsrc[idx, 1 : 1 + len(srcvar)])
        _trace_scalar(idx, "incddsrc_total_nd", incddsrc[idx, -1])

    test_rows: list[np.ndarray] = []
    test_obs: set[int] = set()
    for token in _split_tokens(config.get("test_obs")):
        text = str(token).strip()
        if not text:
            continue
        try:
            test_obs.add(int(float(text)) - 1)
        except ValueError:
            continue

    for i in range(nreach):
        fnode = int(data[i, jfnode]) if jfnode >= 0 else 0
        tnode = int(data[i, jtnode]) if jtnode >= 0 else 0
        upload_before = node[fnode, :].copy()
        rchld[i, :] = incddsrc[i, :] + carryf[i, :] * node[fnode, :]

        if if_test_predict and i in test_obs:
            test_rows.append(
                np.concatenate(
                    [
                        data[i, :],
                        np.array([rchdcayf[i], resdcayf[i]]),
                        incddsrc[i, :],
                        node[fnode, :],
                    ]
                )
            )

        if if_adjust:
            dep_val = data[i, jdepvar] if jdepvar >= 0 else np.nan
            if not np.isnan(dep_val):
                scale = dep_val / rchld[i, 0]
                rchld[i, :n_src] = scale * rchld[i, :n_src]
                node[tnode, :] = node[tnode, :] + iftran_mat[i, :] * adjust_load * rchld[i, :]
                _trace_scalar(i, "if_adjust_scale", scale)
            else:
                node[tnode, :] = node[tnode, :] + iftran_mat[i, :] * rchld[i, :]
        else:
            node[tnode, :] = node[tnode, :] + iftran_mat[i, :] * rchld[i, :]

        if i in debug_rows:
            _trace_scalar(i, "upnode_total_before", upload_before[0])
            _trace_scalar(i, "upnode_total_nd_before", upload_before[-1])
            _trace_scalar(i, "rchld_total", rchld[i, 0])
            _trace_vector(i, "rchld", srcvar, rchld[i, 1 : 1 + len(srcvar)])
            _trace_scalar(i, "rchld_total_nd", rchld[i, -1])
            _trace_scalar(i, "node_tnode_total_after", node[tnode, 0])
            _trace_scalar(i, "node_tnode_total_nd_after", node[tnode, -1])

    if jtarget >= 0:
        node_del = np.zeros((nnode + 1,))
        delfrac = np.zeros((nreach,))
        for i in range(nreach - 1, -1, -1):
            node_del_before = 0.0
            if not np.isnan(data[i, jtarget]) and data[i, jtarget] != 0:
                delfrac[i] = 1.0
            else:
                tnode = int(data[i, jtnode]) if jtnode >= 0 else 0
                node_del_before = node_del[tnode]
                delfrac[i] = data[i, jiftran] * node_del[tnode]
            fnode = int(data[i, jfnode]) if jfnode >= 0 else 0
            node_del[fnode] = node_del[fnode] + delfrac[i] * carryf[i, 0]
            if if_exclude_inc_decay:
                delfrac[i] = delfrac[i] * inc_decay[i]
            if i in debug_rows:
                _trace_scalar(i, "node_del_tnode_before", node_del_before)
                _trace_scalar(i, "delfrac", delfrac[i])
                _trace_scalar(i, "node_del_fnode_after", node_del[fnode])
    else:
        delfrac = np.full((nreach,), np.nan)

    incddsrc_out = incddsrc[:, :n_src]
    rchld_out = np.concatenate(
        [rchld[:, :n_vars], incddsrc_out, delfrac.reshape(-1, 1)], axis=1
    )

    retransform_factor = np.full((rchld_out.shape[1],), mean_exp_weighted_error)
    if exclude_idx:
        retransform_factor[exclude_idx] = 1.0

    predict_values = rchld_out * retransform_factor.reshape(1, -1)

    if if_adjust:
        n_src_basic = jsrcvar.size + 1
        if n_src_basic > 0:
            dep_mask = ~np.isnan(data[:, jdepvar]) if jdepvar >= 0 else np.zeros((nreach,), dtype=bool)
            predict_values[dep_mask, :n_src_basic] = (
                predict_values[dep_mask, :n_src_basic]
                * (1.0 / retransform_factor[:n_src_basic])
            )

    prefix_names: list[str] = []
    prefix_cols: list[np.ndarray] = []
    waterid = _first_token(config.get("waterid"))
    staid = _first_token(config.get("staid"))
    depvar = _first_token(config.get("depvar"))
    if waterid:
        prefix_names.append(waterid)
        prefix_cols.append(data[:, jwaterid] if jwaterid >= 0 else np.full((nreach,), np.nan))
    if staid:
        prefix_names.append(staid)
        prefix_cols.append(data[:, jstaid] if jstaid >= 0 else np.full((nreach,), np.nan))
    if depvar:
        prefix_names.append(depvar)
        prefix_cols.append(data[:, jdepvar] if jdepvar >= 0 else np.full((nreach,), np.nan))

    if prefix_cols:
        predict_matrix = np.column_stack(prefix_cols + [predict_values])
        predict_columns = prefix_names + predlst
    else:
        predict_matrix = predict_values
        predict_columns = predlst

    predict_df = pd.DataFrame(predict_matrix, columns=predict_columns)

    if debug_rows:
        debug_trace_df = pd.DataFrame(list(debug_rows.values()))
        sort_cols = [col for col in ["year", "quarter", "period", waterid_name, "row_index"] if col in debug_trace_df.columns]
        if sort_cols:
            debug_trace_df = debug_trace_df.sort_values(by=sort_cols, kind="stable", na_position="first")
        debug_trace_df = debug_trace_df.reset_index(drop=True)

    chain_targets = _parse_int_tokens(config.get("debug_del_frac_chain_time_comids"))
    if chain_targets:
        if not waterid_name or waterid_name not in data_df.columns:
            warn(
                "DEL_FRAC chain trace requested but waterid is missing from indata; debug_del_frac_chain not generated."
            )
        elif jfnode < 0 or jtnode < 0:
            warn(
                "DEL_FRAC chain trace requested but fnode/tnode is missing from indata; debug_del_frac_chain not generated."
            )
        else:
            water_vals = pd.to_numeric(data_df[waterid_name], errors="coerce").to_numpy(dtype=float)
            water_keys = np.full((nreach,), -1, dtype=np.int64)
            finite_water = np.isfinite(water_vals)
            water_keys[finite_water] = np.rint(water_vals[finite_water]).astype(np.int64)
            row_by_time_comid: dict[int, int] = {}
            for idx, key in enumerate(water_keys.tolist()):
                if key >= 0 and key not in row_by_time_comid:
                    row_by_time_comid[key] = idx

            downstream_map: dict[tuple[int | None, int], list[int]] = {}
            for idx in range(nreach):
                fnode = int(data[idx, jfnode])
                if period_values is not None and np.isfinite(period_values[idx]):
                    map_key = (int(period_values[idx]), fnode)
                else:
                    map_key = (None, fnode)
                downstream_map.setdefault(map_key, []).append(idx)

            chain_rows: list[dict[str, object]] = []
            max_chain_rows = _to_int(config.get("debug_del_frac_chain_max_rows"), 0)
            selected_starts = [target for target in sorted(chain_targets) if target in row_by_time_comid]
            missing_starts = [target for target in sorted(chain_targets) if target not in row_by_time_comid]
            if missing_starts:
                warn(
                    "DEL_FRAC chain trace could not find time_comid values: "
                    + " ".join(str(value) for value in missing_starts)
                )

            for start_rank, start_time_comid in enumerate(selected_starts, start=1):
                start_idx = row_by_time_comid[start_time_comid]
                queue: deque[tuple[int, int | None, int, float]] = deque()
                queue.append((start_idx, None, 0, 1.0))
                seen: set[int] = set()
                while queue:
                    idx, parent_idx, level, cumulative_before = queue.popleft()
                    if idx in seen:
                        continue
                    seen.add(idx)

                    fnode = int(data[idx, jfnode])
                    tnode = int(data[idx, jtnode])
                    step_decay = float(rchdcayf[idx] * resdcayf[idx])
                    cumulative_decay = float(cumulative_before * step_decay)
                    if period_values is not None and np.isfinite(period_values[idx]):
                        child_key = (int(period_values[idx]), tnode)
                    else:
                        child_key = (None, tnode)
                    child_indices = downstream_map.get(child_key, [])
                    child_time_comids = [
                        str(int(value))
                        for value in np.rint(water_vals[child_indices]).astype(np.int64).tolist()
                        if np.isfinite(value)
                    ]

                    row: dict[str, object] = {
                        "chain_start_time_comid": int(start_time_comid),
                        "chain_start_rank": int(start_rank),
                        "chain_level": int(level),
                        "row_index": int(idx),
                        "time_comid": float(water_vals[idx]) if np.isfinite(water_vals[idx]) else np.nan,
                        "fnode": float(fnode),
                        "tnode": float(tnode),
                        "node_transition": f"{fnode}->{tnode}",
                        "parent_time_comid": (
                            float(water_vals[parent_idx])
                            if parent_idx is not None and np.isfinite(water_vals[parent_idx])
                            else np.nan
                        ),
                        "downstream_count": int(len(child_indices)),
                        "downstream_time_comids": " ".join(child_time_comids),
                        "rchdcayf": float(rchdcayf[idx]),
                        "resdcayf": float(resdcayf[idx]),
                        "step_decay": step_decay,
                        "cumulative_decay_from_start": cumulative_decay,
                        "strmload1": (
                            float(data[idx, jdecvar[0]])
                            if jdecvar.size > 0
                            else np.nan
                        ),
                        "resload1": (
                            float(data[idx, jresvar[0]])
                            if jresvar.size > 0
                            else np.nan
                        ),
                        "DEL_FRAC": float(delfrac[idx]),
                    }
                    for col in ["comid", "year", "quarter", "period"]:
                        if col in data_df.columns:
                            row[col] = data_df.iloc[idx][col]
                    chain_rows.append(row)
                    if max_chain_rows > 0 and len(chain_rows) >= max_chain_rows:
                        warn(
                            f"DEL_FRAC chain trace reached debug_del_frac_chain_max_rows={max_chain_rows}; output truncated."
                        )
                        queue.clear()
                        break

                    for child_idx in child_indices:
                        queue.append((child_idx, idx, level + 1, cumulative_decay))

            if chain_rows:
                debug_del_frac_chain_df = pd.DataFrame(chain_rows)
                sort_cols = [
                    col
                    for col in [
                        "chain_start_rank",
                        "chain_level",
                        "year",
                        "quarter",
                        "period",
                        "time_comid",
                        "row_index",
                    ]
                    if col in debug_del_frac_chain_df.columns
                ]
                if sort_cols:
                    debug_del_frac_chain_df = debug_del_frac_chain_df.sort_values(
                        by=sort_cols,
                        kind="stable",
                        na_position="first",
                    )
                debug_del_frac_chain_df = debug_del_frac_chain_df.reset_index(drop=True)

    upmonload_df: pd.DataFrame | None = None
    if if_adjust:
        rchdcayf_up = _get_decay(reach_spec, 1.0)
        resdcayf_up = _get_decay(res_spec, 1.0)
        n_src_basic = jsrcvar.size + 1
        upload = np.zeros((nreach, n_src_basic))
        nodes = np.zeros((nnode + 1, n_src_basic))
        carryf_up = data[:, jfrac] * rchdcayf_up * resdcayf_up

        for i in range(nreach):
            fnode = int(data[i, jfnode]) if jfnode >= 0 else 0
            tnode = int(data[i, jtnode]) if jtnode >= 0 else 0
            dep_val = data[i, jdepvar] if jdepvar >= 0 else np.nan
            if np.isnan(dep_val):
                upload[i, :] = carryf_up[i] * nodes[fnode, :]
                nodes[tnode, :] = nodes[tnode, :] + data[i, jiftran] * upload[i, :]
            else:
                nodes[tnode, :] = nodes[tnode, :] + data[i, jiftran] * predict_values[
                    i, :n_src_basic
                ]

        up_cols: list[np.ndarray] = []
        up_names: list[str] = []
        if waterid:
            up_cols.append(
                data[:, jwaterid] if jwaterid >= 0 else np.full((nreach,), np.nan)
            )
            up_names.append(waterid)
        up_cols.append(upload)
        up_names.extend([f"umpload_{name}" for name in ["total", *srcvar]])
        upmonload_df = pd.DataFrame(np.column_stack(up_cols), columns=up_names)

    test_data_df: pd.DataFrame | None = None
    if test_rows:
        raw = np.vstack(test_rows)
        n_data = len(datalst)
        src_basic = ["total", *srcvar]
        n_src_basic = len(src_basic)

        idx0 = n_data
        rchdcay_col = raw[:, idx0]
        resdcay_col = raw[:, idx0 + 1]
        idx1 = idx0 + 2
        inc_block = raw[:, idx1 : idx1 + n_src_basic]
        inc_nd_total = raw[:, idx1 + n_src_basic]
        idx2 = idx1 + n_src_basic + 1
        node_block = raw[:, idx2 : idx2 + n_src_basic]
        node_nd_total = raw[:, idx2 + n_src_basic]

        out: dict[str, np.ndarray] = {}
        for i_col, col in enumerate(datalst):
            out[col] = raw[:, i_col]
        out["rchdcayf"] = rchdcay_col
        out["resdcayf"] = resdcay_col
        for i_col, name in enumerate(src_basic):
            out[f"incddsrc_{name}"] = inc_block[:, i_col]
        out["nd_incddsrc_total"] = inc_nd_total
        for name in srcvar:
            out[f"nd_incddsrc_{name}"] = np.full((raw.shape[0],), np.nan)
        out["incddsrc_res_loss"] = np.full((raw.shape[0],), np.nan)
        for i_col, name in enumerate(src_basic):
            out[f"node_{name}"] = node_block[:, i_col]
        out["nd_node_total"] = node_nd_total
        for name in srcvar:
            out[f"nd_node_{name}"] = np.full((raw.shape[0],), np.nan)
        out["node_res_loss"] = np.full((raw.shape[0],), np.nan)
        # Compatibility aliases for the current compact ND layout.
        out["incddsrc_total_nd"] = inc_nd_total
        out["node_total_nd"] = node_nd_total

        test_data_df = pd.DataFrame(out)

    return PredictResult(
        predict=predict_df,
        predlst=predlst,
        predlst_str=predlst_str,
        upmonload=upmonload_df,
        test_data=test_data_df,
        debug_trace=debug_trace_df,
        debug_del_frac_chain=debug_del_frac_chain_df,
        warnings=warnings,
        errors=errors,
    )
