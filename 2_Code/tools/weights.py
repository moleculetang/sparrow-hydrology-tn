"""SAS estimate_with_weights.sas replacement helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Callable, Mapping, MutableMapping

import numpy as np
import pandas as pd

from ..cli import FileTableStore


@dataclass
class EstimateWeightsResult:
    ls_weight_betaest: pd.DataFrame | None
    ls_weights: pd.DataFrame | None
    negatives: pd.DataFrame | None
    updated_config: dict[str, object] | None
    updated_indata: pd.DataFrame | None
    run_result: object | None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


_SAS_MISSING_SENTINEL = -np.inf


def estimate_with_weights(
    config: MutableMapping[str, object],
    *,
    resids: pd.DataFrame | None = None,
    results_store: FileTableStore | None = None,
    data_store: FileTableStore | None = None,
    run_main: Callable[[MutableMapping[str, object]], object] | None = None,
) -> EstimateWeightsResult:
    """Estimate heteroscedastic weights from residuals and optionally rerun SPARROW."""
    warnings: list[str] = []
    errors: list[str] = []

    def warn(msg: str) -> None:
        warnings.append(msg)

    def error(msg: str) -> None:
        errors.append(msg)
        config["if_error"] = "yes"

    if "if_error" not in config:
        config["if_error"] = "no"

    home_results = _first_path(config, "home_results", "results_dir")
    home_data = _first_path(config, "home_data", "input_dir")

    if results_store is None:
        results_store = FileTableStore(home_results, config.get("results_tables"))
    if data_store is None:
        data_store = FileTableStore(home_data, config.get("input_tables"))

    if resids is None:
        if results_store.exists("resids"):
            resids = results_store.read("resids")
        else:
            error("resids table not found - stop processing.")

    if resids is None:
        return EstimateWeightsResult(
            None, None, None, None, None, None, warnings, errors
        )

    resid2_transform = str(config.get("resid2_transform") or "").strip()
    if resid2_transform:
        upper = resid2_transform.upper()
        if upper not in {"LOG", "ABS"}:
            error(
                "The resid2_transform control variable must be either LOG or ABS, "
                "or leave blank for no transformation"
            )
        elif upper == "LOG":
            warn("Residual variance regression uses the natural log transform")
        else:
            warn("Residual variance regression uses the absolute value transform")

    if config.get("if_error") == "yes":
        return EstimateWeightsResult(
            None, None, None, None, None, None, warnings, errors
        )

    df = resids.copy()
    if "ln_resid" not in df.columns:
        error("resids does not contain ln_resid - stop processing.")
        return EstimateWeightsResult(
            None, None, None, None, None, None, warnings, errors
        )

    df["resid2"] = df["ln_resid"] * df["ln_resid"]
    if resid2_transform:
        if resid2_transform.upper() == "LOG":
            resid2 = df["resid2"].to_numpy(dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                log_resid2 = np.log(resid2)
            log_resid2[~(resid2 > 0)] = np.nan
            df["resid2"] = log_resid2
        elif resid2_transform.upper() == "ABS":
            df["resid2"] = np.abs(df["resid2"])

    specify_weight_predictors = config.get("specify_weight_predictors")
    if specify_weight_predictors:
        df = _apply_weight_predictors(df, specify_weight_predictors)

    weight_predictors = _split_tokens(config.get("weight_predictors"))
    missing = [name for name in weight_predictors if name and name not in df.columns]
    if missing:
        error(
            "The following weight predictors were not found in resids: "
            + " ".join(missing)
            + "."
        )
        return EstimateWeightsResult(
            None, None, None, None, None, None, warnings, errors
        )

    y = df["resid2"].to_numpy(dtype=float)
    x = df.loc[:, weight_predictors].to_numpy(dtype=float) if weight_predictors else None

    beta, pred, resid, lev, rmse = _ols_with_leverage(y, x)

    weight_name = str(config.get("weight_name") or "").strip()
    if not weight_name:
        error("weight_name is required to output weights.")
        return EstimateWeightsResult(
            None, None, None, None, None, None, warnings, errors
        )

    reg_out = df.copy()
    reg_out[weight_name] = pred
    reg_out["r2resid"] = resid
    reg_out["lev"] = lev

    if resid2_transform.upper() == "LOG":
        reg_out[weight_name] = np.exp(reg_out[weight_name] + rmse * reg_out["lev"] / 2)
    elif resid2_transform.upper() == "ABS":
        reg_out[weight_name] = reg_out[weight_name] ** 2 - rmse * reg_out["lev"]

    reg_out[weight_name] = 1.0 / reg_out[weight_name]

    negatives = reg_out[reg_out[weight_name] < 0].copy()
    if not negatives.empty:
        warn(
            f"{len(negatives)} negative weights have been set to zero - see negatives output."
        )
        reg_out.loc[reg_out[weight_name] < 0, weight_name] = 0

    waterid = _first_token(config.get("waterid"))
    if not waterid or waterid not in reg_out.columns:
        error("waterid column is missing from resids - stop processing.")
        return EstimateWeightsResult(
            None, None, negatives, None, None, None, warnings, errors
        )

    ls_weights = reg_out.loc[:, [waterid, weight_name]].copy()
    ls_weights = ls_weights.sort_values(by=[waterid], kind="stable", na_position="first")

    ls_weight_betaest = _format_betaest(weight_predictors, beta, rmse)

    results_store.write("LS_weight_betaest", ls_weight_betaest)
    results_store.write("ls_weights", ls_weights)
    if not negatives.empty:
        results_store.write("negatives", negatives)

    updated_config = None
    updated_indata = None
    run_result = None

    if _is_yes(config.get("ifrunSPARROW")):
        updated_config = dict(config)
        ls_weight = _first_token(updated_config.get("ls_weight"))
        if ls_weight:
            updated_config["data_modifications"] = _append_data_modifications(
                updated_config.get("data_modifications"), f"{ls_weight} = {weight_name}"
            )
        indata = _read_indata(updated_config, data_store)
        if indata is None:
            error("indata could not be loaded for merging weights.")
        else:
            updated_indata = _merge_weights(indata, ls_weights, waterid)
            data_store.write(_as_table_name(updated_config.get("indata")), updated_indata)
            if updated_config.get("if_error") != "yes":
                runner = run_main
                if runner is None:
                    from ..cli.runner import run_main as default_run_main

                    runner = default_run_main
                run_result = runner(updated_config)

    return EstimateWeightsResult(
        ls_weight_betaest=ls_weight_betaest,
        ls_weights=ls_weights,
        negatives=negatives if not negatives.empty else None,
        updated_config=updated_config,
        updated_indata=updated_indata,
        run_result=run_result,
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


def _first_token(value: object) -> str:
    tokens = _split_tokens(value)
    return tokens[0] if tokens else ""


def _is_yes(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().upper() == "YES"


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


def _merge_weights(
    indata: pd.DataFrame, ls_weights: pd.DataFrame, waterid: str
) -> pd.DataFrame:
    indata_sorted = indata.sort_values(by=[waterid], kind="stable", na_position="first")
    weights_sorted = ls_weights.sort_values(by=[waterid], kind="stable", na_position="first")
    merged = indata_sorted.merge(weights_sorted, on=waterid, how="left", sort=False)
    return merged


def _append_data_modifications(existing: object, extra_assignment: str) -> str:
    extra = extra_assignment.strip().rstrip(";")
    if existing is None:
        return f"{extra} ;"
    if isinstance(existing, str):
        text = existing.strip()
        if not text.endswith(";"):
            text = text + " ;"
        return f"{text} {extra} ;"
    return f"{extra} ;"


def _ols_with_leverage(
    y: np.ndarray, x: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    n = y.shape[0]
    if x is None or x.size == 0:
        X = np.ones((n, 1))
    else:
        X = np.column_stack([np.ones((n, 1)), x])

    valid = np.isfinite(y)
    if x is not None and x.size:
        valid &= np.isfinite(x).all(axis=1)
    yv = y[valid]
    Xv = X[valid, :]

    beta = np.full((X.shape[1],), np.nan)
    pred = np.full((n,), np.nan)
    resid = np.full((n,), np.nan)
    lev = np.full((n,), np.nan)

    if Xv.size and yv.size:
        xtx = Xv.T @ Xv
        xtx_inv = np.linalg.pinv(xtx)
        beta = xtx_inv @ Xv.T @ yv
        yhat = Xv @ beta
        resid_v = yv - yhat
        pred[valid] = yhat
        resid[valid] = resid_v
        h = np.sum((Xv @ xtx_inv) * Xv, axis=1)
        lev[valid] = h

        df_error = max(yv.shape[0] - Xv.shape[1], 1)
        rmse = float(np.sqrt(np.sum(resid_v**2) / df_error))
    else:
        rmse = float("nan")

    return beta, pred, resid, lev, rmse


def _format_betaest(
    predictors: list[str], beta: np.ndarray, rmse: float
) -> pd.DataFrame:
    columns = ["Intercept", *predictors, "_RMSE_"]
    values = np.concatenate([beta, [rmse]])
    return pd.DataFrame([values], columns=columns)


def _normalize_condition_expr(expr: str) -> str:
    expr = expr.strip()
    expr = re.sub(r"(?<![\w.])\.(?![\w.])", "_MISSING_", expr)
    expr = re.sub(r"\bne\b", "!=", expr, flags=re.IGNORECASE)
    expr = re.sub(r"\beq\b", "==", expr, flags=re.IGNORECASE)
    expr = re.sub(r"\bgt\b", ">", expr, flags=re.IGNORECASE)
    expr = re.sub(r"\bge\b", ">=", expr, flags=re.IGNORECASE)
    expr = re.sub(r"\blt\b", "<", expr, flags=re.IGNORECASE)
    expr = re.sub(r"\ble\b", "<=", expr, flags=re.IGNORECASE)
    expr = expr.replace("^=", "!=")
    expr = expr.replace("<>", "!=")
    expr = re.sub(r"(?<![<>=!])=(?!=)", "==", expr)
    expr = re.sub(r"\band\b", "&", expr, flags=re.IGNORECASE)
    expr = re.sub(r"\bor\b", "|", expr, flags=re.IGNORECASE)
    expr = re.sub(r"\bnot\b", "~", expr, flags=re.IGNORECASE)
    return expr


def _normalize_value_expr(expr: str) -> str:
    expr = expr.strip()
    expr = re.sub(r"(?<![\w.])\.(?![\w.])", "nan", expr)
    return expr


def _eval_expr(
    expr: str, df: pd.DataFrame, *, for_condition: bool
) -> pd.Series | np.ndarray | float | int:
    if not expr:
        return np.nan

    if for_condition and re.fullmatch(r"[A-Za-z_]\w*", expr):
        series = df[expr]
        return series.fillna(0) != 0

    if for_condition:
        expr = _normalize_condition_expr(expr)
        temp = df.copy()
        numeric_cols = temp.select_dtypes(include=[np.number]).columns
        temp[numeric_cols] = temp[numeric_cols].fillna(_SAS_MISSING_SENTINEL)
        local_dict: dict[str, object] = {col: temp[col] for col in temp.columns}
        local_dict.update(
            {
                "_MISSING_": _SAS_MISSING_SENTINEL,
                "abs": np.abs,
                "sqrt": np.sqrt,
                "log": np.log,
                "log10": np.log10,
                "exp": np.exp,
                "floor": np.floor,
                "ceil": np.ceil,
                "int": np.int64,
            }
        )
        return pd.eval(expr, local_dict=local_dict, engine="python")

    expr = _normalize_value_expr(expr)
    local_dict = {col: df[col] for col in df.columns}
    local_dict.update(
        {
            "nan": np.nan,
            "abs": np.abs,
            "sqrt": np.sqrt,
            "log": np.log,
            "log10": np.log10,
            "exp": np.exp,
            "floor": np.floor,
            "ceil": np.ceil,
            "int": np.int64,
        }
    )
    return pd.eval(expr, local_dict=local_dict, engine="python")


def _split_data_modifications(script: str) -> list[str]:
    parts = script.split(";")
    statements: list[str] = []
    for part in parts:
        text = part.strip()
        if not text:
            continue
        if text.lower().startswith("else") and statements:
            statements[-1] = f"{statements[-1]} ; {text}"
        else:
            statements.append(text)
    return statements


def _parse_assignment(statement: str) -> tuple[str, str]:
    match = re.match(r"^\s*([A-Za-z_]\w*)\s*=\s*(.+)$", statement)
    if not match:
        raise ValueError(f"Unsupported assignment syntax: {statement}")
    return match.group(1), match.group(2).strip()


def _parse_if_chain(statement: str) -> list[tuple[str | None, tuple[str, str]]]:
    match = re.match(r"^if\s+(.*?)\s+then\s+(.*)$", statement, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Unsupported IF syntax: {statement}")

    chain: list[tuple[str | None, tuple[str, str]]] = []
    cond = match.group(1).strip()
    remainder = match.group(2).strip()

    parts = re.split(r"\s+else\s+", remainder, flags=re.IGNORECASE)
    first_assignment = parts[0].strip().strip(";")
    chain.append((cond, _parse_assignment(first_assignment)))

    for part in parts[1:]:
        segment = part.strip().strip(";")
        if re.match(r"^if\s+", segment, flags=re.IGNORECASE):
            sub_match = re.match(
                r"^if\s+(.*?)\s+then\s+(.*)$", segment, flags=re.IGNORECASE
            )
            if not sub_match:
                raise ValueError(f"Unsupported ELSE IF syntax: {segment}")
            sub_cond = sub_match.group(1).strip()
            sub_assign = _parse_assignment(sub_match.group(2).strip())
            chain.append((sub_cond, sub_assign))
        else:
            chain.append((None, _parse_assignment(segment)))

    return chain


def _apply_weight_predictors(
    df: pd.DataFrame, script: object
) -> pd.DataFrame:
    if script is None:
        return df
    if callable(script):
        return script(df)
    if isinstance(script, (list, tuple)):
        out = df
        for item in script:
            if callable(item):
                out = item(out)
            elif isinstance(item, Mapping):
                out = _apply_ops(out, [item])
            else:
                raise ValueError("Unsupported specify_weight_predictors entry type")
        return out
    if isinstance(script, Mapping):
        return _apply_ops(df, [script])
    if not isinstance(script, str):
        raise ValueError("Unsupported specify_weight_predictors type")

    out = df.copy()
    for statement in _split_data_modifications(script):
        if statement.lower().startswith("if "):
            chain = _parse_if_chain(statement)
            remaining = pd.Series(True, index=out.index)
            for cond, (var, expr) in chain:
                if cond is None:
                    mask = remaining
                else:
                    mask = remaining & pd.Series(
                        _eval_expr(cond, out, for_condition=True), index=out.index
                    )
                if not mask.any():
                    remaining = remaining & ~mask
                    continue
                values = _eval_expr(expr, out, for_condition=False)
                _assign_values(out, var, mask, values)
                remaining = remaining & ~mask
        else:
            var, expr = _parse_assignment(statement)
            values = _eval_expr(expr, out, for_condition=False)
            out[var] = values if not np.isscalar(values) else values

    return out


def _assign_values(
    df: pd.DataFrame,
    column: str,
    mask: pd.Series,
    values: pd.Series | np.ndarray | float | int,
) -> None:
    if np.isscalar(values):
        df.loc[mask, column] = values
        return

    if isinstance(values, pd.Series):
        aligned = values.reindex(df.index)
    else:
        aligned = pd.Series(values, index=df.index)
    df.loc[mask, column] = aligned.loc[mask].to_numpy()


def _apply_ops(df: pd.DataFrame, ops: list[Mapping[str, object]]) -> pd.DataFrame:
    out = df.copy()
    for op in ops:
        when = op.get("when")
        assignments = op.get("set") or {}
        if not isinstance(assignments, Mapping):
            raise ValueError("specify_weight_predictors set must be a mapping")
        else_assignments = op.get("else") or {}
        if when:
            mask = pd.Series(_eval_expr(str(when), out, for_condition=True), index=out.index)
        else:
            mask = pd.Series(True, index=out.index)
        for col, expr in assignments.items():
            values = _eval_expr(str(expr), out, for_condition=False)
            _assign_values(out, str(col), mask, values)
        if else_assignments:
            else_mask = ~mask
            for col, expr in else_assignments.items():
                values = _eval_expr(str(expr), out, for_condition=False)
                _assign_values(out, str(col), else_mask, values)
    return out
