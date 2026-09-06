"""SAS sparrow_setdata.sas replacement helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import floor
import re
from statistics import NormalDist
from typing import Callable, Iterable, Mapping, MutableMapping, Sequence

import numpy as np
import pandas as pd


@dataclass
class SetDataResult:
    indata: pd.DataFrame
    station_data: pd.DataFrame
    ancillary: pd.DataFrame
    fnodes: pd.DataFrame
    tnodes: pd.DataFrame
    labels: dict[str, str]
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class TableStore:
    read: Callable[[str], pd.DataFrame]
    write: Callable[[str, pd.DataFrame], None]
    exists: Callable[[str], bool]


_SAS_MISSING_SENTINEL = -np.inf
_NORMAL_DIST = NormalDist()


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


def _ensure_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col and col not in out.columns:
            out[col] = np.nan
    return out.loc[:, [col for col in columns if col]]


def _check_list(list_a: Iterable[str], list_b: Iterable[str]) -> list[str]:
    list_b_upper = {item.upper() for item in list_b if item}
    excluded: list[str] = []
    for item in list_a:
        if not item:
            continue
        if item.upper() not in list_b_upper:
            excluded.append(item)
    return excluded


def _exclude_list(list1: Iterable[str], list2: Iterable[str]) -> list[str]:
    list1_upper = {item.upper() for item in list1 if item}
    return [item for item in list2 if item and item.upper() not in list1_upper]


def ds_var_list(df: pd.DataFrame, var: str, n: int) -> list[object]:
    if df is None or not var or var not in df.columns:
        return []
    return df[var].iloc[:n].tolist()


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
    binary_term = r"(?:\([^()]+\)|[A-Za-z_][\w\[\],:]*|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
    pattern_max = re.compile(rf"(?P<a>{binary_term})\s*<>\s*(?P<b>{binary_term})")
    pattern_min = re.compile(rf"(?P<a>{binary_term})\s*><\s*(?P<b>{binary_term})")
    while pattern_max.search(expr):
        expr = pattern_max.sub(r"__sas_max(\g<a>, \g<b>)", expr)
    while pattern_min.search(expr):
        expr = pattern_min.sub(r"__sas_min(\g<a>, \g<b>)", expr)
    return expr


def _norm_ppf(x: float | np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    arr = np.clip(arr, np.finfo(float).tiny, 1 - np.finfo(float).eps)
    return np.vectorize(_NORMAL_DIST.inv_cdf)(arr).astype(float)


def _eval_expr(expr: str, df: pd.DataFrame, *, for_condition: bool) -> pd.Series | np.ndarray | float | int:
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
                "exp": np.exp,
                "floor": np.floor,
                "ceil": np.ceil,
                "int": np.int64,
                "choose": lambda cond, if_true, if_false: np.where(cond, if_true, if_false),
                "probit": _norm_ppf,
                "__sas_max": np.maximum,
                "__sas_min": np.minimum,
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
            "exp": np.exp,
            "floor": np.floor,
            "ceil": np.ceil,
            "int": np.int64,
            "choose": lambda cond, if_true, if_false: np.where(cond, if_true, if_false),
            "probit": _norm_ppf,
            "__sas_max": np.maximum,
            "__sas_min": np.minimum,
        }
    )
    return pd.eval(expr, local_dict=local_dict, engine="python")


def _strip_sas_comments(script: str) -> str:
    return re.sub(r"/\*.*?\*/", " ", script, flags=re.DOTALL)


def _split_data_modifications(script: str) -> list[str]:
    script = _strip_sas_comments(script)
    parts = script.split(";")
    statements: list[str] = []
    for part in parts:
        text = part.strip()
        if not text:
            continue
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
            sub_match = re.match(r"^if\s+(.*?)\s+then\s+(.*)$", segment, flags=re.IGNORECASE)
            if not sub_match:
                raise ValueError(f"Unsupported ELSE IF syntax: {segment}")
            sub_cond = sub_match.group(1).strip()
            sub_assign = _parse_assignment(sub_match.group(2).strip())
            chain.append((sub_cond, sub_assign))
        else:
            chain.append((None, _parse_assignment(segment)))

    return chain


def _collect_block_assignments(
    statements: list[str], start_idx: int
) -> tuple[list[tuple[str, str]], int]:
    assigns: list[tuple[str, str]] = []
    i = start_idx
    while i < len(statements):
        token = statements[i].strip()
        if re.fullmatch(r"end", token, flags=re.IGNORECASE):
            return assigns, i + 1
        assigns.append(_parse_assignment(token))
        i += 1
    return assigns, i


def _apply_conditional_assignments(
    df: pd.DataFrame,
    remaining: pd.Series,
    cond: str | None,
    assigns: list[tuple[str, str]],
) -> pd.Series:
    if cond is None:
        mask = remaining
    else:
        mask = remaining & pd.Series(_eval_expr(cond, df, for_condition=True), index=df.index)
    if not mask.any():
        return remaining
    for var, expr in assigns:
        values = _eval_expr(expr, df, for_condition=False)
        _assign_values(df, var, mask, values)
    return remaining & ~mask


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


def apply_data_modifications(
    df: pd.DataFrame, config: Mapping[str, object]
) -> pd.DataFrame:
    script = config.get("data_modifications")
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
                raise ValueError("Unsupported data_modifications entry type")
        return out

    if isinstance(script, Mapping):
        return _apply_ops(df, [script])

    if not isinstance(script, str):
        raise ValueError("Unsupported data_modifications type")

    out = df.copy()
    statements = _split_data_modifications(script)
    i = 0
    while i < len(statements):
        statement = statements[i].strip()
        low = statement.lower()
        if re.match(r"^if\s+.*\s+then\s+do$", statement, flags=re.IGNORECASE):
            m = re.match(r"^if\s+(.*?)\s+then\s+do$", statement, flags=re.IGNORECASE)
            if not m:
                raise ValueError(f"Unsupported IF DO syntax: {statement}")
            remaining = pd.Series(True, index=out.index)
            cond = m.group(1).strip()
            true_assigns, i = _collect_block_assignments(statements, i + 1)
            remaining = _apply_conditional_assignments(out, remaining, cond, true_assigns)

            while i < len(statements):
                token = statements[i].strip()
                if re.match(r"^else\s+if\s+.*\s+then\s+do$", token, flags=re.IGNORECASE):
                    m = re.match(r"^else\s+if\s+(.*?)\s+then\s+do$", token, flags=re.IGNORECASE)
                    if not m:
                        raise ValueError(f"Unsupported ELSE IF DO syntax: {token}")
                    cond = m.group(1).strip()
                    assigns, i = _collect_block_assignments(statements, i + 1)
                    remaining = _apply_conditional_assignments(out, remaining, cond, assigns)
                    continue
                if re.match(r"^else\s+do$", token, flags=re.IGNORECASE):
                    assigns, i = _collect_block_assignments(statements, i + 1)
                    remaining = _apply_conditional_assignments(out, remaining, None, assigns)
                    continue
                if re.match(r"^else\s+if\s+", token, flags=re.IGNORECASE):
                    m = re.match(r"^else\s+if\s+(.*?)\s+then\s+(.*)$", token, flags=re.IGNORECASE)
                    if not m:
                        raise ValueError(f"Unsupported ELSE IF syntax: {token}")
                    cond = m.group(1).strip()
                    assign = _parse_assignment(m.group(2).strip())
                    remaining = _apply_conditional_assignments(out, remaining, cond, [assign])
                    i += 1
                    continue
                if re.match(r"^else\s+", token, flags=re.IGNORECASE):
                    assign = _parse_assignment(re.sub(r"^else\s+", "", token, flags=re.IGNORECASE))
                    remaining = _apply_conditional_assignments(out, remaining, None, [assign])
                    i += 1
                    continue
                break
            continue

        if low.startswith("if "):
            chain = _parse_if_chain(statement)
            remaining = pd.Series(True, index=out.index)
            for cond, (var, expr) in chain:
                remaining = _apply_conditional_assignments(out, remaining, cond, [(var, expr)])
            i += 1
            continue

        var, expr = _parse_assignment(statement)
        values = _eval_expr(expr, out, for_condition=False)
        out[var] = values if not np.isscalar(values) else values
        i += 1

    return out


def _apply_ops(df: pd.DataFrame, ops: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    out = df.copy()
    for op in ops:
        when = op.get("when")
        assignments = op.get("set") or {}
        if not isinstance(assignments, Mapping):
            raise ValueError("data_modifications set must be a mapping")
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


def setdata(indata: pd.DataFrame, config: MutableMapping[str, object]) -> SetDataResult:
    warnings: list[str] = []
    errors: list[str] = []

    def warn(msg: str) -> None:
        warnings.append(msg)

    def error(msg: str) -> None:
        errors.append(msg)
        config["if_error"] = "yes"

    if "if_error" not in config:
        config["if_error"] = "no"

    df = indata.copy()
    df = apply_data_modifications(df, config)

    depvar = _first_token(config.get("depvar"))
    ls_weight = _first_token(config.get("ls_weight"))
    if depvar and ls_weight and depvar in df.columns:
        dep = df[depvar]
        df.loc[dep.isna() | (dep <= 0), ls_weight] = np.nan

    if _is_yes(config.get("if_distribute_yield_by_land_use")):
        land_class_list = _split_tokens(config.get("land_class_list"))
        if land_class_list:
            total = land_class_list[0]
            df["LU_class"] = np.nan
            i_var = 1
            n_entries = len(land_class_list)
            while i_var + 2 < n_entries:
                class_var = land_class_list[i_var]
                cond_var = land_class_list[i_var + 1]
                pct_var = land_class_list[i_var + 2]
                try:
                    pct_val = float(pct_var)
                except ValueError:
                    pct_val = np.nan
                if total in df.columns and cond_var in df.columns:
                    threshold = df[total] * pct_val / 100.0
                    threshold = threshold.fillna(_SAS_MISSING_SENTINEL)
                    mask = df["LU_class"].isna() & (df[cond_var] > threshold)
                    df.loc[mask, "LU_class"] = class_var
                i_var += 3

    indata_list = config.get("indata_list")
    if not indata_list:
        indata_list = _split_tokens(config.get("indata_list"))
    indata_keep = _split_tokens(indata_list)

    staid = _first_token(config.get("staid"))
    arcid = _first_token(config.get("arcid"))
    optional_station_information = _split_tokens(config.get("optional_station_information"))
    waterid = _first_token(config.get("waterid"))
    lat = _first_token(config.get("lat"))
    lon = _first_token(config.get("lon"))

    station_keep = [
        staid,
        arcid,
        *optional_station_information,
        waterid,
        lat,
        lon,
        ls_weight,
    ]

    optional_reach_information = _split_tokens(config.get("optional_reach_information"))
    fnode = _first_token(config.get("fnode"))
    tnode = _first_token(config.get("tnode"))
    hydseq = _first_token(config.get("hydseq"))
    inc_area = _first_token(config.get("inc_area"))
    tot_area = _first_token(config.get("tot_area"))
    mean_flow = _first_token(config.get("mean_flow"))
    frac = _first_token(config.get("frac"))
    iftran = _first_token(config.get("iftran"))
    target = _first_token(config.get("target"))

    ancillary_keep = [
        arcid,
        waterid,
        *optional_reach_information,
        fnode,
        tnode,
        hydseq,
        inc_area,
        tot_area,
        mean_flow,
        frac,
        iftran,
        target,
        ls_weight,
    ]
    if _is_yes(config.get("if_distribute_yield_by_land_use")):
        ancillary_keep.append("LU_class")

    tnodes_keep = [waterid, tnode, hydseq]
    fnodes_keep = [waterid, fnode, hydseq, frac]

    indata_out = _ensure_columns(df, indata_keep)
    ancillary_out = _ensure_columns(df, ancillary_keep)

    if depvar and depvar in df.columns:
        station_mask = df[depvar].notna()
    else:
        station_mask = pd.Series(False, index=df.index)
    station_out = _ensure_columns(df.loc[station_mask], station_keep)

    if fnode and fnode in df.columns:
        node_mask = df[fnode] > 0
    else:
        node_mask = pd.Series(False, index=df.index)
    fnodes_out = _ensure_columns(df.loc[node_mask], fnodes_keep)
    tnodes_out = _ensure_columns(df.loc[node_mask], tnodes_keep)

    labels = _build_labels(config)

    return SetDataResult(
        indata=indata_out,
        station_data=station_out,
        ancillary=ancillary_out,
        fnodes=fnodes_out,
        tnodes=tnodes_out,
        labels=labels,
        warnings=warnings,
        errors=errors,
    )


def sort_data(
    indata: pd.DataFrame, station_data: pd.DataFrame, ancillary: pd.DataFrame, config: Mapping[str, object]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    hydseq = _first_token(config.get("hydseq"))
    waterid = _first_token(config.get("waterid"))
    staid = _first_token(config.get("staid"))

    if hydseq:
        indata_sorted = indata.sort_values(by=[hydseq], kind="stable", na_position="first")
    else:
        indata_sorted = indata.copy()

    if waterid:
        ancillary_sorted = ancillary.sort_values(by=[waterid], kind="stable", na_position="first")
    else:
        ancillary_sorted = ancillary.copy()

    if staid:
        station_sorted = station_data.sort_values(by=[staid], kind="stable", na_position="first")
    else:
        station_sorted = station_data.copy()

    return indata_sorted.reset_index(drop=True), station_sorted.reset_index(drop=True), ancillary_sorted.reset_index(drop=True)


def check_vars(df: pd.DataFrame, search_vars: Iterable[str]) -> list[str]:
    file_vars = " ".join(df.columns)
    file_vars = file_vars.upper()
    not_found: list[str] = []
    for var in search_vars:
        if not var:
            continue
        if file_vars.find(var.upper()) < 0:
            not_found.append(var)
    return not_found


def check_missing_preproc(df: pd.DataFrame, check_vars: Iterable[str]) -> pd.DataFrame:
    flags = []
    cols = []
    for var in check_vars:
        if not var:
            continue
        cols.append(var)
        if var in df.columns:
            flags.append(int(df[var].isna().any()))
        else:
            flags.append(0)
    return pd.DataFrame([flags], columns=cols)


def check_missing_postproc(df: pd.DataFrame, check_vars: Iterable[str]) -> list[str]:
    missing: list[str] = []
    if df.empty:
        return missing
    row = df.iloc[0]
    for var in check_vars:
        if not var:
            continue
        if var in row.index and row[var] == 1:
            missing.append(var)
    return missing


def check_numeric(
    indata: pd.DataFrame,
    config: Mapping[str, object],
    if_macro_nms: bool,
    varnms: Iterable[str],
) -> list[str]:
    errors: list[str] = []
    for token in varnms:
        if not token:
            continue
        if if_macro_nms:
            varnm = _first_token(config.get(token))
            qualifier = f"Variable {varnm} associated with control variable {token}"
        else:
            varnm = token
            qualifier = f"Model variable {varnm}"
        if varnm and varnm in indata.columns:
            if not pd.api.types.is_numeric_dtype(indata[varnm]):
                errors.append(
                    f"{qualifier} is stored as character but should be numeric - stop processing."
                )
    return errors


def check_model_vars(
    indata: pd.DataFrame, config: MutableMapping[str, object]
) -> tuple[list[str], list[str]]:
    warnings: list[str] = []
    errors: list[str] = []

    model_vars_list = _split_tokens(config.get("model_vars_list"))
    vars_not_in_indata = check_vars(indata, model_vars_list)
    vars_in_indata = _check_list(model_vars_list, vars_not_in_indata)

    if vars_not_in_indata:
        config["if_error"] = "yes"
        errors.append(
            "The following model variables were not found: "
            + " ".join(vars_not_in_indata)
            + ". Add to the input data set or specify in a data_modifications statement."
        )

    if vars_in_indata:
        check_df = check_missing_preproc(indata, vars_in_indata)
        missing_vars = check_missing_postproc(check_df, vars_in_indata)
        if missing_vars:
            config["if_error"] = "yes"
            errors.append(
                "The following model variables require a data_modifications statement to remove missing values: "
                + " ".join(missing_vars)
                + "."
            )

    numeric_list = [
        "depvar",
        "ls_weight",
        "staid",
        "lat",
        "lon",
        "waterid",
        "mean_flow",
        "arcid",
        "inc_area",
        "tot_area",
        "fnode",
        "tnode",
        "hydseq",
        "frac",
        "iftran",
        "target",
    ]
    errors.extend(check_numeric(indata, config, True, numeric_list))

    num_vars_list: list[str] = []
    for token in numeric_list:
        num_vars_list.extend(_split_tokens(config.get(token)))

    check_not_num_list = _exclude_list(num_vars_list, model_vars_list)
    errors.extend(check_numeric(indata, config, False, check_not_num_list))

    if errors:
        config["if_error"] = "yes"

    return warnings, errors


def check_station_vars(
    station_data: pd.DataFrame, config: MutableMapping[str, object]
) -> tuple[list[str], list[str]]:
    warnings: list[str] = []
    errors: list[str] = []

    staid = _first_token(config.get("staid"))
    ls_weight = _first_token(config.get("ls_weight"))
    lat = _first_token(config.get("lat"))
    lon = _first_token(config.get("lon"))

    count_staid = int(station_data[staid].isna().sum()) if staid in station_data.columns else 0
    count_ls_weight = int(station_data[ls_weight].isna().sum()) if ls_weight in station_data.columns else 0
    if lat and lon and lat in station_data.columns and lon in station_data.columns:
        count_lat_lon = int((station_data[lat].isna() | station_data[lon].isna()).sum())
    else:
        count_lat_lon = 0

    if count_staid > 0:
        config["if_error"] = "yes"
        errors.append(
            f"Reaches ({count_staid}) with monitored flux are missing a value for the station identifier {staid} - stop processing."
        )
    if count_ls_weight > 0:
        config["if_error"] = "yes"
        errors.append(
            f"Reaches ({count_ls_weight}) with monitored flux are missing a value for the observation weight variable {ls_weight} - stop processing."
        )
    if count_lat_lon > 0:
        warnings.append(
            f"Reaches ({count_lat_lon}) with monitored flux do not have a valid lat/lon and will not be represented in the map of station residuals."
        )

    return warnings, errors


def check_network(
    fnodes: pd.DataFrame, tnodes: pd.DataFrame, config: Mapping[str, object]
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    warnings: list[str] = []
    frac = _first_token(config.get("frac"))
    hydseq = _first_token(config.get("hydseq"))
    fnode = _first_token(config.get("fnode"))
    tnode = _first_token(config.get("tnode"))

    fnodes_sorted = (
        fnodes.sort_values(by=[fnode], kind="stable", na_position="first")
        if fnode and fnode in fnodes.columns
        else fnodes.copy()
    )
    tnodes_sorted = (
        tnodes.sort_values(by=[tnode], kind="stable", na_position="first")
        if tnode and tnode in tnodes.columns
        else tnodes.copy()
    )

    if fnode and frac:
        agg_spec: dict[str, tuple[str, str]] = {
            "sumdivfrac": (frac, "sum"),
            "nreach": (frac, "size"),
        }
        if hydseq:
            agg_spec["mindnhydseq"] = (hydseq, "min")
        sums = fnodes_sorted.groupby(fnode, dropna=False).agg(**agg_spec)

        tol = 1e-10
        # A single outgoing reach with divfrac < 1 represents a valid withdrawal
        # from the modeled network, not a broken split/merge definition.
        single_reach_withdrawal = (
            (sums["nreach"] == 1)
            & (sums["sumdivfrac"] >= -tol)
            & (sums["sumdivfrac"] < 1 - tol)
        )
        badfrac = sums[
            ((sums["sumdivfrac"] - 1).abs() > tol) & ~single_reach_withdrawal
        ].reset_index()
    else:
        badfrac = pd.DataFrame()

    if not badfrac.empty:
        warnings.append(
            f"The variable {frac} fails to sum to one for {len(badfrac)} fromnodes (see badfrac) - Program not terminated"
        )

    if tnode and hydseq:
        maxup = tnodes_sorted.groupby(tnode, dropna=False)[hydseq].max().rename("maxuphydseq")
        mindn = fnodes_sorted.groupby(fnode, dropna=False)[hydseq].min().rename("mindnhydseq")
        merged = pd.concat([maxup, mindn], axis=1, join="inner").reset_index().rename(columns={tnode: "node"})
        badhydseq = merged[merged["maxuphydseq"] > merged["mindnhydseq"]]
    else:
        badhydseq = pd.DataFrame()

    if not badhydseq.empty:
        warnings.append(
            f"There are {len(badhydseq)} nodes with an incorrect {hydseq} hydrosequence relation (see badhydseq) - Program not terminated"
        )

    return badfrac, badhydseq, warnings


def init_beta(
    config: Mapping[str, object],
    *,
    iter_val: int,
    jter_val: int,
    temp_rc: pd.DataFrame | None = None,
    temp_beta: pd.DataFrame | None = None,
    bak_summary_betaest: pd.DataFrame | None = None,
    summary_betaest: pd.DataFrame | None = None,
) -> pd.DataFrame:
    betalst = _split_tokens(config.get("betalst"))
    bretain = _split_tokens(config.get("bretain"))

    out = None
    if iter_val == 0 and jter_val > 0 and temp_rc is not None:
        rc_vals = ds_var_list(temp_rc, "rc", 1)
        if rc_vals and float(rc_vals[0]) == -8 and temp_beta is not None:
            out = temp_beta.copy()
    if out is None and iter_val == 0 and _is_yes(config.get("if_init_beta_w_previous_est")):
        if bak_summary_betaest is not None:
            out = bak_summary_betaest.copy()
    if out is None and iter_val > 0 and summary_betaest is not None:
        out = summary_betaest.loc[:, betalst].copy() if betalst else summary_betaest.copy()

    if out is None:
        data: dict[str, list[float]] = {}
        if betalst:
            init_map: dict[str, float] = {}
            i = 0
            while i + 1 < len(bretain):
                pname = str(bretain[i]).strip()
                pval = str(bretain[i + 1]).strip()
                if pname:
                    try:
                        init_map[pname] = float(pval)
                    except ValueError:
                        init_map[pname] = np.nan
                i += 2
            data = {name: [float(init_map.get(name, np.nan))] for name in betalst if name}
        else:
            data = {name: [np.nan] for name in bretain if name}
        out = pd.DataFrame(data)

    return out


def set_seeds(
    master_seed: int,
    n_boot_iter: int,
    n_extra_jter: int,
    n_seeds: int,
    *,
    ranuni: Callable[[int], float] | None = None,
) -> pd.DataFrame:
    if ranuni is None:
        raise ValueError("ranuni function is required for SAS-compatible seeding")

    rows = []
    for jter in range(0, n_boot_iter + n_extra_jter + 1):
        row: dict[str, object] = {"jter": jter}
        for i in range(1, n_seeds + 1):
            row[f"seed_{i}"] = floor(1e8 * ranuni(master_seed))
        rows.append(row)
    return pd.DataFrame(rows)


def get_seeds(seeds: pd.DataFrame, jter: int, n_seeds: int) -> dict[str, int]:
    if seeds.empty:
        return {}
    if "jter" in seeds.columns:
        match = seeds.loc[seeds["jter"] == jter]
        if match.empty:
            return {}
        row = match.iloc[0]
    else:
        row = seeds.iloc[jter]
    return {f"seed_{i}": int(row[f"seed_{i}"]) for i in range(1, n_seeds + 1)}


def mean_adjust_delivery_vars(
    indata: pd.DataFrame,
    config: MutableMapping[str, object],
    *,
    store: TableStore | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None, list[str]]:
    errors: list[str] = []
    dlvvar = _split_tokens(config.get("dlvvar"))

    mean_df: pd.DataFrame | None = None
    if _is_yes(config.get("if_estimate")):
        if dlvvar:
            means = indata[dlvvar].mean(skipna=True)
            mean_df = pd.DataFrame([means.values], columns=[f"m_{v}" for v in dlvvar])
        else:
            mean_df = pd.DataFrame()
        config["mean_delivery_vars"] = mean_df
        if store is not None:
            store.write("mean_delivery_vars", mean_df)

    if store is not None and store.exists("mean_delivery_vars"):
        mean_df = store.read("mean_delivery_vars")
    elif "mean_delivery_vars" in config:
        mean_df = config.get("mean_delivery_vars")

    if mean_df is None:
        config["if_error"] = "yes"
        errors.append(
            "Specification requests using previous estimation but data set mean_delivery_vars could not be found in the results directory - stop processing."
        )
        return indata, None, errors

    adjusted = indata.copy()
    for var in dlvvar:
        mean_col = f"m_{var}"
        if mean_col in mean_df.columns and var in adjusted.columns:
            adjusted[var] = adjusted[var] - float(mean_df[mean_col].iloc[0])

    return adjusted, mean_df, errors


def make_nested_area(
    indata: pd.DataFrame,
    station_data: pd.DataFrame,
    config: Mapping[str, object],
) -> pd.DataFrame:
    fnode = _first_token(config.get("fnode"))
    tnode = _first_token(config.get("tnode"))
    iftran = _first_token(config.get("iftran"))
    frac = _first_token(config.get("frac"))
    inc_area = _first_token(config.get("inc_area"))
    staid = _first_token(config.get("staid"))
    depvar = _first_token(config.get("depvar"))

    cols = [fnode, tnode, iftran, frac, inc_area, staid, depvar]
    data = indata.loc[:, [c for c in cols if c]]

    selection = config.get("calibrate_selection_criteria")
    if selection:
        mask = pd.Series(_eval_expr(str(selection), data, for_condition=True), index=data.index)
        data = data.loc[mask]

    if data.empty:
        return station_data

    data_values = data.to_numpy()
    jfnode, jtnode, jiftran, jfrac, jinc_area, jstaid, jdepvar = range(7)

    nreach = data_values.shape[0]
    nnode = int(np.nanmax(data_values[:, [jfnode, jtnode]]))

    obsloc = np.where(~np.isnan(data_values[:, jdepvar]))[0]
    nobs = len(obsloc)

    nest_area = np.full((nobs,), np.nan)
    node = np.zeros((nnode + 1,))

    i_obs = 0
    for i in range(nreach):
        dep_val = data_values[i, jdepvar]
        fnode_val = int(data_values[i, jfnode])
        tnode_val = int(data_values[i, jtnode])
        if np.isnan(dep_val):
            node[tnode_val] = node[tnode_val] + data_values[i, jiftran] * (
                data_values[i, jinc_area] + data_values[i, jfrac] * node[fnode_val]
            )
        else:
            nest_area[i_obs] = data_values[i, jinc_area] + data_values[i, jfrac] * node[fnode_val]
            i_obs += 1

    output = pd.DataFrame(
        {
            staid: data_values[obsloc, jstaid],
            "nested_area": nest_area,
        }
    )
    output = output.sort_values(by=staid, kind="stable", na_position="first")

    merged = station_data.merge(output, on=staid, how="outer")
    return merged


def _build_labels(config: Mapping[str, object]) -> dict[str, str]:
    labels: dict[str, str] = {}

    staid = _first_token(config.get("staid"))
    inc_area = _first_token(config.get("inc_area"))
    tot_area = _first_token(config.get("tot_area"))
    mean_flow = _first_token(config.get("mean_flow"))
    waterid = _first_token(config.get("waterid"))
    fnode = _first_token(config.get("fnode"))
    tnode = _first_token(config.get("tnode"))
    iftran = _first_token(config.get("iftran"))
    frac = _first_token(config.get("frac"))
    hydseq = _first_token(config.get("hydseq"))
    ls_weight = _first_token(config.get("ls_weight"))
    lat = _first_token(config.get("lat"))
    lon = _first_token(config.get("lon"))
    target = _first_token(config.get("target"))

    if staid:
        labels[staid] = "SPARROW Monitoring Station Identifier"
    if inc_area:
        labels[inc_area] = "Incremental Drainage Area (km2)"
    if tot_area:
        labels[tot_area] = "Upstream Drainage Area (km2)"
    if mean_flow:
        labels[mean_flow] = "Mean Flow (cfs)"
    if waterid:
        labels[waterid] = "SPARROW Reach Identifier"
    if fnode:
        labels[fnode] = "Upstream Reach Node Identifier"
    if tnode:
        labels[tnode] = "Downstream Reach Node Identifier"
    if iftran:
        labels[iftran] = "If Reach Transmits Flux (1=yes, 0=no)"
    if frac:
        labels[frac] = "Fraction Upstream Flux Diverted to Reach"
    if hydseq:
        labels[hydseq] = "SPARROW Reach Hydrologic Sequencing Code"
    if ls_weight:
        labels[ls_weight] = "Least-Squares Weight"
    if lat:
        labels[lat] = "Station Latitude (decimal degrees)"
    if lon:
        labels[lon] = "Station Longitude (decimal degrees)"
    if target:
        labels[target] = "Destination Reach for Delivery (0/1)"
    if _is_yes(config.get("if_distribute_yield_by_land_use")):
        labels["LU_class"] = "Watershed Land-Cover Type"
    labels["nested_area"] = "Station Nested Basin Area (km2)"

    return labels
