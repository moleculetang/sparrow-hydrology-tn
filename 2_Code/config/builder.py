"""Build SPARROW macro-equivalent configuration derived values."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import floor
from pathlib import Path
from typing import Callable, Iterable, Mapping, MutableMapping
from datetime import datetime


@dataclass
class ModelSpec:
    config: dict[str, object]
    indata_list: list[str]
    betalst: list[str]
    bretain: list[str]
    blbnd: list[str]
    bubnd: list[str]
    blubnd: tuple[list[str], list[str]]
    datalst: list[str]
    model_vars_list: list[str]
    globvar: list[str]
    makecol: dict[str, list[int]]
    n_low: int
    n_hi: int
    n_store: int
    end_jter: int
    predict_prefix: str
    station_priority_list: list[str]
    reach_priority_list: list[str]
    numerator_load_units: str
    adjust_units: int | float | str | None
    adjust_conc_units: int | None
    concentration_units: str
    lu_class_footnote: str
    lu_class_length: int
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


TableExistsFn = Callable[[str], bool]


_SAS_OPTIONS_DETAILS = "source notes mprint macrogen nosymbolgen"
_SAS_OPTIONS_QUIET = "nosource nonotes nomprint nomacrogen nosymbolgen"


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


def _to_float(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except ValueError:
        return default


def _count(tokens: Iterable[str]) -> int:
    return len(list(tokens))


def _check_list(list_a: Iterable[str], list_b: Iterable[str]) -> list[str]:
    list_b_upper = {item.upper() for item in list_b if item}
    excluded: list[str] = []
    for item in list_a:
        if not item:
            continue
        if item.upper() not in list_b_upper:
            excluded.append(item)
    return excluded


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


def _col_indices(master: list[str], query: list[str]) -> list[int]:
    if not query:
        return [0]
    indices: list[int] = []
    for item in query:
        idx = 0
        if item:
            for i, master_item in enumerate(master, start=1):
                if item.upper() == master_item.upper():
                    idx = i
                    break
        indices.append(idx)
    if not indices:
        return [0]
    return indices


def _default_table_exists(name: str, config: Mapping[str, object]) -> bool:
    results_tables = config.get("results_tables")
    if isinstance(results_tables, Mapping):
        path = results_tables.get(name)
        if path:
            return Path(path).exists()
    results_dir = config.get("results_dir") or config.get("home_results")
    if not results_dir:
        return False
    base = Path(str(results_dir))
    if not base.exists():
        return False
    if (base / name).exists():
        return True
    if list(base.glob(f"{name}.*")):
        return True
    return False


def _form_it_lines(intext: str, n_space: int) -> list[str]:
    if not intext:
        return [""]
    out: list[str] = []
    current = ""
    count = 0
    for ch in intext:
        if ch == " ":
            count += 1
            continue
        if count == 0:
            current += ch
        elif count < n_space:
            current += " " + ch
        else:
            out.append(current)
            current = ch
        count = 0
    if current or out:
        out.append(current)
    return out or [""]


def _delim_it_lines(intext: str, del_char: str) -> list[str]:
    if not intext:
        return [""]
    parts = [part for part in (p.strip() for p in intext.split(del_char)) if part]
    return parts or [""]


def _as_macro_str(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return " ".join(str(item) for item in value if item is not None)
    return str(value)


def render_model_summary(config: Mapping[str, object]) -> str:
    """Render the model specification summary text."""
    now = datetime.now()
    date_str = now.strftime("%d%b%Y").upper()
    time_str = now.strftime("%H:%M:%S")

    betailst_raw = _as_macro_str(config.get("betailst"))
    data_modifications_raw = _as_macro_str(config.get("data_modifications"))
    dlvdsgn_raw = _as_macro_str(config.get("dlvdsgn"))

    blines = _form_it_lines(betailst_raw, 3)
    dmlines = _delim_it_lines(data_modifications_raw, ";")
    dldsgn_lines = _delim_it_lines(dlvdsgn_raw, ",")

    lines: list[str] = ["", f"DATE: {date_str} TIME: {time_str}", ""]
    lines.append(f"HOME_RESULTS: {_as_macro_str(config.get('home_results'))}")
    lines.append(f"HOME_DATA: {_as_macro_str(config.get('home_data'))}")
    lines.append(
        "INDATA: "
        + _as_macro_str(config.get("indata"))
        + "  IF_MAKE_INPUT_DATA: "
        + _as_macro_str(config.get("if_make_input_data"))
    )
    lines.append("BETAILST:")
    lines.extend(blines)
    lines.append(
        "DEPVAR: "
        + _as_macro_str(config.get("depvar"))
        + "  LS_WEIGHT: "
        + _as_macro_str(config.get("ls_weight"))
    )
    lines.append("SRCVAR: " + _as_macro_str(config.get("srcvar")))
    lines.append("BSRCVAR: " + _as_macro_str(config.get("bsrcvar")))
    lines.append("DLVVAR: " + _as_macro_str(config.get("dlvvar")))
    lines.append("BDLVVAR: " + _as_macro_str(config.get("bdlvvar")))
    lines.append(
        "IF_MEAN_ADJUST_DELIVERY_VARS: "
        + _as_macro_str(config.get("if_mean_adjust_delivery_vars"))
    )
    lines.append("DLVDSGN:")
    lines.extend(dldsgn_lines)
    lines.append("DECVAR: " + _as_macro_str(config.get("decvar")))
    lines.append("BDECVAR: " + _as_macro_str(config.get("bdecvar")))
    lines.append("RESVAR: " + _as_macro_str(config.get("resvar")))
    lines.append("BRESVAR: " + _as_macro_str(config.get("bresvar")))
    lines.append("OTHVAR: " + _as_macro_str(config.get("othvar")))
    lines.append("BOTHVAR: " + _as_macro_str(config.get("bothvar")))
    lines.append(
        "REACH_DECAY_SPECIFICATION: "
        + _as_macro_str(config.get("reach_decay_specification"))
    )
    lines.append(
        "RESERVOIR_DECAY_SPECIFICATION: "
        + _as_macro_str(config.get("reservoir_decay_specification"))
    )
    lines.append(
        "INCR_DELIVERY_SPECIFICATION: "
        + _as_macro_str(config.get("incr_delivery_specification"))
    )
    if _as_macro_str(config.get("convert_specification")):
        lines.append(
            "CONVERT_SPECIFICATION: "
            + _as_macro_str(config.get("convert_specification"))
        )
    if _as_macro_str(config.get("n_periods")):
        lines.append(
            "N_PERIODS: "
            + _as_macro_str(config.get("n_periods"))
            + "  CATCHMENT_STORAGE_SOURCE: "
            + _as_macro_str(config.get("catchment_storage_source"))
            + "  CATCHMENT_STORAGE_SOURCE_EXCLUDE: "
            + _as_macro_str(config.get("catchment_storage_source_exclude"))
        )
    lines.append(
        "IF_ESTIMATE: "
        + _as_macro_str(config.get("if_estimate"))
        + "  IF_TEST_CALIBRATE: "
        + _as_macro_str(config.get("if_test_calibrate"))
        + "  IF_ACCUMULATE_WITH_DLL: "
        + _as_macro_str(config.get("if_accumulate_with_dll"))
        + "  IF_PREDICT: "
        + _as_macro_str(config.get("if_predict"))
    )
    lines.append(
        "CALIBRATE_SELECTION_CRITERIA: "
        + _as_macro_str(config.get("calibrate_selection_criteria"))
    )
    lines.append("DATA_MODIFICATIONS:")
    lines.extend(dmlines)
    lines.append(" ;")
    lines.append("_" * 102)

    return "\n".join(lines) + "\n"


def write_model_summary(
    config: Mapping[str, object],
    path: str | Path | None = None,
) -> Path:
    """Write the model summary text to the summary specs file."""
    if path is None:
        home_results = config.get("home_results")
        if home_results:
            path = Path(str(home_results)) / "summary_model_specs.txt"
        else:
            path = Path("summary_model_specs.txt")
    else:
        path = Path(path)

    text = render_model_summary(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(text)
    return path


def _format_missing_tables(missing: list[str]) -> str:
    if not missing:
        return ""
    if len(missing) == 1:
        return missing[0]
    if len(missing) == 2:
        return f"{missing[0]} and {missing[1]}"
    return f"{missing[0]}, {missing[1]} and {missing[2]}" if len(missing) == 3 else ", ".join(missing[:-1]) + f" and {missing[-1]}"


def build_model_spec(
    config: MutableMapping[str, object],
    *,
    table_exists: TableExistsFn | None = None,
    n_obs: int | None = None,
) -> ModelSpec:
    warnings: list[str] = []
    errors: list[str] = []

    def warn(msg: str) -> None:
        warnings.append(msg)

    def error(msg: str) -> None:
        errors.append(msg)
        config["if_error"] = "yes"

    if "if_error" not in config:
        config["if_error"] = "no"

    if_print_details = config.get("if_print_details")
    config["sas_options"] = (
        _SAS_OPTIONS_DETAILS if _is_yes(if_print_details) else _SAS_OPTIONS_QUIET
    )

    waterid = _split_tokens(config.get("waterid"))
    arcid = _split_tokens(config.get("arcid"))
    optional_reach_information = _split_tokens(config.get("optional_reach_information"))
    fnode = _split_tokens(config.get("fnode"))
    tnode = _split_tokens(config.get("tnode"))
    hydseq = _split_tokens(config.get("hydseq"))
    inc_area = _split_tokens(config.get("inc_area"))
    tot_area = _split_tokens(config.get("tot_area"))
    mean_flow = _split_tokens(config.get("mean_flow"))
    frac = _split_tokens(config.get("frac"))
    iftran = _split_tokens(config.get("iftran"))
    target = _split_tokens(config.get("target"))
    ls_weight = _split_tokens(config.get("ls_weight"))
    staid = _split_tokens(config.get("staid"))
    optional_station_information = _split_tokens(config.get("optional_station_information"))
    lat = _split_tokens(config.get("lat"))
    lon = _split_tokens(config.get("lon"))
    depvar = _split_tokens(config.get("depvar"))
    srcvar = _split_tokens(config.get("srcvar"))
    dlvvar = _split_tokens(config.get("dlvvar"))
    decvar = _split_tokens(config.get("decvar"))
    resvar = _split_tokens(config.get("resvar"))
    othvar = _split_tokens(config.get("othvar"))

    if _is_yes(config.get("if_distribute_yield_by_land_use")):
        variable_list = (
            waterid
            + arcid
            + optional_reach_information
            + fnode
            + tnode
            + hydseq
            + inc_area
            + tot_area
            + mean_flow
            + frac
            + iftran
            + target
            + ls_weight
            + staid
            + optional_station_information
            + lat
            + lon
            + depvar
            + srcvar
            + dlvvar
            + decvar
            + resvar
            + othvar
            + ["LU_class"]
        )
    else:
        variable_list = (
            waterid
            + arcid
            + optional_reach_information
            + fnode
            + tnode
            + hydseq
            + inc_area
            + tot_area
            + mean_flow
            + frac
            + iftran
            + target
            + ls_weight
            + staid
            + optional_station_information
            + lat
            + lon
            + depvar
            + srcvar
            + dlvvar
            + decvar
            + resvar
            + othvar
        )

    indata_list: list[str] = []
    for var in variable_list:
        if not var:
            continue
        if var not in indata_list:
            indata_list.append(var)

    betailst = _split_tokens(config.get("betailst"))
    betalst: list[str] = []
    bretain: list[str] = []
    blbnd: list[str] = []
    bubnd: list[str] = []

    ib = 0
    while ib < len(betailst):
        bvar = betailst[ib]
        betalst.append(bvar)
        if ib + 1 < len(betailst):
            bretain.extend([bvar, betailst[ib + 1]])
        else:
            bretain.extend([bvar, ""])
        if ib + 2 < len(betailst):
            bound = str(betailst[ib + 2])
            if ":" in bound:
                lower, upper = bound.split(":", 1)
            else:
                lower, upper = bound, ""
            blbnd.append(lower)
            bubnd.append(upper)
        else:
            blbnd.append("")
            bubnd.append("")
        ib += 3

    blubnd = (blbnd, bubnd)

    addlist = depvar + srcvar + dlvvar + decvar + resvar + othvar
    datalst: list[str] = [
        *waterid,
        *staid,
        *fnode,
        *tnode,
        *hydseq,
        *frac,
        *iftran,
        *target,
        *inc_area,
        *tot_area,
        *mean_flow,
        *ls_weight,
    ]
    datalst_upper = {item.upper() for item in datalst if item}
    for addvar in addlist:
        if not addvar:
            continue
        if addvar.upper() not in datalst_upper:
            datalst.append(addvar)
            datalst_upper.add(addvar.upper())

    model_vars_list = _check_list(datalst, staid + target + depvar + tot_area + ls_weight)

    globvar = [
        "data",
        "node",
        "obsloc",
        "nreach",
        "nnode",
        "nobs",
        "ndef",
        "est",
        "if_final_pass",
        "weights",
        "jncnstrn",
        "jfnode",
        "jtnode",
        "jfrac",
        "jtarget",
        "jiftran",
        "jdepvar",
        "jdlvvar",
        "jdecvar",
        "jresvar",
        "jsrcvar",
        "jbdlvvar",
        "jbdecvar",
        "jbresvar",
        "jbsrcvar",
        "jwaterid",
        "jstaid",
        "jtotarea",
        "dlvdsgn",
    ]

    bsrcvar = _split_tokens(config.get("bsrcvar"))
    bdlvvar = _split_tokens(config.get("bdlvvar"))
    bdecvar = _split_tokens(config.get("bdecvar"))
    bresvar = _split_tokens(config.get("bresvar"))
    bothvar = _split_tokens(config.get("bothvar"))

    makecol: dict[str, list[int]] = {
        "jwaterid": _col_indices(datalst, waterid),
        "jstaid": _col_indices(datalst, staid),
        "jfnode": _col_indices(datalst, fnode),
        "jtnode": _col_indices(datalst, tnode),
        "jfrac": _col_indices(datalst, frac),
        "jtarget": _col_indices(datalst, target),
        "jiftran": _col_indices(datalst, iftran),
        "jdepvar": _col_indices(datalst, depvar),
        "jsrcvar": _col_indices(datalst, srcvar),
        "jdlvvar": _col_indices(datalst, dlvvar),
        "jdecvar": _col_indices(datalst, decvar),
        "jresvar": _col_indices(datalst, resvar),
        "jtotarea": _col_indices(datalst, tot_area),
        "jbsrcvar": _col_indices(betalst, bsrcvar),
        "jbdlvvar": _col_indices(betalst, bdlvvar),
        "jbdecvar": _col_indices(betalst, bdecvar),
        "jbresvar": _col_indices(betalst, bresvar),
    }

    for voth in othvar:
        if not voth:
            continue
        globvar.append(f"j{voth}")
        makecol[f"j{voth}"] = _col_indices(datalst, [voth])

    for both in bothvar:
        if not both:
            continue
        globvar.append(f"j{both}")
        makecol[f"j{both}"] = _col_indices(betalst, [both])

    if not dlvvar and _is_yes(config.get("if_mean_adjust_delivery_vars")):
        config["if_mean_adjust_delivery_vars"] = "no"
        warn(
            "No land-to-water delivery variables (dlvvar) are specified. "
            "Control variable if_mean_adjust_delivery_vars is set to no."
        )

    excld_bsrcvar = _check_list(bsrcvar, betalst)
    if excld_bsrcvar:
        error(
            "Coefficient(s) "
            + " ".join(excld_bsrcvar)
            + " of bsrcvar not in the betailst - stop processing"
        )

    excld_bdlvvar = _check_list(bdlvvar, betalst)
    if excld_bdlvvar:
        error(
            "Coefficient(s) "
            + " ".join(excld_bdlvvar)
            + " of bdlvvar not in the betailst - stop processing"
        )

    excld_bdecvar = _check_list(bdecvar, betalst)
    if excld_bdecvar:
        error(
            "Coefficient(s) "
            + " ".join(excld_bdecvar)
            + " of bdecvar not in the betailst - stop processing"
        )

    excld_bresvar = _check_list(bresvar, betalst)
    if excld_bresvar:
        error(
            "Coefficient(s) "
            + " ".join(excld_bresvar)
            + " of bresvar not in the betailst - stop processing"
        )

    excld_bothvar = _check_list(bothvar, betalst)
    if excld_bothvar:
        error(
            "Coefficient(s) "
            + " ".join(excld_bothvar)
            + " of bothvar not in the betailst - stop processing"
        )

    bunion = bsrcvar + bdlvvar + bdecvar + bresvar + bothvar
    excld_betalst = _check_list(betalst, bunion)
    if excld_betalst:
        warn(
            "The following coefficient(s) of betailst are not in the "
            "bsrcvar/bdlvvar/bdecvar/bresvar/other lists: "
            + " ".join(excld_betalst)
        )

    nsrcvar = _count(srcvar)
    if _count(srcvar) != _count(bsrcvar):
        error(
            "Number of source variables ("
            + str(_count(srcvar))
            + ") and source variable coefficients ("
            + str(_count(bsrcvar))
            + ") are unequal - stop processing"
        )

    ndlvvar = _count(dlvvar)
    if _count(dlvvar) != _count(bdlvvar):
        error(
            "Number of delivery variables ("
            + str(_count(dlvvar))
            + ") and delivery variable coefficients ("
            + str(_count(bdlvvar))
            + ") are unequal - stop processing"
        )

    if _count(decvar) != _count(bdecvar):
        error(
            "Number of decay variables ("
            + str(_count(decvar))
            + ") and decay variable coefficients ("
            + str(_count(bdecvar))
            + ") are unequal - stop processing"
        )

    if _count(resvar) != _count(bresvar):
        error(
            "Number of reservoir variables ("
            + str(_count(resvar))
            + ") and reservoir variable coefficients ("
            + str(_count(bresvar))
            + ") are unequal - stop processing"
        )

    dlvdsgn = str(config.get("dlvdsgn") or "").strip()
    if dlvdsgn:
        rows = [row.strip() for row in dlvdsgn.split(",") if row.strip()]
        isrc = 1
        for row in rows:
            if isrc > nsrcvar:
                error(
                    "Too many rows in the delivery design (dlvdsgn) specification: "
                    f"srcvar only has {nsrcvar} variables - stop processing"
                )
                break
            ntest = len(row.split())
            if ntest != ndlvvar:
                error(
                    "Incorrect number of column elements in the delivery design "
                    f"(dlvdsgn) specification for row {isrc} (set at {ntest} - "
                    f"should be {ndlvvar}) - stop processing"
                )
            isrc += 1
        if isrc < nsrcvar + 1:
            error(
                "Too many rows in the delivery design (dlvdsgn) specification: "
                f"srcvar only has {nsrcvar} variables - stop processing"
            )

    if _is_yes(config.get("if_test_calibrate")) and _is_yes(config.get("if_estimate")):
        config["n_boot_iter"] = 0

    if not str(config.get("start_iter") or ""):
        config["start_iter"] = 0
    if not str(config.get("start_jter") or ""):
        config["start_jter"] = 0
    if not str(config.get("end_iter") or ""):
        config["end_iter"] = config.get("n_boot_iter", 0)

    n_boot_iter = _to_int(config.get("n_boot_iter"), 0)
    n_extra_jter = _to_int(config.get("n_extra_jter"), 0)
    end_jter = n_boot_iter + n_extra_jter
    config["end_jter"] = end_jter

    if str(config.get("if_estimate", "")).strip().upper() == "NO":
        checker = table_exists or (lambda name: _default_table_exists(name, config))
        if n_boot_iter == 0:
            if not checker("boot_betaest_all"):
                error(
                    "Requested prediction without estimation but "
                    "dir_rslt.boot_betaest_all was not found - stop processing."
                )
        else:
            missing: list[str] = []
            if not checker("boot_betaest_all"):
                missing.append("dir_rslt.boot_betaest_all")
            if not checker("summary_betaest"):
                missing.append("dir_rslt.summary_betaest")
            if not checker("resids"):
                missing.append("dir_rslt.resids")
            if missing:
                verb = "was" if len(missing) == 1 else "were"
                missing_text = _format_missing_tables(missing)
                error(
                    "Requested bootstrap prediction without estimation but "
                    f"{missing_text} {verb} not found - stop processing."
                )

    cov_prob = _to_float(config.get("cov_prob"), 0.0)
    n_low = int(floor((1 - cov_prob / 100.0) * n_boot_iter / 2.0) + 1)
    n_hi = int(floor((1 - cov_prob / 100.0) * n_boot_iter) + 2 - n_low)
    n_store = n_low + n_hi

    config["n_low"] = n_low
    config["n_hi"] = n_hi
    config["n_store"] = n_store

    n_seeds = _to_int(config.get("n_seeds"), 0)
    config["seed_names"] = [f"seed_{i}" for i in range(1, n_seeds + 1)]
    config["predlst"] = config.get("predlst", "")

    load_units = str(config.get("load_units") or "")
    numerator_load_units = load_units.split("/")[0] if "/" in load_units else load_units

    adjust_units: int | float | str | None = config.get("adjust_units")
    if load_units == "kg/yr" or load_units == "Bcol/yr":
        adjust_units = 1
    elif load_units == "mt/yr":
        adjust_units = 1000

    if _is_yes(config.get("if_concentration_in_micrograms")):
        adjust_conc_units = 1000
        concentration_units = "ug/L"
    else:
        adjust_conc_units = 1
        concentration_units = "mg/L"

    if load_units == "Bcol/yr":
        adjust_conc_units = 100
        concentration_units = "col/100ml"

    if _is_yes(config.get("if_test_calibrate")):
        config["if_accumulate_with_dll"] = "no"

    if _is_yes(config.get("if_accumulate_with_dll")) and str(config.get("syssite")) == "70208402":
        config["if_accumulate_with_dll"] = "no"
        warn(
            "SPARROW is running in SAS University Edition making the accumulation "
            "accelerator DLL unavailable - if_accumulate_with_dll is set to no."
        )

    if n_boot_iter > 0 and _is_yes(config.get("if_print_boot_predictions")):
        predict_prefix = "mean_"
    else:
        predict_prefix = ""

    station_priority_list = (
        staid
        + optional_station_information
        + tot_area
        + mean_flow
        + lat
        + lon
        + waterid
        + arcid
        + ls_weight
    )

    reach_priority_list = (
        waterid
        + optional_reach_information
        + tot_area
        + inc_area
        + mean_flow
        + arcid
        + fnode
        + tnode
        + hydseq
        + frac
        + iftran
        + target
        + ls_weight
        + ["LU_class"]
    )

    lu_class_footnote = ""
    lu_class_length = 0
    if _is_yes(config.get("if_distribute_yield_by_land_use")):
        land_class_list = _split_tokens(config.get("land_class_list"))
        n_lu_class = 1
        i_var = 2
        while i_var <= len(land_class_list):
            class_var = land_class_list[i_var - 1]
            if not class_var:
                break
            n_lu_class += 1
            i_var += 3
        i_var = 2
        i_lu_class = 1
        while i_var <= len(land_class_list):
            class_var = land_class_list[i_var - 1]
            if not class_var:
                break
            pct_index = i_var + 2
            pct_val = land_class_list[pct_index - 1] if pct_index <= len(land_class_list) else ""
            pct_var = f"(>{pct_val}%)"
            if i_lu_class == 1:
                connect = ""
            elif i_lu_class == n_lu_class:
                connect = ", and "
            else:
                connect = ", "
            lu_class_footnote = f"{lu_class_footnote}{connect} {class_var} {pct_var}"
            lu_class_length = max(lu_class_length, len(class_var))
            i_var += 3
            i_lu_class += 1
        if lu_class_footnote:
            lu_class_footnote = f"{lu_class_footnote}."

    catchment_storage_source = str(config.get("catchment_storage_source") or "").strip()
    if catchment_storage_source:
        if not str(config.get("n_periods") or ""):
            error(
                "Storage analysis is requested but n_periods control variable is unspecified "
                "- analysis terminated."
            )
        elif n_obs is not None:
            n_periods = _to_int(config.get("n_periods"), 0)
            if n_periods and n_obs % n_periods != 0:
                error(
                    "The number of observations in the input data ("
                    + str(n_obs)
                    + ") is incompatible with the number of periods specified for n_periods "
                    "- analysis terminated."
                )
        try:
            catchment_source_val = float(catchment_storage_source)
        except ValueError:
            catchment_source_val = None
        if catchment_source_val is not None and catchment_source_val > nsrcvar:
            error(
                "Specified value for catchment_storage_source is greater than number of "
                "declared source variables in srcvar - analysis terminated."
            )
        exclude = _set_unique(_split_tokens(config.get("catchment_storage_source_exclude")))
        config["catchment_storage_source_exclude"] = " ".join(exclude)
        for val in exclude:
            try:
                val_num = float(val)
            except ValueError:
                val_num = None
            if val_num is not None and val_num > nsrcvar:
                error(
                    "A specified value for catchment_storage_source_exclude ("
                    + str(val)
                    + ") is greater than number of declared source variables in srcvar "
                    "- analysis terminated."
                )

    config.update(
        {
            "indata_list": indata_list,
            "betalst": betalst,
            "bretain": bretain,
            "blbnd": blbnd,
            "bubnd": bubnd,
            "blubnd": blubnd,
            "datalst": datalst,
            "model_vars_list": model_vars_list,
            "globvar": globvar,
            "makecol": makecol,
            "predict_prefix": predict_prefix,
            "station_priority_list": station_priority_list,
            "reach_priority_list": reach_priority_list,
            "numerator_load_units": numerator_load_units,
            "adjust_units": adjust_units,
            "adjust_conc_units": adjust_conc_units,
            "concentration_units": concentration_units,
            "LU_class_footnote": lu_class_footnote,
            "LU_class_length": lu_class_length,
        }
    )

    return ModelSpec(
        config=dict(config),
        indata_list=indata_list,
        betalst=betalst,
        bretain=bretain,
        blbnd=blbnd,
        bubnd=bubnd,
        blubnd=blubnd,
        datalst=datalst,
        model_vars_list=model_vars_list,
        globvar=globvar,
        makecol=makecol,
        n_low=n_low,
        n_hi=n_hi,
        n_store=n_store,
        end_jter=end_jter,
        predict_prefix=predict_prefix,
        station_priority_list=station_priority_list,
        reach_priority_list=reach_priority_list,
        numerator_load_units=numerator_load_units,
        adjust_units=adjust_units,
        adjust_conc_units=adjust_conc_units,
        concentration_units=concentration_units,
        lu_class_footnote=lu_class_footnote,
        lu_class_length=lu_class_length,
        warnings=warnings,
        errors=errors,
    )
