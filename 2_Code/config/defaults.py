"""Defaults equivalent to testglobal.sas."""

from __future__ import annotations

from typing import Mapping, MutableMapping


_SAS_OPTIONS_DETAILS = "source notes mprint macrogen nosymbolgen"
_SAS_OPTIONS_QUIET = "nosource nonotes nomprint nomacrogen nosymbolgen"


def _is_yes(value: object) -> bool:
    """Return True when a SAS-style YES value is provided."""
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().upper() == "YES"


def _find_path_delimiter(home_results: str | None) -> str:
    """Mirror SAS findc for '/' or '\\' and return the first delimiter."""
    if not home_results:
        return ""
    for ch in home_results:
        if ch == "/" or ch == "\\":
            return ch
    return ""


def apply_testglobal_defaults(config: MutableMapping[str, object]) -> MutableMapping[str, object]:
    """Apply testglobal.sas defaults to a config mapping.

    This mutates and returns the provided mapping for convenience.
    """
    if_print_details = config.get("if_print_details")
    config["sas_options"] = (
        _SAS_OPTIONS_DETAILS if _is_yes(if_print_details) else _SAS_OPTIONS_QUIET
    )

    home_results = config.get("home_results")
    config["pthdel"] = _find_path_delimiter(
        None if home_results is None else str(home_results)
    )

    for key in (
        "gis_file",
        "convert_specification",
        "n_periods",
        "catchment_storage_source",
        "catchment_storage_source_exclude",
    ):
        if key not in config:
            config[key] = ""

    return config


def with_testglobal_defaults(config: Mapping[str, object]) -> dict[str, object]:
    """Return a copy of config with testglobal.sas defaults applied."""
    new_config: dict[str, object] = dict(config)
    return apply_testglobal_defaults(new_config)
