"""Input semantics and fail-closed real-run gates; no scientific gap filling.

These checks run before integration. ``known_zero`` is an explicit claim with
provenance, ``excluded`` is a stated accounting boundary, and ``unknown`` is a
gap. None is interchangeable with a missing file or NaN.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping
from functools import lru_cache
import hashlib
import json
import numpy as np


class ContractError(ValueError):
    pass


VALID_STATUS = {"available", "known_zero", "excluded", "unknown"}
VALID_ROLE = {"external", "internal", "activity", "land_support", "state", "parameter", "driver", "operator"}
INTERNAL_FLUXES = {"mineralization", "uptake", "residue_return", "stock_release", "retranslocation"}


@dataclass(frozen=True)
class InputField:
    name: str
    status: str
    role: str
    unit: str
    provenance: str
    path: str | None = None
    adopted_year: str | int | None = None
    original_year: str | int | None = None
    is_scenario: bool = False
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    sha256: str | None = None
    payload: Any = None

    def validate(self, *, check_files: bool = True) -> list[str]:
        errors: list[str] = []
        if self.status not in VALID_STATUS:
            errors.append(f"{self.name}:invalid_status:{self.status}")
        if self.role not in VALID_ROLE:
            errors.append(f"{self.name}:invalid_role:{self.role}")
        if not self.unit or not self.provenance:
            errors.append(f"{self.name}:missing_unit_or_provenance")
        if self.status in {"known_zero", "excluded", "unknown"} and not self.reason:
            errors.append(f"{self.name}:status_requires_reason")
        if self.status in {"available", "known_zero"}:
            # Status text is not evidence. Hash and payload/file must agree;
            # known-zero additionally verifies that numeric contents are zero.
            try:
                _verify_artifact(self, check_files=check_files)
            except (ValueError, OSError, KeyError, TypeError) as exc:
                errors.append(f"{self.name}:artifact_invalid:{exc}")
        if self.name in INTERNAL_FLUXES and self.role == "external":
            errors.append(f"{self.name}:internal_transfer_cannot_be_external")
        if self.metadata.get("origin") == "MOD17" and self.role == "external":
            errors.append(f"{self.name}:MOD17_carbon_activity_is_not_external_N")
        if self.metadata.get("value_kind") == "upstream_cumulative_exposure" and self.role == "external":
            errors.append(f"{self.name}:upstream_exposure_would_double_count_sources")
        return errors


def payload_sha256(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


_CORE_SPECS = {
    "old_four_source_monthly": ("external", "kg N/month", (768, 230, 4)),
    "old_removal_proxy": ("activity", "kg N/month", (768, 230)),
    "incremental_area": ("land_support", "ha", (230,)),
    "deposition_monthly_incremental": ("external", "kg N/month", None),
    "deposition_daily_land_components": ("external", "kg N/day", (23376, 230, 4)),
    "h1": ("driver", "mixed_declared_driver_units", None),
}


def _verify_artifact(item: InputField, *, check_files: bool):
    if not item.sha256 or len(item.sha256) != 64:
        raise ContractError("available/known-zero requires an explicit frozen SHA256")
    if item.path is not None and item.payload is not None:
        raise ContractError("choose a file or embedded payload, not both")
    spec = _CORE_SPECS.get(item.name)
    if spec and (item.role, item.unit) != spec[:2]:
        raise ContractError(f"core role/unit mismatch; expected {spec[:2]}")
    year_evidence = item.metadata.get("year_evidence")
    if year_evidence:
        evidence_path = Path(year_evidence["path"])
        if not evidence_path.is_file() or file_sha256(evidence_path) != year_evidence.get("sha256"):
            raise ContractError("adopted-year evidence missing or hash mismatch")
    if item.path:
        path = Path(item.path)
        if not path.is_file():
            raise ContractError("declared artifact file is missing")
        # check_files=False is for small fixtures; it never skips the identity
        # check. It only avoids recursive manifest content validation.
        stat = path.stat()
        _verify_file_cached(str(path.resolve()), stat.st_size, stat.st_mtime_ns,
                            item.sha256, item.name, item.role, item.unit, item.status,
                            json.dumps(item.metadata, sort_keys=True), check_files)
    elif item.payload is not None:
        if payload_sha256(item.payload) != item.sha256:
            raise ContractError("embedded payload hash mismatch")
        kind = item.metadata.get("artifact_kind")
        if kind == "semantic_definition":
            if item.role != "internal":
                raise ContractError("semantic definitions cannot replace external/activity/driver artifacts")
        else:
            values = finite_nonnegative(item.payload, item.name)
            expected = item.metadata.get("shape")
            if expected is None or values.shape != tuple(expected):
                raise ContractError("embedded numeric artifact requires matching explicit shape")
            if spec and spec[2] and values.shape != spec[2]:
                raise ContractError("core shape mismatch")
            if item.status == "known_zero" and np.any(values != 0):
                raise ContractError("known_zero contains nonzero values")
            if spec and item.name != "incremental_area":
                raise ContractError("full-domain time artifacts require explicit date/reach evidence files")
    else:
        raise ContractError("status has no verifiable file or embedded payload")


def _verify_file_cached(path_text, size, mtime_ns, expected_sha, name, role, unit, status, metadata_json, recursive):
    path = Path(path_text)
    if file_sha256(path) != expected_sha:
        raise ContractError("file hash mismatch")
    metadata = json.loads(metadata_json)
    kind = metadata.get("artifact_kind")
    if kind in {"npy", "daily_external_npy"}:
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        expected_shape = metadata.get("shape")
        if expected_shape is None or values.shape != tuple(expected_shape):
            raise ContractError("array does not match explicit shape")
        spec = _CORE_SPECS.get(name)
        if spec and spec[2] is not None and values.shape != spec[2]:
            raise ContractError("core field dimension mismatch")
        finite_nonnegative(values, name)
        if status == "known_zero" and np.any(values != 0):
            raise ContractError("known_zero contains nonzero values")
        if kind == "daily_external_npy":
            date_path = Path(metadata["date_path"])
            reach_path = Path(metadata["reach_path"])
            if file_sha256(date_path) != metadata["date_sha256"] or file_sha256(reach_path) != metadata["reach_sha256"]:
                raise ContractError("daily date/reach evidence hash mismatch")
            validate_daily_mass_bundle(np.load(date_path, allow_pickle=False), np.load(reach_path, allow_pickle=False),
                                       values, unit=unit, source_labels=metadata["source_labels"])
        if name in {"old_four_source_monthly", "old_removal_proxy"}:
            dates_path = Path(metadata["date_path"])
            if file_sha256(dates_path) != metadata["date_sha256"]:
                raise ContractError("date-axis hash mismatch")
            dates = np.load(dates_path, allow_pickle=False).astype("datetime64[M]")
            expected = np.arange(np.datetime64("1961-01", "M"), np.datetime64("2025-01", "M"))
            if not np.array_equal(dates, expected):
                raise ContractError("monthly date support incomplete, duplicated or unordered")
        if name in {"old_four_source_monthly", "old_removal_proxy", "incremental_area"}:
            reaches = metadata.get("reach_ids")
            if reaches != list(range(1, 231)):
                raise ContractError("reach support must explicitly be 1..230 once in order")
    elif kind == "deposition_incremental_csv":
        import pandas as pd
        frame = pd.read_csv(path)
        columns = ["drynhx_kg_n", "drynoy_kg_n", "wetnhx_kg_n", "wetnoy_kg_n"]
        expected = pd.MultiIndex.from_product([range(1961, 2025), range(1, 13), range(1, 231)], names=["year", "month", "reach_id"])
        actual = pd.MultiIndex.from_frame(frame[["year", "month", "reach_id"]])
        if actual.has_duplicates or len(actual) != len(expected) or not actual.sort_values().equals(expected):
            raise ContractError("deposition date/reach support mismatch")
        finite_nonnegative(frame[columns].to_numpy(), name)
        if not np.allclose(frame[columns].sum(axis=1), frame.land_total_kg_n, atol=1e-6, rtol=0):
            raise ContractError("deposition component/total closure failed")
    elif kind == "h1_array_manifest":
        manifest = json.loads(path.read_text(encoding="utf-8"))
        for required in ("contact", "fast_fraction", "lower_release", "upper_water", "percolation", "official_water"):
            record = manifest[required]
            target = path.parent / record["file"]
            values = np.load(target, mmap_mode="r", allow_pickle=False)
            if values.shape != (23376, 230) or list(values.shape) != record["shape"]:
                raise ContractError(f"H1 {required} shape mismatch")
            if recursive and file_sha256(target) != record["sha256"]:
                raise ContractError(f"H1 {required} array hash mismatch")
            finite_nonnegative(values, required)
        date_path = path.parent / "dates.npy"
        dates = np.load(date_path, allow_pickle=False).astype("datetime64[D]")
        if not np.array_equal(dates, np.arange(np.datetime64("1961-01-01"), np.datetime64("2025-01-01"))):
            raise ContractError("H1 daily calendar mismatch")
        if metadata.get("reach_ids") != list(range(1, 231)):
            raise ContractError("H1 reach identity missing")
    elif kind == "frozen_configuration":
        if role not in {"parameter", "operator", "state"}:
            raise ContractError("configuration cannot replace a data artifact")
        json.loads(path.read_text(encoding="utf-8"))
    else:
        raise ContractError("unknown or absent artifact schema")
    after = path.stat()
    if (after.st_size, after.st_mtime_ns) != (size, mtime_ns):
        raise ContractError("artifact changed during validation")


def file_sha256(path: str | Path) -> str:
    target = Path(path)
    stat = target.stat()
    return _file_sha256_cached(str(target.resolve()), stat.st_size, stat.st_mtime_ns)


@lru_cache(maxsize=512)
def _file_sha256_cached(path: str, size: int, mtime_ns: int) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    stat = Path(path).stat()
    if (stat.st_size, stat.st_mtime_ns) != (size, mtime_ns):
        raise ContractError("artifact changed while hashing")
    return h.hexdigest()


def finite_nonnegative(value: Any, name: str, *, shape: tuple[int, ...] | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if shape is not None and array.shape != shape:
        raise ContractError(f"{name}: shape {array.shape}, expected {shape}")
    if not np.all(np.isfinite(array)):
        raise ContractError(f"{name}: nonfinite values cannot enter integration")
    if np.any(array < 0):
        raise ContractError(f"{name}: negative values")
    return array


def validate_land_support(physical_area_ha: Any, harvested_area_ha: Any, *,
                          stock_area_ha: Any, impervious_soil_enabled: bool = False,
                          impervious_soil_evidence: str | None = None) -> None:
    """Multiple harvests may exceed cropland area but cannot multiply soil area."""
    physical = finite_nonnegative(physical_area_ha, "physical_area_ha")
    finite_nonnegative(harvested_area_ha, "harvested_area_ha")
    stock = finite_nonnegative(stock_area_ha, "stock_area_ha", shape=physical.shape)
    if not np.array_equal(stock, physical):
        raise ContractError("soil stocks must use physical land area, never repeated harvested area")
    if impervious_soil_enabled and not impervious_soil_evidence:
        raise ContractError("impervious surfaces require explicit soil/plant evidence")


def validate_budget_semantics(*, external_kind: str, demand_deducted: bool,
                              residue_return_as_external: bool = False,
                              crop_accesses_noncrop_pool: bool = False,
                              total_uptake_from_harvest_only: bool = False,
                              land_model_id: str = "LAND1") -> None:
    if external_kind == "net_surplus" and demand_deducted:
        raise ContractError("net surplus followed by demand subtraction double counts crop removal")
    if residue_return_as_external:
        raise ContractError("residue return is an internal transfer, not new external N")
    if crop_accesses_noncrop_pool:
        raise ContractError("cropland uptake cannot consume non-cropland stocks")
    if total_uptake_from_harvest_only and land_model_id == "LAND1":
        raise ContractError("harvest product removal alone cannot define LAND1 total plant uptake")


def validate_transition(matrix: Any, *, tolerance: float = 1e-12) -> np.ndarray:
    """Rows are donor land units, columns recipients; transfer conserves mass."""
    transfer = finite_nonnegative(matrix, "land_transition")
    if transfer.ndim < 2 or transfer.shape[-2] != transfer.shape[-1]:
        raise ContractError("land transition must end in a square donor-recipient matrix")
    if not np.allclose(transfer.sum(axis=-1), 1.0, atol=tolerance, rtol=0):
        raise ContractError("every donor row must distribute exactly its full stock")
    return transfer


@dataclass(frozen=True)
class SourceScale:
    """Single scaling owner. Input arrays must be raw external mass only."""
    log_multiplier: float
    owner: str = "external_mass_interface"
    applies_to: str = "entire_history_external_mass"
    already_applied: bool = False

    def apply(self, external: Mapping[str, Any], *, roles: Mapping[str, str]) -> dict[str, np.ndarray]:
        if self.owner != "external_mass_interface" or self.applies_to != "entire_history_external_mass":
            raise ContractError("source multiplier must have one full-history external-mass owner")
        if self.already_applied:
            raise ContractError("source multiplier already applied; double scaling forbidden")
        if not np.isfinite(self.log_multiplier) or abs(self.log_multiplier) > np.log(4.0) + 1e-14:
            raise ContractError("source multiplier outside [0.25, 4]")
        if not external:
            raise ContractError("active source multiplier has no external mass targets")
        output = {}
        for name, value in external.items():
            if roles.get(name) != "external" or name in INTERNAL_FLUXES:
                raise ContractError(f"{name}: cannot apply external source multiplier")
            output[name] = finite_nonnegative(value, name) * np.exp(self.log_multiplier)
        return output


def validate_parameter_responsibility(parameters: Iterable[Mapping[str, Any]]) -> None:
    rows = list(parameters)
    names = [row["name"] for row in rows]
    if len(names) != len(set(names)):
        raise ContractError("duplicate parameter coordinates")
    scale_rows = [row for row in rows if row.get("role") == "source_multiplier"]
    if len(scale_rows) > 1:
        raise ContractError("multiple active source multipliers")
    for row in rows:
        if not row.get("physical_targets"):
            raise ContractError(f"{row['name']}: prior-only/inactive parameter is forbidden")
        if row.get("role") == "source_multiplier":
            if row.get("owner") != "external_mass_interface" or row.get("history_scope") != "full":
                raise ContractError("source multiplier must affect all external history exactly once")


def validate_training_years(rows: Any, training_years: Iterable[int], *, year_column: str = "year") -> None:
    allowed = set(int(y) for y in training_years)
    actual = set(int(y) for y in rows[year_column])
    if not actual or not actual <= allowed:
        raise ContractError(f"training labels contain forbidden/empty years: {sorted(actual - allowed)}")


def validate_daily_mass_bundle(dates, reach_ids, values, *, unit: str, source_labels,
                               start="1961-01-01", stop="2025-01-01") -> np.ndarray:
    """Validate explicit daily masses at the integration boundary, not only metadata."""
    if unit != "kg N/day":
        raise ContractError("daily external mass must be kg N/day")
    expected = np.arange(np.datetime64(start, "D"), np.datetime64(stop, "D"))
    actual = np.asarray(dates).astype("datetime64[D]")
    if not np.array_equal(actual, expected):
        raise ContractError("daily calendar contains gaps, duplicates, reordering or wrong coverage")
    if not np.array_equal(np.asarray(reach_ids), np.arange(1, 231)):
        raise ContractError("daily mass reach order must be 1..230")
    labels = list(source_labels)
    if not labels or len(labels) != len(set(labels)) or set(labels) & INTERNAL_FLUXES:
        raise ContractError("external source tags must be distinct and exclude internal transfers")
    return finite_nonnegative(values, "daily_external_mass", shape=(len(expected), 230, len(labels)))


def real_run_gate(fields: Iterable[InputField], required: Iterable[str], *,
                  artifact_kind: str, science_configuration_frozen: bool,
                  common_kernel_passed: bool, formal_dispatch_authorized: bool,
                  check_files: bool = True) -> dict[str, Any]:
    """No model fitting is launched here; this returns an inspectable decision."""
    supplied = list(fields)
    registry = {item.name: item for item in supplied}
    reasons: list[str] = []
    if len(registry) != len(supplied):
        reasons.append("duplicate_input_names")
    for item in supplied:
        reasons.extend(item.validate(check_files=check_files))
    for name in sorted(set(required)):
        item = registry.get(name)
        if item is None:
            reasons.append(f"missing_field:{name}")
        elif item.status not in {"available", "known_zero"}:
            reasons.append(f"required_input_{item.status}:{name}")
    if artifact_kind not in {"real_independent_input", "real_product_input"}:
        reasons.append(f"not_real_independent_input:{artifact_kind}")
    if not science_configuration_frozen:
        reasons.append("science_configuration_not_frozen")
    if not common_kernel_passed:
        reasons.append("common_kernel_not_accepted")
    if not formal_dispatch_authorized:
        reasons.append("this_stage_disallows_formal_fit_dispatch")
    return {"allowed": not reasons, "status": "ready" if not reasons else "blocked",
            "reasons": sorted(set(reasons)), "missing_data_filled": False}


def field_dict(item: InputField) -> dict[str, Any]:
    return asdict(item)
