"""Build an auditable reservoir fact/prior registry.

The registry separates physical facts, qualitative operating evidence and
actual model priors.  A literature statement never becomes a deterministic
monthly release rule.  Unsupported user synthesis is retained for provenance
but is not eligible for calibration until an auditable source is attached.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
STAGE = ROOT / "5_Test" / "20260828_13"
INVENTORY = STAGE / "outputs" / "reservoir_entity_inventory.parquet"
MANUAL = STAGE / "inputs" / "reservoir_specific_evidence.csv"
VERIFIED = STAGE / "inputs" / "reservoir_verified_evidence.csv"
HARD_GATE = (
    ROOT
    / "5_Test"
    / "20260828_17"
    / "inputs"
    / "reservoir_hard_gate_evidence.csv"
)
OUTPUT = STAGE / "outputs" / "literature_prior_registry.parquet"
COVERAGE_OUTPUT = STAGE / "outputs" / "priority_reservoir_evidence_coverage.parquet"
QA = STAGE / "reports" / "literature_prior_registry_qa.json"

REQUIRED_MANUAL_COLUMNS = {
    "reservoir_entity_id",
    "reservoir_name_zh",
    "fact_name",
    "value_text",
    "value_numeric",
    "unit",
    "constraint_class",
    "model_parameter",
    "prior_family",
    "prior_location",
    "prior_scale",
    "evidence_grade",
    "source_type",
    "source_title",
    "source_doi_or_url",
    "source_date",
    "eligible_for_2006_2018_prior",
    "conflict_group",
    "claim_boundary",
    "notes",
}

REQUIRED_VERIFIED_COLUMNS = REQUIRED_MANUAL_COLUMNS | {
    "publication_date",
    "effective_date",
    "data_period_start",
    "data_period_end",
    "eligible_for_development_prior",
    "leakage_status",
    "source_access_status",
    "applicability_window",
    "constraint_scope",
    "operator_eligibility",
    "hard_or_soft",
}

# These are evidence-bearing system concepts, not additional reservoir storage
# states.  They are intentionally absent from the physical inventory.
EVIDENCE_ONLY_ENTITY_IDS = {
    "SYSTEM_DONGJIANG_THREE_RESERVOIRS",
    "TRANSFER_DONGSHEN",
}


def explicit_bool(value: object) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"Expected explicit evidence eligibility boolean, got {value!r}")


def load_hard_gate_evidence() -> pd.DataFrame:
    """Map Stage 17 immutable gate facts into the authoritative schema.

    The compact Stage 17 table records activation, capacity-version and
    represented-domain facts.  This adapter makes the mapping explicit while
    preserving the source row and its claim boundary.  It never turns a fact
    into a fixed release schedule.
    """

    compact = pd.read_csv(HARD_GATE, dtype={"source_url": "string"})
    required = {
        "reservoir_entity_id",
        "reservoir_name_zh",
        "fact_name",
        "value_numeric",
        "unit",
        "effective_date",
        "publication_date",
        "evidence_grade",
        "source_type",
        "source_title",
        "source_url",
        "development_eligibility",
        "model_action",
        "claim_boundary",
    }
    missing = required.difference(compact.columns)
    if missing:
        raise RuntimeError(f"Stage 17 hard-gate evidence is missing columns: {sorted(missing)}")

    activation_facts = {"first_impoundment", "first_generation"}
    diagnostic_facts = {"all_units_operational"}
    capacity_facts = {"phase1_total_capacity"}
    area_facts = {"official_control_area"}
    allowed_facts = activation_facts | diagnostic_facts | capacity_facts | area_facts
    unexpected = sorted(set(compact["fact_name"].astype(str)).difference(allowed_facts))
    if unexpected:
        raise RuntimeError(f"Unmapped Stage 17 hard-gate facts: {unexpected}")

    rows: list[dict[str, object]] = []
    for source in compact.itertuples(index=False):
        fact = str(source.fact_name)
        publication = pd.to_datetime(source.publication_date, errors="coerce")
        published_after_2018 = bool(pd.notna(publication) and publication.year > 2018)
        action = str(source.model_action)
        eligibility = str(source.development_eligibility)
        development_eligible = (
            eligibility
            in {
                "eligible_static_operational_milestone",
                "eligible_static_design_fact",
                "post_2018_static_fact_exception",
            }
            and action != "diagnostic_only"
        )
        extension_only = eligibility.startswith("extension_only")
        provisional = eligibility == "not_yet_verified_for_unlock"

        if development_eligible and published_after_2018:
            leakage_status = "post_2018_publication_static_design_exception"
        elif development_eligible:
            leakage_status = (
                "development_period_publication_static_and_pre2015_operating_facts_only"
            )
        elif extension_only:
            leakage_status = "post_2018_extension_static_milestone_diagnostic_only"
        elif provisional:
            leakage_status = "blocked_provisional_source_not_fit_eligible"
        else:
            leakage_status = "verified_static_fact_diagnostic_only"

        if fact in activation_facts | diagnostic_facts:
            value_text = str(source.effective_date)
            constraint_class = "activation_milestone"
            model_parameter = "operator_start" if action == "activate_storage_operator" else "none"
            prior_family = "immutable_date_lock" if model_parameter == "operator_start" else "none"
            constraint_scope = (
                "immutable_operational_milestone"
                if model_parameter == "operator_start"
                else "diagnostic_activation_history"
            )
        elif fact in capacity_facts:
            value_text = str(source.value_numeric)
            constraint_class = "physical_capacity_boundary"
            model_parameter = "capacity_m3"
            prior_family = "source_fact_with_audit"
            constraint_scope = "immutable_phase_specific_design_fact"
        else:
            value_text = str(source.value_numeric)
            constraint_class = "partial_domain_geometry"
            model_parameter = "domain_storage_scale_or_local_capture_fraction"
            prior_family = "source_fact_with_audit"
            constraint_scope = "immutable_control_area_for_partial_domain_mapping"

        if extension_only:
            operator_eligibility = "extension_operator_only_not_development_fit"
        elif provisional:
            operator_eligibility = "blocked_pending_direct_source"
        elif action == "diagnostic_only":
            operator_eligibility = "diagnostic_only"
        else:
            operator_eligibility = "operator_ready_subject_to_integrated_lock"

        hard_or_soft = (
            "diagnostic_only"
            if action == "diagnostic_only" or provisional
            else "hard"
        )
        rows.append(
            {
                "reservoir_entity_id": source.reservoir_entity_id,
                "reservoir_name_zh": source.reservoir_name_zh,
                "fact_name": fact,
                "value_text": value_text,
                "value_numeric": source.value_numeric,
                "unit": source.unit,
                "constraint_class": constraint_class,
                "model_parameter": model_parameter,
                "prior_family": prior_family,
                "prior_location": source.value_numeric,
                "prior_scale": np.nan,
                "evidence_grade": source.evidence_grade,
                "source_type": source.source_type,
                "source_title": source.source_title,
                "source_doi_or_url": source.source_url,
                "source_date": source.publication_date,
                "publication_date": source.publication_date,
                "effective_date": source.effective_date,
                "data_period_start": None,
                "data_period_end": None,
                "eligible_for_2006_2018_prior": development_eligible,
                "eligible_for_development_prior": development_eligible,
                "leakage_status": leakage_status,
                "source_access_status": (
                    "provisional_indirect_reference"
                    if provisional
                    else "verified_registered_source"
                ),
                "applicability_window": (
                    f"from {source.effective_date}"
                    if fact in activation_facts and action == "activate_storage_operator"
                    else "registered static-fact scope only"
                ),
                "constraint_scope": constraint_scope,
                "operator_eligibility": operator_eligibility,
                "hard_or_soft": hard_or_soft,
                "conflict_group": f"{source.reservoir_entity_id}_{fact}",
                "claim_boundary": source.claim_boundary,
                "notes": (
                    f"Stage 17 action={action}; eligibility={eligibility}. "
                    "No monthly release outcome is imported."
                ),
                "record_origin": "stage17_hard_gate_verified_evidence",
            }
        )
    return pd.DataFrame(rows)


def load_evidence_sources() -> pd.DataFrame:
    """Load legacy user synthesis and independently verified evidence.

    The two files remain separate so that verified facts never overwrite the
    user's original table.  The generated Parquet registry is the sole
    authoritative merged view used by later stages.
    """

    provisional = pd.read_csv(MANUAL, dtype={"source_doi_or_url": "string"})
    missing = REQUIRED_MANUAL_COLUMNS.difference(provisional.columns)
    if missing:
        raise RuntimeError(f"Provisional evidence registry is missing columns: {sorted(missing)}")
    provisional = provisional.copy()
    if "publication_date" not in provisional:
        provisional["publication_date"] = provisional["source_date"]
    for column in ["effective_date", "data_period_start", "data_period_end"]:
        if column not in provisional:
            provisional[column] = None
    if "leakage_status" not in provisional:
        provisional["leakage_status"] = "blocked_unverified"
    for column, default in {
        "source_access_status": "user_supplied_not_independently_verified",
        "applicability_window": None,
        "constraint_scope": "provenance_only",
        "operator_eligibility": "not_applicable_until_verified",
        "hard_or_soft": "diagnostic_only",
    }.items():
        if column not in provisional:
            provisional[column] = default
    provisional["eligible_for_2006_2018_prior"] = provisional[
        "eligible_for_2006_2018_prior"
    ].map(explicit_bool)
    provisional["eligible_for_development_prior"] = provisional[
        "eligible_for_2006_2018_prior"
    ]
    provisional["record_origin"] = "user_synthesis_unverified"

    verified = pd.read_csv(VERIFIED, dtype={"source_doi_or_url": "string"})
    missing = REQUIRED_VERIFIED_COLUMNS.difference(verified.columns)
    if missing:
        raise RuntimeError(f"Verified evidence registry is missing columns: {sorted(missing)}")
    verified = verified.copy()
    for column in ["eligible_for_2006_2018_prior", "eligible_for_development_prior"]:
        verified[column] = verified[column].map(explicit_bool)
    if not (
        verified["eligible_for_2006_2018_prior"]
        == verified["eligible_for_development_prior"]
    ).all():
        raise RuntimeError("Legacy and authoritative eligibility flags disagree in verified evidence")
    verified["record_origin"] = "verified_literature_or_official_evidence"

    hard_gate = load_hard_gate_evidence()
    return pd.concat([provisional, verified, hard_gate], ignore_index=True, sort=False)


def dataset_fact_rows(inventory: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    fields = [
        ("capacity_million_m3_grand", "total_capacity", "million_m3"),
        ("surface_area_km2_grand", "surface_area", "km2"),
        ("catchment_km2_grand", "dam_catchment_area", "km2"),
        ("commissioning_year_grand", "commissioning_year", "year"),
    ]
    for reservoir in inventory.itertuples(index=False):
        if pd.isna(reservoir.grand_id):
            continue
        for column, fact_name, unit in fields:
            value = getattr(reservoir, column)
            if pd.isna(value):
                continue
            rows.append(
                {
                    "reservoir_entity_id": reservoir.reservoir_entity_id,
                    "reservoir_name_zh": reservoir.reservoir_name_zh,
                    "fact_name": fact_name,
                    "value_text": str(value),
                    "value_numeric": float(value),
                    "unit": unit,
                    "constraint_class": "soft_prior_source",
                    "model_parameter": (
                        "capacity_m3"
                        if fact_name == "total_capacity"
                        else "metadata_or_geometry"
                    ),
                    "prior_family": "source_uncertainty_pending",
                    "prior_location": float(value),
                    "prior_scale": np.nan,
                    "evidence_grade": "C_structured_global_dataset",
                    "source_type": "GRanD_dataset_attribute",
                    "source_title": "Global Reservoir and Dam Database (GRanD)",
                    "source_doi_or_url": "https://doi.org/10.1890/100125",
                    "source_date": "2011",
                    "publication_date": "2011",
                    "effective_date": (
                        str(int(reservoir.commissioning_year_grand))
                        if not pd.isna(reservoir.commissioning_year_grand)
                        else None
                    ),
                    "data_period_start": None,
                    "data_period_end": None,
                    "leakage_status": "static_metadata_allowed_subject_to_conflict_audit",
                    "eligible_for_2006_2018_prior": True,
                    "conflict_group": None,
                    "claim_boundary": (
                        "Dataset metadata is a prior source, not a monthly operation record; "
                        "capacity conflicts with authoritative reservoir sources must remain explicit."
                    ),
                    "notes": "Automatically generated from the locked in-domain inventory.",
                    "record_origin": "generated_GRanD_inventory",
                }
            )
    return rows


def main() -> None:
    inventory = pd.read_parquet(INVENTORY)
    manual = load_evidence_sources()
    known_entities = set(inventory["reservoir_entity_id"].astype(str)) | EVIDENCE_ONLY_ENTITY_IDS
    unknown = sorted(set(manual["reservoir_entity_id"].astype(str)).difference(known_entities))
    if unknown:
        raise RuntimeError(f"Evidence references unknown reservoir entities: {unknown}")
    if manual["source_doi_or_url"].isna().any():
        raise RuntimeError("Every manual evidence row needs an explicit provenance identifier")
    if (
        manual["eligible_for_2006_2018_prior"]
        & manual["evidence_grade"].astype(str).str.contains("unverified", case=False)
    ).any():
        raise RuntimeError("Unverified synthesis cannot be eligible for a calibration prior")
    if manual.loc[
        manual["eligible_for_2006_2018_prior"], "leakage_status"
    ].astype(str).str.startswith("blocked").any():
        raise RuntimeError("A blocked leakage record cannot be eligible for calibration")
    allowed_eligible_leakage_prefixes = (
        "static_metadata_allowed_subject_to_conflict_audit",
        "development_period_publication_static_and_pre2015_operating_facts_only",
        "post_2018_publication_static_design_exception",
        "official_regulation_effective_mid_development",
    )
    invalid_leakage = manual[
        manual["eligible_for_development_prior"]
        & ~manual["leakage_status"].astype(str).str.startswith(
            allowed_eligible_leakage_prefixes
        )
    ]
    if not invalid_leakage.empty:
        raise RuntimeError(
            "Eligible evidence has an unregistered leakage status:\n"
            + invalid_leakage[
                ["reservoir_entity_id", "fact_name", "leakage_status"]
            ].to_string(index=False)
        )
    eligible_post_2018 = manual[
        manual["eligible_for_development_prior"]
        & (pd.to_datetime(manual["publication_date"], errors="coerce").dt.year > 2018)
    ]
    invalid_post_2018 = eligible_post_2018[
        ~eligible_post_2018["leakage_status"].astype(str).str.startswith(
            "post_2018_publication_static_design_exception"
        )
    ]
    if not invalid_post_2018.empty:
        raise RuntimeError(
            "Post-2018 evidence can inform development only through the registered immutable "
            "static-design exception:\n"
            + invalid_post_2018[
                ["reservoir_entity_id", "fact_name", "publication_date", "leakage_status"]
            ].to_string(index=False)
        )
    data_period_end_year = pd.to_numeric(manual["data_period_end"], errors="coerce")
    invalid_heldout_period = manual[
        manual["eligible_for_development_prior"]
        & (data_period_end_year > 2018)
        & ~manual["leakage_status"].astype(str).str.startswith(
            "post_2018_publication_static_design_exception"
        )
    ]
    if not invalid_heldout_period.empty:
        raise RuntimeError(
            "Evidence using post-2018 data can inform development only through immutable "
            "static-design facts:\n"
            + invalid_heldout_period[
                ["reservoir_entity_id", "fact_name", "data_period_end", "leakage_status"]
            ].to_string(index=False)
        )
    mid_development_rules = manual[
        manual["leakage_status"].astype(str).str.startswith(
            "official_regulation_effective_mid_development"
        )
    ]
    if (
        mid_development_rules["effective_date"].isna().any()
        or mid_development_rules["applicability_window"].isna().any()
    ):
        raise RuntimeError("Mid-development regulations require effective date and applicability window")

    generated = pd.DataFrame(dataset_fact_rows(inventory))
    generated["eligible_for_development_prior"] = generated[
        "eligible_for_2006_2018_prior"
    ]
    registry = pd.concat([generated, manual], ignore_index=True, sort=False)
    registry["evidence_record_id"] = [
        f"RES_EVID_{index + 1:04d}" for index in range(len(registry))
    ]
    registry = registry[
        ["evidence_record_id"]
        + [column for column in registry.columns if column != "evidence_record_id"]
    ]
    duplicate_key = [
        "reservoir_entity_id",
        "fact_name",
        "value_text",
        "source_doi_or_url",
    ]
    if registry.duplicated(duplicate_key).any():
        duplicates = registry.loc[registry.duplicated(duplicate_key, keep=False), duplicate_key]
        raise RuntimeError(f"Duplicate evidence rows found:\n{duplicates.to_string(index=False)}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    QA.parent.mkdir(parents=True, exist_ok=True)
    registry.to_parquet(OUTPUT, index=False)

    priority = pd.read_parquet(
        STAGE / "outputs" / "priority_reservoir_topology_audit.parquet"
    )
    priority = priority[
        priority["priority_tier"] == "priority_13_artificial_reservoirs"
    ].copy()
    remote = pd.read_parquet(
        STAGE / "outputs" / "reservoir_remote_observation_coverage.parquet"
    )
    evidence_counts = (
        registry.groupby("reservoir_entity_id", dropna=False)
        .agg(
            evidence_record_count=("evidence_record_id", "size"),
            eligible_prior_record_count=("eligible_for_development_prior", "sum"),
            official_A_record_count=(
                "evidence_grade",
                lambda values: int(pd.Series(values).astype(str).str.startswith("A").sum()),
            ),
            peer_reviewed_B_record_count=(
                "evidence_grade",
                lambda values: int(pd.Series(values).astype(str).str.startswith("B").sum()),
            ),
            dataset_C_record_count=(
                "evidence_grade",
                lambda values: int(pd.Series(values).astype(str).str.startswith("C").sum()),
            ),
            unverified_record_count=(
                "evidence_grade",
                lambda values: int(
                    pd.Series(values).astype(str).str.contains("unverified", case=False).sum()
                ),
            ),
        )
        .reset_index()
    )
    coverage = priority.merge(evidence_counts, how="left", on="reservoir_entity_id")
    coverage = coverage.merge(
        remote[
            ["grand_id", "wse_available", "swa_available", "rwsc_available"]
        ],
        how="left",
        on="grand_id",
    )
    count_columns = [
        "evidence_record_count",
        "eligible_prior_record_count",
        "official_A_record_count",
        "peer_reviewed_B_record_count",
        "dataset_C_record_count",
        "unverified_record_count",
    ]
    coverage[count_columns] = coverage[count_columns].fillna(0).astype(int)
    coverage["direct_rule_source_status"] = np.where(
        (coverage["official_A_record_count"] + coverage["peer_reviewed_B_record_count"]) > 0,
        "A_or_B_evidence_present_requires_field_level_classification",
        "NO_DIRECT_RULE_SOURCE",
    )
    coverage["current_modeling_status"] = np.where(
        coverage["operator_authorized"].fillna(False),
        "topology_authorized_generic_or_hierarchical_prior_only",
        "blocked_topology_or_activation",
    )
    coverage.to_parquet(COVERAGE_OUTPUT, index=False)

    qa = {
        "stage": "20260828_13",
        "status": "PASS",
        "records": int(len(registry)),
        "reservoir_entities_with_evidence": int(registry["reservoir_entity_id"].nunique()),
        "manual_records": int(
            registry["record_origin"].isin(
                [
                    "user_synthesis_unverified",
                    "verified_literature_or_official_evidence",
                    "stage17_hard_gate_verified_evidence",
                ]
            ).sum()
        ),
        "unverified_user_synthesis_records": int(
            (registry["record_origin"] == "user_synthesis_unverified").sum()
        ),
        "verified_literature_or_official_records": int(
            (registry["record_origin"] == "verified_literature_or_official_evidence").sum()
        ),
        "stage17_hard_gate_verified_records": int(
            (registry["record_origin"] == "stage17_hard_gate_verified_evidence").sum()
        ),
        "generated_grand_records": int((registry["record_origin"] == "generated_GRanD_inventory").sum()),
        "eligible_prior_records": int(registry["eligible_for_development_prior"].fillna(False).sum()),
        "fit_period": "2010-2015",
        "development_selection_period": "2016-2018",
        "legacy_field_warning": (
            "eligible_for_2006_2018_prior is retained only for source-file compatibility; "
            "eligible_for_development_prior is authoritative."
        ),
        "unverified_records": int(
            registry["evidence_grade"].astype(str).str.contains("unverified", case=False).sum()
        ),
        "priority_reservoirs_with_A_or_B_records": int(
            (
                coverage["official_A_record_count"]
                + coverage["peer_reviewed_B_record_count"]
                > 0
            ).sum()
        ),
        "priority_reservoirs_without_direct_rule_source": int(
            (coverage["direct_rule_source_status"] == "NO_DIRECT_RULE_SOURCE").sum()
        ),
        "evidence_only_system_entities": sorted(EVIDENCE_ONLY_ENTITY_IDS),
        "hard_boundary": (
            "Only evidence rows explicitly marked eligible may inform 2010-2015 fitting priors "
            "or the registered 2016-2018 development selection. "
            "Mid-development regulations retain their effective-date window; post-2018 sources "
            "are limited to immutable static-design exceptions. No row is a fixed monthly release schedule."
        ),
    }
    QA.write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(qa, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
