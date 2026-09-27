"""Build explicit gap registry and the 28 paths; never start a calibration."""
from __future__ import annotations
import json
import csv
from dataclasses import replace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from d29_platform.contracts import InputField, field_dict, file_sha256, payload_sha256, real_run_gate, validate_training_years
from d29_platform.scheduler import dry_run_path


def build_fields():
    old = Path("E:/SPARROW/5_Test/20260917_5/data/domains/FULL24C")
    deposition = Path("E:/SPARROW/5_Test/20260925_1/outputs/deposition_reach")
    vintage = Path("E:/SPARROW/5_Test/20260921_1/evidence/source_year_substitutions.csv")
    hydrology_years = Path("E:/SPARROW/5_Test/20260921_1/evidence/hydrology_source_years.csv")
    vintage_reference = {"path": str(vintage), "sha256": file_sha256(vintage),
                         "grain": "year with separate fertilizer/manure, BNF, deposition, harvested area and crop removal adopted-year columns",
                         "coverage": "2021-2024", "year_substitution_flag_is_not_sufficient": True}
    with vintage.open(encoding="utf-8-sig", newline="") as handle:
        vintage_reference["per_year_source_and_area_records"] = list(csv.DictReader(handle))
    daily_dir = ROOT / "outputs/deposition"
    daily_receipt = daily_dir / "daily_receipt.json"
    conflict_file = daily_dir / "daily_wet_conflicts.csv"
    daily_metadata = {"monthly_source_does_not_certify_complete_daily_input": True}
    reason = "Daily deposition adapter has not produced an accepted complete wet allocation"
    if daily_receipt.is_file():
        daily_info = json.loads(daily_receipt.read_text(encoding="utf-8"))
        daily_metadata.update({"receipt_path": str(daily_receipt), "receipt_sha256": file_sha256(daily_receipt),
                               "producer_status": daily_info.get("status"),
                               "dry_components": "defined; retained in partial artifact", "wet_components": "unknown where monthly rain is zero"})
    if conflict_file.is_file():
        with conflict_file.open(encoding="utf-8-sig", newline="") as handle:
            conflicts = list(csv.DictReader(handle))
        count = len(conflicts)
        reach_months = len({(row["year"], row["month"], row["reach_id"]) for row in conflicts})
        unresolved = sum(float(row["unresolved_kg_n"]) for row in conflicts)
        daily_metadata.update({"conflict_file": str(conflict_file), "conflict_file_sha256": file_sha256(conflict_file),
                               "unresolved_reach_month_components": count, "unresolved_reach_months": reach_months,
                               "unresolved_wet_mass_kg": unresolved})
        reason = f"Nonzero wet deposition with zero monthly H1 rain: {count} reach-month-components in {reach_months} reach-months; no allocation or zero-fill is permitted"
    items = [
        InputField("old_four_source_monthly", "available", "external", "kg N/month",
                   "Original products, monthly day-1 loading; vintage substitutions preserved", str(old / "source_tags.npy"),
                   metadata={"year_evidence": vintage_reference, "source_order": ["fertilizer", "field_manure", "crop_BNF", "land_deposition"]}),
        InputField("old_removal_proxy", "available", "activity", "kg N/month",
                   "Harvest-product removal proxy, not total uptake", str(old / "crop.npy"),
                   metadata={"year_evidence": vintage_reference,
                             "source_year_columns": ["harvested_area_year_used", "crop_removal_year_used"]}),
        InputField("incremental_area", "available", "land_support", "ha",
                   "Frozen FULL24C incremental area, not upstream cumulative area", str(old / "area_ha.npy")),
        InputField("h1", "available", "driver", "mixed_declared_driver_units",
                   "Frozen H1 full-history domain; no recalibration", str(old / "arrays.json"),
                   metadata={"identity_scope": "hash-bearing array manifest; driver hashes independently verified by LAND0 regression",
                             "year_evidence": {"path": str(hydrology_years), "sha256": file_sha256(hydrology_years),
                                               "grain": "year and actual precipitation/PET product identities; no single adopted-year scalar"}}),
        InputField("deposition_monthly_incremental", "available", "external", "kg N/month",
                   "Independently audited input4MIPs drynhx/drynoy/wetnhx/wetnoy; water excluded", str(deposition / "monthly_incremental_reach.csv"),
                   original_year="1961-2024; ScenarioMIP 2023-2024", adopted_year="actual product year", is_scenario=True,
                   metadata={"land_mask": "1985 mask scenario for 1961-1984 and 1986-1989", "four_components_mutually_exclusive": True}),
        InputField("deposition_unknown_area", "unknown", "external", "kg N/month",
                   "Unknown-area mass is separately registered in deposition product", reason="Not assigned to soil/plant land units"),
        InputField("deposition_daily_land_components", "unknown", "external", "kg N/day",
                   "Gregorian daily adapter of audited monthly drynhx/drynoy/wetnhx/wetnoy; complete daily integration is not yet admissible",
                   str(daily_dir / "daily_land_components_kg_n.npy"), reason=reason, metadata=daily_metadata),
        InputField("water_surface_deposition", "excluded", "external", "kg N/month",
                   "Current land-source accounting boundary", reason="Excluded from soil source; not evidence of real zero direct-water input"),
        InputField("mod17_noncrop_production", "unknown", "activity", "kg C/year",
                   "MOD17 constrains noncropland carbon production only", reason="No downloaded, audited reach-aggregated MOD17 product", metadata={"origin": "MOD17"}),
        InputField("crop_activity_ledger", "unknown", "activity", "kg N/day",
                   "CCD and yield material exist but do not yet define complete plant N activities", reason="Crop production/harvest/residue/retranslocation conversion not frozen"),
        InputField("physical_land_units", "unknown", "land_support", "ha",
                   "CLCD data exist; full soil/plant eligibility and historic land-unit contract pending", reason="Land-class aggregation alone does not establish biological eligibility"),
        InputField("land_transition_matrix", "unknown", "land_support", "fraction/day",
                   "Requires independently declared donor-to-recipient stock transfer", reason="No frozen 1961-2024 mass-conserving land transitions"),
        InputField("reconstructed_mineral_external", "unknown", "external", "kg N/day",
                   "S1 source reconstruction", reason="Complete independent source years and calendar unavailable"),
        InputField("reconstructed_organic_external", "unknown", "external", "kg N/day",
                   "S1 source reconstruction", reason="Manure allocation and organic/mineral split not established"),
        InputField("plant_bnf_seed_external", "unknown", "external", "kg N/day",
                   "External plant BNF and seed must not be added again to soil", reason="Plant/soil BNF allocation and seed mass not frozen"),
        InputField("plant_target", "unknown", "activity", "kg N",
                   "Target plant inventory including future planned removals", reason="Missing crop/noncrop phenology and C:N conversion"),
        InputField("plant_potential_demand", "unknown", "activity", "kg N/day",
                   "Potential demand is distinct from realized uptake", reason="Plant demand conversion not frozen"),
        InputField("plant_exclusive_fates", "unknown", "activity", "kg N/day",
                   "Harvest/export/residue return have mutually exclusive destinations", reason="Fates and retranslocation hypotheses not frozen"),
        InputField("initial_land1_stocks", "unknown", "state", "kg N",
                   "1961 stocks require explicit independent assumptions", reason="No accepted partition among P/Sa/Sp/Navail/L"),
        InputField("land1_parameters", "unknown", "parameter", "explicit per-parameter units",
                   "Mineralization, protection, available loss and plant constants; no old M lifetime reuse", reason="Scientific parameter configuration not frozen"),
        InputField("land1_external_partition", "unknown", "activity", "fraction",
                   "Mineral/active organic/protected organic/plant entry mapping", reason="No inferred split from total source; protection and BNF placement require explicit configuration"),
        InputField("plant_export_mapping", "unknown", "activity", "kg N/day",
                   "Harvest, removed residues and other plant exports map once to domain exit", reason="Scientific mutually exclusive fate map is not frozen"),
        InputField("direct_river_source", "unknown", "external", "kg N/day",
                   "Distinct receiving-reach and mass interface", reason="Location and load evidence incomplete; interface disabled, never inserted into soil"),
        InputField("mineralization", "available", "internal", "kg N/day", "Computed internal Sa/Sp to available transfer"),
        InputField("uptake", "available", "internal", "kg N/day", "Computed available-to-plant transfer"),
        InputField("residue_return", "available", "internal", "kg N/day", "Computed plant-to-soil transfer"),
    ]
    identified = []
    for item in items:
        if item.path and Path(item.path).is_file():
            meta = dict(item.metadata)
            if item.name in {"old_four_source_monthly", "old_removal_proxy", "incremental_area"}:
                shape = {"old_four_source_monthly": [768, 230, 4], "old_removal_proxy": [768, 230], "incremental_area": [230]}[item.name]
                meta.update(artifact_kind="npy", shape=shape, reach_ids=list(range(1, 231)))
                if item.name != "incremental_area":
                    meta.update(date_path=str(old / "months.npy"), date_sha256=file_sha256(old / "months.npy"))
            elif item.name == "h1":
                meta.update(artifact_kind="h1_array_manifest", reach_ids=list(range(1, 231)))
            elif item.name == "deposition_monthly_incremental":
                meta.update(artifact_kind="deposition_incremental_csv")
            elif item.name == "deposition_daily_land_components":
                meta.update(artifact_kind="daily_external_npy", shape=[23376, 230, 4],
                            date_path=str(daily_dir / "daily_dates.npy"), date_sha256=file_sha256(daily_dir / "daily_dates.npy"),
                            reach_path=str(daily_dir / "reach_ids.npy"), reach_sha256=file_sha256(daily_dir / "reach_ids.npy"),
                            source_labels=["drynhx", "drynoy", "wetnhx", "wetnoy"])
            item = replace(item, sha256=file_sha256(item.path), metadata=meta)
        elif item.status == "available" and item.role == "internal":
            payload = {"definition": item.provenance, "computed_by": "LAND1 physical kernel"}
            item = replace(item, payload=payload, sha256=payload_sha256(payload), metadata={"artifact_kind": "semantic_definition"})
        identified.append(item)
    return identified


def build_paths():
    combinations = {"C00": ("S0", "LAND0"), "C10": ("S1", "LAND0"),
                    "C01": ("S0", "LAND1"), "C11": ("S1", "LAND1")}
    paths = []
    for configuration, (source, land) in combinations.items():
        for fold, train, evaluation in (("F23", [2021, 2022], [2023]), ("F24", [2021, 2022, 2023], [2024])):
            for entry in (0, 1):
                paths.append({"path_id": f"{configuration}_{fold}_e{entry}", "configuration": configuration,
                              "source_id": source, "land_model_id": land, "mapping_id": "G1_D29_existing",
                              "fold": fold, "entry": entry, "training_years": train, "evaluation_years": evaluation,
                              "initialization": "legal_fold_process_point_recompute_common_objective" if entry == 0 else "preset_entry_1",
                              "allow_inverse_TN_initialization": False, "performance_prerequisites": []})
    for configuration in ("C00", "C11"):
        source, land = combinations[configuration]
        for spatial in ("S56", "S113", "S191"):
            for entry in (0, 1):
                paths.append({"path_id": f"{configuration}_{spatial}_e{entry}", "configuration": configuration,
                              "source_id": source, "land_model_id": land, "mapping_id": "G1_D29_existing",
                              "fold": spatial, "entry": entry, "training_years": [2021, 2022, 2023], "evaluation_years": [2024],
                              "spatial_exclusion": "upstream_closure_plus_downstream_and_shared_reservoir_buffer",
                              "fit_transforms": "legal_training_labels_and_reaches_only",
                              "initialization": "fold_local_preset_only_no_global_TN_fitted_point",
                              "allow_inverse_TN_initialization": False, "performance_prerequisites": []})
    return paths


def required_for(path):
    required = ["h1", "incremental_area"]
    if path["source_id"] == "S0":
        required += ["old_four_source_monthly", "old_removal_proxy"]
    else:
        required += ["deposition_monthly_incremental", "deposition_daily_land_components", "reconstructed_mineral_external",
                     "reconstructed_organic_external", "plant_bnf_seed_external", "crop_activity_ledger",
                     "mod17_noncrop_production"]
    if path["land_model_id"] == "LAND1":
        required += ["physical_land_units", "land_transition_matrix", "plant_target", "plant_potential_demand",
                     "plant_exclusive_fates", "plant_bnf_seed_external", "initial_land1_stocks", "land1_parameters",
                     "land1_external_partition", "plant_export_mapping"]
    return required


def verify_fold_references():
    """Independent support check; old tables are reference labels, never inverse initialization."""
    import pandas as pd
    previous = Path("E:/SPARROW/5_Test/20260921_3")
    folds_file = previous / "configs/folds.json"
    configs = json.loads(folds_file.read_text(encoding="utf-8"))
    topology_file = previous / "data/domains/FULL24C/topology.json"
    topo = json.loads(topology_file.read_text(encoding="utf-8"))
    edges = {(int(a) + 1, int(b) + 1) for a, b in topo["downstream"].items()}
    edges |= {(int(c) + 1, int(row["target"]) + 1) for row in topo["metadata"] for c in row["controls"]}

    def closure(seed, reverse=False):
        seen = set(seed)
        while True:
            extra = {a if reverse else b for a, b in edges if (b if reverse else a) in seen}
            if extra <= seen:
                return seen
            seen |= extra

    result = {}
    names = {"F23": "F23_G_D", "F24": "T24_G_D_H1", **{f"S{x}": f"S{x}_D_H1" for x in (56, 113, 191)}}
    for short, name in names.items():
        config = configs[name]
        path = previous / "data/folds" / name / "train.parquet"
        data = pd.read_parquet(path)
        validate_training_years(data, config["train_years"])
        if file_sha256(path) != config["train_sha256"]:
            raise ValueError(f"frozen fold training hash mismatch: {short}")
        allowed = set(range(1, 231))
        held, buffer = set(), set()
        if short.startswith("S"):
            held = closure({int(short[1:])}, reverse=True)
            buffer = closure(held) - held
            allowed -= held | buffer
            if allowed != set(config["allowed_reaches"]):
                raise ValueError(f"spatial closure mismatch: {short}")
        outside = set(data.reach_id.astype(int)) - allowed
        if outside:
            raise ValueError(f"forbidden station reach in training: {short}: {sorted(outside)}")
        registry = previous / "data/folds" / name / "registry.json"
        result[short] = {"train_path": str(path), "train_sha256": file_sha256(path),
                         "registry_path": str(registry), "registry_sha256": file_sha256(registry),
                         "train_years": config["train_years"], "evaluation_year": config["evaluation_year"],
                         "allowed_reaches": sorted(allowed), "held_reaches": sorted(held), "buffer_reaches": sorted(buffer),
                         "station_count": int(data.station_key.nunique()), "row_count": len(data),
                         "all_training_rows_allowed": True, "hash_verified": True,
                         "usage": "legal label/support reference; new objective scales rebuilt only from these training labels",
                         "reuse_old_fitted_transform_as_new_objective": False}
    return {"folds": result, "source_config": {"path": str(folds_file), "sha256": file_sha256(folds_file)},
            "topology": {"path": str(topology_file), "sha256": file_sha256(topology_file)},
            "NSE": "not_applicable_training_support"}


def summarize_acceptance(records, fixed_checkpoints):
    """Public software acceptance and old optimizer sufficiency are distinct."""
    required = {"LAND1", "legacy_F23", "legacy_F24", "chain", "metrics", "numerical_validator"}
    names = [row["name"] for row in records]
    complete = required <= set(names) and len(names) == len(set(names)) and all(row["status"] == "passed" for row in records)
    public_status = "passed" if complete else "failed" if any(row["status"] == "failed" for row in records) else "pending"
    expected_tags = {"U_W7_joint_b4", "U_D1_joint_b4", "L3_W7_joint_b4", "L3_D1_joint_b4"}
    fixed_complete = len(fixed_checkpoints) == 4 and {row["tag"] for row in fixed_checkpoints} == expected_tags and all(row["status"] == "complete" for row in fixed_checkpoints)
    return {"common_kernel_passed": complete, "common_kernel_status": public_status,
            "public_acceptance_receipts": records,
            "fixed_checkpoint_review_status": "complete" if fixed_complete else "pending",
            "fixed_checkpoint_reviews": fixed_checkpoints,
            "old_optimizer_insufficiency_is_common_kernel_failure": False,
            "formal_calibration_still_disabled": True}


def collect_acceptance(root=ROOT):
    root = Path(root)
    locations = {"LAND1": "evidence/land1_acceptance.json",
                 "legacy_F23": "outputs/legacy/F23/receipt.json",
                 "legacy_F24": "outputs/legacy/F24/receipt.json",
                 "chain": "outputs/chain/receipt.json", "metrics": "outputs/metrics/acceptance.json",
                 "numerical_validator": "outputs/numerics/synthetic_gate_tests.json"}
    records = []
    for name, relative in locations.items():
        path = root / relative
        record = {"name": name, "receipt": str(path), "status": "pending", "hash_coverage": []}
        if not path.is_file():
            record["reason"] = "receipt_not_yet_available"
            records.append(record)
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            record["receipt_sha256"] = file_sha256(path)
            if payload.get("passed") is not True:
                record.update(status="failed", reason="receipt_has_no_positive_pass")
                records.append(record)
                continue
            hashes = {}
            hashes.update({root / key: value for key, value in payload.get("files", {}).items()})
            for key, value in payload.get("implementation_sha256", {}).items():
                normalized = key.replace("\\", "/")
                # Early metrics receipts used module basenames; final receipts
                # use root-relative source paths. Both have explicit roots.
                target = root / normalized if "/" in normalized else root / "d29_platform" / normalized
                hashes[target] = value
            hashes.update({path.parent / key: value for key, value in payload.get("output_hashes", {}).items()})
            bad = []
            for target, expected in hashes.items():
                matches = target.is_file() and file_sha256(target) == expected
                record["hash_coverage"].append({"path": str(target), "expected_sha256": expected, "matched": matches})
                if not matches:
                    bad.append(str(target))
            if bad:
                record.update(status="pending", reason="recorded_artifact_hash_changed_or_missing", unmatched=bad)
            else:
                valid = True
                if name == "LAND1":
                    valid = payload.get("unit_tests", {}).get("passed") is True and payload.get("full_history", {}).get("passed") is True
                elif name.startswith("legacy"):
                    valid = (payload.get("days") == 23376 and payload.get("reaches") == 230 and
                             payload.get("parameters") == 31 and payload.get("independent_objective_error", 1) <= 1e-8 * (1 + abs(payload.get("objective", 0))) and
                             payload.get("prediction_max_error", 1) <= 1e-6 and payload.get("gradient_max_error", 1) <= 1e-6)
                elif name == "chain":
                    directions = payload.get("gradient_directions", [])
                    valid = (payload.get("days") == 23376 and payload.get("reaches") == 230 and
                             bool(directions) and all(row.get("passed") is True for row in directions) and
                             payload.get("local_balance_kg", 1) <= 1e-6 and payload.get("network_relative_error", 1) <= 1e-10)
                elif name in {"metrics", "numerical_validator"}:
                    valid = payload.get("test_count", payload.get("tests_run", 0)) > 0 and payload.get("failures") == 0 and payload.get("errors") == 0
                record.update(status="passed" if valid else "failed", reason="receipt_and_recorded_hashes_verified" if valid else "required_measurements_missing_or_failed")
            record["missing_source_hash_coverage"] = not bool(payload.get("files") or payload.get("implementation_sha256"))
            record["source_identity_note"] = "No source hash is retroactively invented; final implementation manifest/independent review binds current delivery."
        except (OSError, ValueError, KeyError, TypeError) as exc:
            record.update(status="pending", reason=f"receipt_not_readable:{exc}")
        records.append(record)
    fixed = []
    for tag in ("U_W7_joint_b4", "U_D1_joint_b4", "L3_W7_joint_b4", "L3_D1_joint_b4"):
        path = root / "outputs/numerics" / f"{tag}_audit.json"
        row = {"tag": tag, "receipt": str(path), "status": "pending"}
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                checkpoint = Path(data["metadata"]["checkpoint"])
                valid = (data.get("evidence_kind") == "frozen_joint_checkpoint_diagnostic_not_refit" and
                         data.get("all_shared_process_coordinates_checked") is True and data.get("refit_started") is False and
                         checkpoint.is_file() and file_sha256(checkpoint) == data["checkpoint_sha256"])
                row.update(status="complete" if valid else "pending", receipt_sha256=file_sha256(path),
                           checkpoint_sha256=data.get("checkpoint_sha256"), projected_gradient=data.get("projected_gradient"),
                           solver_sufficient=data.get("projected_gradient", 1) <= 1e-5,
                           interpretation="fixed diagnostic completion does not certify old optimizer convergence or full high-dimensional gradient coverage")
            except (OSError, ValueError, KeyError, TypeError) as exc:
                row["reason"] = f"receipt_not_readable:{exc}"
        fixed.append(row)
    return summarize_acceptance(records, fixed)


def main():
    fields = build_fields()
    paths = build_paths()
    fold_audit = verify_fold_references()
    acceptance = collect_acceptance()
    for directory in (ROOT / "configs", ROOT / "outputs/contracts"):
        directory.mkdir(parents=True, exist_ok=True)
    registry = [field_dict(item) for item in fields]
    for row in registry:
        if row["path"] and Path(row["path"]).is_file():
            row["sha256"] = file_sha256(row["path"])
    contract = {"version": "1.0", "reach_count": 230, "history": [1961, 2024],
                "missing_policy": "unknown_never_imputed; no copying years/default zero/TN inversion",
                "fields": registry, "source_scaling": {"owner": "external_mass_interface", "history_scope": "full",
                    "applications": 1, "internal_transfers_scaled": False, "prior_only_theta30_forbidden": True},
                "real_land1_scientific_configuration": "not_frozen",
                "legacy_compatibility": "old source multiplier retained only in exact legacy kernel; new path must not apply it twice"}
    results = []
    for path in paths:
        path["legal_fold_reference"] = fold_audit["folds"][path["fold"]]
        path["required_inputs"] = required_for(path)
        gate = real_run_gate(fields, path["required_inputs"], artifact_kind="real_product_input" if path["source_id"] == "S0" else "real_independent_input",
                             science_configuration_frozen=False, common_kernel_passed=acceptance["common_kernel_passed"],
                             formal_dispatch_authorized=False)
        if acceptance["fixed_checkpoint_review_status"] != "complete":
            gate["reasons"].append("fixed_checkpoint_numerical_review_pending")
            gate["reasons"].sort()
            gate.update(allowed=False, status="blocked")
        gate["common_kernel_status"] = acceptance["common_kernel_status"]
        gate["fixed_checkpoint_review_status"] = acceptance["fixed_checkpoint_review_status"]
        path["status"] = gate["status"]
        path["blocking_reasons"] = gate["reasons"]
        results.append(dry_run_path(path, gate, {"dispatch_allowed": False, "reason": "dry_run_only_no_resource_reservation"}))
    matrix = {"path_count": len(paths), "time_paths": 16, "spatial_paths": 12,
              "formal_dispatch_enabled": False, "C11_performance_gate": None,
              "common_kernel_passed": acceptance["common_kernel_passed"],
              "common_kernel_status": acceptance["common_kernel_status"],
              "fixed_checkpoint_review_status": acceptance["fixed_checkpoint_review_status"],
              "common_objective_requires_C00_refit": True, "paths": paths}
    for relative, value in (("configs/input_contract.json", contract), ("configs/path_matrix.json", matrix),
                            ("outputs/contracts/fold_reference_audit.json", fold_audit),
                            ("outputs/contracts/common_acceptance_status.json", acceptance),
                            ("outputs/contracts/dry_run.json", {"paths": results, "fit_calls": 0,
                             "all_blocked": all(not row["input_gate"]["allowed"] for row in results), "NSE": "not_applicable"})):
        (ROOT / relative).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"paths": len(paths), "fits_started": 0, "all_blocked": True}))


if __name__ == "__main__":
    main()
