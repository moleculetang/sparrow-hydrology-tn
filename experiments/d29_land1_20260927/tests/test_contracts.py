from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import subprocess
import sys
import os
import unittest
import numpy as np

from d29_platform.contracts import (
    ContractError, InputField, SourceScale, finite_nonnegative, real_run_gate,
    validate_budget_semantics, validate_land_support, validate_parameter_responsibility,
    validate_training_years, validate_transition, payload_sha256, file_sha256, validate_daily_mass_bundle)
from d29_platform.identity import RunIdentity, IdentityMismatch, compare_identity, four_corner_identity
from d29_platform.scheduler import PathLedger, resource_gate
from d29_platform.support import NetworkSupport


class ContractTests(unittest.TestCase):
    def artifact(self, status="available", payload=None):
        payload = [0.0] if payload is None else payload
        return InputField("x", status, "external", "kg N/day", "fixture", reason="explicit fixture", payload=payload,
                          sha256=payload_sha256(payload), metadata={"artifact_kind": "embedded_array", "shape": [1]})

    def gate(self, fields, **kw):
        arguments = dict(required=["x"], artifact_kind="real_independent_input", science_configuration_frozen=True,
                         common_kernel_passed=True, formal_dispatch_authorized=True, check_files=False)
        arguments.update(kw)
        return real_run_gate(fields, **arguments)

    def test_unknown_excluded_zero_are_distinct(self):
        for status in ("unknown", "excluded"):
            self.assertFalse(self.gate([InputField("x", status, "external", "kg N/day", "test", reason="explicit")])["allowed"])
        self.assertTrue(self.gate([self.artifact("known_zero")])["allowed"])
        self.assertFalse(self.gate([self.artifact("known_zero", [1.0])])["allowed"])
        self.assertFalse(self.gate([InputField("x", "known_zero", "external", "kg N/day", "test")])["allowed"])

    def test_real_gate_rejects_missing_synthetic_tn_and_unaccepted(self):
        self.assertFalse(self.gate([])["allowed"])
        field = self.artifact()
        for kind in ("synthetic_fixture", "contemporaneous_TN_inverse"):
            self.assertFalse(self.gate([field], artifact_kind=kind)["allowed"])
        for flag in ("science_configuration_frozen", "common_kernel_passed", "formal_dispatch_authorized"):
            self.assertFalse(self.gate([field], **{flag: False})["allowed"])

    def test_available_is_not_self_certifying(self):
        self.assertFalse(self.gate([InputField("x", "available", "external", "kg N/day", "trust me")])["allowed"])
        item = replace(self.artifact(), sha256="0" * 64)
        self.assertFalse(self.gate([item])["allowed"])
        item = replace(self.artifact(), metadata={"artifact_kind": "embedded_array", "shape": [2]})
        self.assertFalse(self.gate([item])["allowed"])

    def test_core_unit_shape_and_reach_axis_checked(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / "work") as directory:
            path = Path(directory) / "area.npy"
            np.save(path, np.ones(230))
            item = InputField("incremental_area", "available", "land_support", "ha", "fixture", str(path),
                              sha256=file_sha256(path), metadata={"artifact_kind": "npy", "shape": [230], "reach_ids": list(range(1, 231))})
            self.assertFalse(item.validate())
            self.assertTrue(replace(item, unit="m2").validate())
            self.assertTrue(replace(item, metadata={"artifact_kind": "npy", "shape": [230], "reach_ids": [1] * 230}).validate())
            np.save(path, np.ones(231))
            self.assertTrue(item.validate())

    def test_nan_not_a_missing_data_representation(self):
        for values in ([1, np.nan], [np.inf], [-1]):
            with self.assertRaises(ContractError):
                finite_nonnegative(values, "mass")

    def test_no_duplicate_crop_soil_or_impervious_default(self):
        validate_land_support([10], [20], stock_area_ha=[10])
        with self.assertRaises(ContractError):
            validate_land_support([10], [20], stock_area_ha=[20])
        with self.assertRaises(ContractError):
            validate_land_support([10], [20], stock_area_ha=[10], impervious_soil_enabled=True)

    def test_semantic_negative_controls(self):
        for extra in ({"external_kind": "net_surplus"}, {"residue_return_as_external": True},
                      {"crop_accesses_noncrop_pool": True}, {"total_uptake_from_harvest_only": True}):
            args = dict(external_kind="gross", demand_deducted=True)
            args.update(extra)
            with self.assertRaises(ContractError):
                validate_budget_semantics(**args)
        validate_budget_semantics(external_kind="gross", demand_deducted=True)
        for item in (InputField("mineralization", "available", "external", "kg N/day", "fixture"),
                     InputField("mod17", "available", "external", "kg N/day", "fixture", metadata={"origin": "MOD17"}),
                     InputField("dep", "available", "external", "kg N/day", "fixture", metadata={"value_kind": "upstream_cumulative_exposure"})):
            self.assertTrue(item.validate())

    def test_transitions_conserve_all_donor_stocks(self):
        matrix = validate_transition([[.8, .2], [.1, .9]])
        np.testing.assert_allclose(np.array([3., 7.]) @ matrix, [3.1, 6.9])
        with self.assertRaises(ContractError):
            validate_transition([[1, .2], [0, 1]])

    def test_external_source_scales_full_history_once(self):
        raw = np.array([[1., 2.], [3., 4.]])
        actual = SourceScale(np.log(2)).apply({"fertilizer": raw}, roles={"fertilizer": "external"})
        np.testing.assert_allclose(actual["fertilizer"], raw * 2)
        np.testing.assert_array_equal(raw, [[1, 2], [3, 4]])
        with self.assertRaises(ContractError):
            SourceScale(0, already_applied=True).apply({"fertilizer": raw}, roles={"fertilizer": "external"})
        with self.assertRaises(ContractError):
            SourceScale(0).apply({"uptake": raw}, roles={"uptake": "internal"})
        with self.assertRaises(ContractError):
            SourceScale(0).apply({}, roles={})

    def test_inactive_prior_only_theta30_rejected(self):
        with self.assertRaises(ContractError):
            validate_parameter_responsibility([{"name": "theta30", "role": "source_multiplier", "physical_targets": []}])
        validate_parameter_responsibility([{"name": "eta", "role": "source_multiplier",
            "physical_targets": ["external_mass"], "owner": "external_mass_interface", "history_scope": "full"}])

    def test_training_gate(self):
        validate_training_years({"year": [2021, 2022]}, [2021, 2022])
        with self.assertRaises(ContractError):
            validate_training_years({"year": [2021, 2023]}, [2021, 2022])

    def test_daily_mass_boundary_identity(self):
        days = np.arange(np.datetime64("2024-02-28"), np.datetime64("2024-03-02"))
        values = np.ones((3, 230, 2))
        kwargs = dict(unit="kg N/day", source_labels=["fertilizer", "deposition"], start="2024-02-28", stop="2024-03-02")
        validate_daily_mass_bundle(days, np.arange(1, 231), values, **kwargs)
        with self.assertRaises(ContractError):
            validate_daily_mass_bundle(days[[0, 0, 2]], np.arange(1, 231), values, **kwargs)
        with self.assertRaises(ContractError):
            validate_daily_mass_bundle(days, np.arange(0, 230), values, **kwargs)
        with self.assertRaises(ContractError):
            validate_daily_mass_bundle(days, np.arange(1, 231), values, **dict(kwargs, unit="g N/day"))

    def test_partial_daily_deposition_blocks_future_S1_even_after_other_gates_clear(self):
        from scripts.prepare_matrix import required_for, build_fields
        for model in ("LAND0", "LAND1"):
            self.assertIn("deposition_daily_land_components", required_for({"source_id": "S1", "land_model_id": model}))
        # Frozen incomplete deposition is a read-only rejection fixture, not the
        # new conditional uniform-calendar product or a generated test output.
        from unittest.mock import patch
        from pathlib import Path
        with patch("scripts.prepare_matrix.ROOT", Path("E:/SPARROW/5_Test/20260926_1")):
            daily = next(row for row in build_fields() if row.name == "deposition_daily_land_components")
        self.assertEqual(daily.status, "unknown")
        self.assertGreater(daily.metadata["unresolved_reach_month_components"], 0)
        gate = real_run_gate([daily], [daily.name], artifact_kind="real_independent_input",
                             science_configuration_frozen=True, common_kernel_passed=True,
                             formal_dispatch_authorized=True)
        self.assertFalse(gate["allowed"])
        self.assertIn("required_input_unknown:deposition_daily_land_components", gate["reasons"])
        # Renaming the partial artifact available cannot conceal its NaNs.
        forged = replace(daily, status="available")
        gate = real_run_gate([forged], [daily.name], artifact_kind="real_independent_input",
                             science_configuration_frozen=True, common_kernel_passed=True,
                             formal_dispatch_authorized=True)
        self.assertFalse(gate["allowed"])
        self.assertTrue(any("nonfinite" in message for message in gate["reasons"]))


def identity(**kw):
    base = dict(source_hash="s", plant_activity_hash="plant", history_hash="history", h1_hash="h1",
        land_model_id="LAND0", equations_hash="eq", mapping_id="G1", operator_hash="ou",
        sampling_support_hash="obs", parameters_hash="par", objective_hash="obj", training_labels_hash="train",
        experiment_role="prediction")
    base.update(kw)
    return RunIdentity(**base)


class IdentityTests(unittest.TestCase):
    def test_changed_history_cannot_be_input_only_gain(self):
        with self.assertRaises(IdentityMismatch):
            compare_identity(identity(), identity(source_hash="new", history_hash="new_history"), changed_factors=["source_hash"])
        result = compare_identity(identity(), identity(history_hash="new_history"), descriptive=True)
        self.assertEqual(result["comparison_type"], "descriptive_only")
        self.assertFalse(result["unique_physical_causation_identified"])

    def test_allowed_input_and_joint_four_corner(self):
        compare_identity(identity(), identity(source_hash="new"), changed_factors=["source_hash"])
        four_corner_identity([identity(), identity(source_hash="new"), identity(parameters_hash="p2"),
                              identity(source_hash="new", parameters_hash="p2")])
        with self.assertRaises(IdentityMismatch):
            four_corner_identity([identity(), identity(), identity(), identity(history_hash="bad")])


class SupportTests(unittest.TestCase):
    def test_full_network_causality_and_zones(self):
        graph = NetworkSupport(5, {0: 1, 1: 3, 2: 3})
        self.assertEqual(graph.nearest_observation_zones([1, 3]).tolist(), [0, 0, 1, 1, -1])
        actual = graph.route_conservative_fixture(np.eye(5))
        self.assertEqual(actual[0].tolist(), [1, 1, 0, 1, 0])
        self.assertEqual(actual[3].tolist(), [0, 0, 0, 1, 0])
        with self.assertRaises(ValueError):
            NetworkSupport(2, {0: 1, 1: 0})


class SchedulerTests(unittest.TestCase):
    def test_public_acceptance_not_old_optimizer_sufficiency(self):
        from scripts.prepare_matrix import summarize_acceptance
        records = [{"name": name, "status": "passed"} for name in ("LAND1", "legacy_F23", "legacy_F24", "chain", "metrics", "numerical_validator")]
        fixed = [{"tag": tag, "status": "complete", "solver_sufficient": False} for tag in
                 ("U_W7_joint_b4", "U_D1_joint_b4", "L3_W7_joint_b4", "L3_D1_joint_b4")]
        result = summarize_acceptance(records, fixed)
        self.assertTrue(result["common_kernel_passed"])
        self.assertEqual(result["fixed_checkpoint_review_status"], "complete")
        result = summarize_acceptance(records, fixed[:3])
        self.assertTrue(result["common_kernel_passed"])
        self.assertEqual(result["fixed_checkpoint_review_status"], "pending")
        result = summarize_acceptance(records + [{"name": "required_fixture", "status": "pending"}], fixed)
        self.assertFalse(result["common_kernel_passed"])
        self.assertEqual(result["common_kernel_status"], "pending")
        self.assertFalse(summarize_acceptance([], fixed)["common_kernel_passed"])
        self.assertFalse(summarize_acceptance(records[:1], fixed)["common_kernel_passed"])

    def test_missing_receipts_are_pending(self):
        from scripts.prepare_matrix import collect_acceptance
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / "work") as directory:
            result = collect_acceptance(directory)
            self.assertFalse(result["common_kernel_passed"])
            self.assertEqual(result["common_kernel_status"], "pending")
            self.assertEqual(result["fixed_checkpoint_review_status"], "pending")

    def test_cpu_ram_commit_hysteresis_and_reserve(self):
        metrics = dict(cpu_percent=20, physical_percent=20, commit_percent=20,
                       physical_available_bytes=100, commit_available_bytes=100)
        self.assertTrue(resource_gate(metrics)["dispatch_allowed"])
        for key in ("cpu_percent", "physical_percent", "commit_percent"):
            self.assertFalse(resource_gate(dict(metrics, **{key: 90}))["dispatch_allowed"])
            self.assertFalse(resource_gate(dict(metrics, **{key: 85}), paused=True)["dispatch_allowed"])
            self.assertTrue(resource_gate(dict(metrics, **{key: 84.9}), paused=True)["dispatch_allowed"])
        self.assertFalse(resource_gate(metrics, reserve_bytes=101)["dispatch_allowed"])
        self.assertFalse(resource_gate({})["dispatch_allowed"])

    def test_concurrent_ledger_and_recovery(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / "work") as directory:
            ledger = PathLedger(directory, "fixture")
            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(lambda _: ledger.append(event="test", calls_delta=1, active_seconds_delta=.5), range(20)))
            state = PathLedger(directory, "fixture").restore()
            self.assertEqual((state["sequence"], state["calls"], state["active_seconds"]), (20, 20, 10))
            # Simulate interruption after durable log append but before checkpoint replace.
            ledger.checkpoint.unlink()
            state = ledger.restore()
            self.assertEqual(state["calls"], 20)

    def test_independent_process_log_concurrency(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=root / "work") as directory:
            program = ("from d29_platform.scheduler import PathLedger; "
                       f"p=PathLedger({directory!r},'workers'); "
                       "[p.append(event='worker',calls_delta=1) for _ in range(3)]")
            environment = dict(os.environ, PYTHONPATH=str(root), PYTHONDONTWRITEBYTECODE="1")
            with ThreadPoolExecutor(max_workers=3) as pool:
                outcomes = list(pool.map(lambda _: subprocess.run([sys.executable, "-c", program],
                    env=environment, capture_output=True, text=True, check=True), range(3)))
            state = PathLedger(directory, "workers").restore()
            self.assertEqual((state["sequence"], state["calls"]), (9, 9))


if __name__ == "__main__":
    unittest.main()
