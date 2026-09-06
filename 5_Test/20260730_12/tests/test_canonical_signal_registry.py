from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import pandas as pd


RUN = Path(__file__).resolve().parents[1]
SCRIPT = RUN / "scripts" / "reconstruct_canonical_signal_registry.py"
SPEC = importlib.util.spec_from_file_location("registry_reconstruction", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class CanonicalSignalRegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.oof = MODULE.load_oof()
        cls.by_fold, cls.summary = MODULE.reconstruct(cls.oof)

    def test_frozen_hashes(self) -> None:
        actual = {
            "oof": MODULE.sha256(MODULE.OOF_PATH),
            "legacy_code": MODULE.sha256(MODULE.LEGACY_CODE_PATH),
            "expected_fold": MODULE.sha256(MODULE.EXPECTED_FOLD_PATH),
            "expected_summary": MODULE.sha256(MODULE.EXPECTED_SUMMARY_PATH),
        }
        self.assertEqual(actual, MODULE.EXPECTED_HASHES)

    def test_oof_contract(self) -> None:
        self.assertEqual(len(self.oof), 9067)
        self.assertEqual(self.oof["q_site"].nunique(), 114)
        self.assertEqual(int(self.oof["year"].min()), 2012)
        self.assertEqual(int(self.oof["year"].max()), 2018)
        self.assertFalse(
            self.oof.duplicated(["q_site", "reach_id", "year", "month", "fold_id"]).any()
        )

    def test_stored_low_flow_flag(self) -> None:
        expected = self.oof["Q_obsv_cfs"].le(self.oof["train_q25_cfs"])
        self.assertTrue((expected == self.oof["is_low_flow"]).all())

    def test_legacy_target_set_is_28(self) -> None:
        targets = self.summary[self.summary["stable_nonreservoir_target"]]
        self.assertEqual(len(targets), 28)
        expected = pd.read_csv(MODULE.EXPECTED_SUMMARY_PATH, encoding="utf-8-sig")
        expected_targets = expected[
            expected["stable_nonreservoir_target"].astype(str).str.lower().eq("true")
        ]
        self.assertEqual(
            set(map(tuple, targets[["q_site", "reach_id"]].to_records(index=False))),
            set(map(tuple, expected_targets[["q_site", "reach_id"]].to_records(index=False))),
        )

    def test_legacy_rule_is_not_three_of_three(self) -> None:
        targets = self.summary[self.summary["stable_nonreservoir_target"]]
        self.assertEqual(int(targets["eligible_folds"].eq(2).sum()), 18)
        self.assertEqual(int(targets["eligible_folds"].eq(3).sum()), 10)
        self.assertEqual(int(targets["overprediction_folds"].eq(2).sum()), 24)
        self.assertEqual(int(targets["overprediction_folds"].eq(3).sum()), 4)

    def test_fixed_exclusions_absent_and_shijiao_present(self) -> None:
        for station in MODULE.EXCLUSIONS:
            self.assertEqual(int((self.oof["q_site"] == station).sum()), 0)
        self.assertTrue((self.oof["q_site"] == "石角站").any())

    def test_perturbed_low_flow_flag_is_detected(self) -> None:
        perturbed = self.oof.copy()
        perturbed.loc[perturbed.index[0], "is_low_flow"] = not bool(
            perturbed.loc[perturbed.index[0], "is_low_flow"]
        )
        expected = perturbed["Q_obsv_cfs"].le(perturbed["train_q25_cfs"])
        self.assertEqual(int((expected != perturbed["is_low_flow"]).sum()), 1)


if __name__ == "__main__":
    unittest.main()

