from __future__ import annotations

import json
import importlib.util
import unittest
from pathlib import Path

import pandas as pd


RUN = Path(__file__).resolve().parents[1]
REPORT = RUN / "reports" / "signal_registry"
AUDIT_SCRIPT = RUN / "scripts" / "audit_canonical_signal_robustness.py"
AUDIT_SPEC = importlib.util.spec_from_file_location("enhanced_audit", AUDIT_SCRIPT)
AUDIT = importlib.util.module_from_spec(AUDIT_SPEC)
assert AUDIT_SPEC.loader is not None
AUDIT_SPEC.loader.exec_module(AUDIT)


class EnhancedSignalRegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = pd.read_csv(
            REPORT / "canonical_signal_registry.csv",
            encoding="utf-8-sig",
        )
        cls.validation = json.loads(
            (REPORT / "enhanced_signal_validation.json").read_text(encoding="utf-8")
        )

    def test_validation_gate(self) -> None:
        self.assertTrue(self.validation["passed"])
        self.assertTrue(self.validation["legacy_registry_reproduced"])
        self.assertEqual(self.validation["legacy_member_count"], 28)
        self.assertEqual(self.validation["enhanced_evidence_status"], "PROVISIONAL")

    def test_contract_columns(self) -> None:
        required = {
            "station",
            "eligible_folds",
            "overprediction_folds",
            "fold1_log_bias",
            "fold2_log_bias",
            "fold3_log_bias",
            "fold1_low_flow_PBIAS",
            "fold2_low_flow_PBIAS",
            "fold3_low_flow_PBIAS",
            "pooled_log_bias",
            "pooled_bootstrap_lower",
            "pooled_bootstrap_upper",
            "BH_q",
            "overprediction_month_fraction",
            "observed_low_flow_volume",
            "absolute_low_flow_error",
            "largest_fold_error_share",
            "cross_model_support",
            "signal_class",
            "source_prediction_sha256",
            "metric_code_sha256",
            "enhanced_evidence_status",
            "numeric_resolution_assessment",
        }
        self.assertFalse(required - set(self.registry.columns))

    def test_signal_and_priority_are_separate(self) -> None:
        legacy = self.registry[self.registry["legacy_canonical_membership"]]
        self.assertEqual(len(legacy), 28)
        self.assertTrue(
            legacy["signal_presence_class"].eq("LEGACY_CANONICAL_SIGNAL").all()
        )
        self.assertEqual(
            dict(legacy["review_priority_evidence_class"].value_counts()),
            {
                "CANDIDATE_SIGNAL": 23,
                "STRONG_SIGNAL": 3,
                "CONFIRMED_STABLE_SIGNAL": 2,
            },
        )

    def test_enhanced_layer_does_not_redefine_legacy_28(self) -> None:
        enhanced_strict = self.registry[
            self.registry["signal_class"].isin(
                ["STRONG_SIGNAL", "CONFIRMED_STABLE_SIGNAL"]
            )
        ]
        self.assertEqual(len(enhanced_strict), 5)
        self.assertTrue(enhanced_strict["legacy_canonical_membership"].all())

    def test_bh_adjust_known_case(self) -> None:
        adjusted = AUDIT.bh_adjust(pd.Series([0.01, 0.04, 0.03]))
        self.assertTrue(
            (adjusted.round(12) == pd.Series([0.03, 0.04, 0.04])).all()
        )

    def test_triple_direction_synthetic_case(self) -> None:
        frame = pd.DataFrame(
            {
                "fold_id": ["f1"] * 4,
                "q_site": ["synthetic"] * 4,
                "reach_id": [1] * 4,
                "is_low_flow": [True] * 4,
                "Q_obsv_cfs": [1.0, 1.0, 1.0, 1.0],
                "q72_interaction_lite_cfs": [2.0, 2.0, 0.5, 2.0],
                "is_reservoir_reach": [0] * 4,
                "downstream_reservoir": [0] * 4,
                "reach_class": ["headwater"] * 4,
            }
        )
        metrics = AUDIT.fold_metrics(frame)
        self.assertEqual(len(metrics), 1)
        self.assertTrue(bool(metrics.iloc[0]["triple_direction_consistent"]))
        self.assertAlmostEqual(
            float(metrics.iloc[0]["overprediction_month_fraction"]),
            0.75,
        )

    def test_year_cluster_bootstrap_is_seed_reproducible(self) -> None:
        low = pd.DataFrame(
            {
                "year": [2012, 2012, 2013, 2013, 2014, 2014],
                "Q_obsv_cfs": [1.0, 1.5, 2.0, 2.5, 1.2, 1.8],
                "q72_interaction_lite_cfs": [1.3, 1.8, 2.4, 2.9, 1.6, 2.2],
            }
        )
        first, first_p = AUDIT.bootstrap_station(
            low, AUDIT.np.random.default_rng(123)
        )
        second, second_p = AUDIT.bootstrap_station(
            low, AUDIT.np.random.default_rng(123)
        )
        self.assertTrue(AUDIT.np.array_equal(first, second))
        self.assertEqual(first_p, second_p)

    def test_station_bootstrap_stream_is_order_invariant(self) -> None:
        left = AUDIT.station_rng("甲站", 100).integers(0, 2**31, size=8)
        _ = AUDIT.station_rng("乙站", 200).integers(0, 2**31, size=8)
        right = AUDIT.station_rng("甲站", 100).integers(0, 2**31, size=8)
        self.assertTrue(AUDIT.np.array_equal(left, right))

    def test_frozen_source_guard(self) -> None:
        checked = AUDIT.validate_frozen_source(AUDIT.pd.read_csv(
            AUDIT.OOF_PATH, encoding="utf-8-sig"
        ).assign(is_low_flow=lambda frame: AUDIT.parse_bool(frame["is_low_flow"])))
        self.assertTrue(checked["passed"])
        self.assertTrue(checked["checks"]["frozen_hashes_match"])

    def test_fold_dominance_rule_matches_contract(self) -> None:
        self.assertTrue(
            (
                self.registry["fold_dominated_signal"]
                == self.registry["largest_fold_error_share"].gt(0.60)
            ).all()
        )
        self.assertTrue(
            self.registry["numeric_resolution_assessment"]
            .eq("NOT_YET_TESTABLE_WITH_OOF_ONLY")
            .all()
        )


if __name__ == "__main__":
    unittest.main()
