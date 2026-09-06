"""Focused regression tests for the Q72 state-calendar repair.

These tests deliberately use a synthetic reach so state recurrences can be
checked without executing the expensive blocked-fold Bayesian fits.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd


RUN = Path(__file__).resolve().parents[2]
COMPONENT = RUN / "scripts" / "components" / "fit_monthly_bayes_seasonal_hysteresis.py"
PARAMS = {
    "rho": 0.70,
    "wm": 480.0,
    "et_gamma": 0.75,
    "sas_rho": 0.93,
    "young_k": 1.5,
    "storage_scale": 720.0,
    "prod_capacity": 240.0,
    "runoff_gamma": 2.5,
    "quick_rho": 0.25,
    "base_rho": 0.85,
    "base_release": 0.10,
}


def load_component():
    spec = importlib.util.spec_from_file_location("q72_state_calendar_test", COMPONENT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {COMPONENT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # State calendar tests must not depend on the shared topology artifact.
    module.reservoir_influence_class_by_reach = lambda max_order=2: {}
    return module


def forcing_frame(periods: pd.PeriodIndex, observed_indices: set[int]) -> pd.DataFrame:
    records = []
    for index, period in enumerate(periods):
        records.append(
            {
                "comid": 999999,
                "year": period.year,
                "month": period.month,
                "quarter": (period.month - 1) // 3 + 1,
                "period": index,
                "q_site": "synthetic" if index in observed_indices else None,
                "Q_obsv_cfs": 10.0 if index in observed_indices else np.nan,
                "CumAreaKm2": 10.0,
                "IncAreaKm2": 10.0,
                "Q_calc_cfs": 10.0,
                "Q_ma_cfs": 10.0,
                "PPT": 100.0 if index == 0 else 0.0,
                "AET": 0.0,
                "PET": 0.0,
            }
        )
    return pd.DataFrame(records)


class StateCalendarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_component()

    def featured(self, frame: pd.DataFrame, mode: str) -> pd.DataFrame:
        return self.module.build_featured_observation_panel(
            frame, **PARAMS, state_calendar_mode=mode
        )

    def test_four_month_internal_q_gap_advances_full_calendar(self):
        frame = forcing_frame(pd.period_range("2006-01", periods=4, freq="M"), {0, 2, 3})
        full = self.featured(frame, "full_forcing")
        observed = self.featured(frame, "observed_only")
        expected_full = (100.0 / 480.0) * 0.70**2
        expected_observed = (100.0 / 480.0) * 0.70
        self.assertEqual(len(full), 3)
        self.assertTrue((full["Q_obsv_cfs"] > 0).all())
        self.assertAlmostEqual(full.loc[full["month"] == 3, "antecedent_wetness"].iloc[0], expected_full)
        self.assertAlmostEqual(observed.loc[observed["month"] == 3, "antecedent_wetness"].iloc[0], expected_observed)
        self.assertNotAlmostEqual(expected_full, expected_observed)

    def test_145_month_internal_gap_is_not_one_recurrence_step(self):
        periods = pd.period_range("2006-01", periods=146, freq="M")
        frame = forcing_frame(periods, {0, 145})
        full = self.featured(frame, "full_forcing")
        observed = self.featured(frame, "observed_only")
        expected_full = (100.0 / 480.0) * 0.70**145
        expected_observed = (100.0 / 480.0) * 0.70
        self.assertAlmostEqual(full.iloc[-1]["antecedent_wetness"], expected_full, places=20)
        self.assertAlmostEqual(observed.iloc[-1]["antecedent_wetness"], expected_observed)

    def test_future_forcing_truncation_does_not_change_prior_states(self):
        frame = forcing_frame(pd.period_range("2006-01", periods=6, freq="M"), set(range(6)))
        full = self.module.add_hydrologic_features(frame, **PARAMS)
        truncated = self.module.add_hydrologic_features(frame.iloc[:3].copy(), **PARAMS)
        fields = [
            "antecedent_wetness", "sas_storage_mm", "production_storage_mm", "routed_base_cfs",
        ]
        for field in fields:
            np.testing.assert_allclose(full.loc[:2, field], truncated[field], rtol=0, atol=1e-12)

    def test_no_q_rows_enter_design_or_training(self):
        periods = pd.period_range("2006-01", periods=24, freq="M")
        frame = forcing_frame(periods, {0, 2, 13, 23})
        featured = self.featured(frame, "full_forcing")
        valid = frame.loc[frame["Q_obsv_cfs"].gt(0), ["comid", "year", "month"]]
        actual = featured[["comid", "year", "month"]]
        self.assertTrue((featured["Q_obsv_cfs"] > 0).all())
        self.assertTrue(featured["q_site"].notna().all())
        self.assertEqual(set(map(tuple, actual.to_numpy())), set(map(tuple, valid.to_numpy())))
        train = featured[featured["year"] <= 2006]
        self.assertTrue((train["Q_obsv_cfs"] > 0).all())


if __name__ == "__main__":
    unittest.main()
