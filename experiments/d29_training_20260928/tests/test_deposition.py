import calendar
import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from d29_platform.deposition import distribute_month, distribute_class_month, normalized_class_area


class DepositionTests(unittest.TestCase):
    def test_leap_month_mass_and_rain_days(self):
        mass = np.array([[31., 57., 29., 87.]])
        rain = np.zeros((29, 1)); rain[[0, 14, 28], 0] = [1, 2, 3]
        daily, gaps = distribute_month(mass, rain, 2024, 2)
        self.assertEqual(gaps, [])
        np.testing.assert_allclose(daily.sum(0), mass, atol=1e-12, rtol=0)
        self.assertEqual(np.count_nonzero(daily[:, 0, 2]), 3)
        self.assertAlmostEqual(daily[14, 0, 2] / daily[0, 0, 2], 2.)

    def test_zero_rain_is_missing_not_zero_wet_mass(self):
        daily, gaps = distribute_month(np.array([[3., 6., 9., 0.]]), np.zeros((28, 1)), 2023, 2)
        self.assertEqual(len(gaps), 1)
        self.assertTrue(np.isnan(daily[:, 0, 2]).all())
        self.assertTrue((daily[:, 0, 3] == 0).all())
        np.testing.assert_allclose(daily[:, :, :2].sum(0), [[3, 6]])

    def test_future_month_independence(self):
        rain = np.ones((31, 2)); m = np.ones((2, 4))
        before = distribute_month(m, rain, 2024, 1)[0]
        distribute_month(m * 100, np.zeros((29, 2)), 2024, 2)
        np.testing.assert_array_equal(before, distribute_month(m, rain, 2024, 1)[0])

    def test_missing_rain_rejects_wet_allocation(self):
        rain = np.ones((31, 1)); rain[0] = np.nan
        d, gaps = distribute_month(np.ones((1, 4)), rain, 2024, 1)
        self.assertEqual(len(gaps), 2)
        self.assertTrue(np.isnan(d[:, :, 2:]).all())

    def test_classification_closes_and_preserves_impervious(self):
        count = np.zeros((2, 10), dtype=int)
        count[0, [1, 5, 8]] = [4, 2, 2]
        count[1, [2, 0]] = [3, 1]
        group, codes = normalized_class_area(count, np.array([7000., 3000.]))
        np.testing.assert_allclose(group.sum(-1), [7000, 3000])
        self.assertEqual(group[0, 2], 1750.)
        self.assertEqual(group[0, 4], 0.)  # 7200m2 centre area scales to7000

    def test_invalid_dimensions_and_negative_rain(self):
        with self.assertRaises(ValueError):
            distribute_month(np.ones((1, 4)), np.ones((28, 1)), 2024, 2)
        with self.assertRaises(ValueError):
            distribute_month(np.ones((1, 4)), -np.ones((29, 1)), 2024, 2)

    def test_class_mass_separate_and_calendar_identical(self):
        mass = np.arange(40, dtype=float).reshape(2, 5, 4)
        rain = np.ones((29, 2)); rain[:3, 0] = 0
        d, gaps = distribute_class_month(mass, rain, 2024, 2)
        self.assertEqual(gaps, [])
        np.testing.assert_allclose(d.sum(0), mass, atol=1e-12, rtol=0)
        whole, _ = distribute_month(mass.sum(1), rain, 2024, 2)
        np.testing.assert_allclose(d.sum(2), whole, atol=1e-12, rtol=0)


if __name__ == "__main__":
    unittest.main()
