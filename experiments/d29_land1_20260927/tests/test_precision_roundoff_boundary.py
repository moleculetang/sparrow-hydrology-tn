import unittest

import numpy as np

from d29_platform.precision_candidate import prepare_inputs, run_land1


class AnnualRoundoffBoundaryTests(unittest.TestCase):
    def fixture(self, small_negative):
        initial = np.zeros((1, 1, 5), dtype=np.float64)
        initial[0, 0, 3] = small_negative
        sources = np.zeros((1, 1, 1, 4), dtype=np.float64)
        sources[0, 0, 0, 3] = 1.0
        return dict(initial=initial, sources=sources, plant_target=0.,
                    plant_outflows=0., probabilities=dict(mineralize_active=0.,
                    mineralize_protected=0., mobilize=0., available_loss=0.,
                    fast_fraction=0., lower_release=0.))

    def test_explicit_initial_remains_strict(self):
        with self.assertRaisesRegex(ValueError, 'INITIAL_MUST_BE_NONNEGATIVE'):
            run_land1(**self.fixture(-9.094947017729282e-13))

    def test_annual_roundoff_retains_mass_without_clipping(self):
        data = self.fixture(-9.094947017729282e-13)
        result = run_land1(**data, compensated=True,
                           intermediate_roundoff_tolerance=1e-9)
        self.assertEqual(result.states[0, 0, 0, 3], data['initial'][0, 0, 3])
        self.assertAlmostEqual(result.final[0, 0, 3],
                               1.0 + data['initial'][0, 0, 3], places=13)
        self.assertLess(result.max_local_balance_kg, 1e-10)

    def test_large_negative_and_large_tolerance_rejected(self):
        with self.assertRaisesRegex(ValueError, 'INITIAL_MUST_BE_NONNEGATIVE'):
            run_land1(**self.fixture(-1.1e-9),
                      intermediate_roundoff_tolerance=1e-9)
        with self.assertRaisesRegex(ValueError, 'INVALID_INTERMEDIATE'):
            prepare_inputs(**self.fixture(0.), intermediate_roundoff_tolerance=1e-7)


if __name__ == '__main__':
    unittest.main()
