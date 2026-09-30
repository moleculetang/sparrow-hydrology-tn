import unittest

import numpy as np

from d29_platform.precision_candidate import run_land1
from d29_platform.mixture_tags_candidate import propagate_source_labels


def probabilities(mobilize):
    return dict(mineralize_active=0., mineralize_protected=0., mobilize=mobilize,
                available_loss=0., fast_fraction=.5, lower_release=.03)


class AuthoritativeTransportTests(unittest.TestCase):
    def test_near_exhaustion_uses_nonnegative_flow_equation(self):
        initial = np.zeros((1, 1, 5))
        initial[0, 0, 3] = 1.3301360013429075e-11
        source = np.zeros((1, 1, 1, 4))
        source[0, 0, 0, 3] = 2500.687221773003
        p = np.nextafter(1., 0.)
        result = run_land1(initial=initial, sources=source, plant_target=0.,
                           plant_outflows=0., probabilities=probabilities(p),
                           compensated=True)
        expected = (1-p)*(initial[0,0,3]+source[0,0,0,3])
        self.assertGreater(result.final[0,0,3], 0.)
        self.assertAlmostEqual(result.final[0,0,3], expected, delta=1e-24)
        self.assertLess(result.max_local_balance_kg, 1e-9)

    def test_source_labels_are_bounded_passive_partition(self):
        days = 40
        initial = np.zeros((1, 1, 5))
        initial[0,0,3] = 47_643_766.39399504
        sources = np.zeros((days, 1, 1, 4))
        sources[:,0,0,3] = 1000.
        result = run_land1(initial=initial, sources=sources, plant_target=0.,
                           plant_outflows=0., probabilities=probabilities(.01),
                           keep_history=True, compensated=True)
        tagged_initial = np.zeros((1,1,2,5))
        tagged_initial[0,0,0,3] = initial[0,0,3]
        tagged_sources = np.zeros((days,1,1,2,4))
        tagged_sources[:,0,0,1,3] = sources[:,0,0,3]
        tags = propagate_source_labels(result, tagged_initial=tagged_initial,
            tagged_sources=tagged_sources, labels=("initial", "new"),
            compensated=True, record_history=True)
        represented = tags.final[0,0,:,3] - tags.compensation[0,0,:,2]
        self.assertAlmostEqual(float(represented.sum()), float(result.final[0,0,3]), delta=1e-6)
        self.assertLess(tags.max_state_sum_error_kg, 1e-6)
        self.assertLess(tags.max_local_balance_kg, 1e-6)


if __name__ == "__main__":
    unittest.main()
