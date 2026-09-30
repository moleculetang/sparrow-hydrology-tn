"""Tests of scientific input semantics and label-free observation boundary."""
import unittest
import math
import numpy as np
import pandas as pd

from d29_platform.inputs import corrected_external_inputs, allocate_external_entries
from d29_platform.coupling import sampling_from_inference_model, response_mapping


class InputBoundaryTests(unittest.TestCase):
    def test_source_correction_once_and_external_only(self):
        raw = np.array([0., 2., 5.])
        val, identity = corrected_external_inputs(raw, math.log(1.5), role='external_land_input')
        np.testing.assert_allclose(val, [0., 3., 7.5])
        for role in ('mineralization', 'plant_uptake', 'residue_return', 'plant_target'):
            with self.assertRaises(ValueError):
                corrected_external_inputs(raw, 0., role=role)
        with self.assertRaises(ValueError):
            corrected_external_inputs(val, 0., role='external_land_input', already_corrected=True)
        for h in (1e-4, 1e-5):
            lo = corrected_external_inputs(raw, math.log(1.5)-h, role='external_land_input')[0]
            hi = corrected_external_inputs(raw, math.log(1.5)+h, role='external_land_input')[0]
            np.testing.assert_allclose((hi-lo)/(2*h), identity['derivative_wrt_eta'], atol=1e-7, rtol=1e-7)

    def test_missing_or_unbounded_sources_rejected(self):
        for a, eta in (([np.nan], 0.), ([-1.], 0.), ([1.], math.log(4)+1e-8), ([1.], np.nan)):
            with self.assertRaises(ValueError):
                corrected_external_inputs(a, eta, role='external_land_input')

    def test_allocation_conserves_tags_not_just_total(self):
        a = np.arange(48., dtype=float).reshape(3,2,2,4)
        f = np.array([[0,0,0,1],[0,.7,.3,0],[1,0,0,0],[0,0,0,1.]])
        out = allocate_external_entries(a, f)
        np.testing.assert_allclose(out.sum(-1), a)
        for invalid in (f*.9, np.full((4,4), np.nan)):
            with self.assertRaises(ValueError):
                allocate_external_entries(a, invalid)

    def test_physical_metadata_never_accepts_tn(self):
        class Sentinel:
            def daily_metadata(self, meta):
                raise AssertionError('LABEL REACHED PHYSICAL OPERATOR')
        for label in ('tn_mg_l', 'fit_weight', 'fit_variance', 'observed', 'y'):
            with self.assertRaisesRegex(ValueError, 'TN_LABEL'):
                sampling_from_inference_model(Sentinel(), pd.DataFrame({label:[1.]}))

    def test_retired_parameters_fail_before_response_evaluation(self):
        class NamesOnly:
            names = ['log_tau_mineral_days'] + [f'gamma_lifetime_{i}' for i in range(7)]
        for name in NamesOnly.names:
            with self.assertRaisesRegex(ValueError, 'RETIRED'):
                response_mapping(NamesOnly(), {name:0.})

    def test_zero_water_is_undefined_not_epsilon(self):
        from d29_platform.legacy import bootstrap
        bootstrap()
        import torch
        from temporal_model import aggregate_daily
        for q in (0., -1., float('nan')):
            with self.assertRaisesRegex(ValueError, 'UNDEFINED_DAILY_CONCENTRATION'):
                aggregate_daily(torch.tensor([1.]), torch.tensor([q]), torch.tensor([0]), torch.tensor([1.]), 1, 'OU')


if __name__ == '__main__':
    unittest.main()
