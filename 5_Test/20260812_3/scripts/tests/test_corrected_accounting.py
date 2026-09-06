from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts" / "components" / "corrected_dynamic_q72_component.py"
spec = importlib.util.spec_from_file_location("corrected_q72", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)
module.TOPOLOGY_PATH = ROOT / "inputs" / "topology" / "topology_edges.csv"


class CorrectedAccountingTests(unittest.TestCase):
    def test_retention_mass_closure_and_literal_rho(self):
        available, release, store = module.retention_step(10.0, 2.0, 0.85)
        self.assertAlmostEqual(available, 12.0)
        self.assertAlmostEqual(release + store, available, places=12)
        self.assertAlmostEqual(store / available, 0.85, places=12)

    def test_no_input_month_uses_rho_not_rho_squared(self):
        available, release, store = module.retention_step(10.0, 0.0, 0.93)
        self.assertAlmostEqual(store, 9.3, places=12)
        self.assertAlmostEqual(release, 0.7, places=12)

    def test_incremental_topology_sum_matches_frozen_explicit_net(self):
        frame = pd.read_parquet(ROOT / "inputs" / "scenarios" / "B0_indata.parquet").sort_values(["comid", "year", "month"]).reset_index(drop=True)
        days = pd.to_datetime(dict(year=frame.year, month=frame.month, day=1)).dt.days_in_month.to_numpy(float)
        seconds = days * 86400.0
        net = np.maximum(frame.PPT.to_numpy(float) - frame.AET.to_numpy(float), 0.0)
        local = net / 1000.0 * frame.IncAreaKm2.to_numpy(float) * 1e6 / seconds * 35.3146667
        aggregate = module._aggregate_local_cfs(frame, local)
        np.testing.assert_allclose(aggregate, frame.explicit_upstream_net_cfs.to_numpy(float), rtol=0.0, atol=1e-8)

    def test_bad_highflow_composite_is_absent_from_all_design_blocks(self):
        for collection in [module.FIXED_FEATURES, module.PRODUCTION_FEATURES, module.RANDOM_SLOPE_FEATURES, module.REGIME_SLOPE_FEATURES, module.SPATIAL_GROUP_FEATURES]:
            self.assertNotIn("log_production_highflow_mass", collection)
        self.assertNotIn("log_basin_net", module.FIXED_FEATURES)
        self.assertNotIn("log_qcalc", module.RANDOM_SLOPE_FEATURES)

    def test_dynamic_beta_zero_is_exact_fixed_point_three_five(self):
        score = np.linspace(-3, 3, 101)
        fraction = module.expit(module.logit(0.35) + 0.0 * score)
        np.testing.assert_allclose(fraction, 0.35, rtol=0.0, atol=1e-15)

    def test_connected_plus_unconnected_closes_local_and_upstream_water(self):
        frame = pd.read_parquet(ROOT / "inputs" / "scenarios" / "B0_indata.parquet").sort_values(["comid", "year", "month"]).reset_index(drop=True)
        local = frame.explicit_upstream_net_cfs.to_numpy(float) * 0.0
        days = pd.to_datetime(dict(year=frame.year, month=frame.month, day=1)).dt.days_in_month.to_numpy(float)
        net = np.maximum(frame.PPT.to_numpy(float)-frame.AET.to_numpy(float),0.0)
        local = net/1000*frame.IncAreaKm2.to_numpy(float)*1e6/(days*86400)*35.3146667
        score = np.linspace(-3,3,len(frame)); c = module.expit(module.logit(.35)+.5*score)
        connected, unconnected = c*local, (1-c)*local
        np.testing.assert_allclose(connected+unconnected,local,rtol=0,atol=1e-10)
        upstream_sum=module._aggregate_local_cfs(frame,connected)+module._aggregate_local_cfs(frame,unconnected)
        np.testing.assert_allclose(upstream_sum,frame.explicit_upstream_net_cfs.to_numpy(float),rtol=0,atol=1e-8)

    def test_fold_reference_does_not_use_future_forcing(self):
        frame = pd.read_parquet(ROOT / "inputs" / "scenarios" / "B0_indata.parquet").sort_values(["comid", "year", "month"]).reset_index(drop=True)
        values=.35*frame.explicit_upstream_net_cfs.to_numpy(float)
        ref1,threshold1=module.fold_pure_qma_reference(frame,values,2011)
        changed=values.copy(); changed[frame.year.to_numpy()>2011]*=1000
        ref2,threshold2=module.fold_pure_qma_reference(frame,changed,2011)
        np.testing.assert_allclose(ref1,ref2,rtol=0,atol=0)
        self.assertEqual(threshold1,threshold2)


if __name__ == "__main__":
    unittest.main()
