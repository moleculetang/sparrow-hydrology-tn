import unittest
import numpy as np
import pandas as pd
from d29_training.metrics import station_table,eligible_nse_stations,basic
from d29_training.bootstrap import paired_resamples


class CoverageTests(unittest.TestCase):
    def test_sparse_perfect_fit_not_qualified_nse(self):
        f=pd.DataFrame(dict(station_key='a',date=pd.date_range('2023-01-01',periods=3),observed=[1.,2.,3.],baseline=[1.,2.,3.],candidate=[1.,2.,3.],read_count=1))
        t=station_table(f,minimum_coverage=True)
        self.assertTrue(t.NSE.isna().all());self.assertTrue(t.month_centered_NSE.isna().all())
        self.assertTrue(t.RMSE.eq(0).all())
        draws,r=paired_resamples(f,4,eligible_sites=eligible_nse_stations(f,True))
        self.assertTrue(draws.common_stations.eq(0).all());self.assertTrue(draws.median_difference.isna().all())

    def test_one_event_read_has_error_but_no_nse(self):
        score=basic([3.],[5.]);self.assertTrue(np.isnan(score['NSE']))
        self.assertEqual(score['RMSE'],2.);self.assertEqual(score['bias'],2.)

    def test_monthly_coverage_is_eight(self):
        f=pd.DataFrame(dict(station_key='a',date=pd.date_range('2023-01-01',periods=8,freq='MS'),observed=np.arange(8),baseline=np.arange(8),candidate=np.arange(8)))
        self.assertTrue(station_table(f,daily=False,minimum_coverage=True).NSE.eq(1).all())
        self.assertTrue(station_table(f.iloc[:7],daily=False,minimum_coverage=True).NSE.isna().all())
