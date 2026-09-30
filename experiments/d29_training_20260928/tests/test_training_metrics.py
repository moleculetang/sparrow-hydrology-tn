import unittest
import numpy as np
import pandas as pd
from d29_training.metrics import basic,centered,station_table,paired_summary,synchronous_blocks

class MetricTests(unittest.TestCase):
    def test_known_bias_amplitude_and_zero_variance(self):
        y=np.array([1.,2.,3.])
        self.assertEqual(basic(y,y)['NSE'],1.)
        self.assertAlmostEqual(basic(y,y+1)['NSE'],-.5)
        self.assertAlmostEqual(basic(y,2*y-2)['NSE'],0.)
        self.assertTrue(np.isnan(basic([2,2],[1,3])['NSE']))
        self.assertTrue(np.isnan(basic([.1,.1,.1],[1,3,2],[1,3,7])['NSE']))
    def test_monthly_bias_removed_only_in_centered_metric(self):
        f=pd.DataFrame(dict(station_key=['x']*4,date=pd.to_datetime(['2023-01-01','2023-01-02','2023-02-01','2023-02-02']),month_key=['a','a','b','b'],observed=[1.,3.,2.,6.],baseline=[1.,3.,2.,6.],candidate=[6.,8.,12.,16.],read_count=[1,3,2,1]))
        self.assertAlmostEqual(centered(f,'candidate')['NSE'],1.)
        t=station_table(f);s=paired_summary(t,'month_centered_NSE')
        self.assertEqual(s['common_stations'],1);self.assertAlmostEqual(s['median_paired_difference'],0.)
        draws=list(synchronous_blocks(pd.concat([f,f.assign(station_key='y')]),3,1))
        for _,d in draws:
            self.assertEqual(d[d.station_key.eq('x')].month_key.tolist(),d[d.station_key.eq('y')].month_key.tolist())

if __name__=='__main__':unittest.main()
