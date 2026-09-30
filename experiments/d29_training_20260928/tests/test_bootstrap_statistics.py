import unittest
import numpy as np
import pandas as pd
from d29_training.metrics import station_table,synchronous_blocks,paired_summary
from d29_training.bootstrap import paired_resamples

class BootstrapTests(unittest.TestCase):
    def test_exact_against_explicit_resampling(self):
        rng=np.random.default_rng(53);rows=[]
        for s in ['a','b','c']:
            for d in pd.date_range('2023-01-01','2023-04-30',freq='4D'):
                y=float(rng.uniform(.2,5));rows.append(dict(station_key=s,date=d,observed=y,baseline=y+float(rng.normal()),candidate=.7*y+float(rng.normal(scale=.4)),read_count=int(rng.integers(1,7))))
        f=pd.DataFrame(rows)
        for block in [1,2]:
            for centered in [False,True]:
                fast,receipt=paired_resamples(f,9,block,centered=centered)
                for b,sample in synchronous_blocks(f,9,block):
                    slow=paired_summary(station_table(sample),'month_centered_NSE' if centered else 'NSE')
                    for key in ['common_stations','median_difference','median_paired_difference','improved_fraction']:
                        self.assertAlmostEqual(fast.iloc[b][key],slow[key],places=11)

    def test_constant_is_undefined(self):
        f=pd.DataFrame(dict(station_key=['a']*3,date=pd.to_datetime(['2023-01-01','2023-01-02','2023-01-03']),observed=[.1]*3,baseline=[1,2,3],candidate=[3,2,1],read_count=[1]*3))
        result,_=paired_resamples(f,2)
        self.assertTrue(result.common_stations.eq(0).all())

if __name__=='__main__':unittest.main()
