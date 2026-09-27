import unittest
import numpy as np
import pandas as pd
from d29_training.objective import StrategyObjective,balanced_weights
from d29_training.observations import deduplicate_readings


def fixture():
    rows=[]
    for station,start in [('long',2016),('other',2021)]:
        for year in range(start,2023):
            for month in range(1,13):
                rows.append(dict(station_key=station,date=pd.Timestamp(year,month,1),tn_mg_l=1+.05*month+.01*(year-2016),eligible=True))
    h=[]
    for station in ['long','other']:
        for year in [2021,2022]:
            for month in [1,2]:
                for day,n in [(1,1),(2,3),(3,2)]:
                    h.append(dict(station_key=station,date=pd.Timestamp(year,month,day),tn_mg_l=1+.2*day,read_count=n,eligible=True))
    return pd.DataFrame(rows),pd.DataFrame(h)


class StrategyTests(unittest.TestCase):
    def test_monitoring_duplicate_adds_no_information(self):
        raw=pd.DataFrame({'station_id':['a','a'],'monitoring_time':['2021-01-01T00:00:00+08:00','2021-01-01T04:00:00+08:00'],'indicator':['TN','TN'],'adopted_value':[1.,3.],'parse_flag':['numeric']*2,'station_status':['正常']*2,'unresolved_conflict':[False]*2})
        original=deduplicate_readings(raw)
        duplicated=deduplicate_readings(pd.concat([raw,raw.iloc[[0]],raw.iloc[[1]]],ignore_index=True))
        pd.testing.assert_frame_equal(original,duplicated)
        bad=raw.iloc[[0]].copy();bad['adopted_value']=99.
        with self.assertRaisesRegex(ValueError,'CONFLICTING_DUPLICATE'):deduplicate_readings(pd.concat([raw,bad]))

    def test_all_strategies_multistep_gradient(self):
        m,h=fixture()
        for strategy in ['T0','T1','T2']:
            obj=StrategyObjective(m,h,strategy,2022)
            rng=np.random.default_rng(1729);p=obj.truth+rng.normal(0,.1,len(obj.truth));direction=rng.normal(size=len(p))
            value,g,_,r=obj.evaluate(p)
            self.assertAlmostEqual(value,.5*r@r,places=12)
            for eps in [1e-4,1e-5,1e-6]:
                fd=(obj.evaluate(p+eps*direction)[0]-obj.evaluate(p-eps*direction)[0])/(2*eps)
                self.assertAlmostEqual(fd,g@direction,places=7)
    def test_monthly_level_and_hf_shift(self):
        m,h=fixture();obj=StrategyObjective(m,h,'T2',2022)
        p=obj.truth.copy();p[obj.rows.kind.eq('HF')]+=99
        self.assertLess(obj.evaluate(p)[0],1e-23)
        p=obj.truth+1
        self.assertGreater(obj.evaluate(p)[2]['long_month'],0)
    def test_duplicate_and_future_rejected(self):
        m,h=fixture()
        with self.assertRaisesRegex(ValueError,'DUPLICATE'):StrategyObjective(m,pd.concat([h,h.iloc[:1]]),'T0',2022)
        future=m.iloc[:1].copy();future['date']=pd.Timestamp('2023-01-01')
        with self.assertRaisesRegex(ValueError,'EVALUATION'):StrategyObjective(pd.concat([m,future]),h,'T0',2022)
    def test_equal_station_year_month_weights(self):
        m,h=fixture();obj=StrategyObjective(m,h,'T1',2022)
        frame=obj.rows[obj.rows.kind.eq('monthly')].copy();frame['w']=balanced_weights(frame)
        np.testing.assert_allclose(frame.groupby('station_key').w.sum(),.5)
        self.assertEqual(obj.long_sites,['long'])
    def test_spatial_group_built_after_exclusion(self):
        m,h=fixture();obj=StrategyObjective(m,h,'T1',2022,excluded=['long'])
        self.assertEqual(obj.long_sites,[])
        self.assertEqual(set(obj.rows.station_key),{'other'})

if __name__=='__main__':unittest.main()
