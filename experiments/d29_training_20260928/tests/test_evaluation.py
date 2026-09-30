import unittest
import numpy as np
import pandas as pd
from d29_training.evaluation import join_pair,hf_monthly,evaluate_events

class EvaluationTests(unittest.TestCase):
    def test_common_support_and_read_weight(self):
        dates=pd.date_range('2023-01-01',periods=3)
        o=pd.DataFrame(dict(station_key=['a']*3,date=dates,tn_mg_l=[1.,3.,7.],read_count=[1,3,1],eligible=[True]*3))
        a=o[['station_key','date']].assign(prediction_mg_l=[2.,4.,8.])
        b=o[['station_key','date']].assign(prediction_mg_l=[1.,3.,np.nan])
        f=join_pair(o,a,b);self.assertEqual(len(f),2)
        m=hf_monthly(f);self.assertEqual(m.iloc[0].observed,2.5)

    def test_event_background_never_training_year(self):
        f=pd.DataFrame(dict(station_key=['a']*8,date=pd.date_range('2022-12-31',periods=8),observed=[100.,1.,1.,1.,1.,2.,4.,3.],baseline=[100.,1.,1.,1.,1.,2.,4.,3.],candidate=[100.,1.,1.,1.,1.,4.,3.,2.]))
        events=pd.DataFrame([dict(station_key='a',event_rank=1,start='2023-01-05',end='2023-01-07',background_start='2022-12-31')])
        t=evaluate_events(f,events,2023)
        self.assertTrue(t.background_days.eq(4).all())
        self.assertEqual(t[t.configuration.eq('baseline')].iloc[0].NSE,1.)
        self.assertEqual(t[t.configuration.eq('candidate')].iloc[0].peak_day_offset,-1)

if __name__=='__main__':unittest.main()
