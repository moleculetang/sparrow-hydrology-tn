"""Scientific support/weighting tests, including deliberate leakage failures."""
import unittest

import numpy as np
import pandas as pd

from d29_platform.objective import TrainingPolicy, build_training_objective
from d29_platform.metrics import (CoveragePolicy, weighted_metrics, evaluate_pair,
                                  monthly_views, calendar_draws, paired_calendar_bootstrap,
                                  evaluate_events)


def training_fixture():
    hf=[]
    for station,offset in [("S1",0),("S2",2)]:
        for month in (1,2,4):
            for day,y,count in [(2,1,1),(7,3,3),(16,2,2)]:
                hf.append(dict(station_key=station,date=f"2021-{month:02d}-{day:02d}",
                               tn_mg_l=y+offset+month/3,read_count=count,eligible=True))
    monthly=pd.DataFrame([
        dict(station_key="S1",date="2021-01-01",tn_mg_l=999.,eligible=True),
        dict(station_key="S1",date="2021-03-01",tn_mg_l=3.,eligible=True),
        dict(station_key="S3",date="2021-01-01",tn_mg_l=4.,eligible=True),
        dict(station_key="S3",date="2021-02-01",tn_mg_l=4.,eligible=True)])
    return pd.DataFrame(hf),monthly


def paired_fixture():
    h,_=training_fixture()
    h=h.rename(columns={"tn_mg_l":"truth"})
    h["baseline"]=h.truth+1.0
    h["candidate"]=h.truth+0.25
    return h


class ObjectiveTests(unittest.TestCase):
    def make(self):
        hf,month=training_fixture()
        return build_training_objective(hf,month,[2021],TrainingPolicy(1.,1.))

    def test_half_squared_coefficient_and_gradient(self):
        obj=self.make(); y=obj.rows.tn_mg_l.to_numpy()
        j,g,t=obj.evaluate(y+2)
        # Every level variance is floored at one and weights sum to one.
        self.assertAlmostEqual(j,1.)
        self.assertAlmostEqual(t["D_month_level"],2.)
        self.assertAlmostEqual(t["D_month_anomaly"],0.)
        self.assertAlmostEqual(g.sum(),1.)

    def test_dynamic_zero_mean_does_not_leak_to_level(self):
        obj=self.make(); p=obj.rows.tn_mg_l.to_numpy().copy()
        group=next(g for g in obj.groups if g["is_hf"])
        delta=np.array([1.,-1.,1.])
        delta-=group["alpha"]@delta
        p[group["indices"]]+=delta
        j,g,t=obj.evaluate(p)
        self.assertAlmostEqual(t["D_month_level"],0.,places=14)
        self.assertGreater(t["D_month_anomaly"],0.)
        self.assertAlmostEqual(g[group["indices"]].sum(),0.,places=14)

    def test_analytic_gradient_multistep(self):
        obj=self.make(); p=obj.rows.tn_mg_l.to_numpy()+np.linspace(-.3,.7,len(obj.rows))
        _,grad,_=obj.evaluate(p)
        for step in (1e-4,1e-5,1e-6):
            for i in range(len(p)):
                up=p.copy(); dn=p.copy(); up[i]+=step; dn[i]-=step
                fd=(obj.evaluate(up)[0]-obj.evaluate(dn)[0])/(2*step)
                self.assertLess(abs(fd-grad[i]),1e-8)

    def test_unique_station_month_priority(self):
        obj=self.make()
        self.assertFalse((obj.rows.tn_mg_l==999).any())
        self.assertEqual(obj.identity["unique_station_months"],9)
        self.assertEqual(len(obj.rows[obj.rows.observation_kind=="monthly"]),3)
        self.assertAlmostEqual(sum(g["level_weight"] for g in obj.groups),1.)
        self.assertAlmostEqual(sum(g["dynamic_weight"] for g in obj.groups),1.)

    def test_train_only_and_eval_mutation(self):
        h,m=training_fixture()
        obj=build_training_objective(h,m,[2021],TrainingPolicy(1.,1.))
        identity=obj.identity.copy(); p=obj.rows.tn_mg_l.to_numpy()+.2
        result=obj.evaluate(p)
        evaluation=h.copy(); evaluation["date"]=pd.to_datetime(evaluation.date)+pd.DateOffset(years=2)
        evaluation["tn_mg_l"]=1e9
        self.assertEqual(identity,obj.identity)
        self.assertEqual(result[0],obj.evaluate(p)[0])
        with self.assertRaisesRegex(ValueError,"non-training-year"):
            build_training_objective(pd.concat([h,evaluation]),m,[2021],TrainingPolicy(1,1))
        # Mutating caller-owned labels after construction cannot mutate truth.
        h.loc[:,"tn_mg_l"]=9999
        self.assertEqual(result[0],obj.evaluate(p)[0])

    def test_explicit_training_floors_and_duplicates(self):
        with self.assertRaises(ValueError): TrainingPolicy(0,1)
        h,m=training_fixture()
        with self.assertRaisesRegex(ValueError,"Duplicate"):
            build_training_objective(pd.concat([h,h.iloc[:1]]),m,[2021],TrainingPolicy(1,1))
        obj=self.make()
        self.assertTrue(obj.scales.set_index("station_key").loc["S3","level_floor_used"])


class MetricsTests(unittest.TestCase):
    def test_bias_changes_raw_but_not_centered_nse(self):
        station,summary=evaluate_pair(paired_fixture())
        self.assertTrue((station.candidate_nse>station.baseline_nse).all())
        np.testing.assert_allclose(station.baseline_month_centered_nse,1,atol=1e-14)
        np.testing.assert_allclose(station.candidate_month_centered_nse,1,atol=1e-14)
        self.assertEqual(summary.set_index("metric").loc["nse","n_common_stations"],2)

    def test_zero_variance_without_epsilon(self):
        result=weighted_metrics([2,2],[2,3])
        self.assertIsNone(result["nse"])
        self.assertEqual(result["nse_status"],"zero_observed_variance")
        t=paired_fixture(); t["truth"]=2.
        station,_=evaluate_pair(t)
        self.assertTrue(station.candidate_nse.isna().all())
        self.assertTrue(station.candidate_month_centered_nse.isna().all())
        self.assertIsNone(weighted_metrics([.1,.1,.1],[.1,.2,.3],[1,2,3])["nse"])

    def test_month_centering_is_equal_month_not_equal_reads(self):
        t=pd.DataFrame([dict(station_key="S",date=f"2021-{m:02d}-{d:02d}",truth=y,baseline=y,
                             candidate=(y if m==1 else 4-y),read_count=(1000 if m==1 else 1))
                        for m in (1,2) for d,y in ((1,1.),(2,3.))])
        station,_=evaluate_pair(t)
        self.assertAlmostEqual(station.candidate_month_centered_nse.iloc[0],-1.)
        self.assertGreater(station.candidate_nse.iloc[0],.99)

    def test_common_support_and_coverage(self):
        t=paired_fixture(); t.loc[0,"candidate"]=np.nan
        station,_=evaluate_pair(t,CoveragePolicy(8,3))
        self.assertEqual(station.loc[station.station_key=="S1","n_common_days"].iloc[0],8)
        station,_=evaluate_pair(t,CoveragePolicy(9,3))
        self.assertTrue(pd.isna(station.loc[station.station_key=="S1","candidate_nse"].iloc[0]))

    def test_monthly_views_stay_separate(self):
        t=paired_fixture()
        m=pd.DataFrame([dict(station_key="S1",date="2021-01-01",truth=99.,baseline=90.,candidate=91.,read_count=1)])
        views=monthly_views(t,m)
        self.assertEqual(len(views["HF_monthly"]),6)
        self.assertEqual(views["original_monthly"].truth.iloc[0],99.)
        self.assertNotEqual(views["HF_monthly"].truth.iloc[0],99.)

    def test_calendar_gaps_and_independent_duplicate_identity(self):
        ledger=calendar_draws(["2021-01","2021-02","2021-04"],50,2)
        self.assertIn("2021-03",ledger.source_month.unique())
        self.assertFalse(ledger.duplicated(["replicate","sample_month_id"]).any())
        for _,g in ledger.groupby(["replicate","block_instance"]):
            if len(g)==2:
                self.assertEqual(pd.Period(g.source_month.iloc[1],"M")-pd.Period(g.source_month.iloc[0],"M"),
                                 pd.offsets.MonthEnd(1))
        self.assertTrue(ledger.duplicated(["replicate","source_month"]).any())

    def test_bootstrap_moments_equal_direct_resampling(self):
        t=paired_fixture()
        for block in (1,2):
            out=paired_calendar_bootstrap(t,n_bootstrap=20,block_months=block)
            for rep,draw in out["ledger"].groupby("replicate"):
                pieces=[]
                for record in draw.to_dict("records"):
                    g=t[pd.to_datetime(t.date).dt.to_period("M").astype(str)==record["source_month"]].copy()
                    g["month_instance"]=record["sample_month_id"]
                    pieces.append(g)
                joined=pd.concat(pieces)
                from d29_platform.metrics import _evaluate_validated
                direct,_summary=_evaluate_validated(joined,CoveragePolicy(),"bootstrap",month_col="month_instance")
                for metric in ("nse","month_centered_nse"):
                    expected=_summary.set_index("metric").loc[metric,"median_paired_difference"]
                    row=out["replicates"][(out["replicates"].replicate==rep)&(out["replicates"].metric==metric)]
                    actual=row.median_paired_difference.iloc[0]
                    if expected is None: self.assertTrue(pd.isna(actual))
                    else: self.assertAlmostEqual(expected,actual,places=11)

    def test_event_peak_and_training_boundary(self):
        t=pd.DataFrame([dict(station_key="S",date=f"2023-01-{d:02d}",truth=y,baseline=b,candidate=y,read_count=1)
                        for d,y,b in [(1,1,1),(2,1,1),(3,2,2),(4,5,2),(5,2,5)]])
        events=pd.DataFrame([dict(station_key="S",event_id="e",start="2023-01-03",end="2023-01-05",
                                  background_start="2023-01-01",background_end="2023-01-02")])
        out=evaluate_events(t,events,evaluation_start="2023-01-01")
        self.assertEqual(out.baseline_peak_date_offset_days.iloc[0],1)
        self.assertEqual(out.candidate_nse.iloc[0],1)
        with self.assertRaisesRegex(ValueError,"crosses"):
            evaluate_events(t,events,evaluation_start="2023-01-02")


if __name__ == "__main__":
    unittest.main()
