"""End-to-end evaluation driver on synthetic observations only."""
import unittest,tempfile,json,importlib.util
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from d29_platform.runtime import sha,write_json


class CampaignDriverTests(unittest.TestCase):
    def test_perfect_candidate_and_no_early_t0_training_claim(self):
        source=Path(__file__).resolve().parents[1]/'scripts/evaluate_frozen_campaign.py'
        spec=importlib.util.spec_from_file_location('synthetic_evaluation_driver',source)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);module.ROOT=root
            (root/'data').mkdir();(root/'outputs').mkdir();(root/'reports').mkdir()
            events_path=root/'data/frozen_evaluation_events.parquet'
            jobs=[dict(id='base',model='U',fold='F23',strategy='T0',entry=0,space_block=None,train_end=2022,evaluate=[2023]),
                  dict(id='candidate',model='U',fold='F23',strategy='T2',entry=0,space_block=None,train_end=2022,evaluate=[2023])]
            write_json(root/'config/jobs.json',jobs);write_json(root/'vendor/legacy22/data/spatial_blocks.json',{})
            write_json(root/'config/experiment_context.json',{'root':str(root),'experiment_id':'20260928_1_review_v1',
                'namespace':'jobs','hashes':{'config/jobs.json':sha(root/'config/jobs.json')}})
            dates=pd.DatetimeIndex([pd.Timestamp(y,m,d) for y in [2018,2022,2023] for m in range(1,13) for d in range(1,6)])
            observed=pd.DataFrame(dict(station_key='a',date=dates,tn_mg_l=np.tile([1.,3.,2.,5.,4.],36)+np.repeat(np.tile(np.arange(12),3),5),read_count=1,eligible=True))
            observed.to_parquet(root/'data/hf_daily.parquet',index=False)
            monthly=observed.assign(year=observed.date.dt.year,month=observed.date.dt.month).groupby(['station_key','year','month'],as_index=False).agg(tn_mg_l=('tn_mg_l','mean'))
            monthly['date']=pd.to_datetime(monthly[['year','month']].assign(day=1));monthly['eligible']=True
            monthly.to_parquet(root/'data/monthly_tn.parquet',index=False)
            pd.DataFrame([dict(station_key='a',event_rank=1,start='2023-01-05',end='2023-01-05',background_start='2023-01-01')]).to_parquet(events_path,index=False)
            write_json(root/'data/event_support_contract.json',{'files':{str(events_path):sha(events_path)}})
            records={}
            for j in jobs:
                folder=root/'outputs/jobs'/j['id'];folder.mkdir(parents=True)
                write_json(folder/'worker.json',{'experiment_id':'20260928_1_review_v1'})
                bias=1. if j['id']=='base' else 0.
                observed[['station_key','date']].assign(prediction_mg_l=observed.tn_mg_l+bias).to_parquet(folder/'frozen_station_days.parquet',index=False)
                monthly[['station_key','year','month','date']].assign(prediction_mg_l=monthly.tn_mg_l+bias).to_parquet(folder/'frozen_station_months.parquet',index=False)
                np.save(folder/'frozen_parameters.npy',np.array([bias]))
                write_json(folder/'prediction_freeze.json',{k:sha(folder/f) for f,k in [('frozen_station_days.parquet','days_sha256'),('frozen_station_months.parquet','months_sha256'),('frozen_parameters.npy','parameter_sha256')]})
                records[j['id']]=dict(status='solver_stopped_numerically_sufficient',objective=1.,objective_passed=True,physical_passed=True,projected_gradient=1e-7,numerically_sufficient=True)
                write_json(root/'data/training_contracts'/j['id']/'identity.json',{'long_stations':['a']})
            manifest=root/'manifest.json';write_json(manifest,dict(selected_jobs=['base','candidate'],excluded_jobs={},accounted_paths=['base','candidate'],selection_training_only=True,training_records=records,event_support_contract_sha256=sha(root/'data/event_support_contract.json'),experiment_id='20260928_1_review_v1',experiment_context_sha256=sha(root/'config/experiment_context.json')))
            # The full 1000-replicate statistic has separate exactness tests.
            # This integration probe uses four draws to cover output wiring.
            native=module.paired_resamples
            with patch.object(module,'paired_resamples',side_effect=lambda f,n,b,s,c,**kw:native(f,4,b,s,c,**kw)):
                module.run(manifest)
            summary=json.loads((root/'outputs/evaluation/paired_summaries.json').read_text())
            daily=next(r for r in summary if r['scope']=='daily' and r['year']==2023 and r['role']=='time')
            self.assertEqual(daily['NSE']['candidate_median'],1.)
            self.assertEqual(daily['NSE']['improved_fraction'],1.)
            early=pd.read_csv(root/'outputs/evaluation/absolute_results/base_monthly_report_2018_all_descriptive.csv')
            self.assertTrue(early.role.eq('historical_backcast_not_training').all())
            event=pd.read_csv(root/'outputs/evaluation/candidate/events_2023.csv')
            self.assertTrue(event.background_days.eq(4).all())
            self.assertTrue((root/'outputs/evaluation/evaluation_completed.json').exists())
            # Report reader must consume the exact evaluated manifest and
            # handle unavailable diagnostics without inventing their values.
            original=manifest.read_bytes();(root/'outputs/campaign_manifest.json').write_bytes(original)
            report_spec=importlib.util.spec_from_file_location('synthetic_report',source.with_name('report_frozen_campaign.py'))
            report=importlib.util.module_from_spec(report_spec);report_spec.loader.exec_module(report);report.ROOT=root
            report.main()
            text=(root/'reports/专家训练策略与全域验证报告.md').read_text(encoding='utf-8')
            self.assertIn('尚无两折完整月报留出证据',text)
            self.assertIn('未完成',text)


if __name__=='__main__':unittest.main()
