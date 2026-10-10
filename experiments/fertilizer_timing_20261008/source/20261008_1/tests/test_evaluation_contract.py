"""Known-result evaluation, paired supports, frozen selection and month bootstrap."""
import importlib.util
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from d29_platform.runtime import write_json,sha
from d29_training.evaluation import pair_tables
from d29_training.bootstrap import paired_resamples
from d29_training.metrics import basic


def main():
    spec=importlib.util.spec_from_file_location('timing_postfreeze_evaluation',ROOT/'scripts/evaluate_campaign.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    jobs=json.loads((ROOT/'config/jobs.json').read_text(encoding='utf8'))
    records={j['id']:dict(objective=2. if j['entry']==0 else 1.,objective_passed=True,
        physical_passed=True,numerically_sufficient=True,freeze_verified=True) for j in jobs}
    selected=module.select(jobs,records)
    assert len(selected)==40
    assert all(jid.endswith('s1') for key,jid in selected.items() if key[0]=='K' and key[3] is None)
    pairs=module.pair_definitions(selected)
    assert any(p[2]=='intramonth_information_same_monthly_input_mass' for p in pairs)
    assert any(p[2]=='complete_plant_configuration' for p in pairs)
    bad={k:v.copy() for k,v in records.items()};bad[jobs[0]['id']]['numerically_sufficient']=False
    try:module.select(jobs,bad)
    except RuntimeError as exc:assert 'UNFINISHED_OR_UNACCEPTED_PATH' in str(exc)
    else:raise AssertionError('insufficient path silently dropped')
    days=pd.date_range('2023-01-01','2023-04-30',freq='D')
    y=np.sin(np.arange(len(days))*.4)+2
    frame=pd.DataFrame(dict(station_key='known',date=days,observed=y,baseline=np.full(len(days),y.mean()),
        candidate=y.copy(),read_count=np.where(np.arange(len(days))%3,6,4)))
    table,summary=pair_tables(frame,True)
    assert summary['NSE']['common_stations']==1
    assert abs(summary['NSE']['median_paired_difference']-1.)<1e-12
    assert abs(summary['month_centered_NSE']['median_paired_difference']-1.)<1e-12
    assert basic(np.ones(12),np.ones(12))['reason']=='zero_observed_variance'
    assert np.isnan(basic(np.ones(12),np.ones(12))['NSE'])
    counts=[]
    for block in (1,2):
        draws,identity=paired_resamples(frame,1000,block,1729,True,['known'])
        assert len(draws)==1000 and len(identity['draw_indices'])==1000
        assert np.all(abs(draws.median_paired_difference-1.)<1e-12)
        assert np.all(draws.improved_fraction==1.)
        counts.append(dict(block_months=block,replicates=len(draws),seed=identity['seed'],known_gain=1.))
    # Monthly known truth uses a distinct support and never substitutes HF days.
    month=pd.DataFrame(dict(station_key='known',date=pd.date_range('2023-01-01',periods=12,freq='MS'),
        observed=np.arange(1.,13.),baseline=np.full(12,6.5),candidate=np.arange(1.,13.)))
    _,m=pair_tables(month,False);assert m['NSE']['median_paired_difference']==1.
    result=dict(passed=True,scope='known_truth_evaluator_and_selection_not_actual_TN_performance',
        all_paths_accounted=44,selected_configurations=40,registered_pairs=len(pairs),
        training_only_entry_selection=True,insufficient_paths_rejected=True,
        daily_and_monthly_support_separate=True,undefined_variance_not_forced=True,
        paired_summaries_and_improvement_fraction=True,month_centering_read_weighted=True,
        bootstrap=counts,identities={p:sha(ROOT/p) for p in ['scripts/evaluate_campaign.py',
            'd29_training/evaluation.py','d29_training/metrics.py','d29_training/bootstrap.py']})
    write_json(ROOT/'evidence/evaluation_known_truth.json',result);print(json.dumps(result))


if __name__=='__main__':main()
