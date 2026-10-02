"""Add only new frozen historical references to byte-verified remote evaluation.

Unchanged registered comparisons retain their original 1000-draw results; all
actual station metrics are subsequently recomputed by the independent scalar
auditor. This starts no fitting and does not select on evaluation performance.
"""
import re
import numpy as np
import pandas as pd
from mltn.common import ROOT, read, write, sha
from mltn.metrics import station_table, paired_summary
from mltn.bootstrap import paired
from evaluate import event_table, paired_event_table


def main():
    assert read(ROOT/'evidence/results_recovery.json')['passed']
    stage=read(ROOT/'outputs/postprocess_stages/evaluate.json')
    assert stage['exit_code']==0 and stage['code_sha256']==sha(ROOT/'evaluate.py')
    for path,digest in stage['source_dependencies'].items():
        assert sha(ROOT/path)==digest, ('EVALUATION_DEPENDENCY_CHANGED',path)
    out=ROOT/'outputs/evaluation'
    previous=read(out/'evaluation_receipt.json')
    assert previous['bootstrap_replicates']==1000 and previous['seed']==1729
    table=pd.read_csv(out/'all_station_metrics.csv')
    pairs=read(out/'paired_summary.json')
    freeze=read(ROOT/'outputs/prediction_freeze.json')
    # The import manifest has already verified every archived result file. Here
    # verify every original raw prediction remains unchanged after import.
    for path,digest in freeze.items():assert sha(ROOT/path)==digest, ('PREDICTION_CHANGED',path)
    selection=read(ROOT/'outputs/frozen_selection.json')
    roles=read(ROOT/'outputs/result_roles.json')
    references=[name for name in roles['include'] if 'reference_' in name]
    assert len(references)==9, ('EXPECTED_NINE_DESCRIPTIVE_REFERENCES',references)
    records=[]
    for name in references:
        source=ROOT/'jobs'/name/'prediction.parquet'
        result=read(source.parent/'result.json')
        assert result['role'] in ['descriptive historical reference only','descriptive legacy comparator']
        context=re.search(r'F23|F24',name)[0]
        task='monthly' if '_monthly_' in name else 'daily'
        q=pd.read_parquet(source).rename(columns={'tn_mg_l':'observed'})
        q['date']=pd.to_datetime(q.date)
        q=q[np.isfinite(q.observed)&np.isfinite(q.prediction)]
        assert not q.duplicated(['station_key','date']).any()
        # Supports idempotent recovery of an interrupted append, not replacement
        # of an existing scientific comparison with different predictions.
        frozen=out/f'{name}_{task}_frozen.parquet'
        if frozen.exists():
            prior=pd.read_parquet(frozen)
            pd.testing.assert_frame_equal(prior,q)
        else:q.to_parquet(frozen,index=False)
        table=table[~((table.configuration==name)&(table.task==task))]
        extra=station_table(q,predictions=('prediction',),daily=task=='daily',minimum_coverage=True)
        extra['configuration']=name;extra['context']=context;extra['task']=task
        table=pd.concat([table,extra],ignore_index=True)
        if task=='daily':event_table(q).assign(configuration=name,context=context,task=task).to_csv(out/f'{name}_{task}_events.csv',index=False,encoding='utf-8-sig')
        family=selection['winners'][task];cfg=selection['selected'][family+'_'+task]
        baselines=[f'{context}_{task}_ridge',f'{context}_{task}_{family}_c{cfg}_seedmean']
        for baseline in baselines:
            b=pd.read_parquet(out/f'{baseline}_{task}_frozen.parquet')
            frame=q.merge(b[['station_key','date','prediction']].rename(columns={'prediction':'baseline'}),on=['station_key','date'],validate='one_to_one')
            frame['candidate']=frame.prediction
            assert len(frame), ('NO_COMMON_REFERENCE_SUPPORT',name,baseline)
            t=station_table(frame,predictions=('baseline','candidate'),daily=task=='daily',minimum_coverage=True)
            summary=paired_summary(t);summary.update(context=context,task=task,baseline=baseline,candidate=name)
            if task=='daily':summary['centered']=paired_summary(t,'month_centered_NSE')
            pairs=[p for p in pairs if (p['candidate'],p['baseline'],p['task'])!=(name,baseline,task)]
            pairs.append(summary)
            tag=f'{name}__versus__{baseline}_{task}'
            t.to_csv(out/f'{tag}_paired.csv',index=False,encoding='utf-8-sig')
            if task=='daily':paired_event_table(frame).to_csv(out/f'{tag}_paired_events.csv',index=False,encoding='utf-8-sig')
            valid=set(t[t.nse_eligible].station_key);f=frame[frame.station_key.isin(valid)]
            if len(f):
                for block in [1,2]:paired(f,1000,block,1729).to_parquet(out/f'{tag}_bootstrap_{block}month.parquet',index=False)
        freeze[source.relative_to(ROOT).as_posix()]=sha(source)
        records.append(dict(configuration=name,task=task,rows=len(q),role=result['role'],source_sha256=sha(source)))
    assert not table.duplicated(['configuration','context','task','station_key']).any()
    table.to_csv(out/'all_station_metrics.csv',index=False,encoding='utf-8-sig')
    write(out/'paired_summary.json',pairs);write(ROOT/'outputs/prediction_freeze.json',freeze)
    write(out/'evaluation_receipt.json',previous|dict(
        configurations=len(table[['configuration','context','task']].drop_duplicates()),
        station_metric_rows=len(table),pair_comparisons=len(pairs),
        original_registered_comparisons_reused='byte-verified import; unchanged source dependencies and all original raw prediction hashes; independent actual scalar audit follows',
        appended_historical_comparisons=len(references)*2))
    write(ROOT/'evidence/historical_comparator_append.json',dict(
        passed=True,records=records,appended_pairs=18,fit_calls=0,selection_calls=0,
        original_evaluation_code_sha256=stage['code_sha256'],
        decision='only new descriptive references scored; original registered 1000-draw results retained unchanged'))


if __name__=='__main__':main()
