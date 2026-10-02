"""Synthetic integration gates: preserve prior comparisons and reject changed identity."""
import shutil
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import append_historical_comparators as app
import evaluate as ev
from mltn.common import ROOT, sha, write, read


def fixture(root):
    for folder in ['outputs/evaluation','outputs/postprocess_stages','jobs','evidence','mltn']:
        (root/folder).mkdir(parents=True,exist_ok=True)
    shutil.copyfile(ROOT/'evaluate.py',root/'evaluate.py')
    shutil.copyfile(ROOT/'mltn/metrics.py',root/'mltn/metrics.py')
    write(root/'evidence/results_recovery.json',{'passed':True})
    write(root/'outputs/postprocess_stages/evaluate.json',dict(exit_code=0,code_sha256=sha(root/'evaluate.py'),source_dependencies={'mltn/metrics.py':sha(root/'mltn/metrics.py')}))
    write(root/'outputs/frozen_selection.json',dict(winners={'monthly':'CatBoost','daily':'CatBoost'},selected={'CatBoost_monthly':0,'CatBoost_daily':0}))
    write(root/'outputs/evaluation/evaluation_receipt.json',dict(bootstrap_replicates=1000,seed=1729,contexts=4))
    oldpair=dict(context='fixture',task='monthly',baseline='preserve_a',candidate='preserve_b',common_stations=2,median_paired_difference=.123)
    write(root/'outputs/evaluation/paired_summary.json',[oldpair])
    (root/'outputs/evaluation/untouched_bootstrap.parquet').write_bytes(b'SYNTHETIC_PRESERVATION_SENTINEL')
    rows=[];names=[];ref=pd.DataFrame()
    for context,year in [('F23',2023),('F24',2024)]:
        for task in ['monthly','daily']:
            dates=pd.date_range(f'{year}-01-01',periods=12 if task=='monthly' else 90,freq='MS' if task=='monthly' else 'D')
            q=pd.DataFrame({'station_key':np.repeat(['test_a','test_b'],len(dates)),'date':list(dates)*2})
            q['observed']=1.5+.5*np.sin(np.arange(len(q))*.2);q['prediction']=q.observed*.8
            if task=='daily':q['read_count']=np.tile(np.arange(len(dates))%6+1,2);ref=q.rename(columns={'observed':'tn_mg_l'})
            for name in [f'{context}_{task}_ridge',f'{context}_{task}_CatBoost_c0_seedmean']:
                q.to_parquet(root/f'outputs/evaluation/{name}_{task}_frozen.parquet',index=False)
                rows.append(dict(configuration=name,context=context,task=task,station_key='test_a'))
            models=['U','LAND1']+(['old_tree'] if context=='F24' and task=='daily' else [])
            for model in models:
                name=f'{context}_{task}_reference_{model}';names.append(name)
                folder=root/'jobs'/name;folder.mkdir()
                q.rename(columns={'observed':'tn_mg_l'}).to_parquet(folder/'prediction.parquet',index=False)
                write(folder/'result.json',dict(role='descriptive historical reference only'))
    pd.DataFrame(rows).to_csv(root/'outputs/evaluation/all_station_metrics.csv',index=False)
    write(root/'outputs/result_roles.json',dict(include=names))
    raw=root/'original_prediction_identity.dat';raw.write_bytes(b'ORIGINAL_SYNTHETIC_PREDICTION')
    write(root/'outputs/prediction_freeze.json',{raw.name:sha(raw)})
    return ref,oldpair


def main():
    records=[]
    original_root=app.ROOT;original_event=app.event_table;original_pair=app.paired_event_table
    empty_events=pd.DataFrame({'eligible':pd.Series(dtype=bool)})
    try:
        for case in ['preserve_and_append','changed_prediction','changed_dependency']:
            with tempfile.TemporaryDirectory(prefix='synthetic_append_',dir=ROOT/'transfer') as temp:
                root=Path(temp);reference,oldpair=fixture(root);app.ROOT=root
                app.event_table=lambda frame:ev.event_table(frame,events=empty_events,reference=reference)
                app.paired_event_table=lambda frame:ev.paired_event_table(frame,events=empty_events,reference=reference)
                sentinel=root/'outputs/evaluation/untouched_bootstrap.parquet';before=sha(sentinel)
                if case=='changed_prediction':(root/'original_prediction_identity.dat').write_bytes(b'CHANGED')
                if case=='changed_dependency':(root/'mltn/metrics.py').write_bytes(b'CHANGED')
                if case=='preserve_and_append':
                    app.main();pairs=read(root/'outputs/evaluation/paired_summary.json')
                    assert pairs[0]==oldpair and len(pairs)==19 and sha(sentinel)==before
                    assert read(root/'evidence/historical_comparator_append.json')['appended_pairs']==18
                    app.main();assert len(read(root/'outputs/evaluation/paired_summary.json'))==19
                    assert sha(sentinel)==before
                else:
                    try:app.main()
                    except AssertionError:pass
                    else:raise AssertionError('IDENTITY_MISMATCH_WAS_NOT_REJECTED '+case)
                    assert read(root/'outputs/evaluation/paired_summary.json')==[oldpair]
                records.append(dict(case=case,passed=True))
    finally:app.ROOT=original_root;app.event_table=original_event;app.paired_event_table=original_pair
    write(ROOT/'evidence/historical_append_acceptance.json',dict(passed=True,tests=records,data='isolated synthetic fixture; no real TN read or fit; original results untouched'))
    print('HISTORICAL_APPEND_GATES_PASS',len(records))


if __name__=='__main__':main()
