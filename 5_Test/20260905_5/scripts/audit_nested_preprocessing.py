"""Audit actual saved fit designs against outer/inner observation exclusions."""
from pathlib import Path
import sys
import json
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'5_Test/20260905_1/scripts'))
from common import RUNTIME,atomic_json,sha256,utc_now,memory_guard
import numpy as np
import pandas as pd


def main():
    run=ROOT/'5_Test/20260905_5'
    registry_path=run/'reports/nested_split_registry.json'
    registry=json.loads(registry_path.read_text(encoding='utf-8'))
    for p,h in registry['inputs'].items():
        if sha256(p)!=h:raise RuntimeError('Registered split inputs changed')
    obs=pd.read_parquet(ROOT/'5_Test/20260905_1/outputs/observations.parquet')
    feature_path=ROOT/'5_Test/20260905_1/outputs/h7_raw_features.parquet'
    feature_hash=sha256(feature_path)
    features=pd.read_parquet(feature_path).set_index('reach_id')
    split_map={};outer_map={}
    for outer in registry['outer_reach']+registry['outer_terminal_tree']:
        split_map[outer['id']]=outer;outer_map[outer['id']]=outer
        for inner in outer['inner']:split_map[inner['id']]=inner;outer_map[inner['id']]=outer
    split_checks=[]
    for name,split in split_map.items():
        train=obs.loc[obs.observation_id.isin(split['train_ids'])]
        test=obs.loc[obs.observation_id.isin(split['eval_ids'])]
        outer=outer_map[name]
        assert len(train)==len(set(split['train_ids'])) and len(test)==len(set(split['eval_ids']))
        assert train.primary_gate.all() and test.primary_gate.all()
        assert train.year.between(2016,2023).all() and test.year.between(2020,2023).all()
        assert not set(train.station_key)&set(test.station_key)
        assert not train.reach_id.isin(split['heldout_reaches']).any()
        assert not train.reach_id.isin(outer['eval_reaches']).any()
        if name!=outer['id']:assert not test.reach_id.isin(outer['eval_reaches']).any()
        assert test.reach_id.isin(split['eval_reaches']).all()
        split_checks.append(dict(split=name,train_rows=len(train),eval_rows=len(test),all_station_histories_on_held_reaches_excluded=True))
    water=None;seen=[];errors=[];cached={};dynamic_count=0
    for path in sorted((run/'reports').glob('*_s[0-4].json')):
        report=json.loads(path.read_text(encoding='utf-8'));spec=report['identity'].get('spec',{})
        if spec.get('evaluation_role')!='nested_spatial':continue
        try:
            assert report['identity']['prepared_features_sha256']==feature_hash
            split=split_map[spec['split_id']]
            assert set(spec['train_ids'])==set(split['train_ids']) and set(spec['eval_ids'])==set(split['eval_ids'])
            assert set(report['design']['training_observation_ids'])==set(split['train_ids'])
            assert set(spec['heldout_reaches'])==set(split['heldout_reaches'])
            design=report['design'];name=spec['split_id']
            if name not in cached:
                train=obs.loc[obs.observation_id.isin(split['train_ids'])]
                reaches=sorted(train.reach_id.unique());raw=features.loc[reaches].to_numpy(float)
                low=np.quantile(raw,.01,axis=0);high=np.quantile(raw,.99,axis=0);clipped=np.clip(raw,low,high)
                grouped=train.groupby('station_key').tn_mg_l
                count=grouped.size();variance=grouped.var(ddof=0)
                floor=float(np.quantile(variance.loc[(count>=12)&(variance>0)],.1))
                cached[name]=dict(fields=features.columns.tolist(),training_reaches=reaches,
                    low=low,high=high,mean=clipped.mean(axis=0),sd=clipped.std(axis=0),training_variance_floor_mg_l_squared=floor)
            expected=cached[name]
            assert design['fields']==expected['fields'] and sorted(design['training_reaches'])==expected['training_reaches']
            for key in ['low','high','mean','sd','training_variance_floor_mg_l_squared']:
                np.testing.assert_allclose(design[key],expected[key],rtol=1e-12,atol=1e-12)
            if spec.get('dynamic'):
                if water is None:
                    hydro=ROOT/'5_Test/20260828_35/outputs/tn_hydrology_reach_daily.parquet'
                    water=pd.read_parquet(hydro,columns=['date','reach_id','upper_response_storage_mm','percolation_to_lower_mm_day'])
                    water['date']=pd.to_datetime(water.date);water=water.loc[water.date.dt.year.between(2016,2023)].copy()
                subset=water.loc[water.reach_id.isin(expected['training_reaches'])]
                fields={'upper_water_mm':'upper_response_storage_mm','percolation_mm_day':'percolation_to_lower_mm_day'}
                for scale in design['dynamic_scales']:
                    values=np.log1p(subset[fields[scale['field']]].to_numpy(float))
                    np.testing.assert_allclose([scale['mean'],scale['sd']],[values.mean(),values.std()],rtol=1e-12,atol=1e-12)
                dynamic_count+=1
            seen.append(dict(path=str(path),sha256=sha256(path),split_id=name,start=spec.get('start',report['identity']['start']),model=spec['model'],loss=spec['loss']))
        except Exception as error:errors.append(dict(path=str(path),error=repr(error)))
    final_summary=run/'reports/nested_validation_summary.json'
    complete=final_summary.exists() and json.loads(final_summary.read_text(encoding='utf-8'))['status']=='NESTED_VALIDATION_COMPLETE'
    result=dict(status='PASS_COMPLETE_NESTED_PREPROCESSING' if complete and not errors else ('FAIL_NESTED_PREPROCESSING' if errors else 'PASS_AVAILABLE_DESIGNS_SPATIAL_RUN_INCOMPLETE'),
        runtime=RUNTIME,created_utc=utc_now(),split_checks=split_checks,actual_fit_reports_checked=len(seen),
        dynamic_designs_checked=dynamic_count,fit_reports=seen,errors=errors,memory=memory_guard(),
        code_sha256=sha256(Path(__file__)),registry_sha256=sha256(registry_path),feature_sha256=feature_hash,
        claim_scope='actual observation exclusion, saved environmental preprocessing and training-only TN variance floor; not a spatial skill claim')
    atomic_json(result,run/'reports/nested_preprocessing_audit.json')
    print('NESTED_PREPROCESSING_AUDIT',result['status'],len(seen),dynamic_count,len(errors),flush=True)
    if errors:raise RuntimeError(errors)


if __name__=='__main__':main()
