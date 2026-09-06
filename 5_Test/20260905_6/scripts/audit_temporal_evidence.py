"""Independently verify temporal/prior cohorts, selection and reported scores."""
from pathlib import Path
import json
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'5_Test/20260905_5/scripts'))
from audit_nested_selection import independent_metrics,error_dynamics
from common import RUNTIME,atomic_json,sha256,utc_now
import numpy as np
import pandas as pd


def main():
    run=ROOT/'5_Test/20260905_6';sources={};checked=[];pending=[];errors=[]
    def read(path):
        sources[str(path)]=sha256(path)
        return json.loads(path.read_text(encoding='utf-8'))
    def frame(path):
        sources[str(path)]=sha256(path)
        return pd.read_parquet(path)
    observations=frame(ROOT/'5_Test/20260905_1/outputs/observations.parquet')
    candidate=read(run/'reports/development_choice.json')['chosen']['candidate']
    def scores(pred,recorded):
        if not len(pred):
            assert recorded is None
            return None
        result=independent_metrics(pred);dynamics=error_dynamics(pred)
        for name,value in dict(result,median_time_r=dynamics['median_time_r']).items():
            if value is None: assert recorded[name] is None
            else: np.testing.assert_allclose(value,recorded[name],rtol=1e-12,atol=1e-12)
        return dict(metrics=result,error_dynamics=dynamics)
    def selected(stage,identifier,year):
        folder=ROOT/'5_Test'/stage
        selection=read(folder/'reports'/f'{identifier}_selected.json')
        starts=selection['all_starts']
        assert len(starts)==5 and {s['start'] for s in starts}==set(range(5))
        assert selection['selected']==min(starts,key=lambda s:(s['objective'],s['start']))
        spec=read(Path(selection['spec_path']))
        assert selection['spec_sha256']==sha256(selection['spec_path'])
        expected_train=set(observations.loc[observations.primary_gate&observations.year.between(2016,year-1),'observation_id'])
        expected_eval=set(observations.loc[observations.year.eq(year),'observation_id'])
        assert set(spec['train_ids'])==expected_train and set(spec['eval_ids'])==expected_eval
        assert not expected_train&expected_eval and not spec.get('final_fit',False)
        for start in starts:
            report=read(Path(start['path']))
            assert report['converged'] and report['projected_gradient_max']<=1e-5
            assert report['objective']==start['objective'] and report['identity']['spec']==spec
            assert report['identity']['spec_sha256']==selection['spec_sha256']
            assert all(sha256(p)==h for p,h in report['identity']['code_sha256'].items())
            assert set(report['design']['training_observation_ids'])==expected_train
        pred=frame(folder/'outputs'/f"{selection['selected']['tag']}_predictions.parquet")
        assert set(pred.observation_id)==expected_eval and not pred.observation_id.duplicated().any()
        assert np.isfinite(pred.prediction_mg_l).all() and pred.prediction_mg_l.ge(0).all()
        columns=['observation_id','station_key','reach_id','terminal_tree_id','year','tn_mg_l','primary_gate']
        truth=observations.loc[observations.observation_id.isin(expected_eval),columns].sort_values('observation_id').reset_index(drop=True)
        pd.testing.assert_frame_equal(pred[columns].sort_values('observation_id').reset_index(drop=True),truth,check_exact=True)
        return pred,spec
    temporal_path=run/'reports/temporal_confirmation.json'
    temporal=read(temporal_path) if temporal_path.exists() else {}
    for year in [2024,2025]:
        if str(year) not in temporal:
            pending.append(f'temporal_{year}');continue
        try:
            entry=temporal[str(year)];result={}
            for label,identifier in [('new',candidate['candidate_id']),('control','refit_control_h7')]:
                pred,spec=selected('20260905_6',f'{identifier}_t{year}',year)
                if label=='new':
                    assert all(spec[key]==candidate[key] for key in candidate)
                else: assert spec['model']=='CONTROL_H7' and spec['loss']=='STUDENT_T4_LOG1P'
                result[label]=scores(pred.loc[pred.primary_gate],entry[label])
                for name,column,value in [('reach17_position_unresolved','reach_id',17),('tree163_open_lake','terminal_tree_id',163)]:
                    subset=pred.loc[pred[column].eq(value)];domain=entry['excluded_domains'][name]
                    assert not subset.primary_gate.any()
                    assert set(subset.observation_id)==set(domain['observation_ids']) and len(subset)==domain['rows']
                    scores(subset,domain[label])
            checked.append(dict(kind='temporal',year=year,**result))
        except Exception as error: errors.append(dict(kind='temporal',year=year,error=repr(error)))
    prior_path=ROOT/'5_Test/20260905_5/reports/prior_sensitivity.json'
    prior=read(prior_path) if prior_path.exists() else {'results':[]}
    for scale in [.5,2.]:
        for year in [2023,2024]:
            matches=[r for r in prior['results'] if r['prior_scale']==scale and r['year']==year]
            if not matches:
                pending.append(f'prior_{scale}_{year}');continue
            try:
                assert len(matches)==1
                identifier=candidate['candidate_id']+f'_prior{scale:g}'
                pred,spec=selected('20260905_5',f'{identifier}_t{year}',year)
                expected=dict(candidate,candidate_id=identifier,prior_scale=scale)
                assert all(spec[key]==expected[key] for key in expected)
                checked.append(dict(kind='prior',year=year,prior_scale=scale,**scores(pred.loc[pred.primary_gate],matches[0]['summary'])))
            except Exception as error: errors.append(dict(kind='prior',year=year,prior_scale=scale,error=repr(error)))
    status='FAIL_TEMPORAL_EVIDENCE' if errors else ('PASS_PARTIAL_TEMPORAL_EVIDENCE' if pending else 'PASS_COMPLETE_TEMPORAL_EVIDENCE')
    atomic_json(dict(status=status,runtime=RUNTIME,created_utc=utc_now(),checked=checked,pending=pending,errors=errors,
        sources=sources,code_sha256={str(Path(__file__)):sha256(Path(__file__)),
        str(ROOT/'5_Test/20260905_5/scripts/audit_nested_selection.py'):sha256(ROOT/'5_Test/20260905_5/scripts/audit_nested_selection.py')},
        scope='Independent median NSE, q25, log-RMSE and correlation, exact temporal/excluded cohorts and five-start lineage; no new fits or corrected predictions'),run/'reports/temporal_evidence_audit.json')
    print('TEMPORAL_EVIDENCE_AUDIT',status,len(checked),len(pending),len(errors),flush=True)
    if errors: raise RuntimeError(errors)


if __name__=='__main__':main()
