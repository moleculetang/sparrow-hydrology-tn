"""Trace nested model selection and OOF summaries back to selected predictions."""
from pathlib import Path
import sys
import json
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'5_Test/20260905_1/scripts'))
from common import RUNTIME, atomic_json, sha256, utc_now
import pandas as pd
import numpy as np


def independent_metrics(frame):
    assert not frame.observation_id.duplicated().any()
    assert np.isfinite(frame[['tn_mg_l', 'prediction_mg_l']]).all().all()
    values=[]; log_errors=[]
    for _, group in frame.groupby('station_key'):
        y=group.tn_mg_l.to_numpy(); p=group.prediction_mg_l.to_numpy()
        denominator=float(np.square(y-y.mean()).sum())
        if len(y)>=8 and denominator>0:
            values.append(1-float(np.square(p-y).sum())/denominator)
        log_errors.append(float(np.sqrt(np.square(np.log1p(p)-np.log1p(y)).mean())))
    return dict(rows=len(frame), stations=frame.station_key.nunique(), defined_nse=len(values),
                median_nse=float(np.median(values)) if values else None,
                q25_nse=float(np.quantile(values,.25)) if values else None,
                mean_station_log_rmse=float(np.mean(log_errors)))


def error_dynamics(frame):
    """MSE identity diagnostics; never create corrected model predictions."""
    rows=[]
    for station,group in frame.groupby('station_key'):
        y=group.tn_mg_l.to_numpy(float); p=group.prediction_mg_l.to_numpy(float)
        vy=float(np.var(y)); vp=float(np.var(p))
        cov=float(np.mean((y-y.mean())*(p-p.mean())))
        bias2=float((p.mean()-y.mean())**2); mse=float(np.mean((p-y)**2))
        variation=vy+vp-2*cov
        np.testing.assert_allclose(mse,bias2+variation,rtol=1e-12,atol=1e-12)
        if len(y)<8 or vy<=0: continue
        rows.append(dict(station_key=station,absolute_bias_mg_l=abs(float(p.mean()-y.mean())),
            bias_fraction=bias2/mse if mse>0 else 0.,sd_ratio=float(np.sqrt(vp/vy)),
            time_r=cov/np.sqrt(vy*vp) if vp>0 else np.nan,
            centered_nse=1-variation/vy))
    table=pd.DataFrame(rows)
    def median(name):
        value=table[name].median() if len(table) else np.nan
        return float(value) if np.isfinite(value) else None
    return dict(stations=len(table),median_absolute_bias_mg_l=median('absolute_bias_mg_l'),
        median_bias_fraction=median('bias_fraction'),median_sd_ratio=median('sd_ratio'),
        median_time_r=median('time_r'),median_centered_nse_diagnostic=median('centered_nse'),
        interpretation='MSE = squared mean bias + observed variance + predicted variance - 2 covariance. Centered NSE uses evaluation means for diagnosis only; it is not an achievable process prediction.')


def main():
    run=ROOT/'5_Test/20260905_5'; sources={}
    def read(path):
        sources[str(path)]=sha256(path)
        return json.loads(path.read_text(encoding='utf-8'))
    def frame(path):
        sources[str(path)]=sha256(path)
        return pd.read_parquet(path)
    def equal(left,right):
        columns=['observation_id','station_key','reach_id','terminal_tree_id','tn_mg_l','prediction_mg_l']
        a=left[columns].sort_values('observation_id').reset_index(drop=True)
        b=right[columns].sort_values('observation_id').reset_index(drop=True)
        pd.testing.assert_frame_equal(a,b,check_exact=True)
    def score_check(actual,recorded):
        for key,value in independent_metrics(actual).items():
            if value is None: assert recorded[key] is None
            else: np.testing.assert_allclose(value,recorded[key],rtol=1e-12,atol=1e-12)
    def selected_predictions(candidate,split):
        selected=read(run/'reports'/f"{candidate}_{split.lower()}_selected.json")
        assert len(selected['all_starts'])==5
        assert {r['start'] for r in selected['all_starts']}==set(range(5))
        assert selected['selected']==min(selected['all_starts'],key=lambda r:(r['objective'],r['start']))
        return frame(run/'outputs'/f"{selected['selected']['tag']}_predictions.parquet")
    registry=read(run/'reports/nested_split_registry.json')
    candidates=read(run/'reports/nested_candidate_registry.json')['candidates']
    observations=pd.read_parquet(ROOT/'5_Test/20260905_1/outputs/observations.parquet')
    sources[str(ROOT/'5_Test/20260905_1/outputs/observations.parquet')]=sha256(ROOT/'5_Test/20260905_1/outputs/observations.parquet')
    completed=[]; pending=[]; errors=[]; inner_checks=[]; outer_diagnostics=[]
    outputs={'reach':{'new':[],'control':[]},'tree':{'new':[],'control':[]}}
    def check_inner(item,outer):
        candidate=item['candidate_id']
        parts=[selected_predictions(candidate,inner['id']) for inner in outer['inner']]
        combined=pd.concat(parts,ignore_index=True)
        expected_ids={oid for inner in outer['inner'] for oid in inner['eval_ids']}
        assert set(combined.observation_id)==expected_ids
        saved=frame(run/'outputs'/f"{candidate}_{outer['id'].lower()}_inner_oof.parquet")
        equal(combined,saved); score_check(saved,item['summary'])
        inner_checks.append(dict(outer_id=outer['id'],candidate_id=candidate,rows=len(saved)))
    for kind,key in [('reach','outer_reach'),('tree','outer_terminal_tree')]:
        for outer in registry[key]:
            path=run/'reports'/f"{outer['id'].lower()}_selection.json"
            if not path.exists():
                pending.append(outer['id'])
                partial=run/'reports'/f"{outer['id'].lower()}_inner_scores_partial.json"
                if partial.exists():
                    try:
                        for item in read(partial): check_inner(item,outer)
                    except Exception as error: errors.append(dict(outer_id=outer['id'],error=repr(error)))
                continue
            try:
                selection=read(path); scores=selection['inner_scores']
                expected={c['candidate_id'] for c in candidates}
                actual={s['candidate_id'] for s in scores}
                assert len(actual)==len(scores) and expected<=actual<=expected|{'timing_plus_dynamic'}
                for item in scores: check_inner(item,outer)
                winner=min((s for s in scores if s['summary']['median_nse'] is not None),key=lambda s:(-s['summary']['median_nse'],s['candidate_id']))
                assert selection['chosen']['candidate_id']==winner['candidate_id']
                outer_metrics={}; outer_dynamics={}
                for label,candidate in [('new',winner['candidate_id']),('control','control_h7')]:
                    pred=selected_predictions(candidate,outer['id'])
                    saved=frame(run/'outputs'/f"{outer['id'].lower()}_{label}_predictions.parquet")
                    equal(pred,saved)
                    assert set(saved.observation_id)==set(outer['eval_ids'])
                    assert not saved.station_seen_in_training.any()
                    truth=observations.loc[observations.observation_id.isin(outer['eval_ids'])]
                    equal(saved,truth.assign(prediction_mg_l=truth.observation_id.map(saved.set_index('observation_id').prediction_mg_l)))
                    outputs[kind][label].append(saved)
                    outer_metrics[label]=independent_metrics(saved)
                    outer_dynamics[label]=error_dynamics(saved)
                n,c=outer_metrics['new'],outer_metrics['control']
                outer_diagnostics.append(dict(outer_id=outer['id'],outer_kind=kind,
                    chosen=winner['candidate_id'],**outer_metrics,error_dynamics=outer_dynamics,
                    median_nse_delta=n['median_nse']-c['median_nse'] if n['median_nse'] is not None and c['median_nse'] is not None else None,
                    q25_nse_delta=n['q25_nse']-c['q25_nse'] if n['q25_nse'] is not None and c['q25_nse'] is not None else None,
                    role='Outer-specific descriptive diagnostic; not used in selection or a replacement for registered aggregate gates'))
                completed.append(outer['id'])
            except Exception as error: errors.append(dict(outer_id=outer['id'],error=repr(error)))
    group_comparisons={}
    for kind,key in [('reach','outer_reach'),('tree','outer_terminal_tree')]:
        expected_outers={outer['id'] for outer in registry[key]}
        if expected_outers and expected_outers<=set(completed):
            expected_ids={oid for outer in registry[key] for oid in outer['eval_ids']}
            group={}
            for label,parts in outputs[kind].items():
                combined=pd.concat(parts,ignore_index=True)
                assert set(combined.observation_id)==expected_ids
                group[label]=dict(metrics=independent_metrics(combined),error_dynamics=error_dynamics(combined))
            group.update(completed_outer_ids=sorted(expected_outers),
                scope='Complete cohort for this holdout type only; other registered holdout types and final stages may still be incomplete. No new selection.')
            group_comparisons[kind]=group
    summary_path=run/'reports/nested_validation_summary.json'
    if not pending and not errors and summary_path.exists():
        result=read(summary_path)
        try:
            assert result['status']=='NESTED_VALIDATION_COMPLETE'
            assert {s['outer_id'] for s in result['selections']}==set(completed)
            for kind in outputs:
                for label,parts in outputs[kind].items():
                    combined=pd.concat(parts,ignore_index=True)
                    saved=frame(run/'outputs'/f'nested_{kind}_{label}_oof.parquet')
                    equal(combined,saved); score_check(saved,result['comparisons'][kind][label])
        except Exception as error: errors.append(dict(stage='final_summary',error=repr(error)))
    else:
        if not pending and not errors: pending.append('final_summary')
    status='FAIL_NESTED_SELECTION' if errors else ('PASS_PARTIAL_NESTED_SELECTION' if pending else 'PASS_COMPLETE_NESTED_SELECTION')
    atomic_json(dict(status=status,runtime=RUNTIME,created_utc=utc_now(),completed=completed,pending=pending,
                     errors=errors,inner_checks=inner_checks,outer_diagnostics=outer_diagnostics,
                     completed_group_comparisons=group_comparisons,
                     code_sha256=sha256(Path(__file__)),sources=sources,
                     scope='Exact prediction lineage and independent median NSE/quantile/log-RMSE recomputation; no new fitting'),run/'reports/nested_selection_audit.json')
    print('NESTED_SELECTION_AUDIT',status,len(completed),len(pending),len(errors),'inner_candidates',len(inner_checks),flush=True)
    if errors: raise RuntimeError(errors)


if __name__=='__main__': main()
