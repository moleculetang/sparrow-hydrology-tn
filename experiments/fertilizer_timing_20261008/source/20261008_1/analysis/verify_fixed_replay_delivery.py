"""Read back all12 fixed-point controls without TN labels or model calls.

Existing historical receipts remain immutable. This checks their actual bytes
and support, including the annotated stock/flux semantics used for contrasts.
It does not treat fixed replay shape changes as heldout improvement.
"""
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'analysis')]
from d29_platform.runtime import sha,write_json


def read(path):return json.loads(Path(path).read_text())


def path_under(root,relative):
    path=(root/relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root.resolve()):
        raise RuntimeError('IN_EXPERIMENT_FIXED_REPLAY_BINDING_REQUIRED')
    return path


def unique(rows,ids,key='job'):
    found=[r[key] for r in rows]
    if len(found)!=len(ids) or len(set(found))!=len(ids) or set(found)!=ids:
        raise RuntimeError('EXACT_UNIQUE12_FIXED_REPLAY_IDS_REQUIRED')


def verify(root=ROOT):
    from physical_waveform_diagnostics import verified
    jobs=read(root/'config/jobs.json')
    selected={j['id']:j for j in jobs if j['space_block'] is None and j['calendar']!='K' and j['entry']==0}
    if len(selected)!=12:raise RuntimeError('EXACT_FIXED_C012_V01_F2324_REQUIRED')
    files=['evidence/fixed_replays.json','outputs/replay_internal_traces/complete.json',
        'outputs/physical_shape_diagnostics_v3/complete.json',
        'outputs/replay_internal_traces/annotated_shapes_v2/complete.json',
        'outputs/replay_internal_traces/paired_complete_v2.json',
        'outputs/physical_shape_diagnostics_v3/paired_complete.json']
    fixed,internal,wave,annotation,internal_pairs,station_pairs=[read(root/p) for p in files]
    ids=set(selected)
    for proof,key in ((fixed,'cases'),(internal,'completed'),(wave,'cases'),(annotation,'cases')):
        if proof.get('passed') is not True or proof.get('count')!=12:raise RuntimeError('ALL12_FIXED_DIAGNOSTICS_REQUIRED')
        unique(proof[key],ids)
    if (internal.get('fixed_replays_sha256')!=sha(root/files[0]) or wave.get('fixed_replays_sha256')!=sha(root/files[0])
            or annotation.get('source_complete_sha256')!=sha(root/files[1])
            or annotation.get('physical_values_changed') is not False or annotation.get('observations_read') is not False):
        raise RuntimeError('FIXED_REPLAY_DIAGNOSTIC_DEPENDENCIES_CHANGED')
    trace_cases={r['job']:r for r in internal['completed']}
    wave_cases={r['job']:r for r in wave['cases']}
    annotations={r['job']:r for r in annotation['cases']}
    points={};warmups={};cases=[]
    end=int((__import__('datetime').date(2016,1,1)-__import__('datetime').date(1961,1,1)).days)
    for case in fixed['cases']:
        jid=case['job'];job=selected[jid];receipt=path_under(root,case['receipt'])
        if sha(receipt)!=case['sha256']:raise RuntimeError('FIXED_REPLAY_RECEIPT_CHANGED')
        physical,identity,kind=verified(receipt.parent)
        if (kind!='fixed_parameter_replay' or identity.get('registered_job')!=jid
                or any(identity.get(k)!=job[k] for k in ('fold','calendar','plant_version'))
                or identity.get('source_multiplier')!=1. or identity.get('evaluation_scores_read') is not False
                or sha(receipt.parent/'training_support_prediction.npy')!=identity.get('prediction_sha256')
                or physical.get('history')!=[1961,2024] or physical.get('reaches')!=230 or physical.get('land_slots')!=13):
            raise RuntimeError('FIXED_REPLAY_SCIENTIFIC_SUPPORT_CHANGED')
        fold=job['fold'];point=identity['point_sha256']
        if fold in points and points[fold]!=point:raise RuntimeError('NONMATCHED_FIXED_PARAMETER_CONTRAST')
        points[fold]=point
        with np.load(receipt.parent/'full_history_physical_source_ledger.npz',allow_pickle=False) as z:
            errors={}
            for key in ('local_kg_day','local_fast_kg_day','local_slow_kg_day'):
                values=z[key]
                if values.shape!=(23376,230) or values.dtype!=np.float64 or not np.isfinite(values).all():
                    raise RuntimeError('EXACT_FLOAT64_FIXED_HISTORY_REQUIRED')
                prior=values[:end]
                if (fold,key) not in warmups:warmups[fold,key]=prior.copy()
                errors[key]=float(np.max(abs(prior-warmups[fold,key]),initial=0.))
                if errors[key]>1e-6:raise RuntimeError('FIXED_POINT_WARMUP_CHANGED_BEFORE2016')
        tc=trace_cases[jid];tp=path_under(root,tc['receipt'])
        if sha(tp)!=tc['sha256']:raise RuntimeError('FIXED_INTERNAL_TRACE_RECEIPT_CHANGED')
        t=read(tp)
        if (t.get('passed') is not True or t.get('registered_job')!=job or t.get('full_history')!=[1961,2024]
                or t.get('source_point_sha256')!=point or t.get('physical_ledger_sha256')!=identity['physical_ledger_sha256']
                or t.get('no_evaluation_outcomes_read') is not True or t.get('prediction_passed') is not True
                or t.get('new_scientific_training_calls')!=0 or t.get('network',{}).get('passed') is not True
                or not np.isfinite(t['objective_replay_error'])
                or t['objective_replay_error']>1e-8*(1+abs(physical['objective']))):
            raise RuntimeError('FULL_HISTORY_FIXED_TRACE_CONTRACT_REQUIRED')
        if [r['year'] for r in t.get('years',[])]!=list(range(1961,2025)):
            raise RuntimeError('EXACT64YEAR_FIXED_TRACE_REQUIRED')
        for y in t['years']:
            yp=path_under(tp.parent,y['file'])
            if (sha(yp)!=y['sha256'] or y.get('passed') is not True or y.get('internal_receipt',{}).get('passed') is not True
                    or set(y.get('source_ledger_export_comparison',{}))!={'local','fast','slow'}
                    or any(c.get('passed') is not True or not 0<=c['max_abs_difference_kg']<=1e-6
                           for c in y['source_ledger_export_comparison'].values())):
                raise RuntimeError('FIXED_ANNUAL_TRACE_BYTES_OR_EXPORTS_CHANGED')
        for name,key in (('routing_trace.npz','routing_trace_sha256'),('internal_and_network_shapes.csv','shapes_sha256')):
            if sha(tp.parent/name)!=t[key]:raise RuntimeError('FIXED_NETWORK_OR_SHAPES_CHANGED')
        an=annotations[jid]
        if (an.get('source_receipt_sha256')!=tc['sha256'] or an.get('original_shapes_sha256')!=t['shapes_sha256']
                or sha(path_under(root,an['annotated_file']))!=an['annotated_sha256']):
            raise RuntimeError('ANNOTATED_FIXED_STOCK_FLUX_SEMANTICS_CHANGED')
        wc=wave_cases[jid];wp=path_under(root,wc['receipt'])
        if sha(wp)!=wc['sha256']:raise RuntimeError('FIXED_STATION_SHAPE_RECEIPT_CHANGED')
        w=read(wp)
        if (w.get('passed') is not True or w.get('parameter_sha256')!=point
                or w.get('physical_receipt_sha256')!=identity['physical_ledger_sha256']
                or w.get('observation_data_read') is not False or w.get('station_count')!=116):
            raise RuntimeError('FULL_STATION_FIXED_SHAPE_SUPPORT_REQUIRED')
        for relative,digest in w['output_sha256'].items():
            if sha(path_under(wp.parent,relative))!=digest:raise RuntimeError('FIXED_STATION_SHAPE_BYTES_CHANGED')
        cases.append(dict(job=jid,point_sha256=point,warmup_export_max_errors_kg=errors,
            replay_sha256=case['sha256'],internal_sha256=tc['sha256'],station_sha256=wc['sha256']))
    if (internal_pairs.get('passed') is not True or internal_pairs.get('count')!=12
            or internal_pairs.get('source_complete_sha256')!=sha(root/files[1])
            or internal_pairs.get('unit_annotation_receipt_sha256')!=sha(root/files[3])
            or sha(root/'outputs/replay_internal_traces/paired_internal_and_network_shapes_v2.csv')!=internal_pairs['csv_sha256']
            or station_pairs.get('passed') is not True or station_pairs.get('contrasts')!=12
            or station_pairs.get('shape_analysis_complete_sha256')!=sha(root/files[2])
            or station_pairs.get('observation_data_read') is not False):
        raise RuntimeError('ALL12_FIXED_MONTH_AND_INTRAMONTH_CONTRASTS_REQUIRED')
    for name,digest in station_pairs['output_sha256'].items():
        if sha(path_under(root/'outputs/physical_shape_diagnostics_v3',name))!=digest:
            raise RuntimeError('FIXED_PAIRED_STATION_TABLE_CHANGED')
    return dict(passed=True,scope='all12_fixed_replays_not44_calibrated_paths',accounted_paths=12,
        fixed_replay_counts=12,internal_years=12*64,certified_station_count=116,cases=cases,
        dependencies={p:sha(root/p) for p in files},evaluation_labels_read=False,
        new_training_calls=0,model_rerun=False,water_quality_improvement_claimed=False,
        NSE='not_applicable_fixed_physical_control',verifier_sha256=sha(Path(__file__)))


if __name__=='__main__':
    result=verify();write_json(ROOT/'evidence/approved_fixed_replay_readback.json',result)
    print(json.dumps(dict(passed=True,accounted_paths=result['accounted_paths'],internal_years=result['internal_years'])))
