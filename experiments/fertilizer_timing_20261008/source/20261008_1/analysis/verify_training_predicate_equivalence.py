"""Full-history oracle equivalence of strict constructor row predicates.

Time reference is an already accepted saved oracle. The spatial/K reference
reproduces the existing logical train slicing in a distinct diagnostic ID.
No evaluation score is calculated and no scientific optimizer is restarted.
"""
import gc
import json
import os
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import numpy as np
import pandas as pd
from unittest.mock import patch
from d29_platform.runtime import write_json,sha
from timing.adapter import TimingTraining
from freeze_job import training_only_adapter


def compare(a,b,tolerance,relative=False):
    a=np.asarray(a);b=np.asarray(b)
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        return dict(passed=False,reason='shape_or_nonfinite')
    delta=abs(a-b)
    return dict(passed=bool(np.all(delta<=tolerance*(1+abs(a)) if relative else delta<=tolerance)),
                bitwise=bool(np.array_equal(a,b)),max_abs_difference=float(delta.max(initial=0.)))


def main():
    from host_resources import registry,registry_root
    from train_job import validate_admission
    validate_admission()
    lease=registry.status(root=registry_root())['leases'].get(os.environ.get('D29_LEASE_TOKEN'))
    if not lease or lease['job']!='analysis/training_predicate_equivalence' or lease['pid']!=os.getpid() or lease['created']!=registry.identity(os.getpid()):
        raise RuntimeError('EXACT_NATIVE_EQUIVALENCE_LEASE_REQUIRED')
    constructor_only='--constructor-only' in sys.argv
    target=ROOT/'outputs'/('training_predicate_access_probe_v2' if constructor_only else 'training_predicate_equivalence_v2');target.mkdir(exist_ok=True)
    if (target/'complete.json').exists():raise RuntimeError('PREDICATE_EQUIVALENCE_ALREADY_COMPLETED')
    jobs=json.loads((ROOT/'config/jobs.json').read_text());cases=[];start=time.monotonic()
    selected=['TIM_F23_C0_V0_s0','TIM_S56_K_V1_s0']
    for index,jid in enumerate(selected):
        job=next(j.copy() for j in jobs if j['id']==jid)
        if constructor_only:
            reference_scope='constructor_access_probe_only_no_oracle'
        elif index==0:
            source=ROOT/'outputs/jobs/measurement_threads1_TIM_F23_C0_V0_s0'
            receipt=json.loads((source/'stage_measurement.json').read_text())
            if not receipt['passed'] or receipt['formal_gate_sha256']!=sha(ROOT/'evidence/formal_gate.json'):
                raise RuntimeError('ACCEPTED_TIME_ORACLE_REQUIRED')
            point=np.load(source/'point.npy',allow_pickle=False)
            expected=dict(value=json.loads((source/'physical_ledger.json').read_text())['objective'],
                gradient=np.load(source/'gradient.npy',allow_pickle=False),
                final=np.load(source/'final_state.npy',allow_pickle=False),
                local=np.load(source/'local_kg_day.npy',allow_pickle=False),
                prediction=np.load(source/'training_prediction.npy',allow_pickle=False))
            reference_scope='existing_accepted_saved_oracle'
        else:
            old=TimingTraining(dict(job,id='predicate_reference_'+jid));point=old.initial.copy()
            value,gradient=old.value_gradient(point)
            expected=dict(value=float(value),gradient=gradient.copy(),final=old.last_chain.final.copy(),
                          local=old.local_prediction.copy(),prediction=old.prediction.copy())
            reference_scope='existing_constructor_full_table_read_then_logical_train_slice'
            del old;gc.collect()
        excluded=set()
        if job['space_block'] is not None:
            blocks=json.loads((ROOT/'vendor/legacy22/data/spatial_blocks.json').read_text());b=blocks[str(job['space_block'])]
            excluded=set(b['held_stations'])|set(b['buffer_stations'])
        reader=pd.read_parquet;reads=[]
        def observed_read(path,*args,**kwargs):
            frame=reader(path,*args,**kwargs);p=Path(path).resolve()
            if p in [(ROOT/'data/monthly_tn.parquet').resolve(),(ROOT/'data/hf_daily.parquet').resolve()]:
                years=frame.year if 'year' in frame else frame.date.dt.year
                excluded_count=int(frame.station_key.isin(excluded).sum())
                invalid_count=int(years.gt(job['train_end']).sum())
                reads.append(dict(file=p.relative_to(ROOT).as_posix(),rows=len(frame),filters=repr(kwargs.get('filters')),
                    evaluation_year_rows_returned=invalid_count,held_or_buffer_rows_returned=excluded_count))
                if invalid_count or excluded_count:raise RuntimeError('EVALUATION_OR_EXCLUDED_ROW_EXPOSED_TO_CONSTRUCTOR')
            return frame
        with patch.object(pd,'read_parquet',side_effect=observed_read):
            a=training_only_adapter(dict(job,id='predicate_strict_'+jid))
        write_json(target/(jid+'_constructor_reads.json'),dict(reads=reads,scope='actual_parquet_returned_frames_before_oracle'))
        if {z['file'] for z in reads}!={'data/monthly_tn.parquet','data/hf_daily.parquet'}:
            raise RuntimeError('BOTH_CONSTRUCTOR_OBSERVATION_TABLE_READS_REQUIRED')
        if constructor_only:
            cases.append(dict(job=jid,passed=True,constructor_reads=reads,
                              scope=reference_scope,oracle_equivalence_checked=False))
            del a;gc.collect();continue
        value,gradient=a.value_gradient(point)
        checks=dict(gradient=compare(expected['gradient'],gradient,1e-6,True),
            final=compare(expected['final'],a.last_chain.final,1e-6),
            local=compare(expected['local'],a.local_prediction,1e-6),
            prediction=compare(expected['prediction'],a.prediction,1e-8,True),
            objective=compare(np.array([expected['value']]),np.array([value]),1e-8,True))
        record=dict(job=jid,passed=all(c['passed'] for c in checks.values()),reference_scope=reference_scope,
            full_history=[1961,2024],coordinate_count=len(point),checks=checks,constructor_reads=reads,
            new_predicate_constructor_exposed_evaluation_or_held_rows=False,
            source_multiplier=1.,evaluation_scores_calculated=False)
        cases.append(record);write_json(target/'progress.json',dict(cases=cases,seconds=time.monotonic()-start))
        if not record['passed']:raise RuntimeError('PREDICATE_ORACLE_EQUIVALENCE_FAILED '+jid)
        del a,expected;gc.collect()
    result=dict(passed=True,cases=cases,seconds=time.monotonic()-start,
        production_scientific_sources_or_live_checkpoints_changed=False,
        oracle_equivalence_checked=not constructor_only,
        counts={'new_time_full_history_calls':0 if constructor_only else 1,'spatial_full_history_reference_and_strict_calls':0 if constructor_only else 2},
        scope='two_representative_full_oracles_not_all44endpoint_audits',
        identities={n:sha(ROOT/n) for n in ['analysis/verify_training_predicate_equivalence.py','scripts/freeze_job.py']})
    write_json(target/'complete.json',result);print(json.dumps(dict(passed=True,cases=len(cases),seconds=result['seconds'])))


if __name__=='__main__':main()
