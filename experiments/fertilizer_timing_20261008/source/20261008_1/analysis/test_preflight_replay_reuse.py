"""Actual two preflight artifact readbacks; final44 refusal and identity controls."""
import json,sys,copy
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'analysis'),str(ROOT/'scripts')]
from d29_platform.runtime import sha,write_json
from preflight_replay_reuse import KINDS,JOB,actual_preflight,verified_for_final

def main():
    job=next(j for j in json.loads((ROOT/'config/jobs.json').read_text()) if j['id']==JOB)
    source=ROOT/'outputs/jobs'/JOB;cases={};controls=[]
    for kind,(folder,prefix,_,_) in KINDS.items():
        destination=ROOT/'outputs/jobs'/(prefix+JOB)
        actual,aggregate=actual_preflight(kind,destination,source,job)
        cases[kind]=dict(actual=actual,aggregate_sha256=aggregate)
        controls.append(kind+'_actual64year230reach_readback')
        try:verified_for_final(kind,destination,source,job)
        except RuntimeError as e:
            if str(e)!='FINAL_CORRECTED44DELIVERY_NOT_READY':raise
            controls.append(kind+'_incomplete44_refused_first')
        else:raise AssertionError('INCOMPLETE44_ACCEPTED')
        bad=copy.deepcopy(job);bad['id']='TIM_F23_C0_V0_s0'
        for name,args in [('wrong_job',(kind,destination,source,bad)),('wrong_output',(kind,destination.parent/'other',source,job)),('wrong_source',(kind,destination,source.parent/'other',job))]:
            try:actual_preflight(*args)
            except RuntimeError:controls.append(kind+'_'+name+'_rejected')
            else:raise AssertionError('UNSAFE_PREFLIGHT_ACCEPTED '+name)
        path=ROOT/'outputs'/folder/'complete.json';original=json.loads(path.read_text());reader=Path.read_text
        for name,edit in [('false_final',lambda v:v.update(mode='final')),('partial',lambda v:v.update(accounted_paths=0)),('read_labels',lambda v:v.update(evaluation_labels_read=True)),('added_training',lambda v:v.update(new_training_calls=1)),('changed_case',lambda v:v['cases'][0].update(complete_sha256='wrong'))]:
            altered=copy.deepcopy(original);edit(altered)
            def read_override(p,*a,**kw):return json.dumps(altered) if p.resolve()==path.resolve() else reader(p,*a,**kw)
            with patch.object(Path,'read_text',read_override):
                try:actual_preflight(kind,destination,source,job)
                except RuntimeError:controls.append(kind+'_'+name+'_rejected')
                else:raise AssertionError('FALSE_PREFLIGHT_ACCEPTED '+name)
    files=('analysis/preflight_replay_reuse.py','analysis/run_approved_internal_trace_delivery.py',
           'analysis/run_approved_bnf_delivery.py','analysis/audit_final_approved_delivery.py','analysis/test_preflight_replay_reuse.py')
    proof=dict(passed=True,controls=controls,actual_cases=cases,exact44_admission_preserved=True,evaluation_labels_read=False,
               additional_forward_calls=0,files={p:sha(ROOT/p) for p in files})
    write_json(ROOT/'evidence/preflight_replay_reuse_controls.json',proof)
    print(json.dumps(dict(passed=True,controls=len(controls),actual_preflight_cases=2,additional_forward_calls=0)))

if __name__=='__main__':main()
