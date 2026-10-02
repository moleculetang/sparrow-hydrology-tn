"""Fail-closed priority coverage: only imminent CPU/GPU requests count."""
from mltn.common import ROOT,write
from controller import immediate_pending_count,jobid
import tempfile,controller
checks=[]
with tempfile.TemporaryDirectory(dir=ROOT/'transfer') as tmp:
    old=controller.ROOT;controller.ROOT=__import__('pathlib').Path(tmp)
    try:
        cfg=dict(cpu_jobs=4,gpu_jobs=3)
        gpu=dict(role='joint',stage='F23',task='joint',family='GraphTCN',config=0,seed=1729)
        cpu=gpu|dict(family='XGBoost')
        assert immediate_pending_count([gpu,cpu],{},cfg,{},set())==2
        checks.append(dict(check='ready_GPU_host_memory_request_has_priority',passed=True))
        active={jobid(gpu):dict(gpu=True),jobid(cpu):dict(gpu=False)}
        assert immediate_pending_count([],active,cfg,{},set())==2
        assert immediate_pending_count([],active,cfg,{},set(active))==0
        checks.append(dict(check='unleased_active_requests_count_and_leased_jobs_do_not',passed=True))
        external={jobid(gpu):dict(host='fixture_remote')}
        child=gpu|dict(stage='S24',block=56)
        assert immediate_pending_count([gpu,child],{},cfg,external,set())==0
        assert immediate_pending_count([gpu,cpu],{},cfg|dict(dispatch_hold=True),{},set())==0
        checks.append(dict(check='foreign_dependency_and_hold_excluded',passed=True))
        many=[gpu|dict(seed=1729+i) for i in range(10)]
        assert immediate_pending_count(many,{},cfg,{},set())==3
        busy={jobid(j):dict(gpu=True) for j in many[:3]}
        assert immediate_pending_count(many[3:],busy,cfg,{},set(busy))==0
        checks.append(dict(check='priority_limited_to_immediate_slots_no_queue_lock',passed=True))
    finally:controller.ROOT=old
write(ROOT/'evidence/gpu_pending_acceptance.json',dict(passed=True,checks=checks,scientific_changes=False,preemption=False))
print(dict(passed=True,checks=len(checks)))
