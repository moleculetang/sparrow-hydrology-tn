"""Cancel only the isolated unadmitted fourth-slot probe, never production training."""
import os,signal,subprocess
from pathlib import Path
from types import SimpleNamespace
from mltn.common import ROOT,read,write
from mltn.resources import registry
from install_gpu_policy import main as policy
name='Linux_GraphTCN_fourth_slot';job='perf_joint_'+name
wait=ROOT/'outputs/resource_wait'/(job+'.json')
mod,state=registry();s=mod.status(state);memory=s['memory']
result=ROOT/'evidence/performance'/(name+'.json')
receipt=dict(probe=name,actual_memory=memory,live_leases=[dict(project=v['project'],job=v['job'],pid=v['pid'],peak_bytes=v.get('peak_bytes'),reserved_bytes=v.get('reserved_bytes'),working_bytes=mod.rss(v['pid']),gpu=v.get('gpu')) for v in s['leases'].values()],production_gpu_jobs=3)
if result.exists():receipt['status']='finished_before_cancellation';receipt['result']=read(result)
else:
    w=read(wait);pid=w['pid'];identity=mod.identity(pid)
    argv=[p.decode() for p in (Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0') if p]
    cwd=(Path('/proc')/str(pid)/'cwd').resolve()
    script=next(p for p in argv if p.endswith('benchmark_joint.py'))
    assert (cwd/script).resolve()==(ROOT/'benchmark_joint.py').resolve() and argv[argv.index('--name')+1]==name,argv
    assert not any(v['pid']==pid for v in s['leases'].values()),'PROBE_ALREADY_ADMITTED_DO_NOT_CANCEL'
    assert identity is not None and mod.identity(pid)==identity
    os.kill(pid,signal.SIGTERM)
    receipt.update(status='unadmitted_probe_cancelled',reason=w['reason'],probe_pid=pid,production_training_interrupted=False)
policy(SimpleNamespace(expected_new=None,slots=3,phase='frozen_production_after_fourth_slot_resource_rejection'))
write(ROOT/'evidence/performance/fourth_slot_admission.json',receipt);print(dict(status=receipt['status'],memory=memory))
