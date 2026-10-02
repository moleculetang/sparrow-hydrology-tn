"""One-time deployment, verified boundary transfer; never a timer or training restart."""
import os,subprocess,time,platform
from mltn.common import ROOT,read,write
assert os.name=='posix'
assert read(ROOT/'evidence/parallel_dispatch_acceptance.json')['passed']
mapping=read(ROOT/'config/external_ownership.json')
for jid,v in mapping['jobs'].items():
    assert v['host']!=platform.node()
    folder=ROOT/'jobs'/jid
    assert not (folder/'owner.lock').exists() and not (folder/'result.json').exists(),'JOB_ALREADY_DISPATCHED '+jid
previous=read(ROOT/'outputs/remaining_owner.json')['pid']
cfg=read(ROOT/'config/execution.json');cfg.update(gpu_jobs=3,neural_threads=1,stagger_seconds=4,status='frozen_after_complete_step_and_concurrency_gate',dispatch_hold=False)
write(ROOT/'config/execution.json',cfg)
from install_gpu_policy import main as policy
from types import SimpleNamespace
policy(SimpleNamespace(expected_new=None,slots=3,phase='frozen_production_after_full_step_gate'))
log=(ROOT/'outputs/dispatch_handoff.log').open('ab')
p=subprocess.Popen([os.sys.executable,'-B',str(ROOT/'handoff_controller.py'),'--previous',str(previous)],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
for _ in range(100):
    marker=ROOT/'outputs/dispatch_handoff.json'
    if marker.exists() and read(marker).get('new_pid')==p.pid:break
    if p.poll() is not None:raise RuntimeError('HANDOFF_FAILED_BEFORE_SUPERVISOR_FREEZE')
    time.sleep(.1)
else:raise RuntimeError('HANDOFF_BOUNDARY_NOT_CONFIRMED')
print(read(marker))
