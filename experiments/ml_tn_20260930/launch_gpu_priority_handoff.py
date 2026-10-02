"""One-time checked supervisor boundary: never stop scientific children."""
import os,time,subprocess
from pathlib import Path
from mltn.common import ROOT,read,write,sha
from mltn.resources import registry
assert os.name=='posix'
assert read(ROOT/'evidence/gpu_pending_acceptance.json')['passed']
assert read(ROOT/'evidence/handoff_lock_acceptance.json')['passed']
assert not (ROOT/'outputs/all_training_done.json').exists()
prior=read(ROOT/'outputs/remaining_owner.json');previous=prior['pid']
mod,state=registry();created=mod.identity(previous);assert created is not None
write(ROOT/'evidence/gpu_priority_handoff_previous_owner.json',dict(owner=prior,created=created,controller_sha256=sha(ROOT/'controller.py'),handoff_sha256=sha(ROOT/'handoff_controller.py')))
log=(ROOT/'outputs/gpu_priority_handoff.log').open('ab')
p=subprocess.Popen([os.sys.executable,'-B',str(ROOT/'handoff_controller.py'),'--previous',str(previous)],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
for _ in range(100):
    marker=ROOT/'outputs/dispatch_handoff.json'
    if marker.exists() and read(marker).get('new_pid')==p.pid:break
    if p.poll() is not None:
        # Constructor failure must not leave delivery waiting for a dead owner.
        write(ROOT/'outputs/remaining_owner.json',prior)
        if mod.identity(previous)==created:
            import signal
            os.kill(previous,signal.SIGCONT)
        raise RuntimeError('NEW_SUPERVISOR_FAILED '+str(p.pid))
    time.sleep(.1)
else:raise RuntimeError('HANDOFF_BOUNDARY_NOT_CONFIRMED')
write(ROOT/'evidence/gpu_priority_handoff_launch.json',dict(previous_pid=previous,new_pid=p.pid,training_signals_sent=False,priority='real immediate CPU and GPU requests, no future queue lock',controller_sha256=sha(ROOT/'controller.py')))
print(read(marker))
