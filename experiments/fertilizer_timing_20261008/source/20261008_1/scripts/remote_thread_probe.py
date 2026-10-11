"""Serial two-/four-thread full-target measurements under campaign load.

An isolated output namespace avoids replacing the accepted one-thread arrays.
This is a diagnostic only; default production workers remain one thread.
"""
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from host_resources import registry,registry_root
from d29_platform.runtime import write_json


def main():
    if os.name!='posix':raise RuntimeError('WORKSTATION_THREAD_PROBE_ONLY')
    if json.loads((ROOT/'config/workstation_hold.json').read_text(encoding='utf8'))['enabled']:raise RuntimeError('WORKSTATION_PAUSED')
    folder=ROOT/'outputs/remote_thread_probe';folder.mkdir(parents=True,exist_ok=True)
    marker=folder/'owner.json'
    if marker.exists():
        old=json.loads(marker.read_text(encoding='utf8'))
        if registry.identity(old['pid'])==old['created']:raise RuntimeError('THREAD_PROBE_ALREADY_LIVE')
    owner=dict(pid=os.getpid(),created=registry.identity(os.getpid()),host=platform.node());write_json(marker,owner)
    completed=[]
    for threads in (2,4):
        namespace=f'threads{threads}';jid='TIM_F23_K_V1_s0'
        output=ROOT/f'outputs/jobs/diagnostic_{namespace}_{jid}/diagnostic_oracle.json'
        if output.exists():completed.append(dict(threads=threads,already_completed=True));continue
        while True:
            lease=registry.acquire('20261008_1',f'diagnostic_threads{threads}/'+jid,threads=threads,peak_bytes=6500000000,root=registry_root())
            write_json(folder/'status.json',dict(owner=owner,phase='resource_wait',requested_threads=threads,lease=lease,completed=completed))
            if lease.get('granted'):break
            time.sleep(20)
        env=os.environ.copy();env.update(PYTHONUTF8='1',D29_WORKER_THREADS=str(threads),D29_RESOURCE_STATE=str(registry_root()),
            D29_H1_ROOT='/home/pc/TWY/SPARROW/Test/20260928_1/linux_land1_12/model/vendor/legacy22/data/domains/FULL24C')
        child=None
        try:
            with (folder/f'threads{threads}.log').open('ab') as log:
                child=subprocess.Popen([sys.executable,'-X','utf8','-B',str(ROOT/'scripts/guarded_worker.py'),lease['token'],
                    'scripts/diagnostic_oracle.py',jid,'--cycles','2','--namespace',namespace],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                registry.bind(lease['token'],child.pid,root=registry_root())
                write_json(folder/'status.json',dict(owner=owner,phase='running_full_target',requested_threads=threads,
                    child=dict(pid=child.pid,created=registry.identity(child.pid)),lease=lease,completed=completed))
                code=child.wait()
        finally:
            if child is None or child.poll() is not None:registry.release(lease['token'],root=registry_root())
        completed.append(dict(threads=threads,exit_code=code))
        if code!=0:
            write_json(folder/'status.json',dict(owner=owner,phase='failure_requires_review',completed=completed));return code
    write_json(folder/'status.json',dict(owner=owner,phase='finished_numerical_comparison_pending',completed=completed));return 0


if __name__=='__main__':sys.exit(main())
