"""Shared host-local leases; current process is its job owner."""
import os,sys,time,atexit,importlib.util,subprocess
from pathlib import Path
from mltn.common import ROOT,write
def registry():
    base=ROOT.parent/'.compute_coordination' if os.name=='posix' else ROOT.parent/'compute_coordination'
    path=base/'resource_registry.py'
    if not path.exists():return None,None
    spec=importlib.util.spec_from_file_location('sparrow_resource_registry',path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod,base/'state' if os.name=='posix' else mod.DEFAULT
def lease(job,threads,device='cpu'):
    mod,root=registry()
    if mod is None:
        if os.name=='posix':raise RuntimeError('SHARED_RESOURCE_REGISTRY_NOT_AVAILABLE')
        return None
    gpu=None
    if device=='cuda':gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()[0].strip()
    # Measured current jobs plus conservative transient preprocessing reserve.
    peak=4*1024**3 if device=='cuda' else 2*1024**3
    while True:
        # A watcher can adopt an already active old worker; reuse only exact PID identity.
        old=next((v|{'token':k} for k,v in mod.status(root)['leases'].items() if v['project']=='20260930_1' and v['job']==job and v['pid']==os.getpid()),None)
        grant=old if old is not None else mod.acquire('20260930_1',job,threads,peak,pid=os.getpid(),gpu=gpu,root=root)
        if old is not None or grant.get('granted'):break
        write(ROOT/'outputs/resource_wait'/f'{job}.json',dict(job=job,pid=os.getpid(),reason=grant.get('reason')));time.sleep(5)
    mod.affinity(grant['cpus']);write(ROOT/'outputs/resource_leases'/f'{job}.json',grant);atexit.register(lambda:mod.release(grant['token'],root=root));return grant
