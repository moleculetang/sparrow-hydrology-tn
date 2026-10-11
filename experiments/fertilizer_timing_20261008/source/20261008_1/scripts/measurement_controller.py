"""Finite stage measurements, separately leased at1/2/4threads; not fitting."""
import argparse
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
from train_job import validate_admission


def main(role):
    from controller_lock import acquire
    lifetime_lock=acquire(ROOT/'outputs'/('measurements_'+role)/'lifetime.lock')
    _,gate_hash=validate_admission()
    if role!=('local' if os.name=='nt' else 'workstation'):raise RuntimeError('MEASUREMENT_ROLE_MISMATCH')
    folder=ROOT/'outputs'/('measurements_'+role);folder.mkdir(parents=True,exist_ok=True)
    ownerfile=folder/'owner.json'
    if ownerfile.exists():
        old=json.loads(ownerfile.read_text());
        if registry.identity(old['pid'])==old['created']:raise RuntimeError('LIVE_MEASUREMENT_CONTROLLER')
    owner=dict(pid=os.getpid(),created=registry.identity(os.getpid()),host=platform.node(),role=role)
    write_json(ownerfile,owner);variants=(1,) if role=='local' else (1,2,4)
    jid='TIM_F23_C0_V0_s0';done=[];running={};remaining=list(variants)
    for threads in tuple(remaining):
        result=ROOT/f'outputs/jobs/measurement_threads{threads}_{jid}/stage_measurement.json'
        if result.exists():
            r=json.loads(result.read_text());
            if r['formal_gate_sha256']!=gate_hash:raise RuntimeError('MEASUREMENT_GATE_CHANGED')
            done.append(dict(threads=threads,exit_code=0 if r['passed'] else 2,reused=True));remaining.remove(threads)
    while remaining or running:
        for threads,r in list(running.items()):
            r['peak']=max(r['peak'],registry.rss(r['process'].pid));code=r['process'].poll()
            if code is None:continue
            r['log'].close();registry.release(r['lease']['token'],root=registry_root())
            done.append(dict(threads=threads,exit_code=code,peak_rss_bytes=r['peak']));del running[threads]
        decision=None
        if remaining and not any(r['exit_code'] for r in done):
            threads=remaining[0];decision=registry.acquire('20261008_1',f'measurement_threads{threads}/'+jid,threads=threads,peak_bytes=6500000000,root=registry_root())
            if decision.get('granted'):
                env=os.environ.copy();env.update(PYTHONUTF8='1',D29_WORKER_THREADS=str(threads))
                if os.name=='posix':env.update(D29_RESOURCE_STATE=str(registry_root()),D29_H1_ROOT='/home/pc/TWY/SPARROW/Test/20260928_1/linux_land1_12/model/vendor/legacy22/data/domains/FULL24C')
                child=None;log=(folder/f'threads{threads}.log').open('ab')
                try:
                    child=subprocess.Popen([sys.executable,'-X','utf8','-B',str(ROOT/'scripts/guarded_worker.py'),decision['token'],
                        'scripts/measure_stages.py',jid],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                    registry.bind(decision['token'],child.pid,root=registry_root())
                except Exception:
                    if child is None or child.poll() is not None:registry.release(decision['token'],root=registry_root())
                    log.close();raise
                running[threads]=dict(process=child,log=log,lease=decision,created=registry.identity(child.pid),peak=0);remaining.remove(threads)
        state=dict(owner=owner,formal_gate_sha256=gate_hash,phase='finite_stage_measurements',remaining=remaining,completed=done,last_resource_decision=decision,
            running=[dict(threads=t,pid=r['process'].pid,created=r['created'],lease=r['lease'],peak_rss_bytes=r['peak']) for t,r in running.items()],
            scientific_fit_paths_dispatched=0)
        write_json(folder/'status.json',state)
        if not running and any(d['exit_code'] for d in done):return 2
        if remaining or running:time.sleep(20)
    state['phase']='all_measurements_returned_comparison_pending';write_json(folder/'status.json',state)
    return 0


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--role',choices=('local','workstation'),required=True);a=p.parse_args();sys.exit(main(a.role))
