"""One local full-history diagnostic, with measured peak and exact lease cleanup."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from host_resources import registry,registry_root
from controller_lock import acquire
from d29_platform.runtime import write_json,sha


def main(jid,namespace):
    folder=ROOT/'outputs'/(namespace+'_'+jid+'_launch');folder.mkdir(parents=True,exist_ok=True)
    lock=acquire(folder/'lifetime.lock')
    if (folder/'returned.json').exists():raise RuntimeError('FINISHED_PILOT_REQUIRES_NEW_ID_NOT_RESTART')
    proof=json.loads((ROOT/'evidence/fixed_replays.json').read_text())
    case=next(c for c in proof['cases'] if c['job']==jid)
    source=ROOT/case['receipt']
    if not case['passed'] or sha(source)!=case['sha256']:raise RuntimeError('ACCEPTED_SOURCE_REPLAY_CHANGED')
    lease=registry.acquire('20261008_1','analysis/'+namespace+'/'+jid,threads=1,peak_bytes=6500000000,root=registry_root())
    write_json(folder/'resource_decision.json',lease)
    if not lease.get('granted'):return 4
    env=os.environ.copy();env.update(PYTHONUTF8='1',PYTHONPATH=str(ROOT/'vendor/legacy22/scripts'))
    child=None;peak=dict(rss=0,private_commit=0,peak_commit=0);started=time.monotonic()
    try:
        with (folder/'pilot.log').open('ab') as log:
            child=subprocess.Popen([sys.executable,'-X','utf8','-B',str(ROOT/'scripts/guarded_worker.py'),
                lease['token'],'analysis/trace_accepted_internal_fluxes.py',str(source.parent),'--namespace',namespace],
                cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            registry.bind(lease['token'],child.pid,root=registry_root())
            owner=dict(pid=os.getpid(),created=registry.identity(os.getpid()),
                       child_pid=child.pid,child_created=registry.identity(child.pid),lease=lease)
            write_json(folder/'owner.json',owner)
            write_json(folder/'status.json',dict(phase='read_only_full_history_trace_running',owner=owner,
                registered_job=jid,scientific_fit_signals=0,new_training_calls=0))
            while child.poll() is None:
                m=registry.process_memory(child.pid)
                for k in peak:
                    if m.get(k) is not None:peak[k]=max(peak[k],m[k])
                time.sleep(5)
        complete=ROOT/'outputs/jobs'/(namespace+'_'+jid)/'complete.json'
        passed=child.returncode==0 and complete.exists() and json.loads(complete.read_text()).get('passed',False)
        record=dict(passed=passed,exit_code=child.returncode,owner=owner,native_peak=peak,
            seconds=time.monotonic()-started,complete_sha256=sha(complete) if passed else None,
            physical_sources_modified=False,no_training_or_evaluation_design_changes=True)
        write_json(folder/'returned.json',record)
        write_json(folder/'status.json',dict(phase='full_history_trace_passed' if passed else 'trace_requires_review',**record))
        return 0 if passed else 2
    finally:
        if child is None or child.poll() is not None:registry.release(lease['token'],root=registry_root())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('job');p.add_argument('--namespace',default='internal_trace_pilot_v2')
    a=p.parse_args();sys.exit(main(a.job,a.namespace))
