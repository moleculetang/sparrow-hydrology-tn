"""Resource-owned twelve fixed replays, never fitting or reading heldout scores."""
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
from d29_platform.runtime import write_json,sha
from train_job import validate_admission


def read(path):return json.loads(path.read_text(encoding='utf8')) if path.exists() else None


def main(role,maximum):
    from controller_lock import acquire
    lifetime_lock=acquire(ROOT/'outputs'/('replays_'+role)/'lifetime.lock')
    _,gate_hash=validate_admission()
    if role!=('local' if os.name=='nt' else 'workstation'):raise RuntimeError('REPLAY_HOST_ROLE_MISMATCH')
    plan=read(ROOT/'deployment/replay_ownership.json')
    if not plan or plan['formal_gate_sha256']!=gate_hash or plan['max_active'][role]!=maximum:
        raise RuntimeError('REGISTERED_REPLAY_OWNERSHIP_REQUIRED')
    jobs=[j for j in read(ROOT/'config/jobs.json') if j['space_block'] is None and j['calendar']!='K' and j['entry']==0]
    if len(jobs)!=12 or set(plan['owners'])!={j['id'] for j in jobs}:raise RuntimeError('ALL12REPLAY_OWNERS_REQUIRED')
    folder=ROOT/'outputs'/('replays_'+role);folder.mkdir(parents=True,exist_ok=True)
    previous=read(folder/'owner.json')
    if previous and registry.identity(previous['pid'])==previous['created']:raise RuntimeError('LIVE_REPLAY_CONTROLLER')
    old=read(folder/'status.json') or {};running={};events=old.get('events',[]);failed=old.get('failed',[])
    for r in old.get('running',[]):
        if registry.identity(r['pid'])==r['created']:raise RuntimeError('LIVE_ORPHAN_REPLAY_REQUIRES_IDENTITY_HANDOFF')
        receipt=read(ROOT/'outputs/jobs'/('replay_'+r['job'])/'replay.json')
        if not receipt or not receipt.get('passed'):failed.append(r['job'])
    owner=dict(pid=os.getpid(),created=registry.identity(os.getpid()),role=role,host=platform.node())
    write_json(folder/'owner.json',owner)
    mine=[j for j in jobs if plan['owners'][j['id']]==role]
    while True:
        for jid,r in list(running.items()):
            r['peak']=max(r['peak'],registry.rss(r['process'].pid));code=r['process'].poll()
            if code is None:continue
            r['log'].close();registry.release(r['lease']['token'],root=registry_root())
            events.append(dict(job=jid,exit_code=code,peak_rss_bytes=r['peak']))
            if code!=0:failed.append(jid)
            del running[jid]
        done=[];queue=[]
        for j in mine:
            jid=j['id'];r=read(ROOT/'outputs/jobs'/('replay_'+jid)/'replay.json')
            if r:
                if r.get('passed') and r['formal_gate_sha256']==gate_hash:done.append(jid)
                elif jid not in failed:failed.append(jid)
            elif jid not in running and jid not in failed:queue.append(jid)
        decision=None
        if queue and len(running)<maximum and not (folder/'pause_requested.json').exists():
            jid=queue[0];decision=registry.acquire('20261008_1','replay/'+jid,threads=1,peak_bytes=plan['peak_bytes'][role],root=registry_root())
            if decision.get('granted'):
                env=os.environ.copy();env.update(PYTHONUTF8='1',D29_WORKER_THREADS='1')
                if os.name=='posix':env.update(D29_RESOURCE_STATE=str(registry_root()),D29_H1_ROOT=plan['remote_h1_root'])
                log=(folder/(jid+'.log')).open('ab');child=None
                try:
                    child=subprocess.Popen([sys.executable,'-X','utf8','-B',str(ROOT/'scripts/guarded_worker.py'),decision['token'],
                        'scripts/replay_config.py',jid],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                    registry.bind(decision['token'],child.pid,root=registry_root())
                except Exception:
                    if child is None or child.poll() is not None:registry.release(decision['token'],root=registry_root())
                    log.close();raise
                running[jid]=dict(process=child,log=log,lease=decision,created=registry.identity(child.pid),peak=0)
        report=dict(owner=owner,formal_gate_sha256=gate_hash,phase='running_replays',completed=done,failed=sorted(set(failed)),events=events,
            running=[dict(job=jid,pid=r['process'].pid,created=r['created'],lease=r['lease'],peak_rss_bytes=r['peak']) for jid,r in running.items()],
            last_resource_decision=decision,scientific_fits_dispatched=0)
        write_json(folder/'status.json',report)
        if not running and not queue:
            report['phase']='all_owned_replays_passed' if not failed else 'replay_failures_require_review'
            write_json(folder/'status.json',report);return 0 if not failed else 2
        if not running and (folder/'pause_requested.json').exists():return 0
        time.sleep(20)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--role',choices=('local','workstation'),required=True);p.add_argument('--max-active',type=int,required=True)
    a=p.parse_args();sys.exit(main(a.role,a.max_active))
