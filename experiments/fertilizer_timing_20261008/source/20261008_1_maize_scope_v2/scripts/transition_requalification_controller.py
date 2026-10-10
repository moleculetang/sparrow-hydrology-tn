"""Two finite full-history rechecks; remote source swap after old science exits."""
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
from controller_lock import acquire
from d29_platform.runtime import write_json,sha


def main(role):
    if role!=('local' if os.name=='nt' else 'workstation'):raise RuntimeError('REQUALIFICATION_ROLE_MISMATCH')
    folder=ROOT/'outputs'/('transition_requalification_'+role);folder.mkdir(parents=True,exist_ok=True)
    lifetime=acquire(folder/'lifetime.lock');owner=dict(pid=os.getpid(),created=registry.identity(os.getpid()),host=platform.node())
    state=dict(owner=owner,phase='waiting_old_coordinate_owner_exit',scientific_fits_dispatched=0)
    write_json(folder/'owner.json',owner);write_json(folder/'status.json',state)
    if role=='workstation':
        old=json.loads((ROOT/'outputs/remote_coordinates/owner.json').read_text())
        while registry.identity(old['pid'])==old['created']:time.sleep(20)
        current=ROOT/'d29_platform/timing_kernel.py';candidate=ROOT/'transfer/timing_kernel_transition_repaired.py'
        expected=json.loads((ROOT/'transfer/transition_repair_transport.json').read_text())
        if sha(current)!=expected['new_source_sha256']:
            if sha(current)!=expected['old_source_sha256'] or sha(candidate)!=expected['new_source_sha256']:
                raise RuntimeError('REMOTE_KERNEL_REPAIR_SOURCE_MISMATCH')
            leases=registry.status(root=registry_root())['leases']
            if any(v['project']=='20261008_1' for v in leases.values()):raise RuntimeError('LIVE_PROJECT_PREVENTS_KERNEL_UPDATE')
            backup=ROOT/'evidence/source_snapshots/timing_kernel_pre_transition_reverse_fix.py';backup.parent.mkdir(exist_ok=True)
            if backup.exists() and sha(backup)!=expected['old_source_sha256']:raise RuntimeError('REMOTE_BEFORE_SOURCE_SNAPSHOT_CONFLICT')
            if not backup.exists():backup.write_bytes(current.read_bytes())
            os.replace(candidate,current)
        write_json(ROOT/'evidence/remote_transition_repair_activation.json',dict(passed=True,old_coordinate_owner=old,
            old_source_sha256=expected['old_source_sha256'],new_source_sha256=sha(current),old_science_interrupted=False,forward_changed=False))
    jobs=['TIM_F23_C0_V0_s0','TIM_F23_K_V1_s0'];running={};done=[]
    for jid in tuple(jobs):
        p=ROOT/'outputs/jobs'/('transition_requalification_'+jid)/'transition_requalification.json'
        if p.exists():
            r=json.loads(p.read_text());
            if not r['passed'] or r['identities']['d29_platform/timing_kernel.py']!=sha(ROOT/'d29_platform/timing_kernel.py'):
                raise RuntimeError('STALE_REQUALIFICATION_REQUIRES_REVIEW')
            done.append(dict(job=jid,exit_code=0,reused=True));jobs.remove(jid)
    while jobs or running:
        for jid,r in list(running.items()):
            r['peak']=max(r['peak'],registry.rss(r['process'].pid));code=r['process'].poll()
            if code is None:continue
            r['log'].close();registry.release(r['lease']['token'],root=registry_root())
            done.append(dict(job=jid,exit_code=code,peak_rss_bytes=r['peak']));del running[jid]
        decision=None
        if jobs and not any(v['exit_code'] for v in done):
            jid=jobs[0];decision=registry.acquire('20261008_1','transition_requalification/'+jid,threads=1,peak_bytes=6500000000,root=registry_root())
            if decision.get('granted'):
                env=os.environ.copy();env.update(PYTHONUTF8='1',D29_WORKER_THREADS='1')
                if role=='workstation':env.update(D29_RESOURCE_STATE=str(registry_root()),D29_H1_ROOT='/home/pc/TWY/SPARROW/Test/20260928_1/linux_land1_12/model/vendor/legacy22/data/domains/FULL24C')
                child=None;log=(folder/(jid+'.log')).open('ab')
                try:
                    child=subprocess.Popen([sys.executable,'-X','utf8','-B',str(ROOT/'scripts/guarded_worker.py'),decision['token'],
                        'scripts/transition_requalifier.py',jid],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                    registry.bind(decision['token'],child.pid,root=registry_root())
                except Exception:
                    if child is None or child.poll() is not None:registry.release(decision['token'],root=registry_root())
                    log.close();raise
                running[jid]=dict(process=child,log=log,lease=decision,created=registry.identity(child.pid),peak=0);jobs.remove(jid)
        state.update(phase='full_history_new_adjoint_requalification',remaining=jobs,completed=done,last_resource_decision=decision,
            running=[dict(job=j,pid=r['process'].pid,created=r['created'],lease=r['lease'],peak_rss_bytes=r['peak']) for j,r in running.items()])
        write_json(folder/'status.json',state)
        if not running and any(v['exit_code'] for v in done):state['phase']='failure_requires_review';write_json(folder/'status.json',state);return 2
        if jobs or running:time.sleep(20)
    state['phase']='full_history_new_adjoint_returned_FD_merge_pending';write_json(folder/'status.json',state);return 0


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--role',choices=('local','workstation'),required=True);a=p.parse_args();sys.exit(main(a.role))
