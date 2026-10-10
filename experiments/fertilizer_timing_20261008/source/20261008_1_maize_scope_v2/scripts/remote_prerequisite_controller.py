"""One-shot two-worker remote prerequisite queue, not44-path training.

Explicit shared registry and host-local owner identity. Does not rerun either
Windows audit, migrate optimizer ownership, or release a live child.
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
    if os.name!='posix':raise RuntimeError('REMOTE_PREREQUISITES_CONTROLLER_REQUIRES_LINUX')
    hold=json.loads((ROOT/'config/workstation_hold.json').read_text(encoding='utf8'))
    if hold['enabled']:raise RuntimeError('WORKSTATION_PAUSED')
    folder=ROOT/'outputs/remote_prerequisites';folder.mkdir(parents=True,exist_ok=True)
    ownerpath=folder/'owner.json'
    if ownerpath.exists():
        old=json.loads(ownerpath.read_text(encoding='utf8'))
        if registry.identity(old['pid'])==old['created']:raise RuntimeError('EXISTING_REMOTE_PREREQUISITES_CONTROLLER_LIVE')
    owner=dict(pid=os.getpid(),created=registry.identity(os.getpid()),host=platform.node(),registry=str(registry_root()))
    write_json(ownerpath,owner)
    jobs=['TIM_F23_C0_V0_s0','TIM_F23_K_V1_s0'];running={};completed=[];started=[]
    state=dict(owner=owner,phase='diagnostic_queue',registered_diagnostics=jobs,scientific_fit_paths_dispatched=0)
    while len(completed)<len(jobs):
        for jid in jobs:
            if jid in started:continue
            lease=registry.acquire('20261008_1','diagnostic/'+jid,threads=1,peak_bytes=6500000000,root=registry_root())
            if not lease.get('granted'):
                state['resource_wait']=lease;break
            env=os.environ.copy();env.update(PYTHONUTF8='1',D29_WORKER_THREADS='1',D29_RESOURCE_STATE=str(registry_root()))
            env['D29_H1_ROOT']='/home/pc/TWY/SPARROW/Test/20260928_1/linux_land1_12/model/vendor/legacy22/data/domains/FULL24C'
            log=(folder/(jid+'.log')).open('ab')
            try:
                child=subprocess.Popen([sys.executable,'-X','utf8','-B',str(ROOT/'scripts/guarded_worker.py'),lease['token'],
                    'scripts/diagnostic_oracle.py',jid,'--cycles','2','--ledger'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                registry.bind(lease['token'],child.pid,root=registry_root())
            except Exception:
                if 'child' not in locals() or child.poll() is not None:registry.release(lease['token'],root=registry_root())
                log.close();raise
            running[jid]=dict(process=child,lease=lease,log=log,created=registry.identity(child.pid),peak_rss=0)
            started.append(jid)
            # Stagger heavy imports/first compilations; this is a controller
            # lifecycle interval, not a scheduled or recurring task.
            break
        for jid,item in list(running.items()):
            child=item['process'];item['peak_rss']=max(item['peak_rss'],registry.rss(child.pid))
            code=child.poll()
            if code is not None:
                item['log'].close();registry.release(item['lease']['token'],root=registry_root())
                completed.append(dict(job=jid,exit_code=code,peak_rss_bytes=item['peak_rss']))
                del running[jid]
        state.update(running=[dict(job=j,pid=r['process'].pid,created=r['created'],lease=r['lease'],peak_rss_bytes=r['peak_rss']) for j,r in running.items()],completed=completed)
        write_json(folder/'status.json',state)
        if any(r['exit_code']!=0 for r in completed):
            # Let already admitted scientific children exit naturally. Preserve
            # their leases; no signal is sent and no new path is admitted.
            for item in running.values():item['process'].wait();item['log'].close();registry.release(item['lease']['token'],root=registry_root())
            state['phase']='failure_requires_review';write_json(folder/'status.json',state);return 2
        if len(completed)<len(jobs):time.sleep(10)
    state['phase']='finished_cross_host_comparison_pending';write_json(folder/'status.json',state);return 0


if __name__=='__main__':sys.exit(main())
