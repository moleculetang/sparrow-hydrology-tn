"""One-shot six-slice K/V1 qualification, using existing shared host leases."""
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
    if os.name!='posix':raise RuntimeError('WORKSTATION_ONLY_COORDINATE_QUEUE')
    if json.loads((ROOT/'config/workstation_hold.json').read_text(encoding='utf8'))['enabled']:
        raise RuntimeError('WORKSTATION_PAUSED')
    folder=ROOT/'outputs/remote_coordinates';folder.mkdir(parents=True,exist_ok=True)
    ownerpath=folder/'owner.json'
    if ownerpath.exists():
        old=json.loads(ownerpath.read_text(encoding='utf8'))
        if registry.identity(old['pid'])==old['created']:raise RuntimeError('COORDINATE_CONTROLLER_ALREADY_LIVE')
    owner=dict(pid=os.getpid(),created=registry.identity(os.getpid()),host=platform.node(),registry=str(registry_root()))
    write_json(ownerpath,owner);running={};done=[];started=set()
    for index in range(6):
        receipt=ROOT/f'outputs/jobs/prerequisite_K_coordinates_{index}/coordinate_slice.json'
        if receipt.exists():
            r=json.loads(receipt.read_text(encoding='utf8'))
            if r['complete']:
                done.append(dict(slice=index,exit_code=0 if r['passed'] else 2,previously_completed=True));started.add(index)
    state=dict(owner=owner,phase='coordinate_queue',scientific_fit_paths_dispatched=0)
    while len(done)<6:
        if not any(r['exit_code']!=0 for r in done):
            for index in range(6):
                if index in started:continue
                lease=registry.acquire('20261008_1',f'prerequisite/K_coordinates_{index}',threads=1,peak_bytes=6500000000,root=registry_root())
                if not lease.get('granted'):state['resource_wait']=lease;break
                env=os.environ.copy();env.update(PYTHONUTF8='1',D29_WORKER_THREADS='1',D29_RESOURCE_STATE=str(registry_root()),
                    D29_H1_ROOT='/home/pc/TWY/SPARROW/Test/20260928_1/linux_land1_12/model/vendor/legacy22/data/domains/FULL24C')
                log=(folder/f'slice_{index}.log').open('ab');child=None
                try:
                    child=subprocess.Popen([sys.executable,'-X','utf8','-B',str(ROOT/'scripts/guarded_worker.py'),lease['token'],
                        'scripts/prerequisite_coordinate_slice.py',str(index)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                    registry.bind(lease['token'],child.pid,root=registry_root())
                except Exception:
                    if child is None or child.poll() is not None:registry.release(lease['token'],root=registry_root())
                    log.close();raise
                running[index]=dict(process=child,lease=lease,log=log,created=registry.identity(child.pid),peak_rss=0)
                started.add(index);break
        for index,item in list(running.items()):
            child=item['process'];item['peak_rss']=max(item['peak_rss'],registry.rss(child.pid))
            code=child.poll()
            if code is not None:
                item['log'].close();registry.release(item['lease']['token'],root=registry_root())
                done.append(dict(slice=index,exit_code=code,peak_rss_bytes=item['peak_rss']));del running[index]
        state.update(running=[dict(slice=i,pid=r['process'].pid,created=r['created'],lease=r['lease'],peak_rss_bytes=r['peak_rss']) for i,r in running.items()],completed=done)
        write_json(folder/'status.json',state)
        if not running and any(r['exit_code']!=0 for r in done):
            state['phase']='failure_requires_review';write_json(folder/'status.json',state);return 2
        if len(done)<6:time.sleep(15)
    failed=any(r['exit_code']!=0 for r in done)
    state['phase']='failure_requires_review' if failed else 'completed_scientific_merge_pending'
    write_json(folder/'status.json',state);return 2 if failed else 0


if __name__=='__main__':sys.exit(main())
