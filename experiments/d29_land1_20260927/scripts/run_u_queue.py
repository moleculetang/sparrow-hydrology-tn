"""Finite U queue for this experiment; no timer, service or external task."""
import sys,json,time,subprocess,os
from pathlib import Path
from datetime import datetime
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from d29_platform.runtime import configure,write_json,dispatch_allowed
configure()
jobs=[j for j in json.loads((ROOT/'config/jobs.json').read_text(encoding='utf-8')) if j['model']=='U']
deadline=datetime.fromisoformat(json.loads((ROOT/'config/clock.json').read_text(encoding='utf-8'))['dispatch_deadline'])
children={};failed={};paused=False
write_json(ROOT/'outputs/u_queue_identity.json',{'pid':os.getpid(),'created':datetime.now(deadline.tzinfo).isoformat(),'maximum_concurrent_U_workers':2,'finite_job_ids':[j['id'] for j in jobs],'not_an_automation':True})
def state(j):
    p=ROOT/'outputs/jobs'/j['id']/'status.json'
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else {'status':'pending'}
while True:
    for key,(p,log) in list(children.items()):
        code=p.poll()
        if code is not None:
            log.close();del children[key]
            if code!=0:
                folder=ROOT/'outputs/jobs'/key
                console=(folder/'worker_console.log').read_text(encoding='utf-8',errors='replace')[-3000:]
                # A second resource check can fail between scheduler dispatch
                # and worker initialization. This is a safe deferral, not a
                # scientific path failure or grounds for discarding its state.
                if console.rstrip().endswith('RuntimeError: RESOURCE_DISPATCH_PAUSED'):
                    paused=True
                    write_json(folder/'dispatch_deferral.json',{'time':datetime.now(deadline.tzinfo).isoformat(),'reason':'resource_changed_between_dispatch_and_worker_check','checkpoint_preserved':True})
                else:
                    failed[key]=code
                    sp=folder/'status.json'
                    if sp.exists():
                        old=json.loads(sp.read_text(encoding='utf-8'))
                        if old.get('status') in ('running','continuing_same_path_zero_ftol'):
                            write_json(folder/'status_before_worker_exception.json',old)
                            old.update(status='worker_exception',exit_code=code)
                            write_json(sp,old)
    statuses={j['id']:state(j) for j in jobs}
    running=[k for k,v in statuses.items() if v['status'] in ('running','continuing_same_path_zero_ftol')]
    pending=[j for j in jobs if statuses[j['id']]['status'] in ('pending','resource_checkpoint') and j['id'] not in failed and j['id'] not in children]
    receipt={'time':datetime.now(deadline.tzinfo).isoformat(),'running':running,'children':list(children),'pending':[j['id'] for j in pending],'isolated_worker_errors':failed,'paused':paused}
    write_json(ROOT/'outputs/u_queue_status.json',receipt)
    if datetime.now(deadline.tzinfo)>=deadline:
        write_json(ROOT/'outputs/u_queue_stop.json',dict(receipt,reason='dispatch_deadline'));break
    if not pending and not running and not children:
        write_json(ROOT/'outputs/u_queue_stop.json',dict(receipt,reason='finite_queue_exhausted'));break
    occupied=set(running)|set(children)
    unfinished_priorities=[j['priority'] for j in jobs if j['id'] in occupied or j in pending]
    first=min(unfinished_priorities) if unfinished_priorities else None
    available=[j for j in pending if j['priority']==first]
    if available and len(occupied)<2:
        j=available[0]
        ok,res=dispatch_allowed(paused=paused or statuses[j['id']]['status']=='resource_checkpoint',reserve_bytes=4_000_000_000)
        with (ROOT/'outputs/u_dispatch_resources.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(receipt,resources=res,allowed=ok,requested_job=j['id']))+'\n')
        if ok:
            paused=False;folder=ROOT/'outputs/jobs'/j['id'];folder.mkdir(parents=True,exist_ok=True)
            log=(folder/'worker_console.log').open('ab')
            p=subprocess.Popen([sys.executable,str(ROOT/'scripts/run_u_job.py'),j['id']],cwd=str(ROOT),stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            children[j['id']]=(p,log)
        else:paused=True
    time.sleep(5)
