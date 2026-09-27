"""Finite LAND1 queue for this experiment; no timer or external task."""
import sys,json,time,subprocess,os,ctypes
from pathlib import Path
from datetime import datetime
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from d29_platform.runtime import configure,write_json,dispatch_allowed
configure()
jobs=[j for j in json.loads((ROOT/'config/jobs.json').read_text(encoding='utf-8')) if j['model']=='LAND1']
deadline=datetime.fromisoformat(json.loads((ROOT/'config/clock.json').read_text(encoding='utf-8'))['dispatch_deadline'])
gate_path=ROOT/'outputs/land1_formal_gate.json'
if not gate_path.exists() or not json.loads(gate_path.read_text(encoding='utf-8')).get('passed'):raise RuntimeError('LAND1_FORMAL_GATE_REQUIRED_BEFORE_QUEUE')
children={};failed={};failure_signatures={};paused=False
MAX_WORKERS=6
PEAK_RESERVE_BYTES=9_000_000_000


class _ProcessMemoryCountersEx(ctypes.Structure):
    _fields_=[('cb',ctypes.c_ulong),('page_fault_count',ctypes.c_ulong)] + [
        (name,ctypes.c_size_t) for name in (
            'peak_working_set','working_set','quota_peak_paged_pool',
            'quota_paged_pool','quota_peak_nonpaged_pool','quota_nonpaged_pool',
            'pagefile_usage','peak_pagefile_usage','private_usage')]


def _private_bytes(process):
    """Read this worker's current commit; zero on failure reserves its full peak."""
    counters=_ProcessMemoryCountersEx()
    counters.cb=ctypes.sizeof(counters)
    ok=ctypes.windll.psapi.GetProcessMemoryInfo(
        ctypes.c_void_p(int(process._handle)),ctypes.byref(counters),counters.cb)
    return int(counters.private_usage) if ok else 0


def _projected_reserve():
    # Current commit already contains running workers. Reserve only the gap
    # from each worker's current commit to its measured peak, plus one full
    # peak for the next worker. This prevents a burst of small new processes
    # from all passing the same instantaneous headroom check.
    active={job_id:_private_bytes(process) for job_id,(process,_) in children.items()}
    reserve=PEAK_RESERVE_BYTES+sum(max(0,PEAK_RESERVE_BYTES-used) for used in active.values())
    return reserve,active


write_json(ROOT/'outputs/land1_queue_identity.json',{'pid':os.getpid(),'created':datetime.now(deadline.tzinfo).isoformat(),'maximum_concurrent_LAND1_workers':MAX_WORKERS,'finite_job_ids':[j['id'] for j in jobs],'not_an_automation':True,
    'resource_basis':'Cloud-drive-daemon stop verified; user authorized up to six workers. Dispatch projects each active worker to the measured 7.86 GB peak with a 9 GB allowance; projected physical and commit memory must also remain below the 90/85 percent hysteresis limit.'})
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
                    signature=next((line.strip() for line in reversed(console.splitlines()) if line.strip()),
                                   'EMPTY_WORKER_CONSOLE')
                    failure_signatures.setdefault(signature,[]).append(key)
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
    receipt={'time':datetime.now(deadline.tzinfo).isoformat(),'running':running,'children':list(children),'pending':[j['id'] for j in pending],'isolated_worker_errors':failed,'failure_signatures':failure_signatures,'paused':paused}
    write_json(ROOT/'outputs/land1_queue_status.json',receipt)
    repeated={k:v for k,v in failure_signatures.items() if len(v)>=2}
    if repeated:
        write_json(ROOT/'outputs/land1_queue_stop.json',dict(receipt,reason='repeated_worker_exception_requires_common_review',repeated=repeated))
        raise RuntimeError('REPEATED_LAND1_WORKER_EXCEPTION '+repr(repeated))
    if datetime.now(deadline.tzinfo)>=deadline:
        write_json(ROOT/'outputs/land1_queue_stop.json',dict(receipt,reason='dispatch_deadline'));break
    if not pending and not running and not children:
        write_json(ROOT/'outputs/land1_queue_stop.json',dict(receipt,reason='finite_queue_exhausted'));break
    occupied=set(running)|set(children)
    unfinished_priorities=[j['priority'] for j in jobs if j['id'] in occupied or j in pending]
    first=min(unfinished_priorities) if unfinished_priorities else None
    available=[j for j in pending if j['priority']==first]
    if available and len(occupied)<MAX_WORKERS:
        j=available[0]
        reserve,worker_private=_projected_reserve()
        ok,res=dispatch_allowed(paused=paused or statuses[j['id']]['status']=='resource_checkpoint',reserve_bytes=reserve)
        projected_commit_percent=100*(1-(res['commit_available_bytes']-reserve)/res['commit_total_bytes'])
        projected_physical_percent=100*(1-(res['physical_available_bytes']-reserve)/res['physical_total_bytes'])
        projected_limit=85. if paused else 90.
        ok=ok and max(projected_commit_percent,projected_physical_percent)<projected_limit
        with (ROOT/'outputs/land1_dispatch_resources.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(receipt,resources=res,allowed=ok,requested_job=j['id'],projected_reserve_bytes=reserve,active_worker_private_bytes=worker_private,projected_commit_percent=projected_commit_percent,projected_physical_percent=projected_physical_percent,projected_limit_percent=projected_limit))+'\n')
        if ok:
            paused=False;folder=ROOT/'outputs/jobs'/j['id'];folder.mkdir(parents=True,exist_ok=True)
            log=(folder/'worker_console.log').open('ab')
            p=subprocess.Popen([sys.executable,str(ROOT/'scripts/run_land1_job.py'),j['id']],cwd=str(ROOT),stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            children[j['id']]=(p,log)
        else:paused=True
    time.sleep(5)
