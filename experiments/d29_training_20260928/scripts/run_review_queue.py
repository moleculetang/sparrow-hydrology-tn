"""One finite queue for both models, six-worker ceiling and staggered starts."""
import ctypes,json,os,subprocess,sys,time
from datetime import datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from d29_platform.runtime import configure,dispatch_allowed,write_json
from d29_training.experiment_context import ExperimentContext
configure()

context=ExperimentContext.load()
gate_path=ROOT/'outputs/land1_formal_gate.json'
if not gate_path.exists():raise RuntimeError('FORMAL_REVIEW_GATE_MISSING')
gate=json.loads(gate_path.read_text(encoding='utf-8'))
if not gate.get('passed') or gate.get('experiment_id')!=context.experiment_id:
    raise RuntimeError('FORMAL_REVIEW_GATE_NOT_PASSED')
from d29_platform.runtime import sha
if any(sha(path)!=digest for path,digest in gate['identities'].items()):
    raise RuntimeError('FORMAL_REVIEW_GATE_IMPLEMENTATION_CHANGED')
jobs=json.loads((ROOT/'config/jobs.json').read_text(encoding='utf-8'))
clock=json.loads((ROOT/'config/clock.json').read_text(encoding='utf-8'))
deadline=datetime.fromisoformat(clock['dispatch_deadline'])
peak={'U':4_000_000_000,'LAND1':9_000_000_000}
MAX_WORKERS=6
MIN_START_GAP_SECONDS=90
children={};failed={};signatures={};paused=False;last_start=0.

class MemoryCounters(ctypes.Structure):
    _fields_=[('cb',ctypes.c_ulong),('page_fault_count',ctypes.c_ulong)]+[(n,ctypes.c_size_t) for n in
        ('peak_working_set','working_set','quota_peak_paged_pool','quota_paged_pool',
         'quota_peak_nonpaged_pool','quota_nonpaged_pool','pagefile_usage','peak_pagefile_usage','private_usage')]

def private_bytes(process):
    counters=MemoryCounters();counters.cb=ctypes.sizeof(counters)
    ok=ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.c_void_p(int(process._handle)),ctypes.byref(counters),counters.cb)
    return int(counters.private_usage) if ok else 0

def status(job):
    path=context.folder(job['id'])/'status.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'status':'pending'}

def projected(job):
    current={key:private_bytes(process) for key,(process,_,_) in children.items()}
    reserve=peak[job['model']]+sum(max(0,peak[old['model']]-current[key]) for key,(_,_,old) in children.items())
    ok,resources=dispatch_allowed(paused=paused,reserve_bytes=reserve)
    limit=85. if paused else 90.
    physical=100*(1-(resources['physical_available_bytes']-reserve)/resources['physical_total_bytes'])
    commit=100*(1-(resources['commit_available_bytes']-reserve)/resources['commit_total_bytes'])
    return bool(ok and max(physical,commit)<limit),dict(resources=resources,private=current,reserve_bytes=reserve,
        projected_physical_percent=physical,projected_commit_percent=commit,limit_percent=limit)

write_json(ROOT/'outputs/review_queue_identity.json',{'pid':os.getpid(),'started':datetime.now(deadline.tzinfo).isoformat(),
    'experiment_id':context.experiment_id,'maximum_concurrent_workers':MAX_WORKERS,
    'minimum_start_gap_seconds':MIN_START_GAP_SECONDS,'peak_reserve_bytes':peak,
    'finite_jobs':[j['id'] for j in jobs],'no_automation':True})

while True:
    for key,(process,log,job) in list(children.items()):
        code=process.poll()
        if code is None:continue
        log.close();del children[key]
        if code!=0:
            folder=context.folder(key);console_path=folder/'worker_console.log'
            tail=console_path.read_text(encoding='utf-8',errors='replace')[-3000:] if console_path.exists() else ''
            signature=next((line.strip() for line in reversed(tail.splitlines()) if line.strip()),'EMPTY_WORKER_CONSOLE')
            if signature.endswith('RuntimeError: RESOURCE_DISPATCH_PAUSED'):
                paused=True
                write_json(folder/'dispatch_deferral.json',{'reason':'resource_changed_after_dispatch','checkpoint_preserved':True})
            else:
                failed[key]=code;signatures.setdefault(signature,[]).append(key)
                sp=folder/'status.json'
                if sp.exists():
                    old=json.loads(sp.read_text(encoding='utf-8'))
                    if old.get('status') in ('running','continuing_same_path_zero_ftol'):
                        write_json(folder/'status_before_worker_exception.json',old)
                        old.update(status='worker_exception',exit_code=code);write_json(sp,old)
    states={j['id']:status(j) for j in jobs}
    pending=[j for j in jobs if states[j['id']]['status'] in ('pending','resource_checkpoint') and j['id'] not in failed and j['id'] not in children]
    active=list(children)
    receipt={'time':datetime.now(deadline.tzinfo).isoformat(),'active':active,'pending':[j['id'] for j in pending],
             'completed':sum(states[j['id']]['status'] not in ('pending','running','resource_checkpoint','continuing_same_path_zero_ftol') for j in jobs),
             'failed':failed,'failure_signatures':signatures,'paused':paused}
    write_json(ROOT/'outputs/review_queue_status.json',receipt)
    repeated={k:v for k,v in signatures.items() if len(v)>=2}
    if repeated:
        write_json(ROOT/'outputs/review_queue_stop.json',dict(receipt,reason='repeated_common_worker_failure',repeated=repeated))
        raise RuntimeError('REPEATED_COMMON_WORKER_FAILURE '+repr(repeated))
    if not pending and not children:
        write_json(ROOT/'outputs/review_queue_stop.json',dict(receipt,reason='finite_queue_exhausted'));break
    if datetime.now(deadline.tzinfo)>=deadline:
        if children:
            time.sleep(15);continue
        write_json(ROOT/'outputs/review_queue_stop.json',dict(receipt,reason='dispatch_deadline',unfinished=[j['id'] for j in pending]));break
    occupied=set(active)
    first=min(j['priority'] for j in jobs if j['id'] in occupied or j in pending)
    choices=[j for j in pending if j['priority']==first]
    if choices and len(children)<MAX_WORKERS and time.monotonic()-last_start>=MIN_START_GAP_SECONDS:
        for j in choices:
            okay,projection=projected(j)
            with (ROOT/'outputs/review_dispatch_resources.jsonl').open('a',encoding='utf-8') as log:
                log.write(json.dumps(dict(time=receipt['time'],job=j['id'],allowed=okay,**projection))+'\n')
            if not okay:continue
            paused=False;folder=context.folder(j['id']);folder.mkdir(parents=True,exist_ok=True)
            output=(folder/'worker_console.log').open('ab')
            script='run_u_job.py' if j['model']=='U' else 'run_land1_job.py'
            process=subprocess.Popen([sys.executable,str(ROOT/'scripts'/script),j['id']],cwd=ROOT,
                stdout=output,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            children[j['id']]=(process,output,j);last_start=time.monotonic()
            break
        else:paused=True
    time.sleep(15)
