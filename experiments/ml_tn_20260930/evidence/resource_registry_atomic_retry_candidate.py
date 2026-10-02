"""Host-local transactional leases. stdlib only; never releases a live job.

Controller acquires with its own PID, starts child with reserved affinity, then
binds lease to child. PID creation identity is mandatory. No TTL-based release.
ML priority is enforced through pending-project registration, without preemption.
Registry must be stored on the host executing the jobs, on a local filesystem.
"""
from pathlib import Path
import os,sys,json,time,uuid,ctypes,subprocess,argparse
from contextlib import contextmanager

DEFAULT=Path(__file__).resolve().parent/('state_'+__import__('platform').node())
ML='20260930_1'

def identity(pid):
    try:
        if os.name=='posix':
            s=Path(f'/proc/{pid}/stat').read_text();fields=s[s.rfind(')')+2:].split()
            if fields[0]=='Z':return None
            return Path('/proc/sys/kernel/random/boot_id').read_text().strip()+':'+fields[19]
        k=ctypes.windll.kernel32;k.OpenProcess.restype=ctypes.c_void_p
        h=k.OpenProcess(0x1000,False,int(pid))
        if not h:return None
        try:
            vals=[ctypes.c_ulonglong() for _ in range(4)]
            ok=k.GetProcessTimes(ctypes.c_void_p(h),*[ctypes.byref(v) for v in vals])
            code=ctypes.c_ulong();k.GetExitCodeProcess(ctypes.c_void_p(h),ctypes.byref(code))
            return str(vals[0].value) if ok and code.value==259 else None
        finally:k.CloseHandle(ctypes.c_void_p(h))
    except (FileNotFoundError,ProcessLookupError):return None

def rss(pid):
    try:
        if os.name=='posix':return int(Path(f'/proc/{pid}/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')
        class PM(ctypes.Structure):
            _fields_=[('cb',ctypes.c_ulong),('faults',ctypes.c_ulong)]+[(n,ctypes.c_size_t) for n in ['peak','working','ppq','pq','pnpq','npq','page','peakpage']]
        k=ctypes.windll.kernel32;k.OpenProcess.restype=ctypes.c_void_p;h=k.OpenProcess(0x1000|0x10,False,int(pid));m=PM();m.cb=ctypes.sizeof(m)
        if not h:return 0
        try:return int(m.working) if ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.c_void_p(h),ctypes.byref(m),m.cb) else 0
        finally:k.CloseHandle(ctypes.c_void_p(h))
    except FileNotFoundError:return 0

def physical_cores():
    if os.name=='posix':
        rows=json.loads(subprocess.check_output(['lscpu','--json','--extended=CPU,CORE,SOCKET,ONLINE'],text=True))['cpus'];cores={}
        allowed=os.sched_getaffinity(0)
        for r in rows:
            cpu=int(r['cpu'])
            if cpu in allowed and str(r.get('online','yes')).lower() not in ['no','false']:cores.setdefault((r['socket'],r['core']),cpu)
        return sorted(cores.values())[:40]
    class INFO(ctypes.Structure):_fields_=[('mask',ctypes.c_size_t),('relation',ctypes.c_int),('data',ctypes.c_byte*16)]
    size=ctypes.c_ulong(0);k=ctypes.windll.kernel32;k.GetLogicalProcessorInformation(None,ctypes.byref(size))
    buf=ctypes.create_string_buffer(size.value)
    if not k.GetLogicalProcessorInformation(buf,ctypes.byref(size)):raise OSError('PHYSICAL_TOPOLOGY_UNAVAILABLE')
    a=ctypes.cast(buf,ctypes.POINTER(INFO));result=[]
    for i in range(size.value//ctypes.sizeof(INFO)):
        if a[i].relation==0:
            bits=[j for j in range(ctypes.sizeof(ctypes.c_size_t)*8) if a[i].mask&(1<<j)]
            if bits:result.append(bits[0])
    if not result:raise OSError('EMPTY_PHYSICAL_TOPOLOGY')
    process=ctypes.c_size_t();system=ctypes.c_size_t()
    if not k.GetProcessAffinityMask(ctypes.c_void_p(-1),ctypes.byref(process),ctypes.byref(system)):raise OSError('PROCESS_AFFINITY_PROBE_FAILED')
    result=[i for i in result if process.value&(1<<i)]
    if not result:raise OSError('NO_ALLOWED_PHYSICAL_CORES')
    return sorted(result)[:40]

def memory():
    if os.name=='posix':
        d={s.split(':')[0]:int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if len(s.split())>1 and s.split()[1].isdigit()}
        # Linux Committed_AS measures reservations, not working RAM; report both.
        return dict(total=d['MemTotal'],used=d['MemTotal']-d['MemAvailable'],commit_total=d['CommitLimit'],commit_used=d['Committed_AS'])
    class MEM(ctypes.Structure):_fields_=[('size',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(n,ctypes.c_ulonglong) for n in ['total','available','ct','ca','vt','va','extra']]
    m=MEM();m.size=ctypes.sizeof(m)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):raise OSError('MEMORY_PROBE_FAILED')
    return dict(total=m.total,used=m.total-m.available,commit_total=m.ct,commit_used=m.ct-m.ca)

def cpu_sample():
    if os.name=='posix':
        a=list(map(int,Path('/proc/stat').read_text().splitlines()[0].split()[1:9]));return sum(a),a[3]+a[4]
    v=[ctypes.c_ulonglong() for _ in range(3)]
    if not ctypes.windll.kernel32.GetSystemTimes(*[ctypes.byref(i) for i in v]):raise OSError('CPU_PROBE_FAILED')
    return v[1].value+v[2].value,v[0].value

def affinity(cpus):
    if os.name=='posix':os.sched_setaffinity(0,cpus)
    elif not ctypes.windll.kernel32.SetProcessAffinityMask(ctypes.c_void_p(-1),ctypes.c_size_t(sum(1<<int(i) for i in cpus))):raise OSError('AFFINITY_FAILED')

def gpu_snapshot(uuid):
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used,memory.total','--format=csv,noheader,nounits'],text=True,timeout=5).splitlines()
    devices={parts[0].strip():dict(used=int(parts[1])*1024**2,total=int(parts[2])*1024**2) for parts in (r.split(',') for r in rows)}
    if uuid not in devices:raise ValueError('GPU_UUID_NOT_FOUND')
    resident={}
    for row in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True,timeout=5).splitlines():
        parts=row.split(',')
        if len(parts)==2 and parts[0].strip().isdigit() and parts[1].strip().isdigit():resident[int(parts[0])]=int(parts[1])*1024**2
    return devices[uuid],resident

def publish_registry(temp,target):
    deadline=time.monotonic()+5
    while True:
        try:os.replace(temp,target);return
        except PermissionError as e:
            if os.name!='nt' or getattr(e,'winerror',None) not in [5,32,33] or time.monotonic()>=deadline:raise
            time.sleep(.05)

@contextmanager
def transaction(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    with (root/'registry.lock').open('a+b') as f:
        if os.name=='posix':
            import fcntl;fcntl.flock(f,fcntl.LOCK_EX)
        else:
            import msvcrt
            if f.seek(0,2)==0:f.write(b'0');f.flush()
            f.seek(0)
            while True:
                try:msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1);break
                except OSError:time.sleep(.02)
        try:
            p=root/'registry.json';s=json.loads(p.read_text()) if p.exists() else dict(version=1,leases={},pending={},paused=False,events=[])
            # Reclaim only after process identity checking, never on SSH timeout.
            for token,v in list(s['leases'].items()):
                current=identity(v['pid'])
                if current!=v['created']:
                    s['events'].append(dict(time=time.time(),action='verified_dead_or_reused_pid',token=token,pid=v['pid'],expected=v['created'],actual=current));del s['leases'][token]
            yield s
            temp=p.with_name(p.name+'.'+str(os.getpid())+'.'+uuid.uuid4().hex+'.tmp');temp.write_text(json.dumps(s,indent=2),encoding='utf-8');publish_registry(temp,p)
        finally:
            if os.name=='posix':fcntl.flock(f,fcntl.LOCK_UN)
            else:f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)

def status(root=DEFAULT):
    with transaction(root) as s:return dict(s,cores=physical_cores(),memory=memory())

def pending(project,count,root=DEFAULT):
    with transaction(root) as s:s['pending'][project]=dict(count=int(count),pid=os.getpid(),created=identity(os.getpid()))

def acquire(project,job,threads,peak_bytes,pid=None,gpu=None,root=DEFAULT,adopt_cpus=None):
    pid=os.getpid() if pid is None else int(pid);created=identity(pid)
    if created is None:raise ValueError('LIVE_IDENTIFIED_OWNER_REQUIRED')
    if threads<1 or peak_bytes<=0:raise ValueError('POSITIVE_MEASURED_RESERVATION_REQUIRED')
    a=cpu_sample();time.sleep(.1);b=cpu_sample();usage=100*(1-(b[1]-a[1])/max(1,b[0]-a[0]))
    with transaction(root) as s:
        if any(v['project']==project and v['job']==job for v in s['leases'].values()):return dict(granted=False,reason='JOB_ALREADY_OWNED')
        for p,v in list(s['pending'].items()):
            if identity(v['pid'])!=v['created']:del s['pending'][p]
        if project!=ML and s['pending'].get(ML,{}).get('count',0)>0:return dict(granted=False,reason='ML_PENDING_PRIORITY')
        cores=physical_cores();used={c for v in s['leases'].values() for c in v['cpus']};free=[c for c in cores if c not in used]
        cpus=list(adopt_cpus) if adopt_cpus is not None else free[:threads]
        if len(cpus)!=threads or len(set(cpus))!=threads or any(c not in free for c in cpus):return dict(granted=False,reason='PHYSICAL_CORES_RESERVED')
        if gpu is not None:
            if project!=ML:return dict(granted=False,reason='GPU_ML_EXCLUSIVE_INITIAL_POLICY')
            policy_path=Path(root)/'gpu_policy.json';policy=json.loads(policy_path.read_text()) if policy_path.exists() else {}
            capacity=int(policy.get('max_same_project_jobs',1))
            if not 1<=capacity<=4:raise ValueError('INVALID_GPU_SLOT_CAPACITY')
            owners=[v for v in s['leases'].values() if v.get('gpu')==gpu]
            if len(owners)>=capacity or any(v['project']!=project for v in owners):return dict(granted=False,reason='GPU_RESERVED')
            try:gm,gr=gpu_snapshot(gpu)
            except (ValueError,subprocess.SubprocessError):return dict(granted=False,reason='GPU_MEMORY_PROBE_FAILED')
            # Legacy leases predate this field; conservatively use the existing4GiB peak.
            gpu_reserve=int(4*1024**3*1.35)
            gpu_headroom=sum(max(0,v.get('reserved_gpu_bytes',gpu_reserve)-gr.get(v['pid'],0)) for v in owners)
            gpu_percent=100*gm['used']/gm['total'];s['gpu_paused']=gpu_percent>=90 or (s.get('gpu_paused',False) and gpu_percent>=85)
            if s['gpu_paused'] or gm['used']+gpu_headroom+gpu_reserve>=.9*gm['total']:return dict(granted=False,reason='GPU_MEMORY_HEADROOM',gpu_memory=gm)
        m=memory();physical=100*m['used']/m['total'];commit=100*m['commit_used']/m['commit_total']
        s['paused']=max(physical,commit,usage)>=90 or (s['paused'] and max(physical,commit,usage)>=85)
        # Only headroom of reservations not already resident is added.
        headroom=sum(max(0,v['reserved_bytes']-rss(v['pid'])) for v in s['leases'].values())
        reserve=int(peak_bytes*1.35)
        if adopt_cpus is None and (s['paused'] or m['used']+headroom+reserve>=.9*m['total'] or m['commit_used']+headroom+reserve>=.9*m['commit_total']):return dict(granted=False,reason='RESOURCE_HEADROOM',memory=m,cpu_percent=usage)
        token=uuid.uuid4().hex;v=dict(project=project,job=job,pid=pid,created=created,cpus=cpus,threads=threads,reserved_bytes=reserve,gpu=gpu,acquired=time.time())
        if gpu is not None:v['reserved_gpu_bytes']=gpu_reserve;v['gpu_slots_at_acquisition']=capacity
        s['leases'][token]=v;s['events'].append(dict(time=time.time(),action='acquire',token=token,**v))
        return dict(granted=True,token=token,**v)

def bind(token,pid,root=DEFAULT):
    current=identity(int(pid))
    if current is None:raise ValueError('LIVE_CHILD_REQUIRED')
    with transaction(root) as s:
        v=s['leases'][token];v.update(pid=int(pid),created=current);s['events'].append(dict(time=time.time(),action='bind',token=token,pid=pid,created=current))

def release(token,root=DEFAULT):
    with transaction(root) as s:
        if token not in s['leases']:return dict(released=False,reason='ABSENT')
        v=s['leases'][token]
        if identity(v['pid'])==v['created'] and v['pid']!=os.getpid():return dict(released=False,reason='OWNER_STILL_ALIVE')
        del s['leases'][token];s['events'].append(dict(time=time.time(),action='release',token=token));return dict(released=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['status','pending','acquire','bind','release']);p.add_argument('--root',type=Path,default=DEFAULT);p.add_argument('--project');p.add_argument('--job');p.add_argument('--threads',type=int,default=1);p.add_argument('--peak-bytes',type=int);p.add_argument('--pid',type=int);p.add_argument('--gpu');p.add_argument('--token');p.add_argument('--count',type=int,default=0);a=p.parse_args()
    if a.action=='status':r=status(a.root)
    elif a.action=='pending':r=pending(a.project,a.count,a.root)
    elif a.action=='acquire':r=acquire(a.project,a.job,a.threads,a.peak_bytes,a.pid,a.gpu,a.root)
    elif a.action=='bind':r=bind(a.token,a.pid,a.root)
    else:r=release(a.token,a.root)
    print(json.dumps(r))
