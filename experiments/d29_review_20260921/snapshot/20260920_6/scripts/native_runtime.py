"""Windows process identity, bounded resource admission and immutable checkpoints."""
import ctypes,os,time,json,hashlib,pickle,uuid,contextlib,sys
from ctypes import wintypes as w
from pathlib import Path
RUN=Path(__file__).resolve().parents[1]
k32=ctypes.WinDLL('kernel32',use_last_error=True)
class Memory(ctypes.Structure):
    _fields_=[('length',w.DWORD),('load',w.DWORD)]+[(n,ctypes.c_ulonglong) for n in ['total','available','page_total','page_available','virtual_total','virtual_available','extended']]
class Counters(ctypes.Structure):
    _fields_=[('cb',w.DWORD),('faults',w.DWORD)]+[(n,ctypes.c_size_t) for n in ['peak','rss','pool_peak','pool','nonpool_peak','nonpool','page','page_peak','private']]
k32.OpenProcess.argtypes=[w.DWORD,w.BOOL,w.DWORD];k32.OpenProcess.restype=w.HANDLE
k32.CloseHandle.argtypes=[w.HANDLE];k32.CloseHandle.restype=w.BOOL
k32.GetProcessTimes.argtypes=[w.HANDLE]+[ctypes.POINTER(w.FILETIME)]*4;k32.GetProcessTimes.restype=w.BOOL
k32.WaitForSingleObject.argtypes=[w.HANDLE,w.DWORD];k32.WaitForSingleObject.restype=w.DWORD
k32.GetSystemTimes.argtypes=[ctypes.POINTER(w.FILETIME)]*3;k32.GetSystemTimes.restype=w.BOOL
k32.GlobalMemoryStatusEx.argtypes=[ctypes.POINTER(Memory)];k32.GlobalMemoryStatusEx.restype=w.BOOL
psapi=ctypes.WinDLL('psapi',use_last_error=True)
psapi.GetProcessMemoryInfo.argtypes=[w.HANDLE,ctypes.POINTER(Counters),w.DWORD];psapi.GetProcessMemoryInfo.restype=w.BOOL
_previous=None

def log(name, record):
    """Durable, cross-process serialized append; timestamps are never reconstructed."""
    import msvcrt
    gate_path=local(RUN/'work/append.lock');gate_path.parent.mkdir(parents=True,exist_ok=True)
    with gate_path.open('a+b') as gate:
        if gate.seek(0,2)==0:gate.write(b'0');gate.flush()
        gate.seek(0)
        while True:
            try:msvcrt.locking(gate.fileno(),msvcrt.LK_NBLCK,1);break
            except OSError:time.sleep(.01)
        try:
            path=local(RUN/'work'/name)
            with path.open('a',encoding='utf-8') as f:
                f.write(json.dumps(dict(time=time.time(),**record),allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
        finally:gate.seek(0);msvcrt.locking(gate.fileno(),msvcrt.LK_UNLCK,1)
def ticks(v):return (v.dwHighDateTime<<32)|v.dwLowDateTime
def resources():
    global _previous
    m=Memory();m.length=ctypes.sizeof(m)
    if not k32.GlobalMemoryStatusEx(ctypes.byref(m)):raise ctypes.WinError(ctypes.get_last_error())
    ts=[w.FILETIME() for _ in range(3)]
    if not k32.GetSystemTimes(*map(ctypes.byref,ts)):raise ctypes.WinError(ctypes.get_last_error())
    now=[ticks(t) for t in ts];cpu=None
    if _previous:
        total=sum(now[1:])-sum(_previous[1:])
        if total>0:cpu=100*(1-(now[0]-_previous[0])/total)
    _previous=now
    return dict(cpu_percent=cpu,ram_percent=100*(1-m.available/m.total),total_gib=m.total/2**30,available_gib=m.available/2**30,logical_cpus=os.cpu_count())

def process(pid,created=None):
    handle=k32.OpenProcess(0x1000|0x10|0x100000,False,int(pid))
    if not handle:raise ctypes.WinError(ctypes.get_last_error())
    try:
        ts=[w.FILETIME() for _ in range(4)]
        if not k32.GetProcessTimes(handle,*map(ctypes.byref,ts)):raise ctypes.WinError(ctypes.get_last_error())
        birth=ticks(ts[0])
        if created is not None and birth!=created:raise RuntimeError('PID_CREATION_MISMATCH')
        c=Counters();c.cb=ctypes.sizeof(c)
        if not psapi.GetProcessMemoryInfo(handle,ctypes.byref(c),c.cb):raise ctypes.WinError(ctypes.get_last_error())
        return dict(pid=int(pid),created=birth,cpu_seconds=(ticks(ts[2])+ticks(ts[3]))/1e7,rss_gib=c.rss/2**30,peak_gib=c.peak/2**30,alive=k32.WaitForSingleObject(handle,0)==258)
    finally:k32.CloseHandle(handle)

def admission(now,peak,running):
    reserve=1.2*peak+sum(max(0,1.2*p['peak_gib']-p['rss_gib']) for p in running)
    return now['cpu_percent'] is not None and now['cpu_percent']<90 and now['ram_percent']<90 and now['available_gib']-reserve>now['total_gib']*.1

def local(path):
    p=Path(path).resolve()
    if not p.is_relative_to(RUN):raise PermissionError('OUTSIDE_EXPERIMENT_WRITE '+str(p))
    return p
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()
def write(path,obj):
    path=local(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')
    until=time.monotonic()+30
    while True:
        try:os.replace(tmp,path);return
        except PermissionError:
            if time.monotonic()>=until:raise
            time.sleep(.1)
def read(path):
    deadline=time.monotonic()+30
    while True:
        try:return json.loads(Path(path).read_text(encoding='utf-8'))
        except PermissionError:
            if time.monotonic()>=deadline:raise
            time.sleep(.1)
def checkpoint(directory,state):
    directory=local(directory);directory.mkdir(parents=True,exist_ok=True)
    old=read(directory/'latest.json') if (directory/'latest.json').exists() else {}
    name=uuid.uuid4().hex+'.pkl';payload=directory/name
    with payload.open('xb') as f:pickle.dump(state,f,protocol=5);f.flush();os.fsync(f.fileno())
    write(directory/'latest.json',dict(payload=name,sha256=sha(payload),identity=state['identity'],previous=old.get('payload')))
    # Only a previously committed superseded generation is removed. Latest and
    # its immediate predecessor remain immutable and recoverable.
    if old.get('previous'):
        stale=local(directory/old['previous'])
        if stale.parent!=directory:raise RuntimeError('BAD_GENERATION_PATH')
        try:stale.unlink(missing_ok=True)
        except PermissionError:pass
    return name
def restore(directory,identity):
    directory=local(directory);ref=read(directory/'latest.json');p=local(directory/ref['payload'])
    if p.parent!=directory or sha(p)!=ref['sha256']:raise RuntimeError('BAD_CHECKPOINT_HASH')
    if ref['identity']!=identity:raise RuntimeError('BAD_CHECKPOINT_IDENTITY')
    with p.open('rb') as f:state=pickle.load(f)
    if state['identity']!=identity:raise RuntimeError('BAD_PAYLOAD_IDENTITY')
    return state
@contextlib.contextmanager
def exclusive(name):
    import msvcrt
    p=local(RUN/'work/locks'/f'{name}.lock');p.parent.mkdir(parents=True,exist_ok=True)
    stream=p.open('a+b');stream.seek(0);stream.write(b'0');stream.flush();stream.seek(0)
    try:msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
    except OSError:stream.close();raise RuntimeError('DUPLICATE_LIVE_PROCESS '+name)
    try:yield
    finally:stream.seek(0);msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1);stream.close()

def label_barrier(fold):
    import pandas as pd
    allowed=(RUN/'data/folds'/fold/'train.parquet').resolve()
    def check(p):
        if not isinstance(p,(str,bytes,os.PathLike)):return
        q=Path(os.fsdecode(p)).resolve();s=str(q).replace('\\','/').lower()
        foreign_fold='/data/folds/' in s and not q.is_relative_to(allowed.parent)
        if foreign_fold or '/data/cohorts/' in s or 'heldout_labels' in s or ('train.parquet' in s and q!=allowed) or '/outputs/observations.parquet' in s or '/evidence/' in s:
            raise PermissionError('TRAINING_LABEL_BARRIER '+str(q))
    def hook(event,args):
        if event=='open':check(args[0])
    sys.addaudithook(hook);original=pd.read_parquet
    def guarded(path,*args,**kw):check(path);return original(path,*args,**kw)
    pd.read_parquet=guarded

def continuation(trace,used=False):
    n=min(50,len(trace)//2)
    if n<5:return dict(continue_training=not used,diagnostic=not used,reason='INSUFFICIENT_TRACE')
    a,b=trace[-2*n:-n],trace[-n:]
    ja=min(v['objective'] for v in a);jb=min(v['objective'] for v in b)
    ga=[v['pg'] for v in a if v.get('pg') is not None];gb=[v['pg'] for v in b if v.get('pg') is not None]
    gain=(ja-jb)/max(abs(ja),1e-30);drop=(min(ga)-min(gb))/max(min(ga),1e-30) if ga and gb else None
    return dict(continue_training=gain>=1e-6 or (drop is not None and drop>=.2),diagnostic=False,relative_gain=gain,pg_drop=drop,window=n)
