"""Experiment-local provenance, atomic outputs and resource guard."""
from pathlib import Path
import json, hashlib, datetime, sys, time, ctypes
import numpy as np
RUN=Path(__file__).resolve().parents[1]
TEST=RUN.parent
ROOT=TEST.parent

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def require_environment():
    if Path(sys.prefix).name.lower()!='sparrow':raise RuntimeError('conda sparrow is required')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def clean(v):
    if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return [clean(x) for x in v]
    if isinstance(v,np.ndarray):return clean(v.tolist())
    if isinstance(v,np.integer):return int(v)
    if isinstance(v,np.bool_):return bool(v)
    if isinstance(v,(float,np.floating)):return float(v) if np.isfinite(v) else None
    return v
def local(p):
    p=Path(p).resolve()
    if not p.is_relative_to(RUN.resolve()):raise RuntimeError('Write outside campaign: '+str(p))
    p.parent.mkdir(parents=True,exist_ok=True)
    return p
def replace(t,p):
    for attempt in range(10):
        try:t.replace(p);return
        except PermissionError:
            if attempt==9:raise
            time.sleep(min(.05*2**attempt,.5))
def write(p,data):
    p=local(p);t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(clean(data),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');replace(t,p)
def parquet(p,df):
    p=local(p);t=p.with_suffix(p.suffix+'.tmp');df.to_parquet(t,index=False);replace(t,p)
def register(p,role):
    p=Path(p).resolve();r=RUN/'reports/input_registry.json'
    registry=read(r) if r.exists() else dict(created_utc=now(),inputs={})
    h=sha(p);key=str(p)
    if key in registry['inputs']:
        if registry['inputs'][key]['sha256']!=h:raise RuntimeError('Registered input changed: '+key)
    else:
        registry['inputs'][key]=dict(sha256=h,role=role,registered_utc=now());write(r,registry)
    return p
class MemoryStatus(ctypes.Structure):
    _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(n,ctypes.c_ulonglong) for n in ['total','available','page','available_page','virtual','available_virtual','extended']]
def memory():
    v=MemoryStatus();v.length=ctypes.sizeof(v)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(v)):raise ctypes.WinError()
    return dict(total_gib=v.total/2**30,available_gib=v.available/2**30,used_percent=100*(v.total-v.available)/v.total)
def guard(reserve_gib=0.):
    m=memory()
    if m['used_percent']>=73 or m['available_gib']-reserve_gib<m['total_gib']*.25:
        raise MemoryError('RESOURCE_GUARD '+json.dumps(m))
    return m
