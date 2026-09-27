"""Single-thread runtime, Windows resource evidence and atomic receipts."""
from __future__ import annotations
import ctypes
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]

def configure():
    for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[k] = '1'
    os.environ['NUMBA_CACHE_DIR'] = str(ROOT/'cache/numba')
    os.environ['MPLCONFIGDIR'] = str(ROOT/'cache/matplotlib')
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    sys.dont_write_bytecode = True
    (ROOT/'cache/numba').mkdir(parents=True, exist_ok=True)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()

def write_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    deadline=time.monotonic()+30.
    while True:
        try:
            os.replace(tmp,path)
            break
        except PermissionError:
            if time.monotonic()>=deadline:raise
            time.sleep(.15)

class _Memory(ctypes.Structure):
    _fields_=[('dwLength',ctypes.c_ulong),('dwMemoryLoad',ctypes.c_ulong)]+[(n,ctypes.c_ulonglong) for n in ('total_phys','avail_phys','total_page','avail_page','total_virtual','avail_virtual','extended')]

def resources():
    m=_Memory();m.dwLength=ctypes.sizeof(m)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
        raise OSError('GlobalMemoryStatusEx failed')
    return {'physical_used_percent':float(m.dwMemoryLoad),'physical_total_bytes':int(m.total_phys),
            'physical_available_bytes':int(m.avail_phys),'commit_used_percent':100*(1-m.avail_page/m.total_page),
            'commit_available_bytes':int(m.avail_page),'commit_total_bytes':int(m.total_page)}

def cpu_percent(interval=.15):
    def sample():
        vals=[ctypes.c_ulonglong() for _ in range(3)]
        if not ctypes.windll.kernel32.GetSystemTimes(*[ctypes.byref(v) for v in vals]):raise OSError('GetSystemTimes')
        return [v.value for v in vals]
    a=sample();time.sleep(interval);b=sample();d=[y-x for x,y in zip(a,b)];total=d[1]+d[2]
    return 100*(1-d[0]/total) if total else 0.

def dispatch_allowed(paused=False, reserve_bytes=0):
    r=resources();r['cpu_used_percent']=cpu_percent()
    # A full CPU is acceptable for bounded single-thread workers; memory has
    # the 90% stop / below-85% resume hysteresis and the measured peak reserve.
    memory_limit=85. if paused else 90.
    allow=(0.<=r['cpu_used_percent']<=100. and
           all(r[k]<memory_limit for k in ('physical_used_percent','commit_used_percent')))
    allow=allow and min(r['physical_available_bytes'],r['commit_available_bytes'])>=reserve_bytes
    return allow,r

configure()
