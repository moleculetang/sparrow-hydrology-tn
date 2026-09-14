"""Native process observations and separate system RAM/VRAM accounting."""
import ctypes
import sys
import os
from fc_io import memory
from fc_legacy import Process


def process_snapshot(pid,created=None):
    process=Process(pid)
    actual=process.create_time()
    if created is not None and abs(actual-created)>.01:raise RuntimeError('PROCESS_IDENTITY_MISMATCH')
    cpu=process.cpu_times()
    native=sys.modules['process_info']
    def read(handle):
        counters=native.Counters();counters.cb=ctypes.sizeof(counters)
        if not native.psapi.GetProcessMemoryInfo(handle,ctypes.byref(counters),counters.cb):raise ctypes.WinError()
        return {'rss_gib':counters.WorkingSetSize/2**30,'peak_rss_gib':counters.PeakWorkingSetSize/2**30}
    return {'pid':pid,'process_created':actual,'alive':process.is_running(),'cpu_seconds':cpu.user+cpu.system,**process.call(read)}


def vram():
    import torch
    if not torch.cuda.is_available():return None
    free,total=torch.cuda.mem_get_info()
    return {'total_gib':total/2**30,'available_gib':free/2**30,'used_percent':100*(total-free)/total}


_previous_cpu=None


def cpu_usage():
    """Whole-system utilization from Windows cumulative kernel/user/idle time."""
    global _previous_cpu
    from ctypes import wintypes
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.GetSystemTimes.argtypes=[ctypes.POINTER(wintypes.FILETIME)]*3
    kernel.GetSystemTimes.restype=wintypes.BOOL
    times=[wintypes.FILETIME() for _ in range(3)]
    if not kernel.GetSystemTimes(*[ctypes.byref(t) for t in times]):raise ctypes.WinError()
    current=tuple((t.dwHighDateTime<<32)|t.dwLowDateTime for t in times)
    percent=None
    if _previous_cpu is not None:
        idle=current[0]-_previous_cpu[0]
        total=sum(current[1:])-sum(_previous_cpu[1:])
        if total>0:percent=max(0.,min(100.,100*(1-idle/total)))
    _previous_cpu=current
    return {'used_percent':percent,'logical_cpus':os.cpu_count(),'source':'Windows_GetSystemTimes'}


def resources():return {'ram':memory(),'vram':vram(),'cpu':cpu_usage()}


def admission(resources_now,reserve_ram,reserve_vram=0.,threshold=90.):
    cpu=resources_now.get('cpu',{}).get('used_percent')
    if cpu is None or cpu>=90.:return False
    r=resources_now['ram']
    if r['used_percent']>=threshold:return False
    if r['available_gib']-reserve_ram<r['total_gib']*(1-threshold/100):return False
    if reserve_vram:
        v=resources_now['vram']
        if v is None or v['used_percent']>=threshold:return False
        if v['available_gib']-reserve_vram<v['total_gib']*(1-threshold/100):return False
    return True
