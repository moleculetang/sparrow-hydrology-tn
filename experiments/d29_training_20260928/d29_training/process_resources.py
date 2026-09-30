"""Windows process peak working-set and commit counters; no psutil dependency."""
import ctypes
from ctypes import wintypes

class Counters(ctypes.Structure):
    _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(k,ctypes.c_size_t) for k in ['PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage','PrivateUsage']]

def process_memory():
    kernel=ctypes.windll.kernel32;psapi=ctypes.windll.psapi
    kernel.GetCurrentProcess.restype=wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
    info=Counters();info.cb=ctypes.sizeof(info)
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(info),info.cb):raise OSError('GetProcessMemoryInfo failed')
    return {k:int(getattr(info,k)) for k in ['PeakWorkingSetSize','WorkingSetSize','PagefileUsage','PeakPagefileUsage','PrivateUsage']}
