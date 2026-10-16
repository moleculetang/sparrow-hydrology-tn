"""Native process counters; Linux RSS/swap are distinct from Windows commit."""
import ctypes,os
from pathlib import Path
from ctypes import wintypes

class Counters(ctypes.Structure):
    _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(k,ctypes.c_size_t) for k in ['PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage','PrivateUsage']]

def process_memory():
    if os.name=='posix':
        fields={}
        for line in Path('/proc/self/status').read_text().splitlines():
            parts=line.split()
            if len(parts)>2 and parts[2]=='kB':fields[parts[0].rstrip(':')]=int(parts[1])*1024
        private=0
        p=Path('/proc/self/smaps_rollup')
        if p.exists():
            for line in p.read_text().splitlines():
                if line.startswith(('Private_Clean:','Private_Dirty:')):private+=int(line.split()[1])*1024
        return dict(PeakWorkingSetSize=fields.get('VmHWM'),WorkingSetSize=fields.get('VmRSS'),PagefileUsage=fields.get('VmSwap'),PeakPagefileUsage=None,PrivateUsage=private,linux_memory_identity='RSS/HWM, private resident and swap; not Windows committed/private bytes')
    kernel=ctypes.windll.kernel32;psapi=ctypes.windll.psapi
    kernel.GetCurrentProcess.restype=wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
    info=Counters();info.cb=ctypes.sizeof(info)
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(info),info.cb):raise OSError('GetProcessMemoryInfo failed')
    return {k:int(getattr(info,k)) for k in ['PeakWorkingSetSize','WorkingSetSize','PagefileUsage','PeakPagefileUsage','PrivateUsage']}
