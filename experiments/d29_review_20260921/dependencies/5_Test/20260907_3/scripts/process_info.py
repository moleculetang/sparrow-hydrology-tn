"""Small native Windows process reader; no extra Conda dependencies or process enumeration."""
import ctypes,os
from ctypes import wintypes
from types import SimpleNamespace

class NoSuchProcess(ProcessLookupError):pass
class AccessDenied(PermissionError):pass

kernel=ctypes.WinDLL('kernel32',use_last_error=True)
psapi=ctypes.WinDLL('psapi',use_last_error=True)
kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];kernel.OpenProcess.restype=wintypes.HANDLE
kernel.CloseHandle.argtypes=[wintypes.HANDLE]
kernel.GetProcessTimes.argtypes=[wintypes.HANDLE,*([ctypes.POINTER(wintypes.FILETIME)]*4)]
kernel.GetExitCodeProcess.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)]
kernel.QueryFullProcessImageNameW.argtypes=[wintypes.HANDLE,wintypes.DWORD,wintypes.LPWSTR,ctypes.POINTER(wintypes.DWORD)]

class Counters(ctypes.Structure):
    _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(n,ctypes.c_size_t) for n in
        ['PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage']]
psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]

def error(pid):
    code=ctypes.get_last_error()
    if code==5:raise AccessDenied(pid)
    raise NoSuchProcess(pid,code)
def seconds(value):return ((int(value.dwHighDateTime)<<32)|int(value.dwLowDateTime))/1e7

class Process:
    def __init__(self,pid=None):self.pid=os.getpid() if pid is None else int(pid)
    def call(self,fn):
        handle=kernel.OpenProcess(0x0400|0x0010,False,self.pid)
        if not handle:error(self.pid)
        try:return fn(handle)
        finally:kernel.CloseHandle(handle)
    def times(self):
        def read(handle):
            values=[wintypes.FILETIME() for _ in range(4)]
            if not kernel.GetProcessTimes(handle,*[ctypes.byref(v) for v in values]):error(self.pid)
            return list(map(seconds,values))
        return self.call(read)
    def create_time(self):return self.times()[0]-11644473600
    def cpu_times(self):
        values=self.times();return SimpleNamespace(user=values[3],system=values[2])
    def memory_info(self):
        def read(handle):
            out=Counters();out.cb=ctypes.sizeof(out)
            if not psapi.GetProcessMemoryInfo(handle,ctypes.byref(out),out.cb):error(self.pid)
            return SimpleNamespace(rss=out.WorkingSetSize)
        return self.call(read)
    def is_running(self):
        def read(handle):
            code=wintypes.DWORD()
            if not kernel.GetExitCodeProcess(handle,ctypes.byref(code)):error(self.pid)
            return code.value==259
        return self.call(read)
    def cmdline(self):
        def read(handle):
            size=wintypes.DWORD(32768);buf=ctypes.create_unicode_buffer(size.value)
            if not kernel.QueryFullProcessImageNameW(handle,0,buf,ctypes.byref(size)):error(self.pid)
            return [buf.value]
        return self.call(read)
