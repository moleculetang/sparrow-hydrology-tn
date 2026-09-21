"""Operational-only recovery: bounded status-read retry and orphan adoption.

The registered fit code/configuration/identity remain byte-identical.
"""
import time,os,ctypes,sys,traceback
from pathlib import Path
import native_runtime as rt
original_read=rt.read
def robust_read(path):
    until=time.monotonic()+30
    while True:
        try:return original_read(path)
        except PermissionError:
            if time.monotonic()>=until:raise
            time.sleep(.1)
rt.read=robust_read
class Adopted:
    def __init__(self,pid,created):
        rt.process(pid,created);self.pid=pid
        self._handle=rt.k32.OpenProcess(0x100000|0x1000,False,pid)
        if not self._handle:raise ctypes.WinError(ctypes.get_last_error())
    def poll(self):
        state=rt.k32.WaitForSingleObject(self._handle,0)
        if state==258:return None
        if state!=0:raise ctypes.WinError(ctypes.get_last_error())
        result=ctypes.c_ulong()
        rt.k32.GetExitCodeProcess.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_ulong)]
        if not rt.k32.GetExitCodeProcess(self._handle,ctypes.byref(result)):raise ctypes.WinError(ctypes.get_last_error())
        return result.value
class NoLog:
    def close(self):pass
def adopt(jobs,proof):
    live={};seen=[]
    for job in jobs:
        p=rt.RUN/'work/jobs'/job['tag']/'status.json'
        if not p.exists():continue
        saved=rt.read(p)
        if saved.get('status')!='RUNNING':continue
        proc=saved.get('process')
        if not proc:raise RuntimeError('RUNNING_WITHOUT_PROCESS '+job['tag'])
        try:actual=rt.process(proc['pid'],proc['created'])
        except OSError as e:
            if getattr(e,'winerror',None)==87:
                # Preserve state and require explicit accounting; never assume a
                # vanished RUNNING process is permission to start a fresh fit.
                saved['status']='FAILED';saved['reason']='ORPHAN_EXIT_WITHOUT_TERMINAL_STATE';rt.write(p,saved);continue
            raise
        if not actual['alive']:continue
        live[job['tag']]=dict(child=Adopted(proc['pid'],proc['created']),log=NoLog(),job=job,audit=False,created=proc['created'],peak=proof['peak_reservations_gib'][job['kind']])
        seen.append(dict(tag=job['tag'],calls=saved.get('calls'),**actual))
    rt.write(rt.RUN/'reports/controller_recovery.json',dict(time=time.time(),cause='Startup reconciliation (may be first launch)',policy='Retry reads for at most 30 seconds; adopt existing PID plus creation time; no fit restart or science changes',adopted=seen,source_sha256=rt.sha(rt.RUN/'scripts/campaign_controller.py')))
    return live

if __name__=='__main__':
    source=(rt.RUN/'scripts/campaign_controller.py').read_text(encoding='utf-8')
    needle='live={};paused=False;rt.resources();lastupdate=0'
    assert source.count(needle)==1
    source=source.replace(needle,'live=adopt(jobs,proof);paused=False;rt.resources();lastupdate=0')
    scope=dict(__name__='controller_recovery_runtime',__file__=str(rt.RUN/'scripts/campaign_controller.py'),adopt=adopt)
    exec(compile(source,str(rt.RUN/'scripts/campaign_controller.py'),'exec'),scope)
    try:scope['main']()
    except Exception:
        rt.write(rt.RUN/'work/recovery_controller_failure.json',dict(error=traceback.format_exc(),time=time.time()));raise
