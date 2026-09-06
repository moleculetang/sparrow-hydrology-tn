"""Run residual discovery and the registered structure queue serially."""
from pathlib import Path
import sys
import ctypes
import subprocess
import argparse
import json
import os
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'5_Test/20260905_1/scripts'))
from common import RUNTIME,atomic_json,utc_now,sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--wait-pid',type=int,required=True);a=p.parse_args()
    run=ROOT/'5_Test/20260905_4';statuspath=run/'reports/stage4_dependency_queue.json'
    state=dict(status='WAITING_ON_PROCESS',pid=os.getpid(),wait_pid=a.wait_pid,runtime=RUNTIME,started_utc=utc_now())
    kernel=ctypes.windll.kernel32;kernel.OpenProcess.argtypes=[ctypes.c_ulong,ctypes.c_int,ctypes.c_ulong];kernel.OpenProcess.restype=ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_ulong];kernel.WaitForSingleObject.restype=ctypes.c_ulong
    kernel.CloseHandle.argtypes=[ctypes.c_void_p]
    h=kernel.OpenProcess(0x100000,False,a.wait_pid)
    if not h:raise ctypes.WinError()
    atomic_json(state,statuspath);print('STAGE4_WAITING_ON_LIVE_HANDLE',a.wait_pid,flush=True)
    try:
        while True:
            r=kernel.WaitForSingleObject(h,30000)
            if r==0:break
            if r!=258:raise ctypes.WinError()
    finally:kernel.CloseHandle(h)
    parent=json.loads((ROOT/'5_Test/20260905_3/reports/post_development_queue.json').read_text(encoding='utf-8'))
    if parent['status']!='DEVELOPMENT_SUMMARIZED':
        state.update(status='PARENT_REQUIRES_DIAGNOSIS',parent_status=parent['status']);atomic_json(state,statuspath)
        print('STAGE4_DEPENDENCY_FAILED',parent['status'],flush=True);return
    for name in ['diagnose_oof_residuals.py','run_structural_trials.py']:
        script=Path(__file__).with_name(name)
        state.update(status='RUNNING',active=name,active_sha256=sha256(script),updated_utc=utc_now());atomic_json(state,statuspath)
        log=run/'logs'/f'{script.stem}.log';log.parent.mkdir(parents=True,exist_ok=True)
        print('STAGE4_PHASE_STARTED',name,flush=True)
        with log.open('a',encoding='utf-8') as stream:
            result=subprocess.run([sys.executable,'-B',str(script)],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:
            state.update(status='REQUIRES_DIAGNOSIS',returncode=result.returncode,updated_utc=utc_now());atomic_json(state,statuspath)
            print('STAGE4_PHASE_FAILED',name,result.returncode,flush=True);return
        print('STAGE4_PHASE_FINISHED',name,flush=True)
    state.update(status='STRUCTURAL_PHASES_FINISHED',active=None,updated_utc=utc_now());atomic_json(state,statuspath)
    print('STAGE4_DEPENDENCY_COMPLETE',flush=True)


if __name__=='__main__':main()
