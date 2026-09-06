"""Continue nested validation and final experiments after the live stage-4 job."""
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
    run=ROOT/'5_Test/20260905_5';statuspath=run/'reports/later_phase_queue.json'
    state=dict(status='WAITING_ON_PROCESS',pid=os.getpid(),wait_pid=a.wait_pid,runtime=RUNTIME,started_utc=utc_now())
    kernel=ctypes.windll.kernel32;kernel.OpenProcess.argtypes=[ctypes.c_ulong,ctypes.c_int,ctypes.c_ulong];kernel.OpenProcess.restype=ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_ulong];kernel.WaitForSingleObject.restype=ctypes.c_ulong
    kernel.CloseHandle.argtypes=[ctypes.c_void_p]
    h=kernel.OpenProcess(0x100000,False,a.wait_pid)
    if not h:raise ctypes.WinError()
    atomic_json(state,statuspath);print('LATER_WAITING_ON_LIVE_HANDLE',a.wait_pid,flush=True)
    try:
        while True:
            r=kernel.WaitForSingleObject(h,30000)
            if r==0:break
            if r!=258:raise ctypes.WinError()
    finally:kernel.CloseHandle(h)
    parent=json.loads((ROOT/'5_Test/20260905_4/reports/stage4_dependency_queue.json').read_text(encoding='utf-8'))
    if parent['status']!='STRUCTURAL_PHASES_FINISHED':
        state.update(status='PARENT_REQUIRES_DIAGNOSIS',parent_status=parent['status']);atomic_json(state,statuspath)
        print('LATER_DEPENDENCY_FAILED',parent['status'],flush=True);return
    residual=json.loads((ROOT/'5_Test/20260905_4/reports/new_residual_evidence.json').read_text(encoding='utf-8'))
    if residual['results']['conditional_air_temperature']['probe_supported']:
        state.update(status='TEMPERATURE_MECHANISM_REVIEW_REQUIRED');atomic_json(state,statuspath)
        print('LATER_TEMPERATURE_REVIEW_REQUIRED',flush=True);return
    scripts=['20260905_5/scripts/run_nested_validation.py','20260905_6/scripts/run_confirmation_and_final.py','20260905_6/scripts/completion_audit.py']
    for relative in scripts:
        script=ROOT/'5_Test'/relative
        state.update(status='RUNNING',active=relative,active_sha256=sha256(script),updated_utc=utc_now());atomic_json(state,statuspath)
        log=run/'logs'/f'{script.stem}.log';log.parent.mkdir(parents=True,exist_ok=True)
        print('LATER_PHASE_STARTED',relative,flush=True)
        with log.open('a',encoding='utf-8') as stream:
            result=subprocess.run([sys.executable,'-B',str(script)],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:
            state.update(status='REQUIRES_DIAGNOSIS',returncode=result.returncode,updated_utc=utc_now());atomic_json(state,statuspath)
            print('LATER_PHASE_FAILED',relative,result.returncode,flush=True);return
        print('LATER_PHASE_FINISHED',relative,flush=True)
    state.update(status='NUMERICAL_PHASES_FINISHED_REQUIRES_SCIENTIFIC_REVIEW',active=None,updated_utc=utc_now());atomic_json(state,statuspath)
    print('LATER_NUMERICAL_PHASES_COMPLETE_REVIEW_REQUIRED',flush=True)


if __name__=='__main__':main()
