"""Continue a live serial queue only after its actual Windows process exits."""
from pathlib import Path
import sys
import argparse
import ctypes
import subprocess
import json
import os
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'5_Test/20260905_1/scripts'))
from common import RUNTIME,atomic_json,utc_now,sha256


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--wait-pid',type=int,required=True);args=parser.parse_args()
    run=ROOT/'5_Test/20260905_3';statuspath=run/'reports/post_development_queue.json'
    status=dict(status='WAITING_ON_PROCESS',pid=os.getpid(),wait_pid=args.wait_pid,runtime=RUNTIME,started_utc=utc_now())
    kernel=ctypes.windll.kernel32
    kernel.OpenProcess.argtypes=[ctypes.c_ulong,ctypes.c_int,ctypes.c_ulong];kernel.OpenProcess.restype=ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_ulong];kernel.WaitForSingleObject.restype=ctypes.c_ulong
    kernel.CloseHandle.argtypes=[ctypes.c_void_p]
    handle=kernel.OpenProcess(0x100000,False,args.wait_pid)
    if not handle:raise ctypes.WinError()
    try:
        atomic_json(status,statuspath)
        print('POST_QUEUE_WAITING_ON_LIVE_HANDLE',args.wait_pid,flush=True)
        while True:
            result=kernel.WaitForSingleObject(handle,30000)
            if result==0:break
            if result!=258:raise ctypes.WinError()
    finally:kernel.CloseHandle(handle)
    previous=json.loads((run/'reports/development_queue.json').read_text(encoding='utf-8'))
    if previous['status']=='RUNNING':raise RuntimeError('Original process exited without a terminal queue record')
    scripts=['20260905_4/scripts/validate_extensions.py','20260905_3/scripts/resolve_development.py','20260905_3/scripts/summarize_development.py']
    for relative in scripts:
        script=ROOT/'5_Test'/relative
        status.update(status='RUNNING',active=relative,active_sha256=sha256(script),updated_utc=utc_now())
        atomic_json(status,statuspath)
        print('POST_QUEUE_STARTED',relative,flush=True)
        with (run/'logs'/f'{script.stem}.log').open('a',encoding='utf-8') as stream:
            done=subprocess.run([sys.executable,'-B',str(script)],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        if done.returncode:
            status.update(status='REQUIRES_DIAGNOSIS',returncode=done.returncode,updated_utc=utc_now())
            atomic_json(status,statuspath)
            print('POST_QUEUE_FAILED',relative,done.returncode,flush=True);return
        print('POST_QUEUE_FINISHED',relative,flush=True)
    status.update(status='DEVELOPMENT_SUMMARIZED',active=None,updated_utc=utc_now());atomic_json(status,statuspath)
    print('POST_QUEUE_COMPLETE',flush=True)


if __name__=='__main__':main()
