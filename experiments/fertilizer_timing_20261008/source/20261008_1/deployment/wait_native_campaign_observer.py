"""Hold a verified Windows process handle until one actual event completes."""
import argparse
import ctypes
import json
import os
import sys
from ctypes import wintypes
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main(kind,sequence):
    if os.name!='nt':raise RuntimeError('WINDOWS_NATIVE_OBSERVER_WAIT_ONLY')
    if kind=='local_event':
        if not sequence.isdigit():raise ValueError('NUMERIC_OBSERVER_SEQUENCE_REQUIRED')
        folder=ROOT/'outputs/campaign_next_local_event'/sequence
        marker=folder/'owner.json';status=folder/'status.json'
    else:
        folder=ROOT/'outputs/maize_scope_revision_candidate'
        marker=folder/'history_launcher_owner.json';status=folder/'history_status.json'
    owner=json.loads(marker.read_text(encoding='utf8'))
    sys.path[:0]=[str(ROOT/'scripts'),str(ROOT)]
    from host_resources import registry
    if registry.identity(owner['pid'])!=owner['created']:
        state=json.loads(status.read_text(encoding='utf8'))
        expected_terminal=('next_stage_event_available_requires_review' if kind=='local_event'
                           else 'candidate_history_finished_requires_review')
        if state.get('phase')!=expected_terminal or state['owner']!=owner:
            raise RuntimeError('OBSERVER_NOT_LIVE_OR_TERMINAL_NO_GUESSED_WAIT')
        print(json.dumps(dict(event='already_terminal_requires_review',state=state)));return 0
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
    kernel.OpenProcess.restype=wintypes.HANDLE
    kernel.GetProcessTimes.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),ctypes.POINTER(wintypes.FILETIME),ctypes.POINTER(wintypes.FILETIME)]
    kernel.GetProcessTimes.restype=wintypes.BOOL
    kernel.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD]
    kernel.WaitForSingleObject.restype=wintypes.DWORD
    kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    handle=kernel.OpenProcess(0x100000|0x1000,False,owner['pid'])
    if not handle:raise ctypes.WinError(ctypes.get_last_error())
    try:
        created,exited,kernel_time,user_time=[wintypes.FILETIME() for _ in range(4)]
        if not kernel.GetProcessTimes(handle,ctypes.byref(created),ctypes.byref(exited),
                                      ctypes.byref(kernel_time),ctypes.byref(user_time)):
            raise ctypes.WinError(ctypes.get_last_error())
        actual=str((created.dwHighDateTime<<32)|created.dwLowDateTime)
        if actual!=owner['created']:raise RuntimeError('NATIVE_HANDLE_CREATION_MISMATCH')
        while True:
            result=kernel.WaitForSingleObject(handle,50000)
            if result==0:break
            if result!=258:raise ctypes.WinError(ctypes.get_last_error())
        state=json.loads(status.read_text(encoding='utf8'))
        print(json.dumps(dict(event='exact_native_observer_exit_requires_review',owner=owner,
                             kind=kind,state=state,training_dispatched=False,scheduled_task_created=False)))
        return 0
    finally:kernel.CloseHandle(handle)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['local_event','candidate_history'],required=True)
    p.add_argument('--sequence',default='')
    a=p.parse_args();sys.exit(main(a.kind,a.sequence))
