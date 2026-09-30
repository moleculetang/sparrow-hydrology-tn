"""Finite experiment continuation: finish current derivative gate, then run registered queue.

This is a foreground child-process controller with the original absolute deadline,
not a scheduled task/service. Failed scientific acceptance stops dispatch.
"""
import sys,json,time,subprocess,os,ctypes
from ctypes import wintypes
from pathlib import Path
from datetime import datetime
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from d29_platform.runtime import configure,write_json,dispatch_allowed
configure()


def process_alive(pid):
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];kernel.OpenProcess.restype=wintypes.HANDLE
    kernel.GetExitCodeProcess.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)]
    kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    handle=kernel.OpenProcess(0x1000,False,pid)
    if not handle:
        error=ctypes.get_last_error()
        if error==87:return False
        raise OSError(error,'Cannot determine derivative-worker state')
    try:
        status=wintypes.DWORD()
        if not kernel.GetExitCodeProcess(handle,ctypes.byref(status)):raise OSError(ctypes.get_last_error())
        return status.value==259
    finally:kernel.CloseHandle(handle)


def main():
    deadline=datetime.fromisoformat(json.loads((ROOT/'config/clock.json').read_text(encoding='utf-8'))['dispatch_deadline'])
    write_json(ROOT/'outputs/land1_continuation_identity.json',dict(pid=os.getpid(),started=datetime.now(deadline.tzinfo).isoformat(),
        scope='complete existing diagnostic checkpoint; gate; original finite LAND1 queue',deadline=deadline.isoformat(),not_scheduled=True))
    child=None;handle=None;paused=False
    try:
        while datetime.now(deadline.tzinfo)<deadline:
            if (ROOT/'work/stop_land1_continuation.flag').exists():return
            if child is not None:
                code=child.poll()
                if code is None:time.sleep(10);continue
                handle.close();handle=None;child=None
                if code:raise RuntimeError('DERIVATIVE_WORKER_EXCEPTION')
            path=ROOT/'outputs/land1_full_history_gradient_v3.json'
            g=json.loads(path.read_text(encoding='utf-8'))
            status=g['status']
            if status in ('passed','requires_branch_review'):
                subprocess.run([sys.executable,str(ROOT/'scripts/build_land1_gate.py')],cwd=ROOT,check=True)
                gate=json.loads((ROOT/'outputs/land1_formal_gate.json').read_text(encoding='utf-8'))
                if not gate['passed']:
                    write_json(ROOT/'outputs/land1_continuation_blocked.json',dict(status='acceptance_review_required',
                        blocked_by=gate['blocked_by'],no_fits_dispatched=True,original_deadline_retained=True))
                    # Keep U sealed and preserve the opportunity for a reviewed
                    # repair/refinement within the registered preparation time.
                    time.sleep(30);continue
                with (ROOT/'outputs/land1_queue_console.log').open('ab') as log:
                    subprocess.run([sys.executable,str(ROOT/'scripts/run_land1_queue.py')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
                write_json(ROOT/'outputs/land1_continuation_finished.json',dict(status='finite_training_queue_returned',
                    finished=datetime.now(deadline.tzinfo).isoformat(),not_equivalent_to_evaluation_complete=True))
                with (ROOT/'outputs/finalization_controller_console.log').open('ab') as log:
                    subprocess.run([sys.executable,str(ROOT/'scripts/finalize_completed_campaign.py')],cwd=ROOT,
                        stdout=log,stderr=subprocess.STDOUT,check=True)
                return
            identity=json.loads((ROOT/'outputs/land1_precision_gradient_worker.json').read_text(encoding='utf-8'))
            if process_alive(identity['pid']):time.sleep(10);continue
            if status!='resource_pause':raise RuntimeError('DERIVATIVE_GATE_NEEDS_REVIEW '+status)
            ok,res=dispatch_allowed(paused=paused or status=='resource_pause',reserve_bytes=9_000_000_000)
            write_json(ROOT/'outputs/land1_continuation_state.json',dict(status='resume_resource_check',allowed=ok,resources=res,
                checked_coordinates=len(g['coordinates']),calls=g['calls']))
            if not ok:paused=True;time.sleep(10);continue
            paused=False;handle=(ROOT/'outputs/land1_precision_gradient.log').open('ab')
            child=subprocess.Popen([sys.executable,str(ROOT/'scripts/verify_land1_precision_gradient.py')],cwd=ROOT,
                stdout=handle,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            time.sleep(10)
        write_json(ROOT/'outputs/land1_continuation_finished.json',dict(status='dispatch_deadline',checkpoints_preserved=True))
        # No LAND1 queue started if execution reached this branch. Preserve
        # the missing paths explicitly instead of evaluating U prematurely.
        identity=json.loads((ROOT/'outputs/land1_precision_gradient_worker.json').read_text(encoding='utf-8'))
        if process_alive(identity['pid']):
            (ROOT/'work/stop_land1_gradient_v3.flag').write_text('Original dispatch deadline: save current diagnostic boundary.',encoding='utf-8')
            final_deadline=datetime.fromisoformat(json.loads((ROOT/'config/clock.json').read_text(encoding='utf-8'))['deadline'])
            while process_alive(identity['pid']):
                if datetime.now(final_deadline.tzinfo)>=final_deadline:raise RuntimeError('DIAGNOSTIC_STILL_LIVE_AT_FINAL_DEADLINE')
                time.sleep(10)
        stop=ROOT/'outputs/land1_queue_stop.json'
        if not stop.exists():
            write_json(stop,dict(reason='original_dispatch_deadline_before_implementation_acceptance',
                no_formal_land1_fit_dispatched=True,missing_paths_preserved=True))
        with (ROOT/'outputs/finalization_controller_console.log').open('ab') as log:
            subprocess.run([sys.executable,str(ROOT/'scripts/finalize_completed_campaign.py')],cwd=ROOT,
                stdout=log,stderr=subprocess.STDOUT,check=True)
    except Exception as e:
        write_json(ROOT/'outputs/land1_continuation_error.json',dict(error=repr(e),no_gate_override=True))
        raise
    finally:
        if handle is not None:handle.close()


if __name__=='__main__':main()
