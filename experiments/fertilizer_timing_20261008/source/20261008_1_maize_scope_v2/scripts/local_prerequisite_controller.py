"""One local serial prerequisite queue; never trains or accesses SSH.

Wait for the already running exploratory oracle using its native creation
identity, then run current-code full audits under the shared resource registry.
No timer jobs are created; the controller has one lifetime and a durable state.
"""
import os,sys,json,time,subprocess,datetime,platform
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT.parent/'compute_coordination')]
from d29_platform.runtime import write_json,sha
import resource_registry as registry

def main(existing_pid):
    folder=ROOT/'outputs/local_prerequisites';folder.mkdir(parents=True,exist_ok=True)
    marker=folder/'owner.json'
    if marker.exists():
        owner=json.loads(marker.read_text())
        if registry.identity(owner['pid'])==owner['created']:raise RuntimeError('LOCAL_PREREQUISITE_CONTROLLER_ALREADY_LIVE')
    owner=dict(host=platform.node(),pid=os.getpid(),created=registry.identity(os.getpid()),project='20261008_1')
    write_json(marker,owner)
    created=registry.identity(existing_pid);state=dict(owner=owner,workstation_hold=True,scientific_fits_dispatched=0,
       phase='waiting_existing_local_probe',existing_pid=existing_pid,existing_created=created,
       jobs=['TIM_F23_C0_V0_s0','TIM_F23_K_V1_s0'],completed=[])
    write_json(folder/'status.json',state)
    while created is not None and registry.identity(existing_pid)==created:
        if (folder/'pause_requested.json').exists():state['phase']='safely_paused_before_new_check';write_json(folder/'status.json',state);return
        time.sleep(30)
    # Use the measured4.44GB private peak, preserving the original1.35 reserve.
    # This is one worker, one CPU thread and no parallel optimization paths.
    for jid in state['jobs']:
        if (folder/'pause_requested.json').exists():state['phase']='safely_paused_before_new_check';write_json(folder/'status.json',state);return
        gate=ROOT/'outputs/jobs'/jid/'prerequisite_audit.json'
        if gate.exists():
            state['completed'].append(dict(job=jid,already_recorded=True));continue
        request=None
        while request is None or not request.get('granted'):
            request=registry.acquire('20261008_1','prerequisite/'+jid,threads=1,peak_bytes=4500000000)
            state.update(phase='awaiting_local_resources',job=jid,resource_receipt=request);write_json(folder/'status.json',state)
            if request.get('granted'):break
            if (folder/'pause_requested.json').exists():return
            time.sleep(30)
        env=os.environ.copy();env['D29_WORKER_THREADS']='1'
        for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[name]='1'
        log=folder/(jid+'.log')
        try:
            with log.open('ab') as out:
                child=subprocess.Popen([sys.executable,'-B',str(ROOT/'scripts/prerequisite_audit.py'),jid],cwd=ROOT,env=env,stdout=out,stderr=subprocess.STDOUT)
                registry.bind(request['token'],child.pid)
                # Native affinity is scoped to this admitted child.
                if os.name=='nt':
                    import ctypes
                    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
                    kernel.OpenProcess.restype=ctypes.c_void_p
                    handle=kernel.OpenProcess(0x0200|0x0400,False,child.pid)
                    if not handle:raise ctypes.WinError(ctypes.get_last_error())
                    try:
                        mask=sum(1<<c for c in request['cpus'])
                        kernel.SetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
                        if not kernel.SetProcessAffinityMask(handle,mask):raise ctypes.WinError(ctypes.get_last_error())
                    finally:
                        kernel.CloseHandle.argtypes=[ctypes.c_void_p];kernel.CloseHandle(handle)
                else:os.sched_setaffinity(child.pid,set(request['cpus']))
                state.update(phase='local_full_history_prerequisite_audit',job=jid,child_pid=child.pid,child_created=registry.identity(child.pid),lease=request)
                write_json(folder/'status.json',state);code=child.wait()
        finally:
            # Never release a running child if the controller loses a connection.
            if 'child' not in locals() or child.poll() is not None:registry.release(request['token'])
        state['completed'].append(dict(job=jid,exit_code=code))
        write_json(folder/'status.json',state)
        if code!=0:
            state['phase']='prerequisite_failure_requires_review';write_json(folder/'status.json',state);return
        receipt=json.loads(gate.read_text())
        if not receipt['derivative_passed'] or not receipt['physical_passed']:
            state['phase']='prerequisite_gate_unresolved_requires_review';write_json(folder/'status.json',state);return
    state.update(phase='local_prerequisite_audits_finished_review_pending',formal_gate_issued=False,
        remote_dispatch=False,train_dispatch=False)
    write_json(folder/'status.json',state)

if __name__=='__main__':main(int(sys.argv[1]))
