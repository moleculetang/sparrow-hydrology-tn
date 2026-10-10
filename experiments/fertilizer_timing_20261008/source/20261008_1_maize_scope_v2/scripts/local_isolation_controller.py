"""One-shot local full prediction isolation under the shared native registry.

Independent of the existing coordinate audit; no fitting or remote access.
Waits for resources, binds child before imports, preserves live leases on loss.
"""
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT.parent/'compute_coordination')]
import resource_registry as registry
from d29_platform.runtime import write_json


def main(test='isolation'):
    specifications={'isolation':('tests/test_full_prediction_isolation.py',5000000000,'local_isolation',[]),
                    'compatibility':('tests/test_full_p0_compatibility.py',9000000000,'local_compatibility',[]),
                    'initial_feedback':('tests/test_full_initial_feedback.py',5000000000,'local_initial_feedback',[]),
                    'diagnostic_K':('scripts/diagnostic_oracle.py',6500000000,'local_diagnostic_K',
                                    ['TIM_F23_K_V1_s0','--cycles','2','--ledger']),
                    'diagnostic_C0':('scripts/diagnostic_oracle.py',6500000000,'local_diagnostic_C0',
                                     ['TIM_F23_C0_V0_s0','--cycles','2']),
                    'uptake_support':('tests/test_full_uptake_support.py',6500000000,'local_uptake_support',[]),
                    'supported_initial':('tests/test_supported_zero_initial_feedback.py',5000000000,'local_supported_initial',[])}
    if test not in specifications:raise ValueError('REGISTERED_LOCAL_PREREQUISITE_REQUIRED')
    script,peak,namespace,arguments=specifications[test]
    folder = ROOT/'outputs'/namespace
    folder.mkdir(parents=True, exist_ok=True)
    status = folder/'status.json'
    if status.exists():
        previous = json.loads(status.read_text(encoding='utf8'))
        for identity in (previous.get('owner'), previous.get('child')):
            if identity and registry.identity(identity['pid']) == identity['created']:
                raise RuntimeError('EXISTING_ISOLATION_OWNER_OR_CHILD_LIVE')
    owner = dict(pid=os.getpid(), created=registry.identity(os.getpid()), host=platform.node())
    state = dict(owner=owner, phase='resource_wait', scientific_fits_dispatched=0,
                 test=test, workstation_used=False)
    while True:
        if (folder/'pause_requested.json').exists():
            state['phase']='paused_before_dispatch'; write_json(status,state); return
        lease=registry.acquire('20261008_1','prerequisite/'+test,threads=1,peak_bytes=peak)
        state['lease']=lease;write_json(status,state)
        if lease.get('granted'):break
        time.sleep(30)
    env=os.environ.copy();env['PYTHONUTF8']='1';env['D29_WORKER_THREADS']='1'
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        env[name]='1'
    child=None
    try:
        with (folder/'run.log').open('ab') as out:
            child=subprocess.Popen([sys.executable,'-X','utf8','-B',str(ROOT/'scripts/guarded_worker.py'),
                lease['token'],script,*arguments],env=env,cwd=ROOT,
                stdout=out,stderr=subprocess.STDOUT)
            registry.bind(lease['token'],child.pid)
            state.update(phase='full_history_'+test,child=dict(pid=child.pid,created=registry.identity(child.pid)))
            write_json(status,state)
            code=child.wait()
        state.update(phase='finished_review_pending' if code==0 else 'failed_requires_review',exit_code=code)
        write_json(status,state)
    finally:
        if child is None or child.poll() is not None:registry.release(lease['token'])


if __name__=='__main__':main(sys.argv[1] if len(sys.argv)>1 else 'isolation')
