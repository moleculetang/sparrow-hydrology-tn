"""One-lifetime completion follower for the accepted12replay internal traces."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from host_resources import registry,registry_root
from controller_lock import acquire
from d29_platform.runtime import write_json,sha


def main():
    folder=ROOT/'outputs/replay_internal_comparison_wait';folder.mkdir(parents=True,exist_ok=True)
    lock=acquire(folder/'lifetime.lock')
    predecessor=json.loads((ROOT/'outputs/replay_internal_traces/owner.json').read_text())
    owner=dict(pid=os.getpid(),created=registry.identity(os.getpid()))
    state=dict(owner=owner,predecessor=predecessor,phase='exact_native_trace_completion_wait',
        no_training_dispatch=True,no_evaluation_labels_read=True,scheduled_task_created=False)
    write_json(folder/'owner.json',owner);write_json(folder/'status.json',state)
    while registry.identity(predecessor['pid'])==predecessor['created']:time.sleep(20)
    proof=ROOT/'outputs/replay_internal_traces/complete.json'
    if not proof.exists() or not json.loads(proof.read_text()).get('passed'):
        state.update(phase='native_trace_exit_requires_review',passed=False)
        write_json(folder/'status.json',state);return 2
    while True:
        lease=registry.acquire('20261008_1','analysis/replay_internal_comparison',threads=1,
            peak_bytes=500000000,root=registry_root())
        if lease.get('granted'):break
        state.update(phase='ready_comparison_resource_wait',last_resource_decision=lease)
        write_json(folder/'status.json',state);time.sleep(20)
    try:
        registry.affinity(lease['cpus'])
        env=os.environ.copy()
        for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS','NUMEXPR_NUM_THREADS','D29_WORKER_THREADS'):
            env[key]='1'
        with (folder/'comparison.log').open('ab') as log:
            p=subprocess.run([sys.executable,'-X','utf8','-B',str(ROOT/'analysis/compare_internal_traces.py')],
                cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        receipt=ROOT/'outputs/replay_internal_traces/paired_complete.json'
        passed=p.returncode==0 and receipt.exists() and json.loads(receipt.read_text()).get('passed')
        state.update(phase='internal_and_network_pairs_returned' if passed else 'comparison_requires_review',
            passed=bool(passed),exit_code=p.returncode,paired_receipt_sha256=sha(receipt) if passed else None)
        write_json(folder/'status.json',state);return 0 if passed else 2
    finally:registry.release(lease['token'],root=registry_root())


if __name__=='__main__':sys.exit(main())
