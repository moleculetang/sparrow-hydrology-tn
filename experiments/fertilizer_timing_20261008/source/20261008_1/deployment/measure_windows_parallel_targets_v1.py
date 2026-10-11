"""Two real native target requests in an isolated, synchronized pilot job.

Uses sealed production modules and training support; never saves a campaign
optimizer, reads evaluation scores, or changes scientific admission.
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts'), str(ROOT/'work'), str(ROOT/'vendor/legacy22/scripts')]
from d29_platform.runtime import configure, write_json
configure()
import numpy as np
from freeze_job import training_only_adapter
from timing.projected_gradient import original_point
from native_bound_adapter_v2 import CanonicalLBFGSB
from train_job import validate_admission
from job_lease import require
from host_resources import registry


def digest(a):
    a = np.ascontiguousarray(a)
    return dict(shape=list(a.shape), dtype=str(a.dtype),
                sha256=hashlib.sha256(memoryview(a).cast('B')).hexdigest())


def main(round_id, index):
    if os.name != 'nt': raise RuntimeError('WINDOWS_FINITE_CAPACITY_WORKER_ONLY')
    _, gate_hash = validate_admission()
    jid = f'{round_id}_{index}'
    lease = require(jid, 'parallel_target')
    jobs = json.loads((ROOT/'config/jobs.json').read_text(encoding='utf8'))
    job = next(j.copy() for j in jobs if j['id']=='TIM_F23_C0_V0_s0')
    job['id'] = 'parallel_target_' + jid
    a = training_only_adapter(job)
    barrier = ROOT/'outputs/parallel_target_pilot'/round_id
    write_json(a.folder/'ready.json', dict(pid=os.getpid(), created=registry.identity(os.getpid()),
                                         gate_sha256=gate_hash, lease=lease))
    while not (barrier/'go.json').exists():
        if (barrier/'cancel.json').exists():
            write_json(a.folder/'cancelled_before_oracle.json', dict(target_calls=0))
            return 3
        time.sleep(.2)
    native = CanonicalLBFGSB(a.initial/a.scale, (a.bounds/a.scale[:,None]).tolist(),
                            stage=1, maxiter=np.iinfo(np.int32).max)
    calls = []
    start = time.monotonic()
    def fun(z):
        x = original_point(z, a.scale, a.bounds)
        before = time.monotonic()
        value, gradient = a.value_gradient(x)
        seconds = time.monotonic() - before
        call = len(calls)
        np.savez(a.folder/f'call_{call}.npz', point=x, gradient=gradient,
                 final_state=a.last_chain.final, prediction=a.prediction)
        calls.append(dict(seconds=seconds, objective=float(value), local_export=digest(a.local_prediction)))
        write_json(a.folder/'calls.json', calls)
        return value, gradient*a.scale
    while len(calls)<2:
        if native.step(fun)=='done':
            break
    from threadpoolctl import threadpool_info
    import torch
    pools = threadpool_info()
    pool_ok = (all(p['num_threads']==1 for p in pools) and
               torch.get_num_threads()==1 and torch.get_num_interop_threads()==1)
    memory = registry.process_memory(os.getpid())
    receipt = dict(passed=len(calls)==2 and pool_ok, seconds=time.monotonic()-start,
                   calls=calls, actual_pools=pools, threads=1,
                   peak_bytes=memory['peak_commit'], process_memory=memory,
                   telemetry_basis='Windows native peak private commit',
                   formal_gate_sha256=gate_hash, lease=lease, scope='two_native_requests_not_fit',
                   evaluation_scores_read=False, source_multiplier=1.)
    write_json(a.folder/'parallel_target.json', receipt)
    return 0 if receipt['passed'] else 2


if __name__=='__main__':
    sys.exit(main(sys.argv[1], int(sys.argv[2])))
