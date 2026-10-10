"""Finite local diagnostic with its own lease; never restart scientific fits."""
import argparse
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


def main(probe):
    folder=ROOT/'outputs'/('training_predicate_access_probe_v2' if probe else 'training_predicate_equivalence_v2')
    folder.mkdir(exist_ok=True)
    lock=acquire(folder/'lifetime.lock')
    if (folder/'returned.json').exists():raise RuntimeError('RETURNED_DIAGNOSTIC_REQUIRES_REVIEW_NOT_RESTART')
    lease=registry.acquire('20261008_1','analysis/training_predicate_equivalence',threads=1,
                           peak_bytes=6500000000,root=registry_root())
    write_json(folder/'resource_decision.json',lease)
    if not lease.get('granted'):return 4
    env=os.environ.copy();env.update(PYTHONUTF8='1',D29_WORKER_THREADS='1',
        PYTHONPATH=str(ROOT/'vendor/legacy22/scripts'))
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        env[name]='1'
    child=None;start=time.monotonic()
    try:
        with (folder/'run.log').open('ab') as log:
            args=[sys.executable,'-X','utf8','-B',str(ROOT/'scripts/guarded_worker.py'),
                  lease['token'],'analysis/verify_training_predicate_equivalence.py']
            if probe:args.append('--constructor-only')
            child=subprocess.Popen(args,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            registry.bind(lease['token'],child.pid,root=registry_root())
            owner=dict(pid=child.pid,created=registry.identity(child.pid),lease=lease)
            write_json(folder/'status.json',dict(phase='constructor_probe' if probe else 'full_oracle_check',owner=owner))
            child.wait()
        complete=folder/'complete.json'
        passed=child.returncode==0 and complete.exists() and json.loads(complete.read_text()).get('passed',False)
        result=dict(passed=passed,exit_code=child.returncode,owner=owner,seconds=time.monotonic()-start,
                    complete_sha256=sha(complete) if passed else None,
                    production_sources_or_checkpoints_changed=False)
        write_json(folder/'returned.json',result)
        write_json(folder/'status.json',dict(phase='passed' if passed else 'requires_review',**result))
        print(json.dumps(result));return 0 if passed else 1
    finally:
        if child is None or child.poll() is not None:registry.release(lease['token'],root=registry_root())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--constructor-only',action='store_true')
    sys.exit(main(p.parse_args().constructor_only))
