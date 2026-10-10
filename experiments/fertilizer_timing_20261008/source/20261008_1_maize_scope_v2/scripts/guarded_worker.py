"""Bind-before-import worker bootstrap; one host-local lease per exact PID.

This is an executor, not a scheduler. It never acquires, steals or releases
leases; the owning controller must bind its live child before scientific imports.
"""
import json
import os
import runpy
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from host_resources import registry,registry_root


def main():
    token, relative, *arguments = sys.argv[1:]
    target = (ROOT/relative).resolve()
    if not target.is_relative_to(ROOT) or target.suffix != '.py' or not target.is_file():
        raise RuntimeError('REGISTERED_LOCAL_SCRIPT_REQUIRED')
    created = registry.identity(os.getpid())
    deadline = time.monotonic()+30.
    while True:
        lease = registry.status(root=registry_root())['leases'].get(token)
        if not lease or lease['project'] != '20261008_1':
            raise RuntimeError('LIVE_PROJECT_LEASE_REQUIRED')
        if lease['pid'] == os.getpid() and lease['created'] == created: break
        if time.monotonic() >= deadline: raise RuntimeError('CHILD_BIND_NOT_CONFIRMED')
        time.sleep(.05)
    registry.affinity(lease['cpus'])
    os.environ['D29_LEASE_TOKEN'] = token
    os.environ['D29_WORKER_THREADS'] = str(lease['threads'])
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                 'NUMBA_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = str(lease['threads'])
    os.environ['PYTHONUTF8'] = '1'
    record = dict(pid=os.getpid(), created=created, lease=lease, script=relative,
                  arguments=arguments, host=__import__('platform').node())
    path = ROOT/'outputs/worker_bootstrap'/f'{os.getpid()}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2), encoding='utf8')
    sys.argv = [str(target), *arguments]
    sys.path.insert(0, str(ROOT))
    runpy.run_path(str(target), run_name='__main__')


if __name__ == '__main__': main()
