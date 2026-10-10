"""Adopt restored finite-stage owners and collect once after completion.

Operational recovery only: no sealed source edits, worker restarts, fits,
evaluation labels, or recurring scheduler. Wait for the prior return writer
before creating archives, so a stale bootstrap PID cannot race the collector.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'deployment')]
from host_resources import registry
from controller_lock import acquire
from d29_platform.runtime import write_json, sha
from train_job import validate_admission
from wait_prerequisites import OPTIONS, REMOTE_ROOT, REMOTE_PYTHON


def run(args, **kwargs):
    p = subprocess.run(args, encoding='utf8', stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, **kwargs)
    if p.returncode:
        raise RuntimeError('FINITE_RETURN_FAILURE ' + p.stderr[-3000:] + p.stdout[-3000:])
    return p.stdout


def main():
    if os.name != 'nt':
        raise RuntimeError('WINDOWS_RETURN_OWNER_REQUIRED')
    folder = ROOT / 'outputs/finite_stage_return_wait'
    folder.mkdir(parents=True, exist_ok=True)
    lock = acquire(folder / 'lifetime.lock')
    _, gate_hash = validate_admission()
    owner = dict(pid=os.getpid(), created=registry.identity(os.getpid()))
    local = []
    for name in ('replays_local', 'measurements_local'):
        local.append(json.loads((ROOT / 'outputs' / name / 'owner.json').read_text()))
    prior = json.loads((ROOT / 'outputs/gate_pipeline/status.json').read_text())['owner']
    state = dict(owner=owner, phase='single_remote_current_owner_completion_wait',
                 formal_gate_sha256=gate_hash, local_owners=local, prior_return_writer=prior,
                 scientific_fits_dispatched=0, scheduled_task_created=False)
    write_json(folder / 'owner.json', owner)
    write_json(folder / 'status.json', state)
    try:
        # A single persistent SSH request waits at the workstation. No remote
        # restart and no repeated network status polling while science runs.
        code = f'''
import json,sys,time
from pathlib import Path
root=Path({REMOTE_ROOT!r});sys.path.insert(0,str(root.parent/'.compute_coordination'))
import resource_registry as registry
names=('replays_workstation','measurements_workstation')
owners=[json.loads((root/'outputs'/n/'owner.json').read_text()) for n in names]
while any(registry.identity(o['pid'])==o['created'] for o in owners):time.sleep(20)
states=[json.loads((root/'outputs'/n/'status.json').read_text()) for n in names]
for n,s,e in zip(names,states,('all_owned_replays_passed','all_measurements_returned_comparison_pending')):
    if s['phase']!=e or s['formal_gate_sha256']!={gate_hash!r}:raise RuntimeError('FINITE_REMOTE_TERMINAL_REVIEW '+n)
print(json.dumps(dict(owners=owners,phases=[s['phase'] for s in states])))
'''
        returned = json.loads(run(['ssh', *OPTIONS, 'pc@10.5.8.38',
                                   REMOTE_PYTHON + ' -X utf8 -B -'], input=code))
        state.update(phase='local_and_previous_writer_completion_wait', remote_return=returned)
        write_json(folder / 'status.json', state)
        while any(registry.identity(o['pid']) == o['created'] for o in local + [prior]):
            time.sleep(20)
        for name, expected in (('replays_local', 'all_owned_replays_passed'),
                               ('measurements_local', 'all_measurements_returned_comparison_pending')):
            s = json.loads((ROOT / 'outputs' / name / 'status.json').read_text())
            if s['phase'] != expected or s['formal_gate_sha256'] != gate_hash:
                raise RuntimeError('FINITE_LOCAL_TERMINAL_REVIEW ' + name)
        old_status = ROOT / 'outputs/gate_pipeline/status.json'
        snapshot = folder / 'previous_pipeline_terminal.json'
        if snapshot.exists() and sha(snapshot) != sha(old_status):
            raise RuntimeError('PRIOR_PIPELINE_TERMINAL_SNAPSHOT_CONFLICT')
        if not snapshot.exists():
            snapshot.write_bytes(old_status.read_bytes())
        state['phase'] = 'verified_completed_archive_return'
        write_json(folder / 'status.json', state)
        code = f'''
import hashlib,json,os,tarfile
from pathlib import Path
root=Path({REMOTE_ROOT!r});results=[]
for kind,name in [('replays','replays_workstation'),('measurements','measurements_workstation')]:
    status=root/'outputs'/name/'status.json'
    archive=root/'transfer'/('completed_'+kind+'_'+{gate_hash[:16]!r}+'.tar.gz')
    if not archive.exists():
        temp=archive.with_name(archive.name+'.finite_return.tmp')
        with tarfile.open(temp,'w:gz') as tar:
            prefixes=('replay_',) if kind=='replays' else ('measurement_threads',)
            for p in sorted((root/'outputs/jobs').iterdir()):
                if p.is_dir() and p.name.startswith(prefixes):tar.add(p,arcname=p.relative_to(root).as_posix())
            tar.add(status,arcname=status.relative_to(root).as_posix())
        os.replace(temp,archive)
    h=hashlib.sha256()
    with archive.open('rb') as s:
        for b in iter(lambda:s.read(4*1024**2),b''):h.update(b)
    results.append(dict(kind=kind,archive=str(archive),sha256=h.hexdigest()))
print(json.dumps(results))
'''
        archives = json.loads(run(['ssh', *OPTIONS, 'pc@10.5.8.38',
                                   REMOTE_PYTHON + ' -X utf8 -B -'], input=code))
        state['returns'] = archives
        write_json(folder / 'status.json', state)
        for r in archives:
            archive = ROOT / 'deployment' / Path(r['archive']).name
            if archive.exists() and sha(archive) != r['sha256']:
                raise RuntimeError('FINITE_ARCHIVE_LOCAL_BYTE_CONFLICT')
            if not archive.exists():
                run(['scp', *OPTIONS, 'pc@10.5.8.38:' + r['archive'], str(archive)])
            if sha(archive) != r['sha256']:
                raise RuntimeError('FINITE_ARCHIVE_RETURN_HASH_FAILURE')
            run([sys.executable, '-B', str(ROOT / 'deployment/import_diagnostics.py'),
                 str(archive), r['sha256'], '--kind', r['kind']])
        run([sys.executable, '-B', str(ROOT / 'deployment/merge_replays.py')])
        run([sys.executable, '-B', str(ROOT / 'deployment/compare_stage_measurements.py')])
        state.update(phase='replays_and_stage_measurements_passed_ownership_review_pending',
                     passed=True)
        write_json(folder / 'status.json', state)
        return 0
    except Exception as exc:
        state.update(phase='failure_requires_review', error=str(exc), passed=False)
        write_json(folder / 'status.json', state)
        raise


if __name__ == '__main__':
    sys.exit(main())
