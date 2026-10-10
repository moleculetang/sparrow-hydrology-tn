"""Restore failed measurement bootstrap through PYTHONPATH; sealed bytes stay fixed.

This operational launcher must not change any module or restart live science.
It records the failed invocation and starts only a dead measurement controller.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'work'),str(ROOT/'vendor/legacy22/scripts')]
from host_resources import registry,registry_root
from train_job import validate_admission
from d29_platform.runtime import write_json,sha


def main(role):
    if role!=('local' if os.name=='nt' else 'workstation'):raise RuntimeError('EXACT_STAGE_HOST_ROLE_REQUIRED')
    _,gate_hash=validate_admission()
    folder=ROOT/'outputs'/('measurements_'+role);owner=json.loads((folder/'owner.json').read_text())
    if registry.identity(owner['pid'])==owner['created']:raise RuntimeError('MEASUREMENT_CONTROLLER_STILL_LIVE')
    leases=registry.status(root=registry_root())['leases']
    if any(v['project']=='20261008_1' and v['job'].startswith('measurement_threads') for v in leases.values()):
        raise RuntimeError('MEASUREMENT_SCIENTIFIC_CHILD_STILL_LIVE')
    status=json.loads((folder/'status.json').read_text());log=folder/'threads1.log'
    if not any(r.get('exit_code')==1 for r in status.get('completed',[])) or "No module named 'serial_solvers'" not in log.read_text(encoding='utf8'):
        raise RuntimeError('SPECIFIC_PRE_ORACLE_IMPORT_FAILURE_REQUIRED')
    snapshot=ROOT/'evidence/failure_snapshots'/('measurement_import_'+role);snapshot.mkdir(parents=True,exist_ok=True)
    preserved={}
    for name in ('owner.json','status.json','threads1.log'):
        source=folder/name;target=snapshot/name
        if target.exists() and sha(target)!=sha(source):raise RuntimeError('IMPORT_FAILURE_SNAPSHOT_CONFLICT')
        if not target.exists():target.write_bytes(source.read_bytes())
        preserved[name]=sha(target)
    vendor=ROOT/'vendor/legacy22/scripts';env=os.environ.copy()
    env.update(PYTHONUTF8='1',PYTHONPATH=str(vendor)+(os.pathsep+env['PYTHONPATH'] if env.get('PYTHONPATH') else ''))
    if role=='workstation':env['D29_RESOURCE_STATE']=str(registry_root())
    # Import the actual frozen solver file without performing a target call.
    probe=subprocess.run([sys.executable,'-X','utf8','-B','-c',
        'import serial_solvers; from pathlib import Path; print(Path(serial_solvers.__file__).resolve())'],
        cwd=ROOT,env=env,encoding='utf8',stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    if probe.returncode or Path(probe.stdout.strip()).resolve()!=(vendor/'serial_solvers.py').resolve():
        raise RuntimeError('EXACT_FROZEN_SERIAL_SOLVER_IMPORT_NOT_RESOLVED '+probe.stderr)
    options={'start_new_session':True} if os.name=='posix' else {'creationflags':subprocess.CREATE_NO_WINDOW}
    with (folder/'controller_environment_resume.log').open('ab') as stream:
        child=subprocess.Popen([sys.executable,'-X','utf8','-B',str(ROOT/'scripts/measurement_controller.py'),'--role',role],
            cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,**options)
    receipt=dict(passed=True,scope='operational_import_environment_resume_no_scientific_code_change',
        formal_gate_sha256=gate_hash,old_owner=owner,old_science_was_pre_oracle_failure=True,
        failure_snapshot=str(snapshot.relative_to(ROOT)),failure_artifacts=preserved,
        PYTHONPATH_added=str(vendor),solver_module_sha256=sha(vendor/'serial_solvers.py'),
        new_owner=dict(pid=child.pid,created=registry.identity(child.pid)),
        sealed_scientific_sources_modified=False,live_replays_interrupted=False)
    write_json(ROOT/'evidence'/('measurement_import_environment_resume_'+role+'.json'),receipt)
    print(json.dumps(receipt))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--role',choices=('local','workstation'),required=True);a=p.parse_args();main(a.role)
