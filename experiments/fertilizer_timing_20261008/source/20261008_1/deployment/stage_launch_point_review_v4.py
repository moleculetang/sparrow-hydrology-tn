"""Launch one explicit reviewed operational version, with immutable receipts."""
import argparse
import base64
import hashlib
import json
import subprocess
from pathlib import Path
from wait_prerequisites import OPTIONS,REMOTE_ROOT,REMOTE_PYTHON


def main(version):
    root=Path(__file__).resolve().parents[1]
    files=[f'deployment/checkpoint_point_review_{version}.py',f'deployment/run_point_review_batch_{version}.py',
        f'evidence/checkpoint_point_review_{version}_controls.json']
    payload={r:base64.b64encode((root/r).read_bytes()).decode() for r in files}
    code=f'''
import ast,base64,hashlib,json,os,subprocess,sys
from pathlib import Path
root=Path({REMOTE_ROOT!r});files={payload!r};hashes={{}}
for rel,b64 in files.items():
 raw=base64.b64decode(b64);p=root/rel
 if rel.endswith('.py'):ast.parse(raw.decode())
 if p.exists() and p.read_bytes()!=raw:raise RuntimeError('CURRENT_OPERATION_ALREADY_STAGED_DIFFERENT_BYTES')
 if not p.exists():p.write_bytes(raw)
 hashes[rel]=hashlib.sha256(p.read_bytes()).hexdigest()
f=root/'outputs/checkpoint_point_review_{version}';f.mkdir(exist_ok=True);launchfile=f/'batch_launch.json'
if launchfile.exists():raise RuntimeError('EXISTING_BATCH_LAUNCH_NO_REPEAT')
sys.path[:0]=[str(root),str(root/'scripts'),str(root/'deployment')]
os.environ['D29_RESOURCE_STATE']=str(root.parent/'.compute_coordination/state')
from host_resources import registry
from approved_scope_policy import load_qualified
load_qualified(root)
from approved_parent_lineage import current
parent=current(root)['new_owner']
if registry.identity(parent['pid'])!=parent['created']:raise RuntimeError('ACTUAL_CURRENT_PARENT_REQUIRED')
env=os.environ.copy();env['PYTHONUTF8']='1';env['D29_WORKER_THREADS']='1'
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS','NUMEXPR_NUM_THREADS']:env[k]='1'
with (f/'batch_launch.log').open('ab') as log:
 p=subprocess.Popen([sys.executable,'-X','utf8','-B',str(root/'deployment/run_point_review_batch_{version}.py')],cwd=root,env=env,
  stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
result=dict(passed=True,owner=dict(pid=p.pid,created=registry.identity(p.pid)),parent=parent,files=hashes,scientific_signals=0)
launchfile.write_text(json.dumps(result,indent=2));print(json.dumps(result))
'''
    p=subprocess.run(['ssh',*OPTIONS,'pc@10.5.8.38',REMOTE_PYTHON+' -X utf8 -B -'],input=code,text=True,encoding='utf8',capture_output=True,timeout=60)
    if p.returncode:raise RuntimeError(p.stderr)
    r=json.loads(p.stdout)
    if r['files']!={rel:hashlib.sha256((root/rel).read_bytes()).hexdigest() for rel in files}:
        raise RuntimeError('ACTUAL_STAGED_BYTES_MUST_MATCH')
    f=root/f'outputs/checkpoint_point_review_{version}_transport';f.mkdir(exist_ok=True)
    with (f/'actual_launch.json').open('x') as stream:json.dump(r,stream,indent=2)
    print(json.dumps(r))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('version',choices=['v4']);a=p.parse_args();main(a.version)
