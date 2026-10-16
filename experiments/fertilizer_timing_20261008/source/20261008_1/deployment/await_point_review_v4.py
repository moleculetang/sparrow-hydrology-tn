"""One SSH wait and hash-verified private return of operational evidence."""
import argparse
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path
from wait_prerequisites import OPTIONS,REMOTE_ROOT,REMOTE_PYTHON


def main(version):
    root=Path(__file__).resolve().parents[1];f=root/f'outputs/checkpoint_point_review_{version}_transport'
    launch=json.loads((f/'actual_launch.json').read_text());owner=launch['owner']
    code=f'''
import hashlib,json,sys,tarfile,time
from pathlib import Path
root=Path({REMOTE_ROOT!r});sys.path.insert(0,str(root.parent/'.compute_coordination'));import resource_registry as reg
owner={owner!r};f=root/'outputs/checkpoint_point_review_{version}'
while reg.identity(owner['pid'])==owner['created']:
 args=(Path('/proc')/str(owner['pid'])/'cmdline').read_bytes().split(b'\\0')
 if str(root/'deployment/run_point_review_batch_{version}.py').encode() not in args:raise RuntimeError('EXACT_BATCH_COMMAND_REQUIRED')
 time.sleep(20)
p=f/'batch_complete.json'
if not p.exists():raise RuntimeError('NATIVE_BATCH_DID_NOT_COMPLETE_NO_RESTART '+(f/'batch_launch.log').read_text())
r=json.loads(p.read_text())
if r['owner']!=owner:raise RuntimeError('ACTUAL_BATCH_RECEIPT_OWNER_CHANGED')
archive=root/'transfer'/'point_review_{version}_actual_return.tar.gz'
if archive.exists():raise RuntimeError('EXISTING_POINT_REVIEW_PACK_NO_OVERWRITE')
with tarfile.open(archive,'w:gz') as t:t.add(f,arcname='checkpoint_point_review_{version}')
print(json.dumps(dict(receipt=r,archive=str(archive),sha256=hashlib.sha256(archive.read_bytes()).hexdigest())))
'''
    p=subprocess.run(['ssh',*OPTIONS,'pc@10.5.8.38',REMOTE_PYTHON+' -X utf8 -B -'],input=code,text=True,encoding='utf8',capture_output=True)
    (f/'actual_wait_stdout.json').write_text(p.stdout);(f/'actual_wait_stderr.txt').write_text(p.stderr)
    if p.returncode:raise RuntimeError(p.stderr)
    r=json.loads(p.stdout);archive=f/f'point_review_{version}_actual_return.tar.gz'
    if archive.exists():raise RuntimeError('LOCAL_PACK_ALREADY_EXISTS_NO_OVERWRITE')
    expected=REMOTE_ROOT+f'/transfer/point_review_{version}_actual_return.tar.gz'
    if r['archive']!=expected:raise RuntimeError('EXACT_REVIEW_ARCHIVE_PATH_REQUIRED')
    p=subprocess.run(['scp',*OPTIONS,'pc@10.5.8.38:'+r['archive'],str(archive)],text=True,capture_output=True)
    if p.returncode:raise RuntimeError(p.stderr)
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=r['sha256']:raise RuntimeError('ACTUAL_RETURNED_ARCHIVE_CHANGED')
    target=f/'actual_native_batch';target.mkdir()
    with tarfile.open(archive) as t:
        for m in t.getmembers():
            p=(target/m.name).resolve()
            if not p.is_relative_to(target.resolve()) or not (m.isdir() or m.isfile()):
                raise RuntimeError('REGULAR_PRIVATE_RETURN_PATH_REQUIRED')
        t.extractall(target,filter='data')
    with (f/'actual_return.json').open('x') as stream:json.dump(r,stream,indent=2)
    print(json.dumps(r['receipt']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('version',choices=['v4']);a=p.parse_args();main(a.version)
