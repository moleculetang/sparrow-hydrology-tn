"""Read-only SSH snapshot of accepted non-K freezes for early local physics.

No live optimizer read, signals, ownership change, scoring or terminal admission.
"""
import ast,json,subprocess,sys,tarfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'analysis'),str(ROOT/'deployment')]
from wait_prerequisites import OPTIONS,REMOTE_ROOT,REMOTE_PYTHON
from d29_platform.runtime import sha,write_json


def safe_members(stream,target):
    members=stream.getmembers()
    names=set()
    for member in members:
        if (member.name in names or not member.isfile() or
                not (target/member.name).resolve().is_relative_to(target.resolve()) or
                '\\' in member.name or ':' in member.name):
            raise RuntimeError('UNSAFE_OR_DUPLICATE_PRIVATE_ARCHIVE_MEMBER')
        names.add(member.name)
    return members


def main():
    ids=sys.argv[1:]
    if not ids or len(ids)!=len(set(ids)):raise RuntimeError('UNIQUE_EXPLICIT_STAGED_IDS_REQUIRED')
    destination=ROOT/'outputs/staged_frozen_import'
    if destination.exists():raise RuntimeError('IMMUTABLE_STAGED_IMPORT_NO_OVERWRITE')
    jobs=json.loads((ROOT/'config/jobs.json').read_text())
    selected=[next(j for j in jobs if j['id']==jid) for jid in ids]
    if any(j['calendar']=='K' for j in selected):raise RuntimeError('FIRST_STAGED_IMPORT_NONK_ONLY')
    gate=sha(ROOT/'evidence/formal_gate.json')
    support=json.loads((ROOT/'evidence/expected_station_support.json').read_text(encoding='utf8'))
    tree=ast.parse((ROOT/'analysis/assemble_approved_delivery.py').read_text(encoding='utf8'))
    validator='\n'.join(ast.unparse(n) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('read','regular','validate_frozen'))
    code='''
import json,sys,tarfile,hashlib
from pathlib import Path
root=Path(REMOTE_ROOT);sys.path[:0]=[str(root),str(root/'scripts'),str(root/'analysis')]
from d29_platform.runtime import sha
exec(compile(VALIDATOR,'readonly_frozen_validator','exec'))
jobs=json.loads((root/'config/jobs.json').read_text());support=SUPPORT
if sha(root/'evidence/formal_gate.json')!=GATE:raise RuntimeError('REMOTE_SCIENCE_GATE_CHANGED')
records=[];sources=[]
for job in SELECTED:
 if next(j for j in jobs if j['id']==job['id'])!=job:raise RuntimeError('REGISTERED_JOB_CHANGED')
 folder=root/'outputs/jobs'/job['id'];bound=validate_frozen(folder,job,GATE,support)
 extra={'audited_training_prediction.npy':sha(folder/'audited_training_prediction.npy')}
 records.append(dict(job=job,bound_files=bound,extra_files=extra))
 sources.extend((folder/rel,'outputs/jobs/'+job['id']+'/'+rel) for rel in {**bound,**extra})
out=root/'transfer/staged_physics_20261010_v1';out.mkdir(exist_ok=False)
manifest=out/'manifest.json';manifest.write_text(json.dumps(dict(partial_only=True,evaluation_admitted=False,records=records),indent=2))
archive=out/'frozen.tar'
with tarfile.open(archive,'w') as stream:
 stream.add(manifest,arcname='manifest.json',recursive=False)
 for path,name in sources:stream.add(path,arcname=name,recursive=False)
for record in records:
 folder=root/'outputs/jobs'/record['job']['id']
 if validate_frozen(folder,record['job'],GATE,support)!=record['bound_files']:raise RuntimeError('FREEZE_CHANGED_DURING_COPY')
 if any(sha(folder/rel)!=v for rel,v in record['extra_files'].items()):raise RuntimeError('EXTRA_CHANGED_DURING_COPY')
print(json.dumps(dict(archive=str(archive),sha256=sha(archive),manifest_sha256=sha(manifest),records=len(records))))
'''
    code=f'REMOTE_ROOT={REMOTE_ROOT!r}\nGATE={gate!r}\nSELECTED={selected!r}\nVALIDATOR={validator!r}\nSUPPORT={support!r}\n'+code
    run=subprocess.run(['ssh',*OPTIONS,'pc@10.5.8.38',REMOTE_PYTHON+' -X utf8 -B -'],
        input=code,text=True,encoding='utf8',capture_output=True,timeout=300)
    if run.returncode:raise RuntimeError(run.stderr+run.stdout)
    packed=json.loads(run.stdout);destination.mkdir()
    archive=destination/'frozen.tar'
    result=subprocess.run(['scp',*OPTIONS,'pc@10.5.8.38:'+packed['archive'],str(archive)],capture_output=True,text=True,timeout=1800)
    if result.returncode:raise RuntimeError(result.stderr)
    if sha(archive)!=packed['sha256']:raise RuntimeError('PRIVATE_ARCHIVE_HASH_MISMATCH')
    target=destination/'data';target.mkdir()
    with tarfile.open(archive,'r') as stream:stream.extractall(target,members=safe_members(stream,target))
    if sha(target/'manifest.json')!=packed['manifest_sha256']:raise RuntimeError('PRIVATE_MANIFEST_HASH_MISMATCH')
    write_json(destination/'complete.json',dict(passed=True,partial_only=True,evaluation_admitted=False,
        training_ownership_changed=False,target=str(target),**packed))
    from staged_frozen_sources import resolve
    for jid in ids:resolve(jid)
    print(json.dumps(dict(passed=True,paths=len(ids),target=str(target),evaluation_admitted=False)))


if __name__=='__main__':main()
