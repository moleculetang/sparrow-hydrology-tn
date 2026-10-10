"""One SSH event wait, immutable verified return and numerical comparison."""
import hashlib
import json
import subprocess
import sys
import tarfile
from pathlib import Path, PurePosixPath

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'deployment')]
from wait_prerequisites import OPTIONS,REMOTE_PYTHON
from compare_hosts import arrays


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()


def main():
    folder=ROOT/'outputs/maize_scope_revision_candidate/transport_v4'
    launch=json.loads((folder/'remote_qualification_launch.json').read_text(encoding='utf8'))
    code=r'''
import hashlib,json,sys,tarfile,time
from pathlib import Path
r=Path('/home/pc/TWY/SPARROW/Test/20261008_1_maize_scope_v2')
sys.path.insert(0,str(r.parent/'.compute_coordination'))
import resource_registry as reg
f=r/'outputs/scope_remote_point_launcher';owner=json.loads((f/'owner.json').read_text())
if owner['pid']!=EXPECTED_PID:raise RuntimeError('EXPECTED_QUALIFICATION_LAUNCHER_CHANGED')
while reg.identity(owner['pid'])==owner['created']:time.sleep(20)
s=json.loads((f/'status.json').read_text())
if s.get('exit_code')!=0 or s['phase']!='qualification_finished_dispatch_restored':
    raise RuntimeError('ACTUAL_QUALIFICATION_FAILED '+json.dumps(s))
q=r/'outputs/scope_remote_point';result=json.loads((q/'complete.json').read_text())
if not result['passed']:raise RuntimeError('ACTUAL_POINT_NOT_PASSED')
names=['outputs/scope_remote_point/'+n for n in ['complete.json','point.npy','gradient.npy',
    'training_prediction.npy','local_kg_day.npy','final_state.npy']]
names+=['outputs/scope_remote_point_launcher/'+n for n in ['status.json','owner.json','launch_receipt.json','worker.log']]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
inventory=dict(files={n:sha(r/n) for n in names},owner=owner)
transport=r/'transfer';transport.mkdir(exist_ok=True)
mf=transport/'scope_point_return_manifest.json'
archive=transport/'scope_point_v4_return.tar.gz'
if archive.exists() or mf.exists():raise RuntimeError('EXISTING_RETURN_ARCHIVE_REVIEW_REQUIRED')
mf.write_text(json.dumps(inventory,indent=2))
with tarfile.open(archive,'w:gz') as t:
 for n in names:t.add(r/n,arcname=n,recursive=False)
 t.add(mf,arcname='return_manifest.json',recursive=False)
print(json.dumps(dict(archive=str(archive),sha256=sha(archive),manifest_sha256=sha(mf))))
'''
    p=subprocess.run(['ssh',*OPTIONS,'pc@10.5.8.38',REMOTE_PYTHON+' -X utf8 -B -'],
        input='EXPECTED_PID='+str(launch['pid'])+'\n'+code,text=True,encoding='utf8',capture_output=True)
    if p.returncode:raise RuntimeError(p.stderr+p.stdout)
    record=json.loads(p.stdout)
    archive=folder/'scope_point_v4_return.tar.gz'
    if archive.exists():raise RuntimeError('EXISTING_LOCAL_RETURN_ARCHIVE_REVIEW_REQUIRED')
    subprocess.run(['scp',*OPTIONS,'pc@10.5.8.38:'+record['archive'],str(archive)],check=True)
    if sha(archive)!=record['sha256']:raise RuntimeError('RETURN_ARCHIVE_BYTES_CHANGED')
    dest=folder/'linux_return';dest.mkdir(exist_ok=False)
    seen=set()
    with tarfile.open(archive,'r:gz') as t:
        for member in t.getmembers():
            rel=PurePosixPath(member.name)
            if rel.is_absolute() or '..' in rel.parts or '\\' in member.name or ':' in member.name or not member.isfile() or member.name in seen:
                raise RuntimeError('UNSAFE_RETURN_MEMBER')
            seen.add(member.name);target=(dest/member.name).resolve()
            if not target.is_relative_to(dest.resolve()):raise RuntimeError('OUTSIDE_RETURN_ROOT')
            target.parent.mkdir(parents=True,exist_ok=True)
            with t.extractfile(member) as src,target.open('xb') as dst:
                for b in iter(lambda:src.read(4*1024**2),b''):dst.write(b)
    if sha(dest/'return_manifest.json')!=record['manifest_sha256']:raise RuntimeError('RETURN_MANIFEST_CHANGED')
    manifest=json.loads((dest/'return_manifest.json').read_text(encoding='utf8'))
    if seen!=set(manifest['files'])|{'return_manifest.json'}:raise RuntimeError('RETURN_MEMBERS_CHANGED')
    for rel,digest in manifest['files'].items():
        if sha(dest/rel)!=digest:raise RuntimeError('RETURN_FILE_CHANGED '+rel)
    reference=ROOT.with_name('20261008_1_maize_scope_v2')/'outputs/scope_strict_point'
    remote=dest/'outputs/scope_remote_point'
    l=json.loads((reference/'complete.json').read_text(encoding='utf8'))
    r=json.loads((remote/'complete.json').read_text(encoding='utf8'))
    compared={n:arrays(reference/(n+'.npy'),remote/(n+'.npy'),tol,relative) for n,tol,relative in [
        ('point',0.,False),('gradient',1e-6,True),('training_prediction',1e-8,True),
        ('local_kg_day',1e-6,False),('final_state',1e-6,False)]}
    error=abs(l['value']-r['value']);jpass=error<=1e-8*(1+abs(l['value']))
    identity=(r['config_sha256']==l['config_sha256'] and r['source_multiplier']==l['source_multiplier']==1.
        and r['reference_receipt_sha256']==sha(reference/'complete.json')
        and r['candidate_manifest_sha256']==sha(folder/'manifest.json'))
    legal=all(d['evaluation_year_rows_returned']==0 for q in [l,r] for d in q['constructor_reads'])
    passed=identity and legal and jpass and r['fresh_canonical_restore_passed'] and all(v['passed'] for v in compared.values())
    result=dict(passed=passed,scope='strict_expanded_K_complete_history_dualhost_not_training',
        arrays=compared,objective_error=error,objective_passed=jpass,identity_passed=identity,
        no_evaluation_rows_returned=legal,remote_fresh_canonical_restore_passed=r['fresh_canonical_restore_passed'],
        actual_remote_receipt_sha256=sha(remote/'complete.json'),archive_sha256=sha(archive),
        reference_sha256=sha(reference/'complete.json'),production_gate_issued=False)
    (folder/'dualhost_comparison.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(dict(passed=passed,objective_error=error,array_errors={n:v.get('max_abs_difference') for n,v in compared.items()},production_gate_issued=False)))
    return 0 if passed else 2


if __name__=='__main__':sys.exit(main())
