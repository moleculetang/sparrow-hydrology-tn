"""Create/apply a source-only immutable seal update, only after prerequisites exit.

Private SSH transport is external to this verifier. It never transfers owners,
checkpoints, secrets or data; existing input bytes must match the complete gate.
"""
import argparse
import datetime
import hashlib
import json
import os
import sys
import tarfile
from pathlib import Path,PurePosixPath
ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024**2),b''):digest.update(block)
    return digest.hexdigest()


def prepare():
    gate_path=ROOT/'evidence/formal_gate.json';gate=json.loads(gate_path.read_text(encoding='utf8'))
    if not gate['passed'] or not gate['sealed']:raise RuntimeError('SEALED_GATE_REQUIRED')
    gate_hash=sha(gate_path);archive=ROOT/'deployment'/('sealed_science_'+gate_hash[:16]+'.tar')
    if archive.exists():raise RuntimeError('EXISTING_SEAL_PAYLOAD_VERIFY_BEFORE_RETRY')
    files=[]
    with tarfile.open(archive,'w:') as tar:
        for rel,digest in gate['identities'].items():
            p=ROOT/rel
            if sha(p)!=digest:raise RuntimeError('SEALED_LOCAL_BYTES_CHANGED '+rel)
            if PurePosixPath(rel).parts[0] in ('data','inputs'):continue
            tar.add(p,arcname=rel,recursive=False);files.append(rel)
        tar.add(gate_path,arcname='evidence/formal_gate.json',recursive=False)
    receipt=dict(archive=str(archive),sha256=sha(archive),gate_sha256=gate_hash,files=len(files),data_transferred=False,
        owner_or_checkpoint_transferred=False,scientific_dispatch=False)
    (ROOT/'deployment/sealed_sync_prepared.json').write_text(json.dumps(receipt,indent=2),encoding='utf8')
    print(json.dumps(receipt))


def activate(archive,expected):
    if os.name!='posix' or ROOT!=Path('/home/pc/TWY/SPARROW/Test/20261008_1'):
        raise RuntimeError('EXACT_WORKSTATION_EXPERIMENT_ONLY')
    archive=Path(archive).resolve()
    if archive.parent!=ROOT/'transfer' or sha(archive)!=expected:raise RuntimeError('SEAL_PAYLOAD_HASH_OR_LOCATION')
    sys.path.insert(0,str(ROOT.parent/'.compute_coordination'));import resource_registry as registry
    state=ROOT.parent/'.compute_coordination/state'
    leases=registry.status(root=state)['leases']
    if any(v['project']=='20261008_1' for v in leases.values()):raise RuntimeError('LIVE_PROJECT_SCIENCE_PREVENTS_SOURCE_UPDATE')
    stage=ROOT/('.sealed_stage_'+expected[:16]);stage.mkdir(exist_ok=False)
    with tarfile.open(archive,'r:') as tar:
        seen=set()
        for member in tar.getmembers():
            rel=PurePosixPath(member.name)
            if rel.is_absolute() or '..' in rel.parts or '\\' in member.name or ':' in member.name or not member.isfile() or member.name in seen:
                raise RuntimeError('UNSAFE_SEAL_PAYLOAD_MEMBER')
            seen.add(member.name);p=stage/member.name;p.parent.mkdir(parents=True,exist_ok=True)
            with tar.extractfile(member) as source,p.open('xb') as target:
                for block in iter(lambda:source.read(4*1024**2),b''):target.write(block)
    gate_path=stage/'evidence/formal_gate.json';gate=json.loads(gate_path.read_text(encoding='utf8'))
    if not gate.get('passed') or not gate.get('sealed'):raise RuntimeError('REMOTE_GATE_NOT_SEALED')
    allowed={rel for rel in gate['identities'] if PurePosixPath(rel).parts[0] not in ('data','inputs')}|{'evidence/formal_gate.json'}
    if seen!=allowed:raise RuntimeError('SEAL_PAYLOAD_MEMBERS_MISMATCH')
    for rel,digest in gate['identities'].items():
        p=ROOT/rel if PurePosixPath(rel).parts[0] in ('data','inputs') else stage/rel
        if not p.is_file() or sha(p)!=digest:raise RuntimeError('REMOTE_SEALED_BYTE_MISMATCH '+rel)
    h1=Path('/home/pc/TWY/SPARROW/Test/20260928_1/linux_land1_12/model/vendor/legacy22/data/domains/FULL24C')
    layout=json.loads((stage/'vendor/legacy22/data/domains/FULL24C/arrays.json').read_text(encoding='utf8'))
    for name,spec in layout.items():
        p=(h1/spec['file']).resolve()
        if not p.is_relative_to(h1) or sha(p)!=spec['sha256']:raise RuntimeError('FINAL_REMOTE_H1_IDENTITY_CHANGED '+name)
    backup=ROOT/'evidence/source_snapshots'/('remote_before_seal_'+sha(gate_path)[:16]);backup.mkdir(exist_ok=False)
    changed=[]
    for rel in sorted(allowed-{'evidence/formal_gate.json'}):
        src=stage/rel;dst=ROOT/rel
        if dst.exists() and sha(dst)==sha(src):continue
        if dst.exists():
            before=backup/rel;before.parent.mkdir(parents=True,exist_ok=True);before.write_bytes(dst.read_bytes())
        dst.parent.mkdir(parents=True,exist_ok=True);os.replace(src,dst);changed.append(rel)
    for rel,digest in gate['identities'].items():
        if sha(ROOT/rel)!=digest:raise RuntimeError('FINAL_REMOTE_SEAL_BYTE_MISMATCH '+rel)
    dst=ROOT/'evidence/formal_gate.json'
    if dst.exists() and sha(dst)!=sha(gate_path):raise RuntimeError('EXISTING_REMOTE_FORMAL_SEAL_CONFLICT')
    if not dst.exists():os.replace(gate_path,dst)
    result=dict(passed=True,scope='final_current_science_all_gate_bytes_verified_before_training',gate_sha256=sha(dst),
        archive_sha256=expected,verified_files=len(gate['identities']),h1_arrays_verified=len(layout),changed_source_files=changed,
        preserved_before_change=str(backup.relative_to(ROOT)),data_transferred=False,scientific_dispatch=0,
        verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (ROOT/'evidence/remote_final_seal.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','activate'));p.add_argument('archive',nargs='?');p.add_argument('sha256',nargs='?')
    a=p.parse_args();prepare() if a.mode=='prepare' else activate(a.archive,a.sha256)
