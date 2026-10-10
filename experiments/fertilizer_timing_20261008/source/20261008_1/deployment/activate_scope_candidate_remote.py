"""Stage a separate Linux candidate from verified bytes, with no dispatch."""
import hashlib
import json
import os
import shutil
import sys
import tarfile
from pathlib import Path, PurePosixPath

BASE = Path('/home/pc/TWY/SPARROW/Test')
ORIGINAL = BASE/'20261008_1'
DESTINATION = BASE/'20261008_1_maize_scope_v2'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main(archive_path, archive_sha, manifest_sha):
    archive_path = Path(archive_path).resolve()
    if os.name != 'posix' or archive_path.parent != ORIGINAL/'transfer/scope_candidate_v4':
        raise RuntimeError('EXACT_PRIVATE_STAGING_LOCATION_REQUIRED')
    if sha(archive_path) != archive_sha:
        raise RuntimeError('ARCHIVE_BYTES_CHANGED')
    stage = BASE/('.incoming_scope_' + manifest_sha[:16])
    if DESTINATION.exists() or stage.exists():
        raise RuntimeError('EXISTING_CANDIDATE_OR_STAGE_REVIEW_REQUIRED')
    stage.mkdir()
    seen = set()
    with tarfile.open(archive_path, 'r:') as tar:
        for member in tar.getmembers():
            rel = PurePosixPath(member.name)
            if (rel.is_absolute() or '..' in rel.parts or '\\' in member.name or ':' in member.name
                    or not member.isfile() or member.name in seen):
                raise RuntimeError('UNSAFE_OR_DUPLICATE_MEMBER')
            seen.add(member.name)
            p = (stage/member.name).resolve()
            if not p.is_relative_to(stage):
                raise RuntimeError('OUTSIDE_STAGING_ROOT')
            p.parent.mkdir(parents=True,exist_ok=True)
            with tar.extractfile(member) as source, p.open('xb') as destination:
                shutil.copyfileobj(source,destination,length=4*1024**2)
    manifest_path = stage/'deployment/candidate_manifest.json'
    if sha(manifest_path) != manifest_sha:
        raise RuntimeError('MANIFEST_BYTES_CHANGED')
    manifest = json.loads(manifest_path.read_text(encoding='utf8'))
    if (manifest['remote_root'] != str(DESTINATION) or manifest['original_remote_root'] != str(ORIGINAL)
            or manifest['production_gate_passed'] or manifest['scientific_dispatch'] != 0):
        raise RuntimeError('CANDIDATE_IDENTITY_OR_ADMISSION_MISMATCH')
    gate = ORIGINAL/'evidence/formal_gate.json'
    if sha(gate) != manifest['original_gate_sha256']:
        raise RuntimeError('REMOTE_ORIGINAL_EPOCH_CHANGED')
    approved = {r['path'] for r in manifest['files'] if r['transport']=='payload'} | {'deployment/candidate_manifest.json'}
    if seen != approved or len({r['path'] for r in manifest['files']}) != len(manifest['files']):
        raise RuntimeError('PAYLOAD_OR_INVENTORY_MEMBERS_MISMATCH')
    copied = 0
    for row in manifest['files']:
        rel = PurePosixPath(row['path'])
        if rel.is_absolute() or '..' in rel.parts or '\\' in row['path'] or ':' in row['path']:
            raise RuntimeError('UNSAFE_MANIFEST_PATH')
        dst = (stage/row['path']).resolve()
        if not dst.is_relative_to(stage):
            raise RuntimeError('OUTSIDE_CANDIDATE_ROOT')
        if row['transport']=='verified_remote_original':
            src = (ORIGINAL/row['path']).resolve()
            if not src.is_relative_to(ORIGINAL) or sha(src)!=row['sha256']:
                raise RuntimeError('REMOTE_ORIGINAL_BYTES_CHANGED ' + row['path'])
            dst.parent.mkdir(parents=True,exist_ok=True)
            if dst.exists():
                raise RuntimeError('PAYLOAD_COLLISION_WITH_ORIGINAL')
            shutil.copyfile(src,dst)
            copied += 1
        elif row['transport'] != 'payload':
            raise RuntimeError('UNKNOWN_TRANSPORT_ROLE')
        if dst.stat().st_size != row['bytes'] or sha(dst)!=row['sha256']:
            raise RuntimeError('STAGED_BYTE_MISMATCH ' + row['path'])
    h1 = BASE/'20260928_1/linux_land1_12/model/vendor/legacy22/data/domains/FULL24C'
    for name, spec in manifest['h1_contract'].items():
        p = (h1/spec['file']).resolve()
        if not p.is_relative_to(h1) or sha(p)!=spec['sha256']:
            raise RuntimeError('H1_BYTES_CHANGED ' + name)
    os.rename(stage,DESTINATION)
    receipt = dict(passed=True,scope='independent_candidate_source_input_H1_stage_not_gate',
        original_gate_sha256=sha(gate),candidate_manifest_sha256=manifest_sha,
        archive_sha256=archive_sha,verified_files=len(manifest['files']),
        original_files_copied=copied,h1_arrays_verified=len(manifest['h1_contract']),
        original_scientific_sources_changed=False,owner_or_optimizer_transferred=False,
        scientific_jobs_dispatched=0,production_gate_passed=False)
    target = DESTINATION/'evidence/remote_candidate_staging.json'
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(receipt,indent=2),encoding='utf8')
    print(json.dumps(receipt))


if __name__=='__main__':
    main(*sys.argv[1:])
