"""Immutable terminal-result transport, not checkpoint migration or evaluation.

The archive admits only host-owned logical job outputs and controller receipts.
Imports remain in a separate namespace; no live training folder is overwritten.
Source budgets, monitored water quality, environments and credentials cannot be
included by an arbitrary archive manifest. Pickle files are hashed, never loaded.
"""
import hashlib
import json
import os
import re
import stat
import tarfile
from pathlib import Path, PurePosixPath

MANIFEST='campaign_return_manifest.json'
CONTROLLER={'outputs/campaign_workstation/status.json','outputs/campaign_workstation/owner.json'}


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()


def permitted(name,jobs):
    if not isinstance(name,str):return False
    p=PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name or p.as_posix()!=name:
        return False
    return name in CONTROLLER or (len(p.parts)>=4 and p.parts[:2]==('outputs','jobs') and p.parts[2] in jobs)


def validate_manifest(manifest,jobs,gate_hash,ownership_hash):
    if manifest.get('version')!=1 or manifest.get('host_role')!='workstation':
        raise ValueError('REGISTERED_CAMPAIGN_RESULT_VERSION_AND_HOST_REQUIRED')
    if set(manifest.get('owned_paths',[]))!=set(jobs) or len(manifest['owned_paths'])!=len(jobs):
        raise ValueError('EXACT42REMOTE_PATH_ACCOUNTING_REQUIRED')
    if manifest.get('formal_gate_sha256')!=gate_hash or manifest.get('ownership_sha256')!=ownership_hash:
        raise ValueError('FROZEN_CAMPAIGN_RETURN_IDENTITY_MISMATCH')
    files=manifest.get('files',[])
    if not 2<=len(files)<=20000:raise ValueError('BOUNDED_RESULT_FILE_ACCOUNTING_REQUIRED')
    seen=set()
    for record in files:
        name=record.get('path')
        if name in seen or not permitted(name,jobs):raise ValueError('UNREGISTERED_OR_DUPLICATE_RESULT_PATH')
        seen.add(name)
        if not isinstance(record.get('bytes'),int) or isinstance(record['bytes'],bool) or record['bytes']<0:
            raise ValueError('EXPLICIT_RESULT_BYTE_SIZE_REQUIRED')
        if not isinstance(record.get('sha256'),str) or not re.fullmatch('[a-f0-9]{64}',record['sha256']):
            raise ValueError('EXPLICIT_RESULT_SHA256_REQUIRED')
    if not CONTROLLER<=seen:raise ValueError('MISSING_CONTROLLER_TERMINAL_IDENTITY')
    if manifest.get('actual_bytes')!=sum(r['bytes'] for r in files):raise ValueError('RESULT_BYTE_TOTAL_MISMATCH')
    return {r['path']:r for r in files}


def terminal_manifest(root,registry,registry_root,expected_owner):
    root=Path(root).resolve()
    if registry.identity(expected_owner['pid'])==expected_owner['created']:
        raise RuntimeError('CAMPAIGN_CONTROLLER_STILL_LIVE_NO_RESULT_SNAPSHOT')
    status_path=root/'outputs/campaign_workstation/status.json'
    status_bytes=status_path.read_bytes();status=json.loads(status_bytes)
    owner=json.loads((root/'outputs/campaign_workstation/owner.json').read_text())
    if owner!=expected_owner or status['owner']!=expected_owner:raise RuntimeError('TERMINAL_CONTROLLER_OWNER_CHANGED')
    if status['phase'] not in ('all_owned_paths_frozen','remaining_paths_require_same_point_review','safely_drained'):
        raise RuntimeError('NO_TERMINAL_CAMPAIGN_STATE_NO_IMPORT')
    ownership_path=root/'deployment/path_ownership.json';ownership=json.loads(ownership_path.read_text())
    jobs=sorted(k for k,v in ownership['owners'].items() if v=='workstation')
    if len(jobs)!=42 or set(status['owned_paths'])!=set(jobs):raise RuntimeError('EXACT_REMOTE_OWNERSHIP_REQUIRED')
    for r in status.get('running',[]):
        if registry.identity(r['pid'])==r['created']:raise RuntimeError('LIVE_SCIENTIFIC_CHILD_NO_SNAPSHOT')
    for lease in registry.status(root=registry_root)['leases'].values():
        jid=lease['job'].split('/',1)[-1]
        if lease['project']=='20261008_1' and jid in jobs and registry.identity(lease['pid'])==lease['created']:
            raise RuntimeError('REMOTE_OWNED_PATH_LEASE_STILL_LIVE')
    files=[]
    paths=[root/p for p in sorted(CONTROLLER)]
    for jid in jobs:
        folder=root/'outputs/jobs'/jid
        if folder.exists():
            if folder.is_symlink():raise RuntimeError('RESULT_DIRECTORY_LINK_REJECTED')
            paths.extend(sorted(p for p in folder.rglob('*') if not p.is_dir()))
            if any(p.is_symlink() for p in folder.rglob('*')):raise RuntimeError('NESTED_RESULT_LINK_REJECTED')
    for p in paths:
        relative=p.relative_to(root).as_posix();info=p.lstat()
        if not permitted(relative,jobs) or not stat.S_ISREG(info.st_mode) or info.st_nlink!=1:
            raise RuntimeError('ONLY_OWNED_REGULAR_RESULT_FILES_ALLOWED '+relative)
        files.append(dict(path=relative,bytes=info.st_size,sha256=digest(p)))
    manifest=dict(version=1,host_role='workstation',owned_paths=jobs,
        formal_gate_sha256=digest(root/'evidence/formal_gate.json'),ownership_sha256=digest(ownership_path),
        expected_owner=expected_owner,terminal_phase=status['phase'],files=files,
        actual_bytes=sum(f['bytes'] for f in files),missing_job_directories=[j for j in jobs if not (root/'outputs/jobs'/j).exists()],
        no_live_checkpoint_import=True,no_evaluation_outcomes_read=True,
        snapshot_is_convergence_certificate=False)
    validate_manifest(manifest,jobs,manifest['formal_gate_sha256'],manifest['ownership_sha256'])
    if status_path.read_bytes()!=status_bytes or registry.identity(expected_owner['pid'])==expected_owner['created']:
        raise RuntimeError('TERMINAL_RESULT_CHANGED_DURING_MANIFEST')
    return manifest


def make_archive(root,manifest,destination):
    root=Path(root).resolve();destination=Path(destination)
    raw=json.dumps(manifest,ensure_ascii=False,sort_keys=True,indent=2).encode('utf8')
    manifest_hash=hashlib.sha256(raw).hexdigest()
    destination.mkdir(parents=True,exist_ok=True)
    archive=destination/('campaign_terminal_'+manifest_hash[:24]+'.tar')
    if archive.exists():return dict(archive=str(archive),sha256=digest(archive),manifest_sha256=manifest_hash)
    temporary=archive.with_name(archive.name+'.'+str(os.getpid())+'.tmp')
    import io
    with tarfile.open(temporary,'w') as stream:
        info=tarfile.TarInfo(MANIFEST);info.size=len(raw);info.mode=0o600
        stream.addfile(info,io.BytesIO(raw))
        for r in manifest['files']:
            p=root/r['path']
            if p.stat().st_size!=r['bytes'] or digest(p)!=r['sha256']:
                raise RuntimeError('RESULT_CHANGED_BEFORE_PACK '+r['path'])
            stream.add(p,arcname=r['path'],recursive=False)
            if digest(p)!=r['sha256']:raise RuntimeError('RESULT_CHANGED_DURING_PACK '+r['path'])
    os.replace(temporary,archive)
    return dict(archive=str(archive),sha256=digest(archive),manifest_sha256=manifest_hash,
                bytes=archive.stat().st_size)


def import_archive(archive,expected_archive_hash,target,jobs,gate_hash,ownership_hash):
    archive=Path(archive);target=Path(target).absolute()
    if target.is_symlink():raise RuntimeError('IMPORT_NAMESPACE_LINK_REJECTED')
    target=target.resolve()
    if digest(archive)!=expected_archive_hash:raise RuntimeError('CAMPAIGN_ARCHIVE_HASH_MISMATCH')
    target.mkdir(parents=True,exist_ok=True)
    with tarfile.open(archive,'r:') as stream:
        members=stream.getmembers()
        if len(members)>20001:raise RuntimeError('BOUNDED_ARCHIVE_MEMBERS_REQUIRED')
        names=[m.name for m in members]
        if len(names)!=len(set(names)) or names.count(MANIFEST)!=1:
            raise RuntimeError('EXACT_SINGLE_MANIFEST_NO_DUPLICATE_MEMBERS')
        item=stream.getmember(MANIFEST)
        if not item.isfile() or item.size>16*1024**2:raise RuntimeError('BOUNDED_REGULAR_JSON_MANIFEST_REQUIRED')
        raw=stream.extractfile(item).read();manifest=json.loads(raw)
        allowed=validate_manifest(manifest,jobs,gate_hash,ownership_hash)
        if manifest['actual_bytes']>archive.stat().st_size:
            raise RuntimeError('SPARSE_OR_UNACCOUNTED_RESULT_SIZE_REJECTED')
        if set(names)!=(set(allowed)|{MANIFEST}):raise RuntimeError('ARCHIVE_NOT_EXACT_MANIFEST_FILE_SET')
        for m in members:
            if not m.isfile() or m.issparse():raise RuntimeError('LINK_DIRECTORY_SPARSE_OR_SPECIAL_RESULT_MEMBER_REJECTED')
            if m.name!=MANIFEST and m.size!=allowed[m.name]['bytes']:
                raise RuntimeError('ARCHIVE_MANIFEST_SIZE_MISMATCH')
        for m in members:
            name=m.name
            original=target/name
            # Existing directory aliases cannot redirect an otherwise safe tar
            # path. Never resolve/write through a symlink during a later resume.
            for parent in [original,*original.parents]:
                if parent==target.parent:break
                if parent.is_symlink():raise RuntimeError('EXISTING_IMPORT_LINK_REJECTED')
            p=original.resolve()
            if not p.is_relative_to(target):raise RuntimeError('RESULT_OUTSIDE_IMPORT_NAMESPACE')
            expected=hashlib.sha256(raw).hexdigest() if name==MANIFEST else allowed[name]['sha256']
            h=hashlib.sha256();source=stream.extractfile(m)
            p.parent.mkdir(parents=True,exist_ok=True)
            if p.exists():
                for b in iter(lambda:source.read(4*1024**2),b''):h.update(b)
                if digest(p)!=expected or h.hexdigest()!=expected:raise RuntimeError('IMMUTABLE_IMPORT_BYTE_CONFLICT')
                continue
            temp=p.with_name(p.name+'.'+str(os.getpid())+'.partial')
            with temp.open('xb') as out:
                for b in iter(lambda:source.read(4*1024**2),b''):h.update(b);out.write(b)
            if h.hexdigest()!=expected:
                raise RuntimeError('IMPORTED_RESULT_FILE_SHA256_MISMATCH '+name)
            os.replace(temp,p)
    return dict(passed=True,archive_sha256=expected_archive_hash,
        manifest_sha256=hashlib.sha256(raw).hexdigest(),target=str(target),
        owned_path_count=len(jobs),file_count=len(allowed),actual_bytes=manifest['actual_bytes'],
        snapshot_terminal_phase=manifest['terminal_phase'],no_live_training_folder_overwritten=True,
        transport_receipt_is_numerical_convergence_certificate=False)
