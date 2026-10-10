"""Immutable two-contract terminal transport; never packages the narrow K epoch.

The job-to-epoch map is fixed by registration, not training or evaluation scores.
The import is a private evidence namespace, never a live checkpoint destination.
"""
import hashlib
import io
import json
import os
import re
import stat
import tarfile
from pathlib import Path, PurePosixPath

MANIFEST = 'approved_scope_return_manifest.json'
CONTROLLER = {'outputs/campaign_workstation/status.json', 'outputs/campaign_workstation/owner.json'}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def job_from_path(name, jobs):
    if not isinstance(name, str):
        raise ValueError('REGISTERED_RESULT_PATH_REQUIRED')
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name or p.as_posix() != name:
        raise ValueError('NORMALIZED_RELATIVE_RESULT_PATH_REQUIRED')
    if name in CONTROLLER:
        return None
    if len(p.parts) < 4 or p.parts[:2] != ('outputs', 'jobs') or p.parts[2] not in jobs:
        raise ValueError('ONLY_REGISTERED_JOB_OR_CONTROLLER_RESULT_ALLOWED')
    return p.parts[2]


def validate_manifest(manifest, jobs, scope, ownership_hash):
    jobs = set(jobs)
    if manifest.get('version') != 2 or manifest.get('host_role') != 'workstation':
        raise ValueError('EXPLICIT_TWO_CONTRACT_RETURN_VERSION_REQUIRED')
    if len(manifest.get('owned_paths', [])) != len(jobs) or set(manifest['owned_paths']) != jobs:
        raise ValueError('EXACT_REGISTERED_REMOTE_JOBS_REQUIRED')
    expected = {j: dict(epoch=scope['epochs'][j], gate_sha256=scope['gates'][j]) for j in jobs}
    if manifest.get('job_contracts') != expected or manifest.get('ownership_sha256') != ownership_hash:
        raise ValueError('EXACT_FIXED_JOB_CONTRACTS_AND_OWNERSHIP_REQUIRED')
    if manifest.get('selection') != 'fixed_contract_epoch_without_score_selection':
        raise ValueError('NON_SCORE_EPOCH_SELECTION_REQUIRED')
    files = manifest.get('files', [])
    if not 2 <= len(files) <= 20000:
        raise ValueError('BOUNDED_REGULAR_RESULT_INVENTORY_REQUIRED')
    allowed = {}
    for record in files:
        name = record.get('path')
        jid = job_from_path(name, jobs)
        epoch = scope['epochs'][jid] if jid is not None else 'controller'
        gate = scope['gates'][jid] if jid is not None else None
        if record.get('epoch') != epoch or record.get('gate_sha256') != gate:
            raise ValueError('RESULT_FILE_CONTRACT_MISMATCH')
        if name in allowed:
            raise ValueError('DUPLICATE_RESULT_PATH')
        if not isinstance(record.get('bytes'), int) or isinstance(record['bytes'], bool) or record['bytes'] < 0:
            raise ValueError('EXPLICIT_FILE_BYTES_REQUIRED')
        if not isinstance(record.get('sha256'), str) or not re.fullmatch('[a-f0-9]{64}', record['sha256']):
            raise ValueError('EXPLICIT_FILE_SHA256_REQUIRED')
        allowed[name] = record
    if not CONTROLLER <= set(allowed):
        raise ValueError('TERMINAL_CONTROLLER_IDENTITY_REQUIRED')
    if manifest.get('actual_bytes') != sum(r['bytes'] for r in files):
        raise ValueError('RESULT_BYTE_TOTAL_MISMATCH')
    return allowed


def source_path(root, name, jobs, scope):
    from approved_scope_policy import route
    jid = job_from_path(name, jobs)
    return (root if jid is None else route(root, jid, scope)) / name


def terminal_manifest(root, registry, registry_root, expected_owner, scope):
    root = Path(root).resolve()
    if registry.identity(expected_owner['pid']) == expected_owner['created']:
        raise RuntimeError('CONTROLLER_STILL_LIVE_NO_TERMINAL_SNAPSHOT')
    status_path = root/'outputs/campaign_workstation/status.json'
    raw = status_path.read_bytes()
    status = json.loads(raw)
    owner = json.loads((root/'outputs/campaign_workstation/owner.json').read_text())
    if owner != expected_owner or status['owner'] != expected_owner or status.get('approved_scope') != scope:
        raise RuntimeError('ACTUAL_TWO_CONTRACT_CONTROLLER_IDENTITY_REQUIRED')
    if status['phase'] not in ('all_owned_paths_frozen', 'remaining_paths_require_same_point_review', 'safely_drained'):
        raise RuntimeError('NONTERMINAL_CONTROLLER_NO_RESULT_IMPORT')
    ownership_path = root/'deployment/path_ownership.json'
    ownership = json.loads(ownership_path.read_text())
    jobs = sorted(j for j, role in ownership['owners'].items() if role == 'workstation')
    if set(status['owned_paths']) != set(jobs):
        raise RuntimeError('ACTUAL_REMOTE_JOB_OWNERSHIP_CHANGED')
    for record in status.get('running', []):
        if registry.identity(record['pid']) == record['created']:
            raise RuntimeError('LIVE_SCIENTIFIC_CHILD_NO_IMPORT')
    for lease in registry.status(root=registry_root)['leases'].values():
        jid = lease['job'].split('/', 1)[-1]
        if (lease['project'] == '20261008_1' and jid in jobs
                and registry.identity(lease['pid']) == lease['created']):
            raise RuntimeError('LIVE_SCIENTIFIC_LEASE_NO_IMPORT')
    names = sorted(CONTROLLER)
    missing = []
    from approved_scope_policy import route
    for jid in jobs:
        science = route(root, jid, scope)
        folder = science/'outputs/jobs'/jid
        if not folder.exists():
            missing.append(jid)
            continue
        if folder.is_symlink():
            raise RuntimeError('RESULT_DIRECTORY_LINK_REJECTED')
        for path in sorted(folder.rglob('*')):
            if path.is_symlink():
                raise RuntimeError('NESTED_RESULT_LINK_REJECTED')
            if not path.is_dir():
                names.append(path.relative_to(science).as_posix())
    files = []
    for name in names:
        jid = job_from_path(name, jobs)
        path = source_path(root, name, jobs, scope)
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise RuntimeError('ONLY_NONLINK_REGULAR_RESULT_FILES_ALLOWED')
        files.append(dict(path=name, bytes=info.st_size, sha256=digest(path),
                          epoch=scope['epochs'][jid] if jid is not None else 'controller',
                          gate_sha256=scope['gates'][jid] if jid is not None else None))
    manifest = dict(version=2, host_role='workstation', owned_paths=jobs,
                    job_contracts={j: dict(epoch=scope['epochs'][j], gate_sha256=scope['gates'][j]) for j in jobs},
                    ownership_sha256=digest(ownership_path), expected_owner=expected_owner,
                    selection='fixed_contract_epoch_without_score_selection',
                    terminal_phase=status['phase'], files=files,
                    actual_bytes=sum(r['bytes'] for r in files), missing_job_directories=missing,
                    no_evaluation_labels_read=True, snapshot_is_convergence_certificate=False)
    validate_manifest(manifest, jobs, scope, manifest['ownership_sha256'])
    if status_path.read_bytes() != raw or registry.identity(expected_owner['pid']) == expected_owner['created']:
        raise RuntimeError('TERMINAL_CONTROLLER_CHANGED_DURING_INVENTORY')
    return manifest


def make_archive(root, manifest, scope, destination):
    root = Path(root).resolve()
    jobs = manifest['owned_paths']
    validate_manifest(manifest, jobs, scope, manifest['ownership_sha256'])
    raw = json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2).encode('utf8')
    mhash = hashlib.sha256(raw).hexdigest()
    destination = Path(destination)
    destination.mkdir(exist_ok=True, parents=True)
    archive = destination/('approved_scope_terminal_'+mhash[:24]+'.tar')
    if archive.exists():
        raise RuntimeError('EXISTING_TERMINAL_ARCHIVE_REQUIRES_HASHED_IMPORT_REVIEW')
    temporary = archive.with_name(archive.name+'.'+str(os.getpid())+'.tmp')
    with tarfile.open(temporary, 'w') as stream:
        member = tarfile.TarInfo(MANIFEST)
        member.size = len(raw)
        member.mode = 0o600
        stream.addfile(member, io.BytesIO(raw))
        for record in manifest['files']:
            path = source_path(root, record['path'], jobs, scope)
            if path.is_symlink() or path.stat().st_size != record['bytes'] or digest(path) != record['sha256']:
                raise RuntimeError('RESULT_CHANGED_BEFORE_ARCHIVING')
            stream.add(path, arcname=record['path'], recursive=False)
            if digest(path) != record['sha256']:
                raise RuntimeError('RESULT_CHANGED_DURING_ARCHIVING')
    os.replace(temporary, archive)
    return dict(archive=str(archive), sha256=digest(archive), manifest_sha256=mhash,
                bytes=archive.stat().st_size, terminal_phase=manifest['terminal_phase'])


def import_archive(archive, archive_hash, target, jobs, scope, ownership_hash):
    archive = Path(archive)
    target = Path(target).absolute()
    if target.is_symlink() or digest(archive) != archive_hash:
        raise RuntimeError('EXACT_ARCHIVE_AND_PRIVATE_NONLINK_NAMESPACE_REQUIRED')
    target = target.resolve()
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, 'r:') as stream:
        members = stream.getmembers()
        names = [m.name for m in members]
        if len(members) > 20001 or len(set(names)) != len(names) or names.count(MANIFEST) != 1:
            raise RuntimeError('BOUNDED_UNIQUE_MEMBERS_SINGLE_MANIFEST_REQUIRED')
        item = stream.getmember(MANIFEST)
        if not item.isfile() or item.size > 16*1024**2:
            raise RuntimeError('BOUNDED_REGULAR_MANIFEST_REQUIRED')
        raw = stream.extractfile(item).read()
        manifest = json.loads(raw)
        allowed = validate_manifest(manifest, jobs, scope, ownership_hash)
        if set(names) != set(allowed)|{MANIFEST} or manifest['actual_bytes'] > archive.stat().st_size:
            raise RuntimeError('EXACT_ARCHIVE_INVENTORY_REQUIRED')
        for m in members:
            if not m.isfile() or m.issparse() or (m.name != MANIFEST and m.size != allowed[m.name]['bytes']):
                raise RuntimeError('REGULAR_NONSPARSE_EXACT_SIZE_MEMBERS_REQUIRED')
        for m in members:
            original = target/m.name
            for parent in [original, *original.parents]:
                if parent == target.parent:
                    break
                if parent.is_symlink():
                    raise RuntimeError('EXISTING_IMPORT_LINK_REJECTED')
            path = original.resolve()
            if not path.is_relative_to(target):
                raise RuntimeError('RESULT_OUTSIDE_PRIVATE_IMPORT_NAMESPACE')
            expected = hashlib.sha256(raw).hexdigest() if m.name == MANIFEST else allowed[m.name]['sha256']
            source = stream.extractfile(m)
            h = hashlib.sha256()
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                for block in iter(lambda: source.read(4*1024**2), b''):
                    h.update(block)
                if digest(path) != expected or h.hexdigest() != expected:
                    raise RuntimeError('IMMUTABLE_IMPORT_CONFLICT')
                continue
            temp = path.with_name(path.name+'.'+str(os.getpid())+'.partial')
            with temp.open('xb') as output:
                for block in iter(lambda: source.read(4*1024**2), b''):
                    h.update(block)
                    output.write(block)
            if h.hexdigest() != expected:
                raise RuntimeError('IMPORTED_RESULT_HASH_CHANGED')
            os.replace(temp, path)
    return dict(passed=True, target=str(target), archive_sha256=archive_hash,
                manifest_sha256=hashlib.sha256(raw).hexdigest(), file_count=len(allowed),
                job_contracts=manifest['job_contracts'], snapshot_terminal_phase=manifest['terminal_phase'],
                no_live_training_folder_overwritten=True, pickle_loaded=False,
                numerical_sufficiency_certified=False)
