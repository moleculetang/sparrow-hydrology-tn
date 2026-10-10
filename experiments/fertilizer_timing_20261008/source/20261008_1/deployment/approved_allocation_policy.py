"""Admit source-allocation artifacts only after actual two-contract import."""
import hashlib
import json
import re
from pathlib import PurePosixPath


def owner_key(owner):
    return hashlib.sha256(json.dumps(owner, sort_keys=True).encode()).hexdigest()[:16]


def qualify_ack(ack, owner, jobs, scope, ownership_hash):
    if ack is None:
        return False
    if ack.get('remote_owner') != owner:
        return False
    expected = {j: dict(epoch=scope['epochs'][j], gate_sha256=scope['gates'][j]) for j in jobs}
    if (ack.get('version') != 2 or ack.get('transport_passed') is not True
            or ack.get('current_host_snapshot_imported') is not True
            or ack.get('ownership_sha256') != ownership_hash or ack.get('job_contracts') != expected):
        raise RuntimeError('EXACT_CURRENT_TWO_CONTRACT_IMPORT_ACK_REQUIRED')
    name = ack.get('archive_relative')
    if not isinstance(name, str):
        raise RuntimeError('EXACT_PRIVATE_ARCHIVE_PATH_REQUIRED')
    path = PurePosixPath(name)
    if (path.parts[:1] != ('transfer',) or len(path.parts) != 2 or path.as_posix() != name
            or not re.fullmatch(r'approved_scope_terminal_[a-f0-9]{24}\.tar', path.name)):
        raise RuntimeError('REGISTERED_PRIVATE_TWO_CONTRACT_ARCHIVE_REQUIRED')
    if not re.fullmatch('[a-f0-9]{64}', str(ack.get('archive_sha256', ''))):
        raise RuntimeError('ACTUAL_IMPORTED_ARCHIVE_SHA256_REQUIRED')
    if not re.fullmatch('[a-f0-9]{64}', str(ack.get('manifest_sha256', ''))):
        raise RuntimeError('ACTUAL_IMPORTED_MANIFEST_SHA256_REQUIRED')
    return True
