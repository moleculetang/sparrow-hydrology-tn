"""Campaign-scoped I/O and immutable provenance; no writes into ancestors."""
from pathlib import Path
import ctypes
import datetime
import hashlib
import json
import os
import sys
import time

RUN = Path(__file__).resolve().parents[1]
TEST = RUN.parent
ROOT = TEST.parent


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def require_environment():
    if Path(sys.prefix).name.lower() != 'sparrow':
        raise RuntimeError('All model work requires conda sparrow')


def read(path):
    # Windows can temporarily deny an open while another process atomically
    # replaces the same status file. Retry boundedly; persistent denial stays
    # an explicit observation error, never an empty/default status.
    for attempt in range(10):
        try:return json.loads(Path(path).read_text(encoding='utf-8-sig'))
        except PermissionError:
            if attempt==9:raise
            time.sleep(min(.01*2**attempt,.5))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def local(path):
    path = Path(path).resolve()
    if not path.is_relative_to(RUN.resolve()):
        raise RuntimeError(f'Write outside registered campaign: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def replace(temp, target):
    # Large checkpoint files can remain briefly locked by Windows readers.
    # The observed failed commit succeeded in the error-handler's subsequent
    # save; retain atomic replacement but allow a bounded 30-second window.
    deadline=time.monotonic()+30.
    attempt=0
    while True:
        try:
            temp.replace(target)
            return
        except PermissionError:
            remaining=deadline-time.monotonic()
            if remaining<=0:raise
            time.sleep(min(.05*2**min(attempt,4),.5,remaining))
            attempt+=1


def write(path, value):
    path = local(path)
    temp = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    replace(temp, path)


def immutable(path, value):
    if Path(path).exists():
        if read(path) != value:
            raise RuntimeError(f'Immutable registration differs: {path}')
    else:
        write(path, value)
    return sha(path)


def register(path, role):
    path = Path(path).resolve()
    registry_path = RUN / 'reports/input_registry.json'
    registry = read(registry_path) if registry_path.exists() else {'inputs': {}}
    item = {'sha256': sha(path), 'role': role}
    key = str(path)
    if key in registry['inputs']:
        if registry['inputs'][key]['sha256'] != item['sha256']:
            raise RuntimeError(f'Registered input changed: {path}')
    else:
        if (RUN/'reports/launch_validation.json').exists():
            raise RuntimeError('Input registry is frozen after launch validation')
        registry['inputs'][key] = item
        write(registry_path, registry)
    return path


def verify_registry():
    for path, item in read(RUN / 'reports/input_registry.json')['inputs'].items():
        if sha(path) != item['sha256']:
            raise RuntimeError(f'Registered input changed: {path}')


class MemoryStatus(ctypes.Structure):
    _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
        (n, ctypes.c_ulonglong) for n in ['total', 'available', 'page', 'available_page', 'virtual', 'available_virtual', 'extended']]


def memory():
    value = MemoryStatus()
    value.length = ctypes.sizeof(value)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value)):
        raise ctypes.WinError()
    return {'total_gib': value.total / 2**30, 'available_gib': value.available / 2**30,
            'used_percent': 100 * (value.total - value.available) / value.total}


def guard(reserve_gib=0):
    value = memory()
    if value['used_percent'] >= 90 or value['available_gib'] - reserve_gib < value['total_gib'] * .10:
        raise MemoryError('RESOURCE_YIELD ' + json.dumps(value))
    return value
