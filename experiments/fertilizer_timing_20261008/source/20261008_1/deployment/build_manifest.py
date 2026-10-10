"""Local deployment inventory only; never SSH, transfer, publish or dispatch.

The remote verifier must match all bytes before using an existing input copy.
Caches, owner records, optimizer states and private keys are never copied by
this inventory. Job ownership is transferred only by a separate reviewed step.
"""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()


def main():
    entries=[]
    for directory in ('timing','d29_platform','d29_training','config','data','inputs','vendor','scripts','tests','work'):
        for path in sorted((ROOT/directory).rglob('*')):
            if not path.is_file() or any(s in ('__pycache__','cache','.git') for s in path.parts):continue
            # Audit hooks and Linux scientific imports keep caches inside the
            # new experiment; control artifacts are explicitly distinguished.
            rel=path.relative_to(ROOT).as_posix()
            entries.append(dict(path=rel,bytes=path.stat().st_size,sha256=digest(path),
                role='mutable_control' if rel=='config/workstation_hold.json' else
                     'data' if directory in ('data','inputs') else 'source_and_contract'))
    layout=ROOT/'vendor/legacy22/data/domains/FULL24C/arrays.json'
    arrays=json.loads(layout.read_text(encoding='utf8'))
    record=dict(scope='local_inventory_not_transfer_or_dispatch_receipt',project='20261008_1',
        remote_root='/home/pc/TWY/SPARROW/Test/20261008_1',files=entries,
        total_bytes=sum(r['bytes'] for r in entries),h1_readonly_array_contract=arrays,
        h1_environment_variable='D29_H1_ROOT',remote_h1_path_verified=False,
        optimizer_or_owner_records_included=False,private_keys_included=False,
        formal_scientific_gate_present=(ROOT/'evidence/formal_gate.json').exists(),
        scientific_dispatch_allowed=False)
    target=ROOT/'deployment/source_input_manifest.json'
    target.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(dict(files=len(entries),bytes=record['total_bytes'],dispatch=False)))


if __name__=='__main__':main()
