"""Create a bounded private SSH payload from the verified local inventory."""
import datetime
import hashlib
import json
import tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()


def main():
    path=ROOT/'deployment/source_input_manifest.json';manifest=json.loads(path.read_text(encoding='utf8'))
    ident=digest(path)[:16];target=ROOT/'deployment'/f'private_science_{ident}.tar'
    if target.exists():raise RuntimeError('EXISTING_TRANSFER_ARCHIVE_CHECK_RECEIPT_BEFORE_RETRY')
    with tarfile.open(target,'w') as archive:
        for row in manifest['files']:
            p=ROOT/row['path']
            if digest(p)!=row['sha256']:raise RuntimeError('LOCAL_SOURCE_CHANGED '+row['path'])
            archive.add(p,arcname=row['path'],recursive=False)
        archive.add(path,arcname='deployment/source_input_manifest.json',recursive=False)
        for name in ('activate_verified.py','workstation_resume.json'):
            p=ROOT/'deployment'/name
            archive.add(p,arcname='deployment/'+name,recursive=False)
    receipt=dict(snapshot=ident,archive=target.name,bytes=target.stat().st_size,sha256=digest(target),
        manifest_sha256=digest(path),private_experiment_transfer=True,keys_included=False,
        owner_or_optimizer_state_included=False,remote_dispatched=False,
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (ROOT/'deployment/transfer_prepared.json').write_text(json.dumps(receipt,indent=2),encoding='utf8')
    print(json.dumps(receipt))


if __name__=='__main__':main()
