"""Strict one-time archive verification and safe private staging."""
import argparse
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()


def main(archive,sha256):
    root=Path('/home/pc/TWY/SPARROW/Test/20261008_1').resolve();archive=Path(archive).resolve()
    if archive.parent!=root/'transfer' or digest(archive)!=sha256:raise RuntimeError('PRIVATE_ARCHIVE_HASH_OR_LOCATION_MISMATCH')
    ident=archive.name.removeprefix('private_science_').removesuffix('.tar')
    if len(ident)!=16 or any(c not in '0123456789abcdef' for c in ident):raise RuntimeError('INVALID_SNAPSHOT_ID')
    stage=root/('.incoming_'+ident)
    if stage.exists():raise RuntimeError('EXISTING_STAGE_INSPECT_BEFORE_RETRY')
    stage.mkdir()
    with tarfile.open(archive,'r:') as tar:
        seen=set()
        for member in tar.getmembers():
            relative=Path(member.name);target=(stage/relative).resolve()
            if relative.is_absolute() or '..' in relative.parts or not target.is_relative_to(stage) or not member.isfile() or member.name in seen:
                raise RuntimeError('UNSAFE_OR_DUPLICATE_ARCHIVE_MEMBER '+member.name)
            seen.add(member.name);target.parent.mkdir(parents=True,exist_ok=True)
            with tar.extractfile(member) as source,target.open('xb') as destination:
                for block in iter(lambda:source.read(4*1024**2),b''):destination.write(block)
    h1='/home/pc/TWY/SPARROW/Test/20260928_1/linux_land1_12/model/vendor/legacy22/data/domains/FULL24C'
    python='/home/pc/TWY/SPARROW/Test/.runtime/envs/sparrow/bin/python'
    result=subprocess.run([python,'-X','utf8','-B',str(stage/'deployment/activate_verified.py'),str(stage),str(root),h1],check=True)
    print(json.dumps(dict(archive_sha256=sha256,activation_exit=result.returncode,stage=str(stage),scientific_dispatch=0)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('archive');p.add_argument('sha256');a=p.parse_args();main(a.archive,a.sha256)
