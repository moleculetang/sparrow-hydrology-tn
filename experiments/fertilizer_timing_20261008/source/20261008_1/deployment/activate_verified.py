"""Verify private extracted science and H1 before first remote activation.

Never overwrites an existing scientific directory, never acquires resources or
dispatches workers. The root's existing control/pause provenance is preserved.
"""
import argparse
import datetime
import hashlib
import json
import os
import sys
from pathlib import Path


def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()


def main(stage,destination,h1):
    stage=Path(stage).resolve();destination=Path(destination).resolve();h1=Path(h1).resolve()
    expected=Path('/home/pc/TWY/SPARROW/Test/20261008_1').resolve()
    if destination!=expected or stage.parent!=destination:raise RuntimeError('EXPECTED_NEW_EXPERIMENT_STAGING_ONLY')
    manifest=stage/'deployment/source_input_manifest.json';record=json.loads(manifest.read_text(encoding='utf8'))
    for row in record['files']:
        path=(stage/row['path']).resolve()
        if not path.is_relative_to(stage) or path.stat().st_size!=row['bytes'] or digest(path)!=row['sha256']:
            raise RuntimeError('REMOTE_SOURCE_HASH_FAILURE '+row['path'])
    h1_verified=[]
    for name,spec in record['h1_readonly_array_contract'].items():
        path=(h1/spec['file']).resolve()
        if not path.is_relative_to(h1) or digest(path)!=spec['sha256']:raise RuntimeError('REMOTE_H1_ARRAY_HASH_FAILURE '+name)
        h1_verified.append(dict(name=name,sha256=spec['sha256'],bytes=path.stat().st_size))
    directories={Path(row['path']).parts[0] for row in record['files']}|{'deployment'}
    for name in directories:
        if (destination/name).exists():raise RuntimeError('ACTIVATION_WOULD_OVERWRITE_EXISTING_DIRECTORY '+name)
    verified=dict(passed=True,scope='remote_source_input_H1_byte_identity_before_science',
        files=len(record['files']),manifest_sha256=digest(manifest),h1_verified=h1_verified,
        destination=str(destination),h1_root=str(h1),scientific_jobs_dispatched=0,
        registered_host_resource_state='/home/pc/TWY/SPARROW/Test/.compute_coordination/state',
        verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    # No scientific process is admitted until every byte is verified. Activation
    # cannot overwrite existing owner/checkpoint directories.
    for name in sorted(directories):os.rename(stage/name,destination/name)
    (destination/'evidence').mkdir(exist_ok=True)
    (destination/'evidence/remote_activation.json').write_text(json.dumps(verified,indent=2),encoding='utf8')
    print(json.dumps(dict(passed=True,files=verified['files'],h1_arrays=len(h1_verified),dispatch=0)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage');p.add_argument('destination');p.add_argument('h1');a=p.parse_args()
    main(a.stage,a.destination,a.h1)
