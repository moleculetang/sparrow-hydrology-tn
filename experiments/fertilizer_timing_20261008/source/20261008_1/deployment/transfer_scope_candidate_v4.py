"""Direct authorized SSH transport and hash-checked independent staging only."""
import json
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'deployment')]
from wait_prerequisites import OPTIONS,REMOTE_PYTHON


def call(args, **kwargs):
    p=subprocess.run(args,text=True,encoding='utf8',capture_output=True,timeout=120,**kwargs)
    if p.returncode:
        raise RuntimeError(p.stdout+p.stderr)
    return p.stdout


def main():
    folder=ROOT/'outputs/maize_scope_revision_candidate/transport_v4'
    if (folder/'actual_remote_staging.json').exists():
        raise RuntimeError('EXISTING_ACTUAL_STAGING_REVIEW_REQUIRED')
    receipt=json.loads((folder/'prepared.json').read_text(encoding='utf8'))
    parent='/home/pc/TWY/SPARROW/Test/20261008_1/transfer/scope_candidate_v4'
    # Exclusive location; retries inspect partial output instead of overwriting.
    call(['ssh',*OPTIONS,'pc@10.5.8.38','mkdir '+parent])
    for source in [folder/'candidate_payload.tar',ROOT/'deployment/activate_scope_candidate_remote.py']:
        call(['scp',*OPTIONS,str(source),'pc@10.5.8.38:'+parent+'/'+source.name])
    command=' '.join([REMOTE_PYTHON,'-X utf8 -B',parent+'/activate_scope_candidate_remote.py',
        parent+'/candidate_payload.tar',receipt['archive_sha256'],receipt['manifest_sha256']])
    result=json.loads(call(['ssh',*OPTIONS,'pc@10.5.8.38',command]))
    if not result['passed'] or result['candidate_manifest_sha256']!=receipt['manifest_sha256']:
        raise RuntimeError('REMOTE_CANDIDATE_STAGING_RECEIPT_MISMATCH')
    (folder/'actual_remote_staging.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result))


if __name__=='__main__':main()
