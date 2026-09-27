"""Finite no-label pass over completed audited U paths; does not select results."""
import sys,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from d29_platform.runtime import configure,dispatch_allowed,sha
configure()

for j in json.loads((ROOT/'config/jobs.json').read_text(encoding='utf-8')):
    if j['model']!='U':continue
    folder=ROOT/'outputs/jobs'/j['id']
    if (folder/'prediction_freeze.json').exists():continue
    files=[folder/k for k in ['independent_audit.json','physical_ledger.json','status.json','best.npy']]
    if not all(p.exists() for p in files):continue
    audit=json.loads(files[0].read_text(encoding='utf-8'));ledger=json.loads(files[1].read_text(encoding='utf-8'))
    status=json.loads(files[2].read_text(encoding='utf-8'))
    if status['status'] in ('running','resource_checkpoint','continuing_same_path_zero_ftol'):continue
    if not audit['objective_passed'] or not ledger['passed']:raise RuntimeError('AUDIT_FAILED '+j['id'])
    digest=sha(files[3])
    if audit['checkpoint_sha256']!=digest or ledger['parameter_sha256']!=digest:raise RuntimeError('STALE_AUDIT '+j['id'])
    ok,_=dispatch_allowed(reserve_bytes=4_000_000_000)
    if not ok:
        print('RESOURCE_PAUSE',j['id'],flush=True);break
    with (folder/'freeze_console.log').open('ab') as f:
        subprocess.run([sys.executable,str(ROOT/'scripts/freeze_u_predictions.py'),j['id']],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    print(j['id'],'frozen; no evaluation labels loaded',flush=True)
