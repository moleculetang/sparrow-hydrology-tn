"""Finite read-only training audits; no selection using held-out outcomes."""
import sys,json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from d29_platform.runtime import configure,dispatch_allowed,write_json
configure()
jobs=json.loads((ROOT/'config/jobs.json').read_text(encoding='utf-8'))
for j in jobs:
    if j['model']!='U':continue
    folder=ROOT/'outputs/jobs'/j['id'];status=folder/'status.json'
    if not status.exists():continue
    s=json.loads(status.read_text(encoding='utf-8'))
    if s['status']!='solver_stopped_numerically_sufficient':continue
    for script,receipt in [('audit_u_solution.py','independent_audit.json'),('verify_u_ledger.py','physical_ledger.json'),('diagnose_training_blocks.py','training_block_gradients.json')]:
        if (folder/receipt).exists():continue
        ok,res=dispatch_allowed(reserve_bytes=4_000_000_000)
        if not ok:
            write_json(ROOT/'outputs/audit_queue_pause.json',{'job':j['id'],'script':script,'resources':res});raise SystemExit(0)
        with (folder/'audit_console.log').open('ab') as f:
            subprocess.run([sys.executable,str(ROOT/'scripts'/script),j['id']],cwd=str(ROOT),stdout=f,stderr=subprocess.STDOUT,check=True)
        print(j['id'],receipt,flush=True)
