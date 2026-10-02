"""Move only not-yet-admitted correction jobs; never interrupt leased training."""
import os,signal,time
from pathlib import Path
from mltn.common import ROOT,read,write
from mltn.resources import registry
mod,state=registry();leases=mod.status(state)['leases']
assert not any(x['project']=='20260930_1' for x in leases.values()),'ACTIVE_ML_LEASE_CANNOT_HANDOFF'
targets=[]
for p in Path('/proc').iterdir():
    if not p.name.isdigit() or int(p.name)==os.getpid():continue
    try:cmd=(p/'cmdline').read_bytes().replace(b'\x00',b' ').decode()
    except (FileNotFoundError,PermissionError):continue
    if 'python' not in cmd:continue
    if ('repair_joint_seed.py' in cmd and (p/'cwd').resolve()==ROOT) or ('joint.py' in cmd and str(ROOT) in cmd and '--family XGBoost' in cmd):targets.append((int(p.name),cmd))
# Parent first; then precisely identified unleased children.
targets.sort(key=lambda x:0 if 'repair_joint_seed.py' in x[1] else 1)
for pid,cmd in targets:os.kill(pid,signal.SIGTERM)
time.sleep(1)
assert all(not Path('/proc',str(pid)).exists() or b'Z' in Path('/proc',str(pid),'stat').read_bytes().split()[2:3] for pid,_ in targets)
write(ROOT/'evidence/seed_repair_handoff.json',dict(passed=True,terminated_only_unleased=targets,completed_preserved=[p.name for p in (ROOT/'jobs').glob('joint_screen_XGBoost*') if (p/'result.json').exists()],new_owner='Windows; same finite matrix',reason='remote shared commit headroom; do not interrupt graybox'))
