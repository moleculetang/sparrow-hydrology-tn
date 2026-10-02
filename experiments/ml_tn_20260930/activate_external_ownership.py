"""Enable a checked pending-job handoff; no active training is interrupted."""
import json,os,hashlib
from pathlib import Path
from mltn.common import ROOT,write
p=ROOT/'transfer/external_ownership_next.json'
assert hashlib.sha256(p.read_bytes()).hexdigest()=='4fb297de4950b2e4bf7edb3d1cce7101b72fc1f84865e9ecc0060ed942f5061e'
old=json.loads((ROOT/'config/external_ownership.json').read_text());new=json.loads(p.read_text())
assert all(new['jobs'][k]==v for k,v in old['jobs'].items())
status=json.loads((ROOT/'outputs/controller_status.json').read_text())
assert status['phase']=='joint_screen_and_independent_historical_TN'
added=sorted(set(new['jobs'])-set(old['jobs']))
for jid in added:
    assert jid not in status['active']
    for name in ['owner.lock','result.json','failure.json']:
        assert not (ROOT/'jobs'/jid/name).exists(),(jid,name)
os.replace(p,ROOT/'config/external_ownership.json')
write(ROOT/'evidence/external_ownership_extension.json',dict(added=added,verified_undispatched=True,signals_sent=False,scientific_grid_unchanged=True))
print('OWNER_MAP_VERIFIED')
