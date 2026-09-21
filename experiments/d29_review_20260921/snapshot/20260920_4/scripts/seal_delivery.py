"""Read-back verification and final immutable-content manifest."""
from pathlib import Path
import json,hashlib
R=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(16*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
for x in read(R/'data/input_manifest.json'):
 assert sha(Path(x['source']))==x['sha256'],x['source']
 if x['target'].startswith(('data/','vendor/')):assert sha(R/x['target'])==x['sha256'],x['target']
assert sha(R/'scripts/bypass_kernel.py')==read(R/'data/science_code_freeze.json')['bypass_kernel.py']
for arm,x in read(R/'data/prediction_freeze.json').items():
 assert sha(R/'outputs'/arm/'summary.json')==x['summary']
 assert sha(R/'outputs'/arm/'daily_station.parquet')==x['daily']
assert not list((R/'work').glob('*.lock'))
files={}
for p in sorted(R.rglob('*')):
 if not p.is_file() or p.name=='delivery_manifest.json' or '__pycache__' in p.parts or 'work' in p.relative_to(R).parts:continue
 files[p.relative_to(R).as_posix()]={'bytes':p.stat().st_size,'sha256':sha(p)}
target=R/'data/delivery_manifest.json';target.write_text(json.dumps(dict(status='SEALED',excluded=['work scratch','__pycache__','manifest itself'],files=files),ensure_ascii=False,indent=2),encoding='utf-8')
for name,x in read(target)['files'].items():assert sha(R/name)==x['sha256'],name
print('SEALED_AND_READBACK_VERIFIED',len(files),'files')
