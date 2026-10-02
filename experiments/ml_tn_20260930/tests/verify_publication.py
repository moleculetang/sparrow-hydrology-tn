"""Verify exact source, complete CSV shards, all published bytes and report links."""
from pathlib import Path
import ast,hashlib,json,re,csv,sys
ROOT=Path(__file__).resolve().parents[1];bad=[]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
m=json.loads((ROOT/'publication_manifest.json').read_text(encoding='utf-8'))
for x in m['files']:
 p=ROOT/x['path']
 if not p.exists() or sha(p)!=x['sha256']:bad.append(x['path'])
for p in ROOT.rglob('*.py'):ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p))
for p in ROOT.rglob('manifest.json'):
 obj=json.loads(p.read_text(encoding='utf-8'))
 if 'parts' not in obj:continue
 count=0
 for part in obj['parts']:
  q=ROOT/part['path'];rows=list(csv.reader(q.open(encoding='utf-8',newline='')))
  assert rows[0]==obj['columns'] and len(rows)-1==part['rows'] and sha(q)==part['sha256'];count+=len(rows)-1
 assert count==obj['rows']
for p in list((ROOT/'reports').glob('*.md'))+[ROOT/'README.md']:
 for href in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',p.read_text(encoding='utf-8')):
  if '://' not in href and not href.startswith('#') and not (p.parent/href).exists():bad.append(f'{p.name}:{href}')
assert not bad,bad
print(json.dumps(dict(passed=True,files=len(m['files']),source_files=len(m['exact_source']),failures=bad),ensure_ascii=False))
