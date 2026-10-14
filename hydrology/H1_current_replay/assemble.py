"""Restore exact authorized H1 assets from bounded GitHub transport parts."""
from pathlib import Path,PurePosixPath
import json,hashlib,zipfile,io,argparse

def main():
 here=Path(__file__).resolve().parent
 m=json.loads((here/'delivery_manifest.json').read_text())
 for row in m.get('supplement_files',[]):
  rel=PurePosixPath(row['path'])
  if rel.is_absolute() or '..' in rel.parts or '\\' in row['path']:raise RuntimeError('Unsafe supplement path')
  content=here.joinpath(*rel.parts).read_bytes()
  if len(content)!=row['bytes'] or hashlib.sha256(content).hexdigest()!=row['sha256']:raise RuntimeError('Bad supplement '+row['path'])
 chunks=[]
 for row in m['parts']:
  b=(here/row['path']).read_bytes()
  if len(b)!=row['bytes'] or hashlib.sha256(b).hexdigest()!=row['sha256']:raise RuntimeError('Bad transport part '+row['path'])
  chunks.append(b)
 b=b''.join(chunks)
 if len(b)!=m['archive_bytes'] or hashlib.sha256(b).hexdigest()!=m['archive_sha256']:raise RuntimeError('Bad complete archive')
 root=here/'runtime'
 with zipfile.ZipFile(io.BytesIO(b)) as z:
  if z.testzip() is not None:raise RuntimeError('ZIP CRC failure')
  for name in z.namelist():
   rel=PurePosixPath(name)
   if rel.is_absolute() or '..' in rel.parts or '\\' in name:raise RuntimeError('Unsafe archive member')
   dst=root.joinpath(*rel.parts)
   content=z.read(name)
   if dst.exists() and dst.read_bytes()!=content:raise RuntimeError('Refusing overwrite '+str(dst))
   dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(content)
 for row in m['runtime_files']:
  content=(root/row['path']).read_bytes()
  if len(content)!=row['bytes'] or hashlib.sha256(content).hexdigest()!=row['sha256']:raise RuntimeError('Bad runtime file '+row['path'])
 print(json.dumps({'passed':True,'runtime':str(root),'files':len(m['runtime_files']),'reservoir_file':'runtime/data/tn_hydrology_reservoir_daily.parquet','reservoir_sha256':hashlib.sha256((root/'data/tn_hydrology_reservoir_daily.parquet').read_bytes()).hexdigest(),'archive_sha256':m['archive_sha256']},ensure_ascii=False))
if __name__=='__main__':main()
