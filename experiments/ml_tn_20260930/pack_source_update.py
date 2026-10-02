import sys,tarfile
from mltn.common import ROOT,sha,write
version=sys.argv[1];files=[*ROOT.glob('*.py'),*(ROOT/'mltn').glob('*.py'),ROOT/'README.md']
for name in ['annual_source_year_identity.parquet','deposition_year_mask_identity.parquet','deposition_unknown_support.parquet','input_semantics.json','hf_daily_availability.parquet','history_availability_identity.json']:
    p=ROOT/'data'/name
    if p.exists():files.append(p)
manifest=ROOT/'evidence'/f'source_update_{version}_manifest.json'
write(manifest,{p.relative_to(ROOT).as_posix():sha(p) for p in files})
dest=ROOT/'transfer'/f'source_update_{version}.tar.gz'
with tarfile.open(dest,'w:gz') as archive:
    for p in [*files,manifest]:archive.add(p,arcname=p.relative_to(ROOT).as_posix())
write(ROOT/'evidence'/f'source_update_{version}.json',dict(archive_sha256=sha(dest),files=len(files),bytes=dest.stat().st_size))
print(dest)
