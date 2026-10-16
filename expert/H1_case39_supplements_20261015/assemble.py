"""Restore the two exact review supplements. No network or training is used."""
from pathlib import Path, PurePosixPath
import argparse, hashlib, json, zipfile

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(4*1024*1024), b''): h.update(b)
    return h.hexdigest()

def safe(name):
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name:
        raise ValueError('Unsafe member: '+name)
    return p

def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--package', choices=['all','case39','catboost'], default='all')
    parser.add_argument('--extract', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    out = args.output or here/'restored'
    out.mkdir(parents=True, exist_ok=True)
    m = json.loads((here/'delivery_manifest.json').read_text(encoding='utf8'))
    receipts = []
    for a in m['archives']:
        if args.package not in ('all', a['id']): continue
        dest = out/a['filename']
        temp = out/(a['filename']+'.partial')
        with temp.open('wb') as f:
            for part in a['parts']:
                p = here.joinpath(*safe(part['path']).parts)
                if p.stat().st_size != part['bytes'] or sha(p) != part['sha256']:
                    raise RuntimeError('Invalid part: '+part['path'])
                with p.open('rb') as source:
                    for b in iter(lambda: source.read(4*1024*1024), b''): f.write(b)
        if temp.stat().st_size != a['bytes'] or sha(temp) != a['sha256']:
            raise RuntimeError('Invalid archive: '+a['filename'])
        if dest.exists():
            if sha(dest) != a['sha256']: raise RuntimeError('Refusing conflicting output')
            temp.unlink()
        else: temp.rename(dest)
        with zipfile.ZipFile(dest) as z:
            if z.testzip(): raise RuntimeError('ZIP CRC failure')
            for info in z.infolist():
                rel = safe(info.filename)
                if (info.external_attr >> 16) & 0o170000 == 0o120000:
                    raise RuntimeError('Symlink archive member')
                if args.extract and not info.is_dir():
                    target = out.joinpath(*rel.parts)
                    data = z.read(info)
                    if target.exists() and target.read_bytes() != data:
                        raise RuntimeError('Refusing conflicting extraction: '+str(target))
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(data)
        receipts.append({'file':str(dest),'sha256':a['sha256'],'passed':True})
    print(json.dumps({'archives':receipts,'daily_deposition_production_allowed':False},ensure_ascii=False))

if __name__ == '__main__': main()
