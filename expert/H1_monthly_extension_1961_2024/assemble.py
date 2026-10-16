from pathlib import Path,PurePosixPath
import hashlib,json,argparse,zipfile,shutil

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def safe(name):
    p=PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name:
        raise ValueError('UNSAFE_PATH '+name)
    return p

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path)
    parser.add_argument('--extract',action='store_true')
    a=parser.parse_args();here=Path(__file__).resolve().parent
    m=json.loads((here/'delivery_manifest.json').read_text(encoding='utf8'))
    fm=json.loads((here/'file_manifest.json').read_text(encoding='utf8'))
    out=a.output or here/'restored';out.mkdir(parents=True,exist_ok=True)
    dest=out/m['archive'];temp=out/(m['archive']+'.partial')
    with temp.open('wb') as f:
        for r in m['parts']:
            p=here.joinpath(*safe(r['path']).parts)
            if p.stat().st_size!=r['bytes'] or sha(p)!=r['sha256']:raise RuntimeError('PART_HASH_MISMATCH '+r['path'])
            with p.open('rb') as stream:shutil.copyfileobj(stream,f)
    if temp.stat().st_size!=m['bytes'] or sha(temp)!=m['sha256']:raise RuntimeError('ARCHIVE_HASH_MISMATCH')
    if dest.exists():
        if sha(dest)!=m['sha256']:raise RuntimeError('REFUSING_CONFLICTING_ARCHIVE')
        temp.unlink()
    else:temp.rename(dest)
    expected={r['path']:r for r in fm['files']}
    if len(expected)!=85 or len(fm['files'])!=85:raise RuntimeError('EXACT_85_SOURCE_FILES_REQUIRED')
    with zipfile.ZipFile(dest) as z:
        if len(z.infolist())!=86 or set(z.namelist())!=set(expected)|{'file_manifest.json'} or z.testzip():
            raise RuntimeError('ARCHIVE_SCOPE_OR_CRC_FAILED')
        if z.read('file_manifest.json')!=(here/'file_manifest.json').read_bytes():raise RuntimeError('FILE_MANIFEST_DIFFERS')
        for name in z.namelist():
            rel=safe(name);b=z.read(name)
            if name in expected:
                r=expected[name]
                if len(b)!=r['bytes'] or hashlib.sha256(b).hexdigest()!=r['sha256']:raise RuntimeError('SOURCE_BYTES_MISMATCH '+name)
            if a.extract:
                p=out.joinpath(*rel.parts)
                if p.exists() and p.read_bytes()!=b:raise RuntimeError('REFUSING_CONFLICTING_EXTRACTION '+name)
                p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
    print(json.dumps({'passed':True,'archive':str(dest),'sha256':m['sha256'],'data_files':85,'members':86,'extracted':a.extract}))

if __name__=='__main__':main()
