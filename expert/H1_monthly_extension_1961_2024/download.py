from pathlib import Path,PurePosixPath
from concurrent.futures import ThreadPoolExecutor
import argparse,hashlib,json,os,re,subprocess,sys,urllib.request

TRUSTED_FILES = {'assemble.py': '22f93b4fadd7aa8e19b194df1a13d8dac762ea46e77ff882f7d6fcd4f43ac62d', 'file_manifest.json': '1e395577846bcb9953d3c1d8e9808e29b8ba5f6348065b806a401867c615c844', 'delivery_manifest.json': '99d980f37bc7901399cc09a038e4dba55c5aca031a46011a4f9adcbba9e1f240'}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--ref',required=True)
    parser.add_argument('--output',type=Path,default=Path('H1_monthly_extension_1961_2024'))
    parser.add_argument('--extract',action='store_true')
    args=parser.parse_args()
    if re.fullmatch('[0-9a-f]{40}',args.ref) is None:raise ValueError('A_PINNED_40_HEX_COMMIT_IS_REQUIRED')
    out=args.output.resolve();cache=out/'.transport';cache.mkdir(parents=True,exist_ok=True)
    base='https://raw.githubusercontent.com/moleculetang/sparrow-hydrology-tn/'+args.ref+'/expert/H1_monthly_extension_1961_2024/'
    def fetch(rel,digest,size=None):
        p=PurePosixPath(rel)
        if p.is_absolute() or '..' in p.parts or '\\' in rel or ':' in rel:raise ValueError('UNSAFE_PATH')
        dest=cache.joinpath(*p.parts)
        if not dest.resolve().is_relative_to(cache.resolve()):raise ValueError('OUTSIDE_DOWNLOAD_CACHE')
        if dest.exists():
            if sha(dest)!=digest or (size is not None and dest.stat().st_size!=size):raise RuntimeError('CONFLICTING_EXISTING_FILE '+rel)
            return
        request=urllib.request.Request(base+rel,headers={'User-Agent':'SPARROW-H1-monthly-extension'})
        with urllib.request.urlopen(request,timeout=120) as response:data=response.read()
        if hashlib.sha256(data).hexdigest()!=digest or (size is not None and len(data)!=size):raise RuntimeError('DOWNLOAD_HASH_MISMATCH '+rel)
        dest.parent.mkdir(parents=True,exist_ok=True)
        temp=dest.with_name(dest.name+'.download');temp.write_bytes(data);os.replace(temp,dest)
    for rel,digest in TRUSTED_FILES.items():fetch(rel,digest)
    m=json.loads((cache/'delivery_manifest.json').read_text(encoding='utf8'))
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda r:fetch(r['path'],r['sha256'],r['bytes']),m['parts']))
    cmd=[sys.executable,'-B',str(cache/'assemble.py'),'--output',str(out)]
    if args.extract:cmd.append('--extract')
    subprocess.run(cmd,check=True)

if __name__=='__main__':main()
