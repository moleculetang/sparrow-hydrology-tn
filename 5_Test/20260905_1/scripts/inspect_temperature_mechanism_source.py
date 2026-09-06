"""Retrieve public SWAT+ process source for a conditional mechanism review."""
from common import ROOT,RUNTIME,atomic_json,sha256,utc_now
from urllib.request import Request,urlopen
import json
import base64


def get(url):
    request=Request(url,headers={'User-Agent':'SPARROW-research-audit','Accept':'application/vnd.github+json'})
    with urlopen(request,timeout=25) as response:return json.load(response)


def main():
    run=ROOT/'5_Test/20260905_1';base='https://api.github.com/repos/swat-model/swatplus'
    repo=get(base);branch=repo['default_branch'];tree=get(base+'/git/trees/'+branch+'?recursive=1')
    if tree.get('truncated'):raise RuntimeError('Incomplete repository tree')
    candidates=[r for r in tree['tree'] if r['type']=='blob' and r['path'].endswith('.f90') and any(k in r['path'].lower() for k in ['denit','nminrl','nitrogen','nutrient','nitrif'])]
    selected=sorted(candidates,key=lambda r:(0 if 'denit' in r['path'].lower() else 1,r['path']))[:6]
    evidence=[]
    for entry in selected:
        blob=get(base+'/git/blobs/'+entry['sha'])
        raw=base64.b64decode(blob['content']);text=raw.decode('utf-8',errors='replace')
        path=run/'sources'/'swatplus'/entry['path'];path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
        hits=[dict(line=i+1,text=line.strip()) for i,line in enumerate(text.splitlines()) if any(k in line.lower() for k in ['denit','temperature','tmp','temp','nitrate','nitrogen'])]
        evidence.append(dict(repository_path=entry['path'],blob_sha=entry['sha'],local_path=str(path),sha256=sha256(path),
            source_url=f"https://github.com/swat-model/swatplus/blob/{tree['sha']}/{entry['path']}",hits=hits[:80]))
    result=dict(status='PUBLIC_SOURCE_RETRIEVED_REVIEW_REQUIRED',runtime=RUNTIME,created_utc=utc_now(),repository='swat-model/swatplus',
        tree_sha=tree['sha'],candidates=[r['path'] for r in candidates],files=evidence,
        caution='Nitrate transformation/removal code is not automatically a valid total-N sink; inspect reservoir/speciation and gaseous-loss semantics')
    atomic_json(result,run/'reports/temperature_mechanism_public_source.json')
    print('MECHANISM_SOURCE',json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
