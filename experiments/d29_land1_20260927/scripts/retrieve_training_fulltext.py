"""Bounded retrieval from OA locations already present in frozen metadata."""
import json,urllib.request,datetime,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'evidence/literature'
for index in [2,5,7]:
    work=json.loads((out/f'openalex_seed_{index}.json').read_text(encoding='utf-8'))['response']
    candidates=[x for x in work['locations'] if x.get('pdf_url')]
    # Prefer an explicit repository deposit when its metadata supplies a PDF.
    candidates.sort(key=lambda x:0 if 'uea.ac.uk' in x['pdf_url'] else 1)
    attempts=[]
    for location in candidates[:2]:
        url=location['pdf_url']
        try:
            request=urllib.request.Request(url,headers={'User-Agent':'D29ResearchEvidence/1.0'})
            with urllib.request.urlopen(request,timeout=25) as response:
                data=response.read(25*1024*1024+1);final=response.url
            if len(data)>25*1024*1024:raise ValueError('bounded document download exceeded')
            if not data.startswith(b'%PDF'):raise ValueError('not a PDF; no access-control workaround')
            file=out/f'seed_{index}_fulltext.pdf';file.write_bytes(data)
            attempts.append({'url':url,'resolved_url':final,'file':str(file),'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'downloaded':True,'reviewed':False});break
        except Exception as e:attempts.append({'url':url,'error':str(e),'downloaded':False})
    (out/f'seed_{index}_fulltext_retrieval.json').write_text(json.dumps({'doi':work['doi'],'retrieved':datetime.datetime.now(datetime.timezone.utc).isoformat(),'attempts':attempts},indent=2),encoding='utf-8')
    print(index,attempts,flush=True)
