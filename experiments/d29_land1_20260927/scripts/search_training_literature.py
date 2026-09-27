"""Bounded Crossref/OpenAlex evidence retrieval; depth is not inferred from DOI."""
import json,sys,urllib.request,urllib.parse,concurrent.futures,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from d29_platform.runtime import write_json
OUT=ROOT/'evidence/literature';OUT.mkdir(exist_ok=True)
queries=['water quality nitrate model calibration high frequency monitoring','water quality multisite calibration long term nitrogen watershed','water quality sampling frequency calibration multi objective nutrient model']
seeds=['10.1016/j.ecolmodel.2006.12.037','10.1016/j.jhydrol.2019.124186','10.1016/j.jenvman.2016.05.002','10.1029/2010WR009525','10.1016/j.watres.2023.120347','10.1002/wat2.1155','10.1016/j.jhydrol.2013.12.036','10.13031/2013.25407']

def request(task):
    name,url=task
    try:
        with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'D29ResearchEvidence/1.0'}),timeout=25) as r:obj=json.load(r)
        write_json(OUT/(name+'.json'),{'url':url,'retrieved':datetime.datetime.now(datetime.timezone.utc).isoformat(),'response':obj})
        return name,obj
    except Exception as e:
        write_json(OUT/(name+'_error.json'),{'url':url,'error':str(e)});return name,None

tasks=[]
for i,q in enumerate(queries):
    tasks.extend([(f'crossref_query_{i}','https://api.crossref.org/works?'+urllib.parse.urlencode({'query.bibliographic':q,'rows':8})),(f'openalex_query_{i}','https://api.openalex.org/works?'+urllib.parse.urlencode({'search':q,'per-page':8}))])
for i,doi in enumerate(seeds):
    tasks.extend([(f'crossref_seed_{i}','https://api.crossref.org/works/'+urllib.parse.quote(doi,safe='')),(f'openalex_seed_{i}','https://api.openalex.org/works/https://doi.org/'+doi)])
results=list(concurrent.futures.ThreadPoolExecutor(max_workers=4).map(request,tasks))
merged={}
for name,result in results:
    if result is None:continue
    cr=name.startswith('crossref')
    items=result['message'].get('items',[result['message']]) if cr else result.get('results',[result])
    for w in items:
        doi=(w.get('DOI') if cr else w.get('doi') or '').lower().removeprefix('https://doi.org/')
        title=(w.get('title') or [''])[0] if cr else w.get('display_name','')
        key=doi or title.lower()
        if not key:continue
        a=merged.setdefault(key,{'doi':doi,'title':title,'sources':[],'abstracts':{},'fulltext_reviewed':False,'code_reviewed':False})
        a['sources'].append(name)
        if cr:
            a['year']=w.get('published',{}).get('date-parts',[[None]])[0][0]
            if w.get('abstract'):a['abstracts']['crossref']=w['abstract']
        else:
            a['year']=w.get('publication_year');inv=w.get('abstract_inverted_index')
            if inv:
                tokens={pos:word for word,positions in inv.items() for pos in positions}
                a['abstracts']['openalex']=' '.join(tokens[p] for p in sorted(tokens))
            a['open_access']=w.get('open_access');a['best_oa_location']=w.get('best_oa_location')
        a['depth']='abstract_available_not_yet_reviewed' if a['abstracts'] else 'metadata_only'
records=list(merged.values())
assert len(records)<=60
write_json(OUT/'deduplicated_candidates.json',{'maximum_screen':60,'candidate_count':len(records),'candidates':records,'queries':queries})
print('candidates',len(records),'successful_requests',sum(v is not None for _,v in results),flush=True)
