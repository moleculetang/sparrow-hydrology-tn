"""Small public metadata/fulltext retrieval; no bulk datasets or credentials."""
import urllib.request,urllib.parse,json,time,hashlib,shutil,re
from pathlib import Path
R=Path(__file__).resolve().parents[1];OUT=R/'evidence/literature';OUT.mkdir(parents=True,exist_ok=True)
DOIS=['10.1016/0022-1694(95)02951-6','10.1029/97WR02171','10.5194/hess-6-559-2002','10.5194/hess-8-695-2004','10.2166/nh.2010.007','10.1016/j.jhydrol.2007.05.020','10.1029/2005WR004368','10.1029/2005WR004376','10.1029/2007WR006720','10.1002/hyp.14847','10.1029/2019WR024922','10.1038/s41467-021-26107-z','10.1029/2022WR032404','10.1029/2023WR036461','10.1137/1.9781611976700.69','10.1016/j.agwat.2013.08.003','10.1016/j.jhydrol.2022.127675']
QUERIES=['nitrogen water quality hybrid process model input uncertainty fertilizer timing','export coefficient nutrient load land use source uncertainty','synthetic data physics guided water quality input bias']
log=[]
def get(url,key):
    path=OUT/(key+'.json' if 'api.' in url else key+'.html')
    if path.exists():return path.read_bytes()
    try:
        req=urllib.request.Request(url,headers={'User-Agent':'SparrowResearch/1.0 (small scientific metadata audit)'})
        with urllib.request.urlopen(req,timeout=35) as r:data=r.read(8_000_000)
        path.write_bytes(data);log.append(dict(url=url,file=path.name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),status='retrieved',time=time.time()));return data
    except Exception as e:log.append(dict(url=url,status='unavailable',error=str(e),time=time.time()));return None
    finally:
        (OUT/'retrieval_log.json').write_text(json.dumps(log,ensure_ascii=False,indent=2),encoding='utf-8');time.sleep(2)
def main():
    records={}
    for doi in DOIS:
        key=hashlib.sha256(doi.encode()).hexdigest()[:16];raw=get('https://api.crossref.org/works/'+urllib.parse.quote(doi,safe=''),key)
        if raw:
            a=json.loads(raw)['message'];records[doi]=dict(doi=doi,title=a.get('title',[''])[0],authors=a.get('author',[]),year=a.get('published',{}).get('date-parts'),abstract=a.get('abstract'),type=a.get('type'),links=a.get('link',[]),evidence='abstract' if a.get('abstract') else 'metadata',file=key+'.json',selected_seed=True)
    for i,q in enumerate(QUERIES):
        raw=get('https://api.crossref.org/works?'+urllib.parse.urlencode({'query':q,'rows':15}),'crossref_query_'+str(i))
        if raw:
            for a in json.loads(raw)['message']['items']:
                doi=a.get('DOI','').lower()
                if doi and doi not in records and len(records)<80:records[doi]=dict(doi=doi,title=a.get('title',[''])[0],abstract=a.get('abstract'),type=a.get('type'),year=a.get('published',{}).get('date-parts'),evidence='abstract' if a.get('abstract') else 'metadata',query=q,selected_seed=False)
    get('https://api.openalex.org/works?'+urllib.parse.urlencode({'search':QUERIES[0],'per-page':10}),'openalex_search')
    for key,url in [('inca_full','https://hess.copernicus.org/articles/6/559/2002/'),('inca_snow_full','https://hess.copernicus.org/articles/8/695/2004/'),('mclstm','https://proceedings.mlr.press/v139/hoedt21a.html'),('dpl_full','https://www.nature.com/articles/s41467-021-26107-z')]:get(url,key)
    old=R.parent/'20260921_3/reports/literature_code_registry.json';shutil.copyfile(old,OUT/'inherited_code_registry.json')
    oldfiles=R.parent/'20260921_3/evidence/literature'
    if oldfiles.exists():shutil.copytree(oldfiles,OUT/'archived_small_evidence',dirs_exist_ok=True)
    (OUT/'screening_candidates.json').write_text(json.dumps(list(records.values()),ensure_ascii=False,indent=2),encoding='utf-8')
    print('LITERATURE RETRIEVED',len(records),'candidates; evidence levels require review',flush=True)
if __name__=='__main__':main()
