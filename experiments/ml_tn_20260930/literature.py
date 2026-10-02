"""Bounded public-source evidence search, no claims of reading unavailable full text."""
import urllib.request,urllib.parse,json,time,re,xml.etree.ElementTree as ET
from mltn.common import ROOT,write
OUT=ROOT/'literature';OUT.mkdir(exist_ok=True)
def get(url):
    req=urllib.request.Request(url,headers={'User-Agent':'SPARROW-research/1.0 (bounded evidence audit)'})
    with urllib.request.urlopen(req,timeout=45) as f:return f.read()
def clean(s):return re.sub('<[^>]+>',' ',s or '').strip()
def main():
    queries=['river nitrate LSTM prediction','river total nitrogen machine learning prediction','water quality XGBoost temporal spatial validation','water quality graph neural network forecasting','water quality random forest deep learning forecasting']
    records={};errors=[]
    for i,q in enumerate(queries):
        url='https://api.crossref.org/works?'+urllib.parse.urlencode({'query':q,'rows':20,'select':'DOI,title,abstract,published,container-title,link'})
        try:
            raw=get(url);(OUT/f'crossref_search_{i}.json').write_bytes(raw)
            for r in json.loads(raw)['message']['items']:
                doi=r['DOI'].lower();records.setdefault(doi,dict(doi=doi,title=' '.join(r.get('title',[])),abstract=clean(r.get('abstract')),year=r.get('published',{}).get('date-parts',[[None]])[0][0],journal=' '.join(r.get('container-title',[])),source='Crossref',url='https://doi.org/'+doi))
        except Exception as e:errors.append(dict(source='Crossref',query=q,error=str(e)))
        time.sleep(1)
    for i,q in enumerate(['(nitrate OR "total nitrogen") AND ("machine learning" OR LSTM) AND (river OR stream)','"water quality" AND (forecast OR prediction) AND (LSTM OR XGBoost OR "graph neural")']):
        url='https://www.ebi.ac.uk/europepmc/webservices/rest/search?'+urllib.parse.urlencode({'query':q,'format':'json','resultType':'core','pageSize':20})
        try:
            raw=get(url);(OUT/f'europepmc_search_{i}.json').write_bytes(raw)
            for r in json.loads(raw)['resultList']['result']:
                key=r.get('doi',r.get('id')).lower()
                records[key]=dict(doi=r.get('doi'),pmcid=r.get('pmcid'),title=r.get('title'),abstract=clean(r.get('abstractText')),year=r.get('pubYear'),journal=r.get('journalInfo',{}).get('journal',{}).get('title'),source='EuropePMC core',url='https://doi.org/'+r['doi'] if r.get('doi') else 'https://europepmc.org/article/MED/'+r['id'])
        except Exception as e:errors.append(dict(source='EuropePMC',query=q,error=str(e)))
        time.sleep(1)
    for doi in ['10.1016/j.wroa.2023.100207','10.1029/2024WR039207','10.3389/frwa.2024.1456647','10.3390/w17081131','10.2166/ws.2023.164','10.1016/j.scitotenv.2023.162930']:
        if doi in records:continue
        try:
            r=json.loads(get('https://api.crossref.org/works/'+doi))['message'];records[doi]=dict(doi=doi,title=' '.join(r.get('title',[])),abstract=clean(r.get('abstract')),year=r.get('published',{}).get('date-parts',[[None]])[0][0],source='Crossref DOI',url='https://doi.org/'+doi)
        except Exception as e:errors.append(dict(doi=doi,error=str(e)))
    rows=list(records.values())[:150];write(OUT/'screened_unique.json',rows);write(OUT/'search_receipt.json',dict(unique_screened=len(rows),cap=150,errors=errors,mcp_available=False,openalex='previous 500/503; public Crossref and EuropePMC fallback',queries=queries,full_text_claim='only explicitly downloaded fullTextXML'))
    candidates=[r for r in rows if r.get('abstract') and any(w in ((r.get('title') or '')+' '+r['abstract']).lower() for w in ['nitrate','nitrogen','water quality'])]
    candidates.sort(key=lambda r:sum(w in (r.get('title') or '').lower() for w in ['river','nitrate','nitrogen','lstm','prediction']),reverse=True)
    detail=[]
    for i,r in enumerate(candidates[:32]):
        r=dict(r);r['evidence_depth']='abstract';r['full_text_available']=False
        if r.get('pmcid') and i<15:
            try:
                raw=get(f'https://www.ebi.ac.uk/europepmc/webservices/rest/{r["pmcid"]}/fullTextXML');(OUT/f'{r["pmcid"]}.xml').write_bytes(raw)
                root=ET.fromstring(raw);r['full_text_available']=True;r['full_text_file']=f'{r["pmcid"]}.xml';r['evidence_depth']='full text retrieved, structured audit pending'
                body=' '.join(root.find('body').itertext()) if root.find('body') is not None else ''
                (OUT/f'{r["pmcid"]}_body.txt').write_text(body,encoding='utf-8')
            except Exception as e:r['full_text_error']=str(e)
        r['extraction_status']='requires human/model paragraph-level interpretation; no invented dataset size or split'
        detail.append(r);time.sleep(.3)
    for pmc in ['PMC10719578','PMC12425153']:
        if not (OUT/f'{pmc}.xml').exists():
            try:(OUT/f'{pmc}.xml').write_bytes(get(f'https://www.ebi.ac.uk/europepmc/webservices/rest/{pmc}/fullTextXML'))
            except Exception as e:errors.append(dict(pmc=pmc,error=str(e)))
    write(OUT/'detailed_candidates.json',detail)
if __name__=='__main__':main()
