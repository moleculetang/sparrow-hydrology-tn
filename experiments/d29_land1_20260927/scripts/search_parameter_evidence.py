"""Small public metadata searches; no credentials or private datasets."""
from pathlib import Path
import urllib.request, urllib.parse, json, hashlib, concurrent.futures, time
OUT=Path(__file__).resolve().parents[1]/'evidence/literature'
QUERIES={
 'son':'SWAT active stable organic nitrogen mineralization initial 2 percent humus',
 'resorption':'Global resorption efficiencies concentrations carbon nutrients leaves Vergutz 2012',
 'plant_cn':'CLM5 leaf fine root live wood dead wood carbon nitrogen ratio parameters',
 'residue':'China crop straw return fraction nitrogen residue management 2020',
 'manure':'China manure nitrogen recycling cropland fraction mineral nitrogen manure management',
 'bnf':'FAOSTAT cropland nutrient budget biological fixation crop coefficients 2024',
 'turnover':'HYPE soil nitrogen fast humus pool degradation mineralisation parameter',
 'soil_initial':'soil nitrogen stock global active stable soil organic nitrogen fraction initialization model',
}

def fetch(item):
 name,query=item
 url='https://api.openalex.org/works?'+urllib.parse.urlencode({'search':query,'per-page':6})
 try:
  with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'D29-parameter-review/1.0'}),timeout=25) as r: b=r.read(3_000_001)
  if len(b)>3_000_000:raise ValueError('METADATA_CAP')
  p=OUT/(name+'_openalex.json');p.write_bytes(b);d=json.loads(b)
  rows=[]
  for x in d['results']:
   inv=x.get('abstract_inverted_index') or {}; words={i:w for w,ix in inv.items() for i in ix}
   rows.append({'title':x['title'],'doi':x.get('doi'),'year':x.get('publication_year'),
    'abstract':' '.join(words[i] for i in sorted(words)), 'open_access':x.get('open_access'),
    'primary_location':x.get('primary_location')})
  return dict(query_id=name,query=query,url=url,status='ok',sha256=hashlib.sha256(b).hexdigest(),results=rows)
 except Exception as e:return dict(query_id=name,query=query,url=url,status='failed',error=str(e))

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(fetch,QUERIES.items()))
 (OUT/'parameter_search.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
 for r in rows:print(r['query_id'],r['status'],[(x['title'],x['doi']) for x in r.get('results',[])],flush=True)
if __name__=='__main__':main()
