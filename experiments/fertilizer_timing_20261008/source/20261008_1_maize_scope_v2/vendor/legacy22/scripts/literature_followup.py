import json,hashlib,urllib.request,urllib.parse,time,re,html
from pathlib import Path
R=Path(__file__).resolve().parents[1];P=R/'evidence/literature'
def main():
    records=json.loads((P/'screening_candidates.json').read_text(encoding='utf-8'));log=[]
    urls={
      'inca_paper.pdf':'https://hess.copernicus.org/articles/6/559/2002/hess-6-559-2002.pdf',
      'inca_snow_paper.pdf':'https://hess.copernicus.org/articles/8/695/2004/hess-8-695-2004.pdf',
      'mclstm_paper.pdf':'https://proceedings.mlr.press/v139/hoedt21a/hoedt21a.pdf',
      'hype_documentation.html':'https://hypeweb.smhi.se/model-water/'}
    for a in records:
        if a.get('selected_seed') and not a.get('abstract'):urls['openalex_'+hashlib.sha256(a['doi'].encode()).hexdigest()[:16]+'.json']='https://api.openalex.org/works/https://doi.org/'+a['doi']
    for name,url in urls.items():
        try:
            req=urllib.request.Request(url,headers={'User-Agent':'SparrowResearch/1.0'})
            with urllib.request.urlopen(req,timeout=35) as f:data=f.read(10_000_000)
            (P/name).write_bytes(data);log.append(dict(url=url,file=name,sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),status='retrieved'))
        except Exception as e:log.append(dict(url=url,status='unavailable',error=str(e)))
        (P/'followup_retrieval.json').write_text(json.dumps(log,indent=2),encoding='utf-8');time.sleep(2)
    from html.parser import HTMLParser
    class Text(HTMLParser):
        def __init__(self):super().__init__();self.parts=[];self.skip=0
        def handle_starttag(self,t,attrs):
            if t in ['script','style']:self.skip+=1
        def handle_endtag(self,t):
            if t in ['script','style']:self.skip=max(0,self.skip-1)
        def handle_data(self,d):
            if not self.skip and d.strip():self.parts.append(d.strip())
    for f in P.glob('*.html'):
        t=Text();t.feed(f.read_text(encoding='utf-8',errors='replace'));f.with_suffix('.txt').write_text('\n'.join(t.parts),encoding='utf-8')
    print('FOLLOWUP RETRIEVAL',len(log),flush=True)
if __name__=='__main__':main()
