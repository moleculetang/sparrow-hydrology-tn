"""Small, bounded public documentation downloads with source/byte receipts."""
from pathlib import Path
import urllib.request,json,hashlib,time
from html.parser import HTMLParser
OUT=Path(__file__).resolve().parents[1]/'evidence/literature'
class Plain(HTMLParser):
    def __init__(self):super().__init__();self.parts=[]
    def handle_data(self,x):
        if x.strip():self.parts.append(x.strip())
SOURCES={
 'era5_land_monthly_official':'https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land-monthly-means?tab=overview',
 'ctsm_phenology_pinned':'https://raw.githubusercontent.com/ESCOMP/CTSM/2ebc042c59649b7f97ac337723ffa4a5eacfae28/doc/source/tech_note/Vegetation_Phenology_Turnover/CLM50_Tech_Note_Vegetation_Phenology_Turnover.rst',
 'soilgrids_layers':'https://docs.isric.org/globaldata/soilgrids/SoilGrids_faqs_01.html',
 'soilgrids_official':'https://docs.isric.org/globaldata/soilgrids/SoilGrids_faqs.html',
 'mod17_official':'https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD17A3HGF',
 'faostat_budget_supplement':'https://essd.copernicus.org/articles/16/525/2024/essd-16-525-2024-supplement.pdf',
 'deposition_official':'https://input4mips-cvs.readthedocs.io/en/latest/dataset-overviews/nitrogen-deposition/',
}
def main():
    receipts=[]
    for name,url in SOURCES.items():
        suffix='.pdf' if url.endswith('.pdf') else '.html';p=OUT/(name+suffix)
        try:
            if not p.exists():
                req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0 D29 scientific source audit'})
                with urllib.request.urlopen(req,timeout=40) as r:b=r.read(12_000_001)
                if len(b)>12_000_000:raise ValueError('bounded_download_exceeded')
                p.write_bytes(b)
            b=p.read_bytes()
            if suffix=='.html':
                parser=Plain();parser.feed(b.decode('utf-8',errors='replace'))
                (OUT/(name+'_plain.txt')).write_text('\n'.join(parser.parts),encoding='utf-8')
            receipts.append(dict(name=name,url=url,path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b),status='retrieved'))
        except Exception as e:receipts.append(dict(name=name,url=url,status='failed',error=str(e)))
        print(name,receipts[-1]['status'],flush=True)
    (OUT/'official_definitions_receipt.json').write_text(json.dumps(receipts,indent=2),encoding='utf-8')
if __name__=='__main__':main()
