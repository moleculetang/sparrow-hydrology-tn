"""Build the authorized real-data review subset; archived experiments are read-only."""
import os,sys,json,hashlib,ast,copy,shutil
from pathlib import Path
BASE=Path(__file__).resolve().parent
ROOT=Path('E:/SPARROW'); OUT=BASE/'repository/expert/tn_challenge'
os.environ['NUMBA_CACHE_DIR']=str(BASE/'work/numba')
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[k]='1'
sys.path.insert(0,str(ROOT/'5_Test/20260911_1/scripts'))
import numpy as np,pandas as pd,torch
from sc_model import data_cache
from fc_io import require_environment
require_environment();torch.set_num_threads(1)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,obj):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
d=data_cache('sensitivity');term={}
for r in reversed(d.order):term[r]=term[d.downstream[r]] if r in d.downstream else r
rr=[r for r in d.order if term[r]+1 in (56,22)];ix={r:i for i,r in enumerate(rr)}
assert all((s in ix)==(t in ix) for s,t in d.downstream.items())
kk=[k for k,m in enumerate(d.metadata) if m['target'] in ix]
assert all(all(c in ix for c in d.metadata[k]['controls']) for k in kk)
rk={k:i for i,k in enumerate(kk)};nd=int((d.dates.year<=2024).sum());nm=int((d.months.year<=2024).sum())
(OUT/'data').mkdir(parents=True,exist_ok=True)
daily=['contact','fast_fraction','lower_release','fast_water','slow_water','official_water','h_day','temperature','upper_water','percolation','soil_wetness']
monthly=['source','crop','h_month','source_tags']
arrays={k:np.ascontiguousarray(getattr(d,k)[:nd,rr]) for k in daily}
arrays.update({k:np.ascontiguousarray(getattr(d,k)[:nm,rr]) for k in monthly})
arrays.update({k:np.ascontiguousarray(getattr(d,k)[:nd,kk]) for k in ['release_fraction','released_water','enabled']})
arrays.update(static_raw=d.static_raw[rr],area_ha=d.area_ha[rr],mid=d.mid[:nd],starts=d.starts[:nm],stops=d.stops[:nm],
              dates=d.dates[:nd].to_numpy('datetime64[D]').astype('int64'),months=d.months[:nm].to_numpy('datetime64[D]').astype('int64'))
# Separate arrays keep individual Git blobs small and permit transparent inspection.
for k,a in arrays.items():np.savez_compressed(OUT/'data'/f'{k}.npz',values=a)
metadata=copy.deepcopy([d.metadata[k] for k in kk])
for m in metadata:m.update(controls=[ix[c] for c in m['controls']],target=ix[m['target']],position=max(ix[c] for c in m['controls']))
top={'product':d.product,'forcing':d.forcing,'global_reach_ids':[r+1 for r in rr], 'terminal_global_ids':[term[r]+1 for r in rr],
 'order':list(range(len(rr))),'downstream':{ix[s]:ix[t] for s,t in d.downstream.items() if s in ix},
 'terminal':[ix[r] for r in d.terminal if r in ix],'metadata':metadata,'static_fields':d.static_fields,
 'date_encoding':'integer days since 1970-01-01','history_start':'1961-01-01','history_end':'2024-12-31','array_files':list(arrays)}
write(OUT/'data/topology.json',top)
obsfile=ROOT/'5_Test/20260906_1/outputs/observations.parquet';obs=pd.read_parquet(obsfile)
o=obs.loc[obs.reach_id.isin(np.array(rr)+1)&obs.year.between(2016,2024)].copy()
o['global_reach_id']=o.reach_id;o['reach_id']=o.reach_id.map(lambda r:ix[int(r)-1]+1)
o['global_reservoir_index']=o.reservoir_index;o['reservoir_index']=o.reservoir_index.map(lambda k:rk.get(int(k),-1))
o['basin']=o.global_reach_id.map(lambda r:'Beijiang' if term[int(r)-1]+1==56 else 'Dongjiang')
o=o.sort_values(['station_key','year','month','observation_id']).reset_index(drop=True)
o.to_csv(OUT/'data/observations_all.csv',index=False,encoding='utf-8')
for name,years in [('train',[2021,2022]),('development',[2023]),('evaluation',[2024]),('hindcast',list(range(2016,2021)))]:
    q=o.loc[o.primary_gate&o.year.isin(years)];q.to_csv(OUT/'data'/f'{name}.csv',index=False,encoding='utf-8')
    q.drop(columns=['tn_mg_l']).to_csv(OUT/'data'/f'{name}_metadata.csv',index=False,encoding='utf-8')
coverage=o.groupby(['basin','station_key','year']).agg(rows=('tn_mg_l','size'),months=('month','nunique'),minimum=('tn_mg_l','min'),maximum=('tn_mg_l','max')).reset_index()
coverage['missing_months_of_12']=12-coverage.months;coverage.to_csv(OUT/'data/coverage.csv',index=False,encoding='utf-8')
quality={'reaches':len(rr),'stations_all':o.station_key.nunique(),'reservoirs':len(kk),'rows_all':len(o),'rows_primary':int(o.primary_gate.sum()),
 'duplicate_ids':int(o.observation_id.duplicated().sum()),'duplicate_station_months':int(o.duplicated(['station_key','year','month']).sum()),
 'missing_TN':int(o.tn_mg_l.isna().sum()),'negative_TN':int((o.tn_mg_l<0).sum()),'TN_range':o.tn_mg_l.agg(['min','max']).to_dict(),
 'primary_exclusions':o.primary_exclusion_reason.fillna('none').value_counts().to_dict(),
 'provenance_tiers':o.source_provenance_tier.value_counts(dropna=False).to_dict(),
 'workbook_link_status':o.raw_workbook_record_link_status.value_counts(dropna=False).to_dict(),
 'sampling_date_status':o.sampling_date_status.value_counts(dropna=False).to_dict(),
 'yearly':o.groupby('year').agg(rows=('tn_mg_l','size'),stations=('station_key','nunique')).reset_index().to_dict('records'),
 'selection':'Two complete small terminal components selected for topology, train coverage and reservoir diversity, not heldout gains; not a probability sample.',
 'topology_closed':True,'all_arrays_finite':all(np.isfinite(a).all() for a in arrays.values()),
 'no_correction':'No label removed/downweighted/corrected by this export. primary_gate is inherited and retained.'}
write(OUT/'data/quality.json',quality)
sources=[ROOT/'5_Test/20260907_3/scripts/tn_reference.py',ROOT/'5_Test/20260907_3/scripts/tn_autograd.py']
def extract(path,names):
    text=path.read_text(encoding='utf-8');tree=ast.parse(text);lines=text.splitlines(True);out=[]
    for node in tree.body:
        if getattr(node,'name',None) in names:
            start=min([node.lineno]+[q.lineno for q in node.decorator_list]);out.append(''.join(lines[start-1:node.end_lineno]))
    return '\n\n'.join(out)
ops='"""Extracted original routing and adjoints; only reach dimension is generalized."""\nimport numpy as np\nimport torch\nfrom numba import njit\n\n'
ops+=extract(sources[0],['reservoir_scan','route'])+'\n\n'+extract(sources[1],['reservoir_backward','RiverN','boundary_mass','monthly_sum'])
ops=ops.replace('len(data.months),230','len(data.months),data.source.shape[1]')
(OUT/'routing.py').write_text(ops,encoding='utf-8')
kernel=ROOT/'5_Test/20260911_1/scripts/sc_kernel.py';shutil.copyfile(kernel,OUT/'sc_kernel.py')
write(OUT/'data/provenance.json',{'observation_input':{'path':str(obsfile),'sha256':sha(obsfile)},
 'cache_manifest':json.loads((ROOT/'5_Test/20260911_1/reports/cache_manifest.json').read_text(encoding='utf-8'))['products']['sensitivity']['data'],
 'original_code':[{ 'path':str(p),'sha256':sha(p)} for p in sources+[kernel]],
 'export_script_sha256':sha(__file__),'selection_terminals':[56,22], 'no_fitted_states':True,
 'source_timing':'S1 monthly source and crop demand pulses on first day, unchanged',
 'adaptations':['numeric NPZ instead of private pickle','local reach/reservoir indices','routing adjoint dimensions inferred from data','standalone parameter/objective adapter'],
 'data_authorization':'User explicitly authorized representative real data and quantitative reports for PR #2 expert fitting review.'})
write(OUT/'data/manifest.json',{p.name:sha(p) for p in sorted((OUT/'data').iterdir()) if p.name!='manifest.json'})
print(json.dumps(quality,ensure_ascii=False));print('Total data MB',sum(p.stat().st_size for p in (OUT/'data').iterdir())/1e6)
