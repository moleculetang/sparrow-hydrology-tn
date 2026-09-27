"""Freeze input products and finite jobs before fitting or evaluation."""
import json,hashlib,ast
from pathlib import Path
R=Path(__file__).resolve().parents[1];OLD=R.parent/'20260921_3'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def patch(name,old,new):
    p=R/'scripts'/name;s=p.read_text(encoding='utf-8');assert old in s,(name,old);p.write_text(s.replace(old,new),encoding='utf-8')
def main():
    originals={str(p.relative_to(OLD)):sha(p) for area in ['scripts','vendor','data'] for p in (OLD/area).rglob('*') if p.is_file() and p.suffix not in ['.pyc','.nbc','.nbi']}
    write(R/'evidence/inherited_file_hashes.json',originals)
    patch('campaign_model.py'," if kind.startswith('REGIONAL_'):"," if 'daily_input' in design:\n  from daily_inputs import make_daily\n  m=make_daily(data,train,kind,design)\n elif kind.startswith('REGIONAL_'):")
    patch('campaign_model.py'," return make_model(d,train,job['kind'],design)"," design['daily_input']=json.loads((RUN/'data/daily_inputs/registry.json').read_text())[job.get('input_mode','P')]\n return make_model(d,train,job['kind'],design)")
    p=R/'scripts/balanced_tags.py';s=p.read_text();s=s.replace('range(4)','range(len(weights))',2)
    s=s.replace('nd,nr=h.shape','nd,nr=h.shape;ns=inputs.shape[-1]').replace('(nr,4)','(nr,ns)').replace('np.zeros(4)','np.zeros(ns)').replace('range(4)','range(ns)').replace('(.25)','(1./ns)').replace('else .25','else 1./ns');p.write_text(s)
    patch('audit_job.py',"rr=a['source_label_global_reaches'];rows=pd.DataFrame(dict(year=np.repeat(d.months.year,len(rr)*4),month=np.repeat(d.months.month,len(rr)*4),global_reach_id=np.tile(np.repeat(rr,4),len(d.months)),source=np.tile(['fertilizer','manure','BNF','deposition'],len(d.months)*len(rr))))", "rr=a['source_label_global_reaches'];names=a.get('source_label_names',['fertilizer','manure','BNF','deposition']);ns=len(names);rows=pd.DataFrame(dict(year=np.repeat(d.months.year,len(rr)*ns),month=np.repeat(d.months.month,len(rr)*ns),global_reach_id=np.tile(np.repeat(rr,ns),len(d.months)),source=np.tile(names,len(d.months)*len(rr))))")
    patch('controller.py','scripts/finalize_regional.py','scripts/finalize_daily.py')
    # parent_tags are warm starts only, recomputed under the candidate input.
    cfg=read(R/'configs/campaign.json');cfg.update(models=['SOURCE_UNIFIED','REGIONAL_L3'],inputs=['P','D','A'],paths=24,budget=dict(campaign_hours=24,preparation_hours=6,fit_hours=12,report_hours=6));cfg.pop('candidate',None);write(R/'configs/campaign.json',cfg)
    folds=read(R/'configs/folds.json');folds={f:folds[f] for f in ['F23_G_D','T24_G_D_H1']};write(R/'configs/folds.json',folds)
    jobs=[]
    for short,fold in [('F23','F23_G_D'),('F24','T24_G_D_H1')]:
        for structure,kind in [('U','SOURCE_UNIFIED'),('L3','REGIONAL_L3')]:
            parents=[f'{short}_{structure}_P_s{i}' for i in range(2)]
            for mode in ['P','D','A']:
                for start in range(2):
                    j=dict(tag=f'{short}_{structure}_{mode}_s{start}',fold=fold,kind=kind,input_mode=mode,start=start,operator_id='OU',mapping_id='G1',observation_operator='MATCH',objective_id='D',dependencies=[] if mode=='P' else parents,priority=0 if mode=='P' or start==0 else 1)
                    if mode=='P':j['reuse_from']=str(OLD/'outputs'/f'{short}_{structure}_s{start}')
                    elif start==0:j['parent_tags']=parents
                    jobs.append(j)
    write(R/'configs/jobs.json',jobs)
    from campaign_model import load_data
    import numpy as np,pandas as pd
    d=load_data('FULL24C');root=R/'data/daily_inputs';root.mkdir(exist_ok=True)
    activity=R.parent/'20260907_3/outputs/crop_specific_annual_n.parquet'
    a=pd.read_parquet(activity,columns=['year','reach_id','harvested_overlap_ha','harvested_area_year_used'])
    area=a.groupby(['year','reach_id']).harvested_overlap_ha.sum().unstack().reindex(index=range(1961,2025),columns=d.global_reach_ids)
    assert area.notna().all().all() and (area>=0).all().all()
    H=area.to_numpy();raw=np.array(d.source_tags);ag=raw[:,:,:3].sum(-1);annual=ag.reshape(64,12,-1).sum(1)
    assert not ((H>0)&(annual==0)).any(),'POSITIVE_AREA_WITHOUT_CALENDAR'
    assert not ((H==0)&(annual>0)).any(),'AGRICULTURE_WITHOUT_AREA'
    kappa=float(annual[:60].sum()/H[:60].sum())
    p=np.divide(ag.reshape(64,12,-1),annual[:,None,:],out=np.zeros((64,12,len(d.area_ha))),where=annual[:,None,:]>0)
    act=np.stack([(kappa*H[:,None,:]*p).reshape(ag.shape),raw[:,:,3]],-1)
    registry={};checks=[]
    for mode,monthly,names in [('P',raw,['fertilizer','manure','BNF','deposition']),('D',raw,['fertilizer','manure','BNF','deposition']),('A',act,['activity_proxy_agricultural_N','deposition'])]:
        if mode=='P':
            tags=np.zeros((*d.contact.shape,len(names)));tags[d.starts]=monthly
            total=np.zeros(d.contact.shape);total[d.starts]=d.source
        else:
            tags=monthly[d.mid]/(d.stops-d.starts)[d.mid,None,None]
            total=tags.sum(-1)
        closure=float(np.max(abs(d.monthly_sum(tags)-monthly)))
        assert closure<=1e-6 and np.isfinite(tags).all() and tags.min()>=0,(mode,closure)
        spec=dict(mode=mode,source_names=names,units='kg N/day',demand_calendar='original_month_start',dates_sha256=sha(R/'data/domains/FULL24C/arrays.json'))
        for key,value in [('total',total),('tags',tags)]:
            path=root/f'{mode}_{key}.npy';np.save(path,value);spec[key]=path.relative_to(R).as_posix();spec[key+'_sha256']=sha(path)
        np.save(root/f'{mode}_monthly_tags.npy',monthly);registry[mode]=spec;checks.append(dict(mode=mode,monthly_closure_kg=closure,total_kg=float(total.sum())))
    area.to_parquet(root/'harvested_area_ha.parquet');a.groupby(['year','harvested_area_year_used']).size().rename('rows').reset_index().to_csv(root/'area_years.csv',index=False)
    write(root/'activity_reference.json',dict(kappa0_kg_N_ha_year=kappa,reference=[1961,2020],agricultural_mass_kg=float(annual[:60].sum()),harvested_area_ha_year=float(H[:60].sum()),activity_file=str(activity),activity_sha256=sha(activity),dependence='Original monthly calendar and 1961-2020 agricultural mass; not independent new data'))
    write(root/'registry.json',registry);write(R/'reports/input_preparation.json',dict(status='PASS',checks=checks,shape=list(d.contact.shape),activity_reference=kappa))
    for f in (R/'scripts').glob('*.py'):ast.parse(f.read_text(encoding='utf-8'))
    print('PASS PREPARATION',checks,'kappa',kappa,flush=True)
if __name__=='__main__':main()
