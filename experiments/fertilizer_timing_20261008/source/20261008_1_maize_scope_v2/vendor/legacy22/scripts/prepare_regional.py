"""Rebuild spatial training before QC; freeze label-free PCA and finite jobs."""
import copy,json,time,shutil
import numpy as np,pandas as pd
import native_runtime as rt
from campaign_model import load_data
from temporal_model import clean_metadata
from prepare_hf import classify,products
R=rt.RUN;P=R.parent/'20260917_5';PREV=R.parent/'20260921_2'

def save(df,path):
    path.parent.mkdir(parents=True,exist_ok=True);df.to_parquet(path,index=False)

def closure(seeds,edges,reverse=False):
    seen=set(seeds)
    while True:
        extra={a if reverse else b for a,b in edges if (b if reverse else a) in seen}
        if extra<=seen:return seen
        seen|=extra

def build_training(normal,obs,station,local,allowed,scope,design):
    years=[2021,2022,2023];keys=['station_key','year','month'];thresholds=[]
    tr=normal[normal.station_key.isin(allowed)&normal.year.isin(years)].copy();tr['excluded']=False
    for (s,se),g in tr.groupby(['station_key','season']):
        flag,stat,_,_=classify(g);tr.loc[g.index,'excluded']=flag;thresholds.append(dict(station_key=s,season=int(se),**stat))
    daily,hfm=products(tr[~tr.excluded],years)
    pub=obs[obs.station_key.isin(allowed)&obs.year.isin(years)].copy();pub['excluded']=False
    for _,g in pub.groupby('station_key'):pub.loc[g.index,'excluded']=classify(g,False)[0]
    hf=hfm[keys+['y']].copy();hf['source_kind']='HF';hf['pub_observation_id']=None
    mon=pub[~pub.excluded][keys+['tn_mg_l','observation_id']].rename(columns={'tn_mg_l':'y','observation_id':'pub_observation_id'});mon['source_kind']='PUB'
    mon=mon.merge(hf[keys].assign(has_hf=True),on=keys,how='left');mon=mon[mon.has_hf.isna()].drop(columns='has_hf')
    union=pd.concat([hf,mon],ignore_index=True).sort_values(keys).reset_index(drop=True);assert not union.duplicated(keys).any()
    stat=union.groupby('station_key').y.agg(n='size',variance=lambda x:float(np.var(x)));valid=set(stat[stat.n>=2].index)
    union=union[union.station_key.isin(valid)].copy();daily=daily[daily.station_key.isin(valid)].copy();stat=stat.loc[sorted(valid)].copy()
    ref=stat[stat.index.isin(local&allowed)&stat.variance.gt(0)];assert len(ref)>0
    floor=float(ref.variance.quantile(.1));assert floor>0
    stat['denominator']=stat.variance.clip(lower=floor);ns=len(stat)
    records=[];registry={};lineage=[];config=scope+'_D_H1'
    for _,u in union.iterrows():
        dd=daily[(daily.station_key==u.station_key)&(daily.year==u.year)&(daily.month==u.month)] if u.source_kind=='HF' else pd.DataFrame()
        for a in (dd.to_dict('records') if u.source_kind=='HF' else [dict(y=u.y,alpha=1.,date=None)]):
            oid=f'{config}_{len(records):06d}';s=station[station.station_key.eq(u.station_key)].iloc[0].to_dict()
            s.update(observation_id=oid,year=int(u.year),month=int(u.month),day_index=-1 if a.get('date') is None else int((a['date']-pd.Timestamp('1961-01-01')).days),tn_mg_l=float(a['y']),fit_variance=float(stat.loc[u.station_key,'denominator']),fit_weight=float(a.get('alpha',1)/(ns*stat.loc[u.station_key,'n']*stat.loc[u.station_key,'denominator'])))
            records.append(s);registry[oid]=dict(day_weights=None,excluded=False)
            ids=list(map(str,a['selected_record_ids'])) if u.source_kind=='HF' else [str(u.pub_observation_id)]
            lineage.append(dict(observation_id=oid,station_key=u.station_key,year=int(u.year),month=int(u.month),source_kind=u.source_kind,source_record_ids=ids))
    table0=pd.DataFrame(records);table=clean_metadata(table0)
    for k in ['tn_mg_l','fit_variance','fit_weight']:table[k]=table0[k]
    return dict(config=config,table=table,registry=registry,union=union,daily=daily,stat=stat,cleaned_hf=tr,cleaned_pub=pub,thresholds=thresholds,lineage=lineage,floor=floor)

def make_basis(d,allowed):
    use=np.isin(d.global_reach_ids,list(allowed));x=np.asarray(d.static_raw,dtype=float)
    w=np.asarray(d.area_ha,dtype=float)[use];w=w/w.sum();mean=w@x[use]
    sd=np.sqrt(w@((x[use]-mean)**2));constant=sd==0
    if not np.isfinite(x).all() or not np.isfinite(sd).all():raise ValueError('BAD_STATIC_ATTRIBUTES')
    z=np.zeros_like(x);z[:,~constant]=(x[:,~constant]-mean[~constant])/sd[~constant]
    _,s,vt=np.linalg.svd(np.sqrt(w)[:,None]*z[use],full_matrices=False)
    cutoff=max(z[use].shape)*np.finfo(float).eps*s[0];rank=int((s>cutoff).sum())
    if rank<3:raise ValueError('SPATIAL_BASIS_RANK_BELOW_3')
    v=vt[:3].copy()
    for row in v:
        if row[np.argmax(abs(row))]<0:row*=-1
    psi=(z@v.T)/s[:3]
    assert np.max(abs(w@psi[use]))<1e-12
    assert np.max(abs((psi[use]*w[:,None]).T@psi[use]-np.eye(3)))<1e-10
    return psi,dict(allowed_reaches=sorted(map(int,allowed)),mean=mean.tolist(),sd=sd.tolist(),constant_columns=np.flatnonzero(constant).tolist(),components=v.tolist(),singular_values=s.tolist(),numerical_rank=rank,rank_cutoff=float(cutoff),principal_scale=s[:3].tolist(),weight='incremental area_ha; only allowed reaches',sign='largest absolute loading positive',input_kind='original seven static attributes, no TN')

def main():
    assert not (R/'reports/preparation_regional.json').exists()
    d=load_data('FULL24C');folds=rt.read(R/'configs/folds.json');station=pd.read_parquet(R/'data/station_registry.parquet')
    topo=rt.read(R/'data/domains/FULL24C/topology.json');edges={(int(a)+1,int(b)+1) for a,b in topo['downstream'].items()}
    edges|={(int(c)+1,int(m['target'])+1) for m in topo['metadata'] for c in m['controls']}
    rawpath=P/'data/heldout_labels/hf_canonical_selected.parquet';obspath=P/'data/heldout_labels/monthly_original.parquet'
    raw=pd.read_parquet(rawpath);obs=pd.read_parquet(obspath)
    normal=raw[raw.indicator.eq('TN')&raw.adopted_value.notna()&np.isfinite(raw.adopted_value)&raw.adopted_value.ge(0)&~raw.unresolved_conflict.fillna(False)&raw.station_status.eq('正常')].copy()
    old=pd.read_parquet(R/'data/old_calendar.parquet').drop_duplicates('station_key');local=set(old[old.cohort.isin(['N','H'])].station_key)
    template=rt.read(R/'data/designs/T24_G_D_H1.json');oldblocks=rt.read(P/'data/spatial_blocks.json');blocks={};summaries=[]
    for outlet in [56,113,191]:
        held=closure({outlet},edges,True);buffer=closure(held,edges)-held
        assert held==set(oldblocks[str(outlet)]['held_reaches']) and buffer==set(oldblocks[str(outlet)]['buffer_reaches'])
        allowed_reaches=set(map(int,d.global_reach_ids))-held-buffer
        allowed=set(station[station.reach_id.isin(allowed_reaches)].station_key);scope='S'+str(outlet)
        a=build_training(normal,obs,station,local,allowed,scope,template);cfg=a['config'];base=R/'data/cohorts'/scope
        for key,name in [('union','station_months'),('daily','hf_days')]:save(a[key],base/(name+'.parquet'))
        save(a['stat'].reset_index(),base/'scales.parquet')
        for key in ['cleaned_hf','cleaned_pub']:save(a[key],R/'evidence'/f'{scope}_{key}.parquet')
        save(pd.DataFrame(a['lineage']),R/'evidence'/f'{scope}_training_lineage.parquet');rt.write(R/'evidence'/f'{scope}_thresholds.json',a['thresholds'])
        folder=R/'data/folds'/cfg;save(a['table'],folder/'train.parquet');rt.write(folder/'registry.json',dict(records=a['registry']))
        des=copy.deepcopy(template);des.update(fold_id=scope,spatial_block_id=str(outlet),source_selection_hash=rt.sha(base/'station_months.parquet'),scale_hash=rt.sha(base/'scales.parquet'),observation_registry_file=(folder/'registry.json').relative_to(R).as_posix(),observation_registry_hash=rt.sha(folder/'registry.json'))
        rt.write(R/'data/designs'/f'{cfg}.json',des)
        folds[cfg]=dict(domain='FULL24C',end_year=2024,evaluation_year=2024,train_years=[2021,2022,2023],scope=scope,mode='D',spatial_block=str(outlet),train_sha256=rt.sha(folder/'train.parquet'),allowed_reaches=sorted(allowed_reaches))
        blocks[str(outlet)]=dict(outlet=outlet,held_reaches=sorted(held),buffer_reaches=sorted(buffer),held_stations=station[station.reach_id.isin(held)].station_key.tolist(),buffer_stations=station[station.reach_id.isin(buffer)].station_key.tolist(),allowed_stations=sorted(allowed))
        # True counterfactual: mutate all forbidden and evaluation TN, rebuild all training quantities.
        mutated=normal.copy();forbidden=~(mutated.station_key.isin(allowed)&mutated.year.isin([2021,2022,2023]));mutated.loc[forbidden,'adopted_value']=777777.
        pm=obs.copy();pm.loc[~(pm.station_key.isin(allowed)&pm.year.isin([2021,2022,2023])),'tn_mg_l']=999999.
        b=build_training(mutated,pm,station,local,allowed,scope,template)
        for key in ['table','union','daily','stat']:pd.testing.assert_frame_equal(a[key],b[key])
        assert a['registry']==b['registry']
        summaries.append(dict(scope=scope,stations=a['table'].station_key.nunique(),hf_stations=a['daily'].station_key.nunique(),training_rows=len(a['table']),hf_days=len(a['daily']),floor=a['floor'],counterfactual_identical=True))
    for fold,cfg in folds.items():
        allowed=cfg.get('allowed_reaches',list(map(int,d.global_reach_ids)))
        psi,info=make_basis(d,allowed);folder=R/'data/bases';folder.mkdir(exist_ok=True);path=folder/(fold+'.npy');np.save(path,psi,allow_pickle=False)
        rt.write(folder/(fold+'.json'),info)
        des=rt.read(R/'data/designs'/f'{fold}.json');des['regional_basis']=dict(file=path.relative_to(R).as_posix(),sha256=rt.sha(path),metadata=(folder/(fold+'.json')).relative_to(R).as_posix())
        rt.write(R/'data/designs'/f'{fold}.json',des)
    jobs=[]
    for short,fold in [('F23','F23_G_D'),('F24','T24_G_D_H1')]+[(f'S{o}',f'S{o}_D_H1') for o in [56,113,191]]:
        arms=[('U','SOURCE_UNIFIED'),('L1','REGIONAL_L1'),('L3','REGIONAL_L3'),('N3','REGIONAL_N3')] if short.startswith('F') else [('U','SOURCE_UNIFIED'),('L3','REGIONAL_L3')]
        previous=[]
        for arm,kind in arms:
            tags=[f'{short}_{arm}_s{i}' for i in [0,1]]
            for start,tag in enumerate(tags):
                job=dict(tag=tag,fold=fold,kind=kind,start=start,operator_id='OU',mapping_id='G1',observation_operator='MATCH',objective_id='D',dependencies=previous.copy(),nested_tags=previous.copy(),priority=1 if arm=='N3' else 0)
                if arm=='U' and short.startswith('F'):job['reuse_from']=str(PREV/'outputs'/tag)
                jobs.append(job)
            previous+=tags
    assert len(jobs)==28 and sum('reuse_from' not in j for j in jobs)==24
    rt.write(R/'configs/folds.json',folds);rt.write(R/'configs/jobs.json',jobs);rt.write(R/'data/spatial_blocks.json',blocks)
    cfg=rt.read(R/'configs/campaign.json');cfg.update(models=['SOURCE_UNIFIED',*['REGIONAL_L1','REGIONAL_L3','REGIONAL_N3']],paths=28,budget=dict(campaign_hours=48,preparation_hours=8,fit_hours=32,report_hours=8),candidate=dict(main='REGIONAL_L3',source='one joint c in [0.25,4], demand unchanged',ranks=[1,3],hidden_units=2,seed=1729,source_prior='21_2 unchanged',mapping_prior='0.5*(0.03/17)*(sum((A/0.5)^2)+sum((V/0.5)^2))',no_hidden_prior=True,public_unlabelled_H1_design=True),temporal_operators=['D'])
    rt.write(R/'configs/campaign.json',cfg)
    rt.write(R/'reports/preparation_regional.json',dict(status='PASS',folds=list(folds),spatial=summaries,paths=28,new_fits=24,maximum_new_fits=28,source_files={str(rawpath):rt.sha(rawpath),str(obspath):rt.sha(obspath)},created=time.time()))
    print('PREPARED',summaries,flush=True)
if __name__=='__main__':main()
