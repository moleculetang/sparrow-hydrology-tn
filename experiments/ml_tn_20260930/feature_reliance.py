"""Post-freeze statistical feature reliance, never model selection or causality."""
import pickle
import numpy as np,pandas as pd
from mltn.common import ROOT,read,write
from mltn.data import Inputs
from mltn.metrics import station_table
from train import NEURAL

def group(name):
    if name in ['annual_sin','annual_cos']:return 'season'
    if name=='downstream_fraction_on_reach':return 'station_geometry'
    if any(x in name for x in ['harvest','residue','bnf_','agriculture_','deposition_','mod17_']):return 'source_and_activity_products'
    if name.startswith(('local_','upstream_')) and any(x in name for x in ['water','soil_wetness','temperature','percolation','precipitation','pet_']):return 'H1_and_weather'
    if name in ['reach_water_log1p','reach_inlet_water_log1p','station_water_log1p']:return 'H1_and_weather'
    return 'land_soil_terrain_static'

def main():
    d=Inputs();selection=read(ROOT/'outputs/frozen_selection.json');names=d.identity['features']+['station_water_log1p','downstream_fraction_on_reach'];native=[];rows=[];skipped=[]
    for task,family in selection['winners'].items():
        if family in NEURAL:
            skipped.append(dict(task=task,family=family,reason='tree diagnostic not applicable; no replacement route selected'))
            continue
        c=selection['selected'][family+'_'+task]
        for fold in ['F23','F24']:
            folder=ROOT/'jobs'/f'{fold}_{task}_{family}_c{c}_s1729'
            if not (folder/'result.json').exists():continue
            ck=pickle.load((folder/'checkpoint.pkl').open('rb'));model=ck['model'];tr=ck['transform'];q=pd.read_parquet(folder/'prediction.parquet').rename(columns={'tn_mg_l':'observed'})
            if family!='CatBoost':model.set_params(n_jobs=1)
            x=tr.apply(d.raw_rows(q,task=='monthly'));n=len(names);assert x.shape[1]==2*n
            importance=np.asarray(model.feature_importances_,float);assert len(importance)==2*n
            for i,name in enumerate(names):native.append(dict(job=folder.name,feature=name,group=group(name),numeric_importance=importance[i],missingness_importance=importance[n+i],total_importance=importance[i]+importance[n+i]))
            baseline=station_table(q,predictions=('prediction',),daily=task=='daily',minimum_coverage=True)
            eligible=baseline[baseline.nse_eligible];base=float(eligible.NSE.median()) if len(eligible) else np.nan
            for category in sorted(set(map(group,names))):
                ix=np.array([i for i,name in enumerate(names) if group(name)==category]);ix=np.r_[ix,n+ix]
                for rep in range(5):
                    rng=np.random.default_rng(1729+rep);xp=x.copy()
                    # Move a group's numeric and mask channels together, within the same
                    # calendar month; never manufacture observations or refit transforms.
                    for _,g in q.groupby(pd.to_datetime(q.date).dt.to_period('M')):
                        idx=g.index.to_numpy();xp[np.ix_(idx,ix)]=x[np.ix_(rng.permutation(idx),ix)]
                    pred=np.maximum(0,model.predict(xp,thread_count=1) if family=='CatBoost' else model.predict(xp));assert np.isfinite(pred).all();v=q.copy();v['prediction']=pred
                    table=station_table(v,predictions=('prediction',),daily=task=='daily',minimum_coverage=True);table=table[table.station_key.isin(eligible.station_key)]
                    rows.append(dict(job=folder.name,task=task,group=category,replicate=rep,common_stations=len(eligible),baseline_median_NSE=base,perturbed_median_NSE=float(table.NSE.median()),median_NSE_loss=base-float(table.NSE.median()),pooled_MSE_change=float(np.mean((pred-q.observed.to_numpy())**2)-np.mean((q.prediction-q.observed)**2))))
    out=ROOT/'outputs/evaluation';out.mkdir(exist_ok=True,parents=True)
    pd.DataFrame(native).to_csv(out/'native_feature_importance.csv',index=False,encoding='utf-8-sig');pd.DataFrame(rows).to_csv(out/'group_permutation_reliance.csv',index=False,encoding='utf-8-sig')
    write(out/'feature_reliance_receipt.json',dict(native_rows=len(native),perturbations=len(rows),skipped=skipped,selection='train-only winner and fixed seed1729; no evaluation-based reselection',interpretation='statistical reliance under group permutation; correlations with unpermuted groups may be broken, so not causal contribution or independent input accuracy',missingness='permuted with numeric channels',source_support='source group follows frozen local/upstream feature supports; no upstream deposition feature added post hoc',seed=1729))

if __name__=='__main__':main()
