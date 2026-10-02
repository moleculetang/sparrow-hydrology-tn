"""Freeze external-driver ML readout at all230 reach outlets, geometry f=1."""
import pickle,numpy as np,pandas as pd,torch
from mltn.common import ROOT,read,write
from mltn.data import Inputs
from mltn.models import Network
from monthly_readouts import predict_checkpoint as direct_predict
from train import predict_network,NEURAL
def full_inputs():
    d=Inputs();d.reg=pd.DataFrame(dict(station_key=['REACH_'+str(i+1) for i in range(230)],reach_id=np.arange(1,231),downstream_fraction_on_reach=np.ones(230)))
    d.rr=np.arange(230);d.station_index=dict(zip(d.reg.station_key,np.arange(230)))
    # Reach outlet logarithmic water is frozen in the independent feature field.
    d.water=np.expm1(np.asarray(d.x[:,:,d.identity['features'].index('reach_water_log1p')],dtype=float));return d
def rows(d,year,task):
    dates=pd.date_range(f'{year}-01-01',f'{year}-12-31',freq='MS' if task=='monthly' else 'D');q=pd.DataFrame(dict(date=np.repeat(dates,230),si=np.tile(np.arange(230),len(dates))))
    q['ri']=q.si;q['station_key']=d.reg.station_key.to_numpy()[q.si];q['year']=year;q['month']=q.date.dt.month;q['end_date']=q.date+pd.offsets.MonthEnd(0) if task=='monthly' else q.date;q['ti']=d.dates.get_indexer(q.end_date);q['start_ti']=d.dates.get_indexer(q.date) if task=='monthly' else q.ti;return q
def main():
    d=full_inputs();selection=read(ROOT/'outputs/frozen_selection.json');joint=read(ROOT/'outputs/frozen_joint_selection.json')['selected'] if (ROOT/'outputs/frozen_joint_selection.json').exists() else {};out=ROOT/'outputs/full_domain';out.mkdir(exist_ok=True)
    for fold,year in [('F23',2023),('F24',2024)]:
        for task,family in selection['winners'].items():
            c=selection['selected'][family+'_'+task];q=rows(d,year,task);pp=[]
            for seed in [1729,1730,1731]:
                folder=ROOT/'jobs'/f'{fold}_{task}_{family}_c{c}_s{seed}'
                if not (folder/'result.json').exists():continue
                pp.append(direct_predict(folder,d,q,'cuda' if torch.cuda.is_available() and family in NEURAL else 'cpu'))
            if len(pp)!=3:continue
            q['prediction']=np.mean(pp,axis=0);q.to_parquet(out/f'{fold}_{task}_{family}_230_reach_outlets.parquet',index=False)
        for family,c in joint.items():
            q=rows(d,year,'daily');pp=[]
            for seed in [1729,1730,1731]:
                folder=ROOT/'jobs'/f'joint_{fold}_{family}_c{c}_s{seed}'
                if not (folder/'result.json').exists():continue
                ck=pickle.load((folder/'checkpoint.pkl').open('rb'));tr=ck['transform'];cfg=ck['cfg']
                if family=='XGBoost':
                    import xgboost as xgb
                    model=xgb.Booster();model.load_model(folder/'booster.json');model.set_param({'nthread':1});pp.append(np.exp(model.predict(xgb.DMatrix(tr.apply(d.raw_rows(q))),output_margin=True)))
                elif family=='LightGBM':
                    import lightgbm as lgb
                    model=lgb.Booster(model_file=str(folder/'booster.txt'));pp.append(np.exp(model.predict(tr.apply(d.raw_rows(q)),raw_score=True,num_threads=1)+ck['base']))
                else:
                    device='cuda' if torch.cuda.is_available() else 'cpu';model=Network(family,len(tr.mean)*2,cfg).to(device);model.load_state_dict(torch.load(folder/'weights.pt',map_location=device,weights_only=True));pp.append(predict_network(model,d,q,cfg,family,tr,device))
            if len(pp)==3:
                q['prediction']=np.mean(pp,axis=0);assert np.isfinite(q.prediction).all() and (q.prediction>=0).all();q.to_parquet(out/f'{fold}_joint_{family}_230_reach_outlets.parquet',index=False)
    write(out/'identity.json',dict(reaches=230,readout='outlet fraction1 with frozen river total water; map names are metadata, never model embeddings',labels='none at unlabeled reach outlets; outputs are extrapolations, not 230-site validation',water_roundtrip='outlet log-water float32 feature inverted only for ML auxiliary readout; no physical load accounting',performance='certified station support metrics separately'))
if __name__=='__main__':main()
