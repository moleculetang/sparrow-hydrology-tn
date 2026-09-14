"""Predict new reaches with an old-package training design, without new TN labels."""
import os,sys,json,copy,hashlib,argparse
from pathlib import Path
HERE=Path(__file__).resolve().parent;CORE=HERE.parent/'tn_challenge_plain'
sys.path.insert(0,str(CORE))
from model import Predictor,load_data
from run import metrics
import numpy as np,pandas as pd,torch

def load_transfer_data():
    d=load_data(HERE/'data')
    for k,s in d.empty_arrays.items():setattr(d,k,np.empty(s['shape'],dtype=np.dtype(s['dtype'])))
    return d

class TransferPredictor(Predictor):
    def map_observations(self,meta):
        # The original sample had reservoirs. For a genuinely reservoir-free
        # graph, never eagerly index a nonexistent release column in np.where.
        if self.data.metadata:return super().map_observations(meta)
        if 'tn_mg_l' in meta:raise ValueError('Prediction refuses TN labels')
        if not meta.station_type.eq('ordinary_internal').all():raise ValueError('This extension contains ordinary internal stations only')
        ti=(meta.year.to_numpy(int)-1961)*12+meta.month.to_numpy(int)-1;ri=meta.reach_id.to_numpy(int)-1
        if (ti<0).any() or (ti>=len(self.data.months)).any() or (ri<0).any() or (ri>=self.data.source.shape[1]).any():raise ValueError('Outside supplied support')
        f=meta.downstream_fraction_on_reach.to_numpy(float);w=self.water['inlet'][ti,ri]+f*self.water['local'][ti,ri]
        if (w<=0).any():raise ValueError('Nonpositive observation water volume')
        # No boundary_code: original boundary_mass then uses its ordinary
        # internal-station formula exactly, without any reservoir indexing.
        return {k:torch.tensor(v) for k,v in dict(ti=ti,ri=ri,f=f,h=self.data.h_month[ti,ri],water=w).items()}

def frozen_transfer(saved,data,check_training=True):
    if check_training:
        allowed=pd.read_csv(CORE/'data/train.csv')
        ids=saved.get('training_ids',[])
        if not ids or set(ids)!=set(allowed.observation_id):raise ValueError('Benchmark requires exactly the original 393 training IDs; full-domain reference weights and new labels are forbidden')
    design=copy.deepcopy(saved['design']);design.pop('inventory_scale_kg',None)
    if saved['variant']=='SC':
        # Extend the registered label-free S rule to new reaches. Static and
        # dynamic scalers are frozen from the OLD training set, never re-fit.
        base=TransferPredictor(data,design,'M0');available=base.ledger(base.initial(0))['available']
        years=design.get('training_years',[2021,2022])
        design['inventory_scale_kg']=np.maximum(1.,np.median(available[np.isin(data.dates.year,years)],axis=0)).tolist()
    return TransferPredictor(data,design,saved['variant'])

def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('predict');a.add_argument('--model',type=Path,required=True);a.add_argument('--metadata',type=Path,default=HERE/'evaluation_metadata.csv');a.add_argument('--out',type=Path,required=True)
    a=sub.add_parser('score');a.add_argument('--predictions',type=Path,required=True);a.add_argument('--labels',type=Path,default=HERE/'evaluation_labels.csv');a.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():raise FileExistsError('Use a new output directory')
    args.out.mkdir(parents=True)
    if args.command=='predict':
        saved=json.loads(args.model.read_text(encoding='utf-8'));meta=pd.read_csv(args.metadata)
        if 'tn_mg_l' in meta:raise ValueError('Metadata file must not contain TN')
        model=frozen_transfer(saved,load_transfer_data());pred=model.predict(saved['theta'],meta)
        meta['prediction_mg_l']=pred;meta.to_csv(args.out/'predictions.csv',index=False)
        receipt={'model_sha256':hashlib.sha256(args.model.read_bytes()).hexdigest(),'data_manifest_sha256':hashlib.sha256((HERE/'data/manifest.json').read_bytes()).hexdigest(),
         'predictions_sha256':hashlib.sha256((args.out/'predictions.csv').read_bytes()).hexdigest(),
         'code_sha256':{str(p.relative_to(HERE.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),CORE/'model.py',CORE/'sc_kernel.py',CORE/'routing.py',CORE/'text_arrays.py']},
         'rows':len(meta),'stations':meta.station_key.nunique(),'new_TN_read':False,'new_scalers_fitted':False,
         'S_rule':'preset0, old static/dynamic scalers, old training years, new frozen reach inputs, no TN',
         'input_model_numerical_sufficient':saved.get('numerical_sufficient'),'input_model_training_ids':saved.get('training_ids')}
        (args.out/'prediction_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    else:
        receipt_path=args.predictions.parent/'prediction_receipt.json'
        receipt=json.loads(receipt_path.read_text(encoding='utf-8'))
        if hashlib.sha256(args.predictions.read_bytes()).hexdigest()!=receipt['predictions_sha256']:raise ValueError('Prediction file changed after freezing')
        pred=pd.read_csv(args.predictions);labels=pd.read_csv(args.labels)
        if pred.observation_id.duplicated().any() or labels.observation_id.duplicated().any():raise ValueError('Duplicate IDs')
        if set(pred.observation_id)!=set(labels.observation_id):raise ValueError('Prediction/label ID mismatch: score the exact predicted cohort')
        frame=labels.merge(pred[['observation_id','prediction_mg_l']],on='observation_id',validate='one_to_one')
        frame.to_csv(args.out/'scored_predictions.csv',index=False)
        tables=[]
        for (year,group),g in frame.groupby(['year','benchmark_group']):
            m=metrics(g);m['year']=year;m['benchmark_group']=group;tables.append(m)
        pd.concat(tables,ignore_index=True).to_csv(args.out/'station_year_metrics.csv',index=False)
        metrics(frame).to_csv(args.out/'all_year_station_metrics.csv',index=False)
    print(json.dumps({'status':'COMPLETE','command':args.command,'output':str(args.out)}))
if __name__=='__main__':main()
