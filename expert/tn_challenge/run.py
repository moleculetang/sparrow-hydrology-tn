"""Small expert harness, NOT a rerun of the full registered eight-path experiment."""
import argparse,json,time,hashlib,os
from pathlib import Path
from model import load_data,Objective,Predictor,projected_gradient,HERE
import numpy as np,pandas as pd
from scipy.optimize import minimize

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(tmp,path)
def metrics(frame):
    rows=[]
    for key,g in frame.groupby('station_key'):
        y=g.tn_mg_l.to_numpy();p=g.prediction_mg_l.to_numpy();v=np.sum((y-y.mean())**2);eligible=len(y)>=8 and v>0
        r=float(np.corrcoef(y,p)[0,1]) if eligible and np.std(p)>0 else None
        rows.append(dict(station_key=key,n=len(y),dynamic_eligible=eligible,nse=float(1-np.sum((p-y)**2)/v) if eligible else None,
            r=r,rmse=float(np.sqrt(np.mean((p-y)**2))),logrmse=float(np.sqrt(np.mean((np.log1p(p)-np.log1p(y))**2))),
            bias=float(np.mean(p-y)),observed_sd=float(np.std(y)),predicted_sd=float(np.std(p))))
    return pd.DataFrame(rows)
def main():
    ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='command',required=True)
    p=sub.add_parser('fit');p.add_argument('--train',type=Path,default=HERE/'data/train.csv');p.add_argument('--variant',choices=['M0','SC'],default='M0')
    p.add_argument('--start',type=int,choices=[0,1],default=0);p.add_argument('--max-calls',type=int,default=1000);p.add_argument('--seconds',type=float,default=3600)
    p.add_argument('--out',type=Path,required=True)
    p=sub.add_parser('evaluate');p.add_argument('--model',type=Path,required=True);p.add_argument('--labels',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p=sub.add_parser('predict');p.add_argument('--model',type=Path,required=True);p.add_argument('--metadata',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();d=load_data()
    if a.command=='fit':
        if a.max_calls<1 or a.seconds<=0:raise ValueError('Positive budget required')
        # No evaluation label path is passed to Objective or opened here.
        if a.out.exists():raise FileExistsError('Choose a fresh output directory; never overwrite a fit')
        a.out.mkdir(parents=True);train=pd.read_csv(a.train);m=Objective(d,train,a.variant);x0=m.initial(a.start);scale=m.variable_scale()
        start=time.monotonic();trajectory=[];best={};reason='';calls=0
        class Budget(Exception):pass
        def fun(q):
            nonlocal calls,best
            if calls>=a.max_calls or (calls and time.monotonic()-start>=a.seconds):raise Budget('BUDGET_REACHED')
            x=q*scale;J,g=m.value_gradient(x);calls+=1
            if not np.isfinite(J) or not np.isfinite(g).all():raise FloatingPointError('Nonfinite objective/gradient')
            pg=float(np.max(np.abs(projected_gradient(x,g,m.bounds))))
            row=dict(call=calls,seconds=time.monotonic()-start,MAP=J,projected_gradient=pg,**m.last_terms);trajectory.append(row)
            if not best or J<best['MAP']:
                best=dict(row,theta=x.tolist(),variant=a.variant,parameter_names=m.names,design=m.design,
                          training_ids=m.train.observation_id.tolist(),training_sha256=hashlib.sha256(a.train.read_bytes()).hexdigest(),
                          optimizer='scaled L-BFGS-B expert starter; not registered TRF',initial_preset=a.start)
                write(a.out/'checkpoint.json',best)
            return J,g*scale
        try:
            res=minimize(fun,x0/scale,jac=True,method='L-BFGS-B',bounds=[(lo/s,hi/s) for (lo,hi),s in zip(m.bounds,scale)],
                options={'maxiter':a.max_calls,'maxfun':a.max_calls,'ftol':1e-14,'gtol':1e-9,'maxls':30})
            reason=str(res.message)
        except Budget as e:reason=str(e)
        finally:
            pd.DataFrame(trajectory).to_csv(a.out/'trajectory.csv',index=False)
        best.update(total_calls=calls,total_seconds=time.monotonic()-start,stop_reason=reason,
                    numerical_sufficient=best.get('projected_gradient',np.inf)<=1e-5)
        write(a.out/'model.json',best)
        pd.DataFrame({'observation_id':m.train.observation_id,'RAW_weight':m.weight}).to_csv(a.out/'training_weights.csv',index=False)
        print(json.dumps({k:best[k] for k in ['total_calls','total_seconds','MAP','data','prior','projected_gradient','numerical_sufficient','stop_reason']}))
    else:
        saved=json.loads(a.model.read_text(encoding='utf-8'));m=Predictor(d,saved['design'],saved['variant']);x=np.asarray(saved['theta'])
        frame=pd.read_csv(a.labels if a.command=='evaluate' else a.metadata)
        if a.command=='evaluate' and set(frame.observation_id)&set(saved['training_ids']):raise ValueError('Train/evaluation IDs overlap')
        meta=frame.drop(columns=['tn_mg_l'],errors='ignore');frame['prediction_mg_l']=m.predict(x,meta)
        a.out.mkdir(parents=True,exist_ok=False);frame.to_csv(a.out/'predictions.csv',index=False)
        ledger=m.ledger(x)
        np.savez_compressed(a.out/'physical_ledger.npz',**ledger)
        audit={'local_balance_max_kg':ledger['local_balance_max_kg'],'network_balance_kg':ledger['network_balance_kg'],
               'minimum_M_kg':float(ledger['M'].min()),'minimum_L_kg':float(ledger['L'].min()),'uptake_excess_kg':float(np.max(ledger['uptake']-ledger['demand']))}
        write(a.out/'physical_summary.json',audit)
        if a.command=='evaluate':metrics(frame).to_csv(a.out/'stations.csv',index=False)
        print(json.dumps(audit))
if __name__=='__main__':main()
