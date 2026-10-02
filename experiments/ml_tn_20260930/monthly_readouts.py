"""Post-freeze daily-to-monthly readouts; no fitting, tuning, or pseudo daily labels."""
import re,pickle
import numpy as np,pandas as pd,torch
from mltn.common import ROOT,read,write,sha
from mltn.data import Inputs
from mltn.inference import direct_predict
from train import NEURAL

def predict_checkpoint(folder,d,q,device='cpu'):
    """Postprocessing uses one CPU compute thread; checkpoint weights do not change."""
    ck=pickle.load((folder/'checkpoint.pkl').open('rb'));family=ck['args']['family']
    for attr in ['_neural_key','_neural_cache']:
        if hasattr(d,attr):delattr(d,attr)
    if family in NEURAL:
        torch.set_num_threads(1);return direct_predict(folder,d,q,device)
    model=ck['model'];x=ck['transform'].apply(d.raw_rows(q,ck['args']['task']=='monthly'))
    if family=='CatBoost':p=model.predict(x,thread_count=1)
    else:
        model.set_params(n_jobs=1);p=model.predict(x)
    return np.maximum(0,p)

def hf_months(q):
    assert not q.duplicated(['station_key','date']).any(),'DUPLICATED_DAILY_SUPPORT'
    z=q.copy();z['month_key']=pd.to_datetime(z.date).dt.to_period('M');rows=[]
    for (station,month),g in z.groupby(['station_key','month_key']):
        if len(g)<2:continue
        w=g.read_count.to_numpy(float);assert (w>0).all();w=w/w.sum()
        y=g.tn_mg_l.to_numpy(float);p=g.prediction.to_numpy(float)
        # Preserve an exact constant through the linear mean operator. A dot
        # product's last-bit noise must not invent variance or correlation.
        ym=float(y[0]) if np.ptp(y)==0 else float(np.dot(w,y))
        pm=float(p[0]) if np.ptp(p)==0 else float(np.dot(w,p))
        rows.append(dict(station_key=station,date=month.to_timestamp(),tn_mg_l=ym,prediction=pm,supported_days=len(g),supported_readings=int(g.read_count.sum()),support='same available HF dates and actual read counts; not official whole-month support'))
    return pd.DataFrame(rows)

def main():
    assert (ROOT/'outputs/all_training_done.json').exists(),'TRAINING_NOT_FROZEN'
    d=Inputs();out=ROOT/'outputs/monthly_readouts';out.mkdir(parents=True,exist_ok=True);records=[];missing=[]
    roles=set(read(ROOT/'outputs/result_roles.json')['include']);selection=read(ROOT/'outputs/frozen_selection.json')
    def save(q,name,context,task,parent,identity):
        if q.empty:return
        assert np.isfinite(q.prediction).all() and (q.prediction>=0).all()
        path=out/(name+'.parquet');q.to_parquet(path,index=False)
        records.append(dict(file=path.relative_to(ROOT).as_posix(),sha256=sha(path),configuration=name,context=context,task=task,parent=parent,identity=identity))
    # Separate HF supported-month means for every frozen daily readout, including references.
    for jid in sorted(roles):
        folder=ROOT/'jobs'/jid;match=re.search(r'(F23|F24|S23|S24)',jid)
        if match is None:continue
        b=re.search(r'_B(56|113|191)',jid);context=match[1]+('' if b is None else '_B'+b[1])
        for filename in ['prediction.parquet','prediction_daily_hf.parquet','closed_loop_prediction.parquet']:
            path=folder/filename
            if not path.exists():continue
            q=pd.read_parquet(path)
            if 'read_count' not in q or 'tn_mg_l' not in q:continue
            name='hfmean_'+jid+('_closed_loop' if filename.startswith('closed_loop') else '')
            save(hf_months(q),name,context,'hf_monthly',jid,dict(operator='read-count weighted mean of shared HF daily support; at least two dates per month',parent_prediction_file=path.relative_to(ROOT).as_posix(),parent_prediction_sha256=sha(path)))
    # The direct daily algorithms can also be queried on full calendar supports.
    # Their training still uses HF only; this does not make a new joint model.
    for fold,year in [('F23',2023),('F24',2024)]:
        m=d.labels('monthly');m=m[m.year.eq(year)].reset_index(drop=True)
        parts=[]
        for row in m.itertuples():
            days=pd.date_range(row.date,row.end_date);parts.append(pd.DataFrame(dict(station_key=row.station_key,date=days,si=row.si,ri=row.ri,year=year,month=row.month)))
        q=pd.concat(parts,ignore_index=True);q['ti']=d.dates.get_indexer(q.date);q['start_ti']=q.ti
        for key,c in selection['selected'].items():
            family,task=key.rsplit('_',1)
            if task!='daily':continue
            seeds=[1729,1730,1731] if family==selection['winners']['daily'] else [1729]
            for seed in seeds:
                jid=f'{fold}_daily_{family}_c{c}_s{seed}';folder=ROOT/'jobs'/jid
                if jid not in roles or not (folder/'result.json').exists():missing.append(jid);continue
                # A new deserialized transform must never inherit another checkpoint's cache.
                device='cuda' if family in NEURAL and torch.cuda.is_available() else 'cpu'
                pred=np.asarray(predict_checkpoint(folder,d,q,device),dtype=float);daily=q[['station_key','date']].copy();daily['prediction']=pred
                dayfile=out/('dailyagg_'+jid+'_calendar_daily.parquet');daily.to_parquet(dayfile,index=False)
                grouped=daily.assign(date=daily.date.dt.to_period('M').dt.to_timestamp())
                means=grouped.groupby(['station_key','date'],as_index=False).prediction.mean()
                result=m.merge(means,on=['station_key','date'],validate='one_to_one');assert len(result)==len(m)
                save(result,'dailyagg_'+jid,fold,'monthly',jid,dict(operator='equal-calendar-day approximation for official monthly readout; actual official daily read counts unavailable',training_support='direct daily model HF labels only; no official monthly refit',labels='official monthly TN only for post-freeze scoring; calendar daily predictions carry no synthetic daily labels',parent_checkpoint_sha256=sha(folder/'checkpoint.pkl'),daily_prediction_file=dayfile.relative_to(ROOT).as_posix(),daily_prediction_sha256=sha(dayfile)))
    write(out/'manifest.json',dict(records=records,missing=missing,fit_calls=0,selection='existing frozen internal configurations; no score-based reselection',comparability='official month, HF supported month, and direct daily forecasts are distinct observation supports'))

if __name__=='__main__':main()
