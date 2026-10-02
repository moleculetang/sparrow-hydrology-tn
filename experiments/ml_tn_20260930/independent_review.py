"""Prediction bytes, exact support, independent objective and coverage receipts."""
import numpy as np,pandas as pd
import pickle
from mltn.common import ROOT,write,read,sha
from mltn.data import Inputs
from mltn.metrics import basic,centered
from mltn.bootstrap import paired

def aggregate_receipt(folder):
    """Reconstruct full checkpoint data terms without the sparse objective class."""
    files=['training_prediction_daily.parquet','training_prediction_monthly.parquet','training_prediction_daily_hf.parquet','independent_training_objective.json','checkpoint.pkl']
    if not all((folder/name).exists() for name in files):return None
    from mltn.data import balanced_weights
    daily=pd.read_parquet(folder/files[0]);daily['prediction']=daily.prediction.astype(float);m=pd.read_parquet(folder/files[1]);h=pd.read_parquet(folder/files[2]);identity=pickle.load((folder/'checkpoint.pkl').open('rb'))['identity']
    month_readout=daily.groupby(['station_key','year','month']).prediction.mean().rename('independent_prediction')
    check=m.join(month_readout,on=['station_key','year','month'])
    np.testing.assert_allclose(check.prediction,check.independent_prediction,rtol=1e-8,atol=1e-9)
    checkh=h.merge(daily[['station_key','date','prediction']].rename(columns={'prediction':'independent_prediction'}),on=['station_key','date'],validate='one_to_one')
    np.testing.assert_allclose(checkh.prediction,checkh.independent_prediction,rtol=1e-8,atol=1e-9)
    wm=balanced_weights(m,identity['scales'],identity['floor']);wh=balanced_weights(h,identity['hscales'],identity['hfloor'],True)
    em=check.independent_prediction.to_numpy()-m.tn_mg_l.to_numpy();eh=checkh.independent_prediction.to_numpy()-h.tn_mg_l.to_numpy()
    for _,g in h.groupby(['station_key','year','month']):
        ix=g.index.to_numpy();eh[ix]-=np.average(eh[ix],weights=g.read_count)
    jm=.4*float(sum(float(w)*float(e)**2 for w,e in zip(wm,em)));jh=.1*float(sum(float(w)*float(e)**2 for w,e in zip(wh,eh)))
    saved=read(folder/'independent_training_objective.json');diff=abs(jm+jh-saved['data_objective']);tol=1e-8*(1+abs(saved['data_objective']))
    if diff>tol:raise ValueError('INDEPENDENT_FULL_AGGREGATE_OBJECTIVE '+folder.name)
    return dict(job=folder.name,monthly=jm,hf_anomaly=jh,total=jm+jh,absolute_difference=diff,tolerance=tol,passed=True,method='scalar residual arithmetic and pandas date-support readout; no sparse objective class')
def main():
    d=Inputs();rows=[];aggregate=[]
    for folder in (ROOT/'jobs').glob('*'):
        if not (folder/'result.json').exists():continue
        if folder.name.startswith('joint_'):
            receipt=aggregate_receipt(folder)
            if receipt is not None:aggregate.append(receipt)
        for name in ['prediction.parquet','prediction_monthly.parquet','prediction_daily_hf.parquet','closed_loop_prediction.parquet']:
            file=folder/name
            if not file.exists():continue
            q=pd.read_parquet(file);finite=bool(np.isfinite(q.prediction).all());nonnegative=bool((q.prediction>=0).all());unique=not q.duplicated(['station_key','date']).any();identity=True
            if 'tn_mg_l' in q:
                task='monthly' if 'monthly' in folder.name or name=='prediction_monthly.parquet' else 'daily';labels=d.labels(task);joined=q.merge(labels[['station_key','date','tn_mg_l']],on=['station_key','date'],suffixes=('','_source'),validate='one_to_one');identity=bool(len(joined)==len(q) and np.array_equal(joined.tn_mg_l,joined.tn_mg_l_source))
            rows.append(dict(job=folder.name,file=name,sha256=sha(file),rows=len(q),finite=finite,nonnegative=nonnegative,unique=unique,label_identity=identity))
            if not finite or not nonnegative or not unique or not identity:raise ValueError('PREDICTION_GATE '+str(file))
    derived=[]
    manifest=ROOT/'outputs/monthly_readouts/manifest.json'
    if manifest.exists():
        for r in read(manifest)['records']:
            file=ROOT/r['file'];assert sha(file)==r['sha256'];q=pd.read_parquet(file)
            meta=r['identity']
            if 'daily_prediction_file' in meta:
                dayfile=ROOT/meta['daily_prediction_file'];assert sha(dayfile)==meta['daily_prediction_sha256']
                days=pd.read_parquet(dayfile);days['month']=pd.to_datetime(days.date).dt.to_period('M').dt.to_timestamp()
                expected=days.groupby(['station_key','month']).prediction.apply(lambda p:sum(map(float,p))/len(p)).rename('independent')
                check=q.join(expected,on=['station_key','date']);np.testing.assert_allclose(check.prediction,check.independent,rtol=1e-10,atol=1e-12)
                assert 'tn_mg_l' not in days,'MONTHLY_LABELS_COPIED_TO_DAYS'
            else:
                parentfile=ROOT/meta['parent_prediction_file'];assert sha(parentfile)==meta['parent_prediction_sha256']
                days=pd.read_parquet(parentfile);days['month']=pd.to_datetime(days.date).dt.to_period('M').dt.to_timestamp()
                for row in q.itertuples():
                    g=days[days.station_key.eq(row.station_key)&days.month.eq(row.date)];n=float(sum(map(float,g.read_count)))
                    p=sum(float(w)*float(v) for w,v in zip(g.read_count,g.prediction))/n;y=sum(float(w)*float(v) for w,v in zip(g.read_count,g.tn_mg_l))/n
                    np.testing.assert_allclose([row.prediction,row.tn_mg_l],[p,y],rtol=1e-10,atol=1e-12)
            derived.append(dict(readout=r['configuration'],rows=len(q),passed=True))
    # Independent scalar arithmetic for the vectorized bootstrap with exact multiplicities.
    rng=np.random.default_rng(1);f=pd.DataFrame({'station_key':['a']*8,'date':pd.date_range('2023-01-01',periods=8,freq='MS'),'observed':rng.normal(size=8),'baseline':rng.normal(size=8),'candidate':rng.normal(size=8)})
    b=paired(f,1,1,1729);draws=np.random.default_rng(1729).integers(0,8,size=8);g=f.iloc[draws];diff=basic(g.observed,g.candidate)['NSE']-basic(g.observed,g.baseline)['NSE'];np.testing.assert_allclose(b.median_paired_difference.iloc[0],diff,atol=1e-12)
    write(ROOT/'evidence/independent_predictions_review.json',dict(passed=True,files=rows,full_checkpoint_aggregate=aggregate,monthly_readouts=derived,bootstrap_scalar_check=True,selection_isolation='internal config keys and data split registrations separately audited'))
if __name__=='__main__':main()
