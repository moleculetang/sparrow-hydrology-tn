"""Bias/amplitude/phase decomposition of frozen results, with no model reselection."""
import numpy as np,pandas as pd
from mltn.common import ROOT,write
from mltn.metrics import basic

def decomposition(y,p):
    y=np.asarray(y,float);p=np.asarray(p,float);ym=y.mean();pm=p.mean();sy=y.std();sp=p.std();cov=np.mean((y-ym)*(p-pm))
    correlation=cov/(sy*sp) if sy>0 and sp>0 else np.nan
    # Algebraic identity holds even when a constant prediction has no defined r.
    bias2=(pm-ym)**2;amp=(sp-sy)**2;phase=2*(sy*sp-cov);mse=np.mean((p-y)**2)
    np.testing.assert_allclose(bias2+amp+phase,mse,rtol=1e-10,atol=1e-12)
    return dict(mse=mse,bias_squared=bias2,amplitude_mismatch_squared=amp,decorrelation_error=phase,correlation=correlation,
        bias_fraction=bias2/mse if mse>0 else np.nan,amplitude_fraction=amp/mse if mse>0 else np.nan,decorrelation_fraction=phase/mse if mse>0 else np.nan)

def main():
    out=ROOT/'outputs/evaluation';rows=[];season=[];within=[]
    for path in sorted(out.glob('*_frozen.parquet')):
        q=pd.read_parquet(path)
        if q.empty:continue
        name=path.stem.removesuffix('_frozen');daily=path.stem.endswith('_daily_frozen')
        for station,g in q.groupby('station_key'):
            if len(g)<2:continue
            # Both independent pieces include correlation. A mapping merge is
            # intentional; repeated keyword arguments to dict() raise before
            # any diagnostics can be saved. The metric definition is unchanged.
            rows.append(dict(model_readout=name,station_key=station,scope='level') | decomposition(g.observed,g.prediction) | basic(g.observed,g.prediction))
            if daily:
                yc=[];pc=[];weights=[]
                for _,m in g.groupby(pd.to_datetime(g.date).dt.to_period('M')):
                    if len(m)<2:continue
                    w=m.read_count.to_numpy(float);w/=w.sum();yc.extend(m.observed-np.dot(w,m.observed));pc.extend(m.prediction-np.dot(w,m.prediction));weights.extend(w)
                if yc:
                    w=np.asarray(weights);w/=w.sum();y=np.asarray(yc);p=np.asarray(pc);mse=np.dot(w,(y-p)**2);sy=np.sqrt(np.dot(w,y*y));sp=np.sqrt(np.dot(w,p*p));cov=np.dot(w,y*p)
                    rows.append(dict(model_readout=name,station_key=station,scope='month_centered',mse=mse,bias_squared=0.,amplitude_mismatch_squared=(sp-sy)**2,decorrelation_error=2*(sy*sp-cov),NSE=1-mse/(sy*sy) if sy>0 else np.nan,amplitude_ratio=sp/sy if sy>0 else np.nan,correlation=cov/(sy*sp) if sy>0 and sp>0 else np.nan))
            qdate=pd.to_datetime(g.date)
            for s,ss in g.groupby((qdate.dt.month%12//3).map({0:'DJF',1:'MAM',2:'JJA',3:'SON'})):
                season.append(dict(model_readout=name,station_key=station,season=s,**basic(ss.observed,ss.prediction)))
    if rows:pd.DataFrame(rows).to_csv(out/'error_decomposition_station.csv',index=False,encoding='utf-8-sig')
    if season:pd.DataFrame(season).to_csv(out/'seasonal_station_metrics.csv',index=False,encoding='utf-8-sig')
    reads=pd.read_parquet(ROOT/'data/hf_readings.parquet');reads=reads[reads.eligible]
    for (station,day),g in reads.groupby(['station_key','date']):
        if len(g)<2:continue
        y=g.adopted_value.to_numpy(float);within.append(dict(station_key=station,date=day,n=len(g),intra_day_sd=float(y.std()),intra_day_range=float(np.ptp(y)),daily_best_constant_sse=float(np.dot(y-y.mean(),y-y.mean()))))
    pd.DataFrame(within).to_parquet(out/'unresolved_four_hour_variation.parquet',index=False)
    write(out/'diagnostics_receipt.json',dict(decomposition='MSE=bias²+(sd_prediction-sd_observed)²+2(sd_prediction*sd_observed-covariance); decorrelation includes timing and other shape mismatch, not uniquely causal lag',rows=len(rows),season_rows=len(season),four_hour_days=len(within),model_choice='none; all frozen identities only'))
if __name__=='__main__':main()
