"""Post-freeze amplitude profiles and objective-direction attribution.

No new fit, feature or model selection. Evaluation-label oracles are explicitly
diagnostic bounds for an existing waveform, never accepted predictions.
"""
import math
import pickle
import numpy as np
import pandas as pd
from mltn.common import ROOT,read,write,sha
from mltn.resources import lease


def moments(y,p,w):
    y=np.asarray(y,float);p=np.asarray(p,float);w=np.asarray(w,float);w=w/math.fsum(w)
    ym=math.fsum(float(a*b) for a,b in zip(y,w));pm=math.fsum(float(a*b) for a,b in zip(p,w))
    yc=y-ym;pc=p-pm
    vy=math.fsum(float(a*a*b) for a,b in zip(yc,w));vp=math.fsum(float(a*a*b) for a,b in zip(pc,w))
    cov=math.fsum(float(a*b*c) for a,b,c in zip(yc,pc,w))
    if np.ptp(y[w>0])==0:vy=0.
    if np.ptp(p[w>0])==0:vp=0.
    if vy==0:return None
    a=math.sqrt(vp/vy);r=cov/math.sqrt(vy*vp) if vp>0 else math.nan
    bias2=(pm-ym)**2/vy
    mse=math.fsum(float((v-u)**2*z) for u,v,z in zip(y,p,w))
    nse=1-mse/vy
    beta=max(0,cov/vp) if vp>0 else 0.
    optimal_r=max(0,r) if math.isfinite(r) else 0.
    ceiling=optimal_r**2-bias2
    oracle=optimal_r**2
    unit=2*r-1-bias2 if vp>0 else math.nan
    if vp>0:
        assert abs(nse-(2*r*a-a*a-bias2))<=1e-9*(1+abs(nse))
        pp=pm+beta*pc
        independent=1-math.fsum(float((v-u)**2*z) for u,v,z in zip(y,pp,w))/vy
        assert abs(ceiling-independent)<=1e-9*(1+abs(ceiling))
    return dict(NSE=nse,RMSE=math.sqrt(mse),bias=pm-ym,amplitude_ratio=a,correlation=r,bias_penalty=bias2,
        gain_beta_opt=beta,positive_shape_amplitude_opt=optimal_r,
        fixed_mean_amplitude_ceiling_NSE=ceiling,observed_mean_affine_ceiling_NSE=oracle,
        unit_amplitude_NSE=unit,amplitude_only_room=ceiling-nse,
        unit_amplitude_change=unit-nse,remaining_after_affine=1-oracle,
        amplitude_gradient_at1_normalized=2*(a*a-(r*a if vp>0 else 0)))


def center_frame(g):
    ys=[];ps=[];weights=[];parts=[]
    for _,m in g.groupby(pd.to_datetime(g.date).dt.to_period('M')):
        if len(m)<2:continue
        w=m.read_count.to_numpy(float);w=w/w.sum()
        y=m.observed.to_numpy(float);p=m.prediction.to_numpy(float)
        ys.extend(y-np.dot(w,y));ps.extend(p-np.dot(w,p));weights.extend(w);parts.append(m.index)
    return np.array(ys),np.array(ps),np.array(weights),parts


def main():
    lease('amplitude_diagnostic',1,'cpu')
    assert read(ROOT/'evidence/independent_actual_effect_metrics.json')['passed']
    # Discriminating software controls: correct phase can be rescued by gain;
    # wrong phase with full amplitude is harmed by insisting on unit amplitude.
    t=np.arange(100);y=np.sin(2*np.pi*t/100)
    good=moments(y,.2*y,np.ones(100));bad=moments(y,np.cos(2*np.pi*t/100),np.ones(100))
    assert abs(good['gain_beta_opt']-5)<1e-12 and abs(good['fixed_mean_amplitude_ceiling_NSE']-1)<1e-12
    assert abs(bad['observed_mean_affine_ceiling_NSE'])<1e-12 and abs(bad['unit_amplitude_NSE']+1)<1e-12
    out=ROOT/'outputs/amplitude_diagnostic';out.mkdir(exist_ok=True)
    selection=read(ROOT/'outputs/frozen_selection.json');jointselected=read(ROOT/'outputs/frozen_joint_selection.json')['selected'];names=[]
    for fold in ['F23','F24']:
        for family in read(ROOT/'config/design.json')['families']:
            for task in ['daily','monthly']:
                names.append((f'{fold}_{task}_{family}_c{selection["selected"][family+"_"+task]}_s1729',task))
        names.extend([(f'{fold}_daily_CatBoost_c4_seedmean','daily'),(f'{fold}_monthly_CatBoost_c2_seedmean','monthly')])
        names.extend((f'joint_{fold}_{family}_c{jointselected[family]}_seedmean',task)
                     for family in ['XGBoost','Transformer','GraphTCN'] for task in ['daily','monthly'])
    metrics=pd.read_csv(ROOT/'outputs/evaluation/all_station_metrics.csv');rows=[];readouts=[]
    for name,task in names:
        file=ROOT/'outputs/evaluation'/f'{name}_{task}_frozen.parquet'
        q=pd.read_parquet(file);q.date=pd.to_datetime(q.date)
        valid=set(metrics[(metrics.configuration.eq(name))&metrics.task.eq(task)&metrics.nse_eligible.eq(True)].station_key)
        for station,g in q[q.station_key.isin(valid)].groupby('station_key'):
            z=moments(g.observed,g.prediction,np.ones(len(g)))
            if z is not None:rows.append(dict(configuration=name,task=task,station_key=station,scope='level',**z))
            if task=='daily':
                y,p,w,_=center_frame(g);z=moments(y,p,w) if len(y) else None
                if z is not None:rows.append(dict(configuration=name,task=task,station_key=station,scope='month_centered',**z))
        readouts.append(dict(configuration=name,task=task,rows=len(q),zero_predictions=int(q.prediction.eq(0).sum()),
            minimum_prediction=float(q.prediction.min()),maximum_prediction=float(q.prediction.max()),sha256=sha(file)))
    station=pd.DataFrame(rows);assert not station.empty
    station.to_csv(out/'station_shape_gain_profiles.csv',index=False,encoding='utf-8-sig')
    agg=station.groupby(['configuration','task','scope']).median(numeric_only=True).reset_index()
    agg['stations']=station.groupby(['configuration','task','scope']).size().to_numpy()
    agg.to_csv(out/'shape_gain_summary.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(readouts).to_csv(out/'output_range_and_zero_activity.csv',index=False,encoding='utf-8-sig')
    # Does averaging registered seeds explain small amplitude? Exact covariance
    # identity: mean(parent variance) = variance(mean) + disagreement variance.
    seedrows=[]
    for name,task in names:
        if '_seedmean' not in name:continue
        avg=pd.read_parquet(ROOT/'outputs/evaluation'/f'{name}_{task}_frozen.parquet').sort_values(['station_key','date']).reset_index(drop=True)
        parents=[]
        for seed in [1729,1730,1731]:
            q=pd.read_parquet(ROOT/'outputs/evaluation'/f'{name.replace("_seedmean","_s"+str(seed))}_{task}_frozen.parquet').sort_values(['station_key','date']).reset_index(drop=True)
            assert q[['station_key','date']].equals(avg[['station_key','date']]);parents.append(q.prediction.to_numpy(float))
        matrix=np.column_stack(parents);valid=set(metrics[metrics.configuration.eq(name)&metrics.task.eq(task)&metrics.nse_eligible].station_key)
        for station,g in avg[avg.station_key.isin(valid)].groupby('station_key'):
            for scope in (['level','month_centered'] if task=='daily' else ['level']):
                ix=g.index.to_numpy();v=matrix[ix];w=np.ones(len(ix))
                if scope=='month_centered':
                    parts=[];allw=[]
                    for _,m in g.groupby(pd.to_datetime(g.date).dt.to_period('M')):
                        if len(m)<2:continue
                        mw=m.read_count.to_numpy(float);mw/=mw.sum();pv=matrix[m.index]
                        parts.append(pv-np.sum(pv*mw[:,None],axis=0));allw.extend(mw)
                    if not parts:continue
                    v=np.concatenate(parts);w=np.asarray(allw)
                w=w/w.sum();v=v-np.sum(w[:,None]*v,axis=0)
                vv=np.sum(w[:,None]*v*v,axis=0);mean=v.mean(axis=1);vm=np.dot(w,mean*mean)
                disagreement=np.dot(w,((v-mean[:,None])**2).mean(axis=1))
                assert abs(vv.mean()-vm-disagreement)<1e-10*(1+vv.mean())
                seedrows.append(dict(configuration=name,task=task,station_key=station,scope=scope,
                    rms_parent_sd=float(np.sqrt(vv.mean())),mean_prediction_sd=float(np.sqrt(vm)),
                    sd_retained_fraction=float(np.sqrt(vm/vv.mean())) if vv.mean()>0 else np.nan,
                    disagreement_variance=float(disagreement)))
    pd.DataFrame(seedrows).to_csv(out/'seed_averaging_variance.csv',index=False,encoding='utf-8-sig')
    # Full-history TRAINING-only joint data objective. Output-space null mode
    # keeps every official month mean unchanged. Head-margin mode is a feasible
    # common output-head/tree-margin transformation; no inference rerun needed.
    from mltn.data import Inputs
    from joint import support,objective
    d=Inputs();profiles=[];directions=[]
    for fold,cut in [('F23',2022),('F24',2023)]:
        q,m,h,B,H,_=support(d,cut)
        for family in ['XGBoost','Transformer','GraphTCN']:
            for seed in [1729,1730,1731]:
                name=f'joint_{fold}_{family}_c{jointselected[family]}_s{seed}'
                folder=ROOT/'jobs'/name;ck=pickle.load((folder/'checkpoint.pkl').open('rb'))
                saved=pd.read_parquet(folder/'training_prediction_daily.parquet')
                assert saved[['station_key','date']].equals(q[['station_key','date']])
                p=saved.prediction.to_numpy(float);o,_=objective(m,h,B,H,**ck['identity'])
                # Saved GPU predictions can be FP32. Profile arithmetic and
                # the month-null identity must be computed entirely in FP64.
                mean=pd.Series(p).groupby([saved.station_key,saved.year,saved.month]).transform('mean').to_numpy()
                pc=p-mean
                assert np.max(np.abs(B@pc))<=1e-10
                j0=o.value_gradient(p)[0];monthly=.4*np.dot(o.wm,(B@p-o.ym)**2)
                hz=o.center(H@pc);hy=o.center(o.yh);den=np.dot(o.wh,hz*hz)
                optimum=max(0,np.dot(o.wh,hz*hy)/den) if den>0 else 0.
                # Raw margin transformations are attainable by scaling the
                # final linear head and changing its bias (or booster scores).
                raw=np.log(p) if family=='XGBoost' else np.log(np.expm1(p))
                center=float(raw.mean());link_deriv=p if family=='XGBoost' else -np.expm1(-p)
                headv=link_deriv*(raw-center)
                gm=.8*(B.T@(o.wm*(B@p-o.ym)))
                gh=.2*(H.T@o.transpose_center(o.wh*o.center(H@p-o.yh)))
                dm=float(np.dot(gm,headv));dh=float(np.dot(gh,headv))
                # Finite difference follows the exact nonlinear link.
                def head(a):
                    z=center+a*(raw-center)
                    return np.exp(z) if family=='XGBoost' else np.logaddexp(0,z)
                slopes=[]
                for eps in [1e-3,1e-4,1e-5]:
                    fd=(o.value_gradient(head(1+eps))[0]-o.value_gradient(head(1-eps))[0])/(2*eps)
                    assert abs(fd-dm-dh)<=(1e-6 if eps<=1e-4 else 1e-5)*(1+abs(dm+dh));slopes.append(float(fd))
                directions.append(dict(job=name,J=j0,monthly=monthly,hf=j0-monthly,
                    output_null_monthly_derivative=float(np.dot(gm,pc)),output_null_hf_derivative=float(np.dot(gh,pc)),
                    output_null_hf_optimal_gain=float(optimum),head_monthly_derivative=dm,head_hf_derivative=dh,
                    head_total_derivative=dm+dh,block_opposition=bool(dm*dh<0),
                    softplus_derivative_minimum=float(link_deriv.min()) if family!='XGBoost' else np.nan,
                    finite_difference=slopes))
                for a in [0.,.5,.75,1.,1.25,1.5,2.,float(optimum)]:
                    ptest=p+(a-1)*pc;j=o.value_gradient(ptest)[0]
                    mn=.4*np.dot(o.wm,(B@ptest-o.ym)**2)
                    assert abs(mn-monthly)<=1e-8*(1+abs(monthly))
                    profiles.append(dict(job=name,mode='official_month_mean_preserving',gain=a,J=j,monthly=mn,hf=j-mn,
                        negative_concentrations=int((ptest<0).sum()),identity='diagnostic output-space mode; not necessarily a shared-model parameter mode'))
                for a in [.5,.75,1.,1.25,1.5,2.]:
                    ptest=head(a);j=o.value_gradient(ptest)[0];mn=.4*np.dot(o.wm,(B@ptest-o.ym)**2)
                    profiles.append(dict(job=name,mode='shared_head_margin_gain',gain=a,J=j,monthly=mn,hf=j-mn,
                        negative_concentrations=0,identity='feasible shared final head/margin direction; fixed existing waveform and no regularization value included'))
    pd.DataFrame(directions).to_csv(out/'joint_training_amplitude_directions.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(profiles).to_csv(out/'joint_training_amplitude_profiles.csv',index=False,encoding='utf-8-sig')
    assert len(directions)==18
    write(out/'receipt.json',dict(passed=True,station_rows=len(rows),registered_readouts=len(readouts),joint_training_checkpoints=18,
        synthetic_controls=dict(aligned_underamplitude=good,wrong_phase_full_amplitude=bad),
        independent_method='math.fsum station moments; exact analytic gain identities; true aggregate objective with 3-step head finite differences',
        train_oracle='training data only for joint profiles',evaluation_oracle='descriptive labels-assisted fixed-waveform bound, never used as forecast, new point or model selection',
        fit_calls=0,scientific_predictions_changed=False,
        parameter_claim='shared-head direction is feasible; arbitrary per-month output null direction need not be representable by shared parameters',
        regularization='data objective only; no numerical optimality assertion and no automatic weight/loss change',
        source_identity={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'mltn/objectives.py',ROOT/'mltn/models.py',ROOT/'train.py',ROOT/'joint.py']}))
    print('AMPLITUDE_DIAGNOSIS_PASSED',len(rows),18,flush=True)


if __name__=='__main__':main()
