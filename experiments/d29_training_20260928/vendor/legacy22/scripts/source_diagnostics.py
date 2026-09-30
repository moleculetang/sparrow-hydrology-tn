"""Training-only weighted source directions and selected full four-source accounting."""
import argparse,time,gc
import native_runtime as rt
from campaign_model import *
from source_corrected import embed,GROUPS
from routing import route
from balanced_tags import balanced_scan

def directions(short):
    selected=rt.read(RUN/'data/selected.json');tag=selected[short+'_R'];record=rt.read(RUN/'outputs'/tag/'model.json');job=record['job'];rt.label_barrier(job['fold'])
    m=for_job(dict(job,kind='SOURCE_SEPARATE'));x=embed(record['parameters'],'D29_BE','SOURCE_SEPARATE');y=m.predict(x,m.meta);columns=[];half=[];calls=1
    for factor,target in [(1.,columns),(.5,half)]:
        for i in range(34):
            step=2e-5*max(1,abs(x[i]))*factor;lo,hi=m.bounds[i];p=x.copy();n=x.copy()
            if lo<=x[i]-step and x[i]+step<=hi:
                p[i]+=step;n[i]-=step;col=(m.predict(p,m.meta)-m.predict(n,m.meta))/(2*step)
            else:
                sign=1 if x[i]-step<lo else -1;p[i]+=sign*step;n[i]+=sign*2*step;col=sign*(-3*y+4*m.predict(p,m.meta)-m.predict(n,m.meta))/(2*step)
            target.append(col);calls+=2
    w=np.sqrt(m.weight)[:,None];jac=w*np.column_stack(columns);other=w*np.column_stack(half);A=jac[:,:30]*m.variable_scale()[:30];Z=jac[:,30:]
    U,s,V=np.linalg.svd(A,full_matrices=False);noise=float(np.linalg.norm((other[:,:30]-jac[:,:30])*m.variable_scale()[:30],ord=2));cut=max(max(A.shape)*np.finfo(float).eps*s[0],5*noise);rank=int((s>cut).sum());remainder=Z-U[:,:rank]@(U[:,:rank].T@Z);rows=[]
    for i,name in enumerate(['fertilizer','manure','BNF','deposition']):
        norm=np.linalg.norm(Z[:,i]);err=np.linalg.norm(Z[:,i]-other[:,30+i]);defined=norm>max(10*err,1e-12)
        unexplained=float(np.linalg.norm(remainder[:,i])**2/norm**2) if defined else None
        rows.append(dict(source=name,weighted_norm=float(norm),step_error=float(err),reliable=bool(defined),unexplained_fraction=unexplained,explained_fraction=None if unexplained is None else 1-unexplained))
    np.savez_compressed(RUN/'reports'/f'{short}_source_directions.npz',weighted_jacobian=jac,half_step_jacobian=other,residual_source_directions=remainder,singular_values=s)
    # Direction cosines describe local process/source compensation, not causal
    # attribution or coefficient uncertainty. Use no prior in this geometry.
    names=['fertilizer','manure','BNF','deposition'];cosines=[]
    for i,name in enumerate(names):
        for j,parameter in enumerate(m.names):
            a=Z[:,i];b=jac[:,j];den=np.linalg.norm(a)*np.linalg.norm(b)
            cosines.append(dict(source=name,parameter=parameter,weighted_cosine=float(a@b/den) if den>0 else np.nan,source_direction_reliable=rows[i]['reliable']))
    pd.DataFrame(cosines).to_csv(RUN/'reports'/f'{short}_source_process_direction_cosines.csv',index=False)
    rt.write(RUN/'reports'/f'{short}_source_directions.json',dict(status='PASS',rows=rows,rank=rank,cutoff=cut,numerical_noise=noise,singular_values=s.tolist(),remaining_source_singular_values=np.linalg.svd(remainder,compute_uv=False).tolist(),calls=calls,notes='No prior in projection; local diagnostic, not a fit cancellation or source identification test.'))

def tags(tag):
    out=RUN/'outputs'/tag;rec=rt.read(out/'model.json');rt.label_barrier(rec['job']['fold']);m=for_job(rec['job']);d=m.data;x=np.array(rec['parameters']);base=m.ledger(x)
    mult=np.repeat(np.exp(x[30]),4)
    raw=np.array(d.source_tags,copy=True);corr=raw*mult;inp=np.zeros((*d.contact.shape,4));inp[d.starts]=corr
    with torch.no_grad():h,s,f,k=[v.numpy() for v in m.flux_parameters(torch.tensor(x))]
    ti=np.zeros_like(h);ti[d.starts]=base.get('corrected_source_monthly',d.source)
    v=balanced_scan(h,s,f,d.lower_release,inp,ti,m.demand);errors={}
    assert v[8]<=1e-6 and v[9]>=-1e-7,(v[8],v[9])
    frames=[];network=[];nd,nr=d.contact.shape;end=d.stops-1
    for name,i in [('fast',0),('slow',1),('M',3),('L',4),('uptake',5),('mineral_loss',6)]:
        errors[name]=float(np.max(abs(v[i].sum(-1)-base[name])))
    assert max(errors.values())<=1e-6,errors
    total_channel=np.zeros_like(base['channel_loss']);total_terminal=np.zeros_like(base['terminal']);total_stocks=np.zeros_like(base['reservoir_stocks'])
    for j,name in enumerate(['fertilizer','manure','BNF','deposition']):
        row=pd.DataFrame(dict(year=np.repeat(d.months.year,nr),month=np.repeat(d.months.month,nr),global_reach_id=np.tile(d.global_reach_ids,len(d.months)),source=name,original_input_kg=raw[:,:,j].ravel(),corrected_input_kg=corr[:,:,j].ravel()))
        for key,i in [('fast',0),('slow',1),('M',3),('L',4),('uptake',5),('mineral_loss',6)]:row[key+'_kg']=(v[i][end,:,j] if key in ['M','L'] else d.monthly_sum(v[i][:,:,j])).ravel()
        rr=route(d,v[0][:,:,j]+v[1][:,:,j],vf=float(x[2]));row['channel_loss_kg']=d.monthly_sum(rr['channel_removed']).ravel();frames.append(row)
        network.append(pd.DataFrame(dict(year=d.months.year,month=d.months.month,source=name,terminal_kg=d.monthly_sum(rr['terminal']),reservoir_end_kg=rr['stocks'][end].sum(1))))
        total_channel+=rr['channel_removed'];total_terminal+=rr['terminal'];total_stocks+=rr['stocks'];del rr
    for key,a,b in [('channel',total_channel,base['channel_loss']),('terminal',total_terminal,base['terminal']),('reservoir',total_stocks,base['reservoir_stocks'])]:errors[key]=float(np.max(abs(a-b)))
    assert max(errors.values())<=1e-6,errors
    pd.concat(frames,ignore_index=True).to_parquet(out/'full_monthly_source_ledger.parquet',index=False);pd.concat(network,ignore_index=True).to_parquet(out/'full_monthly_source_network.parquet',index=False)
    rt.write(out/'full_source_audit.json',dict(status='PASS',sources=4,reaches=nr,days=nd,multipliers=mult.tolist(),max_errors=errors,maximum_rounding_adjustment=v[7],per_source_balance=v[8],minimum=v[9],full_aggregate_ledger_calls=1,full_tagged_land_calls=1,full_single_source_network_calls=4,process=rt.process(os.getpid())))
    print('PASS full source',tag,errors,flush=True)

def replay(tag):
    rec=rt.read(RUN/'outputs'/tag/'model.json');job=rec['job'];short=tag[:3];selected=rt.read(RUN/'data/selected.json');baseline=rt.read(RUN/'outputs'/selected[short+'_R']/'model.json');rt.label_barrier(job['fold'])
    m=for_job(job);design=dict(m.design,observation_registry_file='data/prediction_registry.json',observation_registry_hash=rt.sha(RUN/'data/prediction_registry.json'));m.registry=rt.read(RUN/'data/prediction_registry.json')['records'];m._daily_meta_cache={}
    meta=pd.read_parquet(RUN/'data/prediction_calendar.parquet');end=rt.read(RUN/'configs/folds.json')[job['fold']]['end_year'];meta=meta[meta.year.le(end)]
    for mode in ['source_only','process_only']:
        x=np.array(rec['parameters']);x[:30]=baseline['parameters'] if mode=='source_only' else x[:30]
        if mode=='process_only':x[30:]=0
        with torch.no_grad():a=m.daily_boundary(torch.tensor(x),meta)
        i=a['record'].numpy();di=a['day_index'].numpy();mass=a['mass'].numpy();water=a['water'].numpy();path=RUN/'outputs'/tag/'replays';path.mkdir(exist_ok=True)
        pd.DataFrame(dict(station_key=meta.station_key.to_numpy()[i],date=m.data.dates[di],mass_kg_day=mass,water_m3_day=water,concentration_mg_l=1000*mass/water)).to_parquet(path/(mode+'.parquet'),index=False)
        physical=m.ledger(x);flux=max(1.,(physical['fast']+physical['slow']).sum());ok=physical['local_balance_max_kg']<=1e-6 and abs(physical['network_balance_kg'])<=flux*1e-10
        rt.write(path/(mode+'.json'),dict(parameters=x.tolist(),physical=bool(ok),local=physical['local_balance_max_kg'],network=physical['network_balance_kg'],full_history_calls=2,role='fixed-parameter diagnostic, no fitting or point selection'))
        assert ok

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('operation',choices=['directions','tags','replay']);p.add_argument('target');a=p.parse_args();globals()[a.operation](a.target)
