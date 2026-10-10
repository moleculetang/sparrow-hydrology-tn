"""Frozen-model local sensitivity and response/accounting diagnostics, never selection rules."""
import argparse,gc,time
import native_runtime as rt
from campaign_model import *
from regional_response import embed,numpy_delta

def directions(short):
    selected=rt.read(RUN/'data/selected.json');tag=selected[short+'_U'];record=rt.read(RUN/'outputs'/tag/'model.json');job=record['job'];rt.label_barrier(job['fold'])
    m=for_job(dict(job,kind='REGIONAL_L3'));x=embed(record['parameters'],'SOURCE_UNIFIED','REGIONAL_L3');y=m.predict(x,m.meta);sets=[];calls=1;propagation=[]
    for factor in [1.,.5]:
        columns=[]
        for i in range(len(x)):
            step=2e-5*max(1,abs(x[i]))*factor;lo,hi=m.bounds[i];p=x.copy();n=x.copy()
            if lo<=x[i]-step and x[i]+step<=hi:p[i]+=step;n[i]-=step;col=(m.predict(p,m.meta)-m.predict(n,m.meta))/(2*step)
            else:
                sign=1 if x[i]-step<lo else -1;p[i]+=sign*step;n[i]+=sign*2*step;col=sign*(-3*y+4*m.predict(p,m.meta)-m.predict(n,m.meta))/(2*step)
            columns.append(col);calls+=2
            if factor==.5 and i>=31:
                # The same local map perturbation, with all original/source parameters fixed.
                totals=[]
                for point in [p,n]:
                    ledger=m.ledger(point);annual={}
                    for year in np.unique(m.data.dates.year):
                        take=m.data.dates.year==year
                        annual[int(year)]={key:float(ledger[key][take].sum()) for key in ['fast','slow','uptake','mineral_loss','channel_loss','terminal']}
                    totals.append(annual);del ledger;gc.collect();calls+=1
                for year in totals[0]:
                    propagation.append(dict(parameter=m.names[i],year=year,**{key+'_derivative':(totals[0][year][key]-totals[1][year][key])/(2*step) for key in totals[0][year]},training_weighted_station_derivative_norm=float(np.linalg.norm(np.sqrt(m.weight)*col))))
        sets.append(np.sqrt(m.weight)[:,None]*np.column_stack(columns))
    jac,other=sets;sc=m.variable_scale()[:31];a=jac[:,:31]*sc;z=jac[:,31:];u,s,v=np.linalg.svd(a,full_matrices=False)
    noise=float(np.linalg.norm((other[:,:31]-jac[:,:31])*sc,ord=2));cut=max(max(a.shape)*np.finfo(float).eps*s[0],5*noise);rank=int((s>cut).sum());remaining=z-u[:,:rank]@(u[:,:rank].T@z);rows=[];cosines=[]
    for i,name in enumerate(m.names[31:]):
        norm=np.linalg.norm(z[:,i]);err=np.linalg.norm(z[:,i]-other[:,31+i]);reliable=norm>max(10*err,1e-12);fraction=float(np.linalg.norm(remaining[:,i])**2/norm**2) if reliable else None
        rows.append(dict(parameter=name,weighted_norm=float(norm),step_error=float(err),reliable=bool(reliable),unexplained_fraction=fraction,explained_fraction=None if fraction is None else 1-fraction))
        for j,pname in enumerate(m.names[:31]):
            den=np.linalg.norm(z[:,i])*np.linalg.norm(jac[:,j]);cosines.append(dict(new_parameter=name,baseline_parameter=pname,cosine=float(z[:,i]@jac[:,j]/den) if den>0 else None))
    residual_noise=np.linalg.norm(other[:,31:]-z,ord=2);sv=np.linalg.svd(remaining,compute_uv=False);rcut=max(max(remaining.shape)*np.finfo(float).eps*sv[0],5*residual_noise)
    neural=make_model(m.data,m.train,'REGIONAL_N3',m.design);nx=embed(record['parameters'],'SOURCE_UNIFIED','REGIONAL_N3');neural_sets=[]
    for factor in [1.,.5]:
        columns=[]
        for i in range(73,79):
            step=2e-5*factor;p=nx.copy();n=nx.copy();p[i]+=step;n[i]-=step;columns.append(np.sqrt(m.weight)*(neural.predict(p,neural.meta)-neural.predict(n,neural.meta))/(2*step));calls+=2
        neural_sets.append(np.column_stack(columns))
    nz,nz_half=neural_sets;nr=nz-u[:,:rank]@(u[:,:rank].T@nz);neural_rows=[]
    linear=jac*m.variable_scale();lu,ls,lv=np.linalg.svd(linear,full_matrices=False);linear_noise=np.linalg.norm((other-jac)*m.variable_scale(),ord=2);linear_cut=max(max(linear.shape)*np.finfo(float).eps*ls[0],5*linear_noise);linear_rank=int((ls>linear_cut).sum());nr_linear=nz-lu[:,:linear_rank]@(lu[:,:linear_rank].T@nz)
    for i in range(6):
        norm=np.linalg.norm(nz[:,i]);err=np.linalg.norm(nz[:,i]-nz_half[:,i]);reliable=norm>max(10*err,1e-12);unexplained=float(np.linalg.norm(nr[:,i])**2/norm**2) if reliable else None
        neural_rows.append(dict(parameter=neural.names[73+i],weighted_norm=float(norm),step_error=float(err),reliable=bool(reliable),unexplained_fraction=unexplained,explained_fraction=None if unexplained is None else 1-unexplained,linear_parent_explained_fraction=float(1-np.linalg.norm(nr_linear[:,i])**2/norm**2) if reliable else None))
    np.savez_compressed(RUN/'reports'/f'{short}_neural_zero_output_directions.npz',weighted_derivatives=nz,half_step_derivatives=nz_half,residual_directions=nr)
    rt.write(RUN/'reports'/f'{short}_neural_zero_output_directions.json',dict(status='PASS',rows=neural_rows,anchor='unified source baseline; A=V=a=0 and fixed seeded H',hidden_directions='H and a derivatives are zero at V=0 by construction; not interpreted as no neural response',baseline_rank=rank,cutoff=cut,linear_parent_rank=linear_rank,linear_parent_cutoff=float(linear_cut),linear_parent_singular_values=ls.tolist(),interpretation='Overlap with the 55-coordinate linear parent helps distinguish additional local nonlinear directions from reparameterization; not a global equivalence test.'))
    del neural;gc.collect()
    np.savez_compressed(RUN/'reports'/f'{short}_regional_directions.npz',weighted_jacobian=jac,half_step_jacobian=other,residual_directions=remaining,singular_values=s)
    pd.DataFrame(cosines).to_csv(RUN/'reports'/f'{short}_regional_direction_cosines.csv',index=False)
    pd.DataFrame(propagation).to_csv(RUN/'reports'/f'{short}_local_perturbation_propagation.csv',index=False)
    rt.write(RUN/'reports'/f'{short}_regional_directions.json',dict(status='PASS',rows=rows,baseline_rank=rank,cutoff=cut,numerical_noise=noise,singular_values=s.tolist(),remaining_singular_values=sv.tolist(),remaining_cutoff=float(rcut),remaining_rank=int((sv>rcut).sum()),calls=calls,notes='Data residual weighting only; local overlap is not mechanistic identity or prediction success.'))
    print('PASS regional directions',short,flush=True)

def responses():
    selected=rt.read(RUN/'data/selected.json');response=[];parameters=[];budgets=[];propagation=[];sourceyears=[]
    for name,tag in selected.items():
        record=rt.read(RUN/'outputs'/tag/'model.json');x=np.asarray(record['parameters']);job=record['job'];short=name.split('_')[0]
        for pname,value in zip(record['names'],x):parameters.append(dict(model=name,parameter=pname,value=value))
        ledger=pd.read_parquet(RUN/'outputs'/tag/'monthly_physical_ledger.parquet');evaluation_year=2023 if short=='F23' else 2024
        ledger['period']=np.select([ledger.year<=2020,ledger.year<evaluation_year,ledger.year==evaluation_year],['1961-2020','training','evaluation'],default='unlabelled_after_evaluation')
        network=pd.read_parquet(RUN/'outputs'/tag/'network_ledger.parquet')
        for period,g in ledger.groupby('period'):
            fields={k:float(g[k].sum()) for k in ['source_kg','original_source_kg','uptake_kg','demand_kg','mineral_loss_kg','channel_loss_kg','fast_kg','slow_kg']}
            fields['demand_satisfaction']=fields['uptake_kg']/fields['demand_kg'] if fields['demand_kg']>0 else None
            last=g[(g.year==g.year.max())&g.month.eq(12)];net=network[network.year.between(g.year.min(),g.year.max())]
            fields.update(M_end_kg=float(last.M_end_kg.sum()),L_end_kg=float(last.L_end_kg.sum()),terminal_kg=float(net.terminal_kg.sum()),reservoir_end_kg=float(net[net.year.eq(g.year.max())&net.month.eq(12)].reservoir_end_kg.sum()))
            budgets.append(dict(model=name,period=period,source_multiplier=float(np.exp(x[30])),**fields))
        sourcefile=RUN/'outputs'/tag/'full_monthly_source_ledger.parquet'
        if sourcefile.exists():
            sl=pd.read_parquet(sourcefile);flux=['original_input_kg','corrected_input_kg','fast_kg','slow_kg','uptake_kg','mineral_loss_kg','channel_loss_kg'];annual=sl.groupby(['year','global_reach_id','source'])[flux].sum().reset_index();stock=sl[sl.month.eq(12)][['year','global_reach_id','source','M_kg','L_kg']];annual=annual.merge(stock,on=['year','global_reach_id','source'],validate='one_to_one');annual['model']=name;sourceyears.append(annual)
        parent=selected.get(short+'_U')
        if parent and parent!=tag:
            b=pd.read_parquet(RUN/'outputs'/parent/'monthly_physical_ledger.parquet');q=ledger.merge(b,on=['year','month','global_reach_id'],suffixes=('','_base'),validate='one_to_one')
            for year,g in q.groupby('year'):
                for field in ['fast_kg','slow_kg','source_kg','uptake_kg','mineral_loss_kg','M_end_kg','L_end_kg']:
                    diff=g[field]-g[field+'_base'];propagation.append(dict(model=name,year=int(year),level='land',quantity=field,change_sum=float(diff.sum()),absolute_change_sum=float(abs(diff).sum())))
            s=pd.read_parquet(RUN/'outputs'/tag/'daily_station_mass_water.parquet');b=pd.read_parquet(RUN/'outputs'/parent/'daily_station_mass_water.parquet');keys=['observation_id','station_key','date'];q=s.merge(b,on=keys,suffixes=('','_base'),validate='one_to_one')
            for (station,year),g in q.groupby(['station_key',q.date.dt.year]):
                for field in ['mass_kg_day','concentration_mg_l']:
                    diff=g[field]-g[field+'_base'];propagation.append(dict(model=name,station_key=station,year=int(year),level='station',quantity=field,change_sum=float(diff.sum()),absolute_change_sum=float(abs(diff).sum())))
        if not job['kind'].startswith('REGIONAL_'):continue
        m=for_job(job);phi=m.dynamic_basis.numpy();d=m.data
        # Grouped exact sums preserve all days without storing another huge tensor.
        accum={}
        for start in range(0,len(phi),256):
            stop=min(start+256,len(phi));p=phi[start:stop];delta=numpy_delta(x[31:],p,m.psi,m.rank,m.neural);old=np.einsum('trj,j->tr',p,x[21:29]);sat=np.tanh((old+delta)/np.log(10))
            for year in np.unique(d.dates[start:stop].year):
                keep=d.dates[start:stop].year==year;a=delta[keep];z=sat[keep];store=accum.setdefault(int(year),[0,np.zeros(230),np.zeros(230),np.full(230,np.inf),np.full(230,-np.inf),np.zeros(230),np.zeros(230)])
                store[0]+=len(a);store[1]+=a.sum(0);store[2]+=(a*a).sum(0);store[3]=np.minimum(store[3],a.min(0));store[4]=np.maximum(store[4],a.max(0));store[5]+=abs(z).sum(0);store[6]+=(1-z*z).sum(0)
        for year,a in accum.items():
            for r in range(230):response.append(dict(model=name,year=year,reach_id=int(d.global_reach_ids[r]),delta_mean=a[1][r]/a[0],delta_rms=np.sqrt(a[2][r]/a[0]),delta_min=a[3][r],delta_max=a[4][r],mean_abs_tanh=a[5][r]/a[0],mean_tanh_slope=a[6][r]/a[0]))
        del m,phi;gc.collect()
    for filename,data in [('regional_response',response),('parameter_comparison',parameters),('physical_period_summary',budgets),('land_station_propagation',propagation)]:pd.DataFrame(data).to_csv(RUN/'reports'/f'{filename}.csv',index=False)
    if sourceyears:pd.concat(sourceyears,ignore_index=True).to_parquet(RUN/'reports/annual_source_ledger.parquet',index=False)
    rt.write(RUN/'reports/response_diagnostics.json',dict(status='PASS',models=list(selected),no_extra_fits=True,interpretation='Paired jointly calibrated changes, not isolated causal attribution; source-demand and process compensation retained.'))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['directions','responses']);p.add_argument('short',nargs='?');a=p.parse_args();directions(a.short) if a.mode=='directions' else responses()
