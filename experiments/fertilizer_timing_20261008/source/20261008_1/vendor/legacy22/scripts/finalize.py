"""Freeze ALL paths before evaluation; run separate recomputations and reports."""
import os,sys,time,subprocess,traceback
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import native_runtime as rt
R=rt.RUN

def run(script,*args):
    peak=max(rt.read(R/'reports/launch_validation.json')['peak_reservations_gib'].values())
    deadline=rt.read(R/'work/experiment_clock.json')['delivery_deadline'];paused=False
    while True:
        now=rt.resources()
        if now['ram_percent']>=90 or (now['cpu_percent'] is not None and now['cpu_percent']>=90):paused=True
        elif now['ram_percent']<85 and now['cpu_percent'] is not None and now['cpu_percent']<85:paused=False
        if not paused and rt.admission(now,peak,[]):break
        if time.time()>=deadline:
            rt.log('resource_events.jsonl',dict(event='POSTPROCESS_BUDGET_OMISSION',script=script,args=args));return False
        rt.log('resource_events.jsonl',dict(event='POSTPROCESS_WAIT_RESOURCE',script=script,resources=now));time.sleep(2)
    log=R/'work'/('_'.join([Path(script).stem,*args])+'.log')
    with log.open('ab') as f:
        p=subprocess.run([sys.executable,'-B',str(R/'scripts'/script),*args],cwd=R,stdout=f,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
    rt.log('resource_events.jsonl',dict(event='POSTPROCESS_EXIT',script=script,args=args,exit_code=p.returncode,resources=rt.resources()))
    return p.returncode==0

def main():
    import numpy as np,pandas as pd
    jobs=rt.read(R/'configs/jobs.json');selected={};paths=[];freeze={};missing=[]
    for job in jobs:
        out=R/'outputs'/job['tag'];a=rt.read(out/'audit.json') if (out/'audit.json').exists() else {}
        row=dict(tag=job['tag'],kind=job['kind'],reused=bool(job.get('reuse_from')),status=a.get('status','MISSING'),physical=a.get('physical_reasonable'),numerical=a.get('numerical_sufficient'),objective=a.get('objective'),pg=a.get('pg'))
        if (out/'model.json').exists():
            model=rt.read(out/'model.json');row.update(model.get('terms',{}));row['eta']=model['parameters'][-1] if job['kind']=='STATE_MODULATED' else 0.
            row['original_prior']=row.get('original_prior',row.get('prior'));row['new_prior']=row.get('new_prior',0.)
        paths.append(row)
        for name in ['audit.json','model.json','daily_station_mass_water.parquet','daily_physical_ledger.npz','predictions.parquet']:
            if (out/name).exists():freeze[str((out/name).relative_to(R))]=rt.sha(out/name)
    for fold in ['F23','F24']:
        for arm in ['R','X']:
            key=fold+'_'+arm;candidates=[r for r in paths if r['tag'].startswith(key+'_') and r['physical'] and r['objective'] is not None]
            if candidates:selected[key]=min(candidates,key=lambda a:a['objective'])['tag']
            else:missing.append(key)
    rt.write(R/'data/selected.json',selected);rt.write(R/'data/prediction_freeze.json',dict(created=time.time(),selection='training objective among physical legal saved points only',files=freeze))
    pd.DataFrame(paths).to_csv(R/'reports/path_summary.csv',index=False)
    stability=[]
    for fold in ['F23','F24']:
        for arm in ['R','X']:
            tags=[f'{fold}_{arm}_s{i}' for i in [0,1]]
            if not all((R/'outputs'/t/'model.json').exists() for t in tags):continue
            models=[rt.read(R/'outputs'/t/'model.json') for t in tags];aa=[rt.read(R/'outputs'/t/'audit.json') for t in tags]
            params=[np.array(m['parameters']) for m in models];gap=abs(models[0]['objective']-models[1]['objective'])
            stability.append(dict(fold=fold,arm=arm,both_numerically_sufficient=all(a['numerical_sufficient'] for a in aa),both_physical=all(a['physical_reasonable'] for a in aa),objective_gap=gap,relative_objective_gap=gap/(1+abs(min(m['objective'] for m in models))),max_absolute_parameter_difference=float(abs(params[0]-params[1]).max()),eta_difference=float(params[1][-1]-params[0][-1]) if arm=='X' else 0.))
    pd.DataFrame(stability).to_csv(R/'reports/start_stability.csv',index=False)
    audits={tag:run('independent_audit.py',tag) for tag in selected.values()}
    directions={fold:run('direction_diagnostic.py',fold) for fold in ['F23','F24'] if fold+'_R' in selected}
    evaluation=None
    if any(f+'_R' in selected and f+'_X' in selected for f in ['F23','F24']):
        from evaluation import main as evaluate
        evaluation=evaluate(selected)
    evaluation_audit=run('audit_evaluation.py') if evaluation is not None else False
    # Reconstruct month-level and centered contributions from frozen train predictions.
    components=[];compensation=[]
    for job in jobs:
        out=R/'outputs'/job['tag']
        if not (out/'model.json').exists():continue
        train=pd.read_parquet(R/'data/folds'/job['fold']/'train.parquet')
        pred=pd.read_parquet(out/'training_predictions.parquet')[['observation_id','prediction_mg_l']]
        z=train.merge(pred,on='observation_id',validate='one_to_one');z['residual']=z.prediction_mg_l-z.tn_mg_l
        level=0.;dynamic=0.
        for _,g in z.groupby(['station_key','year','month']):
            w=g.fit_weight.to_numpy();e=g.residual.to_numpy();mean=float(w@e/w.sum());level+=.5*w.sum()*mean*mean;dynamic+=.5*float(w@((e-mean)**2))
        rec=rt.read(out/'model.json');assert abs(level+dynamic-rec['terms']['data'])<1e-10*(1+abs(rec['terms']['data']))
        terms=dict(rec['terms']);terms.setdefault('original_prior',terms['prior']);terms.setdefault('new_prior',0.)
        components.append(dict(tag=job['tag'],monthly_level=level,within_month_dynamic=dynamic,**terms))
    for fold in ['F23','F24']:
        if fold+'_R' in selected and fold+'_X' in selected:
            r=rt.read(R/'outputs'/selected[fold+'_R']/'model.json');x=rt.read(R/'outputs'/selected[fold+'_X']/'model.json')
            for i,name in enumerate(r['names']):compensation.append(dict(fold=fold,parameter=name,baseline=r['parameters'][i],candidate=x['parameters'][i],change=x['parameters'][i]-r['parameters'][i]))
    pd.DataFrame(components).to_csv(R/'reports/training_components.csv',index=False);pd.DataFrame(compensation).to_csv(R/'reports/parameter_compensation.csv',index=False)
    if selected:
        from physical_comparison import main as compare_physics
        compare_physics(selected)
    # Old fixed replay is ONLY a F24 descriptive bridge, never a training input.
    bridge=None
    if 'F24_R' in selected:
        from evaluation import metrics,event_scores
        src=R.parent/'20260920_3/outputs/H1_D29/daily_station.parquet'
        if src.exists():
            registered=rt.read(R.parent/'20260920_4/data/evaluation/additional_sources.json')['D29_sha256']
            assert rt.sha(src)==registered,'OLD_REPLAY_HASH_CHANGED'
            assert rt.read(src.parent/'summary.json')['gate']['pass'],'OLD_REPLAY_PHYSICAL_FAILURE'
            old=pd.read_parquet(src);new=pd.read_parquet(R/'outputs'/selected['F24_R']/'daily_station_mass_water.parquet').rename(columns={'concentration_mg_l':'p'})
            obs=pd.read_parquet(R/'data/heldout_labels/2024_days.parquet');rows=[]
            for label,pred in [('旧参数H1回放',old),('原D29重新校准',new)]:
                z=obs.merge(pred[['station_key','date','p']],on=['station_key','date'],validate='one_to_one');assert len(z)==len(obs)
                for s,g in z.groupby('station_key'):
                    eligible=bool(len(g)>=30 and g.date.dt.month.nunique()>=3 and np.var(g.y)>0);stat=metrics(g.y,g.p)
                    if not eligible:stat['nse']=np.nan
                    rows.append(dict(arm=label,station_key=s,n_months=g.date.dt.month.nunique(),nse_eligible=eligible,**stat))
            pd.DataFrame(rows).to_csv(R/'reports/F24_replay_refit_bridge.csv',index=False);bridge=dict(source=str(src),sha256=rt.sha(src),not_F23_baseline=True)
    # Preserve a complete computational accounting distinction for reused paths.
    counts=[]
    for j in jobs:
        path=R/'work/jobs'/j['tag']/'status.json';s=rt.read(path) if path.exists() else {}
        counts.append(dict(tag=j['tag'],new_fit_calls=0 if j.get('reuse_from') else s.get('calls',0),active_seconds=0 if j.get('reuse_from') else s.get('active_seconds',0),new_primary_audit_calls=5 if (R/'outputs'/j['tag']/'audit.json').exists() else 0))
    rt.write(R/'reports/computation_accounting.json',dict(paths=counts,preflight_calls=202,extended_acceptance_calls=15,initial_F24_s0_audit_repeated_calls=5,directions={f:rt.read(R/'reports'/f'direction_{f}.json').get('full_history_calls',0) for f in directions},selected_independent_calls=3*len(audits),note='Inherited prior-fit call totals in reused audit metadata are historical, not new campaign calls. Failed attempts and checkpoint charges remain in per-path logs.'))
    verdict=dict(status='COMPLETE' if not missing and all(audits.values()) and all(directions.values()) and evaluation_audit else 'COMPLETE_WITH_CONDITIONAL_GAPS',selected=selected,missing=missing,independent_processes=audits,independent_evaluation=evaluation_audit,directions=directions,replay_bridge=bridge,all_selected_numerically_sufficient=all(rt.read(R/'outputs'/t/'audit.json').get('numerical_sufficient',False) for t in selected.values()),wall_hours=(time.time()-rt.read(R/'work/experiment_clock.json')['started'])/3600)
    rt.write(R/'reports/completion.json',verdict)
    from write_reports import main as write_reports
    write_reports()
    # Seal deliverables, not mutable controller logs (controller adds terminal event).
    files={str(p.relative_to(R)):rt.sha(p) for folder in ['scripts','configs','reports','outputs','evidence','data'] for p in (R/folder).rglob('*') if p.is_file() and p.name!='delivery_manifest.json'}
    rt.write(R/'data/delivery_manifest.json',dict(files=files,created=time.time(),mutable_logs='work/*.jsonl retained outside immutable seal'))
    for name,h in files.items():assert rt.sha(R/name)==h,name
    print(verdict,flush=True)
if __name__=='__main__':
    try:main()
    except Exception:
        rt.write(R/'reports/finalization_failure.json',dict(error=traceback.format_exc(),time=time.time()));raise
