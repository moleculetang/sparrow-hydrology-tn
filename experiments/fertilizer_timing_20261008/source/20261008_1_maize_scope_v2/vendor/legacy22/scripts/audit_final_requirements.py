"""Independent event reconstruction and requirement-level final evidence audit."""
import json,time,math
from collections import defaultdict
import numpy as np,pandas as pd
import native_runtime as rt
R=rt.RUN
checks=[]
def check(name,ok,evidence,detail=None):
    checks.append(dict(requirement=name,passed=bool(ok),evidence=evidence,detail=detail))
jobs=rt.read(R/'configs/jobs.json');selected=rt.read(R/'data/selected.json');freeze=rt.read(R/'data/prediction_freeze.json')
check('finite matrix: 12 new fits, 4 reused, all 8 fold/model selections',len(jobs)==16 and sum(not bool(j.get('reuse_from')) for j in jobs)==12 and len(selected)==8 and not freeze['missing'],'configs/jobs.json; data/selected.json')
for fold in ['F23_G_D','T24_G_D_H1']:
    a=rt.read(R/'reports'/f'preflight_{fold}.json');groups=defaultdict(list)
    for v in a['gradient_checks']:groups[(v['eta'],v['index'])].append(v)
    check('full-history gradients '+fold,all(len(v)>=2 and all(t['error']<=t['tolerance'] for t in v[-2:]) for v in groups.values()),f'reports/preflight_{fold}.json',dict(calls=a['full_history_calls'],directions=len(groups)))
    check('exact unit-source nesting '+fold,all(v['objective']==0 and v['prediction']==0 and v['gradient']<1e-12 and max(v['states'].values())==0 for v in a['nested']),f'reports/preflight_{fold}.json')
    check('group prior and derivative embeddings '+fold,all(v['objective_error']<1e-12 and v['gradient_error']<1e-12 for v in a['group_embeddings']),f'reports/preflight_{fold}.json')
    check('reference, depletion, source-demand scaling, monotonicity and causality '+fold,all(v is True for k,v in a['fixtures'].items() if k!='input_adjoint_max_error') and a['fixtures']['input_adjoint_max_error']<1e-7,f'reports/preflight_{fold}.json',a['fixtures'])
for name in ['infrastructure_checks','solver_resume_test','label_counterfactual','sampling_reconstruction','full_source_fixture','evaluation_fixture_tests','evaluation_pipeline_fixture','maintenance_v2','manifest_recovery_v3','evaluation_boolean_repair']:
    a=rt.read(R/'reports'/f'{name}.json');check(name,a['status']=='PASS',f'reports/{name}.json')
for fold in ['F23_G_D','T24_G_D_H1']:
    a=rt.read(R/'reports/launch_by_fold'/f'{fold}.json')
    check('frozen science '+fold,all(rt.sha(R/p)==h for p,h in a['frozen_hashes'].items()),f'reports/launch_by_fold/{fold}.json')
check('frozen prediction bytes',all(rt.sha(R/p)==h for p,h in freeze['files'].items()),'data/prediction_freeze.json')
original=rt.read(R/'evidence/inherited_manifest.json');check('old experiments unchanged',all(rt.sha(p)==h for p,h in original.items()),'evidence/inherited_manifest.json')
replay_count=0;alive=[]
for job in jobs:
    tag=job['tag'];out=R/'outputs'/tag;a=rt.read(out/'audit.json');s=rt.read(R/'work/jobs'/tag/'status.json')
    p=s.get('process')
    if p:
        try:
            if rt.process(p['pid'],p['created'])['alive']:alive.append(tag)
        except (OSError,RuntimeError):pass
    check('path exit '+tag,s['status'] in ['NUMERICALLY_SUFFICIENT','BUDGET_STOPPED','NUMERICALLY_INSUFFICIENT_STATIONARY'] and a['status']=='AUDITED_FIT',f'work/jobs/{tag}/status.json',dict(status=s['status'],pg=a['pg'],sufficient=a['numerical_sufficient']))
    check('physical gate '+tag,a['balance_max_kg']<=1e-6 and abs(a['network_balance_kg'])<=a['network_scale_kg']*1e-10 and min(a['minima'].values())>=-1e-7 and a['uptake_excess_kg']<=1e-7 and a['physical_reasonable'],f'outputs/{tag}/audit.json')
    if not job.get('reuse_from'):
        cp=R/'work/jobs'/tag/'checkpoints';ref=rt.read(cp/'latest.json');state=rt.restore(cp,ref['identity']);cfg=rt.read(R/'configs/campaign.json')['solver'];stage=state['stage']
        check('cumulative budget '+tag,state['calls']<=cfg['cumulative_calls'][stage] and stage<=2 and state['active_seconds']<=cfg['cumulative_hours'][stage]*3600+60,f'work/jobs/{tag}/checkpoints/latest.json',dict(calls=state['calls'],stage=stage,hours=state['active_seconds']/3600))
check('scientific workers exited',not alive,'verified PID plus creation time',alive)
for key,tag in selected.items():
    short,arm=key.split('_');out=R/'outputs'/tag;model=rt.read(out/'model.json');ind=rt.read(R/'reports/independent'/f'{tag}.json');source=rt.read(out/'full_source_audit.json')
    check('selected numerical and independent audit '+key,model['pg']<=1e-5 and ind['status']=='PASS' and abs(model['objective']-ind['objective'])<=1e-8*(1+abs(model['objective'])),f'reports/independent/{tag}.json')
    check('full four-source history '+key,source['reaches']==230 and source['sources']==4 and source['status']=='PASS' and max(source['max_errors'].values())<=1e-6 and source['per_source_balance']<=1e-6 and source['minimum']>=-1e-7,f'outputs/{tag}/full_source_audit.json')
    x=np.array(model['parameters']);eta=np.zeros(4) if arm=='R' else x[30:][{'U':[0,0,0,0],'G':[0,0,0,1],'S':[0,1,2,3]}[arm]]
    prior=.5*(.03/17)*np.mean((eta/math.log(2))**2)
    check('bounds and expanded prior '+key,np.max(abs(eta))<=math.log(4)+1e-12 and abs(model['terms'].get('new_prior',0)-prior)<1e-13,f'outputs/{tag}/model.json')
    if arm!='R':
        for mode in ['source_only','process_only']:
            a=rt.read(out/'replays'/f'{mode}.json');replay_count+=1
            check('four-corner replay '+key+' '+mode,a['physical'] and (out/'replays'/f'{mode}.parquet').exists(),f'outputs/{tag}/replays/{mode}.json')
    a=pd.read_parquet(out/'full_monthly_source_ledger.parquet');mapping=dict(zip(['fertilizer','manure','BNF','deposition'],np.exp(eta)))
    err=np.max(abs(a.original_input_kg*a.source.map(mapping)-a.corrected_input_kg))
    check('original/corrected source mass identity '+key,err<=1e-6 and a.year.min()==1961 and a.year.max()==2024,f'outputs/{tag}/full_monthly_source_ledger.parquet',float(err))
check('exactly 12 fixed-parameter replays',replay_count==12,'outputs/*/replays')
# Independent event construction from frozen raw support and daily predictions.
obs=pd.read_parquet(R/'data/heldout_labels/events/observed_days.parquet');ev=pd.read_parquet(R/'data/heldout_labels/events/events_frozen.parquet');rebuilt={}
for key,tag in selected.items():
    year=2023 if key.startswith('F23') else 2024;pred=pd.read_parquet(R/'outputs'/tag/'daily_station_mass_water.parquet');rows=[]
    for s,sg in obs[obs.date.dt.year==year].groupby('station_key'):
        p=pred[pred.station_key==s].set_index('date').concentration_mg_l;o=sg.set_index('date').y
        for e in ev[(ev.station_key==s)&(ev.start.dt.year==year)&(ev.end.dt.year==year)].itertuples():
            bg=o.loc[(o.index>=e.background_start)&(o.index<e.start)];pk=o.loc[(o.index>=e.start)&(o.index<=e.end)]
            if len(bg)<4 or len(pk)<1 or e.background_start.year!=year:continue
            ob=float(np.median(bg));op=float(np.max(pk));pb=float(np.median(p.loc[bg.index]));pp=float(np.max(p.loc[pk.index]));defined=min(ob,op,pb,pp)>0
            rows.append(dict(station_key=s,event_rank=e.event_rank,event_month=e.start.month,obs_base=ob,obs_peak=op,pred_base=pb,pred_peak=pp,amplitude_error=abs(math.log(pp/pb)-math.log(op/ob)) if defined else np.nan,peak_error=abs(pp-op),base_error=abs(pb-ob),ratio_defined=defined))
    a=pd.DataFrame(rows);saved=pd.read_parquet(R/'outputs'/tag/'event_scores.parquet');z=a.merge(saved,on=['station_key','event_rank'],suffixes=('_direct','_saved'),validate='one_to_one')
    fields=['obs_base','obs_peak','pred_base','pred_peak','amplitude_error','peak_error','base_error'];errors={k:float(np.nanmax(abs(z[k+'_direct']-z[k+'_saved']))) for k in fields}
    check('independent common-date event reconstruction '+key,len(a)==len(saved) and max(errors.values())<1e-12,f'outputs/{tag}/event_scores.parquet',dict(events=len(a),stations=a.station_key.nunique(),errors=errors));rebuilt[key]=a
for folder in [p for p in (R/'reports').iterdir() if p.is_dir() and '-' in p.name]:
    high,low=folder.name.split('-')
    for block in [1,2]:
        d=folder/f'{block}month';summary=pd.read_csv(d/'event_centered_summary.csv');draws=rt.read(d/'bootstrap_draws.json');boot=pd.read_csv(d/'bootstrap_changes.csv')
        for fold in ['F23','F24']:
            r=rebuilt[fold+'_'+low];x=rebuilt[fold+'_'+high];row=summary[summary.fold==fold].iloc[0]
            for field in ['amplitude_error','peak_error','base_error']:
                delta=x.groupby('station_key')[field].median().mean()-r.groupby('station_key')[field].median().mean()
                check(f'event aggregate {folder.name} {block} {fold} {field}',np.isclose(delta,row[field+'_change'],atol=1e-12),str(d.relative_to(R))+'/event_centered_summary.csv',float(delta))
            p=r[['station_key','event_rank','event_month','amplitude_error']].merge(x[['station_key','event_rank','amplitude_error']],on=['station_key','event_rank'],suffixes=('_R','_X'),validate='one_to_one')
            errors=[];b=boot[(boot.fold==fold)&(boot.scale=='events')].set_index('replicate')
            for rep,seq in enumerate(draws[fold]):
                sample=pd.concat([p[p.event_month==m] for m in seq],ignore_index=True)
                if sample.empty:continue
                v=sample.groupby('station_key')[['amplitude_error_R','amplitude_error_X']].median().mean();errors.append(abs((v.iloc[1]-v.iloc[0])-b.loc[rep,'amplitude_error_change']))
            check(f'1000 amplitude resamples {folder.name} {block} {fold}',len(errors)==1000 and max(errors)<1e-12,str(d.relative_to(R))+'/bootstrap_changes.csv',max(errors))
clock=rt.read(R/'work/experiment_clock.json');check('total 72-hour budget',time.time()<=clock['delivery_deadline'],'work/experiment_clock.json',dict(elapsed_hours=(time.time()-clock['started'])/3600))
failed=[c for c in checks if not c['passed']]
rt.write(R/'reports/final_requirements_audit.json',dict(status='PASS' if not failed else 'FAIL',checks=checks,failed=failed,scientific_calls_added=0,time=time.time()))
print('FINAL_REQUIREMENTS',len(checks),'failed',failed,flush=True)
if failed:raise SystemExit(2)
