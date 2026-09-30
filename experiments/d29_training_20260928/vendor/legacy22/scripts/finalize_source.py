"""Freeze finite scientific paths, then diagnostics and evaluation, then report."""
import os,sys,time,subprocess
from pathlib import Path
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import native_runtime as rt
R=rt.RUN
def run(script,*args):
    deadline=rt.read(R/'work/experiment_clock.json')['delivery_deadline'];peak=max(rt.read(R/'reports/launch_validation.json')['peak_reservations_gib'].values())
    paused=False
    while True:
        now=rt.resources()
        if now['ram_percent']>=90 or (now['cpu_percent'] or 0)>=90:paused=True
        elif now['ram_percent']<85 and now['cpu_percent'] is not None and now['cpu_percent']<85:paused=False
        if time.time()>=deadline:return False
        if not paused and rt.admission(now,peak,[]):break
        time.sleep(2)
    with (R/'work'/('_'.join([Path(script).stem,*args])+'.log')).open('ab') as f:
        p=subprocess.run([sys.executable,'-B',str(R/'scripts'/script),*args],cwd=R,stdout=f,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
    rt.log('resource_events.jsonl',dict(event='POSTPROCESS_EXIT',script=script,args=args,exit_code=p.returncode,resources=rt.resources()))
    return p.returncode==0

def main():
    import numpy as np,pandas as pd
    jobs=rt.read(R/'configs/jobs.json');rows=[];selected={};freeze={};missing=[]
    for j in jobs:
        out=R/'outputs'/j['tag'];a=rt.read(out/'audit.json') if (out/'audit.json').exists() else {};row=dict(tag=j['tag'],kind=j['kind'],reused=bool(j.get('reuse_from')),status=a.get('status','MISSING'),physical=a.get('physical_reasonable'),numerical=a.get('numerical_sufficient'),objective=a.get('objective'),pg=a.get('pg'),calls=a.get('calls'))
        if (out/'model.json').exists():
            m=rt.read(out/'model.json');row.update(m['terms']);row['original_prior']=row.get('original_prior',row.get('prior'));row['new_prior']=row.get('new_prior',0.)
        rows.append(row)
        for name in ['model.json','audit.json','daily_station_mass_water.parquet','daily_physical_ledger.npz','predictions.parquet']:
            if (out/name).exists():freeze[str((out/name).relative_to(R))]=rt.sha(out/name)
    for short in ['F23','F24']:
        for arm in ['R','U','G','S']:
            candidates=[r for r in rows if r['tag'].startswith(short+'_'+arm+'_') and r['physical'] and r['objective'] is not None]
            if candidates:selected[short+'_'+arm]=min(candidates,key=lambda r:r['objective'])['tag']
            else:missing.append(short+'_'+arm)
    pd.DataFrame(rows).to_csv(R/'reports/path_summary.csv',index=False);rt.write(R/'data/selected.json',selected)
    rt.write(R/'data/prediction_freeze.json',dict(time=time.time(),selection='training objective among physical legal points; includes nested candidates',files=freeze,missing=missing))
    tasks=[]
    for tag in selected.values():
        tasks.append(dict(script='independent_audit.py',args=[tag],passed=run('independent_audit.py',tag)))
        tasks.append(dict(script='source_diagnostics.py',args=['tags',tag],passed=run('source_diagnostics.py','tags',tag)))
    for short in ['F23','F24']:
        if short+'_R' in selected:tasks.append(dict(script='source_diagnostics.py',args=['directions',short],passed=run('source_diagnostics.py','directions',short)))
        for arm in ['U','G','S']:
            if short+'_'+arm in selected and short+'_R' in selected:
                tag=selected[short+'_'+arm];tasks.append(dict(script='source_diagnostics.py',args=['replay',tag],passed=run('source_diagnostics.py','replay',tag)))
    rt.write(R/'reports/postprocess_tasks.json',tasks)
    import evaluation
    summaries={}
    for low,high in [('R','G'),('R','U'),('U','G'),('G','S'),('R','S')]:
        pair={}
        for short in ['F23','F24']:
            if short+'_'+low in selected and short+'_'+high in selected:
                pair[short+'_R']=selected[short+'_'+low];pair[short+'_X']=selected[short+'_'+high]
        if not pair:continue
        for block in [1,2]:
            if time.time()>=rt.read(R/'work/experiment_clock.json')['delivery_deadline']:
                missing.append(f'evaluation_{high}-{low}_{block}month');continue
            summaries[f'{high}-{low}_{block}month']=evaluation.main(pair,high+'-'+low,block)
    def clean(v):
        if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
        if isinstance(v,(list,tuple)):return [clean(x) for x in v]
        if isinstance(v,(float,np.floating)) and not np.isfinite(v):return None
        return v
    rt.write(R/'reports/evaluation_summary.json',clean(summaries))
    # The report always preserves conditional omissions instead of replacing candidates.
    result=run('report_source.py');return result
if __name__=='__main__':
    if not main():raise SystemExit(2)
