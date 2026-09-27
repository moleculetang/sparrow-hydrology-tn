"""Training selection, immutable prediction freeze, then paired evaluation and independent audit."""
import time
import native_runtime as rt
from finalize_source import run
R=rt.RUN

def clean(v):
    import numpy as np
    if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [clean(x) for x in v]
    if isinstance(v,(float,np.floating)) and not np.isfinite(v):return None
    return v

def main():
    import pandas as pd
    jobs=rt.read(R/'configs/jobs.json');rows=[];selected={};freeze={};missing=[]
    for j in jobs:
        out=R/'outputs'/j['tag'];a=rt.read(out/'audit.json') if (out/'audit.json').exists() else {}
        row=dict(tag=j['tag'],kind=j['kind'],reused=bool(j.get('reuse_from')),status=a.get('status','MISSING'),physical=a.get('physical_reasonable'),numerical=a.get('numerical_sufficient'),objective=a.get('objective'),pg=a.get('pg'),calls=a.get('calls'))
        if (out/'model.json').exists():row.update(rt.read(out/'model.json')['terms'])
        rows.append(row)
        for name in ['model.json','audit.json','daily_station_mass_water.parquet','daily_physical_ledger.npz','predictions.parquet']:
            if (out/name).exists():freeze[(out/name).relative_to(R).as_posix()]=rt.sha(out/name)
    for short in ['F23','F24','S56','S113','S191']:
        for arm in (['U','L1','L3','N3'] if short.startswith('F') else ['U','L3']):
            candidates=[r for r in rows if r['tag'].startswith(short+'_'+arm+'_') and r['physical'] and r['objective'] is not None]
            if candidates:selected[short+'_'+arm]=min(candidates,key=lambda r:r['objective'])['tag']
            else:missing.append(short+'_'+arm)
    pd.DataFrame(rows).to_csv(R/'reports/path_summary.csv',index=False);rt.write(R/'data/selected.json',selected)
    rt.write(R/'data/prediction_freeze.json',dict(time=time.time(),selection='training objective among physically legal points including embedded parents',files=freeze,missing=missing))
    tasks=[]
    def task(script,*args):
        ok=run(script,*args);tasks.append(dict(script=script,args=list(args),passed=ok));rt.write(R/'reports/postprocess_tasks.json',tasks);return ok
    task('audit_registered_paths.py')
    for tag in selected.values():
        task('independent_audit.py',tag);task('source_diagnostics.py','tags',tag)
    for short in ['F23','F24']:
        if short+'_U' in selected:task('regional_diagnostics.py','directions',short)
    task('regional_diagnostics.py','responses')
    task('evaluate_lowrank.py')
    import evaluation
    summaries={}
    for low,high in [('U','L3'),('U','L1'),('U','N3'),('L1','L3'),('L3','N3')]:
        pair={}
        for short in ['F23','F24']:
            if short+'_'+low in selected and short+'_'+high in selected:pair.update({short+'_R':selected[short+'_'+low],short+'_X':selected[short+'_'+high]})
        if not pair:continue
        for block in [1,2]:
            if time.time()>=rt.read(R/'work/experiment_clock.json')['delivery_deadline']:missing.append(f'{high}-{low}_{block}month');continue
            summaries[f'{high}-{low}_{block}month']=evaluation.main(pair,high+'-'+low,block)
    blocks=rt.read(R/'data/spatial_blocks.json')
    for short in ['S56','S113','S191']:
        if short+'_U' not in selected or short+'_L3' not in selected:continue
        pair={short+'_R':selected[short+'_U'],short+'_X':selected[short+'_L3']}
        for block in [1,2]:summaries[f'{short}_L3-U_{block}month']=evaluation.main(pair,short+'_L3-U',block,scopes=[(short,2024)],held_stations=blocks[short[1:]]['held_stations'])
    rt.write(R/'reports/evaluation_summary.json',clean(summaries))
    task('audit_regional_evaluation.py')
    rt.write(R/'reports/finalization_omissions.json',dict(missing=missing,failed_tasks=[x for x in tasks if not x['passed']]))
    return run('report_regional.py')
if __name__=='__main__':
    if not main():raise SystemExit(2)
