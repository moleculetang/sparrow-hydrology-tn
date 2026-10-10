"""Freeze selection before any real held-out evaluation; preserve conditional omissions."""
import time
import native_runtime as rt
from finalize_source import run
from finalize_regional import clean
R=rt.RUN
def main():
    import pandas as pd
    jobs=rt.read(R/'configs/jobs.json');rows=[];selected={};freeze={};missing=[]
    for j in jobs:
        out=R/'outputs'/j['tag'];a=rt.read(out/'audit.json') if (out/'audit.json').exists() else {}
        row=dict(tag=j['tag'],kind=j['kind'],input_mode=j['input_mode'],reused=bool(j.get('reuse_from')),status=a.get('status','MISSING'),physical=a.get('physical_reasonable'),numerical=a.get('numerical_sufficient'),objective=a.get('objective'),pg=a.get('pg'),calls=a.get('calls'))
        if (out/'model.json').exists():row.update(rt.read(out/'model.json')['terms'])
        rows.append(row)
        for name in ['model.json','audit.json','daily_station_mass_water.parquet','daily_physical_ledger.npz','predictions.parquet']:
            if (out/name).exists():freeze[(out/name).relative_to(R).as_posix()]=rt.sha(out/name)
    for short in ['F23','F24']:
        for structure in ['U','L3']:
            for mode in ['P','D','A']:
                key=f'{short}_{structure}_{mode}';candidates=[r for r in rows if r['tag'].startswith(key+'_') and r['physical'] and r['objective'] is not None]
                if candidates:selected[key]=min(candidates,key=lambda r:r['objective'])['tag']
                else:missing.append(key)
    pd.DataFrame(rows).to_csv(R/'reports/path_summary.csv',index=False);rt.write(R/'data/selected.json',selected)
    rt.write(R/'data/prediction_freeze.json',dict(time=time.time(),selection='Only training total objective among physical legal points; different input products are not nested',files=freeze,missing=missing))
    tasks=[]
    for tag in selected.values():
        for script in ['independent_audit.py','daily_source_accounting.py']:
            passed=run(script,tag);tasks.append(dict(script=script,tag=tag,passed=passed));rt.write(R/'reports/postprocess_tasks.json',tasks)
        if rt.read(R/'outputs'/tag/'model.json')['job']['input_mode']!='P':
            passed=run('input_compensation.py',tag);tasks.append(dict(script='input_compensation.py',tag=tag,passed=passed));rt.write(R/'reports/postprocess_tasks.json',tasks)
    import evaluation
    summaries={}
    for structure in ['U','L3']:
        for low,high in [('P','D'),('D','A'),('P','A')]:
            pair={}
            for short in ['F23','F24']:
                if f'{short}_{structure}_{low}' in selected and f'{short}_{structure}_{high}' in selected:pair.update({short+'_R':selected[f'{short}_{structure}_{low}'],short+'_X':selected[f'{short}_{structure}_{high}']})
            if not pair:continue
            for block in [1,2]:
                key=f'{structure}_{high}-{low}'
                if time.time()>=rt.read(R/'work/experiment_clock.json')['delivery_deadline']:missing.append(f'{key}_{block}month');continue
                summaries[f'{key}_{block}month']=evaluation.main(pair,key,block)
                rt.write(R/'reports/evaluation_summary.json',clean(summaries))
    rt.write(R/'reports/finalization_omissions.json',dict(missing=missing,failed_tasks=[t for t in tasks if not t['passed']]))
    rt.write(R/'reports/computation_phase_complete.json',dict(status='COMPUTED_PENDING_SCIENTIFIC_REPORT',finished=time.time(),selected=selected))
    return not missing and all(t['passed'] for t in tasks)
if __name__=='__main__':
    if not main():raise SystemExit(2)
