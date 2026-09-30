"""Repair evaluation boolean coercion only; all fit and prediction hashes frozen."""
import shutil,time,subprocess,sys
import numpy as np
import native_runtime as rt
import evaluation
R=rt.RUN
archive=R/'evidence/evaluation_boolean_v1'
archive.mkdir(exist_ok=False)
shutil.copytree(R/'reports',archive/'reports')
shutil.copy2(R/'delivery_manifest.json',archive/'delivery_manifest.json')
selected=rt.read(R/'data/selected.json');frozen=rt.read(R/'data/prediction_freeze.json')
assert all(rt.sha(R/p)==h for p,h in frozen['files'].items())
summaries={}
for low,high in [('R','G'),('R','U'),('U','G'),('G','S'),('R','S')]:
    pair={}
    for fold in ['F23','F24']:
        pair[fold+'_R']=selected[fold+'_'+low];pair[fold+'_X']=selected[fold+'_'+high]
    for block in [1,2]:summaries[f'{high}-{low}_{block}month']=evaluation.main(pair,high+'-'+low,block)
def clean(v):
    if isinstance(v,dict):return {k:clean(a) for k,a in v.items()}
    if isinstance(v,(list,tuple)):return [clean(a) for a in v]
    if isinstance(v,(float,np.floating)) and not np.isfinite(v):return None
    return v
rt.write(R/'reports/evaluation_summary.json',clean(summaries))
assert all(rt.sha(R/p)==h for p,h in frozen['files'].items())
rt.write(R/'reports/evaluation_boolean_repair.json',dict(status='PASS',cause='mixed eligible/ineligible rows retained object dtype; bitwise invert yielded integers rather than logical negation',repair='explicit bool conversion on qualified events before inversion',fits_changed=False,predictions_changed=False,scientific_calls_added=0,tests='test_evaluation.py and test_evaluation_pipeline.py',archive='evidence/evaluation_boolean_v1',time=time.time()))
subprocess.run([sys.executable,'-B',str(R/'scripts/report_source.py')],check=True,cwd=R)
print('EVALUATION_REPAIR_PASS',flush=True)
