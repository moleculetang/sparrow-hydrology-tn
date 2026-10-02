"""Regenerate affected readouts and ALL official metric evidence after seed repair."""
import os, subprocess, sys, time, shutil
import numpy as np
import pandas as pd
from mltn.common import ROOT,read,write,sha
from mltn.resources import lease,registry
from monthly_readouts import hf_months
from full_domain import full_inputs,rows
def main():
    mod,resource_root=registry();original_cpus=mod.physical_cores() if mod is not None else None
    parent_grant=lease('seed_repair_postprocessing',1,'cpu')
    assert read(ROOT/'evidence/joint_seed_repair_import.json')['passed']
    from freeze_results import main as freeze
    freeze()
    manifest=read(ROOT/'outputs/monthly_readouts/manifest.json')
    kept=[i for i in manifest['records'] if not (i['parent'].startswith('joint_') and 'XGBoost' in i['parent'])]
    import re
    for jid in read(ROOT/'outputs/result_roles.json')['include']:
        if not jid.startswith('joint_') or 'XGBoost' not in jid:continue
        path=ROOT/'jobs'/jid/'prediction_daily_hf.parquet'
        q=hf_months(pd.read_parquet(path));name='hfmean_'+jid;dest=ROOT/'outputs/monthly_readouts'/(name+'.parquet');q.to_parquet(dest,index=False)
        stage=re.search(r'F23|F24|S23|S24',jid)[0];b=re.search(r'_B(56|113|191)',jid)
        kept.append(dict(file=dest.relative_to(ROOT).as_posix(),sha256=sha(dest),configuration=name,context=stage+('' if b is None else b[0]),task='hf_monthly',parent=jid,identity=dict(operator='read-count weighted mean of shared HF daily support; at least two dates per month',parent_prediction_file=path.relative_to(ROOT).as_posix(),parent_prediction_sha256=sha(path))))
    manifest['records']=kept;write(ROOT/'outputs/monthly_readouts/manifest.json',manifest)
    d=full_inputs();c=read(ROOT/'outputs/frozen_joint_selection.json')['selected']['XGBoost']
    import xgboost as xgb,pickle
    for fold,year in [('F23',2023),('F24',2024)]:
        q=rows(d,year,'daily');pp=[]
        for seed in [1729,1730,1731]:
            f=ROOT/'jobs'/f'joint_{fold}_XGBoost_c{c}_s{seed}';ck=pickle.load((f/'checkpoint.pkl').open('rb'));model=xgb.Booster();model.load_model(f/'booster.json');model.set_param({'nthread':1})
            pp.append(np.exp(model.predict(xgb.DMatrix(ck['transform'].apply(d.raw_rows(q))),output_margin=True)))
        q['prediction']=np.mean(pp,axis=0);q.to_parquet(ROOT/'outputs/full_domain'/f'{fold}_joint_XGBoost_230_reach_outlets.parquet',index=False)
    old=ROOT/'superseded/seed_not_forwarded/outputs/evaluation'
    assert not old.exists();old.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(ROOT/'outputs/evaluation'),str(old))
    (ROOT/'outputs/evaluation').mkdir()
    unchanged=['exact_input_collision_lower_bound.parquet','feature_collision_identity.json','feature_reliance_receipt.json','group_permutation_reliance.csv','native_feature_importance.csv']
    for name in unchanged:
        shutil.copyfile(old/name,ROOT/'outputs/evaluation'/name)
        assert sha(old/name)==sha(ROOT/'outputs/evaluation'/name)
    write(ROOT/'evidence/seed_repair_unaffected_diagnostics_reuse.json',dict(passed=True,
        files={name:sha(old/name) for name in unchanged},scope='input-only checks and unchanged direct CatBoost diagnostics; no joint-tree cache reuse'))
    stages=['evaluate.py','training_metrics.py','diagnostics.py','independent_review.py','training_seedmeans.py','expert_tables.py','audit_effect_metrics.py','audit_final_protocol.py','diagnose_amplitude.py','author_final_report.py']
    env=os.environ.copy()
    for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:env[k]='1'
    records=[]
    for s in stages:
        # These scripts acquire their OWN lease. Do not leave the waiting
        # controller's sole inherited CPU reserved against its child.
        if s in ['diagnose_amplitude.py','author_final_report.py'] and parent_grant is not None:
            mod.release(parent_grant['token'],root=resource_root)
            mod.affinity(original_cpus);parent_grant=None
        t=time.monotonic()
        with (ROOT/'logs'/('seed_repair_'+s+'.log')).open('ab') as log:code=subprocess.call([sys.executable,'-B',str(ROOT/s)],env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        records.append(dict(script=s,exit_code=code,elapsed_s=time.monotonic()-t,sha256=sha(ROOT/s)));write(ROOT/'outputs/seed_repair_postprocess_stages.json',records)
        print(s,code,round(time.monotonic()-t,2),flush=True)
        assert code==0, s
    write(ROOT/'outputs/seed_repair_postprocess_done.json',dict(passed=True,stages=records,scientific_changes='only registered joint-tree seed correction; all metrics rebuilt on unchanged frozen support'))
if __name__=='__main__':main()
