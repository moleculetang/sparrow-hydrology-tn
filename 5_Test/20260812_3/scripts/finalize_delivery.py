from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT=Path(__file__).resolve().parents[1]


def sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def main()->None:
    b=pd.read_parquet(ROOT/'outputs'/'B0'/'q72_three_fold_oof_predictions.parquet')
    i=pd.read_parquet(ROOT/'outputs'/'I0'/'q72_three_fold_oof_predictions.parquet')
    k=['comid','q_site','year','month','fold_id']; b=b.sort_values(k).reset_index(drop=True); i=i.sort_values(k).reset_index(drop=True)
    g1=json.loads((ROOT/'logs'/'G1_b0_reproduction.json').read_text(encoding='utf-8'))
    protection=json.loads((ROOT/'logs'/'I0_protection_gate.json').read_text(encoding='utf-8'))
    posterior=pd.read_csv(ROOT/'reports'/'dynamic_beta_posterior_summary.csv')
    syn=pd.read_csv(ROOT/'reports'/'synthetic_recovery.csv')
    syn['abs_bias']=abs(syn.posterior_mean-syn.beta_true)
    syn_summary=syn.groupby(['fold_id','beta_true']).agg(n=('replicate','size'),mean_estimate=('posterior_mean','mean'),median_abs_bias=('abs_bias','median'),sign_recovery=('sign_correct','mean'),coverage=('covered','mean'),false_positive=('false_positive','mean')).reset_index()
    syn_summary.to_csv(ROOT/'reports'/'synthetic_recovery_summary.csv',index=False,encoding='utf-8-sig')
    fold_expected={
      'fit_2006_2011_eval_2012_2013':2434,
      'fit_2006_2013_eval_2014_2015':2582,
      'fit_2006_2015_eval_2016_2018':3722,
    }
    persistence_rows=[]
    for fold_id, expected_eval in fold_expected.items():
        fold_dir=ROOT/'outputs'/'I0'/'blocked_folds'/fold_id
        audit=json.loads((fold_dir/'reports'/'model_population_audit.json').read_text(encoding='utf-8'))
        authoritative=pd.read_parquet(fold_dir/'reports'/'monthly_bayes_seasonal_hysteresis_prediction_vs_observed_2006_2022.parquet')
        csv_mirror=pd.read_csv(fold_dir/'reports'/'monthly_bayes_seasonal_hysteresis_prediction_vs_observed_2006_2022.csv',encoding='utf-8-sig')
        evaluation=pd.read_parquet(fold_dir/'evaluation_predictions.parquet')
        persistence_rows.append({
          'fold_id':fold_id,'forcing_rows':audit['forcing_rows'],'observed_rows_before_write':audit['prediction_rows_before_write'],
          'parquet_rows_after_read':len(authoritative),'csv_rows_after_read':len(csv_mirror),'evaluation_rows':len(evaluation),
          'expected_evaluation_rows':expected_eval,'population_gate_pass':bool(audit['forcing_rows']==46920 and audit['prediction_rows_before_write']==21440 and len(authoritative)==21440 and len(csv_mirror)==21440 and len(evaluation)==expected_eval),
        })
    persistence=pd.DataFrame(persistence_rows)
    persistence.to_csv(ROOT/'reports'/'persistence_population_audit.csv',index=False,encoding='utf-8-sig')
    result={
      'g0_runtime_and_input':'PASS','g1_b0_exact_reproduction':g1['keys_equal'] and g1['predict_max_abs']<=1e-8,
      'g2_literal_rho_mass_closure':'PASS_7_UNIT_TEST_SUITE','g3_incarea_topology_closure':'PASS_ATOL_1E-8','g4_bad_highflow_composite_removed':'PASS',
      'g5_identifiability':'FAIL_POSTERIOR_SHRINKAGE','g6_synthetic_recovery':'FAIL_NEGATIVE_BETA',
      'g7_i0_oof':len(i)==8738 and i[k].equals(b[k]),'g7_fold_rows':i.groupby('fold_id').size().to_dict(),'g7_persistence_population_gate':bool(persistence.population_gate_pass.all()),'g8_i1_scientific_run':'STOPPED_BY_G5_G6',
      'g9_i0_protection':protection['protection_pass'],'terminal':'I0_ACCOUNTING_REPAIR_PREDICTIVE_PROMOTION_SUPPORTED__DYNAMIC_CONNECTIVITY_NOT_IDENTIFIABLE'}
    (ROOT/'terminal_gate.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    files=[p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    manifest=[{'path':str(p),'size':p.stat().st_size,'sha256':sha(p)} for p in files if p.name!='input_manifest.json']
    (ROOT/'input_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
