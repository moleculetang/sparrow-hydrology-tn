from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from runtime_guard import assert_sparrow_runtime

RUNTIME=assert_sparrow_runtime(); ROOT=Path(__file__).resolve().parents[1]
KEY=['comid','q_site','year','month','fold_id']
def sha(p:Path)->str: return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p0=pd.read_parquet(ROOT/'outputs'/'P0'/'q72_three_fold_oof_predictions.parquet').sort_values(KEY).reset_index(drop=True)
 p1=pd.read_parquet(ROOT/'outputs'/'P1'/'q72_three_fold_oof_predictions.parquet')
 ref=pd.read_parquet(ROOT/'inputs'/'reference'/'B1_oof.parquet').sort_values(KEY).reset_index(drop=True)
 evaluation=json.loads((ROOT/'terminal_gate.json').read_text(encoding='utf-8'))
 folds=[]
 for fold,g in p1.groupby('fold_id'):
  y=g.actual.to_numpy(float); q=g.predict.to_numpy(float); ly=np.log(y); lq=np.log(q)
  folds.append({'fold_id':fold,'rows':len(g),'raw_nse':float(1-np.sum((q-y)**2)/np.sum((y-y.mean())**2)),'log_nse':float(1-np.sum((lq-ly)**2)/np.sum((ly-ly.mean())**2)),'pbias_pct':float(100*np.sum(q-y)/np.sum(y))})
 gate={'runtime':RUNTIME,'P0_reproduction':{'keys_equal':bool(p0[KEY].equals(ref[KEY])),'actual_max_abs':float(np.max(np.abs(p0.actual-ref.actual))),'predict_max_abs':float(np.max(np.abs(p0.predict-ref.predict)))},'P1_oof_sha256':sha(ROOT/'outputs'/'P1'/'q72_three_fold_oof_predictions.parquet'),'fold_metrics':folds,'strict_dual_nse_gt_095':bool(all(x['raw_nse']>.95 and x['log_nse']>.95 for x in folds)),'protection_gate':bool(evaluation['all_promotion_gates_pass']),'terminal':['FOLD_PURE_HYPERPARAMETER_LINEAGE_CLOSED','FOLD_PURE_BASELINE_REBUILT_AND_PROTECTED','STRICT_DUAL_NSE_TARGET_NOT_REACHED']}
 (ROOT/'terminal_gate.json').write_text(json.dumps(gate,ensure_ascii=False,indent=2),encoding='utf-8')
 files=[ROOT/'inputs'/'A1_indata.parquet',ROOT/'inputs'/'topology'/'topology_edges.csv',ROOT/'inputs'/'reference'/'B1_oof.parquet',ROOT/'scripts'/'run_fold_pure_hyperparameters.py',ROOT/'scripts'/'components'/'q72_fold_pure_component.py',ROOT/'outputs'/'P1'/'q72_three_fold_oof_predictions.parquet']
 (ROOT/'input_manifest.json').write_text(json.dumps({'runtime':RUNTIME,'files':[{'path':str(p),'size':p.stat().st_size,'sha256':sha(p)} for p in files]},ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps(gate,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
