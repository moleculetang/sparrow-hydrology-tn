"""Compare complete structural OOF predictions with the refitted old control."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'5_Test/20260905_3/scripts'))
sys.path.insert(0,str(ROOT/'5_Test/20260905_1/scripts'))
from summarize_development import station_metrics,summary,paired_bootstrap
from common import RUNTIME,atomic_json,sha256,utc_now
import json
import pandas as pd


def main():
    run=ROOT/'5_Test/20260905_4'
    structural=json.loads((run/'reports/structural_summary.json').read_text(encoding='utf-8'))
    if structural['status']!='STRUCTURAL_OOF_COMPLETE':raise RuntimeError('Structural matrix incomplete')
    control_path=ROOT/'5_Test/20260905_3/outputs/control_h7__student_t4_log1p_oof_predictions.parquet'
    control=pd.read_parquet(control_path);cm=station_metrics(control);cs=summary(cm,control);results={}
    for key in structural['outcomes']:
        path=run/'outputs'/f'{key}_oof_predictions.parquet';new=pd.read_parquet(path)
        if set(new.observation_id)!=set(control.observation_id):raise ValueError('Nonidentical OOF cohort')
        nm=station_metrics(new);ns=summary(nm,new);paired=paired_bootstrap(nm,cm)
        gates=dict(median_positive=ns['median_nse']>0,median_delta_ge_010=paired['delta_median_nse']>=.1,
            station_ci_positive=paired['station_ci95'][0]>0,tree_ci_positive=paired['tree_cluster_ci95'][0]>0,
            q25_noninferior=ns['q25_nse']>=cs['q25_nse']-.05,
            station_log_rmse_noninferior=ns['mean_station_log_rmse']<=cs['mean_station_log_rmse']*1.05)
        results[key]=dict(summary=ns,paired_vs_old_control=paired,development_gates=gates,
            all_development_gates=all(gates.values()),prediction_sha256=sha256(path))
    atomic_json(dict(status='STRUCTURE_VS_OLD_CONTROL_COMPLETE_REQUIRES_SPATIAL_CONFIRMATION',runtime=RUNTIME,created_utc=utc_now(),
        control=cs,control_sha256=sha256(control_path),comparisons=results,promotion=False),run/'reports/structure_vs_old_control.json')
    print('STRUCTURE_CONTROL_COMPARISON',json.dumps({k:dict(median=v['summary']['median_nse'],paired=v['paired_vs_old_control'],gates=v['development_gates']) for k,v in results.items()}),flush=True)


if __name__=='__main__':main()
