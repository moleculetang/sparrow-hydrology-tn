"""Forensic check of the observed progress-log gap; preserve the live log."""
from pathlib import Path
import sys
import json
import re
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'5_Test/20260905_1/scripts'))
from common import RUNTIME,atomic_json,sha256,utc_now
import pandas as pd
import numpy as np


def main():
    run=ROOT/'5_Test/20260905_5'
    logfile=run/'logs/resumed_run_nested_validation.log'
    data=logfile.read_bytes()
    gaps=[dict(offset=m.start(),length=len(m.group()),
        before=data[max(0,m.start()-160):m.start()].decode('utf-8'),
        after=data[m.end():m.end()+220].decode('utf-8')) for m in re.finditer(b'\x00+',data)]
    prefix='h7_contact_lifetime_station_normalized_mse_reach_o1_i0'
    checks=[]
    for start in range(5):
        tag=prefix+f'_s{start}';reportpath=run/'reports'/f'{tag}.json'
        report=json.loads(reportpath.read_text(encoding='utf-8'));identity=report['identity']
        assert report['converged'] and report['projected_gradient_max']<=1e-5
        assert identity['start']==start
        assert all(sha256(p)==h for p,h in identity['code_sha256'].items())
        predpath=run/'outputs'/f'{tag}_predictions.parquet';p=pd.read_parquet(predpath)
        assert not p.observation_id.duplicated().any()
        assert set(p.observation_id)==set(identity['spec']['eval_ids'])
        assert np.isfinite(p.prediction_mg_l).all() and p.prediction_mg_l.ge(0).all()
        assert not p.station_seen_in_training.any()
        tracepath=run/'logs'/f'{tag}.log';trace=tracepath.read_bytes()
        assert b'\x00' not in trace and f'SPEC_FINISHED {tag} True'.encode() in trace
        checks.append(dict(start=start,objective=report['objective'],gradient=report['projected_gradient_max'],
            rows=len(p),artifacts=[dict(path=str(path),sha256=sha256(path)) for path in [reportpath,predpath,tracepath]]))
    state=json.loads((run/'reports/nested_queue.json').read_text(encoding='utf-8'))
    result=dict(status='PROGRESS_LOG_GAP_AFFECTED_FIT_ARTIFACTS_PASS',runtime=RUNTIME,
        created_utc=utc_now(),code_sha256=sha256(Path(__file__)),progress_log=str(logfile),
        inspected_log_prefix_bytes=len(data),null_runs=gaps,affected_fits=checks,
        controller_state_snapshot=state,
        cause='Unknown; log contains a null-byte gap. Do not infer a numerical failure or a specific filesystem cause.',
        scope='Five affected completed fit reports, code identity, exact evaluation IDs, finite predictions and separate completion traces; not a new forward replay.',
        action='Keep the original live progress log unchanged. Continue verified live queue; final numerical audits remain required.')
    atomic_json(result,run/'reports/queue_log_gap_audit.json')
    print('QUEUE_LOG_GAP_AUDIT',result['status'],len(gaps),len(checks),flush=True)


if __name__=='__main__':main()
