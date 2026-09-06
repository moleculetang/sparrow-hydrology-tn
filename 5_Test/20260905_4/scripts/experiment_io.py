"""Immutable specifications and serial fitting shared by later stages."""
from pathlib import Path
import sys
import json
import subprocess
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'5_Test/20260905_1/scripts'))
from common import atomic_json,atomic_parquet,sha256,utc_now


def save_spec(spec,stage):
    path=ROOT/'5_Test'/stage/'configs'/f"{spec['id']}.json"
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8'))!=spec:raise RuntimeError(f'Conflicting registered specification {path}')
    else:atomic_json(spec,path)
    return path


def fit_starts(specpath,statuspath,status):
    spec=json.loads(Path(specpath).read_text(encoding='utf-8'));run=ROOT/spec['output_dir']
    runner=ROOT/'5_Test/20260905_4/scripts/fit_spec.py'
    rows=[]
    for start in range(5):
        tag=spec['id']+f'_s{start}';log=run/'logs'/f'{tag}.log';log.parent.mkdir(parents=True,exist_ok=True)
        status.update(active=tag,status='RUNNING',updated_utc=utc_now());atomic_json(status,statuspath)
        print('PROGRAM_JOB_STARTED',tag,flush=True)
        with log.open('a',encoding='utf-8') as stream:
            done=subprocess.run([sys.executable,'-B',str(runner),'--spec',str(specpath),'--start',str(start)],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        if done.returncode:raise RuntimeError(f'Fit process failed: {tag}; see {log}')
        path=run/'reports'/f'{tag}.json';r=json.loads(path.read_text(encoding='utf-8'))
        if not r['converged'] or r['projected_gradient_max']>1e-5:raise RuntimeError(f'Unconverged physical start {tag}')
        rows.append(dict(tag=tag,path=str(path),objective=r['objective'],start=start))
        print('PROGRAM_JOB_FINISHED',tag,r['projected_gradient_max'],flush=True)
    best=min(rows,key=lambda r:(r['objective'],r['start']))
    selected=dict(spec_id=spec['id'],spec_path=str(specpath),spec_sha256=sha256(specpath),
        selected=best,all_starts=rows,selection_rule='lowest training objective among all five stationary starts')
    atomic_json(selected,run/'reports'/f"{spec['id']}_selected.json")
    p=run/'outputs'/f"{best['tag']}_predictions.parquet"
    return pd.read_parquet(p),selected


def temporal_spec(candidate,year,obs,stage):
    tr=obs.loc[obs.primary_gate&obs.year.between(2016,year-1)]
    ev=obs.loc[obs.year.eq(year)]
    return dict(candidate, id=f"{candidate['candidate_id']}_t{year}",train_ids=tr.observation_id.tolist(),
        eval_ids=ev.observation_id.tolist(),product='sensitivity' if year>=2025 else 'formal',
        output_dir=f'5_Test/{stage}',evaluation_role='development' if year<=2023 else ('confirmation' if year==2024 else 'sensitivity'))
