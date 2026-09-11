"""Finite cumulative budgets, guarded checkpoint identity, and label read barrier."""
import contextlib
import os
import sys
import time
import numpy as np
import torch
from fc_io import RUN,read,write,sha,local,memory
from fc_checkpoint import save,ResourceYield,BudgetStop
from fc_resources import process_snapshot

@contextlib.contextmanager
def exclusive(name):
    import msvcrt
    stream=local(RUN/f'work/locks/{name}.lock').open('a+b')
    stream.seek(0);stream.write(b'0');stream.flush();stream.seek(0)
    try:msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
    except OSError:
        stream.close();raise RuntimeError('DUPLICATE_LIVE_PROCESS '+name)
    try:yield
    finally:
        stream.seek(0);msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1);stream.close()

def label_barrier():
    import functools
    import pandas as pd
    def check_path(value):
        if not isinstance(value,(str,bytes,os.PathLike)):
            value=getattr(value,'name',None)
        if isinstance(value,(str,bytes,os.PathLike)):
            p=os.fsdecode(value).replace('\\','/').lower()
            if ('heldout_labels' in p or '/outputs/observations.parquet' in p or
                '/outputs/bundles/' in p or '/primary_integration' in p or '/diagnostic_observations' in p):
                raise PermissionError('TN_LABEL_OR_INVERSION_ACCESS_FORBIDDEN '+p)
    def hook(event,args):
        if event=='open':check_path(args[0])
    sys.addaudithook(hook)
    # PyArrow's native file opens need not emit Python's open audit event.
    # Intercept the actual dataframe API used by fitting/auditing as well.
    if not getattr(pd.read_parquet,'_sc_label_barrier',False):
        original=pd.read_parquet
        @functools.wraps(original)
        def guarded_read_parquet(path,*args,**kwargs):
            check_path(path)
            return original(path,*args,**kwargs)
        guarded_read_parquet._sc_label_barrier=True
        pd.read_parquet=guarded_read_parquet

def identity(fold,variant,start):
    proof=read(RUN/'reports/launch_validation.json');config=RUN/f'configs/{fold}.json'
    if sha(RUN/'configs/campaign.json')!=proof['registration_sha256']:
        raise RuntimeError('CAMPAIGN_REGISTRATION_HASH_CHANGED')
    if sha(config)!=proof['fold_config_sha256'][fold]:raise RuntimeError('CONFIG_HASH_CHANGED')
    for p,h in proof['code_sha256'].items():
        if sha(p)!=h:raise RuntimeError('CODE_HASH_CHANGED '+p)
    c=read(config)
    for path,h in [(RUN/f'outputs/data/{fold}_train.parquet',c['train_sha256']),
                   (RUN/f'outputs/data/{fold}_heldout_metadata.parquet',c['metadata_sha256']),
                   (c['cache']['path'],c['cache']['sha256'])]:
        if sha(path)!=h:raise RuntimeError('TRAINING_INPUT_HASH_CHANGED '+str(path))
    return {'fold':fold,'variant':variant,'start':start,'config_sha256':sha(config),
            'registration_sha256':proof['registration_sha256'],'code_sha256':proof['code_sha256']}

def checkpoint(path,state):
    # Commit the payload first, then atomically publish one small manifest.
    # Two alternating slots retain the last committed checkpoint if interrupted
    # between writes; there is no inconsistent payload/sidecar window.
    from pathlib import Path
    path=local(path);manifest=Path(str(path)+'.sha256.json')
    old=read(manifest) if manifest.exists() else {}
    slot=1-int(old.get('slot',1))
    payload=path.with_name(path.stem+f'.slot{slot}'+path.suffix)
    digest=save(payload,state)
    write(manifest,{'format':'atomic_two_slot_v1','slot':slot,
                    'payload':str(payload),'sha256':digest})
    return digest

def restore(path,ident):
    from pathlib import Path
    path=Path(path);record=read(str(path)+'.sha256.json')
    if record.get('format')=='atomic_two_slot_v1':
        expected=path.with_name(path.stem+f'.slot{int(record["slot"])}'+path.suffix).resolve()
        payload=Path(record['payload']).resolve()
        if payload!=expected or int(record['slot']) not in (0,1):raise RuntimeError('BAD_CHECKPOINT_SLOT')
    else:payload=path  # Read-only compatibility with the original format.
    if sha(payload)!=record['sha256']:raise RuntimeError('BAD_CHECKPOINT_HASH')
    value=torch.load(payload,weights_only=False,map_location='cpu')
    if value['identity']!=ident:raise RuntimeError('BAD_CHECKPOINT_IDENTITY')
    return value

def continuation(trace,diagnostic_used=False):
    if any(not np.isfinite(r['objective']) or
           (r.get('projected_gradient') is not None and
            (not np.isfinite(r['projected_gradient']) or r['projected_gradient']<0)) for r in trace):
        raise ValueError('INVALID_CONTINUATION_TRAJECTORY')
    n=min(50,len(trace)//2)
    if n<5:return {'continue':not diagnostic_used,'diagnostic':not diagnostic_used,'reason':'INSUFFICIENT_TRAINING_TRAJECTORY'}
    a,b=trace[-2*n:-n],trace[-n:]
    ja=min(r['objective'] for r in a);jb=min(r['objective'] for r in b)
    improvement=(ja-jb)/max(abs(ja),1e-30)
    ga=[r.get('projected_gradient') for r in a if r.get('projected_gradient') is not None]
    gb=[r.get('projected_gradient') for r in b if r.get('projected_gradient') is not None]
    drop=(min(ga)-min(gb))/max(min(ga),1e-30) if ga and gb else None
    return {'continue':improvement>=1e-6 or (drop is not None and drop>=.2),'diagnostic':False,
            'window':n,'relative_objective_decrease':improvement,'minimum_pg_decrease':drop,'reason':'TRAINING_ONLY_TREND'}

def resource_read():
    from fc_resources import cpu_usage
    return {'ram':memory(),'cpu':cpu_usage()}

def admission(now,peak,running):
    cpu=now['cpu']['used_percent'];ram=now['ram']
    if not np.isfinite(peak) or peak<=0:raise ValueError('UNKNOWN_WORKER_PEAK_RESERVATION')
    if any(not np.isfinite(v[k]) or v[k]<0 for v in running for k in ('peak','rss')):
        raise ValueError('INVALID_LIVE_WORKER_RESERVATION')
    reserve=1.2*peak+sum(max(0.,1.2*v['peak']-v['rss']) for v in running)
    return cpu is not None and cpu<90 and ram['used_percent']<90 and ram['available_gib']-reserve>=ram['total_gib']*.1

def legal_point(x,bounds,rounding=False):
    """Validate original coordinates; only documented ULP restoration allowed."""
    x=np.asarray(x,dtype=np.float64).copy()
    if x.ndim!=1 or len(x)!=len(bounds) or not np.isfinite(x).all():
        raise ValueError('NONFINITE_OR_WRONG_SHAPE_PARAMETER_POINT')
    for i,(lo,hi) in enumerate(bounds):
        for bound,violates in ((lo,lo is not None and x[i]<lo),(hi,hi is not None and x[i]>hi)):
            if violates:
                allowance=8*np.finfo(float).eps*max(1.,abs(float(bound)))
                if not rounding or abs(x[i]-bound)>allowance:raise ValueError('PARAMETER_BOUND_VIOLATION')
                x[i]=bound
    return x

def sufficient(row,best,pg_tolerance=1e-5,objective_tolerance=1e-8):
    return bool(row is not None and best is not None and
                row.get('projected_gradient') is not None and
                np.isfinite(row['projected_gradient']) and row['projected_gradient']<=pg_tolerance and
                np.isfinite(row['objective']) and
                row['objective']<=best['objective']+objective_tolerance*(1+abs(best['objective'])))

def validate_policy(policy):
    if policy['max_calls']!=[4000,6000,8000] or policy['max_hours']!=[4,6,8]:
        raise RuntimeError('UNREGISTERED_PATH_BUDGET')
    if policy['projected_gradient']!=1e-5 or policy['minimum_objective_tolerance']!=1e-8:
        raise RuntimeError('UNREGISTERED_ORIGINAL_COORDINATE_GATE')
