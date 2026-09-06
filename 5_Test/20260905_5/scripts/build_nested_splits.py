"""Register spatial splits without reading any fitted predictions."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'5_Test/20260905_1/scripts'))
from common import RUNTIME,atomic_json,sha256,utc_now
import pandas as pd
import numpy as np


def main():
    obs_path=ROOT/'5_Test/20260905_1/outputs/observations.parquet'
    group_path=ROOT/'5_Test/20260905_1/outputs/reach_folds.parquet'
    obs=pd.read_parquet(obs_path);groups=pd.read_parquet(group_path)
    trainbase=obs.loc[obs.primary_gate&obs.year.between(2016,2023)].copy()
    evalbase=trainbase.loc[trainbase.year.between(2020,2023)].copy()
    allreach=set(trainbase.reach_id)
    def split(label,held,extra=None):
        held=set(int(x) for x in held);excluded=held|set(extra or [])
        tr=trainbase.loc[~trainbase.reach_id.isin(excluded)]
        ev=evalbase.loc[evalbase.reach_id.isin(held)]
        if set(tr.station_key)&set(ev.station_key):raise ValueError('Station leakage')
        return dict(id=label,heldout_reaches=sorted(excluded),eval_reaches=sorted(held),
            train_ids=tr.observation_id.tolist(),eval_ids=ev.observation_id.tolist(),
            train_stations=int(tr.station_key.nunique()),eval_stations=int(ev.station_key.nunique()),
            require_eval_heldout=True,years=dict(train=[2016,2023],evaluation=[2020,2023]))
    spatial=[]
    for outer in range(5):
        held=set(groups.loc[groups.spatial_fold.eq(outer),'reach_id'])&allreach
        item=split(f'REACH_O{outer}',held)
        remaining=[k for k in range(5) if k!=outer]
        item['inner']=[]
        for inner in range(2):
            inner_groups=remaining[inner::2]
            validation=set(groups.loc[groups.spatial_fold.isin(inner_groups),'reach_id'])&allreach
            item['inner'].append(split(f'REACH_O{outer}_I{inner}',validation,held))
        if set().union(*(set(i['eval_reaches']) for i in item['inner']))!=allreach-held:raise ValueError('Inner coverage gap')
        spatial.append(item)
    eligible=[];ineligible=[]
    for tree,g in evalbase.groupby('terminal_tree_id',sort=True):
        stats=g.groupby('station_key').tn_mg_l.agg(['size','var'])
        n=int((stats['size'].ge(8)&stats['var'].gt(0)).sum())
        held=set(trainbase.loc[trainbase.terminal_tree_id.eq(tree),'reach_id'])
        outside=trainbase.loc[~trainbase.reach_id.isin(held)]
        if n<3 or outside.station_key.nunique()<20:
            ineligible.append(dict(tree=int(tree),valid_stations=n,outside_stations=int(outside.station_key.nunique())));continue
        item=split(f'TREE_O{int(tree)}',held);item['terminal_tree_id']=int(tree);item['inner']=[]
        available=groups.loc[groups.reach_id.isin(allreach-held)].sort_values(['spatial_fold','reach_id'])
        # Use pre-existing reach groups for deterministic tree-exterior inner
        # folds, then alternate reaches if one side would be empty.
        blocks=[set(available.loc[available.spatial_fold.mod(2).eq(k),'reach_id']) for k in range(2)]
        if any(not b for b in blocks):blocks=[set(available.reach_id.to_list()[k::2]) for k in range(2)]
        for k,b in enumerate(blocks):item['inner'].append(split(f'TREE_O{int(tree)}_I{k}',b,held))
        eligible.append(item)
    result=dict(status='REGISTERED_SPLITS_NOT_FITS',created_utc=utc_now(),runtime=RUNTIME,
        inputs={str(p):sha256(p) for p in [obs_path,group_path]},outer_reach=spatial,
        outer_terminal_tree=eligible,ineligible_terminal_tree=ineligible,
        candidate_selection='all candidates on nested inner OOF median station NSE',
        no_outer_TN_leakage=True,interpretation='frozen-water conditional TN transfer')
    atomic_json(result,ROOT/'5_Test/20260905_5/reports/nested_split_registry.json')
    print('NESTED_SPLITS',len(spatial),len(eligible),[(x['id'],x['train_stations'],x['eval_stations']) for x in spatial+eligible],flush=True)


if __name__=='__main__':main()
