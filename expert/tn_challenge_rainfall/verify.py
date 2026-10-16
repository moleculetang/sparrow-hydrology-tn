"""Verify the authorized rainfall subset and return date x original-case-reach values."""
import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
def load_rainfall():
    manifest=json.loads((HERE/'manifest.json').read_text(encoding='utf-8'))
    scope=json.loads((HERE/'case_scope.json').read_text(encoding='utf-8'))
    parts=[]
    for record in manifest['files']:
        p=HERE/record['path']
        assert hashlib.sha256(p.read_bytes()).hexdigest()==record['sha256'],p
        f=pd.read_csv(p,float_precision='round_trip')
        assert list(f.columns)==scope['columns'],p
        assert len(f)==record['rows'],p
        parts.append(f)
    f=pd.concat(parts,ignore_index=True)
    dates=pd.DatetimeIndex(pd.to_datetime(f.pop('date'),format='%Y-%m-%d'))
    expected=pd.date_range(scope['history_start'],scope['history_end'])
    assert dates.equals(expected)
    a=f.to_numpy(dtype=np.float64)
    assert list(a.shape)==manifest['shape']
    assert np.isfinite(a).all() and (a>=0).all()
    assert hashlib.sha256(np.ascontiguousarray(a,dtype='<f8').tobytes()).hexdigest()==manifest['logical_float64_c_order_sha256']
    annual=pd.read_csv(HERE/'annual_totals.csv',float_precision='round_trip')
    assert not annual.duplicated(['year','global_reach_id']).any()
    for y in range(1961,2025):
        rows=annual.loc[annual.year.eq(y)].set_index('global_reach_id').loc[scope['global_reach_ids']]
        assert rows.days.eq(int((dates.year==y).sum())).all()
        # Reductions may reorder addition; data identity above remains exact.
        np.testing.assert_allclose(a[dates.year==y].sum(axis=0),rows.precipitation_sum_mm.to_numpy(),rtol=1e-14,atol=1e-10)
    return dates,scope['global_reach_ids'],a

if __name__=='__main__':
    dates,ids,a=load_rainfall()
    print(json.dumps(dict(passed=True,days=len(dates),reaches=len(ids),readings=int(a.size),units='mm/day')))
