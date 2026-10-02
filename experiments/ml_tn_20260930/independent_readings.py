"""Read-level arithmetic audit; no residual-based cleaning or new quality rule."""
import numpy as np,pandas as pd
from mltn.common import ROOT,write

def daily_mean(readings):
    keys=['station_key','monitoring_time'];q=readings.copy()
    for _,g in q[q.duplicated(keys,keep=False)].groupby(keys):
        if g.adopted_value.nunique(dropna=False)>1 or g.eligible.nunique(dropna=False)>1:raise ValueError('CONFLICTING_DUPLICATE_READ')
    q=q.drop_duplicates(keys);q=q[q.eligible & np.isfinite(q.adopted_value) & q.adopted_value.ge(0)]
    q=q.groupby(['station_key','date']).adopted_value.agg(tn_mg_l='mean',read_count='count').reset_index()
    return q

def main():
    r=pd.read_parquet(ROOT/'data/hf_readings.parquet');daily=daily_mean(r)
    duplicate=daily_mean(pd.concat([r,r.iloc[:10]],ignore_index=True));pd.testing.assert_frame_equal(daily,duplicate)
    adopted=pd.read_parquet(ROOT/'data/hf_daily_accepted.parquet');m=adopted.merge(daily,on=['station_key','date'],suffixes=('_adopted','_recomputed'),validate='one_to_one')
    assert len(m)==len(adopted)
    np.testing.assert_allclose(m.tn_mg_l_adopted,m.tn_mg_l_recomputed,rtol=0,atol=1e-12)
    np.testing.assert_array_equal(m.read_count_adopted,m.read_count_recomputed)
    write(ROOT/'evidence/independent_HF_reading_mean.json',dict(passed=True,accepted_days=len(m),unique_readings=len(r),max_absolute_difference=float(abs(m.tn_mg_l_adopted-m.tn_mg_l_recomputed).max()),read_count_exact=True,duplicate_replication_invariant=True,coverage='accepted day support inherited unchanged; daily mean and counts recomputed from eligible TN reads'))
if __name__=='__main__':main()
