"""Independent actual seed-mean and synchronous month-block result checks."""
import math
import numpy as np
import pandas as pd
from audit_effect_metrics import scalar
from mltn.common import ROOT, read, write, sha


def same(a, b):
    assert math.isnan(a) == math.isnan(b), ('UNDEFINED_MISMATCH', a, b)
    if math.isfinite(a):
        assert abs(a-b) <= 1e-8*(1+abs(a)), ('ACTUAL_BOOTSTRAP_MISMATCH', a, b)


def main():
    out = ROOT/'outputs/evaluation'
    seeds = []
    for file in sorted(out.glob('*_seedmean*_frozen.parquet')):
        keys = ['station_key', 'date']
        average = pd.read_parquet(file).sort_values(keys).reset_index(drop=True)
        parents = []
        for seed in [1729, 1730, 1731]:
            path = file.with_name(file.name.replace('_seedmean', '_s'+str(seed)))
            assert path.exists(), ('MISSING_SEED_PARENT', path)
            q = pd.read_parquet(path).sort_values(keys).reset_index(drop=True)
            assert not q.duplicated(keys).any() and q[keys].equals(average[keys])
            np.testing.assert_array_equal(q.observed, average.observed)
            if 'read_count' in q:
                np.testing.assert_array_equal(q.read_count, average.read_count)
            parents.append(q.prediction.to_numpy(float))
        expected = np.mean(np.stack(parents), axis=0)
        np.testing.assert_allclose(average.prediction, expected, rtol=1e-12, atol=1e-12)
        seeds.append(dict(file=file.name, rows=len(average), sha256=sha(file)))
    assert seeds, 'NO_ACTUAL_SEED_MEANS_CHECKED'
    pairs = read(out/'paired_summary.json')
    # One identity-selected seed-mean comparison per context/task, covering time
    # and each frozen spatial block, without using any performance criterion.
    selected = {}
    for pair in sorted(pairs, key=lambda p:(p['context'], p['task'], p['candidate'], p['baseline'])):
        if '_seedmean' in pair['candidate']:
            selected.setdefault((pair['context'], pair['task']), pair)
    records = []
    for pair in selected.values():
        task = pair['task']; name = pair['candidate']; base = pair['baseline']
        q = pd.read_parquet(out/f'{name}_{task}_frozen.parquet')
        b = pd.read_parquet(out/f'{base}_{task}_frozen.parquet')
        f = q.merge(b[['station_key','date','prediction']].rename(columns={'prediction':'baseline'}),
                    on=['station_key','date'], validate='one_to_one')
        tag = f'{name}__versus__{base}_{task}'
        eligible = pd.read_csv(out/f'{tag}_paired.csv')
        f = f[f.station_key.isin(eligible[eligible.nse_eligible].station_key)].copy()
        if f.empty:
            continue
        f['calendar_month'] = pd.to_datetime(f.date).dt.to_period('M')
        months = pd.period_range(f.calendar_month.min(), f.calendar_month.max(), freq='M')
        for block in [1,2]:
            stored = pd.read_parquet(out/f'{tag}_bootstrap_{block}month.parquet')
            assert len(stored)==1000
            rng = np.random.default_rng(1729)
            for rep in range(3):
                draws = []
                while len(draws)<len(months):
                    start = int(rng.integers(0,max(1,len(months)-block+1)))
                    draws.extend(months[start:start+block])
                frames=[]
                for index,month in enumerate(draws[:len(months)]):
                    part=f[f.calendar_month.eq(month)].copy();part['copy_month']=index
                    frames.append(part)
                sample=pd.concat(frames,ignore_index=True)
                na=[];nb=[];ca=[];cb=[]
                for _,g in sample.groupby('station_key'):
                    a=scalar(g.observed,g.baseline)['NSE'];bval=scalar(g.observed,g.prediction)['NSE']
                    if math.isfinite(a) and math.isfinite(bval):na.append(a);nb.append(bval)
                    ys=[];aa=[];bb=[];ww=[]
                    if 'read_count' in g:
                        for _,m in g.groupby('copy_month'):
                            if len(m)<2:continue
                            w=m.read_count.to_numpy(float);w=w/math.fsum(w)
                            y=m.observed.to_numpy(float);a=m.baseline.to_numpy(float);bval=m.prediction.to_numpy(float)
                            for values,output in [(y,ys),(a,aa),(bval,bb)]:
                                mean=math.fsum(float(v)*float(z) for v,z in zip(values,w))
                                output.extend(values-mean if np.ptp(values)>0 else np.zeros_like(values))
                            ww.extend(w)
                        av=scalar(ys,aa,ww)['NSE'];bv=scalar(ys,bb,ww)['NSE']
                        if math.isfinite(av) and math.isfinite(bv):ca.append(av);cb.append(bv)
                row=stored.iloc[rep];assert int(row.common_stations)==len(na)
                values=dict(median_difference=float(np.median(nb)-np.median(na)) if na else math.nan,
                            median_paired_difference=float(np.median(np.array(nb)-na)) if na else math.nan,
                            improved_fraction=float(np.mean(np.array(nb)>na)) if na else math.nan,
                            centered_median_paired_difference=float(np.median(np.array(cb)-ca)) if ca else math.nan)
                for key,value in values.items():same(value,float(row[key]))
            records.append(dict(context=pair['context'],task=task,baseline=base,candidate=name,
                                block_months=block,independently_recomputed_replicates=3))
    assert records, 'NO_ACTUAL_BLOCK_REPLICATES_CHECKED'
    write(ROOT/'evidence/independent_final_protocol.json',dict(
        passed=True,seedmeans=seeds,actual_block_checks=records,
        method='exact parent support and prediction mean; explicit calendar-block copies, math.fsum scalar metrics',
        coverage='original jointly eligible stations frozen before resampling; sampled zero variance remains undefined',
        scope='all frozen seed means; first 3 actual draws for both block lengths in one identity-selected pair per context/task; not all 1000 scalar recomputations'))


if __name__=='__main__':main()
