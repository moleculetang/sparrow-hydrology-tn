"""Counterfactual future-label mutation repeats the actual cleaning/union rules."""
from campaign_model import *
from prepare_hf import classify,products
import native_runtime as rt

def build(normal,pub,years,local):
    tr=normal[normal.year.isin(years)].copy();tr['excluded']=False
    for _,g in tr.groupby(['station_key','season']):tr.loc[g.index,'excluded']=classify(g)[0]
    days,hf=products(tr[~tr.excluded],years)
    pp=pub[pub.year.isin(years)].copy();pp['excluded']=False
    for _,g in pp.groupby('station_key'):pp.loc[g.index,'excluded']=classify(g,False)[0]
    keys=['station_key','year','month'];hf=hf[keys+['y']].copy();hf['source_kind']='HF'
    mo=pp[~pp.excluded][keys+['tn_mg_l']].rename(columns={'tn_mg_l':'y'})
    mo=mo.merge(hf[keys].assign(has_hf=True),on=keys,how='left');mo=mo[mo.has_hf.isna()].drop(columns='has_hf');mo['source_kind']='PUB'
    union=pd.concat([hf,mo],ignore_index=True).sort_values(keys).reset_index(drop=True)
    stat=union.groupby('station_key').y.agg(n='size',variance=lambda x:float(np.var(x)));stat=stat[stat.n>=2]
    union=union[union.station_key.isin(stat.index)];ref=stat[stat.index.isin(local)&stat.variance.gt(0)]
    floor=float(ref.variance.quantile(.1));stat['denominator']=stat.variance.clip(lower=floor)
    return union,stat,days

def main():
    p=RUN.parent/'20260917_5';raw=pd.read_parquet(p/'data/heldout_labels/hf_canonical_selected.parquet');pub=pd.read_parquet(p/'data/heldout_labels/monthly_original.parquet')
    normal=raw[raw.indicator.eq('TN')&raw.station_status.eq('正常')&raw.adopted_value.notna()&np.isfinite(raw.adopted_value)&raw.adopted_value.ge(0)&~raw.unresolved_conflict.fillna(False)].copy()
    old=pd.read_parquet(RUN/'data/old_calendar.parquet');local=set(old[old.cohort.isin(['N','H'])].station_key)
    rows=[]
    for fold,years,cohort in [('F23_G_D',[2021,2022],RUN/'data/cohorts/F23_G'),('T24_G_D_H1',[2021,2022,2023],p/'data/cohorts/T24_G')]:
        baseline=build(normal,pub,years,local);nr=normal.copy();pr=pub.copy()
        nr.loc[~nr.year.isin(years),'adopted_value']=1234567.;pr.loc[~pr.year.isin(years),'tn_mg_l']=7654321.
        mutant=build(nr,pr,years,local)
        for a,b in zip(baseline,mutant):pd.testing.assert_frame_equal(a,b)
        actual=pd.read_parquet(cohort/'station_months.parquet')[baseline[0].columns].reset_index(drop=True)
        pd.testing.assert_frame_equal(baseline[0].reset_index(drop=True),actual)
        scales=pd.read_parquet(cohort/'scales.parquet').set_index('station_key').sort_index()
        pd.testing.assert_frame_equal(baseline[1].sort_index(),scales,check_dtype=False)
        rows.append(dict(fold=fold,months=len(actual),counterfactual_identical=True,actual_preparation_identical=True))
    rt.write(RUN/'reports/label_counterfactual.json',dict(status='PASS',folds=rows,scope='Actual preprocessing identical; deterministic initialization and training consume only these frozen tables, not withheld labels.'))
    print('PASS counterfactual')
if __name__=='__main__':main()
