from pathlib import Path
import pandas as pd

root=Path(__file__).resolve().parents[1]
b=pd.read_parquet(root/'outputs'/'B0'/'q72_three_fold_oof_predictions.parquet')
for fold in ['fit_2006_2011_eval_2012_2013','fit_2006_2013_eval_2014_2015','fit_2006_2015_eval_2016_2018']:
    p=root/'outputs'/'I0'/'blocked_folds'/fold/'reports'/'monthly_bayes_seasonal_hysteresis_prediction_vs_observed_2006_2022.csv'
    d=pd.read_csv(p,encoding='utf-8-sig')
    e=pd.read_csv(root/'outputs'/'I0'/'blocked_folds'/fold/'evaluation_predictions.csv',encoding='utf-8-sig')
    bb=b[b.fold_id.eq(fold)]
    keys=['comid','q_site','year','month']
    miss=bb[keys].merge(e[keys],on=keys,how='left',indicator=True)
    print(fold,'source',len(d),'eval',len(e),'years',d.year.value_counts().sort_index().to_dict(),'stations',d.q_site.nunique(),'missing',miss._merge.eq('left_only').sum(),'missing years',miss[miss._merge.eq('left_only')].year.value_counts().to_dict())
