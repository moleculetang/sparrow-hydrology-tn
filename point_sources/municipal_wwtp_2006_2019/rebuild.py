"""Same annual mass equations on reach/province aggregates, no facility records."""
import calendar
import numpy as np
import pandas as pd

SCENARIOS={'tdn_equals_tn_lower_bound':1.,'tdn_fraction_of_tn_0.9':.9,'tdn_fraction_of_tn_0.8':.8}
def rebuild_reach(anchor,factors):
    rows=[]
    for year in range(2006,2020):
        a=anchor.copy();a['year']=year
        a['wang_province']=a.province.replace({'Guangxi Zhuang Autonomous Region':'Guangxi'})
        a=a.merge(factors[['province','year','province_tn_load_factor_relative_2012']].rename(columns={'province':'wang_province'}),
                  on=['wang_province','year'],how='left',validate='many_to_one')
        if a.province_tn_load_factor_relative_2012.isna().any():raise ValueError('MISSING_PROVINCE_YEAR_FACTOR')
        for name,fraction in SCENARIOS.items():
            b=a.copy();b['tn_scenario']=name
            b['tn_load_kg_n_yr']=b.tdn_2012_t_n_yr*b.province_tn_load_factor_relative_2012/fraction*1000.
            rows.append(b[['model_reach_id','year','tn_scenario','facilities','tn_load_kg_n_yr']])
    annual=pd.concat(rows,ignore_index=True).groupby(['model_reach_id','year','tn_scenario'],as_index=False).agg(
        facilities=('facilities','sum'),tn_load_kg_n_yr=('tn_load_kg_n_yr','sum'))
    monthly=[]
    for month in range(1,13):
        a=annual.copy();a['month']=month
        a['tn_load_kg_n_month']=a.tn_load_kg_n_yr*np.array([calendar.monthrange(int(y),month)[1] for y in a.year])/np.array([366 if calendar.isleap(int(y)) else 365 for y in a.year])
        monthly.append(a[['model_reach_id','year','month','tn_scenario','facilities','tn_load_kg_n_month']])
    return annual,pd.concat(monthly,ignore_index=True)
