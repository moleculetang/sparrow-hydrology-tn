"""Audit exposure-feature semantics independently of TN labels and model fits."""
import pandas as pd
from mltn.common import ROOT,write,sha

def main():
    path=ROOT.parent/'20260926_1/outputs/deposition_landclass/monthly_reach_clcd_class_deposition_1961_2024.parquet'
    columns=['drynhx_kg_n','drynoy_kg_n','wetnhx_kg_n','wetnoy_kg_n']
    q=pd.read_parquet(path,columns=['year','clcd_class',*columns]);q=q[q.year.between(2015,2024)].copy()
    assert q[columns].notna().all().all() and q[columns].ge(0).all().all()
    q['mass']=q[columns].sum(axis=1);rows=[]
    for year,g in q.groupby('year'):
        included=float(g.loc[g.clcd_class.ne(5),'mass'].sum());unknown=float(g.loc[g.clcd_class.eq(0),'mass'].sum())
        rows.append(dict(year=int(year),included_nonwater_exposure_kg_n=included,unknown_landclass_exposure_kg_n=unknown,unknown_fraction_of_feature_exposure=unknown/included if included>0 else None,excluded_water_kg_n=float(g.loc[g.clcd_class.eq(5),'mass'].sum()),known_nonwater_exposure_kg_n=included-unknown))
    write(ROOT/'evidence/deposition_feature_class_identity.json',dict(source=str(path),source_sha256=sha(path),rows=rows,actual_feature='prepare.py excludes class5 water; class0 estimated exposure is included in four monthly covariates',identity='unknown class remains uncertain; feature is not a physical known-land external-source mass interface',monthly_support='constant current-month retrospective kg N/ha/month covariate; no wet-deposition daily rainfall division',labels_read=False,fitting_calls=0,changed_input_arrays=False))

if __name__=='__main__':main()
