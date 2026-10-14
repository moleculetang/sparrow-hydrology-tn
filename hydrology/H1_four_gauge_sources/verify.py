"""Independent source/readback checks; no original workspace is needed."""
import json,hashlib,calendar,csv
from pathlib import Path
import pandas as pd
import numpy as np

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify(home):
    home=Path(home);assets=home/'assets';manifest=json.loads((home/'delivery_manifest.json').read_text())
    for row in manifest['members']:
        path=assets/row['path']
        if path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:raise RuntimeError('ASSET_BYTES_CHANGED '+row['path'])
    lineage=pd.read_parquet(assets/'four_gauge_daily_lineage.parquet');index=pd.read_csv(assets/'station_year_source_index.csv')
    if len(lineage)!=13148 or lineage.duplicated(['station_norm','date']).any() or len(index)!=36:
        raise RuntimeError('EXACT_FOUR_STATION_YEAR_COVERAGE_REQUIRED')
    dates=pd.date_range('2010-01-01','2018-12-31');csv_records=[]
    months=['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
    for row in index.itertuples():
        p=assets/row.raw_csv
        if sha(p)!=row.source_sha256:raise RuntimeError('ORIGINAL_CSV_HASH_DIFFERS')
        raw=pd.read_csv(p,dtype=str)
        for i,v in raw.iterrows():
            day=pd.to_numeric(v.day,errors='coerce')
            if not np.isfinite(day) or day!=int(day):continue
            for m,column in enumerate(months,1):
                if not 1<=day<=calendar.monthrange(int(row.year),m)[1]:continue
                number=pd.to_numeric(v[column],errors='coerce')
                q=float(number) if np.isfinite(number) and number>=0 else np.nan
                csv_records.append(dict(station_norm=row.station_norm,date=pd.Timestamp(int(row.year),m,int(day)),q_m3_s=q,
                    source_row=int(i)+2,source_column=column))
    recreated=pd.DataFrame(csv_records)
    c=lineage.merge(recreated,on=['station_norm','date'],validate='one_to_one',how='outer',suffixes=('_delivered','_recreated'),indicator=True)
    same=c.q_m3_s_delivered.eq(c.q_m3_s_recreated)|c.q_m3_s_delivered.isna()&c.q_m3_s_recreated.isna()
    if not c['_merge'].eq('both').all() or not same.all() or not c.source_row_delivered.eq(c.source_row_recreated).all() or not c.source_column_delivered.eq(c.source_column_recreated).all():
        raise RuntimeError('RECREATED_DAILY_CSV_LINEAGE_DIFFERS')
    for station,group in lineage.groupby('station_norm'):
        if not pd.DatetimeIndex(group.date).sort_values().equals(dates):raise RuntimeError('REAL_CALENDAR_OR_LEAP_DAY_GAP')
    panel=pd.read_csv(assets/'primary_Q_53_station_manifest.csv')
    if len(panel)!=53 or panel.station_norm.nunique()!=53 or set(panel.station_norm)&set(lineage.station_norm):
        raise RuntimeError('EXACT53_PRIMARY_EXCLUDING_FOUR_REQUIRED')
    semantics=pd.read_csv(assets/'station_identity_and_semantics.csv')
    if len(semantics)!=4 or semantics.official_stable_station_code.notna().any() or semantics.sensitivity_57_admitted.any():
        raise RuntimeError('MISSING_OFFICIAL_IDENTITIES_MUST_STAY_MISSING')
    failure=json.loads((assets/'wrong_table_failure_receipt.json').read_text())
    cmp=pd.read_csv(assets/'lujiang_2018_wrong_table_comparison.csv')
    if len(cmp)!=365 or int(cmp.screen_matches_neighbor.sum())!=failure['screen_matches_neighbor_days'] or int((~cmp.screen_matches_main_table&cmp.q_m3_s_main_table.notna()).sum())!=failure['screen_differs_main_table_days']:
        raise RuntimeError('WRONG_TABLE_EVIDENCE_CHANGED')
    jan1=cmp[cmp.date.eq('2018-01-01')].iloc[0]
    if jan1.q_m3_s_screen!=.608 or jan1.q_m3_s_main_table!=23.:
        raise RuntimeError('VISIBLE_RAW_PAGE_COUNTEREXAMPLE_CHANGED')
    flags=pd.read_parquet(assets/'existing_ocr_cell_review_flags.parquet')
    if len(flags)!=5429 or not flags.flag_kind.eq('existing_OCR_review_metadata_not_instrument_QC').all():
        raise RuntimeError('EXISTING_FLAGS_OR_SCOPE_CHANGED')
    return dict(passed=True,members=len(manifest['members']),station_years=36,station_days=13148,
        raw_daily_recreation_max_error=0.,primary_Q_stations=53,existing_OCR_flags=len(flags),
        lujiang_2018_wrong_table_confirmed=True,sensitivity_57_admitted=False,
        official_codes_and_instrument_QC_certified=False,NSE='not_applicable_to_source_verification')

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--root',default=str(Path(__file__).resolve().parent));args=parser.parse_args()
    print(json.dumps(verify(args.root),ensure_ascii=False,indent=2))
