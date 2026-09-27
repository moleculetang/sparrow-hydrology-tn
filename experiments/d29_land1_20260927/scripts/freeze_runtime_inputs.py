"""Record source bytes and independently match all prefit training contracts."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from d29_platform.runtime import configure,write_json,sha
configure()
import pandas as pd
from d29_training.objective import StrategyObjective
out=ROOT/'evidence/runtime_input_identity.json'
if out.exists():raise RuntimeError('RUNTIME_IDENTITY_ALREADY_FROZEN')
monthly=pd.read_parquet(ROOT/'data/monthly_tn.parquet');hf=pd.read_parquet(ROOT/'data/hf_daily.parquet')
receipts=[]
for j in json.loads((ROOT/'config/jobs.json').read_text(encoding='utf-8')):
    folder=ROOT/'data/training_contracts'/j['id'];identity=json.loads((folder/'identity.json').read_text(encoding='utf-8'))
    saved=pd.read_parquet(folder/'rows.parquet')
    if sha(folder/'rows.parquet')!=identity['rows_hash']:raise RuntimeError('PREFIT_CONTRACT_CHANGED')
    o=StrategyObjective(monthly[monthly.year.le(j['train_end'])],hf[hf.date.dt.year.le(j['train_end'])],j['strategy'],j['train_end'],identity['excluded_stations'])
    # Parquet restores absent mixed boolean metadata as None rather than NaN.
    # Normalize only missing-value representation; compare every nonmissing
    # scalar exactly (no floating-value tolerance for a frozen training label).
    current=o.rows.astype(object).where(o.rows.notna(),None)
    restored=saved.astype(object).where(saved.notna(),None)
    pd.testing.assert_frame_equal(current,restored,check_dtype=False,check_exact=True)
    if o.scales!=identity['scales'] or o.long_sites!=identity['long_stations']:raise RuntimeError('PREFIT_TRANSFORM_CHANGED')
    receipts.append({'path':j['id'],'training_contract_unchanged':True,'rows_sha256':sha(folder/'rows.parquet')})
paths=list((ROOT/'data').glob('*.parquet'))+[ROOT/'config/jobs.json',ROOT/'config/conditional_reference.json',ROOT/'config/mineralization_reference.json',ROOT/'d29_platform/conditional_inputs.py',ROOT/'d29_platform/luh3_adapter.py']
paths+=list((ROOT/'outputs/agriculture_reference').glob('*.parquet'))+[ROOT/'outputs/soil_reference/son_soc14_reference.csv']
parent=ROOT.parent/'20260926_1'
paths+=[parent/'outputs/luh3_reach/luh3_reach_state_1961_2024.csv',parent/'outputs/luh3_reach/luh3_land_area_transfer_operator_1961_2023.csv',parent/'outputs/deposition_landclass/monthly_reach_clcd_class_deposition_1961_2024.parquet',parent/'outputs/mod17_gee_reachcover/reach_clcd_class_annual_npp_qc_2001_2025.csv']
write_json(out,{'files':{str(p):sha(p) for p in paths},'training_contracts':receipts,'captured_after_U_started':True,'training_data_identity_supported_by_original_prefit_contract_hashes':True,'H1_and_original_U_sources':'independently hash-verified by cm.load_data(...,verify=True)','no_new_evaluation_readout':True})
print('Verified',len(receipts),'prefit contracts; frozen',len(paths),'runtime inputs')
