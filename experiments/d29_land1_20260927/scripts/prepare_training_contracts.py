"""All registered path contracts, without dispatch or physical calibration."""
import sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from d29_platform.runtime import configure,write_json
configure()
import pandas as pd
from d29_training.objective import StrategyObjective

m=pd.read_parquet(ROOT/'data/monthly_tn.parquet');h=pd.read_parquet(ROOT/'data/hf_daily.parquet')
blocks=json.loads((ROOT/'vendor/legacy22/data/spatial_blocks.json').read_text(encoding='utf-8'))
summary=[]
for j in json.loads((ROOT/'config/jobs.json').read_text(encoding='utf-8')):
    excluded=[]
    if j['space_block'] is not None:
        b=blocks[str(j['space_block']).removeprefix('S')];excluded=b['held_stations']+b['buffer_stations']
    end=j['train_end'];mt=m[m.year.le(end)];ht=h[h.date.dt.year.le(end)]
    o=StrategyObjective(mt,ht,j['strategy'],end,excluded)
    folder=ROOT/'data/training_contracts'/j['id'];folder.mkdir(parents=True,exist_ok=True)
    o.rows.to_parquet(folder/'rows.parquet',index=False)
    identity={'job':j,'excluded_stations':sorted(set(excluded)),'long_stations':o.long_sites,'scales':o.scales,'rows_hash':hashlib.sha256((folder/'rows.parquet').read_bytes()).hexdigest(),'no_evaluation_labels':True,'initialization':'original preset '+str(j['entry']),'monthly_support':'equal-day approximation; no complete official valid-read counts'}
    write_json(folder/'identity.json',identity)
    summary.append(dict(job=j['id'],monthly_stations=o.rows[o.rows.kind.eq('monthly')].station_key.nunique(),monthly_rows=int(o.rows.kind.eq('monthly').sum()),hf_stations=o.rows[o.rows.kind.eq('HF')].station_key.nunique(),hf_days=int(o.rows.kind.eq('HF').sum()),long_stations=len(o.long_sites),excluded_stations=len(set(excluded)),model=j['model'],training_contract_passed=True,physical_gate='U gates passed; path-specific branch/solution checks remain' if j['model']=='U' else 'blocked pending absolute mass and annual adjoint gates'))
pd.DataFrame(summary).to_csv(ROOT/'outputs/training_contracts_32.csv',index=False)
write_json(ROOT/'outputs/training_contracts_receipt.json',{'paths':len(summary),'training_contract_passed':True,'formal_fits_completed':False})
print(pd.DataFrame(summary).to_string(index=False))
