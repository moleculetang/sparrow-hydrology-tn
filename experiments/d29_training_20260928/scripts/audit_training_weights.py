"""Report actual training-block support and weight changes before fitting."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from d29_platform.runtime import write_json
from d29_training.objective import StrategyObjective
import numpy as np
import pandas as pd

m=pd.read_parquet(ROOT/'data/monthly_tn.parquet')
h=pd.read_parquet(ROOT/'data/hf_daily.parquet')
blocks=json.loads((ROOT/'vendor/legacy22/data/spatial_blocks.json').read_text(encoding='utf-8'))
rows=[]
for j in json.loads((ROOT/'config/jobs.json').read_text(encoding='utf-8')):
    if j['model']!='U':continue  # LAND1 has exactly the same observation contracts.
    excluded=[] if j['space_block'] is None else blocks[str(j['space_block'])]['held_stations']+blocks[str(j['space_block'])]['buffer_stations']
    o=StrategyObjective(m[m.year.le(j['train_end'])],h[h.date.dt.year.le(j['train_end'])],j['strategy'],j['train_end'],excluded)
    for block in o.blocks:
        frame=o.rows.iloc[block.indices]
        kind='HF' if block.name=='HF_anomaly' else 'monthly'
        variance=frame.station_key.map(o.scales[kind]['station_variance']).fillna(o.scales[kind]['variance_floor']).to_numpy(float)
        multiplier=.2 if kind=='HF' else .4 if j['strategy']=='T2' else .8
        raw=(block.sqrt_weight**2)*variance/multiplier
        standardized=block.sqrt_weight**2
        rows.append({'job':j['id'],'fold':j['fold'],'strategy':j['strategy'],'space_block':j['space_block'],
            'block':block.name,'stations':int(frame.station_key.nunique()),'rows':len(frame),
            'long_group_stations':len(o.long_sites),'other_group_stations':int(o.rows[o.rows.kind.eq('monthly')].station_key.nunique()-len(o.long_sites)),
            'raw_weight_total':float(raw.sum()),'early_raw_weight_fraction':float(raw[frame.year.to_numpy()<2021].sum()/raw.sum()),
            'recent_raw_weight_fraction':float(raw[frame.year.to_numpy()>=2021].sum()/raw.sum()),
            'maximum_raw_record_weight':float(raw.max()),'median_raw_record_weight':float(np.median(raw)),
            'effective_standardized_record_count':float(standardized.sum()**2/(standardized@standardized)),
            'scale_floor':o.scales[kind]['variance_floor']})
table=pd.DataFrame(rows)
table.to_csv(ROOT/'outputs/training_weight_audit.csv',index=False)
write_json(ROOT/'outputs/training_weight_audit.json',{'rows':len(table),'T2_note':'The 0.4/0.4 group allocation is not intrinsically more weight per long station; inspect group counts.','comparison':'shared recent-fold scales; historical years also change balanced observation weights, so T0/T1/T2 are complete strategy comparisons','NH4_DO':'excluded','file':'training_weight_audit.csv'})
