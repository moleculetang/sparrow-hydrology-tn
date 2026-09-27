"""Read-only audit of accepted LUH3 transfer rows and area mapping."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from d29_platform.luh3_adapter import load_area_transfers,LAND_STATES
from d29_platform.runtime import sha,write_json


def main():
    base=ROOT.parent/'20260926_1/outputs/luh3_reach'
    states=base/'luh3_reach_state_1961_2024.csv'
    operator=base/'luh3_land_area_transfer_operator_1961_2023.csv'
    dates=pd.date_range('1961-01-01','2024-12-31',freq='D')
    area,events=load_area_transfers(states,operator,dates)
    worst_row=(-1.,None)
    worst_area=(-1.,None)
    worst_row_mass_proxy=(-1.,None)
    # Multiplying row error by the actual initial SON scenario gives a scale
    # check, not a claim about the stock present in each later year.
    # Initial SON is reconstructed directly from the saved density to avoid
    # reading unrelated vegetation/observation products.
    soil=pd.read_csv(ROOT/'outputs/soil_reference/son_soc14_reference.csv')
    config=__import__('json').loads((ROOT/'config/conditional_reference.json').read_text(encoding='utf-8'))
    density=soil[soil.depth==config['son']['depth']].set_index('reach_id').reindex(range(1,231)).son_reference_kg_m2.to_numpy()
    initial_son=area[0]*density[:,None]
    for day,matrix in events.items():
        effective=dates[day]
        year=effective.year-1
        old=area[year-1961]
        new=area[year+1-1961]
        row=np.abs(matrix.sum(-1)-1.)
        rid,state=np.unravel_index(np.argmax(row),row.shape)
        if float(row[rid,state])>worst_row[0]:
            worst_row=(float(row[rid,state]),dict(operator_year=int(year),reach_id=int(rid+1),
                                                   source_state=LAND_STATES[state]))
        predicted=np.einsum('ri,rij->rj',old,matrix)
        difference=np.abs(predicted-new)
        rid,state=np.unravel_index(np.argmax(difference),difference.shape)
        if float(difference[rid,state])>worst_area[0]:
            worst_area=(float(difference[rid,state]),dict(operator_year=int(year),reach_id=int(rid+1),
                                                          target_state=LAND_STATES[state],
                                                          predicted_area_m2=float(predicted[rid,state]),
                                                          saved_area_m2=float(new[rid,state])))
        proxy=row*initial_son
        rid,state=np.unravel_index(np.argmax(proxy),proxy.shape)
        if float(proxy[rid,state])>worst_row_mass_proxy[0]:
            worst_row_mass_proxy=(float(proxy[rid,state]),dict(operator_year=int(year),
                reach_id=int(rid+1),source_state=LAND_STATES[state],
                initial_SON_scenario_kg=float(initial_son[rid,state])))
    receipt=dict(transition_count=len(events),maximum_row_sum_abs_error=worst_row,
                 maximum_area_mapping_abs_error_m2=worst_area,
                 maximum_row_error_times_initial_SON_scenario_kg=worst_row_mass_proxy,
                 source_sha256={str(states):sha(states),str(operator):sha(operator)},
                 role='read-only coefficient and area audit, not a full state-budget audit')
    write_json(ROOT/'outputs/luh3_transfer_precision_audit.json',receipt)


if __name__=='__main__':main()
