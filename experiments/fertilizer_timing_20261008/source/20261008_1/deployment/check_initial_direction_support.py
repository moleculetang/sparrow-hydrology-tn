"""Expose the general zero-pool fixture's off-area directions, preserve evidence."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from d29_platform.luh3_adapter import LAND_STATES
from d29_platform.runtime import write_json,sha


def main():
    source=ROOT/'inputs/land_support/outputs/luh3_reach/luh3_reach_state_1961_2024.csv'
    frame=pd.read_csv(source);frame=frame[(frame.year==1961)&frame.state.isin(LAND_STATES)]
    index=pd.MultiIndex.from_product([range(1,231),LAND_STATES],names=['reach_id','state'])
    values=frame.set_index(['reach_id','state']).physical_area_m2.reindex(index)
    if values.isna().any():raise RuntimeError('MISSING1961LAND_SUPPORT')
    area=np.zeros((230,13));area[:,:12]=values.to_numpy().reshape(230,12)
    config=json.loads((ROOT/'config/conditional_reference.json').read_text(encoding='utf8'))
    mask=np.zeros((230,13),bool);mask[:,config['land1_integration']['active_land_slots']]=True
    wrong=mask&(area==0);valid=mask&(area>0)
    initialfile=ROOT/'outputs/jobs/prerequisite_full_initial_feedback/initial_point.npy'
    initial=np.load(initialfile,allow_pickle=False)
    positive_wrong={name:int(np.count_nonzero((initial[:,:,k]>0)&(area==0))) for k,name in enumerate(('P','Sa','Sp'))}
    receipt=dict(passed=bool(np.any(wrong) and not any(positive_wrong.values())),
        scope='fixture_direction_support_failure_and_corrected_admissible_initial_qualification',
        original_zero_pool_fixture_support_passed=not bool(np.any(wrong)),
        original_zero_pool_direction_cells=int(mask.sum()),off_area_cells=int(wrong.sum()),
        corrected_positive_area_active_cells=int(valid.sum()),positive_pool_off_area_cells=positive_wrong,
        original_initial_input_modified=False,original_partial_receipt_modified=False,
        correct_zero_pool_receipt='evidence/supported_zero_initial_feedback.json',
        original_general_fixture_remains_diagnostic=True,formal_gate_issued=False,
        examples=[dict(reach_id=int(r+1),land_slot=int(l)) for r,l in np.argwhere(wrong)[:12]],
        identities={source.relative_to(ROOT).as_posix():sha(source),'config/conditional_reference.json':sha(ROOT/'config/conditional_reference.json')},
        initial_point_sha256=sha(initialfile))
    write_json(ROOT/'evidence/initial_direction_support_failure_and_fix.json',receipt)
    print(json.dumps(receipt))


if __name__=='__main__':main()
