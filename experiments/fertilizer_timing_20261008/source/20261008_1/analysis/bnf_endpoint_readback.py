"""Independent compensated endpoint check against the original BNF tracer.

Calendar diagnostic tags are grouped back to the existing crop_BNF label,
separately from the other sources/initial state. Absolute tolerances remain
per reach/physical land/state, never on differently rounded basin totals.
"""
import math
from pathlib import Path
import numpy as np


def grouped_errors(new_high,new_compensation,original_high,original_compensation):
    arrays=[np.asarray(a,dtype=np.float64) for a in
            (new_high,new_compensation,original_high,original_compensation)]
    high,corr,old,old_corr=arrays
    if (high.shape!=(230,13,5,5) or old.shape!=high.shape
            or corr.shape!=(230,13,5,4) or old_corr.shape!=corr.shape
            or any(not np.isfinite(a).all() for a in arrays)):
        raise ValueError('FINITE230REACH13LAND5STATE_BNF_ENDPOINT_REQUIRED')
    worst={'all_BNF':0.,'other_sources_and_initial':0.}
    for r in range(230):
        for land in range(13):
            for state in range(5):
                for key,new_indices,old_indices in (
                        ('all_BNF',(1,2,3,4),(3,)),
                        ('other_sources_and_initial',(0,),(0,1,2,4))):
                    terms=[high[r,land,k,state] for k in new_indices]
                    terms.extend(-old[r,land,k,state] for k in old_indices)
                    if state:
                        terms.extend(-corr[r,land,k,state-1] for k in new_indices)
                        terms.extend(old_corr[r,land,k,state-1] for k in old_indices)
                    worst[key]=max(worst[key],abs(math.fsum(terms)))
    return dict(passed=all(x<=1e-6 for x in worst.values()),max_abs_error_kg=worst,
                tolerance_kg=1e-6,scope='per_reach_physical_land_state',
                NSE='not_applicable',equivalence_is_not_causal_BNF_removal_effect=True)


def verify(destination,source):
    from d29_platform.runtime import sha
    from bnf_calendar_partition import LABELS
    import json
    destination=Path(destination);source=Path(source)
    value=json.loads((destination/'complete.json').read_text())
    physical=json.loads((source/'physical_ledger.json').read_text())
    original_file=source/'full_history_physical_source_ledger.npz'
    if (value.get('passed') is not True or value.get('labels')!=list(LABELS)
            or physical.get('labels')!=['deposition','fertilizer','field_manure','crop_BNF','initial_research_state']
            or physical.get('passed') is not True or sha(original_file)!=physical.get('full_history_ledger_sha256')
            or value.get('bound_files',{}).get('full_history_physical_source_ledger.npz')!=sha(original_file)):
        raise RuntimeError('EXACT_ORIGINAL_AND_BNF_DIAGNOSTIC_ENDPOINT_BINDINGS_REQUIRED')
    row=next(r for r in value['years'] if r['year']==2024)
    yearfile=(destination/row['file']).resolve()
    if not yearfile.is_relative_to((destination/'yearly').resolve()) or sha(yearfile)!=row['sha256']:
        raise RuntimeError('BNF_FINAL_YEAR_BYTES_CHANGED')
    with np.load(yearfile,allow_pickle=False) as new,np.load(original_file,allow_pickle=False) as old:
        result=grouped_errors(new['end_year_high'],new['end_year_compensation'],
                             old['final_label_stocks'],old['final_label_compensation'])
    if not result['passed']:raise RuntimeError('REGROUPED_BNF_INVENTORY_DIFFERS_FROM_ORIGINAL_TRACER')
    return dict(**result,diagnostic_complete_sha256=sha(destination/'complete.json'),
        original_ledger_sha256=sha(original_file),diagnostic_final_year_sha256=sha(yearfile),
        readback_sha256=sha(Path(__file__)),evaluation_labels_read=False)
