"""Training-only selection and complete campaign accounting; no label access."""
import math

LIVE = {'pending', 'running', 'continuing_same_path_zero_ftol', 'resource_checkpoint'}


def select_campaign(jobs, records, exclusions):
    ids = {j['id'] for j in jobs}
    if (set(records) | set(exclusions)) != ids or set(records) & set(exclusions):
        raise ValueError('EXACT_DISJOINT_PATH_ACCOUNTING_REQUIRED')
    if any(not str(reason).strip() for reason in exclusions.values()):
        raise ValueError('EXCLUSION_REASON_REQUIRED')
    groups = {}
    for j in jobs:
        key = (j['model'], j['fold'], j['strategy'], j['space_block'])
        groups.setdefault(key, [])
        if j['id'] in exclusions:
            continue
        r = records[j['id']]
        if r['status'] in LIVE:
            raise ValueError('LIVE_CAMPAIGN')
        if not r.get('objective_passed') or not r.get('physical_passed'):
            raise ValueError('UNACCEPTED_CHECKPOINT')
        if not math.isfinite(r['objective']):
            raise ValueError('NONFINITE_TRAINING_OBJECTIVE')
        groups[key].append((float(r['objective']), j['entry'], j['id']))
    selected = [min(candidates)[2] for candidates in groups.values() if candidates]
    selected_set = set(selected)
    for j in jobs:
        if j['id'] in selected_set and j['strategy'] != 'T0':
            if not any(k['id'] in selected_set and k['model'] == j['model'] and
                       k['fold'] == j['fold'] and k['space_block'] == j['space_block'] and
                       k['strategy'] == 'T0' for k in jobs):
                raise ValueError('SELECTED_STRATEGY_WITHOUT_BASELINE')
    return {'selected_jobs': selected, 'excluded_jobs': exclusions,
            'unselected_legal_jobs': sorted(set(records)-selected_set),
            'accounted_paths': sorted(ids), 'selection_training_only': True,
            'selection_rule': 'minimum independently verified training total objective within configuration; numerical sufficiency reported separately',
            'training_records': records}


def year_role(job, year, heldout=False):
    start = 2021 if job['strategy'] == 'T0' else 2016
    if job['space_block'] is not None:
        if not heldout:
            return 'mixed_training_buffer_holdout_descriptive'
        return 'same_year_spatial' if year <= job['train_end'] else 'space_time'
    if start <= year <= job['train_end']:
        return 'training_reconstruction'
    if year < start:
        return 'historical_backcast_not_training'
    return 'retrospective_time_evaluation'
