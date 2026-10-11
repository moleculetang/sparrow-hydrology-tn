"""Strict legal-label complete-history point on Linux; no production fit."""
import json
import pickle
import sys
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts'), str(ROOT/'work'), str(ROOT/'vendor/legacy22/scripts')]
from d29_platform.runtime import configure, sha, write_json
configure()
import numpy as np
import pandas as pd
from training_only_constructor import build_training
from timing.independent_objective import recompute
from native_bound_adapter_v2 import CanonicalLBFGSB, restore_canonical_solver


def main():
    out = ROOT/'outputs/scope_remote_point'
    out.mkdir(parents=True, exist_ok=True)
    if (out/'complete.json').exists():
        raise RuntimeError('EXISTING_POINT_RECEIPT_REVIEW_REQUIRED')
    local = json.loads((ROOT/'outputs/scope_strict_point/complete.json').read_text(encoding='utf8'))
    manifest = json.loads((ROOT/'deployment/candidate_manifest.json').read_text(encoding='utf8'))
    for row in manifest['files']:
        if sha(ROOT/row['path']) != row['sha256']:
            raise RuntimeError('CANDIDATE_BYTES_CHANGED ' + row['path'])
    jobs = json.loads((ROOT/'config/jobs.json').read_text(encoding='utf8'))
    job = next(j.copy() for j in jobs if j['id']=='TIM_F23_K_V1_s0')
    job['id'] = 'scope_remote_point_F23_K_V1'
    reader = pd.read_parquet
    reads = []
    def observed(path, *args, **kwargs):
        result = reader(path, *args, **kwargs)
        p = Path(path).resolve()
        if p in [(ROOT/'data/monthly_tn.parquet').resolve(), (ROOT/'data/hf_daily.parquet').resolve()]:
            years = result.year if 'year' in result else result.date.dt.year
            invalid = int(years.gt(job['train_end']).sum())
            reads.append(dict(file=p.relative_to(ROOT).as_posix(), rows=len(result),
                              evaluation_year_rows_returned=invalid, filters=repr(kwargs.get('filters'))))
            if invalid:
                raise RuntimeError('EVALUATION_ROWS_EXPOSED')
        return result
    started = time.monotonic()
    with patch.object(pd, 'read_parquet', side_effect=observed):
        a = build_training(job, acceptance=True)
    if len(reads) != 2:
        raise RuntimeError('BOTH_LEGAL_TRAINING_READS_REQUIRED')
    point = np.load(ROOT/'outputs/scope_strict_point/point.npy', allow_pickle=False)
    if np.count_nonzero(np.all(a.bounds == [-.1,0.], axis=1)) != 1:
        raise RuntimeError('ONE_CORRECT_EXPANDED_MAIZE_BOUND_REQUIRED')
    value, gradient = a.value_gradient(point)
    independent = sum(recompute(a.objective.rows,a.prediction,a.objective.scales,a.objective.long_sites).values()) + a.last['prior'] + a.last['management_prior']
    if abs(value-independent) > 1e-8*(1+abs(value)) or a.last['fixed_source_multiplier'] != 1.:
        raise RuntimeError('INDEPENDENT_OBJECTIVE_OR_MULTIPLIER_FAILURE')
    # Fresh native memory, never loaded from an old narrow-K optimizer.
    bounds = a.bounds/a.scale[:,None]
    solver = CanonicalLBFGSB(point/a.scale, bounds)
    if solver.step(None) != 'request_evaluation':
        raise RuntimeError('NATIVE_INITIAL_REQUEST_REQUIRED')
    resumed = restore_canonical_solver(pickle.loads(pickle.dumps(solver.s, protocol=5)))
    def current_oracle(x):
        if not np.array_equal(x, point/a.scale):
            raise RuntimeError('THIS_PROBE_ONLY_REPLAYS_ITS_ACTUAL_CURRENT_POINT')
        return value, gradient*a.scale
    if solver.step(current_oracle) != 'evaluation' or resumed.step(current_oracle) != 'evaluation':
        raise RuntimeError('FRESH_CANONICAL_RESTORE_EVALUATION_MISMATCH')
    if solver.step(None) != resumed.step(None):
        raise RuntimeError('FRESH_CANONICAL_RESTORE_NEXT_EVENT_MISMATCH')
    for key, state in solver.s.items():
        other = resumed.s[key]
        equal = np.array_equal(state, other) if isinstance(state, np.ndarray) else state == other
        if not equal:
            raise RuntimeError('FRESH_CANONICAL_RESTORE_MEMORY_MISMATCH ' + key)
    arrays = [('point',point), ('gradient',gradient), ('training_prediction',a.prediction),
              ('local_kg_day',a.local_prediction), ('final_state',a.last_chain.final)]
    for name, array in arrays:
        np.save(out/(name+'.npy'),array)
    from train_job import backend
    from threadpoolctl import threadpool_info
    pools = threadpool_info()
    if any(p['num_threads'] != 1 for p in pools):
        raise RuntimeError('ONE_LIBRARY_THREAD_REQUIRED')
    write_json(out/'complete.json',dict(passed=True,
        scope='strict_legal_label_complete_history_expanded_point_and_fresh_native_restore_not_fit',
        value=float(value), independent_objective=float(independent), constructor_reads=reads,
        source_multiplier=1., config_sha256=sha(ROOT/'config/timing_evidence.json'),
        candidate_manifest_sha256=sha(ROOT/'deployment/candidate_manifest.json'),
        reference_receipt_sha256=sha(ROOT/'outputs/scope_strict_point/complete.json'),
        array_sha256={n:sha(out/(n+'.npy')) for n,_ in arrays},backend=backend(),
        actual_library_pools=pools, fresh_canonical_restore_passed=True,
        seconds=time.monotonic()-started, evaluation_labels_read=False,
        production_gate_passed=False,dualhost_comparison_pending=True))
    print(json.dumps(dict(passed=True,seconds=time.monotonic()-started)))


if __name__ == '__main__':
    main()
