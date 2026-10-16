"""Real1961--2024 physical isolation and parameter-point replay.

Changes only future/heldout TN values at the data-reader boundary. No evaluation
score is read or computed. Uses a dedicated output ID, never a production job.
Run through guarded_worker.py under a real host-local scientific resource lease.
"""
import gc
import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from d29_platform.runtime import write_json, sha
import numpy as np
import pandas as pd
from timing.adapter import TimingTraining


def array_hash(array):
    a = np.ascontiguousarray(array)
    return hashlib.sha256(str(a.shape).encode()+str(a.dtype).encode()+a.tobytes()).hexdigest()


def main():
    jobs = json.loads((ROOT/'config/jobs.json').read_text(encoding='utf8'))
    blocks = json.loads((ROOT/'vendor/legacy22/data/spatial_blocks.json').read_text(encoding='utf8'))
    chosen = ['TIM_F23_C0_V0_s0', 'TIM_F24_K_V1_s0'] + [f'TIM_S{s}_K_V1_s0' for s in (56,113,191)]
    receipt_path = ROOT/'evidence/full_prediction_isolation.json'
    identities = {str(p.relative_to(ROOT)).replace('\\','/'):sha(p)
                  for directory in ('timing','d29_platform','d29_training','config')
                  for p in (ROOT/directory).glob('*.py' if directory!='config' else '*.json')}
    records = []
    if receipt_path.exists():
        old = json.loads(receipt_path.read_text(encoding='utf8'))
        if old['identities'] != identities: raise RuntimeError('ISOLATION_IDENTITY_CHANGED')
        records = old['cases']
    reader = pd.read_parquet
    for jid in chosen:
        if any(r['registered_job']==jid for r in records): continue
        job = next(j.copy() for j in jobs if j['id']==jid)
        job['id'] = 'prerequisite_isolation_'+jid
        excluded = []
        if job['space_block'] is not None:
            b = blocks[str(job['space_block'])]
            excluded = b['held_stations']+b['buffer_stations']
        def read_changed(path, *args, **kwargs):
            frame = reader(path, *args, **kwargs)
            if Path(path).resolve() in [(ROOT/'data/monthly_tn.parquet').resolve(), (ROOT/'data/hf_daily.parquet').resolve()]:
                frame = frame.copy()
                years = frame.year if 'year' in frame else frame.date.dt.year
                outside = (years>job['train_end']) | frame.station_key.isin(excluded)
                frame.loc[outside, 'tn_mg_l'] = 1e9
            return frame
        original = TimingTraining(job, acceptance=True)
        point = original.initial.copy()
        v, _ = original.value_gradient(point, True)
        prediction = original.prediction.copy(); local = original.local_prediction.copy()
        initial = original.inputs.initial().copy()
        design = json.dumps(original.design, sort_keys=True)
        scales = original.objective.scales.copy(); groups = original.objective.long_sites.copy()
        rows = original.objective.rows.copy()
        # Persist and reload the exact original-coordinate point, then rebuild
        # all yearly warmup states; never carry warmup state from the first call.
        pointfile = original.folder/'isolation_point.npy'
        np.save(pointfile, point)
        del original; gc.collect()
        with patch.object(pd, 'read_parquet', side_effect=read_changed):
            changed = TimingTraining(job, acceptance=True)
            assert np.array_equal(point, changed.initial)
            assert np.array_equal(initial, changed.inputs.initial())
            assert design == json.dumps(changed.design, sort_keys=True)
            assert scales == changed.objective.scales and groups == changed.objective.long_sites
            pd.testing.assert_frame_equal(rows, changed.objective.rows)
            reloaded = np.load(pointfile, allow_pickle=False)
            w, _ = changed.value_gradient(reloaded, True)
        assert v == w and np.array_equal(prediction, changed.prediction)
        assert np.array_equal(local, changed.local_prediction)
        records.append(dict(registered_job=jid, history=[1961,2024], reaches=230,
            initial_parameters_unchanged=True, initial_stocks_unchanged=True,
            physical_design_unchanged=True, training_groups_scales_rows_unchanged=True,
            exact_warmup_rebuild_point_replay=True, prediction_equal=True, local_flux_equal=True,
            prediction_sha256=array_hash(prediction), local_flux_sha256=array_hash(local),
            independent_point_file_sha256=sha(pointfile), objective=v, excluded_count=len(set(excluded))))
        write_json(receipt_path, dict(passed=len(records)==len(chosen), cases=records,
            identities=identities, full_history_forward_calls=2*len(records),
            scope='full_physical_prediction_isolation_and_parameter_point_replay_not_optimizer_state_recovery',
            no_heldout_scores_computed=True, optimizer_state_recovery_separate=True))
        del changed; gc.collect()
    print(json.dumps(dict(passed=True, completed_supports=len(records))))


if __name__ == '__main__': main()
