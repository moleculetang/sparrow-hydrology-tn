"""Freeze all44owners only from completed, verified prerequisite/pilot receipts.

Selection uses computation throughput, never heldout outcomes. Actual dispatch
still obeys the shared host registry and can be lower than the measured ceiling.
"""
import json
import math
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from train_job import validate_admission
from d29_platform.runtime import sha,write_json


def read(relative):return json.loads((ROOT/relative).read_text(encoding='utf8'))


def main():
    gate,gate_hash=validate_admission()
    for rel,digest in gate['proof_sha256'].items():
        if sha(ROOT/rel)!=digest:raise RuntimeError('SEALED_PREREQUISITE_PROOF_CHANGED '+rel)
    replays=read('evidence/fixed_replays.json')
    stages=read('evidence/stage_measurement_consistency.json')
    transport=read('evidence/workstation_parallel_target_import.json')
    pilot_rel='outputs/imported_workstation_parallel_targets/outputs/parallel_target_pilot/status.json'
    pilot=read(pilot_rel)
    if not replays['passed'] or replays['count']!=12 or replays['formal_gate_sha256']!=gate_hash:
        raise RuntimeError('ALL12CURRENT_FIXED_REPLAYS_REQUIRED')
    if not stages['passed'] or stages['pending'] or len(stages['cases'])!=3:
        raise RuntimeError('DUAL_HOST_ALL_STAGE_COMPARISON_REQUIRED')
    if not transport['passed'] or not pilot.get('passed') or pilot['formal_gate_sha256']!=gate_hash:
        raise RuntimeError('CURRENT_VERIFIED_PARALLEL_PILOT_REQUIRED')
    for f in transport['files']:
        if sha(Path(transport['import_root'])/f['path'])!=f['sha256']:
            raise RuntimeError('IMPORTED_PILOT_ARTIFACT_CHANGED '+f['path'])
    passing=[r for r in pilot['rounds'] if r.get('passed')]
    if not passing or any(r['workers'] not in (4,8,12) for r in passing):
        raise RuntimeError('BOUNDED_REAL_TARGET_PILOT_REQUIRED')
    best=max(r['calls_per_second'] for r in passing)
    # Same workload, synchronized after initialization. Avoid a larger cohort
    # for a less-than5% measured benefit; finite shared-load limits are retained.
    chosen=min((r for r in passing if r['calls_per_second']>=.95*best),key=lambda r:r['workers'])
    local_peak=stages['windows_peak'].get('peak_commit',stages['windows_peak'].get('peak_bytes',0))
    remote_peak=max([r['native_peak_bytes'] for r in passing]+[
        c['native_peak'].get('peak_bytes',0) for c in stages['cases']])
    if min(local_peak,remote_peak)<=0:raise RuntimeError('MEASURED_NATIVE_PEAK_REQUIRED')
    # Align upwards to0.5GB with an additional5% measurement allowance. The
    # registry applies the prescribed1.35margin exactly once on these peaks.
    peak=lambda n:max(6500000000,math.ceil(n*1.05/500000000)*500000000)
    jobs=read('config/jobs.json')
    if len(jobs)!=44 or len({j['id'] for j in jobs})!=44:raise RuntimeError('EXACT44UNIQUE_JOBS_REQUIRED')
    local_ids={'TIM_F23_C0_V0_s0','TIM_F24_C0_V0_s0'}
    for jid in local_ids:
        folder=ROOT/'outputs/jobs'/jid
        if (folder/'optimizer.pkl').exists():raise RuntimeError('EXISTING_FIT_OWNER_REVIEW_REQUIRED '+jid)
    plan=dict(frozen_after_measurement=True,formal_gate_sha256=gate_hash,
        owners={j['id']:'local' if j['id'] in local_ids else 'workstation' for j in jobs},
        max_active=dict(local=2,workstation=chosen['workers']),
        production_threads=1,peak_bytes=dict(local=peak(local_peak),workstation=peak(remote_peak)),
        measured_native_peak_bytes=dict(local=local_peak,workstation=remote_peak),
        selected_parallel_target_workers=chosen['workers'],
        selection='smallest cohort within5percent of highest measured full-native-target throughput',
        selection_reads_evaluation_scores=False,shared_load_not_isolated=True,
        independent_paths_have_no_training_communication=True,
        admission_still_requires_shared_registry=True,registry_peak_margin=1.35,
        registry_shared_physical_core_cap=40,resource_pause_percent=90,resource_resume_below_percent=85,
        no_hard_training_time_or_call_budget=True,
        insufficient_native_stop_requires_same_point_diagnosis=True,
        audit_concurrency_note='1/2/4thread full audit measured; path concurrency expanded only with measured target ceiling and all-stage peak reservation; actual audit throughput is reported separately',
        local_small_batch_rule='two new time-baseline fits local;42independent bulk paths workstation',
        evidence_sha256={p:sha(ROOT/p) for p in ('evidence/formal_gate.json','evidence/fixed_replays.json',
            'evidence/stage_measurement_consistency.json','evidence/workstation_parallel_target_import.json',pilot_rel)},
        selector_sha256=sha(Path(__file__)))
    target=ROOT/'deployment/path_ownership.json'
    if target.exists() and read('deployment/path_ownership.json')!=plan:
        raise RuntimeError('EXISTING44OWNERSHIP_MUST_NOT_BE_OVERWRITTEN')
    write_json(target,plan)
    print(json.dumps(dict(passed=True,owners=44,max_active=plan['max_active'],peak_bytes=plan['peak_bytes'])))


if __name__=='__main__':main()
