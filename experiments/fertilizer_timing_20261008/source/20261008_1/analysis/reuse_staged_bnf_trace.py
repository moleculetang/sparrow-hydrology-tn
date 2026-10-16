"""Reuse actual passive BNF executions only after exact44 terminal admission."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def verified_for_final(destination,source,job):
    from approved_epoch_delivery import admitted_folders
    from staged_frozen_sources import resolve
    from run_approved_bnf_delivery import verified_case
    from diagnose_approved_waveforms import accepted_point
    from reuse_staged_internal_trace import compare_bindings
    from d29_platform.runtime import sha
    final=admitted_folders(ROOT)
    if Path(source).resolve()!=final[job['id']].resolve():
        raise RuntimeError('EXACT_FINAL44_BNF_SOURCE_REQUIRED')
    expected=ROOT/'outputs/jobs'/('bnf_trace_approved_staged_v1_'+job['id'])
    if job['calendar']=='K' or Path(destination).resolve()!=expected.resolve():
        raise RuntimeError('EXACT_NONK_STAGED_BNF_EXECUTION_REQUIRED')
    staged=resolve(job['id']);result=verified_case(destination,staged,job,'staged')
    support=json.loads((ROOT/'evidence/expected_station_support.json').read_text())
    _,_,early=accepted_point(staged,job,support)
    _,_,actual=accepted_point(source,job,support)
    compare_bindings(early,actual,sha(staged/'audited_training_prediction.npy'),sha(source/'audited_training_prediction.npy'))
    result.update(execution_mode='staged_BNF_reused_after_final44',
        final_source_folder=str(Path(source).resolve()),final_delivery_sha256=sha(ROOT/'evidence/approved_epoch_delivery.json'),
        execution_worker_sha256=sha(ROOT/'analysis/trace_staged_bnf_history.py'),
        reuse_verifier_sha256=sha(Path(__file__)),additional_forward_calls=0)
    return result
