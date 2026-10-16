"""Reuse actual early physical executions only after exact44 final admission.

Keeps original staged receipts unchanged; compares them to the actual terminal
imports. A staged replay is never relabelled as an execution by another worker.
"""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def compare_bindings(staged,final,staged_prediction_hash,final_prediction_hash):
    if staged!=final or staged_prediction_hash!=final_prediction_hash:
        raise RuntimeError('TERMINAL_AND_STAGED_FREEZE_BYTES_MUST_MATCH')


def verified_for_final(destination,source,job):
    from approved_epoch_delivery import admitted_folders
    from staged_frozen_sources import resolve
    from run_approved_internal_trace_delivery import verified_case
    from diagnose_approved_waveforms import accepted_point
    from d29_platform.runtime import sha
    # Hard gate before reuse: partial imports cannot self-admit final delivery.
    final_sources=admitted_folders(ROOT)
    if Path(source).resolve()!=final_sources[job['id']].resolve():
        raise RuntimeError('EXACT_FINAL44_SOURCE_REQUIRED_FOR_REUSE')
    expected=ROOT/'outputs/jobs'/('internal_trace_approved_staged_v1_'+job['id'])
    if job['calendar']=='K' or Path(destination).resolve()!=expected.resolve():
        raise RuntimeError('EXACT_NONK_STAGED_EXECUTION_REQUIRED')
    staged=resolve(job['id'])
    result=verified_case(destination,staged,job,'staged')
    support=json.loads((ROOT/'evidence/expected_station_support.json').read_text())
    _,_,early=accepted_point(staged,job,support)
    _,_,actual=accepted_point(source,job,support)
    compare_bindings(early,actual,sha(staged/'audited_training_prediction.npy'),sha(source/'audited_training_prediction.npy'))
    result.update(execution_mode='staged_reused_after_final44',
        final_source_folder=str(Path(source).resolve()),
        final_delivery_sha256=sha(ROOT/'evidence/approved_epoch_delivery.json'),
        execution_worker_sha256=sha(ROOT/'analysis/trace_staged_internal_fluxes.py'),
        reuse_verifier_sha256=sha(Path(__file__)),additional_forward_calls=0)
    return result
