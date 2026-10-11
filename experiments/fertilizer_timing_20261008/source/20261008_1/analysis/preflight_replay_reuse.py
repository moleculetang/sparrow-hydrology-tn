"""Reuse the actual F24/C0/V0 preflight without relabelling its execution.

Final reuse always begins with independently admitted all44 frozen sources.
Original preflight receipts stay byte-identical; full annual artifact readers
are run again before the returned result can enter final delivery.
"""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
KINDS={'internal':('approved_internal_trace_preflight','internal_trace_approved_preflight_v1_',
                  'run_approved_internal_trace_delivery','preflight_reused_after_final44'),
       'bnf':('approved_bnf_preflight','bnf_trace_approved_preflight_v1_',
              'run_approved_bnf_delivery','preflight_BNF_reused_after_final44')}
JOB='TIM_F24_C0_V0_s0'

def actual_preflight(kind,destination,source,job):
    from d29_platform.runtime import sha
    import importlib
    if kind not in KINDS:raise RuntimeError('EXACT_PREFLIGHT_KIND_REQUIRED')
    folder,prefix,reader,_=KINDS[kind]
    if job['id']!=JOB or job['calendar']!='C0' or job['plant_version']!='V0' or job['fold']!='F24':
        raise RuntimeError('ONLY_ORIGINAL_F24_C0_V0_PREFLIGHT_REUSABLE')
    if Path(source).resolve()!=(ROOT/'outputs/jobs'/JOB).resolve() or Path(destination).resolve()!=(ROOT/'outputs/jobs'/(prefix+JOB)).resolve():
        raise RuntimeError('EXACT_ORIGINAL_PREFLIGHT_PATHS_REQUIRED')
    path=ROOT/'outputs'/folder/'complete.json';saved=json.loads(path.read_text())
    if (saved.get('passed') is not True or saved.get('mode')!='preflight' or saved.get('accounted_paths')!=1
        or len(saved.get('cases',[]))!=1 or saved.get('evaluation_labels_read') is not False
        or saved.get('new_training_calls')!=0
        or saved.get('full44_internal_replay_completed',False) is not False
        or saved.get('full44_BNF_completed',False) is not False):
        raise RuntimeError('ACTUAL_ONE_CASE_PREFLIGHT_EXECUTION_REQUIRED')
    actual=importlib.import_module(reader).verified_case(Path(destination),Path(source),job,'preflight')
    if actual!=saved['cases'][0]:raise RuntimeError('ORIGINAL_PREFLIGHT_AGGREGATE_BYTES_AND_CASE_REQUIRED')
    return actual,sha(path)

def verified_for_final(kind,destination,source,job):
    from approved_epoch_delivery import admitted_folders
    from d29_platform.runtime import sha
    # No provisional table, local freeze, or preflight may self-admit final44.
    final=admitted_folders(ROOT)
    if Path(source).resolve()!=final[job['id']].resolve():raise RuntimeError('EXACT_FINAL44_SOURCE_REQUIRED_FOR_PREFLIGHT_REUSE')
    actual,aggregate=actual_preflight(kind,destination,source,job)
    result=dict(actual)
    result.update(execution_mode=KINDS[kind][3],original_preflight_aggregate_sha256=aggregate,
        final_source_folder=str(Path(source).resolve()),
        final_delivery_sha256=sha(ROOT/'evidence/approved_epoch_delivery.json'),
        reuse_verifier_sha256=sha(Path(__file__)),additional_forward_calls=0)
    return result

def require_controls(kind,job):
    from d29_platform.runtime import sha
    path=ROOT/'evidence/preflight_replay_reuse_controls.json';proof=json.loads(path.read_text())
    expected=('analysis/preflight_replay_reuse.py','analysis/run_approved_internal_trace_delivery.py',
              'analysis/run_approved_bnf_delivery.py','analysis/audit_final_approved_delivery.py',
              'analysis/test_preflight_replay_reuse.py')
    if (proof.get('passed') is not True or proof.get('exact44_admission_preserved') is not True
        or proof.get('evaluation_labels_read') is not False or proof.get('additional_forward_calls')!=0
        or proof.get('files')!={p:sha(ROOT/p) for p in expected}
        or job['id']!=JOB or kind not in proof.get('actual_cases',{})):
        raise RuntimeError('CURRENT_ACTUAL_PREFLIGHT_REUSE_CONTROLS_REQUIRED')
