"""Final evidence audit for the unchanged 44-path contract.

Missing deliveries remain missing. Inventory mode never reads evaluation labels,
starts workers, creates leases or certifies completion. Verification reuses the
independent artifact readers, rather than accepting their old success flags.
"""
import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts'), str(ROOT/'analysis'), str(ROOT/'deployment')]
from d29_platform.runtime import sha, write_json

RECEIPTS = {
    'scientific_freeze': 'evidence/approved_epoch_delivery.json',
    'input_allocations': 'evidence/approved_allocation_delivery.json',
    'evaluation': 'outputs/approved_evaluation_delivery/complete.json',
    'internal_network_replay': 'outputs/approved_internal_trace_final/complete.json',
    'station_waveforms': 'outputs/approved_waveforms_final/complete.json',
    'plant_output_debt': 'outputs/approved_plant_debt/complete.json',
    'BNF_growth_inventory': 'outputs/approved_bnf_final/complete.json',
    'fixed_parameter_replays': 'evidence/approved_fixed_replay_readback.json',
}
REPORTS = ('reports/专家施肥时序与TN全域验证报告.md', 'reports/实际方法与偏离.md')
CONCLUSIONS = {'path_definitions', 'month_between', 'month_within', 'plant_versions',
               'time_holdout', 'space_and_joint_holdout', 'plant_output_cost',
               'numerical_limits', 'input_and_hydrology_limits', 'conditional_calendar_identity',
               'BNF_outside_growth_and_inventory'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def exact_ids(rows, ids, key='job'):
    found = [r[key] for r in rows]
    if len(found) != len(ids) or len(set(found)) != len(ids) or set(found) != set(ids):
        raise RuntimeError('EXACT_UNIQUE_ALL44_IDS_REQUIRED')


def bound_files(directory, files):
    directory = Path(directory).resolve()
    if not isinstance(files, dict) or not files:
        raise RuntimeError('NONEMPTY_BYTE_BINDINGS_REQUIRED')
    for relative, digest in files.items():
        path = directory/relative
        if (Path(relative).is_absolute() or '..' in Path(relative).parts or path.is_symlink()
                or not path.resolve().is_relative_to(directory) or not path.is_file() or sha(path) != digest):
            raise RuntimeError('BOUND_OUTPUT_BYTES_CHANGED '+relative)


def trajectory_accounting(rows, final_calls):
    calls=[r['calls'] for r in rows]
    if (not rows or any(type(c) is not int or c<1 for c in calls)
            or len(set(calls))!=len(calls) or calls!=sorted(calls) or calls[-1]!=final_calls
            or any(not math.isfinite(r['objective']) or not math.isfinite(r['projected_gradient'])
            or r['projected_gradient']<0 or r.get('fixed_source_multiplier')!=1.
            or not math.isfinite(r['local_balance_kg']) or not 0<=r['local_balance_kg']<=1e-6
            for r in rows)):
        raise RuntimeError('COMPLETE_FINITE_CUMULATIVE_TRAJECTORY_REQUIRED')
    return dict(committed_calls=len(calls),final_call=calls[-1])


def signoff(root, value):
    if (value.get('manually_reviewed') is not True or value.get('actual_paths') != 44
            or set(value.get('conclusions_reviewed', [])) != CONCLUSIONS
            or len(value.get('conclusions_reviewed', [])) != len(CONCLUSIONS)
            or set(value.get('report_sha256', {})) != set(REPORTS)
            or set(value.get('scientific_receipt_sha256', {})) != set(RECEIPTS.values())):
        raise RuntimeError('ACTUAL_COMPLETE_MANUAL_EXPERT_REVIEW_REQUIRED')
    bound_files(root, value['report_sha256'])
    bound_files(root, value['scientific_receipt_sha256'])
    return dict(manually_reviewed=True, conclusions=len(CONCLUSIONS), reports=list(REPORTS))


def inventory(root=ROOT):
    """Only presence and bytes: present is explicitly NOT validated."""
    paths = list(RECEIPTS.values())+list(REPORTS)+['evidence/final_expert_signoff.json']
    rows = [dict(file=p, status='present_not_validated' if (root/p).is_file() else 'missing',
                 sha256=sha(root/p) if (root/p).is_file() else None) for p in paths]
    return dict(mode='inventory', completion_proven=False, actual_final_validation_executed=False,
                evaluation_labels_read=False, new_workers=0, rows=rows)


def verify(root=ROOT):
    from approved_epoch_delivery import admitted_folders
    from assemble_approved_delivery import validate_frozen
    from diagnose_approved_waveforms import approved_gate
    jobs = read(root/'config/jobs.json'); ids = {j['id'] for j in jobs}
    if len(jobs) != 44 or len(ids) != 44:
        raise RuntimeError('REGISTERED44_REQUIRED')
    folders = admitted_folders(root)  # Must pass before evaluation file access.
    support = read(root/'evidence/expected_station_support.json')
    checks = []
    for job in jobs:
        folder=folders[job['id']]
        validate_frozen(folder, job, approved_gate(job), support)
        status=read(folder/'status.json')
        rows=[]
        for line in (folder/'trajectory.jsonl').read_bytes().splitlines(keepends=True):
            if not line.endswith(b'\n'):
                raise RuntimeError('FINAL_TRAJECTORY_UNCOMMITTED_TAIL')
            rows.append(json.loads(line))
        trajectory_accounting(rows,status['calls'])
    checks.append(dict(requirement='full44_audits_freezes_and_call_trajectories', passed=True, paths=44))

    from verify_fixed_replay_delivery import verify as verify_fixed
    fixed=read(root/RECEIPTS['fixed_parameter_replays'])
    if verify_fixed(root)!=fixed:
        raise RuntimeError('ACTUAL_ALL12_FIXED_REPLAY_READBACK_REQUIRED')
    checks.append(dict(requirement='all12_same_parameter_replays_and_layered_contrasts',passed=True,
        paths=12,internal_years=768,certified_station_count=116,
        fixed_physical_shapes_are_not_heldout_NSE=True))

    from assemble_approved_allocation_delivery import assemble
    allocation=read(root/RECEIPTS['input_allocations'])
    # Re-read the actual transport and all 128 artifacts/path, not a cached flag.
    returns=list((root/'outputs/approved_allocation_terminal_return').rglob('complete.json'))
    matches=[p for p in returns if sha(p)==allocation.get('actual_allocation_return_sha256')]
    if len(matches)!=1 or assemble(root,matches[0])!=allocation:
        raise RuntimeError('INDEPENDENT_FULL44_ALLOCATION_REASSEMBLY_REQUIRED')
    checks.append(dict(requirement='all44_64year_21crop_input_allocations', passed=True, artifacts=44*128))

    from run_approved_evaluation_delivery import verify_outputs
    evaluation=read(root/RECEIPTS['evaluation']); verified=verify_outputs(root)
    if evaluation.get('passed') is not True or any(evaluation.get(k)!=v for k,v in verified.items()):
        raise RuntimeError('ACTUAL_EVALUATION_BYTES_AND1000_DRAWS_REQUIRED')
    checks.append(dict(requirement='all_station_metrics_and_both_bootstrap_blocks', passed=True,
                       bootstrap_tables=len(verified['bootstrap'])))

    from run_approved_internal_trace_delivery import verified_case as trace_case
    from run_approved_waveform_delivery import verified_case as waveform_case
    from trace_approved_internal_fluxes import science_root
    for name in ('internal_network_replay','station_waveforms'):
        result=read(root/RECEIPTS[name]); exact_ids(result.get('cases',[]),ids)
        if result.get('passed') is not True or result.get('mode')!='final' or result.get('accounted_paths')!=44:
            raise RuntimeError('ACTUAL_FINAL_DIAGNOSTICS_REQUIRED '+name)
        for job in jobs:
            saved=next(r for r in result['cases'] if r['job']==job['id'])
            if name=='internal_network_replay':
                if saved.get('execution_mode')=='staged_reused_after_final44':
                    from reuse_staged_internal_trace import verified_for_final
                    destination=root/'outputs/jobs'/('internal_trace_approved_staged_v1_'+job['id'])
                    actual=verified_for_final(destination,folders[job['id']],job)
                else:
                    destination=science_root(job)/'outputs/jobs'/('internal_trace_approved_final_v1_'+job['id'])
                    actual=trace_case(destination,folders[job['id']],job,'final')
            else:
                destination=root/'outputs/approved_waveforms_final'/job['id']
                actual=waveform_case(destination,folders[job['id']],job['id'],approved_gate(job))
            if actual!=saved:raise RuntimeError('DIAGNOSTIC_REPLAY_RECEIPT_CHANGED '+job['id'])
        checks.append(dict(requirement=name,passed=True,paths=44))

    from run_approved_bnf_delivery import verified_case as bnf_case, require_controls
    from bnf_endpoint_readback import verify as bnf_endpoint
    bnf_controls=read(root/'evidence/approved_bnf_delivery_controls.json')
    control_files=('analysis/run_approved_bnf_delivery.py','analysis/trace_approved_bnf_history.py',
                   'analysis/bnf_calendar_partition.py','analysis/bnf_endpoint_readback.py',
                   'analysis/test_approved_bnf_delivery.py')
    if (bnf_controls.get('passed') is not True or bnf_controls.get('actual_cases')!=1
            or bnf_controls.get('evaluation_labels_read') is not False
            or bnf_controls.get('files')!={p:sha(root/p) for p in control_files}):
        raise RuntimeError('CURRENT_BNF_DELIVERY_AND_ACTUAL_PREFLIGHT_CONTROLS_REQUIRED')
    endpoint_controls=read(root/'outputs/bnf_partition_software_controls/endpoint_controls.json')
    if (any(endpoint_controls.get(k) is not True for k in ('passed','known_compensated_equivalence',
            'changed_BNF_inventory_rejected','changed_compensation_rejected','incomplete_and_nonfinite_rejected'))
            or endpoint_controls.get('implementation_sha256')!={p:sha(root/p) for p in
                ('analysis/bnf_endpoint_readback.py','analysis/test_bnf_endpoint_readback.py')}):
        raise RuntimeError('CURRENT_BNF_ENDPOINT_CONTROLS_REQUIRED')
    bnf=read(root/RECEIPTS['BNF_growth_inventory'])
    exact_ids(bnf.get('cases',[]),ids)
    if (bnf.get('passed') is not True or bnf.get('mode')!='final'
            or bnf.get('accounted_paths')!=44 or bnf.get('actual_K_cases')!=14
            or bnf.get('full44_BNF_completed') is not True
            or bnf.get('evaluation_labels_read') is not False or bnf.get('new_training_calls')!=0
            or bnf.get('controls_sha256')!=require_controls()
            or bnf.get('launcher_sha256')!=sha(root/'analysis/run_approved_bnf_delivery.py')
            or bnf.get('worker_sha256')!=sha(root/'analysis/trace_approved_bnf_history.py')
            or bnf.get('partition_sha256')!=sha(root/'analysis/bnf_calendar_partition.py')):
        raise RuntimeError('ACTUAL44_BNF_GROWTH_INVENTORY_DIAGNOSTICS_REQUIRED')
    endpoint_checks=[]
    for job in jobs:
        saved=next(r for r in bnf['cases'] if r['job']==job['id'])
        if saved.get('execution_mode')=='staged_BNF_reused_after_final44':
            from reuse_staged_bnf_trace import verified_for_final
            destination=root/'outputs/jobs'/('bnf_trace_approved_staged_v1_'+job['id'])
            actual=verified_for_final(destination,folders[job['id']],job)
        else:
            destination=science_root(job)/'outputs/jobs'/('bnf_trace_approved_final_v1_'+job['id'])
            actual=bnf_case(destination,folders[job['id']],job,'final')
        if actual!=saved:
            raise RuntimeError('BNF_DIAGNOSTIC_RECEIPT_CHANGED '+job['id'])
        endpoint_checks.append(dict(job=job['id'],**bnf_endpoint(destination,folders[job['id']])))
    checks.append(dict(requirement='BNF_outside_growth_and_inventory',passed=True,paths=44,
        BNF_timing_changed=False,inventory_is_model_provenance_not_causal_effect=True,
        original_BNF_endpoint_independent_readbacks=endpoint_checks))

    debt=read(root/RECEIPTS['plant_output_debt'])
    if (debt.get('passed') is not True or debt.get('accounted_paths')!=44
            or debt.get('threshold_percentage_points')!=1 or debt.get('selected_by_training_objective') is not True
            or debt.get('approved_delivery_sha256')!=sha(root/RECEIPTS['scientific_freeze'])):
        raise RuntimeError('ACTUAL44_PLANT_OUTPUT_GUARD_REQUIRED')
    exact_ids([dict(job=j) for j in debt.get('dependencies',{})],ids)
    for jid,bindings in debt['dependencies'].items():bound_files(folders[jid],bindings)
    bound_files(root/'outputs/approved_plant_debt',debt['files'])
    guards=read(root/'outputs/approved_plant_debt/upgrade_guards.json')
    if not guards or len(guards)!=debt.get('comparisons'):
        raise RuntimeError('NONEMPTY_ALL_SELECTED_PLANT_COMPARISONS_REQUIRED')
    checks.append(dict(requirement='plant_output_cost_and_upgrade_guards',passed=True,
                       blocked_recommendations=sum(g['upgrade_recommendation_blocked'] for g in guards)))
    checks.append(dict(requirement='manual_expert_reports',passed=True,
                       **signoff(root,read(root/'evidence/final_expert_signoff.json'))))
    return dict(mode='verify', passed=True, completion_proven=False,
                deliverable_contents_verified=True, resource_closeout_still_required=True,
                expert_review_does_not_imply_TN_improvement=True, checks=checks,
                scientific_receipt_sha256={p:sha(root/p) for p in RECEIPTS.values()},
                new_workers=0, new_training_calls=0)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--verify',action='store_true');args=parser.parse_args()
    result=verify() if args.verify else inventory()
    target=ROOT/'outputs'/('final_approved_contents_audit.json' if args.verify else 'final_approved_delivery_inventory.json')
    write_json(target,result)
    print(json.dumps(dict(output=str(target),mode=result['mode'],completion_proven=False)))


if __name__=='__main__':main()
