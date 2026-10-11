"""Seal the corrected K contract only from actual compatible evidence.

Original receipts are retained with their original hashes and access scope.
Missing Linux evidence issues a pending receipt, never a formal gate.
"""
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CANDIDATE=ROOT.with_name('20261008_1_maize_scope_v2')
sys.path[:0]=[str(ROOT)]
from d29_platform.runtime import sha,write_json


def main():
    original_path=ROOT/'evidence/formal_gate.json'
    original=json.loads(original_path.read_text(encoding='utf8'))
    folder=ROOT/'outputs/maize_scope_revision_candidate/transport_v4'
    manifest_path=folder/'manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf8'))
    pending=[];errors=[];proofs={}
    if not original.get('passed') or not original.get('sealed') or sha(original_path)!=manifest['original_gate_sha256']:
        errors.append('ORIGINAL_SEALED_EPOCH_CHANGED')
    for rel,digest in original['identities'].items():
        if sha(ROOT/rel)!=digest:errors.append('ORIGINAL_DEPENDENCY_CHANGED '+rel)
    for rel,digest in original['proof_sha256'].items():
        if sha(ROOT/rel)!=digest:errors.append('ORIGINAL_PROOF_CHANGED '+rel)
        proofs[str((ROOT/rel).resolve())]=digest
    for row in manifest['files']:
        if sha(CANDIDATE/row['path'])!=row['sha256']:errors.append('CANDIDATE_DEPENDENCY_CHANGED '+row['path'])
    def actual(path):
        path=Path(path)
        if not path.exists():pending.append(str(path));return None
        record=json.loads(path.read_text(encoding='utf8'));proofs[str(path.resolve())]=sha(path)
        if not record.get('passed'):errors.append('ACTUAL_PROOF_NOT_PASSED '+str(path))
        return record
    history=actual(CANDIDATE/'outputs/maize_scope_history/complete.json')
    profile=actual(CANDIDATE/'outputs/maize_scope_profiles/complete.json')
    strict=actual(CANDIDATE/'outputs/scope_strict_point/complete.json')
    staged=actual(folder/'actual_remote_staging.json')
    dual=actual(folder/'dualhost_comparison.json')
    revision=actual(CANDIDATE/'evidence/constructor_access_epoch_v3/source_revision.json')
    config_path=CANDIDATE/'config/timing_evidence.json'
    config=json.loads(config_path.read_text(encoding='utf8'))
    old=json.loads((ROOT/'config/timing_evidence.json').read_text(encoding='utf8'))
    maize=next(s for s in config['fertilizer_profiles'] if s['crop']=='Maize' and s['source']=='Nfer')
    old_maize=next(s for s in old['fertilizer_profiles'] if s['crop']=='Maize' and s['source']=='Nfer')
    reconstructed=json.loads(json.dumps(config))
    item=next(s for s in reconstructed['fertilizer_profiles'] if s['crop']=='Maize' and s['source']=='Nfer')
    item['basal_delta_bounds']=old_maize['basal_delta_bounds'];item['bounds_basis']=old_maize['bounds_basis']
    if reconstructed!=old or maize['basal_delta_bounds']!=[-.1,0.]:errors.append('UNREGISTERED_SCIENTIFIC_CONFIG_CHANGE')
    change_paths={row['path'] for row in manifest['files'] if row.get('original_sha256') and row['original_sha256']!=row['sha256']}
    if change_paths!={'config/timing_evidence.json','scripts/train_job.py','scripts/audit_solution.py'}:
        errors.append('UNREGISTERED_CHANGED_ORIGINAL_SOURCE')
    # Verify the constructor replacements against preserved exact original bytes.
    if revision:
        for row in revision['changes']:
            path=row['path'];preserved=CANDIDATE/'evidence/constructor_access_epoch_v3'/(path.replace('/','_')+'.original')
            if sha(preserved)!=row['old_sha256'] or original['identities'][path]!=row['old_sha256'] or sha(CANDIDATE/path)!=row['new_sha256']:
                errors.append('CONSTRUCTOR_SOURCE_REVISION_CHANGED '+path)
            original_source=preserved.read_text(encoding='utf8')
            source=(CANDIDATE/path).read_text(encoding='utf8')
            prefix='    from training_only_constructor import build_training\n'
            if path=='scripts/train_job.py':
                replacement=prefix+"    a=build_training(job);scale=a.scale.copy();cp=folder/'optimizer.pkl';identity=backend()"
                expected="    a=TimingTraining(job);scale=a.scale.copy();cp=folder/'optimizer.pkl';identity=backend()"
            else:
                replacement=prefix+"    a=build_training(job);folder=a.folder;pointfile=folder/'best.npy'"
                expected="    a=TimingTraining(job);folder=a.folder;pointfile=folder/'best.npy'"
            if source.count(replacement)!=1 or source.replace(replacement,expected)!=original_source:
                errors.append('UNREGISTERED_CONSTRUCTOR_ALGORITHM_CHANGE '+path)
        if sha(CANDIDATE/'scripts/training_only_constructor.py')!=revision['helper_sha256']:
            errors.append('STRICT_ACCESS_HELPER_CHANGED')
    expanded_derivatives=False;ledger=False;compatibility=False
    if history:
        if history['config_sha256']!=sha(config_path) or history['source_multiplier']!=1.:
            errors.append('EXPANDED_HISTORY_CONFIG_OR_MASS_CHANGED')
        cases=history['cases']
        expanded_derivatives=(len(cases)==3 and sorted(c['delta'] for c in cases)==[-.1,-.075,0.]
            and all(c['passed'] and c['derivative']['passed'] and c['derivative']['passing_adjacent_pairs']
                and c['derivative']['branch_consistent_pairs'] for c in cases))
        coverage=history['reused']['coordinate_coverage']
        compatibility=(len(history['regression'])==2 and all(c['passed'] and all(x['bitwise_equal'] for x in c['arrays'].values()) for c in history['regression'])
            and sorted(c['count'] for c in coverage)==[22,24] and all(c['passed'] for c in coverage)
            and history['reused']['initial_family_count']==5)
        reuse_path=Path(history['reused']['proof_path'])
        if sha(reuse_path)!=history['reused']['proof_sha256']:errors.append('REUSED_ZERO_POINT_PROOF_CHANGED')
        physical=next(c['physical'] for c in cases if c['delta']==-.1)
        ledger=(physical['passed'] and physical['history']==[1961,2024] and physical['reaches']==230
            and sorted(r['year'] for r in physical['year_ledger'])==list(range(1961,2025))
            and all(v<=1e-6 for v in physical['maxima'].values())
            and max(physical['network_relative'])<=1e-10 and physical['station_source_relative']<=1e-10)
    strict_equivalent=bool(strict and strict['config_sha256']==sha(config_path)
        and strict['objective_bitwise_equal_to_pre_access_epoch'] and strict['gradient_bitwise_equal_to_pre_access_epoch']
        and len(strict['constructor_reads'])==2 and all(r['evaluation_year_rows_returned']==0 for r in strict['constructor_reads']))
    if strict:
        for name,digest in strict['array_sha256'].items():
            if sha(CANDIDATE/'outputs/scope_strict_point'/(name+'.npy'))!=digest:errors.append('STRICT_REFERENCE_ARRAY_CHANGED '+name)
    dual_ok=bool(dual and dual['identity_passed'] and dual['no_evaluation_rows_returned']
        and dual['objective_passed'] and dual['remote_fresh_canonical_restore_passed'] and all(c['passed'] for c in dual['arrays'].values()))
    if dual:
        actual_path=folder/'linux_return/outputs/scope_remote_point/complete.json'
        if sha(actual_path)!=dual['actual_remote_receipt_sha256']:errors.append('ACTUAL_LINUX_ORACLE_RECEIPT_CHANGED')
        remote=json.loads(actual_path.read_text(encoding='utf8'))
        for name,digest in remote['array_sha256'].items():
            if sha(actual_path.parent/(name+'.npy'))!=digest:errors.append('ACTUAL_LINUX_ARRAY_CHANGED '+name)
    requirements=dict(original['requirements'])
    requirements.update(full_history_derivatives=bool(original['requirements']['full_history_derivatives'] and compatibility and expanded_derivatives),
        source_network_ledger=bool(original['requirements']['source_network_ledger'] and ledger and profile and profile['passed']),
        independent_objective=bool(strict_equivalent and history and all(c['independent_objective']['error']<=1e-8*(1+abs(c['independent_objective']['value'])) for c in history['cases']) and dual_ok),
        source_multiplier_one=bool(history and history['source_multiplier']==1. and strict and strict['source_multiplier']==1.),
        training_prediction_isolation=bool(original['requirements']['training_prediction_isolation'] and strict_equivalent),
        checkpoint_restore=bool(original['requirements']['checkpoint_restore'] and dual_ok),
        dual_host_consistency=bool(dual_ok and staged and staged['h1_arrays_verified']==26),
        immutable_input_identity=bool(not errors))
    passed=not pending and not errors and all(requirements.values())
    identities={row['path']:row['sha256'] for row in manifest['files']
        if row['path'].split('/')[0] not in ('outputs','evidence')}
    result=dict(passed=passed,sealed=passed,
        scope='approved_maize_range_K_only_with_exact_kernel_continuity_and_actual_strict_dualhost',
        requirements=requirements,pending=pending,errors=errors,identities=identities,proof_sha256=proofs,
        original_gate_sha256=sha(original_path),candidate_manifest_sha256=sha(manifest_path),
        source_multiplier=1.,K_coordinate_count=24,C0_coordinate_count=22,
        derivative_reuse_scope='same_zero_management_point_only; maize expanded points independently checked',
        access_scope='new_K_train_audit_filtered_constructor; original_nonK_access_facts_not_recertified',
        scientific_fits_dispatched=0,all_station_performance_claimed=False,
        sealer_sha256=sha(Path(__file__)))
    target=CANDIDATE/'evidence/formal_gate.json' if passed else CANDIDATE/'evidence/scope_gate_pending.json'
    if passed and target.exists():raise RuntimeError('PRESERVE_EXISTING_FORMAL_SEAL_NO_OVERWRITE')
    write_json(target,result)
    print(json.dumps(dict(passed=passed,pending=pending,errors=errors,requirements=requirements,sealed=passed)))
    return 0 if passed else 2


if __name__=='__main__':sys.exit(main())
