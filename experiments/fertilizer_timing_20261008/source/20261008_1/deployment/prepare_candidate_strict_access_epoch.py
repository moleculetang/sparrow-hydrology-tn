"""Post-diagnostic, versioned train/audit access correction; no live edits."""
import ast
import json
import shutil
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT)]
from host_resources import registry,registry_root
from d29_platform.runtime import sha,write_json

def main():
    c=ROOT.with_name('20261008_1_maize_scope_v2')
    status=json.loads((ROOT/'outputs/maize_scope_revision_candidate/history_status.json').read_text(encoding='utf8'))
    if registry.identity(status['owner']['pid'])==status['owner']['created'] or status.get('exit_code')!=0:
        raise RuntimeError('ACTUAL_CANDIDATE_HISTORY_NATIVE_EXIT_REQUIRED')
    if any(v['job']=='scope_revision_probe/maize_scope_v2_history' for v in registry.status(root=registry_root())['leases'].values()):
        raise RuntimeError('CANDIDATE_HISTORY_LEASE_NOT_RELEASED')
    history=c/'outputs/maize_scope_history/complete.json'
    q=json.loads(history.read_text(encoding='utf8'))
    if not q['passed']:raise RuntimeError('ACTUAL_EXPANDED_HISTORY_PASS_REQUIRED')
    stage=c/'evidence/constructor_access_epoch_v3';stage.mkdir(parents=True,exist_ok=True)
    if (stage/'source_revision.json').exists():raise RuntimeError('EXISTING_ACCESS_REVISION_REVIEW_REQUIRED')
    changes=[]
    for rel,needle,replacement in [
        ('scripts/train_job.py',"    a=TimingTraining(job);scale=a.scale.copy();cp=folder/'optimizer.pkl';identity=backend()",
         "    from training_only_constructor import build_training\n    a=build_training(job);scale=a.scale.copy();cp=folder/'optimizer.pkl';identity=backend()"),
        ('scripts/audit_solution.py',"    a=TimingTraining(job);folder=a.folder;pointfile=folder/'best.npy'",
         "    from training_only_constructor import build_training\n    a=build_training(job);folder=a.folder;pointfile=folder/'best.npy'")]:
        source=c/rel;raw=source.read_bytes();text=raw.decode('utf8')
        original_gate=json.loads((ROOT/'evidence/formal_gate.json').read_text(encoding='utf8'))
        if sha(source)!=original_gate['identities'][rel] or text.count(needle)!=1:
            raise RuntimeError('EXPECTED_ORIGINAL_STAGE_SOURCE_REQUIRED '+rel)
        archived=stage/(rel.replace('/','_')+'.original');archived.write_bytes(raw)
        revised=text.replace(needle,replacement)
        if revised.replace(replacement,needle)!=text:raise RuntimeError('UNREGISTERED_ACCESS_CHANGE')
        compile(revised,str(source),'exec');source.write_bytes(revised.encode('utf8'))
        changes.append(dict(path=rel,old_sha256=sha(archived),new_sha256=sha(source),
                            only_change='constructor now uses qualified legal-row predicate'))
    access=dict(scope='clarification_of_completed_pre_access_epoch_diagnostic',
        history_sha256=sha(history),original_claim_field='evaluation_labels_read',original_claim_value=q['evaluation_labels_read'],
        raw_evaluation_rows_returned_to_constructor=True,
        evaluation_scores_calculated=False,evaluation_values_entered_objective=False,
        reason='UTraining reads full Parquet frames then slices legal years; diagnostic used that unfiltered constructor',
        qualified_training_predicate='scripts/freeze_job.py::training_only_adapter',
        predicate_proof_path=str(ROOT/'outputs/training_predicate_equivalence_v2/complete.json'),
        original_record_retained_without_rewriting=True,
        strict_current_oracle_equivalence_pending=True,production_gate_passed=False)
    write_json(stage/'completed_diagnostic_access_clarification.json',access)
    write_json(stage/'source_revision.json',dict(passed=True,changes=changes,
        helper_sha256=sha(c/'scripts/training_only_constructor.py'),
        history_sha256=sha(history),scientific_numerical_methods_changed=False,
        current_oracle_equivalence_passed=False,production_gate_passed=False))
    print(json.dumps(dict(source_change_prepared=True,strict_oracle_equivalence_pending=True,
                         old_diagnostic_access_claim_corrected_by_sidecar=True)))

if __name__=='__main__':main()
