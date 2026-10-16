"""Compile current entry points and aggregate actual known-error controls.

This readiness receipt is not a full-history or end-to-end campaign certificate.
The scientific sealer checks separate physical and derivative proof families.
"""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from d29_platform.runtime import write_json,sha


def main():
    scripts=['scripts/train_job.py','scripts/audit_solution.py','scripts/freeze_job.py','scripts/evaluate_campaign.py',
        'scripts/campaign_controller.py','scripts/replay_controller.py','scripts/replay_config.py','scripts/measure_stages.py',
        'scripts/measurement_controller.py','scripts/job_lease.py','scripts/controller_lock.py','scripts/guarded_worker.py',
        'scripts/export_timing_allocations.py','scripts/source_ledger.py']
    compiled=[]
    for rel in scripts:
        compile((ROOT/rel).read_text(encoding='utf8'),rel,'exec');compiled.append(rel)
    names=['production_lifecycle_known_truth','campaign_admission_known_states','stage_admission_known_errors',
        'solution_audit_policy_known_errors','evaluation_known_truth','transition_return_import_known_errors',
        'allocation_export_known_truth','plant_activity_ledger_known_truth']
    proofs={};passed=True
    for name in names:
        p=ROOT/'evidence'/f'{name}.json';r=json.loads(p.read_text(encoding='utf8'))
        passed &= bool(r['passed']);proofs[name]=sha(p)
        for rel,digest in r.get('identities',{}).items():passed &= sha(ROOT/rel)==digest
    receipt=dict(passed=bool(passed),scope='compiled_current_production_interfaces_and_known_error_controls_not_actual44path_execution',
        compiled=compiled,known_error_proof_sha256=proofs,identities={rel:sha(ROOT/rel) for rel in scripts},
        scientific_fit_paths_dispatched=0,formal_gate_issued=False)
    write_json(ROOT/'evidence/production_code_review.json',receipt);print(json.dumps(dict(passed=passed,compiled=len(compiled))))
    return 0 if passed else 2


if __name__=='__main__':sys.exit(main())
