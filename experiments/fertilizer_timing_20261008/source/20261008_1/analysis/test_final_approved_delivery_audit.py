"""Known false-completion controls; fixtures never contain water-quality labels."""
import copy
import json
import tempfile
from pathlib import Path
import audit_final_approved_delivery as audit


def main():
    cases=[]
    def rejects(name,fn):
        try:fn()
        except (RuntimeError,FileNotFoundError,KeyError,ValueError):cases.append(dict(case=name,passed=True))
        else:raise AssertionError('FALSE_COMPLETION_ACCEPTED '+name)
    ids={f'job{i}' for i in range(44)}
    rows=[dict(job=j) for j in sorted(ids)]
    audit.exact_ids(rows,ids)
    rejects('empty_path_coverage',lambda:audit.exact_ids([],ids))
    rejects('only_two_baselines',lambda:audit.exact_ids(rows[:2],ids))
    duplicate=copy.deepcopy(rows);duplicate[-1]=duplicate[0]
    rejects('duplicate_replaces_missing_path',lambda:audit.exact_ids(duplicate,ids))
    from verify_fixed_replay_delivery import unique,path_under
    fixed_ids={f'fixed{i}' for i in range(12)}
    fixed_rows=[dict(job=j) for j in sorted(fixed_ids)]
    unique(fixed_rows,fixed_ids)
    rejects('fixed_replay_empty_coverage',lambda:unique([],fixed_ids))
    rejects('fixed_replay_missing_control',lambda:unique(fixed_rows[:-1],fixed_ids))
    duplicated=copy.deepcopy(fixed_rows);duplicated[-1]=duplicated[0]
    rejects('fixed_replay_duplicate_instead_of_missing',lambda:unique(duplicated,fixed_ids))
    rejects('fixed_replay_path_escape',lambda:path_under(audit.ROOT,'../outside/receipt.json'))
    trial=dict(calls=1,objective=2.,projected_gradient=1e-7,fixed_source_multiplier=1.,local_balance_kg=1e-8)
    audit.trajectory_accounting([trial],1)
    rejects('duplicate_cumulative_call',lambda:audit.trajectory_accounting([trial,trial],1))
    rejects('status_omits_committed_call',lambda:audit.trajectory_accounting([trial],2))
    for name,key,value in [('nonfinite_objective','objective',float('nan')),
                           ('changed_source_mass','fixed_source_multiplier',2.),
                           ('mass_outside_existing_tolerance','local_balance_kg',2e-6)]:
        altered=dict(trial);altered[key]=value
        rejects(name,lambda r=altered:audit.trajectory_accounting([r],1))
    with tempfile.TemporaryDirectory(dir=audit.ROOT/'outputs',prefix='final_audit_controls_') as name:
        root=Path(name)
        for relative in list(audit.RECEIPTS.values())+list(audit.REPORTS):
            path=root/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('synthetic control only',encoding='utf8')
        value=dict(manually_reviewed=True,actual_paths=44,conclusions_reviewed=sorted(audit.CONCLUSIONS),
                   report_sha256={p:audit.sha(root/p) for p in audit.REPORTS},
                   scientific_receipt_sha256={p:audit.sha(root/p) for p in audit.RECEIPTS.values()})
        audit.signoff(root,value)
        missing=copy.deepcopy(value);missing['conclusions_reviewed'].pop()
        rejects('missing_scientific_conclusion',lambda:audit.signoff(root,missing))
        duplicate=copy.deepcopy(value);duplicate['conclusions_reviewed'].append(duplicate['conclusions_reviewed'][0])
        rejects('duplicate_manual_conclusion',lambda:audit.signoff(root,duplicate))
        changed=copy.deepcopy(value);changed['scientific_receipt_sha256'][audit.RECEIPTS['evaluation']]='0'*64
        rejects('changed_evaluation_receipt',lambda:audit.signoff(root,changed))
        (root/audit.REPORTS[0]).write_text('altered report',encoding='utf8')
        rejects('report_changed_after_signoff',lambda:audit.signoff(root,value))
        rejects('relative_escape_binding',lambda:audit.bound_files(root,{'../escape':'0'*64}))
        inventory=audit.inventory(root)
        assert inventory['completion_proven'] is False and inventory['evaluation_labels_read'] is False
        assert all(r['status'] in ('missing','present_not_validated') for r in inventory['rows'])
        cases.append(dict(case='presence_never_certifies_completion',passed=True))
    # Actual incomplete campaign: admission must fail before evaluation access.
    import run_approved_evaluation_delivery as evaluation
    touched=[];original=evaluation.verify_outputs
    evaluation.verify_outputs=lambda *args,**kwargs:touched.append('evaluation_access')
    try:
        try:audit.verify()
        except RuntimeError as error:
            assert str(error)=='FINAL_CORRECTED44DELIVERY_NOT_READY',repr(error)
            cases.append(dict(case='actual_incomplete44_rejected_before_evaluation',passed=True,
                              actual_rejection=str(error)))
        else:raise AssertionError('ACTUAL_INCOMPLETE_CAMPAIGN_ACCEPTED')
        assert not touched
    finally:evaluation.verify_outputs=original
    result=dict(passed=True,cases=cases,source_sha256=audit.sha(Path(audit.__file__)),
                test_sha256=audit.sha(Path(__file__)),actual_final_contents_validated=False,
                evaluation_labels_read=False,physical_calls=0,new_workers=0)
    audit.write_json(audit.ROOT/'evidence/final_approved_contents_audit_controls.json',result)
    print(json.dumps(dict(passed=True,controls=len(cases),actual_final_contents_validated=False)))


if __name__=='__main__':main()
