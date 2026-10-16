"""Stage/transport controls only, never dispatch a scientific or data worker."""
import copy
import sys
import tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'deployment')]
from frozen_allocation_controller import stage,recovered_exit,transport_admitted
from d29_platform.runtime import sha,write_json


def fixture(root):
    jid='synthetic_delivery';f=root/'outputs/jobs'/jid;f.mkdir(parents=True)
    (f/'frozen_parameters.npy').write_bytes(b'synthetic immutable point; not loaded')
    verifier=root/'analysis/verify_timing_allocation_artifacts.py';verifier.parent.mkdir();verifier.write_text('# fixture')
    point=sha(f/'frozen_parameters.npy')
    freeze=dict(numerically_sufficient=True,formal_gate_sha256='gate',parameter_sha256=point)
    write_json(f/'prediction_freeze.json',freeze)
    return jid,f,freeze


def main():
    controls=[]
    assert recovered_exit('allocation_export','allocation_review')==0
    assert recovered_exit('allocation_review','complete')==0
    for first,last in [('allocation_review','allocation_review'),('allocation_export','allocation_export'),
                       ('allocation_review','await_scientific_freeze')]:
        assert recovered_exit(first,last)==2;controls.append(dict(control=first+'_'+last,passed=True))
    with tempfile.TemporaryDirectory(dir=ROOT/'outputs/test_fixtures') as tmp:
        root=Path(tmp);jid,f,freeze=fixture(root)
        assert stage(root,'missing','gate')=='await_scientific_freeze'
        assert stage(root,jid,'gate')=='allocation_export'
        allocation=dict(passed=True,source_multiplier=1.,formal_gate_sha256='gate',parameter_sha256=freeze['parameter_sha256'])
        write_json(f/'input_allocation/complete.json',allocation)
        assert stage(root,jid,'gate')=='allocation_review'
        review=dict(passed=True,years=64,reaches=230,crops=21,
                    allocation_receipt_sha256=sha(f/'input_allocation/complete.json'),
                    parameter_sha256=freeze['parameter_sha256'],
                    verifier_sha256=sha(root/'analysis/verify_timing_allocation_artifacts.py'))
        write_json(f/'input_allocation/independent_artifact_review.json',review)
        assert stage(root,jid,'gate')=='complete'
        def reject(name,path,value):
            original=path.read_bytes();write_json(path,value)
            try:stage(root,jid,'gate')
            except RuntimeError:controls.append(dict(control=name,passed=True))
            else:raise AssertionError(name)
            path.write_bytes(original)
        reject('insufficient_freeze',f/'prediction_freeze.json',dict(freeze,numerically_sufficient=False))
        reject('changed_gate',f/'prediction_freeze.json',dict(freeze,formal_gate_sha256='wrong'))
        reject('old_multiplier',f/'input_allocation/complete.json',dict(allocation,source_multiplier=2.))
        reject('different_point',f/'input_allocation/complete.json',dict(allocation,parameter_sha256='wrong'))
        reject('partial_year_readback',f/'input_allocation/independent_artifact_review.json',dict(review,years=9))
        reject('stale_verifier',f/'input_allocation/independent_artifact_review.json',dict(review,verifier_sha256='wrong'))
        assert transport_admitted(root,'local','gate') is True
        assert transport_admitted(root,'workstation','gate') is False
        transfer=root/'transfer';transfer.mkdir();a=transfer/'synthetic.tar';a.write_bytes(b'known synthetic archive')
        owner=dict(pid=0,created='synthetic stopped controller')
        write_json(root/'outputs/campaign_workstation/owner.json',owner)
        ack=dict(transport_passed=True,formal_gate_sha256='gate',current_host_snapshot_imported=True,
                 archive_relative='transfer/synthetic.tar',archive_sha256=sha(a),remote_owner=owner)
        write_json(root/'evidence/terminal_snapshot_imported_before_allocations.json',ack)
        assert transport_admitted(root,'workstation','gate') is True
        write_json(root/'outputs/campaign_workstation/owner.json',dict(owner,created='new owner'))
        assert transport_admitted(root,'workstation','gate') is False
        write_json(root/'outputs/campaign_workstation/owner.json',owner)
        controls.append(dict(control='new_controller_requires_new_snapshot_ack',passed=True))
        for name,change in [('unchecked_import',dict(current_host_snapshot_imported=False)),
                            ('archive_outside_transfer',dict(archive_relative='../outside.tar')),
                            ('changed_archive',dict(archive_sha256='0'*64))]:
            write_json(root/'evidence/terminal_snapshot_imported_before_allocations.json',dict(ack,**change))
            try:transport_admitted(root,'workstation','gate')
            except RuntimeError:controls.append(dict(control=name,passed=True))
            else:raise AssertionError(name)
    result=dict(passed=True,negative_controls=controls,scope='synthetic_stage_and_native_snapshot_admission_no_actual_path_export',
                source_sha256=sha(ROOT/'deployment/frozen_allocation_controller.py'),
                scientific_dispatch=False,scientific_checkpoint_change=False)
    write_json(ROOT/'evidence/frozen_allocation_controller_known_errors.json',result)
    print('passed: frozen-delivery stages and '+str(len(controls))+' recovery/admission controls')


if __name__=='__main__':main()
