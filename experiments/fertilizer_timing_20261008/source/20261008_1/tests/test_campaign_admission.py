"""Known-state orchestration barriers, not a44pathscientific execution."""
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import campaign_controller as controller
from d29_platform.runtime import write_json,sha


def main():
    checks={}
    with tempfile.TemporaryDirectory(prefix='campaign_admission_',dir=ROOT/'cache') as temp:
        p=Path(temp);folder=p/'job';folder.mkdir()
        assert controller.stage_for(folder)=='train'
        write_json(folder/'status.json',dict(status='PG_met_full_audit_pending'))
        assert controller.stage_for(folder)=='audit'
        write_json(folder/'independent_audit.json',dict(numerically_sufficient=False))
        assert controller.stage_for(folder)=='review_required'
        checks['failed_audit_blocks_restart_and_freeze']=True
        write_json(folder/'independent_audit.json',dict(numerically_sufficient=True))
        assert controller.stage_for(folder)=='freeze'
        write_json(folder/'prediction_freeze.json',dict(numerically_sufficient=True))
        assert controller.stage_for(folder)=='complete'
        write_json(folder/'prediction_freeze.json',dict(numerically_sufficient=False))
        assert controller.stage_for(folder)=='review_required'
        checks['freeze_does_not_hide_insufficient_status']=True
        (folder/'prediction_freeze.json').unlink();(folder/'independent_audit.json').unlink()
        write_json(folder/'status.json',dict(status='numerical_review_required'))
        assert controller.stage_for(folder)=='same_point_audit'
        checks['insufficient_stop_enters_same_point_audit']=True
        with patch.object(controller,'ROOT',p),patch.object(controller,'validate_admission',return_value=({},'fixture_gate')), \
            patch.object(controller.registry,'acquire',side_effect=AssertionError('resource acquired before complete admission')):
            # No actual resource operation may happen without measured ownership.
            try:controller.main('local' if controller.os.name=='nt' else 'workstation',1)
            except RuntimeError as exc:assert 'MEASURED_OWNERSHIP' in str(exc)
            else:raise AssertionError('missing ownership admitted a controller')
        checks['missing_measurement_prevents_resource_acquisition']=True
        live=dict(pid=999,created='known_creation',job='known_job',stage='train',peak_rss_bytes=123,
            lease=dict(token='fixture_token'))
        lease=dict(pid=999,created='known_creation',project='20261008_1',job='train/known_job',token='fixture_token')
        previous=dict(formal_gate_sha256='fixture_gate',running=[live])
        with patch.object(controller.registry,'identity',return_value='known_creation'), \
            patch.object(controller.registry,'status',return_value=dict(leases={'fixture_token':lease})), \
            patch.object(controller,'registry_root',return_value=p):
            adopted,events=controller.adopt(previous,'fixture_gate')
            assert adopted['known_job']['pid']==999 and adopted['known_job']['process'] is None and not events
            with patch.object(controller.registry,'status',return_value=dict(leases={})): 
                try:controller.adopt(previous,'fixture_gate')
                except RuntimeError as exc:assert 'LEASE_MISMATCH' in str(exc)
                else:raise AssertionError('live child without exact lease adopted')
        checks['identified_live_child_adopted_without_redispatch']=True
        checks['adoption_rejects_missing_exact_lease']=True
        with patch.object(controller,'ROOT',p),patch.object(controller.registry,'identity',return_value=None), \
            patch.object(controller.registry,'status',return_value=dict(leases={})),patch.object(controller,'registry_root',return_value=p):
            _,events=controller.adopt(previous,'fixture_gate')
            assert events[0]['exit_code']==3
            target=p/'outputs/jobs/known_job';target.mkdir(parents=True)
            write_json(target/'status.json',dict(status='PG_met_full_audit_pending'))
            _,events=controller.adopt(previous,'fixture_gate')
            assert events[0]['exit_code']==0 and events[0]['recovered_exit_inferred_from_scientific_receipt']
        checks['dead_child_without_terminal_evidence_requires_review']=True
    receipt=dict(passed=all(checks.values()),scope='known_state_campaign_admission_not_production_fit',cases=checks,
        scientific_jobs_started=0,formal_gate_issued=False,identities={'scripts/campaign_controller.py':sha(ROOT/'scripts/campaign_controller.py')})
    write_json(ROOT/'evidence/campaign_admission_known_states.json',receipt);print(json.dumps(receipt))


if __name__=='__main__':main()
