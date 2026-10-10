"""Synthetic transaction controls; never uses or mutates a real campaign point."""
import copy
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'analysis')]
from rearm_reviewed_campaign import qualified_epoch, rearmed_state
from d29_platform.runtime import sha, write_json


def build_fixture(root):
    jid = 'synthetic_reviewed_point'
    folder = root / 'outputs/jobs' / jid
    archive = folder / 'reentry_epochs/001_control'
    archive.mkdir(parents=True)
    probe = root / 'analysis/probe_same_point_descent.py'
    probe.parent.mkdir(parents=True)
    probe.write_text('# synthetic probe identity only\n')
    (folder / 'best.npy').write_bytes(b'synthetic point, no numpy load')
    (folder / 'optimizer.pkl').write_bytes(b'new canonical control, no pickle load')
    point = sha(folder / 'best.npy')
    contract = dict(names=['x', 'y'], source_multiplier=1., formal_gate_sha256='gate')
    audit = dict(source_multiplier=1., formal_gate_sha256='gate', parameter_sha256=point,
                 full_history=[1961, 2024], reaches=230, objective_passed=True,
                 physical_passed=True, gradient_recovery_passed=True, derivative_passed=True,
                 checks=[dict(coordinate=n, derivative_qualified=True) for n in contract['names']],
                 objective=3., projected_gradient=.02, scaled_projected_gradient=.01,
                 numerically_sufficient=False)
    witness = dict(parameter_sha256=point, formal_gate_sha256='gate', source_multiplier=1.,
                   evaluation_scores_used=False, base_objective=3., direction=[-1., 0.],
                   probes=[dict(step=h, slope=-.02, analytic=-.02, objective=3.-.02*h,
                                feasible=True, branch_consistent=True, branch_signature='same')
                           for h in [.01, .005]])
    diagnostic = folder / 'continuation_diagnostics' / point / 'complete.json'
    write_json(diagnostic, dict(passed=True, new_solver_started=False, witness=witness,
                               source_sha256=sha(probe)))
    record = dict(continuation_prepared=True, continuation_started=False, training_started=False,
                  same_point=True, new_canonical_memory=True, old_workspace_reencoded=False,
                  formal_gate_sha256='gate', parameter_sha256=point,
                  new_checkpoint_sha256=sha(folder/'optimizer.pkl'),
                  diagnostic_sha256=sha(diagnostic), archive='reentry_epochs/001_control')
    history = folder / 'continuation_history.json'
    write_json(history, dict(attempts=[record]))
    write_json(archive/'training_contract.json', contract)
    write_json(archive/'independent_audit.json', audit)
    for name in ['optimizer.pkl', 'status.json', 'best.npy', 'best_gradient.npy']:
        (archive/name).write_bytes(('original control '+name).encode())
    files = {p.name: dict(sha256=sha(p), bytes=p.stat().st_size) for p in archive.iterdir()}
    write_json(archive/'archive_manifest.json', dict(job=jid, gate_sha256='gate',
               parameter_sha256=point, diagnostic_sha256=sha(diagnostic), files=files,
               qualification=dict(qualified_for_reviewed_same_point_reentry=True)))
    write_json(archive/'prepared.json', record)
    write_json(archive/'preparation_journal.json', dict(phase='prepared_not_dispatched', record=record))
    write_json(folder/'status.json', dict(status='safe_checkpoint_paused',
               reviewed_reentry_prepared=True, audit_completed=False, numerically_sufficient=False,
               source_multiplier=1., reentry_history_sha256=sha(history)))
    return jid, folder, archive


def main():
    controls = []
    def reject(name, change):
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs/test_fixtures') as name_dir:
            root = Path(name_dir)
            jid, folder, archive = build_fixture(root)
            change(root, folder, archive)
            try:
                qualified_epoch(root, jid, 'gate')
            except (RuntimeError, FileNotFoundError, ValueError, KeyError):
                controls.append(dict(control=name, passed=True))
            else:
                raise AssertionError(name)
    def alter(p, **values):
        d = json.loads(p.read_text()); d.update(values); write_json(p, d)
    def rehash_archive_audit(root, folder, archive):
        # A self-consistent archive hash cannot conceal missing derivative coverage.
        alter(archive/'independent_audit.json', checks=[])
        m=json.loads((archive/'archive_manifest.json').read_text())
        p=archive/'independent_audit.json'
        m['files'][p.name]=dict(sha256=sha(p),bytes=p.stat().st_size)
        write_json(archive/'archive_manifest.json',m)
    with tempfile.TemporaryDirectory(dir=ROOT/'outputs/test_fixtures') as d:
        root=Path(d);jid,folder,archive=build_fixture(root)
        q=qualified_epoch(root,jid,'gate')
        state=dict(phase='remaining_paths_require_same_point_review',running=[],
                   owned_paths=[jid,'other'],completed=[],events=[dict(old=True)],
                   reviews=[dict(job=jid,reason='same_point_review_required'),
                            dict(job='other',reason='scientific_child_failure')])
        before=copy.deepcopy(state);updated=rearmed_state(state,[q])
        assert state==before and updated['reviews']==[state['reviews'][1]]
        assert updated['events'][0]==state['events'][0]
        assert updated['events'][-1]['training_dispatched'] is False
        for label,change in [
            ('empty_reentry',lambda s:None),
            ('live_child_record',lambda s:s.update(running=[dict(job=jid)])),
            ('nonterminal_campaign',lambda s:s.update(phase='running')),
            ('completed_path',lambda s:s.update(completed=[jid])),
            ('foreign_job',lambda s:s.update(owned_paths=['other'])),
            ('scientific_failure',lambda s:s['reviews'][0].update(reason='scientific_child_failure'))]:
            s=copy.deepcopy(state);change(s)
            try:rearmed_state(s,[] if label=='empty_reentry' else [q])
            except RuntimeError:controls.append(dict(control=label,passed=True))
            else:raise AssertionError(label)
    reject('modified_checkpoint',lambda r,f,a:(f/'optimizer.pkl').write_bytes(b'changed'))
    reject('modified_point',lambda r,f,a:(f/'best.npy').write_bytes(b'changed'))
    reject('damaged_archive',lambda r,f,a:(a/'optimizer.pkl').write_bytes(b'changed'))
    reject('incomplete_transaction',lambda r,f,a:alter(a/'preparation_journal.json',phase='archive_verified_before_new_memory'))
    reject('changed_descending_probe',lambda r,f,a:(r/'analysis/probe_same_point_descent.py').write_text('# changed'))
    reject('false_qualification_with_updated_archive_hash',rehash_archive_audit)
    reject('already_started_epoch',lambda r,f,a:alter(f/'status.json',status='running'))
    reject('old_audit_still_active',lambda r,f,a:write_json(f/'independent_audit.json',{}))
    reject('old_freeze_still_active',lambda r,f,a:write_json(f/'prediction_freeze.json',{}))
    reject('wrong_source_multiplier',lambda r,f,a:alter(f/'status.json',source_multiplier=2.))
    result=dict(passed=True,negative_controls=controls,source_sha256=sha(ROOT/'analysis/rearm_reviewed_campaign.py'),
                scope='synthetic_native_archive_and_rearm_only_no_real_path_restarted',
                scientific_files_changed=False,training_dispatched=False)
    write_json(ROOT/'evidence/rearm_reviewed_campaign_known_errors.json',result)
    print('passed: archived qualification and '+str(len(controls))+' rearm negative controls; no real restart')


if __name__=='__main__':main()
