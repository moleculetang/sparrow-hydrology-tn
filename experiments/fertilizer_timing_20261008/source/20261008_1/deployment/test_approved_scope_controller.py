"""Offline real-function lifecycle controls; no live child or TN access."""
import ast
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'deployment')]
from approved_scope_policy import plan,route,qualify_record,corrected_reviews


class Controls(unittest.TestCase):
    def setUp(self):
        jobs=json.loads((ROOT/'config/jobs.json').read_text(encoding='utf8'))
        self.scope=plan(jobs,'a'*64,'b'*64)
        self.k=next(j['id'] for j in jobs if j['calendar']=='K')
        self.n=next(j['id'] for j in jobs if j['calendar']=='C0')
        parent=ROOT/'outputs/test_fixtures/approved_scope_controller';parent.mkdir(parents=True,exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=parent)
        self.base=Path(self.tmp.name)/'20261008_1';self.base.mkdir()
        self.leases={};self.identities={};self.command_checks=[]
        self.registry=SimpleNamespace(identity=lambda p:self.identities.get(p),status=lambda **k:dict(leases=self.leases))
        tree=ast.parse((ROOT/'deployment/approved_scope_controller.py').read_text(encoding='utf8'))
        selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('read','stage_for','recovered_exit','adopt','job_root')]
        self.env=dict(ROOT=self.base,SCIENCE_SCOPE=self.scope,Path=Path,json=json,route=route,
            qualify_record=qualify_record,registry=self.registry,registry_root=lambda:None,
            check_child_command=lambda r,l:self.command_checks.append((r,l)))
        exec(compile(ast.Module(body=selected,type_ignores=[]),'actual_reviewed_functions','exec'),self.env)

    def tearDown(self):self.tmp.cleanup()

    def record(self,jid):
        record=dict(job=jid,stage='train',pid=123,created='real-create',peak_rss_bytes=10,
            lease=dict(project='20261008_1',job='train/'+jid,pid=123,created='real-create',cpus=[1],threads=1,reserved_bytes=100,gpu=None))
        self.identities[123]='real-create';self.leases['exact-token']=dict(record['lease'])
        return record

    def previous(self,records):return dict(formal_gate_sha256='a'*64,running=records)

    def test_real_root_function_selects_corrected_K(self):
        self.assertEqual(self.env['job_root'](self.k),self.base.with_name('20261008_1_maize_scope_v2'))
        self.assertEqual(self.env['job_root'](self.n),self.base)

    def test_old_narrow_K_freeze_is_not_selected(self):
        old=self.base/'outputs/jobs'/self.k;old.mkdir(parents=True)
        (old/'prediction_freeze.json').write_text(json.dumps(dict(numerically_sufficient=True)))
        self.assertEqual(self.env['stage_for'](self.env['job_root'](self.k)/'outputs/jobs'/self.k),'train')

    def test_old_nonK_child_is_retained_with_exact_key(self):
        record=self.record(self.n)
        running,events=self.env['adopt'](self.previous([record]),'a'*64)
        self.assertFalse(events);self.assertEqual(running[self.n]['pid'],123)
        self.assertEqual(running[self.n]['lease']['token'],'exact-token')
        self.assertEqual(running[self.n]['epoch'],'original_nonK')
        self.assertEqual(running[self.n]['job_gate_sha256'],'a'*64)
        self.assertEqual(len(self.command_checks),1)

    def test_live_old_K_refused_even_with_valid_lease(self):
        record=self.record(self.k)
        with self.assertRaises(RuntimeError):self.env['adopt'](self.previous([record]),'a'*64)

    def test_current_K_retained_only_with_correct_epoch(self):
        record=self.record(self.k);record.update(epoch='approved_maize_K',job_gate_sha256='b'*64)
        running,events=self.env['adopt'](self.previous([record]),'a'*64)
        self.assertEqual(running[self.k]['epoch'],'approved_maize_K');self.assertFalse(events)

    def test_wrong_K_gate_refused(self):
        record=self.record(self.k);record.update(epoch='approved_maize_K',job_gate_sha256='a'*64)
        with self.assertRaises(RuntimeError):self.env['adopt'](self.previous([record]),'a'*64)

    def test_wrong_original_gate_refused(self):
        record=self.record(self.n)
        with self.assertRaises(RuntimeError):self.env['adopt'](self.previous([record]),'different')

    def test_reserved_core_changes_refused(self):
        record=self.record(self.n);self.leases['exact-token']['cpus']=[2]
        with self.assertRaises(RuntimeError):self.env['adopt'](self.previous([record]),'a'*64)

    def test_current_K_dead_child_uses_corrected_terminal_receipt(self):
        record=self.record(self.k);record.update(epoch='approved_maize_K',job_gate_sha256='b'*64)
        self.identities.clear()
        old=self.base/'outputs/jobs'/self.k;old.mkdir(parents=True)
        (old/'status.json').write_text(json.dumps(dict(status='PG_met_full_audit_pending')))
        running,events=self.env['adopt'](self.previous([record]),'a'*64)
        self.assertFalse(running);self.assertEqual(events[0]['exit_code'],3)
        new=self.env['job_root'](self.k)/'outputs/jobs'/self.k;new.mkdir(parents=True)
        (new/'status.json').write_text(json.dumps(dict(status='PG_met_full_audit_pending')))
        _,events=self.env['adopt'](self.previous([record]),'a'*64)
        self.assertEqual(events[0]['exit_code'],0)

    def test_pending_review_not_freeze(self):
        new=self.env['job_root'](self.k)/'outputs/jobs'/self.k;new.mkdir(parents=True)
        (new/'status.json').write_text(json.dumps(dict(status='numerical_review_required')))
        self.assertEqual(self.env['stage_for'](new),'same_point_audit')
        (new/'independent_audit.json').write_text(json.dumps(dict(numerically_sufficient=False)))
        self.assertEqual(self.env['stage_for'](new),'review_required')


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Controls))
    from d29_platform.runtime import sha,write_json
    write_json(ROOT/'evidence/approved_scope_controller_controls.json',dict(passed=result.wasSuccessful(),tests=result.testsRun,
        controller_sha256=sha(ROOT/'deployment/approved_scope_controller.py'),test_sha256=sha(Path(__file__)),
        policy_sha256=sha(ROOT/'deployment/approved_scope_policy.py'),
        actual_functions_exercised=True,live_commands_checked=False,native_handoff_passed=False,
        scientific_processes_signaled=0,model_calls=0,production_jobs_dispatched=0))
    sys.exit(0 if result.wasSuccessful() else 2)
