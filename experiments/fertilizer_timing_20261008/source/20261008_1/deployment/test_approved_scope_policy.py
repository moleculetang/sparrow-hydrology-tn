"""Wrong-epoch, duplicate, multiplier and review-release refusal controls."""
import copy
import json
import sys
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'deployment')]
from approved_scope_policy import plan,route,qualify_record,corrected_reviews,HOLD_REASON


class Controls(unittest.TestCase):
    def setUp(self):
        self.jobs=json.loads((ROOT/'config/jobs.json').read_text(encoding='utf8'))
        self.p=plan(self.jobs,'a'*64,'b'*64)
        self.k=next(j['id'] for j in self.jobs if j['calendar']=='K')
        self.n=next(j['id'] for j in self.jobs if j['calendar']=='C0')

    def test_exact_registration(self):
        self.assertEqual(sum(v=='approved_maize_K' for v in self.p['epochs'].values()),14)
        self.assertEqual(route(ROOT,self.k,self.p),ROOT.with_name('20261008_1_maize_scope_v2'))
        self.assertEqual(route(ROOT,self.n,self.p),ROOT)

    def test_missing_and_duplicate_job_rejected(self):
        for jobs in [self.jobs[:-1],self.jobs[:-1]+[self.jobs[0]]]:
            with self.assertRaises(RuntimeError):plan(jobs,'a','b')

    def test_extra_K_rejected(self):
        jobs=copy.deepcopy(self.jobs);next(j for j in jobs if j['calendar']=='C0')['calendar']='K'
        with self.assertRaises(RuntimeError):plan(jobs,'a','b')

    def test_unregistered_calendar_rejected(self):
        jobs=copy.deepcopy(self.jobs);jobs[0]['calendar']='N'
        with self.assertRaises(RuntimeError):plan(jobs,'a','b')

    def test_inherited_multiplier_rejected(self):
        jobs=copy.deepcopy(self.jobs);jobs[0]['fixed_log_source_multiplier']=.3
        with self.assertRaises(RuntimeError):plan(jobs,'a','b')

    def test_same_and_missing_gate_rejected(self):
        for a,b in [('a','a'),('a',''),('','b')]:
            with self.assertRaises(RuntimeError):plan(self.jobs,a,b)

    def test_only_original_nonK_can_be_unlabelled_live_child(self):
        self.assertEqual(qualify_record(dict(job=self.n),self.p),('original_nonK','a'*64))
        with self.assertRaises(RuntimeError):qualify_record(dict(job=self.k),self.p)

    def test_wrong_K_epoch_rejected(self):
        with self.assertRaises(RuntimeError):
            qualify_record(dict(job=self.k,epoch='original_nonK',job_gate_sha256='a'*64),self.p)

    def test_wrong_K_gate_rejected(self):
        with self.assertRaises(RuntimeError):
            qualify_record(dict(job=self.k,epoch='approved_maize_K',job_gate_sha256='a'*64),self.p)

    def test_approved_K_identity_passes(self):
        self.assertEqual(qualify_record(dict(job=self.k,epoch='approved_maize_K',job_gate_sha256='b'*64),self.p),('approved_maize_K','b'*64))

    def test_only_explicit_contract_hold_is_removed(self):
        held=dict(job=self.k,reason=HOLD_REASON)
        failed=dict(job=self.k,reason='scientific_child_failure')
        self.assertEqual(corrected_reviews([held,failed],self.p),[failed])

    def test_nonK_and_unknown_hold_rejected(self):
        for jid in [self.n,'unknown']:
            with self.assertRaises(RuntimeError):corrected_reviews([dict(job=jid,reason=HOLD_REASON)],self.p)
        with self.assertRaises(RuntimeError):route(ROOT,'unknown',self.p)


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Controls))
    from d29_platform.runtime import sha,write_json
    write_json(ROOT/'evidence/approved_scope_policy_controls.json',dict(passed=result.wasSuccessful(),tests=result.testsRun,
        policy_sha256=sha(ROOT/'deployment/approved_scope_policy.py'),test_sha256=sha(Path(__file__)),
        scientific_sources_changed=False,live_controller_changed=False,training_dispatched=False,
        proves_actual_corrected_gate=False))
    sys.exit(0 if result.wasSuccessful() else 2)
