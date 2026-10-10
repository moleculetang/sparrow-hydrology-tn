"""Fail-closed accounting controls for the original narrow K dispatch hold."""
import copy
import json
import sys
import tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'deployment')]
from contract_hold_policy import held_jobs,append_holds
from d29_platform.runtime import write_json,sha


def main():
    revision=json.loads((ROOT/'deployment/maize_range_revision_v2.json').read_text(encoding='utf8'))
    jobs=json.loads((ROOT/'config/jobs.json').read_text(encoding='utf8'))
    raw=(ROOT/'config/timing_evidence.json').read_bytes()
    failures=[]
    with tempfile.TemporaryDirectory(dir=ROOT/'outputs/test_fixtures') as temporary:
        folder=Path(temporary);(folder/'config').mkdir();(folder/'deployment').mkdir()
        def stage(r,j,config=raw):
            (folder/'config/timing_evidence.json').write_bytes(config)
            write_json(folder/'deployment/maize_range_revision_v2.json',r)
            write_json(folder/'config/jobs.json',j)
        stage(revision,jobs)
        expected=held_jobs(folder,revision['original_gate_sha256'])
        assert len(expected)==14 and set(expected)==set(revision['affected_paths'])
        for name,change in (
            ('wrong_gate',lambda r,j:r.update(original_gate_sha256='bad')),
            ('missing_path',lambda r,j:r['affected_paths'].pop()),
            ('duplicate_path',lambda r,j:r['affected_paths'].__setitem__(0,r['affected_paths'][1])),
            ('nonK_in_hold',lambda r,j:r['affected_paths'].__setitem__(0,'TIM_F23_C0_V0_s0')),
            ('narrow_new_range',lambda r,j:r.update(planned_delta_bounds=[-.05,0.])),
            ('wrong_prior_strength',lambda r,j:r.update(current_normalized_prior_strength_relative_to_planned=1.)),
            ('missing_registered_job',lambda r,j:j.pop()),
            ('duplicate_registered_job',lambda r,j:j.__setitem__(0,j[1])),
            ('changed_calendar',lambda r,j:next(x for x in j if x['calendar']=='K').update(calendar='C2')),
        ):
            r=copy.deepcopy(revision);j=copy.deepcopy(jobs);change(r,j);stage(r,j)
            try:held_jobs(folder,revision['original_gate_sha256'])
            except RuntimeError as exc:failures.append(dict(control=name,reason=str(exc)))
            else:raise AssertionError('CONTROL_WRONGLY_ADMITTED '+name)
        stage(revision,jobs,raw+b'\n')
        try:held_jobs(folder,revision['original_gate_sha256'])
        except RuntimeError as exc:failures.append(dict(control='source_epoch_changed',reason=str(exc)))
        else:raise AssertionError('CHANGED_SOURCE_ADMITTED')
    prior=[dict(job='TIM_F23_C0_V0_s0',reason='scientific_child_failure')]
    held=append_holds(prior,expected)
    assert held[0]==prior[0] and len(held)==15 and append_holds(held,expected)==held
    assert len(prior)==1 and {r['job'] for r in held[1:]}==set(expected)
    write_json(ROOT/'evidence/contract_hold_policy_known_errors.json',dict(passed=True,
        negative_controls=failures,preserves_existing_scientific_reviews=True,
        idempotent=True,holds_exactly14K_leaves30nonK=True,
        source_sha256=sha(ROOT/'deployment/contract_hold_policy.py')))
    print('passed: 10 negative controls; exact14K hold, review preservation and idempotence')


if __name__=='__main__':main()
