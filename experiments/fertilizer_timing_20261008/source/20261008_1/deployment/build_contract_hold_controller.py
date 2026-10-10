"""One exact startup-only operational insertion; original scientific seal intact."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'deployment')]
from d29_platform.runtime import sha,write_json
from contract_hold_policy import held_jobs,append_holds


def main():
    base=ROOT/'deployment/extended_campaign_controller.py'
    build=json.loads((ROOT/'evidence/extended_controller_build.json').read_text(encoding='utf8'))
    if not build['passed'] or sha(base)!=build['extension_sha256']:
        raise RuntimeError('VERIFIED_EXTENDED_BASE_REQUIRED')
    gate=sha(ROOT/'evidence/formal_gate.json')
    held=held_jobs(ROOT,gate)
    text=base.read_text(encoding='utf8')
    needle="    reviews=list(previous.get('reviews',[])) if previous else []\n"
    insertion=needle+"    from contract_hold_policy import held_jobs,append_holds\n    reviews=append_holds(reviews,held_jobs(ROOT,gate_hash))\n"
    if text.count(needle)!=1:
        raise RuntimeError('EXACT_ONE_STARTUP_REVIEW_BLOCK_REQUIRED')
    new=text.replace(needle,insertion)
    if new.replace(insertion,needle)!=text:
        raise RuntimeError('UNREGISTERED_CONTROLLER_CHANGE')
    target=ROOT/'deployment/contract_hold_controller.py'
    compile(new,str(target),'exec')
    target.write_bytes(new.encode('utf8'))
    # Executable policy controls cover exact accounting and preservation.
    reviews=[dict(job='TIM_F23_C0_V0_s0',reason='prior_failure')]
    actual=append_holds(reviews,held)
    assert len(actual)==15 and actual[0]==reviews[0]
    assert append_holds(actual,held)==actual
    assert reviews==[dict(job='TIM_F23_C0_V0_s0',reason='prior_failure')]
    assert {r['job'] for r in actual if r.get('not_convergence_failure')}==set(held)
    receipt=dict(passed=True,original_extended_controller_sha256=sha(base),
                 controller_sha256=sha(target),policy_sha256=sha(ROOT/'deployment/contract_hold_policy.py'),
                 scientific_gate_sha256=gate,affected_paths=held,
                 unchanged_outside_exact_startup_hold_block=True,
                 actual_running_children_not_restarted=True,
                 stage_lifecycle_and_resource_gates_unchanged=True,
                 new_source_epoch_admitted=False,scheduled_task_created=False)
    write_json(ROOT/'evidence/contract_hold_controller_build.json',receipt)
    print(json.dumps(receipt))


if __name__=='__main__':main()
