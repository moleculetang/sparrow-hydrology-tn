"""Exact operational hold plus keyed lease recovery, no scientific edits."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'deployment')]
from d29_platform.runtime import sha,write_json
from contract_hold_policy import held_jobs


def replace_one(text,needle,replacement):
    if text.count(needle)!=1:raise RuntimeError('EXACT_ONE_REPLACEMENT_REQUIRED '+needle)
    return text.replace(needle,replacement)


def main():
    base=ROOT/'deployment/extended_campaign_controller.py'
    original=base.read_text(encoding='utf8')
    proof=json.loads((ROOT/'evidence/extended_controller_build.json').read_text(encoding='utf8'))
    if not proof['passed'] or proof['extension_sha256']!=sha(base):
        raise RuntimeError('ORIGINAL_EXTENDED_SOURCE_CHANGED')
    control=json.loads((ROOT/'evidence/controller_recovery_policy_known_errors.json').read_text(encoding='utf8'))
    if not control['passed'] or control['source_sha256']!=sha(ROOT/'deployment/controller_recovery_policy.py'):
        raise RuntimeError('CURRENT_KEYED_RECOVERY_CONTROLS_REQUIRED')
    pairs=[
      ("    reviews=list(previous.get('reviews',[])) if previous else []\n",
       "    reviews=list(previous.get('reviews',[])) if previous else []\n    from contract_hold_policy import held_jobs,append_holds\n    reviews=append_holds(reviews,held_jobs(ROOT,gate_hash))\n"),
      ("        token=record['lease']['token'];lease=resources.get(token)\n",
       "        from controller_recovery_policy import resolve_record_lease\n        lease=resolve_record_lease(record,resources);token=lease['token']\n")]
    text=original
    for needle,new in pairs:text=replace_one(text,needle,new)
    reverse=text
    for needle,new in reversed(pairs):reverse=replace_one(reverse,new,needle)
    if reverse!=original:raise RuntimeError('UNREGISTERED_LIFECYCLE_CHANGE')
    target=ROOT/'deployment/contract_hold_controller_v3.py'
    compile(text,str(target),'exec');target.write_bytes(text.encode('utf8'))
    h=ROOT/'deployment/handoff_contract_hold.py'
    handoff=h.read_text(encoding='utf8')
    for old,new in [
        ("folder=root/'outputs/contract_hold_native_controls'","folder=root/'outputs/contract_hold_native_controls_v3'"),
        ("root/'evidence/contract_hold_controller_build.json'","root/'evidence/contract_hold_controller_build_v3.json'"),
        ("root/'deployment/contract_hold_controller.py'","root/'deployment/contract_hold_controller_v3.py'"),
        ("folder=root/'outputs/contract_hold_handoff_v2'","folder=root/'outputs/contract_hold_handoff_v3'"),
        ("    from contract_hold_policy import held_jobs\n","    from contract_hold_policy import held_jobs\n    from controller_recovery_policy import resolve_record_lease\n"),
        ("                lease=state['leases'].get(x['lease']['token'])\n","                lease=resolve_record_lease(x,state['leases'])\n"),
        ("    held=held_jobs(root,gate)\n","    if proof['recovery_policy_sha256']!=sha(root/'deployment/controller_recovery_policy.py'):\n        raise RuntimeError('CURRENT_KEYED_RECOVERY_POLICY_REQUIRED')\n    held=held_jobs(root,gate)\n")]:
        handoff=replace_one(handoff,old,new)
    new_handoff=ROOT/'deployment/handoff_contract_hold_v3.py'
    compile(handoff,str(new_handoff),'exec');new_handoff.write_bytes(handoff.encode('utf8'))
    gate=sha(ROOT/'evidence/formal_gate.json')
    write_json(ROOT/'evidence/contract_hold_controller_build_v3.json',dict(passed=True,
        original_extended_controller_sha256=sha(base),controller_sha256=sha(target),
        policy_sha256=sha(ROOT/'deployment/contract_hold_policy.py'),
        recovery_policy_sha256=sha(ROOT/'deployment/controller_recovery_policy.py'),
        scientific_gate_sha256=gate,affected_paths=held_jobs(ROOT,gate),
        exact_modified_blocks=['startup original-epoch K dispatch eligibility','keyed child lease recovery'],
        unchanged_outside_registered_blocks=True,scientific_sources_changed=False,
        prior_failed_handoff_source_sha256=sha(h),handoff_source_sha256=sha(new_handoff),
        prior_failed_handoff_preserved=True,source_epoch_changed=False))
    print(json.dumps(dict(passed=True,controller_sha256=sha(target),handoff_sha256=sha(new_handoff))))


if __name__=='__main__':main()
