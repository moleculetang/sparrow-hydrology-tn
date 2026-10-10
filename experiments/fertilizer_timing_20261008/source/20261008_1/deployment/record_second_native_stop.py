"""Record the actual F23 native stop and separate audit lease; no promotion."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from host_resources import registry,registry_root
from d29_platform.runtime import sha,write_json

def main():
    jid='TIM_F23_C0_V0_s0';f=ROOT/'outputs/jobs'/jid
    state=json.loads((f/'status.json').read_text(encoding='utf8'))
    old=dict(pid=61972,created='134359598718239829')
    if registry.identity(old['pid'])==old['created']:
        raise RuntimeError('ORIGINAL_TRAINING_PROCESS_STILL_LIVE')
    if state['owner']['pid']!=old['pid'] or state['owner']['created']!=old['created'] or state['status']!='PG_met_full_audit_pending':
        raise RuntimeError('ACTUAL_NATIVE_STOP_RECEIPT_REQUIRED')
    if state.get('numerically_sufficient') or state.get('audit_completed'):
        raise RuntimeError('STOP_AND_FULL_AUDIT_MUST_BE_DISTINCT')
    controller=json.loads((ROOT/'outputs/campaign_local/status.json').read_text(encoding='utf8'))
    if registry.identity(controller['owner']['pid'])!=controller['owner']['created']:
        raise RuntimeError('ACTUAL_LOCAL_CONTROLLER_REQUIRED')
    record=next(x for x in controller['running'] if x['job']==jid and x['stage']=='audit')
    if registry.identity(record['pid'])!=record['created']:
        raise RuntimeError('ACTUAL_INDEPENDENT_AUDIT_PROCESS_REQUIRED')
    token=record['lease']['token']
    lease=registry.status(root=registry_root())['leases'].get(token)
    if not lease or lease['pid']!=record['pid'] or lease['created']!=record['created'] or lease['job']!='audit/'+jid:
        raise RuntimeError('ACTUAL_SEPARATE_AUDIT_LEASE_REQUIRED')
    contract=json.loads((f/'training_contract.json').read_text(encoding='utf8'))
    receipt=dict(job=jid,original_training_owner=old,training_native_exited=True,
        stopped_status=state,audit_owner=dict(pid=record['pid'],created=record['created']),
        audit_lease=lease,formal_gate_sha256=sha(ROOT/'evidence/formal_gate.json'),
        training_contract_sha256=contract['contract_sha256'],
        artifacts={name:sha(f/name) for name in ('best.npy','best_gradient.npy','optimizer.pkl','training_contract.json')},
        numerical_sufficiency_certified=False,evaluation_scores_calculated=False,
        source_multiplier=1.,optimizer_state_reencoded=False)
    target=ROOT/'outputs/second_native_training_stop/receipt.json'
    if target.exists():
        if json.loads(target.read_text(encoding='utf8'))!=receipt:raise RuntimeError('STOP_RECEIPT_CONFLICT')
    else:write_json(target,receipt)
    print(json.dumps(dict(job=jid,calls=state['calls'],J=state['best_objective'],PG=state['projected_gradient'],
                         audit_pid=record['pid'],numerical_sufficiency_certified=False)))

if __name__=='__main__':main()
