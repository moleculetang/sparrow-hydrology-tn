"""Verify all fixed replays and same-fold parameter identity before fitting."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from d29_platform.runtime import write_json,sha


def main():
    gate=ROOT/'evidence/formal_gate.json';gate_hash=sha(gate)
    jobs=[j for j in json.loads((ROOT/'config/jobs.json').read_text(encoding='utf8'))
        if j['space_block'] is None and j['calendar']!='K' and j['entry']==0]
    plan=json.loads((ROOT/'deployment/replay_ownership.json').read_text(encoding='utf8'))
    cases=[];errors=[];points={}
    for j in jobs:
        jid=j['id'];relative='outputs/jobs/replay_'+jid
        if plan['owners'][jid]=='workstation':relative='outputs/imported_workstation_replays/'+relative
        folder=ROOT/relative;path=folder/'replay.json'
        if not path.exists():errors.append('MISSING '+jid);continue
        r=json.loads(path.read_text(encoding='utf8'))
        valid=r['passed'] and r['formal_gate_sha256']==gate_hash and r['source_multiplier']==1. and not r['evaluation_scores_read']
        for name,field in (('point.npy','point_sha256'),('physical_ledger.json','physical_ledger_sha256'),
            ('training_support_prediction.npy','prediction_sha256')):
            valid &= sha(folder/name)==r[field]
        for rel,digest in r.get('identities',{}).items():valid &= sha(ROOT/rel)==digest
        key=(j['fold'],j['plant_version']);points.setdefault(key,set()).add(r['point_sha256'])
        if not valid:errors.append('FAILED_OR_STALE '+jid)
        cases.append(dict(job=jid,passed=valid,receipt=relative+'/replay.json',sha256=sha(path),seconds=r['seconds'],native_peak=r['native_peak']))
    if any(len(p)!=1 for p in points.values()):errors.append('FIXED_CALENDARS_HAVE_DIFFERENT_PARENT_PARAMETERS')
    passed=len(jobs)==12 and len(cases)==12 and not errors
    receipt=dict(passed=passed,count=len(cases),expected_count=12,cases=cases,errors=errors,formal_gate_sha256=gate_hash,
        scope='twelve_same_parameter_full_history_fixed_replays_not_fit_or_TN_gain',source_multiplier=1.)
    write_json(ROOT/'evidence/fixed_replays.json',receipt);print(json.dumps(dict(passed=passed,count=len(cases),errors=errors)))
    return 0 if passed else 2


if __name__=='__main__':sys.exit(main())
