"""Phase 2 -- G2-G5.  REFUSES TO RUN unless Phase 1 opened the gate.

THE EARLY STOP IS ENFORCED HERE IN CODE, NOT BY RESTRAINT
---------------------------------------------------------
The round carries a mandate no earlier round carried:

    "this round looks at L1 first.  If even L1 cannot produce amplitude, stop
     immediately; there is no need to run the complex spatial validation."

A mandate that lives only in a plan is a mandate that gets reinterpreted once the
numbers are in.  So the gate is an executable precondition: this module reads
`reports/phase1_l1.json`, and unless `phase1.proceed_to_phase2` is true it writes
`status = NOT_RUN` with the reason and exits non-zero WITHOUT building a model, without
touching a frozen file and without computing a single statistic.

The refusal is a result and is written to `reports/phase2_full.json`, whose first field is
`status`, so the file cannot be mistaken for a Phase 2 that ran.

WHAT PHASE 2 WOULD HAVE DONE (for the record, not executed)
-----------------------------------------------------------
G2 L3 amplitude, G3 the 15-station `|log(SD_pred/SD_obs)|` median, G4 the F1/F3
`beta`/`alpha` distances, G5 the monthly NSE pair, plus the non-gating
`mean_concentration_relative_change`.  All of it is unreachable: G1 was not reached by any
point of the registered grid nor by the limit case, and G2 was not reached either, so even
under the `L1_ONLY` branch there is nothing to carry forward.
"""
import hashlib
import json
from pathlib import Path

ROUND = Path(__file__).resolve().parent.parent
REPORTS = ROUND / 'reports'
PRE = REPORTS / 'pre_registration.json'
L1 = REPORTS / 'phase1_l1.json'
OUT = REPORTS / 'phase2_full.json'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    pre = json.loads(PRE.read_text(encoding='utf-8'))
    if sha(REPORTS / '预注册_判据与门槛.md') != pre['pre_registration']['sha256']:
        raise SystemExit('PRE_REGISTRATION_CHANGED')
    l1 = json.loads(L1.read_text(encoding='utf-8'))
    ph = l1['phase1']
    allowed = bool(ph['proceed_to_phase2'])
    body = dict(
        status='RUN' if allowed else 'NOT_RUN',
        early_stop_mandate=ph['early_stop_mandate'],
        phase1_verdict=ph['verdict'], phase1_branch=ph['branch'],
        phase1_qualifier=ph['qualifier'],
        g_passing_G1=ph['g_passing_G1'], g_passing_G2=ph['g_passing_G2'],
        limit_case_passes_G1=ph['limit_case_passes_G1'],
        reason=(None if allowed else
                'no point of the registered grid reached G1, and neither did the limit '
                'case; G2 was not reached either, so the L1_ONLY branch does not apply. '
                'Per the mandate the complex spatial validation is not run.'),
        gates_evaluated=[] if not allowed else ['G2', 'G3', 'G4', 'G5'],
        gates_not_evaluated=['G2', 'G3', 'G4', 'G5'] if not allowed else [],
        note=('a NOT_RUN is not a failure of G2-G5: those gates were never tested, and '
              'nothing here may be cited as evidence about them'),
        sources=dict(pre_registration=pre['pre_registration']['sha256'],
                     phase1_readings=sha(L1)))
    OUT.write_text(json.dumps(body, indent=1, ensure_ascii=False, sort_keys=True),
                   encoding='utf-8')
    print('PHASE 2: %s' % body['status'])
    print('  reason: %s' % body['reason'])
    print('  gates not evaluated: %s' % body['gates_not_evaluated'])
    print('WROTE reports/phase2_full.json')
    return 0 if allowed else 3


if __name__ == '__main__':
    raise SystemExit(main())
