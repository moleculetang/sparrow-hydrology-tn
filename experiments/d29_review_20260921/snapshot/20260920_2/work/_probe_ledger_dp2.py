"""PROBE, not a deliverable: does the round-2 ledger/tag pair satisfy the six-channel
label identity, on a real arm, before `phase1_arms.py` is written against it?

WHY THIS PROBE EXISTS
---------------------
Round 2 splits the single land-phase pool `M` into `ML` (legacy) + `MM` (mobile):

  * `closures_dp2.ledger_dp2` returns `M = ML + MM`  (the SUM of the two states);
  * `closures_dp2._tag_dp2_nb` returns `ms[t,r,j] = MM[r,j]`  (the MOBILE state alone).

`campaign_model.Matched.ledger:105` maps tag index 3 to the label name `'M'` and then
computes, at :107,

    a['source_label_sum_errors']['M'] = max|tags[3].sum(-1) - a['M'][:, rr]|

which compares the tagged MOBILE pool against the LEDGER's summed pool.  If those are
different objects the residual is the legacy pool's size -- not a rounding -- and
`local_balance_max_kg` / `all_hold` would be graded against a broken channel.

The only honest way to settle it is to measure, not to reason about it: run the real
ledger through the real binding on a real arm and print every channel.

OUTCOME (measured, then fixed in `closures_dp2._tag_dp2_nb`)
-----------------------------------------------------------
Slot 3 WAS wrong.  On arm `P-1e2` the residual was `M: 1.831372e+07` against a 1e-6 kg
tolerance -- seven orders out -- while the other five channels were 8.7e-10, 2.6e-10,
7.3e-08, 1.7e-09 and 1.0e-11.  Decomposition: `|tag_M.sum(-1) - (ML+MM)| = 1.831372e+07`
but `|tag_M.sum(-1) - MM| = 1.443550e-08`.  So the tag kernel's `ms` was the right KIND of
object and only the ledger's `M` spelling was being compared against the wrong half.

The fix is in the KERNEL, not in the gate: `ms[t,r,j] = ML[r,j] + MM[r,j]`, the same
association `ledger_dp2:447` uses.  No threshold was touched and no channel was exempted;
the six-channel gate keeps exactly its registered teeth.  At `q_m = 1` the change is
inert (see the slot-3 note in `_tag_dp2_nb`), so Phase 0's N1' bitwise reduction stands.

This file writes nothing and is not imported by anything.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

WORK = Path(__file__).resolve().parent
sys.path.insert(0, str(WORK))

import common25 as C      # noqa: E402
import closures_dp2 as MC  # noqa: E402

TAG = C.TAG


def main():
    model = C.build(TAG)
    arms = C.read_json(C.ROUND / 'reports' / 'arms.json')
    arm = [a for a in arms['arms'] if a['arm'] == 'P-1e2'][0]
    print('[probe] arm', arm['arm'], 'q_m', repr(arm['q_m']), 'tau_m', arm['tau_m'])

    C.dp_arrays(model, q_m=float(arm['q_m']), form=arms['closure_form'])
    pack = model.dp_fractions
    print('[probe] pack q_m', repr(pack['q_m']), 'tau_m', pack['tau_m'],
          'k_m', pack['k_m'])
    info = C.install_kernel(model)
    print('[probe] installed all_bound', info['all_bound'],
          'ledger_is_frozen', info['ledger_is_frozen'],
          'q_m_installed', info['q_m_installed'])
    print('[probe] ledger bound is ledger_dp2:', C._ledger_is_dp2())

    try:
        a = model.ledger(C.parameters(TAG))
        print('\n[probe] a.keys() =', sorted(a.keys()))
        print('[probe] local_balance_max_kg      = %r' % a['local_balance_max_kg'])
        print('[probe] network_balance_kg        = %r' % a['network_balance_kg'])
        lab = a.get('source_label_sum_errors')
        print('[probe] source_label_sum_errors   = %r' % lab)
        if lab:
            for k, v in sorted(lab.items()):
                print('          %-14s %.6e   (tol 1e-6 -> %s)'
                      % (k, float(v), 'OK' if float(v) <= 1e-6 else 'FAIL'))
            print('[probe] max label error = %.6e' % max(float(v) for v in lab.values()))
        # The decisive decomposition: is the 'M' residual a rounding or the legacy pool?
        rr = np.asarray(C.pilot_indices(model), int)
        if a.get('source_labels') and 'legacy_state' in a:
            Msum = np.asarray(a['M'], float)[:, rr]
            ML = np.asarray(a['legacy_state'], float)[:, rr]
            MM = np.asarray(a['mobile_state'], float)[:, rr]
            tagsM = np.asarray(a['source_labels']['M'], float)
            d_sum = np.abs(tagsM.sum(-1) - Msum)
            d_mob = np.abs(tagsM.sum(-1) - MM)
            print('\n[probe] |tag_M.sum(-1) - (ML+MM)| max = %.6e   <- the GRADED channel'
                  % d_sum.max())
            print('[probe] |tag_M.sum(-1) -  MM   | max = %.6e   <- NOT the graded object'
                  % d_mob.max())
            print('[probe] legacy pool magnitude |ML| max      = %.6e' % np.abs(ML).max())
            print('[probe] the two disagree on %d of %d cells'
                  % (int((d_sum != d_mob).sum()), int(d_sum.size)))
        print('[probe] ledger_body_ran = %r' % a.get('ledger_spelling'))
    finally:
        print('\n[probe] is_installed before restore:', C.is_installed()['all_bound'])
        C.restore_kernel()
        model.dp_fractions = None
        print('[probe] is_installed after restore :', C.is_installed()['all_bound'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
