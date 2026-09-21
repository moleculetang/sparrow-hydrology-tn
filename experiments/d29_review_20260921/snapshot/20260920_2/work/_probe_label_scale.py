"""PROBE, not a deliverable: is the `K-slowend` label-sum residual ROUNDOFF that scales with
the pool magnitude, or a defect that happens to show up only at small `q_m`?

Method is the one that settled the A6 defect: decompose the residual against each half of the
summed channel separately, and read it against the size of the object being compared.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

WORK = Path(__file__).resolve().parent
sys.path.insert(0, str(WORK))
import common25 as C      # noqa: E402
import layers25 as LY     # noqa: E402

TAG = C.TAG
ROWS = []


def probe(arm, model, arms):
    C.dp_arrays(model, q_m=float(arm['q_m']), form=arms['closure_form'])
    C.install_kernel(model)
    try:
        a = model.ledger(C.parameters(TAG))
        rr = np.asarray(C.pilot_indices(model), int)
        ML = np.asarray(a['legacy_state'], float)[:, rr]
        MM = np.asarray(a['mobile_state'], float)[:, rr]
        tM = np.asarray(a['source_labels']['M'], float)
        d_sum = float(np.abs(tM.sum(-1) - (ML + MM)).max())
        d_ml = float(np.abs(tM.sum(-1) - ML).max())
        d_mm = float(np.abs(tM.sum(-1) - MM).max())
        scale = float(np.abs(ML + MM).max())
        tot = float(np.abs(ML + MM).sum())
        ROWS.append(dict(arm=arm['arm'], q_m=float(arm['q_m']), d_sum=d_sum, d_ml=d_ml,
                         d_mm=d_mm, max_M=scale, sum_M=tot,
                         rel=d_sum / scale if scale else 0.0,
                         e=a['source_label_sum_errors']))
    finally:
        C.restore_kernel()
        model.dp_fractions = None


def main():
    arms = C.read_json(C.ROUND / 'reports' / 'arms.json')
    for arm in arms['arms']:
        if not arm['installs_kernel']:
            continue
        model = C.build(TAG)
        probe(arm, model, arms)
    print('%-10s %-10s %-12s %-12s %-12s %-12s %-12s %-10s'
          % ('arm', 'q_m', 'M label err', 'vs ML only', 'vs MM only', 'max|M|', 'sum|M|', 'rel'))
    for r in ROWS:
        print('%-10s %-10.3e %-12.4e %-12.4e %-12.4e %-12.4e %-12.4e %-10.2e'
              % (r['arm'], r['q_m'], r['d_sum'], r['d_ml'], r['d_mm'],
                 r['max_M'], r['sum_M'], r['rel']))
    print('\nwhat the OTHER five channels read (max abs per arm):')
    print('%-10s %s' % ('arm', ' '.join('%-11s' % k for k in
          ('fast', 'slow', 'M', 'L', 'uptake', 'mineral_loss'))))
    for r in ROWS:
        e = r['e']
        print('%-10s %s' % (r['arm'], ' '.join('%-11.3e' % e[k] for k in
              ('fast', 'slow', 'M', 'L', 'uptake', 'mineral_loss'))))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
