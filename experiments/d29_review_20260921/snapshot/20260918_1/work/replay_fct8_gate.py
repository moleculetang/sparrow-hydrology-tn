"""F1-7 additivity gate: editing `campaign_model.py` must not move D29_BE.

`campaign_model.py` is one of the 9 frozen core scripts (`seal_global.py:30`), so
adding the FCT8 branch did change its bytes. The claim is not "nothing changed"
but "the change is additive": the source diff against the parent round is three
inserted lines and zero modified ones, and `kind='D29_BE'` cannot reach the new
branch.

This gate turns that source-level argument into a numerical one, using the
mechanism `20260917_5` already proved (`work/replay_arm.py` reproduced
`T24_G_M_s0` bitwise on 5568 monthly + 169476 daily rows):

    H1 domain + H1 published parameters -> forward -> must equal the published
    `outputs/<h1_tag>/{predictions,daily_station_mass_water}.parquet` exactly.

It imports `replay_arm.forward` rather than re-implementing it, because that
transcription is the thing that was already validated against the fitted pathway.

One fold per process (`native_runtime.label_barrier` permits exactly one fold's
`train.parquet`), so each pair runs in a subprocess.

Output: work/fct8_replay_gate.json
"""
from __future__ import annotations

import argparse
import gc
import os
import subprocess
import sys
import time
from pathlib import Path

RUN = Path(__file__).resolve().parents[1]
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
             'NUMBA_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_key, '1')
os.environ['NUMBA_CACHE_DIR'] = str(RUN / 'work/numba')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path[:0] = [str(RUN / 'scripts'), str(RUN / 'work')]

import numpy as np                                                            # noqa: E402
import pandas as pd                                                           # noqa: E402

import native_runtime as rt                                                   # noqa: E402
from replay_arm import forward                                                # noqa: E402

#: the two published H1-D29_BE baselines that FCT8 reuses at zero new fits
TAGS = ['T24_G_D_H1_s0', 'T24_G_M_H1_s0']
#: this round's `configs/jobs.json` holds only the 4 FCT8 jobs by design, so the
#: baseline job graph is read from the frozen parent round.
PARENT = RUN.parent / '20260917_5'


def one(tag):
    """Replay one published baseline path and compare bitwise."""
    jobs = rt.read(PARENT / 'configs/jobs.json')
    job = next(j for j in jobs if j['tag'] == tag)
    if job['kind'] != 'D29_BE':
        raise ValueError('NOT_A_D29_BE_PATH ' + tag)
    ref_path = RUN / 'outputs' / tag / 'model.json'
    ref = rt.read(ref_path)
    x = np.asarray(ref['parameters'], float)
    if len(x) != 30:
        raise ValueError('BASELINE_NOT_30 %s %d' % (tag, len(x)))
    out = forward(job, x)

    pub = pd.read_parquet(RUN / 'outputs' / tag / 'predictions.parquet')
    both = pub[['observation_id', 'prediction_mg_l']].merge(
        out['predicted'][['observation_id', 'prediction_mg_l']], on='observation_id',
        validate='one_to_one', suffixes=('_ref', '_got'))
    if len(both) != len(pub):
        raise ValueError('MONTHLY_ROWS_LOST ' + tag)
    conc = float(np.max(np.abs(both.prediction_mg_l_ref - both.prediction_mg_l_got)))

    pubd = pd.read_parquet(RUN / 'outputs' / tag / 'daily_station_mass_water.parquet')
    bothd = pubd[['observation_id', 'date', 'concentration_mg_l']].merge(
        out['frame'][['observation_id', 'date', 'concentration_mg_l']],
        on=['observation_id', 'date'], validate='one_to_one', suffixes=('_ref', '_got'))
    if len(bothd) != len(pubd):
        raise ValueError('DAILY_ROWS_LOST ' + tag)
    daily = float(np.max(np.abs(bothd.concentration_mg_l_ref - bothd.concentration_mg_l_got)))

    # the frozen FCT8 weight must be h_preD29 at exactly this parameter vector
    w_path = RUN / 'data/folds' / job['fold'] / 'w_preD29.npy'
    w_note = None
    if w_path.exists():
        w = np.load(w_path, allow_pickle=False)
        w_note = dict(path=str(w_path.relative_to(RUN)), shape=list(w.shape),
                      sha256=rt.sha(w_path))
    row = dict(tag=tag, fold=job['fold'], parameters=len(x),
               monthly_rows=int(len(both)), daily_rows=int(len(bothd)),
               max_abs_monthly_concentration_gap=conc,
               max_abs_daily_concentration_gap=daily,
               bit_identical=bool(conc == 0.0 and daily == 0.0), weight=w_note)
    del out
    gc.collect()
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag')
    ap.add_argument('--out', default='work/fct8_replay_gate.json')
    a = ap.parse_args()
    started = time.time()
    if a.tag:
        rt.write(RUN / ('work/fct8_replay_%s.json' % a.tag), one(a.tag))
        return
    rows = []
    for tag in TAGS:
        run = subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()),
                              '--tag', tag], cwd=RUN,
                             creationflags=subprocess.CREATE_NO_WINDOW)
        if run.returncode != 0:
            raise RuntimeError('replay subprocess failed for ' + tag)
        rows.append(rt.read(RUN / ('work/fct8_replay_%s.json' % tag)))
        print('  %-16s monthly gap %.3e  daily gap %.3e  %s'
              % (tag, rows[-1]['max_abs_monthly_concentration_gap'],
                 rows[-1]['max_abs_daily_concentration_gap'],
                 'IDENTICAL' if rows[-1]['bit_identical'] else 'DIVERGED'), flush=True)
    report = dict(gate='F1-7', claim='editing campaign_model.py is additive for D29_BE',
                  parent_round='20260917_5', rows=rows,
                  bit_identical=all(r['bit_identical'] for r in rows),
                  seconds=time.time() - started)
    rt.write(RUN / a.out, report)
    print('\nF1-7 %s' % ('PASS' if report['bit_identical'] else 'FAIL'), flush=True)
    if not report['bit_identical']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
