"""The compensation control: corrected hydrology driven by H0's fitted parameters.

§5.2 P2 asks whether H0's parameters were absorbing the error in the old water.
That is a forward question and needs no fitting -- build the H1 model, hand it
H0's optimum, read the prediction.  Three objectives then separate the two
effects at fixed training data:

    E_H0(x_H0)   what the published arm achieved
    E_H1(x_H0)   the water changed, the parameters did not  -> the water's own effect
    E_H1(x_H1)   what re-fitting under the new water buys back

If E_H1(x_H0) is much worse than E_H0(x_H0) the old parameters were tuned to the
old water and do not transfer; if E_H1(x_H1) recovers most of that gap the refit
did real work rather than walking back to the same optimum.

Two jobs, and they are different:

  1. PATHWAY NEGATIVE CONTROL (§6.4).  The same code driven with the H0 domain and
     H0 parameters must reproduce `outputs/<h0_tag>/predictions.parquet` bit for
     bit.  Otherwise the replay pathway is not the fitted pathway, and nothing the
     replay arm says afterwards is interpretable.  It runs first; the arm is not
     written if it fails.

  2. THE REPLAY ARM.  H1 domain, H0 parameters, both starts, all four fold pairs.

One fold per process.  `native_runtime.label_barrier` installs an audit hook that
permits exactly one fold's `train.parquet`; calling it four times in one process
would make the four hooks forbid each other.  So the parent runs the pathway
control and then re-invokes itself once per fold, which is also the isolation the
fit workers themselves run under.

This module produces predictions only.  It scores nothing and reads no held-out
label: `work/audit_arms.py` rebuilds every reported number from the parquets
without importing this file, the model code, or the metric module.

Output: work/replay/<h1_tag>/{statistical_products,daily_station_mass_water,predictions}.parquet
        work/replay/replay_arm.json
"""

from __future__ import annotations

import argparse
import gc
import json
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
sys.path.insert(0, str(RUN / 'scripts'))

import numpy as np                                                            # noqa: E402
import pandas as pd                                                           # noqa: E402
import torch                                                                  # noqa: E402

import native_runtime as rt                                                   # noqa: E402
from campaign_model import for_job, load_data, make_model                     # noqa: E402
from temporal_model import aggregate_daily                                    # noqa: E402

PAIRS = ['T24_L_M', 'T24_L_D', 'T24_G_M', 'T24_G_D']
STARTS = [0, 1]
CHECK_FOLD = 'T24_G_M'   # the pathway control runs on one pair; it is code, not data


def forward(job, x):
    """`audit_job.py`'s prediction block, for an arbitrary parameter vector.

    A transcription rather than a refactor, on purpose: the pathway control exists
    to show this reproduces the fitted pathway, and a refactor made in step with
    `audit_job.py` would hide a divergence instead of exposing it.
    """
    cfg = json.loads((RUN / 'configs/folds.json').read_text(encoding='utf-8'))[job['fold']]
    model = for_job(job)
    design, names, bounds = dict(model.design), list(model.names), list(model.bounds)
    del model
    gc.collect()
    design['observation_registry_file'] = 'data/prediction_registry.json'
    design['observation_registry_hash'] = rt.sha(RUN / design['observation_registry_file'])
    d = load_data(cfg['domain'])
    pm = make_model(d, None, job['kind'], design)
    meta = pd.read_parquet(RUN / 'data/prediction_calendar.parquet')
    meta = meta[meta.year.le(cfg['end_year'])].copy()
    p = pm.predict(x, meta)
    assert np.isfinite(p).all() and p.min() >= -1e-9, 'non-physical prediction'
    with torch.no_grad():
        daily = pm.daily_boundary(torch.tensor(x), meta)
    ri = daily['record'].numpy()
    di = daily['day_index'].numpy()
    F = daily['mass'].numpy()
    V = daily['water'].numpy()
    w = daily['weights'].numpy()
    frame = pd.DataFrame(dict(observation_id=meta.observation_id.to_numpy()[ri],
                              station_key=meta.station_key.to_numpy()[ri],
                              date=d.dates[di], mass_kg_day=F, water_m3_day=V,
                              concentration_mg_l=1000 * F / V, weight=w))
    products = meta.copy()
    for operator in ['FW', 'MATCH']:
        products[operator + '_mg_l'] = aggregate_daily(
            daily['mass'], daily['water'], daily['record'], daily['weights'],
            len(meta), operator).numpy()
    predicted = meta.copy()
    predicted['prediction_mg_l'] = p
    del pm, d, daily
    gc.collect()
    return dict(names=names, bounds=bounds, products=products, frame=frame, predicted=predicted)


def pathway_control():
    """§6.4: H0 domain plus H0 parameters must reproduce the fitted H0 outputs."""
    jobs = rt.read(RUN / 'configs/jobs.json')
    base = f'{CHECK_FOLD}_s0'
    job = next(j for j in jobs if j['tag'] == base)
    x = np.asarray(rt.read(RUN / 'outputs' / base / 'model.json')['parameters'], float)
    out = forward(job, x)
    ref = pd.read_parquet(RUN / 'outputs' / base / 'predictions.parquet')
    both = ref[['observation_id', 'prediction_mg_l']].merge(
        out['predicted'][['observation_id', 'prediction_mg_l']], on='observation_id',
        validate='one_to_one', suffixes=('_ref', '_got'))
    assert len(both) == len(ref), 'replay pathway lost monthly rows'
    conc_gap = float(np.max(np.abs(both.prediction_mg_l_ref - both.prediction_mg_l_got)))
    refd = pd.read_parquet(RUN / 'outputs' / base / 'daily_station_mass_water.parquet')
    bothd = refd[['observation_id', 'date', 'concentration_mg_l']].merge(
        out['frame'][['observation_id', 'date', 'concentration_mg_l']],
        on=['observation_id', 'date'], validate='one_to_one', suffixes=('_ref', '_got'))
    assert len(bothd) == len(refd), 'replay pathway lost daily rows'
    day_gap = float(np.max(np.abs(bothd.concentration_mg_l_ref - bothd.concentration_mg_l_got)))
    del out
    gc.collect()
    return dict(tag=base, rows_monthly=int(len(both)), rows_daily=int(len(bothd)),
                max_abs_monthly_concentration_gap=conc_gap,
                max_abs_daily_concentration_gap=day_gap,
                bit_identical=bool(conc_gap == 0.0 and day_gap == 0.0))


def run_fold(base):
    """One fold's two starts, in this process, under this fold's label barrier."""
    fold = base + '_H1'
    rt.label_barrier(fold)
    jobs = rt.read(RUN / 'configs/jobs.json')
    rows = []
    for start in STARTS:
        h0_tag, h1_tag = f'{base}_s{start}', f'{fold}_s{start}'
        job = next(j for j in jobs if j['fold'] == fold and j['start'] == start)
        x = np.asarray(rt.read(RUN / 'outputs' / h0_tag / 'model.json')['parameters'], float)
        # E_H1 at H0's point and at H1's own point use the same H1 training frame, which
        # is byte-identical to H0's, so the pair reads what the water change did at fixed data.
        model = for_job(job)
        at_h0, _ = model.value_gradient(x)
        assert all(lo <= v <= hi for v, (lo, hi) in zip(x, model.bounds)), 'H0 point off H1 bounds'
        del model
        gc.collect()
        own = rt.read(RUN / 'outputs' / h1_tag / 'model.json')
        out = forward(job, x)
        root = RUN / 'work/replay' / h1_tag
        root.mkdir(parents=True, exist_ok=True)
        out['products'].to_parquet(root / 'statistical_products.parquet', index=False)
        out['frame'].to_parquet(root / 'daily_station_mass_water.parquet', index=False)
        out['predicted'].to_parquet(root / 'predictions.parquet', index=False)
        row = dict(h1_tag=h1_tag, h0_tag=h0_tag, fold=fold, start=start,
                   objective_h0_arm=float(rt.read(RUN / 'outputs' / h0_tag / 'model.json')['objective']),
                   objective_h1_at_h0_parameters=float(at_h0),
                   objective_h1_arm=float(own['objective']),
                   parameter_max_abs_change=float(np.max(np.abs(
                       x - np.asarray(own['parameters'], float)))))
        rows.append(row)
        print(f"  [replay] {h1_tag:<18} E_H0(x_H0) {row['objective_h0_arm']:.6f}  "
              f"E_H1(x_H0) {row['objective_h1_at_h0_parameters']:.6f}  "
              f"E_H1(x_H1) {row['objective_h1_arm']:.6f}", flush=True)
        del out
        gc.collect()
    rt.write(RUN / 'work/replay' / f'fold_{base}.json', dict(fold=fold, rows=rows))
    return rows


def main():
    assert Path(sys.prefix).name.lower() == 'sparrow', 'conda sparrow required'
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', default=None, help='run one fold in this process')
    args = parser.parse_args()
    if args.fold:
        run_fold(args.fold)
        return

    started = time.time()
    report = dict(stage='20260917_5/work/replay_arm',
                  started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    control = pathway_control()
    report['pathway_control'] = control
    print(f"[pathway] {control['tag']} monthly gap "
          f"{control['max_abs_monthly_concentration_gap']:.3e} daily gap "
          f"{control['max_abs_daily_concentration_gap']:.3e} "
          f"{'IDENTICAL' if control['bit_identical'] else 'DIVERGED'}", flush=True)
    assert control['bit_identical'], (
        'the replay pathway does not reproduce the fitted pathway; the replay arm '
        'would be measuring the pathway rather than the hydrology')

    rows = []
    for base in PAIRS:
        print(f'[{base}]', flush=True)
        run = subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()),
                              '--fold', base], cwd=RUN, creationflags=subprocess.CREATE_NO_WINDOW)
        assert run.returncode == 0, f'replay subprocess failed for {base}'
        rows.extend(rt.read(RUN / 'work/replay' / f'fold_{base}.json')['rows'])

    report['replays'] = rows
    report['peak_gib'] = float(rt.process(os.getpid())['peak_gib'])
    report['seconds'] = time.time() - started
    report['status'] = 'REPLAY_ARM_COMPLETE'
    rt.write(RUN / 'work/replay/replay_arm.json', report)
    print(f"\n{report['status']}  {len(rows)} replays  {report['seconds']:.1f}s", flush=True)


if __name__ == '__main__':
    main()
