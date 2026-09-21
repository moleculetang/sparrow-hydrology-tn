"""F0.4b -- where the published baseline sits inside its own soft saturation.

The daily phenotype (work/f0_phenotype.py) shows `sd(pred)/sd(obs)` between 0.25
and 0.50 at 13 of the 15 HF stations: the model under-disperses badly. The D29_BE
multiplier is `exp(A_t)` with `A_t = ln10 * tanh(u / ln10)`, so `u` is pushed
through a tanh that rails at +-ln10. Two very different causes of a damped signal
are possible and this audit separates them:

  * the *multiplier* is pinned near its rails, so it cannot move day to day;
  * the multiplier moves fine, and the damping is introduced downstream by the
    M/L reservoir and routing operators.

The discriminator is the within-month spread of the multiplier itself, measured
against the observed daily spread. Zero fits: this reads the published optimum and
the frozen domain arrays only.

Output: reports/f0_saturation.json
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

RUN = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(RUN / 'scripts')]

import campaign_model as CM                                                   # noqa: E402

FOLD, TAG = 'T24_G_D_H1', 'T24_G_D_H1_s0'
B = math.log(10.0)


def q(a, extra=None):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    d = dict(n=int(a.size), mean=float(a.mean()), sd=float(a.std()),
             min=float(a.min()), p05=float(np.percentile(a, 5)),
             p25=float(np.percentile(a, 25)), median=float(np.median(a)),
             p75=float(np.percentile(a, 75)), p95=float(np.percentile(a, 95)),
             max=float(a.max()))
    if extra:
        d.update(extra)
    return d


def main():
    rec = json.loads((RUN / 'outputs' / TAG / 'model.json').read_text(encoding='utf-8'))
    t_ref = np.asarray(rec['parameters'], float)
    assert len(t_ref) == 30, len(t_ref)
    m = CM.for_job(dict(fold=FOLD, kind='D29_BE', tag=TAG), verify=True)
    t = torch.tensor(t_ref)

    u = torch.einsum('trj,j->tr', m.dynamic_basis, t[21:29])
    scaled = (u / B).detach().numpy()                 # the tanh argument
    mult = torch.exp(B * torch.tanh(u / B)).detach().numpy()
    # the multiplier's own bounds are the claim being audited, so check them here
    assert mult.min() >= 0.1 - 1e-12 and mult.max() <= 10.0 + 1e-12, (mult.min(), mult.max())

    # within-month spread, per (month, reach), on the same day grouping the FCT8
    # weight uses; reach-month cells with fewer than 5 days are skipped
    starts = np.asarray(m.data.starts)
    count = np.bincount(np.asarray(m.data.mid), minlength=len(m.data.months))
    sd_mult, sd_lat = [], []
    for s, c in zip(starts, count):
        if c < 5:
            continue
        blk_m = mult[s:s + c]
        blk_u = scaled[s:s + c]
        sd_mult.append(blk_m.std(axis=0))
        sd_lat.append(blk_u.std(axis=0))
    sd_mult = np.concatenate(sd_mult)
    sd_lat = np.concatenate(sd_lat)

    tanh_abs = np.abs(np.tanh(scaled))
    rails = np.isclose(mult, 10.0, rtol=1e-2) | np.isclose(mult, 0.1, rtol=1e-2)
    out = dict(
        status='F0_SATURATION_AUDIT', fold=FOLD, tag=TAG, zero_fits=True,
        reference_pg=rec['pg'], B=dict(ln10=B, multiplier_range=[0.1, 10.0]),
        tanh_argument=dict(name='u/ln10 (the tanh argument)', **q(scaled)),
        tanh_output=dict(name='tanh(u/ln10)', **q(np.tanh(scaled)),
                         fraction_abs_gt_0p9=float((tanh_abs > 0.9).mean()),
                         fraction_abs_gt_0p5=float((tanh_abs > 0.5).mean()),
                         fraction_abs_lt_0p5=float((tanh_abs < 0.5).mean()),
                         note=('|tanh| > 0.9 means the multiplier is within ~1% of a rail; '
                               '|tanh| < 0.5 is the near-linear region where the multiplier '
                               'still responds proportionally to the latent.')),
        multiplier=dict(name='exp(ln10*tanh(u/ln10))', **q(mult),
                        fraction_at_a_rail_within_1pct=float(rails.mean()),
                        fraction_in_linear_band=float((tanh_abs < 0.5).mean())),
        within_month_spread=dict(
            name='per (month, reach) daily sd, cells with >= 5 days',
            multiplier_sd=q(sd_mult), latent_tanh_argument_sd=q(sd_lat),
            ratio_multiplier_sd_over_latent_sd=float(sd_mult.mean() / max(sd_lat.mean(), 1e-300)),
            d_multiplier_per_d_latent_at_zero=float(B),
            note=('If the multiplier is pinned, `multiplier_sd` is small and the observed '
                  'daily spread cannot be reproduced by *any* reshaping of u within the month '
                  '-- the rails are the binding constraint. If `multiplier_sd` is healthy, the '
                  'damping is downstream (M/L reservoir and routing), not in this factor.')),
    )

    print('=== D29_BE baseline saturation, %s ===' % TAG)
    print('  tanh argument u/ln10 : median %.3f  p05 %.3f  p95 %.3f  min %.3f  max %.3f'
          % (out['tanh_argument']['median'], out['tanh_argument']['p05'],
             out['tanh_argument']['p95'], out['tanh_argument']['min'],
             out['tanh_argument']['max']))
    print('  |tanh| > 0.9 (railed)  : %.2f%%    |tanh| < 0.5 (linear): %.2f%%'
          % (100 * out['tanh_output']['fraction_abs_gt_0p9'],
             100 * out['tanh_output']['fraction_abs_lt_0p5']))
    print('  multiplier           : median %.4f  p05 %.4f  p95 %.4f'
          % (out['multiplier']['median'], out['multiplier']['p05'],
             out['multiplier']['p95']))
    print('  multiplier at a rail : %.4f%% of (day,reach) cells'
          % (100 * out['multiplier']['fraction_at_a_rail_within_1pct']))
    print('  within-month sd      : multiplier %.5f   latent argument %.5f'
          % (out['within_month_spread']['multiplier_sd']['mean'],
             out['within_month_spread']['latent_tanh_argument_sd']['mean']))
    print('  multiplier sd is %.3f%% of what a fully linear latent would give'
          % (100 * out['within_month_spread']['ratio_multiplier_sd_over_latent_sd'] / B))

    (RUN / 'reports/f0_saturation.json').write_text(
        json.dumps(out, ensure_ascii=False, indent=1, sort_keys=True), encoding='utf-8')
    print('\nwrote reports/f0_saturation.json')


if __name__ == '__main__':
    sys.exit(main())
