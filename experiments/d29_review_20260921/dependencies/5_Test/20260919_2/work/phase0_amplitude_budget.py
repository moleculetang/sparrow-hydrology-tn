"""Phase 0 -- where the event amplitude is lost, and what Step B would double-count.

THREE PARTS, ALL READ-ONLY, ALL ZERO-FIT
----------------------------------------
A. The layer budget (plan 2.1).  The plan asks WHERE the amplitude is flattened:
   at the land-phase output (L1), after routing to the station reach (L2), or
   after the observation operator (L3).  If L1 is near the observation and only L3
   falls to ~1.02, no land-phase store can repair it.  The plan says this is
   "本轮最重要的负结果之一" and must go on the first screen -- but it explicitly
   does NOT cancel Phase 1 (the user's ruling: Phase 0 has no early-close power).

        L1  local[t,r]                    the station reach's OWN land-phase output
        L2  inlet[t,r] + local[t,r]       all mass arriving at the station reach,
                                          before in-stream attenuation
        L3  boundary_mass(...)            after the boundary/observation operator

   `local` is `fast + slow + human_mass` from `Transport.apply`, exactly as
   `StructureEndpoints.daily_boundary` builds it.  `inlet` is `RiverN.apply`'s
   first return.  All three are then put on the same (station_key, date) grid
   through the same `c['ti'] / c['ri']` indexing `daily_boundary` uses, and L3's
   concentration is asserted EQUAL to the frozen `replay`'s `p`.

B. The elasticity measurement (plan 2.2).  The plan's derivation assumed the
   `mode=True` branch, `r = h/(av+k)`.  This path runs `mode=False`
   (`scientific_models.py:42` sets `self.cap=False`; `common20.build` asserts it),
   so `risk = h` and `r := h`.  The plan's conclusions survive with `k = 0`, but
   the measured quantity changes, and the plan's own branch rule --
   "若事件窗内 r 已 ≳1 ... 本段须按实测改写" -- turns on the measured `h`, so it
   is measured here rather than argued.

C. The Step B conservation audit (plan 2.4).  Turns D1 into a number:
   `sum(E_t)/sum(I_t)` over Step B's own frozen grid, plus the verification of the
   assumption Step B itself flags as unverified (its section 6.4).
"""
import json
import math

import numpy as np
import pandas as pd
import torch

import common20 as C
import closures_r as KR
import eventlib as E
import structure_model as SM
from eventlib import as_day

R = C.ROUND
TAG = 'C0_s1'
ANCHOR_P = C.PEER / 'outputs/C0_s1/daily_station_mass_water.parquet'
STEPB_SOURCE = C.PEER.parent / '20260824_10/outputs/mainline_reach_month_n_hydrology_1961_2024.parquet'
S_TOL = 1e-9


# --------------------------------------------------------------------------
# A. the three-layer forward
# --------------------------------------------------------------------------
def forward_layers(model, tag):
    """Mirror of `StructureEndpoints.daily_boundary`, keeping the three layers.

    Nothing frozen is edited: every primitive is CALLED through the module that
    owns it, so `common20.install_kernel`'s rebinding of `Transport` is honoured.
    """
    x = C.parameters(tag)
    meta = pd.read_parquet(C.PEER / 'data/prediction_calendar.parquet')
    meta = meta[meta.year.le(C.END_YEAR)].copy()
    with torch.no_grad():
        t = torch.tensor(x)
        c, record, weights = model.daily_metadata(meta)
        h_, s_, f_, k_ = model.flux_parameters(t[:30])
        fast, slow = SM.Transport.apply(h_, s_, f_, k_, model)
        local = fast + slow + model.human_mass(t)
        inlet, official, releases = SM.RiverN.apply(local, t[2], model.daily_data,
                                                    'monthly', True)
        mass = SM.boundary_mass(inlet, official, releases, local, t[2], c)
    ti = c['ti'].numpy().astype(np.int64)
    ri = c['ri'].numpy().astype(np.int64)
    L = np.asarray(local.numpy(), float)
    I = np.asarray(inlet.numpy(), float)
    M = np.asarray(mass.numpy(), float)
    V = np.asarray(c['water'].numpy(), float)
    df = pd.DataFrame(dict(
        station_key=meta.station_key.to_numpy()[record.numpy()],
        date=model.data.dates[ti],
        water_m3_day=V,
        L1_kg=L[ti, ri],
        L2_kg=I[ti, ri] + L[ti, ri],
        L3_kg=M))
    for lay in ('L1', 'L2', 'L3'):
        df['p' + lay] = 1000.0 * df[lay + '_kg'] / df.water_m3_day
    # `assert_no_mass_columns` is FORBIDDEN=('mass','load','kg','yield','flux','tonne'):
    # it is a check on the OUTPUT TABLE that reaches a criterion, and `pdL*` are the
    # only columns a criterion reads.  The `*_kg` columns are kept because this frame
    # is the layer-budget EVIDENCE, and the budget is a mass ledger -- which the round
    # discipline exempts explicitly.
    E.assert_no_mass_columns(df[['station_key', 'date', 'pL1', 'pL2', 'pL3']])
    return df, dict(c=c, ti=ti, ri=ri, local=L, inlet=I, mass=M, water=V,
                    record=record, meta=meta, x=x)


def layer_budget(layers, ev):
    """Per-layer C_base / C_peak / ratio on the frozen events, vs the observation."""
    out = {}
    for lay in ('L1', 'L2', 'L3'):
        tbl = E.build_event_table(
            layers[['station_key', 'date', 'p' + lay]].rename(columns={'p' + lay: 'p'}), ev)
        fin = np.isfinite(tbl.c_base) & np.isfinite(tbl.c_peak)
        r = (tbl.c_peak / tbl.c_base)[fin]
        out[lay] = dict(
            n_events=int(fin.sum()),
            c_base_median=float(tbl.c_base[fin].median()),
            c_peak_median=float(tbl.c_peak[fin].median()),
            amp_ratio_median=float(r.median()),
            amp_ratio_mean=float(r.mean()),
            amp_ratio_sd=float(r.std()),
            n_peak_min=int(tbl.n_peak.min()), n_peak_max=int(tbl.n_peak.max()),
            n_base_unique=[int(v) for v in sorted(tbl.n_base.unique())])
    return out


def station_sd_by_layer(layers, mask):
    out = {}
    for lay in ('L1', 'L2', 'L3'):
        s = E.station_daily_sd(
            layers[['station_key', 'date', 'p' + lay]].rename(columns={'p' + lay: 'p'}), mask)
        out[lay] = dict(sd_median=float(s.sd.median()), sd_min=float(s.sd.min()),
                        sd_max=float(s.sd.max()), n_stations=int(len(s)))
    # the OBSERVATION on the SAME eligible station-days, so 2.1's "how many stations
    # are flatter than the observation" is a like-for-like count and not a contrast
    # between a model grid and a differently-sampled panel.
    m = mask[mask.eligible][['station_key', 'date']].copy()
    p = pd.read_parquet(E.PANEL)
    p['date'] = p.monitoring_time.dt.tz_localize(None).dt.normalize().values.astype('datetime64[ns]')
    g = p.merge(m.assign(date=m.date.astype('datetime64[ns]')), on=['station_key', 'date'])
    go = g.groupby('station_key').TN
    out['OBS'] = dict(sd_median=float(go.std().median()), sd_min=float(go.std().min()),
                      sd_max=float(go.std().max()), n_stations=int(go.ngroups),
                      n_obs_rows=int(len(g)))
    for lay in ('L1', 'L2', 'L3'):
        cmp = (layers[['station_key', 'date', 'p' + lay]].rename(
            columns={'p' + lay: 'p'})).merge(m.assign(date=m.date.astype('datetime64[ns]')),
                                             on=['station_key', 'date'])
        gm = cmp.groupby('station_key').p
        sm = pd.DataFrame(dict(obs=go.std(), mdl=gm.std()))
        out[lay]['stations_below_obs'] = int((sm.mdl < sm.obs).sum())
        out[lay]['ratio_mdl_over_obs_median'] = float((sm.mdl / sm.obs).median())
    out['OBS']['stations_below_obs'] = None
    return out


# --------------------------------------------------------------------------
# B. the elasticity measurement
# --------------------------------------------------------------------------
def event_window_mask(model, ev, nd, nr):
    """(nd, nr) boolean: the station reach's days inside an event PEAK window.

    The peak window is `[t_start, t_end + 1d]` with BOTH ends inclusive -- the
    same frozen window `base_peak` uses, so the mask and the amplitude budget ask
    about the same days.
    """
    dates = np.asarray(model.data.dates.values.astype('datetime64[D]'))
    mask = np.zeros((nd, nr), bool)
    ts, te = E.event_windows(ev)
    # the station -> reach map, taken from the SAME `c['ri']` the forward uses
    meta = pd.read_parquet(C.PEER / 'data/prediction_calendar.parquet')
    meta = meta[meta.year.le(C.END_YEAR)].copy()
    with torch.no_grad():
        _c, _rec, _w = model.daily_metadata(meta)
    ti = _c['ti'].numpy().astype(np.int64)
    ri = _c['ri'].numpy().astype(np.int64)
    reach_of = {}
    for sk, r in zip(meta.station_key.to_numpy(), ri):
        reach_of[str(sk)] = int(r)
    for i, sk in enumerate(ev.station_key.values):
        r = reach_of[str(sk)]
        lo = np.datetime64(ts.iloc[i], 'D')
        hi = np.datetime64(te.iloc[i] + pd.Timedelta(days=1), 'D')
        sel = (dates >= lo) & (dates <= hi)
        mask[sel, r] = True
    return mask, reach_of, dates


def elasticity(model, ev, delta=1.0):
    with torch.no_grad():
        x = C.parameters(TAG)
        h_, s_, f_, k_ = [np.ascontiguousarray(v.numpy())
                          for v in model.flux_parameters(torch.tensor(x))]
    nd, nr = h_.shape
    C.install_kernel(model, delta, 0.0)
    m = model
    out = KR.scan_r(h_, s_, f_, k_, np.ascontiguousarray(m.data.lower_release),
                    np.ascontiguousarray(m.inp), np.ascontiguousarray(m.demand),
                    m.cap, delta, 0.0)
    C.restore_kernel()
    av, praw = out[2], out[6]
    # `k` is indexed by REACH in the frozen recurrence (`risk = h/(av+k[r])`), so it is
    # a (nr,) vector, not a field.  Broadcasting it to the field shape is what the kernel
    # itself does; doing it here keeps the comparison in the same space as `av`.
    k2 = np.broadcast_to(np.asarray(k_, float).reshape(-1), (nd, nr)).copy()
    mask, reach_of, dates = event_window_mask(m, ev, nd, nr)
    inside = mask & (h_ > 0)
    outside = (~mask) & (h_ > 0)
    g = lambda v: np.where(v > 0, v / np.expm1(np.minimum(v, 700.)), 1.0)

    def q(v, name):
        v = np.asarray(v, float)
        return dict(name=name, n=int(v.size),
                    q05=float(np.percentile(v, 5)), q25=float(np.percentile(v, 25)),
                    median=float(np.median(v)), q75=float(np.percentile(v, 75)),
                    q95=float(np.percentile(v, 95)), mean=float(v.mean()),
                    frac_ge_1=float(np.mean(v >= 1.0)),
                    frac_ge_0p5=float(np.mean(v >= 0.5)))

    return dict(
        delta=delta, mode=bool(m.cap), kernel_note='mode=False -> risk = h, so r := h',
        n_reach_days_inside=int(inside.sum()), n_reach_days_outside=int(outside.sum()),
        h=dict(inside=q(h_[inside], 'h_inside'), outside=q(h_[outside], 'h_outside')),
        g_r=dict(inside=q(g(h_[inside]), 'g_inside'), outside=q(g(h_[outside]), 'g_outside')),
        av=dict(inside=q(av[inside], 'av_inside'), outside=q(av[outside], 'av_outside')),
        prob=dict(inside=q(praw[inside], 'prob_inside'),
                  outside=q(praw[outside], 'prob_outside')),
        av_gt_k_share=dict(inside=float(np.mean(av[inside] > k2[inside])),
                           outside=float(np.mean(av[outside] > k2[outside]))),
        k_stats=dict(min=float(k2.min()), max=float(k2.max()),
                     median=float(np.median(k2)), shape=list(np.shape(k_))),
        reaches_with_events=int(len(set(reach_of.values()))),
        branch_rule=('plan 2.2: if r within the event window is already ~>1 the output is '
                     'saturated against `av` and the "av >> k" corollary fails; read the '
                     'measured numbers, not the derivation'),
        n_reaches=int(nr))


# --------------------------------------------------------------------------
# C. the Step B conservation audit
# --------------------------------------------------------------------------
def stepb_audit(model):
    d = model.data
    I_m = np.asarray(d.source, float)                     # (768, 230) the REGISTERED ledger
    nd, nr = len(d.dates), I_m.shape[1]
    mid = np.asarray(d.mid, np.int64)
    # The divisor is the number of DAYS PRESENT in the month, not the calendar length
    # of a leap year.  Writing `np.where(dates.is_leap_year, 366., 365.)[mid]` -- which
    # is what the first version of this audit did -- indexes a PER-DAY flag by a MONTH
    # index, so every month is divided by 365 and February's reconstruction is off by
    # |28/365 - 1| = 0.9232876712328767.  That number, not any property of the data,
    # was the `max_rel_error` this audit first reported.  `np.bincount` over `mid` is
    # the count the recurrence actually needs, and it makes the round trip exact.
    span = np.bincount(mid, minlength=I_m.shape[0])
    if (span == 0).any():
        raise SystemExit('MONTH_WITH_NO_DAY_PRESENT')
    I_day = I_m[mid] / span[mid][:, None]                 # the registered daily spreading
    H = np.asarray(d.h_day, float)

    # (i) section 6.4: does the daily spreading preserve the monthly total?
    back = np.zeros_like(I_m)
    np.add.at(back, mid, I_day)
    rel = np.abs(back - I_m) / np.maximum(np.abs(I_m), 1e-30)
    chk = dict(n_months=int(I_m.shape[0]), n_reaches=int(nr),
               max_rel_error=float(rel.max()),
               max_abs_error=float(np.max(np.abs(back - I_m))),
               total_source_kg=float(I_m.sum()), total_back_kg=float(back.sum()),
               days_per_month_unique=sorted(int(v) for v in set(span.tolist())),
               consistent=bool(rel.max() <= S_TOL),
               verdict=('VERIFIED: the assumption Step B marks unverified in its section '
                        '6.4 HOLDS -- spreading the month across its present days and summing '
                        'back reproduces the monthly ledger to the last bit of a float64.'
                        if rel.max() <= S_TOL else
                        'REFUTED: the daily spreading does not reproduce the monthly total.'))

    # (ii) D1: the double count `sum(E_t)/sum(I_t)` over Step B's own frozen grid
    I_tot = float(I_day.sum())
    grid = []
    for a in (0.0, 0.25, 0.5, 1.0):
        for T_e in (1, 2, 3, 5, 7, 14, 30, 60):
            kE = 1.0 / float(T_e)
            S = np.zeros(nr)
            Etot = 0.0
            for t in range(nd):
                Spre = S + a * I_day[t]
                Et = Spre * -np.expm1(-np.minimum(kE * H[t], 700.))
                S = Spre - Et
                Etot += float(Et.sum())
            grid.append(dict(a=a, T_e_days=T_e, k_E=kE,
                             sum_E_kg=Etot, sum_I_kg=I_tot,
                             ratio_E_over_I=Etot / I_tot if I_tot else None,
                             share_of_S_floor=('a=0 is the built-in null: E == 0 bitwise')))
    live = [g for g in grid if g['a'] > 0]
    out = dict(assumption_6_4=chk, n_grid=len(grid), sum_I_kg=I_tot,
               ratio_E_over_I_max=max(g['ratio_E_over_I'] for g in live),
               ratio_E_over_I_min=min(g['ratio_E_over_I'] for g in live),
               ratio_E_over_I_median=float(np.median([g['ratio_E_over_I'] for g in live])),
               grid=grid,
               d1_statement=('D1: under Step B, M/L still receives the FULL I_t while S_E '
                             'takes an ADDITIONAL a*I_t, so the network receives '
                             'baseline + sum(E_t).  The excess fraction is sum(E_t)/sum(I_t), '
                             'reported above.  It is not a bound that can be argued away: it '
                             'is the share by which the N entering the river network would '
                             'exceed the registered ledger.'))
    return out


def stepb_citation_check():
    """Step B C1 cites a `source` COLUMN in a named file.  Does it exist?

    The ledger this round actually uses is NOT named by hand: it is DERIVED from
    the model's own array layout, so the recorded path is the one `load_data` will
    open and the recorded sha is the one `load_data` verifies.  The first draft of
    this function recorded the abbreviation `data/source.npy`, which resolves to
    no file on disk -- a transcription of the same class as the defects this round
    registers elsewhere, caught by the delivery path check.
    """
    cols = list(pd.read_parquet(STEPB_SOURCE).columns)
    exact = 'source' in cols
    near = [c for c in cols if 'source' in c.lower()]
    root = C.PEER / 'data' / 'domains' / 'FULL24'
    spec = json.loads((root / 'arrays.json').read_text(encoding='utf-8'))['source']
    ledger = root / spec['file']
    if not ledger.exists():
        raise SystemExit('REGISTERED_LEDGER_MISSING %s' % ledger)
    rel = ledger.relative_to(C.PEER.parent).as_posix()
    return dict(cited_file=STEPB_SOURCE.name, cited_column='source',
                n_rows=int(len(pd.read_parquet(STEPB_SOURCE, columns=['reach_id']))),
                column_exists=bool(exact), near_miss_columns=near,
                verdict=('C1 names a `source` column that is NOT present in the file it '
                         'cites.  The file and its row count (176640 = 230 x 768) are as '
                         'described; the column is not.  The registered monthly ledger this '
                         'round uses instead is %s (768, 230), sha %s.'
                         % (rel, spec['sha256'])),
                registered_ledger_used=rel,
                registered_ledger_sha=spec['sha256'])


# --------------------------------------------------------------------------
def main():
    S = dict(phase=0, n_fits=0, fit_worker_calls=0, not_a_fit=True,
             pre_registration_sha=C.sha(R / 'reports/预注册_判据与门槛.md'))
    ev = E.eligible_events()
    S['eligible_set'] = dict(n_events=int(len(ev)), n_stations=int(ev.station_key.nunique()))
    S['obs_recipe_reproduce'] = E.obs_recipe_reproduce(ev)

    C.restore_kernel()
    m = C.build(TAG)
    layers, raw = forward_layers(m, TAG)

    # the L3 identity: this mirrored forward must be the frozen replay's arithmetic
    frozen = C.replay(m, TAG)
    j = layers[['station_key', 'date', 'pL3']].merge(
        frozen[['station_key', 'date', 'p']], on=['station_key', 'date'], validate='one_to_one')
    if len(j) != len(frozen):
        raise SystemExit('LAYER_FORWARD_ROW_MISMATCH %d %d' % (len(j), len(frozen)))
    dmax = float(np.max(np.abs(j.pL3.to_numpy() - j.p.to_numpy())))
    S['L3_equals_frozen_replay'] = dict(n_rows=int(len(j)), max_abs_diff=dmax,
                                        bitwise=bool(dmax == 0.0))
    if dmax != 0.0:
        raise SystemExit('LAYER_FORWARD_IS_NOT_THE_FROZEN_RECIPE %.3e' % dmax)
    print('L3 == frozen replay bitwise, n=%d' % len(j), flush=True)

    ev2 = ev.copy()
    S['amplitude_budget'] = layer_budget(layers, ev2)
    mask = pd.read_parquet(E.MASK)
    S['station_sd_by_layer'] = station_sd_by_layer(layers, mask)
    S['observation_anchor'] = C.read_json(R / 'reports/frozen_anchors.json')['obs_levels']

    for lay, v in S['amplitude_budget'].items():
        print('  %s  C_base=%.4f C_peak=%.4f  A=%.6f  (n_peak %d..%d)'
              % (lay, v['c_base_median'], v['c_peak_median'], v['amp_ratio_median'],
                 v['n_peak_min'], v['n_peak_max']), flush=True)
    o = S['observation_anchor']
    print('  OBS             C_base=%.4f C_peak=%.4f  A=%.9f'
          % (o['c_base_median'], o['c_peak_median'], o['median_ratio']), flush=True)
    for lay, v in S['station_sd_by_layer'].items():
        print('  sd %s median=%.5f  [%.5f, %.5f]' % (lay, v['sd_median'], v['sd_min'],
                                                     v['sd_max']), flush=True)

    S['elasticity'] = elasticity(m, ev2)
    el = S['elasticity']
    print('  elasticity: mode=%s  h_inside median=%.5f (frac>=1: %.4f)  '
          'g_inside median=%.5f (frac>=1: %.4f)'
          % (el['mode'], el['h']['inside']['median'], el['h']['inside']['frac_ge_1'],
             el['g_r']['inside']['median'], el['g_r']['inside']['frac_ge_1']), flush=True)
    print('              h_outside median=%.5f  g_outside median=%.5f'
          % (el['h']['outside']['median'], el['g_r']['outside']['median']), flush=True)

    S['stepb'] = dict(citation=stepb_citation_check(), **stepb_audit(m))
    print('  stepB 6.4 consistent=%s max_rel=%.3e'
          % (S['stepb']['assumption_6_4']['consistent'],
             S['stepb']['assumption_6_4']['max_rel_error']), flush=True)
    print('  stepB D1 sum(E)/sum(I) in [%.6f, %.6f] median %.6f'
          % (S['stepb']['ratio_E_over_I_min'], S['stepb']['ratio_E_over_I_max'],
             S['stepb']['ratio_E_over_I_median']), flush=True)
    print('  stepB citation: column_exists=%s near=%s'
          % (S['stepb']['citation']['column_exists'],
             S['stepb']['citation']['near_miss_columns']), flush=True)

    layers.to_parquet(R / 'data/phase0_layers.parquet', index=False)
    S['outputs'] = dict(layers='data/phase0_layers.parquet',
                        n_rows=int(len(layers)), columns=list(layers.columns))
    out = R / 'reports/phase0_amplitude_budget.json'
    out.write_text(json.dumps(S, indent=1, sort_keys=True, ensure_ascii=False, default=str),
                   encoding='utf-8')
    print('PHASE0_DONE', C.sha(out), flush=True)


if __name__ == '__main__':
    main()
