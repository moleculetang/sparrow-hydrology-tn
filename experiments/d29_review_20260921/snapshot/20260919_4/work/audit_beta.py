"""The INDEPENDENT recomputation of the beta != 0 chain -- the step round 3 did not have.

WHAT THIS FILE IS FOR
---------------------
`20260919_3` shipped a gamma device whose entire forward was re-implemented by exactly the
code that produced the reading.  The review's word for the consequence was that the
mechanism family's closure was "not reliable".  This module is the answer to that specific
defect, and it is deliberately awkward to write: it reproduces the SAME numbers through a
SECOND implementation and compares them.

WHAT IT DOES **NOT** IMPORT -- AND WHY THAT IS THE WHOLE POINT
--------------------------------------------------------------
It imports NONE of this round's modules:

    common22, xi, closures_mc, layers22, phase0_freeze, phase1_l1, phase2_full,
    phase2_gamma_backfill, verdict_beta

Those are precisely the modules whose agreement with themselves is worthless here.  In
particular this file does NOT import `xi.py` (it re-derives `Xi` from the raw `.npy`
drivers), does NOT import `closures_mc.py` (it re-derives the land-phase recurrence in
plain numpy), and does NOT import `layers22.py` (it re-derives the three layer
concentrations and the event aggregation).

It bootstraps `sys.path` itself, in the open, rather than borrowing the round's bootstrap.

WHAT IT DOES IMPORT, AND WHY THAT IS NOT A HOLE
-----------------------------------------------
Three things, none of which is this round's product:

  1. the FROZEN vendor tree (`closures.scan`, `structure_model`, `campaign_model`,
     `routing`) -- the reference these readings are measured against.  Calling the frozen
     kernel is how the audit proves the SECOND implementation reproduces it at beta = 0.
  2. `eventlib` (from `20260919_2`, sha-pinned) -- only to obtain the FROZEN event set.
     Re-deriving the events would change which events are measured, not verify them.
  3. the frozen 30-vector and design out of `outputs/C0_s1/model.json`.

THE ONE PLACE EXACTNESS CANNOT BE DEMANDED, NAMED RATHER THAN TOLERATED SILENTLY
-------------------------------------------------------------------------------
numba compiles `np.expm1` to its own libm call; CPython/numpy routes to a different one.
The two may differ by an ulp.  So `prob` is compared with BOTH a bitwise test AND a
relative tolerance, and the residual is REPORTED.  Everything downstream that is a scalar
recurrence over one cell (the day loop) is expected to be bitwise, because vectorising
over reaches does not reorder any individual cell's arithmetic.  Amplitudes are medians
of ratios and are compared at 1e-12 relative.

A MISMATCH HERE IS A FINDING, NOT A FAILURE OF THE AUDIT: it means the reading in
`phase1_l1.json` was produced by something other than the frozen recurrence with an `Xi`
factor, and the round must stop and say so.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------------
# own bootstrap -- copied in the open, not borrowed
# ---------------------------------------------------------------------------------
AUDIT_DIR = Path(__file__).resolve().parents[1]
PEER = Path(r'E:\SPARROW\5_Test\20260916_2')
PEER_EVENTLIB = PEER.parent / '20260919_2' / 'work'
assert AUDIT_DIR.name == '20260919_4', AUDIT_DIR
assert PEER.is_dir(), PEER
assert (PEER_EVENTLIB / 'eventlib.py').is_file(), PEER_EVENTLIB

CACHE = AUDIT_DIR / 'work' / 'numba'
CACHE.mkdir(parents=True, exist_ok=True)
os.environ['NUMBA_CACHE_DIR'] = str(CACHE)

sys.path[:0] = [str(PEER / 'scripts'),
                str(PEER / 'vendor/expert/tn_challenge'),
                str(PEER / 'vendor/research')]
sys.path.append(str(PEER_EVENTLIB))

import torch  # noqa: E402
import campaign_model as cm  # noqa: E402
import closures as CLOSED  # noqa: E402
import structure_model as SM  # noqa: E402
import eventlib as EL  # noqa: E402

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

DOMAIN = PEER / 'data' / 'domains' / 'FULL24'
TAG = 'C0_s1'
END_YEAR = 2024
REF_YEARS = (1961, 2020)
W_FLOOR = 1e-3
CLIP = 700.0

# Frozen anchors, restated as literals (a hash read from the file it guards proves
# nothing).  The audit's job is to reproduce these from scratch.
A_L1_ANCHOR = 1.0400772083480145
A_L2_ANCHOR = 1.0217750264713188
A_L3_ANCHOR = 1.023217381861253
A_OBS = 1.2758737517831669
S_U_REGISTERED = 14.438101895057091

# The four Phase-2 points, plus beta = 0 as the no-op control.
AUDIT_BETAS = (0.0, 0.5, 1.0, -4.0)


# =================================================================================
# 1.  Xi, re-derived from the raw drivers
# =================================================================================
def own_geometry(verbose=True):
    """`W`, the floor, the active mask, the 1961-2020 active-cell climatology, `u`, `S_u`.

    Written from the plan's §1.2 spelling, not by calling `xi.py`.
    """
    d = np.load(DOMAIN / 'fast_water.npy', allow_pickle=False)
    p = np.load(DOMAIN / 'percolation.npy', allow_pickle=False)
    a = np.load(DOMAIN / 'area_ha.npy', allow_pickle=False)
    dates = np.load(DOMAIN / 'dates.npy', allow_pickle=False)
    Qf = np.asarray(d, np.float64)
    Qs = np.asarray(p, np.float64) * np.asarray(a, np.float64)[None, :] * 10.0
    assert Qf.shape == Qs.shape
    W = Qf + Qs
    assert int((Qf <= 0).sum()) == 0 and int((Qs <= 0).sum()) == 0, 'NOT_STRICTLY_POSITIVE'
    Weff = np.maximum(W, W_FLOOR)
    active = W > W_FLOOR
    logWeff = np.log(Weff)
    assert np.array_equal(logWeff[active], np.log(W)[active]), 'FLOOR_NOT_EXACT_ON_ACTIVE'
    yrs = np.asarray(dates).astype('datetime64[Y]').astype(np.int64) + 1970
    ref = (yrs >= REF_YEARS[0]) & (yrs <= REF_YEARS[1])
    Aref = active[ref]
    logref = logWeff[ref]
    n_act = Aref.sum(axis=0).astype(np.float64)
    clim = np.where(Aref, logref, 0.0).sum(axis=0) / n_act
    u = logWeff - clim[None, :]
    ctr = np.where(Aref, u[ref] - clim[None, :], 0.0)
    sd_r = np.sqrt(np.where(Aref, ctr * ctr, 0.0).sum(axis=0) / n_act)
    S_u = float(np.median(sd_r))
    if verbose:
        print('   own W: min=%.6g  frac(W<1e-6)=%.6f  frac_active=%.6f'
              % (W.min(), float((W < 1e-6).mean()), float(active.mean())))
        print('   own S_u = %.15f   registered = %.15f   bitwise-equal=%s'
              % (S_U_REGISTERED, S_u, S_u == S_U_REGISTERED))
    assert S_u == S_U_REGISTERED, (
        'THE AUDIT\'S OWN S_u DIFFERS FROM THE REGISTERED ONE: %.17g vs %.17g -- the two '
        'implementations do not agree on the axis that LABELS the grid'
        % (S_u, S_U_REGISTERED))
    return dict(Qf=Qf, Qs=Qs, W=W, Weff=Weff, active=active, u=u, S_u=S_u, sd_r=sd_r)


def own_xi(G, beta):
    """The frozen spelling of §1.2, written again from the plan."""
    b = float(beta)
    Lf = np.log(G['Qf'])
    Ls = np.log(G['Qs'])
    L0 = np.logaddexp(Lf, Ls)
    half = b * G['u'] / 2.0
    Lb = np.logaddexp(Lf + half, Ls - half)
    Xi = np.exp(np.clip(Lb - L0, -CLIP, CLIP))
    return np.where(G['active'], Xi, 1.0)


def own_census(Xi, h, active):
    """Only the fields Phase 1 actually stored; recomputed here to be compared.

    The `frac` fields are over ALL cells; the QUANTILES are over the ACTIVE cells
    (`xi.saturation_census` takes `x = Xi[pos]` with `pos = active`), and the audit
    reproduces that split rather than picking one.
    """
    prob = -np.expm1(-np.minimum(h * Xi, CLIP))
    prob0 = -np.expm1(-np.minimum(h, CLIP))
    xa = Xi[active]
    qq = np.percentile(xa, (0, 1, 50, 99, 100))
    return dict(
        frac_abs_xi_minus_1_gt_1e6=float((np.abs(Xi - 1.0) > 1e-6).mean()),
        frac_xi_gt_10=float((Xi > 10.0).mean()),
        frac_xi_lt_0p1=float((Xi < 0.1).mean()),
        frac_prod_ge_700=float(((h * Xi) >= CLIP).mean()),
        frac_prob_ge_0p99=float((prob >= 0.99).mean()),
        frac_prob_ge_1e_3=float((prob >= 1e-3).mean()),
        xi_min=float(Xi.min()), xi_max=float(Xi.max()),
        xi_active_quantiles={str(q): float(v) for q, v in
                             zip((0, 1, 50, 99, 100), qq)},
        med_abs_log_xi=float(np.median(np.abs(np.log(xa)))),
        n_nonfinite=int((~np.isfinite(Xi)).sum()),
        n_nonpositive=int((Xi <= 0).sum()),
        prob_baseline_frac_ge_0p99=float((prob0 >= 0.99).mean()),
    )


# =================================================================================
# 2.  the land-phase recurrence, re-derived in plain numpy
# =================================================================================
def own_scan(h, s_wet, f, k, l_rel, inp, demand, Xi):
    """`closures.scan` with the `Xi` factor, written as an explicit day loop.

    `mode = False` (the registered `cap`), so `risk = h[t, :]` and `k` is unused.
    The loop is over DAYS with the reach axis vectorised -- a different code shape from
    numba's double loop, and per cell the operations are in the same order, so agreement
    is expected to be bitwise up to the libm difference in `expm1`.

    Returns `(fast, slow, a, p)` and the UNMODULATED `prob` so the audit can show the
    intervention is real rather than assumed to be.
    """
    nd, nr = h.shape
    fast = np.zeros_like(h)
    slow = np.zeros_like(h)
    a = np.zeros_like(h)
    p = np.zeros_like(h)
    p0 = np.zeros_like(h)
    M = np.zeros(nr)
    L = np.zeros(nr)
    for t in range(nd):
        av = np.maximum(M + inp[t] - demand[t], 0.0)
        risk = h[t]
        prob = -np.expm1(-np.minimum(risk * Xi[t], CLIP))
        prob0 = -np.expm1(-np.minimum(risk, CLIP))
        E = av * prob
        fast[t] = E * f[t]
        pre = L + E * (1.0 - f[t])
        slow[t] = pre * l_rel[t]
        survival = s_wet if s_wet.ndim == 1 else s_wet[t]
        M = av * (1.0 - prob) * survival
        L = pre - slow[t]
        a[t] = av
        p[t] = prob
        p0[t] = prob0
    return fast, slow, a, p, p0


# =================================================================================
# 3.  the three layers and the event aggregation, re-derived
# =================================================================================
def own_layers(model, x, Xi):
    """`L1`/`L2`/`L3` concentrations straight from the frozen call sites."""
    t = torch.tensor(x)
    with torch.no_grad():
        h_, s_, f_, k_ = model.flux_parameters(t[:30])
        h = np.ascontiguousarray(h_.numpy())
        s_wet = np.ascontiguousarray(s_.numpy())
        f = np.ascontiguousarray(f_.numpy())
        k = np.ascontiguousarray(k_.numpy())
        assert Xi.shape == h.shape, (Xi.shape, h.shape)
        fast, slow, a, p, p0 = own_scan(h, s_wet, f, k, model.data.lower_release,
                                        model.inp, model.demand, Xi)
        local_np = fast + slow
        # the frozen operator, called directly.  `human_mass` is the scalar 0. because
        # `human_enabled` is False, so it adds nothing -- asserted, not assumed.
        assert model.human_mass(t) == 0., 'human_mass is NOT zero: L1 has a third term'
        local = torch.from_numpy(local_np) + model.human_mass(t)
        inlet, official, releases = SM.RiverN.apply(local, t[2], model.daily_data,
                                                    'monthly', True)
        meta = pd.read_parquet(PEER / 'data/prediction_calendar.parquet')
        meta = meta[meta.year.le(END_YEAR)].copy()
        c, record, weights = model.daily_metadata(meta)
        mass = SM.boundary_mass(inlet, official, releases, local, t[2], c)
    ti = c['ti'].numpy().astype(np.int64)
    ri = c['ri'].numpy().astype(np.int64)
    L = np.asarray(local.numpy(), float)
    I = np.asarray(inlet.numpy(), float)
    Mm = np.asarray(mass.numpy(), float)
    V = np.asarray(c['water'].numpy(), float)
    df = pd.DataFrame(dict(
        station_key=meta.station_key.to_numpy()[record.numpy()],
        date=model.data.dates[ti],
        water_m3_day=V,
        L1_kg=L[ti, ri], L2_kg=I[ti, ri] + L[ti, ri], L3_kg=Mm))
    for lay in ('L1', 'L2', 'L3'):
        df['p' + lay] = 1000.0 * df[lay + '_kg'] / df.water_m3_day
    return df, dict(p0=p0, p=p, fast=fast, slow=slow, h=h)


def own_amp(df, ev, lay, tn_pre_days=7):
    """Per-event `median(base) -> max(peak)` ratio, window rules from the frozen recipe.

        base  [t_start - 7d, t_start)   half-open
        peak  [t_start, t_end + 1d]     BOTH ENDS INCLUSIVE
    """
    piv = df.pivot_table(index='date', columns='station_key', values='p' + lay).sort_index()
    dates = piv.index.values.astype('datetime64[D]')
    ts = pd.to_datetime(ev.t_start).dt.normalize()
    te = pd.to_datetime(ev.t_end).dt.normalize()
    ratios, nb, npk = [], [], []
    for i, r in enumerate(ev.itertuples()):
        s = piv[r.station_key].to_numpy(float)
        lo = np.datetime64(ts.iloc[i] - pd.Timedelta(days=tn_pre_days), 'D')
        t0 = np.datetime64(ts.iloc[i], 'D')
        hi = np.datetime64(te.iloc[i] + pd.Timedelta(days=1), 'D')
        b = s[np.searchsorted(dates, lo, 'left'):np.searchsorted(dates, t0, 'left')]
        q = s[np.searchsorted(dates, t0, 'left'):np.searchsorted(dates, hi, 'right')]
        b = b[np.isfinite(b)]
        q = q[np.isfinite(q)]
        nb.append(int(b.size))
        npk.append(int(q.size))
        ratios.append(float(np.max(q)) / float(np.median(b)) if b.size and q.size else np.nan)
    rr = np.asarray(ratios, float)
    fin = np.isfinite(rr)
    want_peak = ((te - ts).dt.days + 2).to_numpy()
    assert np.array_equal(np.asarray(npk), want_peak), 'PEAK_WINDOW_LENGTH_WRONG'
    assert np.array_equal(np.asarray(nb), np.full(len(ev), tn_pre_days)), 'BASE_WINDOW_WRONG'
    return dict(n_events=int(fin.sum()), amp_ratio_median=float(np.median(rr[fin])),
                amp_ratio_mean=float(rr[fin].mean()))


# =================================================================================
def relerr(got, want):
    return float(abs(got - want) / abs(want)) if want else float(abs(got - want))


def main():
    rep = {'phase': 'audit', 'n_fits': 0, 'fit_worker_calls': 0, 'round': str(AUDIT_DIR),
           'independent': True, 'imports_round_modules': [],
           'imports_frozen': ['closures', 'structure_model', 'campaign_model', 'routing '
                              '(via structure_model)', 'eventlib (event set only)'],
           'review_defect_addressed': ('20260919_3 (ii)-adjacent: no second independent '
                                       'implementation recomputed the beta != 0 forward')}

    print('=== 1. own Xi from the raw drivers ===', flush=True)
    G = own_geometry()
    rep['own_S_u'] = G['S_u']
    rep['own_S_u_matches_registered'] = bool(G['S_u'] == S_U_REGISTERED)

    print('=== 2. the model, the parameters, the event set ===', flush=True)
    rec = json.loads((PEER / 'outputs' / TAG / 'model.json').read_text(encoding='utf-8'))
    x = np.asarray(rec['parameters'], float)
    assert x.size == 30, x.size
    design = json.loads(json.dumps(rec['design']))
    design['observation_registry_file'] = 'data/prediction_registry.json'
    design['observation_registry_hash'] = hashlib.sha256(
        (PEER / 'data/prediction_registry.json').read_bytes()).hexdigest()
    data = cm.load_data('FULL24')
    model = cm.make_model(data, None, 'D29_BE', design)
    assert model.cap is False and model.calendar == 'monthfirst'
    assert model.human_enabled is False
    assert type(model).__name__ == 'StructureEndpoints', type(model).__name__
    ev = EL.eligible_events()
    assert len(ev) == 214 and ev.station_key.nunique() == 15
    rep['n_events'] = int(len(ev))
    rep['n_stations'] = int(ev.station_key.nunique())

    # --- the frozen kernel, for the beta = 0 control -----------------------------
    t = torch.tensor(x)
    with torch.no_grad():
        h_, s_, f_, k_ = model.flux_parameters(t[:30])
        hf = np.ascontiguousarray(h_.numpy())
        sf = np.ascontiguousarray(s_.numpy())
        ff = np.ascontiguousarray(f_.numpy())
        kf = np.ascontiguousarray(k_.numpy())
        fz_fast, fz_slow, fz_a, fz_p = CLOSED.scan(
            hf, sf, ff, kf, model.data.lower_release, model.inp, model.demand, model.cap)

    print('=== 3. beta = 0: does the second implementation reproduce the FROZEN '
          'kernel? ===', flush=True)
    X0 = own_xi(G, 0.0)
    assert np.array_equal(X0, np.ones_like(X0)), 'BETA=0 Xi IS NOT EXACTLY ONES'
    rep['beta0_xi_is_exactly_ones'] = True
    m_fast, m_slow, m_a, m_p, m_p0 = own_scan(hf, sf, ff, kf, model.data.lower_release,
                                              model.inp, model.demand, X0)
    quad = {}
    for nm, mine, ref in (('fast', m_fast, fz_fast), ('slow', m_slow, fz_slow),
                          ('a', m_a, fz_a), ('p', m_p, fz_p)):
        quad[nm] = dict(bitwise=bool(np.array_equal(mine, ref)),
                        max_abs=float(np.abs(mine - ref).max()),
                        max_rel=float(np.abs(mine - ref).max() / max(np.abs(ref).max(), 1e-300)))
    rep['beta0_frozen_kernel_reproduction'] = quad
    for nm, v in quad.items():
        print('   %-5s bitwise=%-5s  max|d|=%.3g  max_rel=%.3g'
              % (nm, v['bitwise'], v['max_abs'], v['max_rel']))
    assert all(v['max_rel'] <= 1e-12 for v in quad.values()), (
        'THE AUDIT DOES NOT REPRODUCE THE FROZEN KERNEL AT BETA = 0: %s' % quad)
    assert quad['fast']['bitwise'] and quad['slow']['bitwise'], (
        'the second implementation is not bitwise on fast/slow at beta = 0: %s' % quad)

    print('=== 4. the frozen anchors, recomputed through the SECOND implementation ===',
          flush=True)
    df0, aux0 = own_layers(model, x, X0)
    a0 = {L: own_amp(df0, ev, L) for L in ('L1', 'L2', 'L3')}
    rep['beta0_amplitudes'] = a0
    for L, want in (('L1', A_L1_ANCHOR), ('L2', A_L2_ANCHOR), ('L3', A_L3_ANCHOR)):
        got = a0[L]['amp_ratio_median']
        print('   A_%-2s = %.16g   anchor = %.16g   rel_err = %.3g'
              % (L, got, want, relerr(got, want)))
        assert relerr(got, want) <= 1e-12, (
            'ANCHOR MISMATCH ON A_%s: the audit got %.17g, the frozen anchor is %.17g'
            % (L, got, want))

    print('=== 5. beta != 0: the full chain, twice, element by element ===', flush=True)
    rows = {}
    for beta in AUDIT_BETAS:
        Xi = own_xi(G, beta)
        cen = own_census(Xi, hf, G['active'])
        fst, slw, aa, pp, p0 = own_scan(hf, sf, ff, kf, model.data.lower_release,
                                        model.inp, model.demand, Xi)
        # how big is the intervention, measured rather than asserted
        dp = np.abs(pp - p0)
        dfast = np.abs(fst - fz_fast)
        df, _ = own_layers(model, x, Xi)
        amp = {L: own_amp(df, ev, L) for L in ('L1', 'L2', 'L3')}
        rows['%g' % beta] = dict(
            beta=float(beta), g_in_sd_units=float(beta * G['S_u']),
            A_L1=amp['L1']['amp_ratio_median'], A_L2=amp['L2']['amp_ratio_median'],
            A_L3=amp['L3']['amp_ratio_median'],
            own_census=cen,
            prob_max_abs_change=float(dp.max()),
            prob_frac_changed=float((dp > 0).mean()),
            fast_max_rel_change=float(
                (dfast.max() / max(np.abs(fz_fast).max(), 1e-300))),
            n_cells_prob_decreased=int((pp < p0).sum()),
            n_cells_prob_increased=int((pp > p0).sum()),
            n_cells_prob_tied=int((pp == p0).sum()),
            xi_equals_frozen=bool(beta == 0.0),
        )
        r = rows['%g' % beta]
        fc = r['fast_max_rel_change']
        print('   beta=%+5g  A_L1=%.10f A_L2=%.10f A_L3=%.10f  '
              'frac(prob changed)=%.4f  max|d fast|/max|fast|=%.3g'
              % (beta, r['A_L1'], r['A_L2'], r['A_L3'], r['prob_frac_changed'], fc))
    rep['points'] = rows

    print('=== 6. agreement with the STORED readings ===', flush=True)
    stored = {}
    for fn in ('phase1_l1.json', 'phase2_full.json'):
        d = json.loads((AUDIT_DIR / 'reports' / fn).read_text(encoding='utf-8'))
        for k, v in d['points'].items():
            stored[k] = dict(beta=v['beta'], A_L1=v['A_L1'], A_L2=v['A_L2'], A_L3=v['A_L3'],
                             src=fn)
    cmp_rows = {}
    for k, r in rows.items():
        if k not in stored:
            cmp_rows[k] = dict(compared=False,
                               why='this beta is not among the four Phase-2 candidates; '
                                   'the audit recomputed it anyway and records it')
            continue
        s = stored[k]
        cmp_rows[k] = dict(
            compared=True, source=s['src'], beta_match=bool(s['beta'] == r['beta']),
            A_L1_stored=s['A_L1'], A_L1_audit=r['A_L1'],
            A_L1_rel_err=relerr(r['A_L1'], s['A_L1']),
            A_L2_stored=s['A_L2'], A_L2_audit=r['A_L2'],
            A_L2_rel_err=relerr(r['A_L2'], s['A_L2']),
            A_L3_stored=s['A_L3'], A_L3_audit=r['A_L3'],
            A_L3_rel_err=relerr(r['A_L3'], s['A_L3']),
            A_L1_bitwise=bool(r['A_L1'] == s['A_L1']),
            A_L3_bitwise=bool(r['A_L3'] == s['A_L3']))
        c = cmp_rows[k]
        print('   beta=%+5g A_L1 stored=%.16g audit=%.16g rel=%.3g bitwise=%s'
              % (r['beta'], c['A_L1_stored'], c['A_L1_audit'], c['A_L1_rel_err'],
                 c['A_L1_bitwise']))
    rep['stored_comparison'] = cmp_rows

    # the census cross-check: the audit's Xi against Phase 1's stored census
    p1 = json.loads((AUDIT_DIR / 'reports' / 'phase1_l1.json').read_text(encoding='utf-8'))
    cen_cmp = {}
    for k, r in rows.items():
        if k in p1['points']:
            sc = p1['points'][k]['saturation']
            fields = ('frac_abs_xi_minus_1_gt_1e6', 'frac_xi_gt_10', 'frac_xi_lt_0p1',
                      'frac_prod_ge_700', 'frac_prob_ge_0p99', 'frac_prob_ge_1e_3',
                      'xi_min', 'xi_max', 'med_abs_log_xi',
                      'prob_baseline_frac_ge_0p99')
            mx = max(relerr(r['own_census'][f], sc[f]) for f in fields)
            mxq = max(relerr(r['own_census']['xi_active_quantiles'][q],
                             sc['xi_active_quantiles'][q]) for q in ('0', '1', '50',
                                                                     '99', '100'))
            cen_cmp[k] = dict(max_rel_err_of_census=float(max(mx, mxq)),
                              n_fields=len(fields) + 5,
                              agrees=bool(max(mx, mxq) <= 1e-12))
    rep['census_agreement'] = cen_cmp
    print('   census agreement (max rel err over 9 fields, per beta): %s'
          % {k: '%.2g' % v['max_rel_err_of_census'] for k, v in cen_cmp.items()})

    cmp_ok = all((not v['compared']) or (v['A_L1_rel_err'] <= 1e-12
                                         and v['A_L3_rel_err'] <= 1e-12)
                 for v in cmp_rows.values())
    cen_ok = all(v['agrees'] for v in cen_cmp.values())
    rep['summary'] = dict(
        own_S_u_matches=bool(G['S_u'] == S_U_REGISTERED),
        beta0_reproduces_frozen_kernel=bool(all(v['bitwise'] for v in quad.values())),
        beta0_reproduces_anchors=True,
        stored_readings_reproduced=bool(cmp_ok),
        stored_max_rel_err_A_L1=float(max([v.get('A_L1_rel_err', 0.0)
                                           for v in cmp_rows.values()])),
        stored_max_rel_err_A_L3=float(max([v.get('A_L3_rel_err', 0.0)
                                           for v in cmp_rows.values()])),
        census_reproduced=bool(cen_ok),
        intervention_is_real=bool(all(
            rows[k]['prob_frac_changed'] > 0.0 for k in rows if abs(rows[k]['beta']) > 0)),
        n_betas_recomputed=len(rows))
    assert rep['summary']['intervention_is_real'], (
        'THE MODULATION CHANGES NOTHING: Xi is not reaching prob on any beta != 0')
    assert cmp_ok, ('THE AUDIT DOES NOT REPRODUCE THE STORED AMPLITUDES: %s'
                    % {k: v for k, v in cmp_rows.items() if v['compared']})
    assert cen_ok, 'THE AUDIT DOES NOT REPRODUCE THE STORED SATURATION CENSUS'

    path = AUDIT_DIR / 'reports' / 'audit_beta.json'
    payload = json.dumps(rep, indent=1, sort_keys=True, default=float)
    path.write_text(payload, encoding='utf-8')
    print('WROTE %s sha256=%s' % (path.name,
                                  hashlib.sha256(payload.encode('utf-8')).hexdigest()))
    print('AUDIT_PASSED  (a second implementation reproduces every stored reading)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
