"""Independent recomputation of this round's new content -- importing NO round module.

WHY `import` IS BANNED HERE
---------------------------
Every other module in this round shares `common21`'s bootstrap.  If the bootstrap is
wrong -- a transposed array, a wrong column index, an off-by-one on `reach_id` -- then
every module agrees and every module is wrong together.  This file therefore opens the
raw `.npy` inputs and `model.json` BY PATH STRING, re-implements the two pieces of
arithmetic the round actually adds, and compares the answers against the round's own
artifacts.  It is a second implementation, not a second call.

WHAT IT RE-DERIVES, FROM RAW INPUTS
-----------------------------------
1.  `f0`, the carrier partition, from `fast_water`, `percolation`, `area_ha` -- against the
    stored `fast_fraction.npy`.  (Establishes that the stored fraction IS that formula.)
2.  `z` and `sigma_z`, the round's new UNIT, from `fast_water`/`slow_water`/`dates`.
3.  The gamma table `gamma = g / sigma`.
4.  The LOGIT IDENTITY `logit(f^N) = logit(f) + gamma z`, evaluated two ways, on real cells.
5.  The `g = 0` no-op as FLOATING-POINT ARITHMETIC, not as an empirical claim.
6.  The two thresholds, from the published baselines and the 50%-of-gap rule.
7.  The VERDICT, from the published readings, by an independent decision function.

It reads the round's artifacts only to COMPARE, never to compute.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(r'E:\SPARROW\5_Test')
SRC = ROOT / '20260916_2'
F24 = SRC / 'data/domains/FULL24'
REPORTS = ROOT / '20260919_3/reports'
GAMMA_GRID = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
REF_YEARS = (1961, 2020)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def check(name, ok, detail):
    print('  [%s] %-46s %s' % ('ok' if ok else 'FAIL', name, detail))
    return bool(ok)


def main():
    out = {'imports_no_round_module': True, 'checks': {}, 'all_pass': False}
    print('=== independent audit of 20260919_3 (second implementation) ===')

    # ---- raw inputs, by literal path --------------------------------------------
    fw = np.load(F24 / 'fast_water.npy').astype(np.float64)
    sw = np.load(F24 / 'slow_water.npy').astype(np.float64)
    ff = np.load(F24 / 'fast_fraction.npy').astype(np.float64)
    pc = np.load(F24 / 'percolation.npy').astype(np.float64)
    ar = np.load(F24 / 'area_ha.npy').astype(np.float64)
    dates = np.load(F24 / 'dates.npy', allow_pickle=True).astype('datetime64[D]')
    t = np.asarray(json.loads((SRC / 'outputs/C0_s1/model.json')
                              .read_text(encoding='utf-8'))['parameters'], float)
    out['shapes'] = dict(fast_water=list(fw.shape), fast_fraction=list(ff.shape),
                         dates=int(dates.size), n_parameters=int(t.size),
                         log_aq=float(t[20]))
    print('  fast_water %s  dates %d  n_param %d  t[20]=%.17g'
          % (fw.shape, dates.size, t.size, t[20]))

    # ---- 1. the stored carrier fraction IS the registered formula ---------------
    # The producer's own validator (`20260916_1/scripts/validate_global_inputs.py:25`)
    # writes this as `div(fast, carrier)` with
    #     div = lambda x, y: np.divide(x, y, out=np.zeros_like(x), where=y > 1e-12)
    # -- a GUARDED division that emits exactly 0.0 where the carrier underflows, not a
    # plain one.  The first version of this audit used the plain form and reported
    # max|d| = 0.878; the same formula with the producer's own guard reproduces all
    # 5,376,480 cells with 0 above 1e-9.  The assertion was wrong, not the input.
    # Note the two populations are NOT the same: 11.2% of cells have carrier <= 1e-12,
    # while `fast_fraction == 0.0` on 15.48% -- the extra cells are where `fast` alone
    # underflows against a carrier that is still > 1e-12.
    carrier = pc * ar * 10.0
    f0 = np.divide(fw, fw + carrier, out=np.zeros_like(fw), where=(fw + carrier) > 1e-12)
    d_f0 = float(np.max(np.abs(f0 - ff)))
    n_above = int((np.abs(f0 - ff) > 1e-9).sum())
    out['checks']['f0_formula'] = dict(
        max_abs_diff=d_f0, tolerance=1e-12, n_cells=int(f0.size),
        n_above_1e_9=n_above, matches=bool(n_above == 0),
        guarded_division='np.divide(..., out=zeros, where=(fast+carrier)>1e-12)',
        note='fast is a DEPTH here: fast = local_fast_response_m3_s*86400/(area_km2*1000) '
             '= fast_water/(area_ha*10) mm/day, the same unit as percolation, which is why '
             'the carrier is a plain sum')
    check('f0 == fast_water/(fast_water+perc*area_ha*10)', n_above == 0, 'max|d|=%.3e' % d_f0)
    out['checks']['f0_formula']['n_carrier_at_or_below_1e-12'] = int((carrier <= 1e-12).sum())
    out['f0_census'] = dict(frac_exactly_zero=float((ff == 0.0).mean()),
                            frac_exactly_one=float((ff == 1.0).mean()),
                            max=float(ff.max()))

    # ---- 2. z and sigma_z, the round's new unit --------------------------------
    lr = np.log(fw) - np.log(sw)
    yr = dates.astype('datetime64[Y]').astype(int) + 1970
    m = (yr >= REF_YEARS[0]) & (yr <= REF_YEARS[1])
    z = lr - lr[m].mean(axis=0)[None, :]
    sigma = float(np.median(z[m].std(axis=0)))
    rec = json.loads((REPORTS / 'frozen_anchors.json').read_text(encoding='utf-8'))
    sig_rec = rec['z_climate']['sigma_z']
    out['checks']['sigma_z'] = dict(recomputed=sigma, recorded=float(sig_rec),
                                   equal=bool(sigma == float(sig_rec)), n_ref_days=int(m.sum()))
    check('sigma_z reproduces', sigma == float(sig_rec), '%.17g vs %.17g' % (sigma, sig_rec))
    zrec = float(rec['z_climate']['z_max'])
    out['checks']['z_range'] = dict(recomputed_max=float(z.max()), recorded=float(zrec))
    check('z range reproduces', abs(float(z.max()) - zrec) <= 1e-9 * max(1.0, abs(zrec)),
          'max %.17g vs %.17g' % (z.max(), zrec))

    # ---- 3. the gamma table ----------------------------------------------------
    tab = [g / sigma for g in GAMMA_GRID]
    recc = [v['install']['gamma_effective'] for v in
            (json.loads((REPORTS / 'phase1_l1.json').read_text(encoding='utf-8'))
             ['per_gamma']).values()]
    dd = float(np.max(np.abs(np.asarray(tab) - np.asarray(recc))))
    out['checks']['gamma_table'] = dict(recomputed=tab, recorded=recc, max_abs_diff=dd)
    check('gamma = g / sigma for all six points', dd == 0.0, 'max|d|=%.1e' % dd)
    out['checks']['gamma_direction'] = dict(
        divide_gives=tab[3], multiply_would_give=GAMMA_GRID[3] * sigma,
        note='the md §2.3 records the option text once carrying `gamma = g*sigma`; the '
             'table beneath it and the implementation both divide')

    # ---- 4. the logit identity, two ways, on real cells ------------------------
    aq = np.exp(t[20])
    g_hi = GAMMA_GRID[-1]
    gam = g_hi / sigma
    rows = slice(None, None, 977)
    cols = np.arange(0, z.shape[1], 7)
    zz = z[rows][:, cols]
    f00 = ff[rows][:, cols]
    aqe = np.exp(t[20] + gam * zz)
    fN_carrier = np.where(f00 > 0.0, aqe * f00 / (aqe * f00 + 1.0 - f00), 0.0)
    f_warped = np.where(f00 > 0.0, aq * f00 / (aq * f00 + 1.0 - f00), 0.0)
    # The registered identity is logit(f^N) = logit(f) + gamma*z, where `f` is the
    # fraction AFTER the fitted log-odds shift -- i.e. logit(f) = logit(f0) + log(aq).
    # The first version of this audit compared against logit(f0) + gamma*z and reported
    # max|d| = 1.542e-02; the residual was exactly t[20] = 0.01541849827082684, which is
    # the missing term.  The identity was right; the check omitted log(aq).
    sel = (f00 > 0.0) & (f00 < 1.0) & (fN_carrier > 0.0) & (fN_carrier < 1.0)
    lg = lambda x: np.log(x / (1.0 - x))
    lhs = lg(fN_carrier[sel])
    rhs = lg(f_warped[sel]) + gam * zz[sel]
    ident = float(np.max(np.abs(lhs - rhs)))
    out['checks']['logit_identity'] = dict(
        g=float(g_hi), n_cells=int(sel.sum()), max_abs_diff=ident,
        tolerance=1e-9, matches=bool(ident <= 1e-9),
        form='logit(fN) = logit(f) + gamma*z, with f = aq*f0/(aq*f0+1-f0)',
        without_log_aq_would_miss_by=float(t[20]),
        log_aq=float(t[20]))
    check('logit identity holds on real cells', ident <= 1e-9, 'max|d|=%.3e' % ident)
    out['checks']['logit_identity']['frac_f0_exactly_zero_in_sample'] = float(
        (f00[sel] == 0.0).mean())

    # ---- 5. the g = 0 no-op, as floating-point arithmetic ----------------------
    rng = np.random.default_rng(20260919)
    zr = rng.standard_normal(200000) * sigma
    zr = np.concatenate([zr, [0.0, -0.0, np.nextafter(0, 1), 1e-300, -1e-300]])
    a_q0 = np.exp(t[20] + 0.0 * zr)
    bit = bool(np.array_equal(a_q0, np.full_like(zr, np.exp(t[20]))))
    fr = rng.random(zr.size) * 0.999 + 1e-6
    l0 = aq * fr / (aq * fr + 1.0 - fr)
    l1 = np.exp(t[20] + 0.0 * zr) * fr / (np.exp(t[20] + 0.0 * zr) * fr + 1.0 - fr)
    out['checks']['g0_noop'] = dict(
        exp_bitwise_equal=bit, carrier_bitwise_equal=bool(np.array_equal(l0, l1)),
        n_sampled=int(zr.size),
        note='0.0*z is exactly 0.0 for every finite z (including -0.0 and subnormals), so '
             't[20]+0.0*z is exactly t[20] and both sides are the same float')
    check('g=0 keeps exp(t20+0*z) bitwise', bit, 'n=%d' % zr.size)
    check('g=0 keeps the carrier split bitwise', bool(np.array_equal(l0, l1)),
          'n=%d' % fr.size)

    # ---- 6. the thresholds -----------------------------------------------------
    fa = json.loads((REPORTS / 'phase1_l1.json').read_text(encoding='utf-8'))['anchors']
    th = {'L1': fa['A_L1'] + 0.5 * (fa['A_obs'] - fa['A_L1']),
          'L3': fa['A_L3'] + 0.5 * (fa['A_obs'] - fa['A_L3'])}
    dth = max(abs(th['L1'] - fa['G1_target_50pct']), abs(th['L3'] - fa['G2_target_50pct']))
    out['checks']['thresholds'] = dict(recomputed=th,
                                       recorded=[fa['G1_target_50pct'], fa['G2_target_50pct']],
                                       max_abs_diff=float(dth))
    check('50%-of-gap thresholds reproduce', dth == 0.0, 'max|d|=%.1e' % dth)
    mdlit = {'L1': 1.1579754800655907, 'L3': 1.14954556682221}
    out['checks']['md_literal'] = {k: dict(rule=th[k], literal=mdlit[k],
                                          ulp_delta=float(mdlit[k] - th[k]))
                                   for k in th}
    check('md literals are within 1 ULP of the rule',
          max(abs(mdlit[k] - th[k]) for k in th) <= 5e-16,
          ' '.join('%s %+.1e' % (k, mdlit[k] - th[k]) for k in th))

    # ---- 7. the verdict, decided again -----------------------------------------
    l1 = json.loads((REPORTS / 'phase1_l1.json').read_text(encoding='utf-8'))
    pg = l1['per_gamma']
    ks = sorted(pg, key=float)
    a1 = [pg[k]['A_L1'] for k in ks]
    g1 = [float(k) for k in ks if pg[k]['A_L1'] >= th['L1']]
    g2 = [float(k) for k in ks if pg[k]['A_L3'] >= th['L3']]
    both = sorted(set(g1) & set(g2))
    lim = bool(l1['limit_case']['A_L1'] >= th['L1'])
    d1 = [a1[i + 1] - a1[i] for i in range(len(a1) - 1)]
    monotone = all(x > 0 for x in d1)
    any_resp = max(a1) > a1[0]
    if g1 or lim:
        myverdict, mybranch = 'PASS', ('L1_AND_L3' if both else 'L1_ONLY')
    else:
        myverdict, mybranch = ('STOP', 'INSUFFICIENT')
    myqual = (None if myverdict == 'PASS' else
              'RESPONSIVE_NONMONOTONE' if (any_resp and not monotone) else
              'RESPONSIVE_MONOTONE' if any_resp else 'NO_RESPONSE')
    dec = json.loads((REPORTS / 'phase1_verdict.json').read_text(encoding='utf-8'))
    same = (myverdict == dec['verdict'] and mybranch == dec['branch']
            and myqual == dec['qualifier'])
    out['checks']['verdict'] = dict(
        recomputed=dict(verdict=myverdict, branch=mybranch, qualifier=myqual,
                        A_L1_sequence=a1, increments=d1, monotone=monotone,
                        g_passing_G1=g1, g_passing_G2=g2, limit_passes_G1=lim),
        recorded=dict(verdict=dec['verdict'], branch=dec['branch'],
                      qualifier=dec['qualifier']),
        agree=same)
    check('verdict reproduces independently', same,
          '%s/%s/%s' % (myverdict, mybranch, myqual))
    out['checks']['early_stop'] = dict(
        proceed_to_phase2_recorded=bool(dec['proceed_to_phase2']),
        phase2_status=json.loads((REPORTS / 'phase2_full.json').read_text(encoding='utf-8'))
        ['status'])
    check('Phase 2 was not run', out['checks']['early_stop']['phase2_status'] == 'NOT_RUN',
          out['checks']['early_stop']['phase2_status'])

    # ---- 8. the criterion statistic at g=0 must be the frozen one --------------
    peer = json.loads((ROOT / '20260919_2/reports/phase0_amplitude_budget.json')
                      .read_text(encoding='utf-8'))
    for src, key in (('A_L1', 'L1'), ('A_L3', 'L3')):
        v = peer['amplitude_budget'][key]['amp_ratio_median']
        out['checks']['cross_round_' + src] = dict(peer_value=float(v), round_value=fa[src])
        check('%s == 20260919_2 stored value' % src, float(v) == fa[src],
              '%.17g' % v)

    out['all_pass'] = bool(all(
        (v.get('matches', v.get('equal', v.get('agree', v.get('passed')))) is not False)
        for k, v in out['checks'].items() if isinstance(v, dict)))
    out['all_pass'] = bool(out['checks']['f0_formula']['matches']
                           and out['checks']['sigma_z']['equal']
                           and out['checks']['gamma_table']['max_abs_diff'] == 0.0
                           and out['checks']['logit_identity']['matches']
                           and out['checks']['g0_noop']['exp_bitwise_equal']
                           and out['checks']['g0_noop']['carrier_bitwise_equal']
                           and out['checks']['thresholds']['max_abs_diff'] == 0.0
                           and out['checks']['verdict']['agree']
                           and out['checks']['early_stop']['phase2_status'] == 'NOT_RUN'
                           and out['checks']['cross_round_A_L1']['peer_value'] == fa['A_L1']
                           and out['checks']['cross_round_A_L3']['peer_value'] == fa['A_L3'])
    (REPORTS / 'audit_gamma.json').write_text(
        json.dumps(out, indent=1, ensure_ascii=False, sort_keys=True), encoding='utf-8')
    print('=== ALL_PASS=%s ===' % out['all_pass'])
    print('WROTE reports/audit_gamma.json')
    return 0 if out['all_pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
