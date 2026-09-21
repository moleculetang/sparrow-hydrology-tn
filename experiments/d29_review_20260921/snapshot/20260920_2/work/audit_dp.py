"""`20260920_1` -- the independent recomputation.  ZERO FORWARDS.  ZERO FITS.

WHAT "INDEPENDENT" MEANS HERE, AND WHAT IT DOES NOT
---------------------------------------------------
This file imports NO round-`20260920_1` module: not `common24`, not `dp_kernel`, not
`closures_dp`, not `layers24`, not `phase0_gates`, not `phase1_arms`, not
`level_variance`.  Everything it needs is rebuilt from the READ-ONLY PEER tree
(`20260916_2`, `20260828_38`, `20260919_1`, `20260918_1`) plus its own numpy.  It reads
this round's JSON artifacts as DATA and re-derives their numbers, which is the only way a
round's own report can be checked by something other than itself.

It independently recomputes, from the definition:

  1. THE THREE PATHWAYS AND THE TWO VOLUMES.  `Qf/Qp/Qs/Qu` are rebuilt in mm/day with
     the file's own `area_ha`; `V_u` and `V_s` are rebuilt at the instant outflow begins
     (`S_post + Q_out`), with the LOWER STORAGE re-read straight from the producer
     parquet and its date/reach alignment re-proven here.  `V_s` is then rebuilt a SECOND
     way -- by inverting `lower_release` -- which never touches the parquet: a calendar
     slip or a reach permutation cannot survive both routes.
  2. THE CLOSURE AND THE CONCENTRATION FORM.  `g`, `phi_f`, `g_s` are recomputed from
     `x = Q/V`, and `x_u <= 1` is checked globally rather than taken from the arm table.
     Both candidate closures are computed and their difference is reported (P4).
  3. ASSERTION N1, AND THIS IS THE ONE THAT MATTERS.  The round's whole claim to be "the
     same model with one layer replaced" rests on the DP kernel REDUCING BITWISE to the
     frozen kernel when `g_u := p_frozen`, `phi_f := f_frozen`, `g_s := l_frozen` are
     injected.  The audit rebuilds BOTH spellings -- the DP one from `a*g_u*phi_f`, the
     frozen one from `E*f` -- in ONE day-loop, and asserts `np.array_equal` on all seven
     channels.  Nothing in this round is evidenced if this fails.
  4. THE MASS LEDGER, rebuilt from the frozen recurrence's own formula rather than read
     off the producer, reported as `local_balance_max_kg` against the 1e-6 kg tolerance.
  5. THE SECTION 2.6 P1 IDENTITY, re-derived rather than quoted: under the chosen linear
     closure `1 - g_u` must equal `S_post/V_u`, the retained water share.  `verdict.json`
     cites this identity as the reason P1 is falsified, so it is checked here numerically
     instead of being taken from the round's own prose.
  6. LEVEL / VARIANCE / G5b, recomputed from the delivered `daily_arms.parquet` with the
     audit's own NSE and recentring, against `level_variance.json`.
  7. N12 AS A SOURCE CHECK: the round's own source files are read as TEXT and searched
     for the naive exponential spelling, so the assertion does not depend on the round's
     own scanner being right.

SCOPE LIMITS, STATED RATHER THAN HIDDEN
---------------------------------------
* THE ROUTING AND THE STATION AGGREGATION ARE SHARED WITH THE PRODUCER.  Re-deriving
  `RiverN`/`boundary_mass` would be a second implementation of frozen peer code, not
  evidence about this round.  Every LAND-PHASE claim is audited; the downstream chain is
  not, and that is the same limitation round 5's `audit_beta.py` declared for itself.
* THE EVENT STATISTIC `A_L1` IS NOT RECOMPUTED.  It is built from a DENSE 169,476-row
  forward frame, and the delivered `daily_arms.parquet` is the 12,152-row ELIGIBLE grid.
  `eventlib.build_event_table` asserts `n_base == 7` and `n_peak == window + 2` on EVERY
  event, which the eligible grid cannot satisfy.  So the one number that carries G1 cannot
  be independently re-derived from what this round delivered -- a real gap in the
  deliverable set, registered here rather than papered over.
* `s_M` IS TAKEN FROM THE PEER'S OWN `flux_parameters`, not rebuilt.  Rebuilding it would
  be a second implementation of frozen peer code.
* `inp` / `demand` are the peer model's own objects, as in every arm.
* THE TAG AND TRANSPORT CHANNEL GROUPS OF N1 ARE NOT RE-DERIVED HERE.  They live in
  `closures_dp`, which the audit may not import, and the transport pair is the same `scan`
  object in any case.  `phase0_n1.json` is read as data and its counts restated with that
  restatement labelled as one.
* THE 1.15 GB PRODUCER PARQUET IS READ ONCE, FOR FOUR COLUMNS, to rebuild `V_s` route
  (a).  The two algebraic routes would both survive a corruption present in the cache
  itself; what they cover is a calendar slip or a reach-axis permutation, which is the
  failure they exist for.
"""
import ast
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROUND = HERE.parent
PEER = ROUND.parent / '20260916_2'
UP = ROUND.parent
sys.path.insert(0, str(PEER / 'scripts'))
sys.path.insert(0, str(PEER / 'vendor' / 'research'))

import campaign_model as cm                                          # noqa: E402
from closures import scan as frozen_scan                             # noqa: E402

OUT = ROUND / 'reports'
TAG = 'C0_s1'
CLIP = 700.0
MM_PER_HA = 10.0
START_YEAR, END_YEAR = 2021, 2024
BASELINE_NSE = 0.7029748157444711
BASELINE_MEAN_C = 2.8234476727379607

# Registered constants, hard-coded HERE so that the audit verifies them rather than
# inheriting them from a module that could have moved in the same commit.
MASK = UP / '20260919_1' / 'data' / 'phase1_eligible_mask.parquet'
PANEL = UP / '20260918_1' / 'reports' / 'ammonia_TN_quality_only.parquet'
MASK_SHA = '872280128ed6261ca936544379d54e63800497e0ee66625bf1cf5b83c310af41'
PANEL_SHA = '7ed9e6179705affc00494fafb2119ea2c656b32e661c44fee7dd5fd162d56a18'
HYDRO_PARQUET = UP / '20260828_38' / 'outputs' / 'tn_hydrology_reach_daily.parquet'
R4 = UP / '20260919_4'
R5 = UP / '20260919_5'
N12_WATCH = ('dp_kernel.py', 'closures_dp.py', 'common24.py', 'layers24.py',
             'phase0_gates.py', 'phase1_arms.py')
N12_NAIVE = ('1 - np.exp(', '1 - exp(', '1.0 - np.exp(', '1.0 - exp(')
N12_PERMITTED = ('expm1_underflow_probe',)


def _p(msg):
    print(msg, flush=True)


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arr_sha(a):
    """`phase0_gates.arr_sha`, reproduced: contiguous float64, raw bytes, sha256."""
    return hashlib.sha256(np.ascontiguousarray(a, np.float64).tobytes()).hexdigest()


# --------------------------------------------------------------------------
# the audit's own kernel: a day loop, vectorised over reaches
# --------------------------------------------------------------------------
def dp_land(inp, demand, gu, phi_f, gs, sm):
    """The DP kernel, written here from the plan's own six lines.

    `Lp_` is `L_pre` and `Fj` is `J` -- the fast/slow split of the SAME upper
    concentration, which is why no ordering appears anywhere.  `Lo` is built as the
    RECURRENCE `L_t = L_pre_t - F_s_t`, not as a `cumsum`: the frozen `tag_scan` uses the
    recurrence, and `cumsum(a+b-c)` is not bitwise `(prev+b)-c`, so a cumsum here would
    make N1 fail on rounding rather than on substance.
    """
    nd, nr = gu.shape
    M = np.zeros(nr)
    L = np.zeros(nr)
    Eu = np.zeros((nd, nr))
    Ff = np.zeros((nd, nr))
    Fj = np.zeros((nd, nr))
    Fs = np.zeros((nd, nr))
    Lp_ = np.zeros((nd, nr))
    Mo = np.zeros((nd, nr))
    Lo = np.zeros((nd, nr))
    for t in range(nd):
        A = np.maximum(M + inp[t] - demand[t], 0.0)
        e = A * gu[t]
        Eu[t] = e
        Ff[t] = e * phi_f[t]
        Fj[t] = e * (1.0 - phi_f[t])
        pre = L + Fj[t]
        Lp_[t] = pre
        Fs[t] = pre * gs[t]
        M = A * (1.0 - gu[t]) * sm
        L = pre - Fs[t]
        Mo[t] = M
        Lo[t] = L
    return dict(Eu=Eu, Ff=Ff, J=Fj, Fs=Fs, Lpre=Lp_, M=Mo, L=Lo)


def tau_of(surv):
    """Phase 0's estimator, verbatim, including its own internal filter.

    The `p10`/`p90` keys are inverted in the original (`1/percentile(ls, 90)` under the
    name `p10`) and are kept that way on purpose: this is a RECOMPUTATION, and quietly
    re-orienting them here would make the two columns disagree for a reason that has
    nothing to do with the model.
    """
    ls = -np.log(surv[np.isfinite(surv) & (surv > 0.0) & (surv < 1.0)])
    if not len(ls):
        return dict(geometric_mean=None, median=None, p10=None, p90=None,
                    max_survival_lifetime=None, n=0)
    return dict(geometric_mean=float(1.0 / np.mean(ls)),
                median=float(1.0 / np.median(ls)),
                p10=float(1.0 / np.percentile(ls, 90)),
                p90=float(1.0 / np.percentile(ls, 10)),
                max_survival_lifetime=float(1.0 / np.min(ls)), n=int(len(ls)))


def _is_exp_call(node):
    """`exp(...)`, `np.exp(...)`, `numpy.exp(...)`, `math.exp(...)`, `torch.exp(...)`."""
    if not isinstance(node, ast.Call):
        return False
    fn = node.func
    if isinstance(fn, ast.Name):
        return fn.id == 'exp'
    if isinstance(fn, ast.Attribute):
        return fn.attr == 'exp'
    return False


def naive_exp_sites(path):
    """Every EXPRESSION in `path` of the form `1 - <something>.exp(...)`.

    AST-BASED, NOT SUBSTRING-BASED, AND THAT IS THE POINT.  The plan states N12 as
    "the source must not contain `1 - np.exp(`", and a literal reading of that string
    matches three things in this round that are all innocent: `dp_kernel.py`'s module
    docstring and `g_of`'s docstring, which QUOTE the forbidden form precisely in order to
    forbid it, and `phase1_arms.py`'s own scanner literal, which contains the pattern as
    data.  A substring scanner therefore cannot answer the question it was written for --
    it fires on the rule's own statement.  This walks the parse tree instead and reports
    only a `Sub` whose LEFT operand is the numeric constant 1 and whose right subtree calls
    `exp`, which is the thing the rule is actually about.  The permitted site,
    `dp_kernel.expm1_underflow_probe`, is matched by name and reported separately.
    """
    src = Path(path).read_text(encoding='utf-8')
    tree = ast.parse(src)
    lines = src.splitlines()
    owner = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for sub in ast.walk(node):
                owner[id(sub)] = node.name
    hits = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Sub)):
            continue
        left, right = node.left, node.right
        if not (isinstance(left, ast.Constant) and left.value in (1, 1.0)):
            continue
        if not any(_is_exp_call(x) for x in ast.walk(right)):
            continue
        where = owner.get(id(node), '<module>')
        if any(p in where for p in N12_PERMITTED):
            continue
        hits.append([Path(path).name, where, node.lineno,
                     lines[node.lineno - 1].strip()[:120]])
    return hits


def quoted_naive_sites(path):
    """The docstring/literal occurrences the AST check deliberately ignores.

    Reported beside the real check rather than dropped, because "the source contains this
    text three times and every one of them is inside a string" is a fact a reviewer should
    be able to see without re-running the scan.
    """
    out = []
    for i, line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(), 1):
        if any(t in line for t in N12_NAIVE):
            out.append([Path(path).name, i, line.strip()[:120]])
    return out


def main():
    arms = json.loads((OUT / 'arms.json').read_text(encoding='utf-8'))
    ph0 = json.loads((OUT / 'phase0_gates.json').read_text(encoding='utf-8'))
    p1 = json.loads((OUT / 'phase1_arms.json').read_text(encoding='utf-8'))
    lvf = json.loads((OUT / 'level_variance.json').read_text(encoding='utf-8'))
    n1j = json.loads((OUT / 'phase0_n1.json').read_text(encoding='utf-8'))
    v = json.loads((OUT / 'verdict.json').read_text(encoding='utf-8'))
    H = arms['hashes']
    form = arms['closure_form']
    if form != 'x':
        raise SystemExit('THE_AUDIT_WAS_WRITTEN_FOR_THE_LINEAR_CLOSURE; got %r' % form)

    rep = {'phase': 'audit_dp', 'round': str(ROUND), 'n_fits': 0, 'fit_worker_calls': 0,
           'zero_forwards': True, 'imports_this_round': False,
           'sources': {f: sha_file(OUT / f) for f in
                       ('arms.json', 'phase0_gates.json', 'phase0_n1.json',
                        'phase1_arms.json', 'level_variance.json', 'verdict.json',
                        'daily_arms.parquet')},
           'checks': {}}
    chk = rep['checks']

    def eq(name, got, want, **kw):
        node = dict(got=got, frozen=want, ok=bool(got == want))
        node.update(kw)
        chk[name] = node
        return node['ok']

    # ---- the peer model, built here and NOT through this round ------------------
    rec = json.loads((PEER / 'outputs' / TAG / 'model.json').read_text(encoding='utf-8'))
    design = json.loads(json.dumps(rec['design']))
    design['observation_registry_file'] = 'data/prediction_registry.json'
    design['observation_registry_hash'] = sha_file(PEER / 'data/prediction_registry.json')
    data = cm.load_data('FULL24')
    model = cm.make_model(data, None, 'D29_BE', design)
    params = np.asarray(rec['parameters'], float)
    if len(params) != 30:
        raise SystemExit('THE_FROZEN_PARAMETER_VECTOR_IS_NOT_30')
    if model.cap is not False:
        raise SystemExit('THE_REFERENCE_SCAN_FLAG_IS_NOT_FALSE')
    d = model.data
    nd, nr = d.fast_water.shape
    dates = np.asarray(d.dates).astype('datetime64[D]')
    _p('=== audit: peer model built, %d days x %d reaches ===' % (nd, nr))

    # ---- 1. the three pathways, in mm/day --------------------------------------
    area = np.asarray(d.area_ha, np.float64)[None, :]
    Qf = np.asarray(d.fast_water, np.float64) / (area * MM_PER_HA)
    Qp = np.asarray(d.percolation, np.float64)
    Qs = np.asarray(d.slow_water, np.float64) / (area * MM_PER_HA)
    Qu = Qf + Qp
    for k, arr in (('Qf', Qf), ('Qp', Qp), ('Qs', Qs), ('Qu', Qu), ('dates', dates)):
        eq('hashes.%s' % k, arr_sha(arr), H[k])
    rep['lower_bounds'] = {k: float(np.min(a)) for k, a in
                           (('Qf', Qf), ('Qp', Qp), ('Qs', Qs), ('Qu', Qu))}
    rep['lower_bounds']['all_strictly_positive'] = bool(
        all(np.all(a > 0.0) for a in (Qf, Qp, Qs, Qu)))

    # ---- 2. the guard, the lower storage, and the two volumes -------------------
    ff = np.asarray(d.fast_fraction, np.float64)
    guard = np.where(ff == 0.0, 0.0, 1.0)
    eq('hashes.guard', arr_sha(guard), H['guard'])
    contact = np.asarray(d.contact, np.float64)
    rep['guard'] = dict(
        n_zero_cells=int((ff == 0.0).sum()), n_cells=int(ff.size),
        zero_set_is_contact_le_0=bool(np.array_equal(ff == 0.0, contact <= 0.0)),
        note='the guard is 1.0 where live and 0.0 on the frozen fast_fraction == 0 set; '
             'rebuilt here from the cache, not read from the round')

    # route (a): the producer's own storage column, with the ALIGNMENT RE-PROVEN here
    w = pd.read_parquet(HYDRO_PARQUET, columns=['date', 'reach_id',
                                                'lower_slow_storage_mm',
                                                'local_slow_response_m3_s'])
    w = w.sort_values(['date', 'reach_id'])
    if len(w) % nr or len(w) // nr < nd:
        raise SystemExit('HYDRO_SHAPE %d %d %d' % (len(w), nr, nd))
    dax = w.date.to_numpy().reshape(-1, nr)[:, 0].astype('datetime64[D]')
    if not np.array_equal(dax[:nd], dates):
        raise SystemExit('AUDIT_HYDRO_CALENDAR_MISALIGNED')
    slow_s = w.local_slow_response_m3_s.to_numpy(np.float64).reshape(-1, nr)[:nd]
    if not np.array_equal(slow_s * 86400.0, np.asarray(d.slow_water, np.float64)[:nd]):
        raise SystemExit('AUDIT_HYDRO_REACH_AXIS_MISALIGNED')
    S_low = np.ascontiguousarray(
        w.lower_slow_storage_mm.to_numpy(np.float64).reshape(-1, nr)[:nd])
    eq('hashes.lower_storage', arr_sha(S_low), H['lower_storage'])
    rep['alignment_reproven'] = dict(
        calendar=True, reach_axis=True,
        n_rows_parquet=int(len(w)), n_days_used=nd,
        producer_rows_beyond_the_model=int(len(w) // nr - nd),
        note='the reach axis is proven against the producer own '
             '`local_slow_response_m3_s`, which IS bitwise slow_water/86400, and not '
             'against a column name')

    up = np.asarray(d.upper_water, np.float64)
    Vu = up + Qu
    Vs = S_low + Qs
    eq('hashes.Vu_P_upper', arr_sha(Vu), H['Vu_P_upper'])
    eq('hashes.Vs', arr_sha(Vs), H['Vs'])
    rep['volumes'] = dict(
        Vu_min=float(Vu.min()), Vs_min=float(Vs.min()),
        both_strictly_positive=bool(Vu.min() > 0.0 and Vs.min() > 0.0),
        naming_trap='the exported state is the POST-outflow carry, and THIS `Vu` is '
                    '`Vu_post + Qu`; the arm table keeps the two apart as `Vu_post` and '
                    '`Vu_at_outflow` for exactly that reason, and swapping them would '
                    'silently change every x in the kernel')

    # route (b): invert lower_release.  Independent of the parquet.
    #
    # ROUTE (b) DOES NOT RECONSTRUCT `V_s`; IT RECONSTRUCTS THE POST-OUTFLOW STORAGE.
    # The frozen slow step is `slow = pre*l` and `S_post = pre - slow = pre*(1-l)`, so
    # `S_post = Q_s*(1-l)/l` and the PRE-outflow volume is `V_s = S_post + Q_s`. Comparing
    # the inverted expression against `V_s` therefore differs by exactly `Q_s`, i.e. by
    # `l` relative -- 0.78% here. That is not a provenance failure, it is the two sides of
    # the timing convention the plan calls the single most important thing to get right:
    # the FROZEN array is the post-outflow carry and the KERNEL's `V_s` is the pre-outflow
    # volume. This was written as a comparison against `V_s` on the first run, it fired at
    # 7.79e-3, and the comparison -- not the artifact -- was what was wrong.
    lr = np.asarray(d.lower_release, np.float64)
    S_post_b = Qs * (1.0 - lr) / lr
    rel_b = float(np.max(np.abs(S_post_b - S_low) / np.maximum(np.abs(S_low), 1e-300)))
    rep['Vs_route_b'] = dict(
        what_route_b_reconstructs='S_post, the POST-outflow lower storage (== S_low)',
        max_rel_diff_vs_the_parquet_storage=rel_b,
        agrees_within_tolerance=bool(rel_b <= 1e-12), tolerance=1e-12, bitwise=False,
        why_not_bitwise='route (a) reads S_low and route (b) is Qs*(1-lr)/lr; different '
                        'float expressions of the same real number, so the frozen hash '
                        '(taken on route (a)) cannot and must not be claimed to hold '
                        'bitwise for route (b)',
        max_rel_diff_if_mistaken_for_the_pre_outflow_volume=float(
            np.max(np.abs(Qs * (1.0 - lr) / lr - Vs) / np.maximum(np.abs(Vs), 1e-300))),
        timing_convention='V_s = S_low + Qs is the PRE-outflow volume the kernel divides '
                          'by; the frozen column is the POST-outflow carry. The gap '
                          'between the two is Qs, which is exactly the `l` relative gap '
                          'recorded above, and neither side is an error.',
        lower_release_range=[float(lr.min()), float(lr.max())],
        lower_release_relative_band=float((lr.max() - lr.min()) / np.median(lr)),
        note='the narrow band is the CIRCULARITY of the inverted formula, not hydrology: '
             'inverting it makes pi_s nearly constant per reach by construction, which is '
             'why the producer-written storage column is the registered source and this '
             'inversion is only a cross-check')

    # ---- 3. the closure, the concentration form, x <= 1 ------------------------
    xu = Qu / Vu
    xs = Qs / Vs
    gu_inst = xu * guard
    pf = Qf / Qu
    gs = xs
    eq('hashes.phi_f', arr_sha(pf), H['phi_f'])
    eq('hashes.gs', arr_sha(gs), H['gs'])
    rep['x_and_fractions'] = dict(
        max_xu=float(xu.max()), n_xu_gt_1=int((xu > 1.0).sum()),
        x_le_1_globally=bool(xu.max() <= 1.0),
        pf_min=float(pf.min()), pf_max=float(pf.max()),
        pf_strictly_inside_0_1=bool(pf.min() > 0.0 and pf.max() < 1.0),
        gs_min=float(gs.min()), gs_max=float(gs.max()),
        n_gu_out_of_range=int(((gu_inst < 0.0) | (gu_inst > 1.0)).sum()),
        n_gu_exactly_zero=int((gu_inst == 0.0).sum()),
        gu_zero_set_is_the_mask=bool(np.array_equal(gu_inst == 0.0, ff == 0.0)),
        note='x_u <= 1 globally is the Phase-0 reading that SELECTED the linear closure, '
             'and it is re-derived here rather than read from the arm table')
    xact = xu > 0.0
    phi_exp = np.ones_like(xu)
    phi_exp[xact] = -np.expm1(-xu[xact]) / xu[xact]
    rel = np.zeros_like(xu)
    rel[xact] = np.abs(phi_exp[xact] - 1.0)
    rep['closure_forms'] = dict(
        phi_linear_is_identically_one=True,
        phi_exp_median_where_xu_gt_0=float(np.median(phi_exp[xact])),
        frac_where_forms_differ_over_10pct=float(np.mean(rel[xact] > 0.10)),
        n_active_cells=int(xact.sum()),
        round_P4_on_the_primary_arm=p1['arms']['P-upper']['P4_closure_sensitivity'],
        selection='the audit figure is GLOBAL (every cell with x_u > 0, 230 reaches x '
                  '23376 days); the round figure is on the primary arm own active '
                  'selection, so the two are not the same denominator and are reported '
                  'beside each other rather than reconciled',
        note='P4 independently re-derived. The chosen form gives phi == 1 identically, so '
             'what this measures is how much the round did NOT use -- and it is large, so '
             'the closure choice is load-bearing and belongs beside the verdict rather '
             'than in a footnote.')

    # ---- 4. ASSERTION N1, BOTH SPELLINGS, ONE LOOP ----------------------------
    _p('=== audit: N1, the DP kernel against the frozen kernel ===')
    h_t, s_t, f_t, k_t = model.flux_parameters(torch.tensor(params))
    h = np.ascontiguousarray(h_t.detach().numpy(), np.float64)
    sm = np.ascontiguousarray(s_t.detach().numpy(), np.float64)
    f = np.ascontiguousarray(f_t.detach().numpy(), np.float64)
    k = np.ascontiguousarray(k_t.detach().numpy(), np.float64)
    if sm.shape != (nr,):
        raise SystemExit('S_M_IS_NOT_PER_REACH %r' % (sm.shape,))
    l_ = np.ascontiguousarray(lr, np.float64)
    inp = np.ascontiguousarray(np.asarray(model.inp, np.float64))
    demand = np.ascontiguousarray(np.asarray(model.demand, np.float64))

    fast0, slow0, a0, p0 = frozen_scan(h, sm, f, k, l_, inp, demand, bool(model.cap))
    eq('N1.frozen_p_is_the_audit_expm1_spelling', arr_sha(p0),
       arr_sha(-np.expm1(-np.minimum(h, CLIP))))

    # the frozen recurrence channels, written here, in the frozen spelling
    Lf_pre = np.zeros((nd, nr))
    Lf_Fs = np.zeros((nd, nr))
    Lf_L = np.zeros((nd, nr))
    Lcur = np.zeros(nr)
    for t in range(nd):
        pre = Lcur + a0[t] * p0[t] * (1.0 - f[t])
        Lf_pre[t] = pre
        Lf_Fs[t] = pre * l_[t]
        Lcur = pre - Lf_Fs[t]
        Lf_L[t] = Lcur
    eq('N1.frozen_recurrence_matches_the_returned_slow', arr_sha(Lf_Fs), arr_sha(slow0))

    dk = dp_land(inp, demand, p0, f, l_, sm)
    pairs = (('Eu', dk['Eu'], a0 * p0), ('F_f', dk['Ff'], fast0),
             ('J', dk['J'], a0 * p0 * (1.0 - f)),
             ('M', dk['M'], a0 * (1.0 - p0) * sm[None, :]),
             ('L_pre', dk['Lpre'], Lf_pre), ('F_s', dk['Fs'], slow0),
             ('L', dk['L'], Lf_L))
    for name, mine, ref in pairs:
        eq('N1.reduce.%s' % name, bool(np.array_equal(mine, ref)), True,
           max_abs_diff=float(np.max(np.abs(mine - ref))))
    rep['N1'] = dict(
        channels=len(pairs),
        all_bitwise=bool(all(chk['N1.reduce.%s' % n]['ok'] for n, _a, _b in pairs)),
        spellings_compared='DP: a*g_u, e*phi_f, e*(1-phi_f), a*(1-g_u)*s | '
                           'FROZEN: a*prob, E*f, E*(1-f), a*(1-prob)*s',
        injected='g_u := p0 (the frozen kernel own prob return), phi_f := f, g_s := l',
        round_channel_inventory=n1j['verdict'],
        tag_and_transport_groups=(
            'READ AS DATA from phase0_n1.json, not re-derived here: they live in '
            'closures_dp, which this file may not import. The transport pair is the same '
            'scan object, so the audit scan-level reduction covers it in substance, and '
            'this restatement is labelled as a restatement.'),
        note='if this block is False the round has no evidence that the new kernel is the '
             'same model with one layer replaced')

    chk['Vs.route_b_reconstructs_the_parquet_storage'] = dict(
        got=rel_b, frozen='<= 1e-12', ok=bool(rel_b <= 1e-12),
        note='route (b) never touches the parquet, so a calendar slip or a reach '
             'permutation could not survive this; the pre/post gap is reported separately '
             'and is the timing convention, not an error')

    # ---- 5. the mass ledger, written from the frozen formula -------------------
    Mf = a0 * (1.0 - p0) * sm[None, :]
    Lf = np.cumsum(a0 * p0 * (1.0 - f) - slow0, axis=0)
    before = np.vstack([np.zeros_like(Mf[:1]), Mf[:-1]])
    uptake = np.minimum(before + inp, demand)
    loss = a0 * (1.0 - p0) * (1.0 - sm[None, :])
    bal = (inp - uptake - loss - fast0 - slow0
           - np.diff(Mf + Lf, axis=0, prepend=np.zeros_like(Mf[:1])))
    lbal = float(np.abs(bal).max())
    rep['ledger'] = dict(
        local_balance_max_kg=lbal, tolerance_kg=1e-6, passed=bool(lbal <= 1e-6),
        written_as='<=, so a NaN would fail rather than pass',
        reported_in_N10_kg=float(p1['N10']['max_local_balance_kg']),
        M_matches_the_frozen_ledger_bitwise=bool(np.array_equal(
            Mf, a0 * (1.0 - p0) * sm[None, :])),
        cumsum_L_equals_the_recurrence_L=bool(np.array_equal(Lf, Lf_L)),
        scope='the MASS-LEDGER identity -- the kernel returned fluxes against an '
              'independently written recurrence -- NOT a physical conservation law and '
              'NOT a load criterion',
        note='the ledger is NOT vacuous here only because the new ledger is spelled from '
             'the injected FRACTIONS (A*g_u*phi_f, A*g_u*(1-phi_f)) and never from the '
             'kernel RETURNED F_f/J; a ledger written from return values would be an '
             'identity by construction and would check nothing')

    # ---- 6. the section 2.6 P1 identity, re-derived ---------------------------
    _p('=== audit: the P1 identity ===')
    interior = np.ones_like(contact, bool)
    interior[0] = False
    sel = (contact > 0.0) & interior
    one_minus_g = 1.0 - gu_inst
    true_share = up / Vu
    ident = one_minus_g - true_share
    rel_id = np.abs(ident) / np.maximum(np.abs(one_minus_g), np.finfo(float).tiny)
    # THE RELATIVE FORM IS ILL-CONDITIONED HERE AND THE ABSOLUTE FORM IS THE READING.
    # `1 - Q_u/V_u` cancels catastrophically as x_u -> 1, and x_u reaches 0.9999999999999986
    # on this grid, so cells where the retained share is ~1e-15 show a relative gap of ~0.1
    # while their ABSOLUTE gap is one ulp. Reporting the relative maximum alone would turn a
    # conditioning artefact into a "the identity does not hold" claim, which is the same
    # class of mistake the round made with the label floor.
    cond = one_minus_g >= 1e-6
    rep['P1_identity'] = dict(
        identity='1 - g_u == S_post/V_u, i.e. the retained WATER fraction',
        max_abs_diff=float(np.abs(ident[sel]).max()),
        max_rel_diff=float(rel_id[sel].max()),
        max_rel_diff_where_retained_share_ge_1e6=(
            float(rel_id[sel & cond].max()) if int((sel & cond).sum()) else None),
        n_selected=int(sel.sum()), n_selected_well_conditioned=int((sel & cond).sum()),
        n_selected_retained_share_lt_1e15=int((sel & (one_minus_g < 1e-15)).sum()),
        max_abs_diff_in_ulps=float(np.abs(ident[sel]).max() / np.finfo(float).eps),
        holds_to_one_ulp=bool(float(np.abs(ident[sel]).max()) <= 2 * np.finfo(float).eps),
        bitwise=False,
        why_not_bitwise='the two sides are different float expressions of the same real '
                        'number (1 - Qu/Vu against S_post/(S_post+Qu)); they agree to '
                        'ROUNDING, and asserting bit-equality -- or a 1e-12 RELATIVE '
                        'tolerance -- would be a claim about float subtraction rather than '
                        'about the model. The relative form is only meaningful on the '
                        'well-conditioned subset, reported above.',
        consequence='with g = x the daily carry of the M pool is (S_post/V_u)*s_M, so the '
                    'N memory lifetime cannot exceed the upper store water residence time. '
                    'This is a property of the CHOSEN closure, not a measurement of the '
                    'catchment, and it is why P1 is falsified by IDENTITY and not by '
                    'approximation.')

    chk['P1.identity_holds_to_one_ulp_in_absolute_terms'] = dict(
        got=rep['P1_identity']['max_abs_diff_in_ulps'], frozen='<= 2',
        ok=bool(rep['P1_identity']['holds_to_one_ulp']),
        note='the ABSOLUTE form is the reading; the relative form is reported on the '
             'well-conditioned subset only, because 1 - Q_u/V_u cancels as x_u -> 1')

    carry = (1.0 - gu_inst) * sm[None, :]
    t_eff = tau_of(carry[sel])
    xh = xu[sel]
    t_hyd = tau_of(1.0 - xh[np.isfinite(xh) & (xh > 0.0) & (xh <= 1.0)])
    tau_rep = p1['arms']['P-upper']['tau']
    ratio_audit = float(t_eff['median'] / t_hyd['median'])
    ratio_rep = float(tau_rep['ratio__median_over_median'])
    rep['P1_lifetimes'] = dict(
        audit=dict(tau_eff=t_eff, tau_hydro=t_hyd, ratio__median_over_median=ratio_audit,
                   n_sel=int(sel.sum())),
        round_reported=dict(tau_eff_median=tau_rep['tau_eff_block']['median'],
                            tau_hydro_median=tau_rep['tau_hydro_block']['median'],
                            ratio__median_over_median=ratio_rep, n_sel=tau_rep['n_sel']),
        agrees=bool(abs(ratio_audit - ratio_rep) <= 1e-9),
        names_kept_separate_in_the_round=bool(tau_rep.get('names_kept_separate')),
        s_M_band_relative=float((sm.max() - sm.min()) / np.median(sm)),
        s_M_implied_lifetime_days=[float(1.0 / -np.log(sm.max())),
                                   float(1.0 / -np.log(sm.min()))],
        note='the two lifetimes must never share a name. s_M alone implies a memory of '
             'hundreds to thousands of days; the measured tau_eff is single-digit days, '
             'which is the upper store water residence time. That gap IS the falsification.')

    # ---- 7. level / variance / G5b, recomputed from the delivered parquet -------
    _p('=== audit: level / variance / G5b ===')
    if sha_file(MASK) != MASK_SHA:
        raise SystemExit('THE_ELIGIBLE_MASK_DRIFTED')
    if sha_file(PANEL) != PANEL_SHA:
        raise SystemExit('THE_4h_PANEL_CHANGED')
    m0 = pd.read_parquet(MASK)
    elig = m0[m0.eligible][['station_key', 'date']].copy()
    elig['date'] = pd.to_datetime(elig.date).dt.normalize().astype('datetime64[ns]')
    if len(elig) != 12152:
        raise SystemExit('ELIGIBLE_GRID_IS_NOT_12152_ROWS %d' % len(elig))
    pan = pd.read_parquet(PANEL)
    pan = pan.assign(date=pan.monitoring_time.dt.tz_localize(None).dt.normalize(),
                     y=pan.monitoring_time.dt.year, mo=pan.monitoring_time.dt.month)
    pday = pan.groupby(['station_key', 'date'], as_index=False).TN.mean()
    pday = pday.assign(y=pday.date.dt.year, mo=pday.date.dt.month)
    pday = pday[(pday.y >= START_YEAR) & (pday.y <= END_YEAR)]
    obs_m = pday.groupby(['station_key', 'y', 'mo']).TN.mean().rename('obs')

    daily = pd.read_parquet(OUT / 'daily_arms.parquet')
    daily = daily.assign(date=pd.to_datetime(daily.date).dt.normalize()
                         .astype('datetime64[ns]'))
    arm_names = sorted(daily.arm.unique())
    n_per = len(daily) // len(arm_names)
    rep['daily_arms'] = dict(
        n_rows=int(len(daily)), n_arms=int(len(arm_names)), n_per_arm=int(n_per),
        rows_equal_the_eligible_grid=bool(n_per == len(elig)),
        arms=arm_names, eligible_rows=int(len(elig)),
        note='verification item 16: the delivered frame carries every arm including '
             'R5-ref, at exactly the eligible-grid row count')

    def monthly(ly):
        m = elig.merge(ly[['station_key', 'date', 'pL3']].rename(columns={'pL3': 'p'}),
                       on=['station_key', 'date'], how='left', validate='one_to_one')
        if int(m.p.isna().sum()):
            raise SystemExit('AUDIT_DROPPED_AN_ELIGIBLE_DAY')
        m = m.assign(y=m.date.dt.year, mo=m.date.dt.month)
        pm = m.groupby(['station_key', 'y', 'mo']).p.mean().rename('pred')
        return m, obs_m.to_frame().join(pm, how='inner').dropna()

    null = daily[daily.arm == 'B0'].reset_index(drop=True)
    _, znull = monthly(null)
    o = znull.obs.to_numpy(float)
    nse_null = float(1.0 - np.mean((znull.pred.to_numpy(float) - o) ** 2) / np.var(o))
    rep['audit_level'] = dict(
        null_nse=nse_null, frozen_baseline_nse=BASELINE_NSE,
        null_reproduces_the_baseline=bool(abs(nse_null - BASELINE_NSE) <= 1e-9),
        null_mean_concentration=float(null.pL3.to_numpy(float).mean()),
        note='B0 IS the frozen arm, so its own monthly NSE must be the frozen baseline; '
             'this is what makes every ratio below a ratio against the frozen kernel')
    for a in arm_names:
        sub = daily[daily.arm == a].reset_index(drop=True)
        if not np.array_equal(np.asarray(sub.station_key.astype(str)),
                              np.asarray(null.station_key.astype(str))):
            raise SystemExit('ARM_%s_IS_NOT_ROW_ALIGNED_WITH_B0' % a)
        if not np.array_equal(sub.date.to_numpy(), null.date.to_numpy()):
            raise SystemExit('ARM_%s_HAS_A_DIFFERENT_DAY_AXIS' % a)
        lv = lvf['level_variance'].get(a)
        if lv is None:
            continue
        pv = sub.pL3.to_numpy(float)
        nv = null.pL3.to_numpy(float)
        m, z = monthly(sub)
        if not z.index.equals(znull.index):
            raise SystemExit('RECENTRING_ON_A_DIFFERENT_INDEX %s' % a)
        pr = z.pred.to_numpy(float)
        pn = znull.pred.to_numpy(float)
        node = dict(
            level_ratio=float(pv.mean() / nv.mean()),
            variance_ratio=float(np.var(pv) / np.var(nv)),
            nse=float(1.0 - np.mean((pr - o) ** 2) / np.var(o)),
            nse_level_removed=float(1.0 - np.mean((pr - pr.mean() + pn.mean() - o) ** 2)
                                    / np.var(o)),
            n_station_months=int(len(z)), n_eligible_rows=int(len(m)),
            mean_arm=float(m.p.to_numpy(float).mean()),
            reported=dict(level_ratio=lv['level_ratio'],
                          variance_ratio=lv['amplitude_ratio'], nse=lv['nse'],
                          nse_level_removed=lv['nse_level_removed'],
                          n_station_months=lv['n_station_months'],
                          n_eligible_rows=lv['n_eligible_rows'], mean_arm=lv['mean_arm']))
        node['matches_the_round'] = bool(all(
            abs(node[k] - node['reported'][k]) <= 1e-9
            for k in ('level_ratio', 'variance_ratio', 'nse', 'nse_level_removed',
                      'mean_arm'))
            and node['n_station_months'] == node['reported']['n_station_months']
            and node['n_eligible_rows'] == node['reported']['n_eligible_rows'])
        rep['audit_level'][a] = node

    # ---- 8. N12 as a source check, independent of the round's own scanner -------
    n12_all = [h for f in N12_WATCH if (HERE / f).is_file()
               for h in naive_exp_sites(HERE / f)]
    rep['N12'] = dict(
        expression_sites={f: naive_exp_sites(HERE / f) for f in N12_WATCH
                          if (HERE / f).is_file()},
        n_expression_sites=len(n12_all),
        quoted_sites={f: quoted_naive_sites(HERE / f) for f in N12_WATCH
                      if (HERE / f).is_file()},
        permitted=['dp_kernel.expm1_underflow_probe'],
        scanner_was_corrected_here=(
            'the audit first ran this as a SUBSTRING scan and it fired three times: '
            'dp_kernel.py module docstring line 77, dp_kernel.g_of docstring line 145, '
            'and phase1_arms.py own scanner literal line 1120. Every one of those is the '
            'forbidden form QUOTED in order to forbid it, or the pattern held as data. A '
            'substring scanner cannot answer this question because it fires on the rule '
            'own statement, so the audit was fixed rather than the artifact (the artifact '
            'reproduces byte for byte). The parse-tree form above is the check.'),
        round_scanner_result=n1j['N12']['naive_exp_sites'],
        note='the round own scanner reported an empty list for three files; this '
             'independently confirms it on SIX files by walking the parse tree, and also '
             'reports the quoted occurrences that a text scan cannot distinguish')
    chk['N12.no_naive_exp_expression'] = dict(
        got=len(n12_all), frozen=0, ok=bool(not n12_all))
    xp = 1e-200
    naive = 1.0 - np.exp(-xp)
    rep['N12']['underflow_probe'] = dict(
        x=xp, one_minus_exp=float(naive), neg_expm1=float(-np.expm1(-xp)),
        argument_returned_exactly=str(float(-np.expm1(-xp)) == xp),
        note='the direct subtraction is EXACTLY zero and expm1 returns the argument itself: '
             'a numerical-correctness assertion, not a style rule. This line is itself a '
             'deliberate occurrence of the naive form in the AUDIT, which is why the audit '
             'is not in N12_WATCH -- it is the measurement, the same status as '
             'dp_kernel.expm1_underflow_probe.')

    # ---- 9. the verdict's arithmetic, read rather than re-run ------------------
    prim = arms['primary_arm']
    VGRID = ('DUAL_PATH_CAPABILITY_DEMONSTRATED', 'VOLUME_MAPPING_SENSITIVE',
             'AMPLITUDE_WITHOUT_LEVEL', 'LEVEL_WITHOUT_AMPLITUDE',
             'AMPLITUDE_BUT_VARIANCE_COLLAPSES', 'MOBILE_POOL_MAPPING_LIMITED',
             'NO_AMPLITUDE_MECHANISM', 'NO_ADMISSIBLE_ARM', 'STATE_GATE_FAILED',
             'BLOCKED', 'UNCLASSIFIED_BY_THE_REGISTERED_GRID')
    fired = {a: bool(p1['arms'][a].get('tau', {}).get('P1_fired'))
             for a in p1['arms'] if a not in ('B0', 'R5-ref')}
    rep['verdict_arithmetic'] = dict(
        primary_arm_matches_phase0=bool(prim == ph0['3.2']['primary_arm']),
        primary_arm=prim, outcome=v['outcome'],
        outcome_is_registered=bool(v['outcome'] in VGRID),
        n_primary_gates_passed=int(v['n_primary_gates_passed']),
        primary_gates=v['primary_gates'],
        next_round_choice=v['next_round']['chosen'],
        n_forwards=int(p1['forward_count']), n_fits=int(p1['n_fits']),
        fit_worker_calls=int(p1['fit_worker_calls']),
        p1_falsifier_fired_on_every_kernel_arm=bool(fired and all(fired.values())),
        p1_falsifier_per_arm={a: p1['arms'][a]['tau']['P1_falsifier'] for a in fired},
        main_five_gates={a: p1['arms'][a].get('n_gates_passed_main_five')
                         for a in p1['arms']},
        reference_arm_carries_no_gate=bool(all(
            p1['arms']['R5-ref'].get(g) is None for g in
            ('G1_pass', 'G2_pass', 'G3_pass', 'G5_pass', 'G5b_pass'))),
        note='the audit does not re-run the grid; it checks that the grid ran on the arm '
             'Phase 0 froze, that its outcome is a registered row, and that the condition '
             'the verdict names DID fire on every kernel arm')
    chk['verdict.primary_arm_is_the_frozen_one'] = dict(
        got=prim, frozen=ph0['3.2']['primary_arm'],
        ok=bool(prim == ph0['3.2']['primary_arm']))
    chk['verdict.outcome_is_registered'] = dict(
        got=v['outcome'], frozen=list(VGRID), ok=bool(v['outcome'] in VGRID))

    # ---- 10. the neighbours, so item 17 can be checked by re-running this -------
    rep['neighbour_json_shas'] = {
        str(p.relative_to(UP).as_posix()): sha_file(p)
        for root in (R4 / 'reports', R5 / 'reports') if root.is_dir()
        for p in sorted(root.glob('*.json'))}
    rep['neighbour_note'] = ('verification item 17 compares these against a before-image; '
                             'the round writes nowhere outside its own directory, so any '
                             'change here is a foreign write and a STOP')

    # ---- roll-up ---------------------------------------------------------------
    bad = [k for k, val in chk.items() if val.get('ok') is False]
    rep['n_checks'] = len(chk)
    rep['n_failed'] = len(bad)
    rep['failed'] = bad
    rep['all_reproduce'] = bool(not bad)
    rep['scope_limits'] = [
        'routing and station aggregation are SHARED with the producer; re-deriving them '
        'would be a second implementation of frozen peer code',
        'A_L1 is NOT independently recomputed: it needs the dense 169,476-row forward '
        'frame, and the delivered parquet is the 12,152-row eligible grid. This is a real '
        'gap in the deliverable set, registered rather than papered over.',
        's_M comes from the peer own flux_parameters, not rebuilt',
        'the tag and transport channel groups of N1 live in closures_dp and are read as '
        'data, not re-derived; the transport pair is the same scan object',
        'V_s route (b) agrees to rounding and NOT bitwise, because the frozen hash was '
        'taken on route (a)',
        'the producer parquet is read once for four columns; both V_s routes would survive '
        'a corruption present in the cache itself',
        'the audit re-reads the eligible mask and the 4h TN panel directly with their '
        'registered shas hard-coded here, so a change in a round helper cannot mask itself',
    ]
    (OUT / 'audit_dp.json').write_text(
        json.dumps(rep, indent=1, ensure_ascii=False, default=str), encoding='utf-8')
    _p('=== wrote %s ===' % (OUT / 'audit_dp.json'))
    _p('checks=%d failed=%d all_reproduce=%s' % (len(chk), len(bad),
                                                 rep['all_reproduce']))
    for k in bad:
        _p('   FAILED %s got=%r frozen=%r' % (k, chk[k].get('got'), chk[k].get('frozen')))
    if bad:
        raise SystemExit('THE_AUDIT_FAILED %r' % bad)
    return rep


if __name__ == '__main__':
    main()
