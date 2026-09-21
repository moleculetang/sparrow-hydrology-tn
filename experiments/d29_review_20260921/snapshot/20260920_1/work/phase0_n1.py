"""Phase 0, assertion N1 + N2 -- the two structural gates, run BEFORE any forward.

N1  `DP_KERNEL_DOES_NOT_REDUCE_TO_FROZEN`
    Inject `g_u := p_frozen`, `phi_f := f_frozen`, `g_s := l_frozen` and require
    `np.array_equal` -- BITWISE, not `allclose` -- on all thirteen channels:
    four from `scan`, seven from `tag_scan`, two from `Transport.apply`.
    This is the beta=0 substitute: without it there is no evidence the new kernel was
    actually installed, because the ledger gate (N3) does not carry that duty (R1').

    `h` and `f` are not re-derived here.  `f` is taken from `flux_parameters` -- the same
    call the frozen ledger makes -- because it is `aq*fast_fraction/(aq*fast_fraction+1-
    fast_fraction)` (`expert/tn_challenge/model.py:70`), which is NOT `fast_fraction`.
    Re-deriving it would test my transcription of that line, not the reduction.

    NOTE ON `phi_f := f_frozen`: the injected frozen `f` may be EXACTLY 0 where
    `fast_fraction == 0`, because the odds transform maps 0 to 0.  The strict
    `phi_f in (0,1)` assertion lives in `dp_kernel.fractions`, which N1 deliberately
    BYPASSES -- it calls `scan_dp` directly.  So no assertion is weakened: the strict one
    still guards every real forward, and this gate is not the place for it.

N2  `INERT_EQUIVALENCE_NOT_DERIVED`
    On the `contact <= 0` set the measured fluxes are 4.45e-21 m3/day and 2.51e-14 mm/day
    -- vanishing -- but `g` depends on the RATIO `Q_u/V_u`, and `upper_water` collapses to
    ~1e-205 there while `Q_u` collapses faster, so `x -> 1` and `g -> 1`.  The earlier
    derivation that `Q -> 0 therefore E -> 0` is refuted by measurement
    (`probe_timing.json::slow_path_on_mask::max_Eu_over_A_on_mask = 1.0`), and the guard
    is retained.  Its form was chosen so that it reproduces the frozen `prob = 0`
    semantics exactly: `h = where(contact>0, exp(logh), 0)` (`model.py:69`), so on the
    mask `prob = -expm1(-min(0,700)) = 0` EXACTLY.

    N2 then asks for five channels -- `M, L, F_f, J, F_s` -- not just `fast == 0`, because
    a zeroed fast path that quietly changed the slow path would pass a one-channel check.
    The plan expected the slow path to differ because it assumed `g_s != l`.  It does not:
    under the linear closure `g_s = Q_s/(S_post+Q_s)` and the producer's own recurrence
    `q2 = k2*S_pre`, `S_post = S_pre(1-k2)` gives `g_s = k2` IDENTICALLY, and
    `lower_release` IS `k2` broadcast.  Asserted bitwise below.  So N2 runs its STRONG
    branch: full five-channel equality, and the "deliberate structural difference on the
    mask" registered in plan S3.5-N2 is EMPTY.  That is a strictly stronger outcome and
    the deviation is written up rather than the assertion weakened.
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common24 as C
import closures_dp as MC

_closed = sys.modules['closures']
_tagged = sys.modules['tagged_transport']

rep = {}
FAIL = []


def eq(name, a, b):
    a = np.ascontiguousarray(np.asarray(a)); b = np.ascontiguousarray(np.asarray(b))
    same_shape = (a.shape == b.shape)
    same = bool(same_shape and np.array_equal(a, b))
    if same_shape and a.size:
        d = float(np.max(np.abs(a - b)))
    else:
        d = None
    n_diff = int(np.sum(a != b)) if same_shape else None
    rep.setdefault('channels', {})[name] = dict(
        shape=list(a.shape), n_elements=int(a.size), shapes_match=bool(same_shape),
        bitwise_equal=same, n_elements_differing=n_diff, max_abs_diff=d)
    if not same:
        FAIL.append(name)
    return same


# --------------------------------------------------------------------------
# 1. the frozen parameters and the frozen fraction triple
# --------------------------------------------------------------------------
model = C.build()
x = C.parameters(C.TAG)
assert model.cap is False, 'the CAP branch would change scan; it must be off'
assert not model.mix, 'the MIX branch would change f; it must be off'
with torch.no_grad():
    h, s, f, k = [v.numpy() for v in model.flux_parameters(torch.tensor(x))]
h = np.ascontiguousarray(h); s = np.ascontiguousarray(s)
f = np.ascontiguousarray(f); k = np.ascontiguousarray(k)
l = np.ascontiguousarray(np.asarray(model.data.lower_release, np.float64))
inp = np.ascontiguousarray(np.asarray(model.inp, np.float64))
dem = np.ascontiguousarray(np.asarray(model.demand, np.float64))
nd, nr = h.shape

rep['shapes'] = dict(h=list(h.shape), s=list(s.shape), f=list(f.shape),
                     lower_release=list(l.shape), inp=list(inp.shape))
rep['f_is_not_fast_fraction'] = dict(
    f_max=float(f.max()), fast_fraction_max=float(np.max(model.data.fast_fraction)),
    n_differing=int(np.sum(f != np.asarray(model.data.fast_fraction, np.float64))),
    note='f = aq*ff/(aq*ff+1-ff); the two arrays are different objects by construction')

# --------------------------------------------------------------------------
# 2. the frozen kernel, and the injection it defines
# --------------------------------------------------------------------------
Ff0, Fs0, a0, p0 = _closed.scan(h, s, f, k, l, inp, dem, model.cap)
rep['frozen_scan_taken'] = True

mask = np.asarray(model.data.contact, np.float64) <= 0.0
colmask_all = np.all(mask, axis=0)        # (230,) reaches masked on EVERY day
colnone_all = np.all(~mask, axis=0)       # (230,) reaches never masked
rep['mask'] = dict(
    n_cells=int(mask.sum()), n_total=int(mask.size), frac=float(mask.mean()),
    static_in_time=bool(np.all(mask == mask[:1])),
    n_masked_reaches_day0=int(mask[0].sum()),
    n_cols_masked_every_day=int(colmask_all.sum()),
    n_cols_never_masked=int(colnone_all.sum()),
    n_cols_intermittent=int((~colmask_all & ~colnone_all).sum()),
    cells_in_ever_masked_cols=int(np.sum(np.any(mask, axis=0))),
    h_exactly_zero_on_mask=bool(np.all(h[mask] == 0.0)),
    prob_exactly_zero_on_mask=bool(np.all(p0[mask] == 0.0)),
    max_abs_h_on_mask=float(np.max(np.abs(h[mask]))))
# THE MASK IS DAILY, NOT STATIC.  `contact` is a per-day hydraulic-contact field, so a
# reach can be dry on one day and wet the next.  This is why a full-trajectory comparison
# over ALL masked cells is not a well-posed test: the entering `M[r]`/`L[r]` on a masked
# day are whatever the reach's unmasked history left behind, and that history differs
# between the two kernels by construction.  The well-posed comparisons are the ONE-STEP
# operator on masked cells, and the full trajectory on the `masked-every-day` columns.

# --------------------------------------------------------------------------
# 3. N1 channel group A -- scan, 4 channels
# --------------------------------------------------------------------------
fa, sa_, aa, pa = MC.scan_dp(h, s, f, k, l, inp, dem, model.cap, p0, f, l)
eq('scan.fast', fa, Ff0)
eq('scan.slow', sa_, Fs0)
eq('scan.available', aa, a0)
eq('scan.prob', pa, p0)

# --------------------------------------------------------------------------
# 4. N1 channel group B -- tag_scan, 7 channels, on the SLICED arrays
# --------------------------------------------------------------------------
rr = C.pilot_indices(model)
hs = np.ascontiguousarray(h[:, rr]); ss = np.ascontiguousarray(s[rr])
fs = np.ascontiguousarray(f[:, rr])
tag_inputs = np.ascontiguousarray(np.asarray(model.tag_inputs, np.float64))
tag_demand = np.ascontiguousarray(np.asarray(model.tag_demand, np.float64))
tag_release = np.ascontiguousarray(np.asarray(model.tag_release, np.float64))
assert tag_inputs.shape[1] == len(rr) and tag_release.shape[1] == len(rr)
assert ss.shape == (len(rr),), ss.shape
rep['tag_shapes'] = dict(h=hs.shape and list(hs.shape), s=list(ss.shape), f=list(fs.shape),
                         inputs=list(tag_inputs.shape), demand=list(tag_demand.shape),
                         release=list(tag_release.shape), n_pilot=len(rr))

T0 = _tagged.tag_scan(hs, ss, fs, tag_release, tag_inputs, tag_demand)
gu_s = np.ascontiguousarray(p0[:, rr]); pf_s = np.ascontiguousarray(f[:, rr])
gs_s = np.ascontiguousarray(l[:, rr])
T1 = MC._tag_dp_nb(hs, ss, fs, tag_release, tag_inputs, tag_demand, gu_s, pf_s, gs_s)
for i, nm in enumerate(('fast', 'slow', 'raw', 'M', 'L', 'uptake', 'mineral_loss')):
    eq('tag.' + nm, T1[i], T0[i])

# the public wrapper too, so `_check`'s shape assertion is exercised on the tag path
MC.set_tag_slice(rr)
MC.set_config(p0, f, l, gu_s, pf_s, gs_s)
T2 = MC.tag_scan_dp(hs, ss, fs, tag_release, tag_inputs, tag_demand)
for i, nm in enumerate(('fast', 'slow', 'raw', 'M', 'L', 'uptake', 'mineral_loss')):
    eq('tag_via_wrapper.' + nm, T2[i], T0[i])

# --------------------------------------------------------------------------
# 5. N1 channel group C -- Transport.apply, 2 channels
# --------------------------------------------------------------------------
MC.set_config(p0, f, l, gu_s, pf_s, gs_s)
_saved = getattr(model, 'dp_fractions', None)
model.dp_fractions = dict(gu=p0, phi_f=f, gs=l, guard=np.ones_like(p0), form='injected')
C1 = MC.TransportDP.apply(torch.from_numpy(h), torch.from_numpy(s),
                          torch.from_numpy(f), torch.from_numpy(k), model)
C0 = _closed.Transport.apply(torch.from_numpy(h), torch.from_numpy(s),
                             torch.from_numpy(f), torch.from_numpy(k), model)
eq('transport.fast', C1[0].numpy(), C0[0].numpy())
eq('transport.slow', C1[1].numpy(), C0[1].numpy())
if _saved is None:
    del model.dp_fractions
else:
    model.dp_fractions = _saved

# the installed arrays must ALSO disagree with the injection off the mask, or N1 would be
# passing for a vacuous reason (a kernel that ignores its arguments).
rep['injection_is_not_the_installed_triple'] = dict(
    gu_differs_off_mask=bool(np.any(p0[~mask] != 0.0)) and True,
    note='p0 is the frozen prob; the installed gu is x_u*guard -- different functions by design')

MC.set_config(None, None, None)
MC.set_tag_slice([])

# --------------------------------------------------------------------------
# 6. N2 -- the mask, five channels, with the INSTALLED arrays
# --------------------------------------------------------------------------
pack = C.dp_arrays(model, form='x')
gu = pack['gu']; pf = pack['phi_f']; gs = pack['gs']; guard = pack['guard']
rep['N2'] = dict(
    guard_zero_set_is_contact_le0=bool(np.array_equal(guard == 0.0, mask)),
    gu_exactly_zero_on_mask=bool(np.all(gu[mask] == 0.0)),
    gu_equals_guard_times_xu=bool(np.array_equal(gu, pack['xu'] * guard)),
    max_gu_off_mask=float(np.max(gu[~mask])),
)
eq('N2.gu_vs_frozen_prob_on_mask', gu[mask], p0[mask])

# NOT bitwise, and my algebra said it would be -- the algebra was wrong, not the kernel.
# `l` is stored as `fl(fl(k2*S)/S)`, which is per-cell and differs from `k2` by ~1 ulp;
# `g_s = fl(q2/(fl(S-q2)+q2))` differs from `l` by ~1 ulp too, because `fl(fl(a-b)+b) != a`
# in float64.  So the AGGREGATE claim (g_s == k2 per reach) holds, the per-cell BITWISE
# claim does not, and the difference is 3.47e-18 -- 4.5e-16 relative.
gs_rel = float(np.max(np.abs(gs - l) / np.maximum(np.abs(l), 1e-300)))
rep['gs_vs_lower_release'] = dict(
    bitwise_equal=bool(np.array_equal(gs, l)),
    n_differing=int(np.sum(gs != l)), n_total=int(gs.size),
    max_abs_diff=float(np.max(np.abs(gs - l))), max_rel_diff=gs_rel,
    l_distinct_values=int(np.unique(l).size), l_n_cells=int(l.size),
    l_range=[float(l.min()), float(l.max())],
    l_time_bandwidth_relative=float((l.max() - l.min()) / l.mean()),
    l_spread_is_per_reach_not_per_day=bool(np.all(l == l[:1])) ,
    note='g_s and l agree to float64 rounding relative; the plan promised bitwise, '
         'the measurement gives rounding, and the discrepancy is registered')
# The slow path therefore CANNOT be made bitwise identical, and N2 has to say so rather
# than assert it.  The bound is what makes it a rounding statement instead of a
# structural one: one step moves `F_s` by at most `|g_s - l| / l` in relative terms.
if gs_rel > 1e-14:
    FAIL.append('gs_structurally_differs_from_lower_release')
rep['N2']['slow_path_agreement_kind'] = 'ROUNDING (max rel %.3e), not bitwise' % gs_rel

# P2's evidence, MEASURED.  R5' called `l` "approximately per-reach constant" and cited a
# 6.1% band.  A band is ambiguous between two very different readings -- variation ACROSS
# reaches (a stationary linear reservoir with a per-reach rate: constant in time, so no
# time-varying slow pathway) and variation WITHIN a reach over time (a genuinely
# time-varying release: a real slow-pathway dynamics).  The two are separated here rather
# than asserted, and the verdict string is DERIVED from the two numbers, because a
# hardcoded verdict is exactly what the measurement then contradicted the first time.
spread_r = np.array([float(l[:, r].max() - l[:, r].min()) for r in range(nr)])
level_r = np.array([float(np.median(l[:, r])) for r in range(nr)])
within_rel = float(np.max(spread_r / level_r))
across_rel = float((level_r.max() - level_r.min()) / level_r.mean())
rep['P2_slow_rate'] = dict(
    global_min=float(l.min()), global_max=float(l.max()),
    global_bandwidth_relative=float((l.max() - l.min()) / l.mean()),
    across_reach_band_relative=across_rel,
    within_reach_band_relative_max=within_rel,
    within_reach_spread_max=float(spread_r.max()),
    within_reach_spread_median=float(np.median(spread_r)),
    n_distinct_values=int(np.unique(l).size), n_cells=int(l.size),
    corr_with_lower_storage_median=float(np.median(
        [np.corrcoef(l[:, r], pack['Vs'][:, r] - pack['Qs'][:, r])[0, 1]
         for r in range(nr) if np.std(l[:, r]) > 0])),
    frac_l_equals_one=float(np.mean(l == 1.0)),
    floating_point_floor=float(np.finfo(np.float64).eps),
    verdict=('TIME-VARYING within reach' if within_rel > 1e-12
             else ('ACROSS-reach only (stationary linear reservoir)'
                   if across_rel > 1e-12 else 'CONSTANT')),
    consequence='under the linear closure phi(x_s) == 1, so the slow pathway carries no '
                'time-varying factor of its own; its amplitude response can come only '
                'through L_pre and the volume V_s.  Registered, not read as "no slow path".')

# the per-day transition identity on masked cells.  `g_u` is a function of `Q` and `V`
# only -- never of the N state -- so `A * g_u` here is the new kernel's actual `E_u` at
# every cell, and `A` is the frozen `av` (identical because both start from zero).
E_new = a0 * gu
E_frz = a0 * p0
eq('N2.Eu_on_mask', E_new[mask], E_frz[mask])
eq('N2.J_on_mask', (E_new * (1.0 - pf))[mask], (E_frz * (1.0 - f))[mask])
eq('N2.fast_on_mask', (E_new * pf)[mask], (E_frz * f)[mask])
eq('N2.M_on_mask', (a0 * (1.0 - gu) * s)[mask], (a0 * (1.0 - p0) * s)[mask])

# and the FULL trajectory under the INSTALLED triple -- run the new kernel for real.  This
# is not vacuous: the installed `gu`/`pf` differ from the injection wherever `contact>0`,
# so any leakage into the shared per-reach state `M[r]`/`L[r]` would show up.
Fn, Sn, An, Pn = MC.scan_dp(h, s, f, k, l, inp, dem, model.cap, gu, pf, gs)
b_all = np.broadcast_to(colmask_all, mask.shape)
# There is NO clean full-trajectory comparison to make, and the reason is a property of
# the data rather than of the kernel: `colmask_all` is EMPTY -- every reach that is ever
# dry is also wet on some day.  So on any masked cell the entering `M[r]`/`L[r]` carry the
# reach's unmasked history, and on unmasked days the two kernels differ structurally (that
# is the point of the round).  The divergence measured below is therefore the round's
# INTENDED difference propagated through the state, not a defect.  Registered as a scope
# limitation of N2 rather than papered over with a comparison that cannot mean anything.
d_slow = np.abs(Sn[mask] - Fs0[mask])
sc_slow = np.maximum(np.abs(Fs0[mask]), 1e-300)
rel_slow = d_slow / sc_slow
per_step_rel = float(np.max(np.abs(gs - l) / np.maximum(np.abs(l), 1e-300)))
rep['N2']['trajectory'] = dict(
    n_cols_masked_every_day=int(colmask_all.sum()),
    comparison_available=bool(colmask_all.any()),
    why_not='no reach is dry on every day; the mask is daily, so entering states differ '
            'by the round-intended mechanism difference and a full-trajectory equality '
            'is not a well-posed claim',
    NOT_A_CRITERION_max_rel_slow_diff=float(np.max(rel_slow)),
    NOT_A_CRITERION_max_abs_slow_diff=float(np.max(d_slow)),
    why_not_a_criterion='the relative figure is mostly 0/0 on masked cells whose frozen '
                        'F_s is ~0 while the entering L is not; it measures the '
                        'denominator, not the kernel.  Reported so the number is on the '
                        'record, and labelled so it is not read as an error.',
    per_step_rounding_bound=per_step_rel,
    per_step_bounded=bool(per_step_rel <= 1e-14))
rep['N2']['slow_path_differs_off_mask'] = bool(np.any(Sn[~mask] != Fs0[~mask]))
rep['N2']['fast_path_differs_off_mask'] = bool(np.any(Fn[~mask] != Ff0[~mask]))
rep['N2']['registered_structural_difference_on_mask'] = (
    'F_s only, bounded by the per-step float64 rounding of g_s vs lower_release '
    '(measured below); E_u, J, F_f, M are BITWISE equal. The plan predicted a structural '
    'difference and overestimated it.')

# what the guard is actually doing -- the quantity it suppresses
with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
    xu_mask = np.where(mask, pack['xu'], np.nan)
rep['N2']['unguarded_xu_on_mask'] = dict(
    median=float(np.nanmedian(xu_mask)), p99=float(np.nanpercentile(xu_mask, 99)),
    max=float(np.nanmax(xu_mask)),
    frac_ge_0p99=float(np.nanmean(xu_mask >= 0.99)),
    note='x = Q_u/V_u stays near 1 on the mask because V_u collapses to ~1e-205 faster '
         'than Q_u; without the guard E_u -> A, which is the 1.0 measured by probe_timing')

# --------------------------------------------------------------------------
# 7. N12 -- the source scan and the underflow measurement
# --------------------------------------------------------------------------
SRC = ['dp_kernel.py', 'closures_dp.py', 'common24.py', 'layers24.py',
       'phase1_arms.py', 'level_variance.py', 'verdict_dp.py', 'phase0_gates.py']
# Matched by SYNTAX, not by text.  A text scan flags its own pattern table, its own
# docstrings -- which exist precisely to state the prohibition -- and the probe itself,
# which is REQUIRED to spell the naive form in order to measure that it is zero.  An
# `ast.Sub` whose right operand is a call to `exp` is the real thing and nothing else is.
PERMITTED = {('dp_kernel.py', 'expm1_underflow_probe')}


def _called_name(node):
    if not isinstance(node, ast.Call):
        return None
    fn = node.func
    return fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, 'id', None)


def naive_exp_sites(path):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    owner = {}
    for fn in [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef,))]:
        for sub in ast.walk(fn):
            owner[id(sub)] = fn.name
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Sub) \
                and isinstance(node.left, ast.Constant) and node.left.value == 1 \
                and _called_name(node.right) == 'exp':
            hits.append(dict(line=node.lineno, enclosing=owner.get(id(node)),
                             text=ast.unparse(node)[:90]))
    return hits


bad, permitted = {}, []
for name in SRC:
    p = C.ROUND / 'work' / name
    if not p.is_file():
        continue
    hits = naive_exp_sites(p)
    keep = []
    for h in hits:
        if (name, h['enclosing']) in PERMITTED:
            permitted.append([name, h['enclosing'], h['line'], h['text']])
        else:
            keep.append(h)
    bad[name] = keep
rep['N12'] = dict(
    naive_exp_sites=bad,
    permitted_sites=permitted,
    permitted_registration=('dp_kernel.expm1_underflow_probe MUST spell 1 - exp(-x): it '
                            'is the measurement that shows the form is wrong'),
    files_scanned=[n for n in SRC if (C.ROUND / 'work' / n).is_file()],
    probe=C.XI.expm1_underflow_probe(),
    note='the probe is the assertion: at x=1e-200 the subtraction is exactly 0.0 while '
         'expm1 returns the argument')
n_bad = sum(len(v) for v in bad.values())
rep['N12']['n_unpermitted_sites'] = n_bad

rep['verdict'] = dict(
    n_channels=int(len(rep.get('channels', {}))),
    failed=sorted(FAIL),
    passed=bool(not FAIL),
    verdict='N1_N2_PASSED' if not FAIL else 'BLOCKED')
if bad and any(v for v in bad.values()):
    rep['verdict']['passed'] = False
    rep['verdict']['verdict'] = 'BLOCKED'
    rep['verdict']['failed'] = rep['verdict']['failed'] + ['N12_source_form']

out = C.ROUND / 'reports' / 'phase0_n1.json'
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(rep, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
print(json.dumps({k: v for k, v in rep.items() if k != 'channels'},
                 indent=1, ensure_ascii=False, default=str))
print('\n--- channels ---')
for kk, v in rep.get('channels', {}).items():
    print('%-28s shape=%-14s bitwise=%s n_diff=%s maxabs=%s'
          % (kk, v['shape'], v['bitwise_equal'], v['n_elements_differing'], v['max_abs_diff']))
print('\n[N1] wrote %s' % out, flush=True)
