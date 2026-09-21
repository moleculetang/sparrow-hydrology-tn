"""Phase 0, assertions N1' + N2' + N12 + N13 -- the structural gates, run BEFORE any
forward.  Read-only, no arm forward.

DERIVED FROM `20260920_1/work/phase0_n1.py`.  THE DELTA, EXHAUSTIVELY
--------------------------------------------------------------------
  1. `import common24 as C` -> `import common25 as C`; `import closures_dp as MC` ->
     `import closures_dp2 as MC`.  A THIRD kernel import is added: `import closures_dp as
     MC1`, this round's own seeded copy of the PARENT kernel.
     WHY IT IS NEEDED.  Round 1's N1 asked the new kernel to reproduce the FROZEN
     `closures.scan` under the injection `g_u := prob_frozen`, `phi_f := f_frozen`,
     `g_s := l_frozen`.  That is still available here and is kept (`frozen.` columns
     below), but it is no longer the interesting edge: round 1 SHIPPED a kernel of its own
     (`closures_dp`), round 1's arm `P-upper` reports THAT kernel's readings, and this
     round's `q_m = 1` boundary has to reduce to it.  So N1' is stated against `MC1`, and
     the frozen comparison is carried as a second column.  `MC1` is byte-identical to
     `20260920_1/work/closures_dp.py`; verify with
     `diff -q work/closures_dp.py ../20260920_1/work/closures_dp.py`.
  2. Every parent-reduction call gains the FOURTH argument `q_m = 1.0`.  `q_m` is the
     `CONFIG` scalar in `closures_dp2`, not a fourth positional on the tag path, so the
     tag injection sets it through `set_config` while the direct `_tag_dp2_nb` call takes
     it positionally.  Both are exercised.
  3. The THIRTEEN channels are the same thirteen: four from the untagged kernel, seven
     from the tagged kernel, two from `Transport.apply`.  The channel NAMES and ORDER are
     round 1's, so the two JSON files can be read side by side.
  4. N2' is now a TWO-POOL mask statement.  Round 1 asked for five channels (`M`, `L`,
     `F_f`, `J`, `F_s`) on the mask.  This round has to state the plan S0.2 red line
     instead: `contact <= 0` forbids WATER EXPORT and does NOT forbid `N^L -> N^M`.  That
     is a statement about the INSTALLED arrays at a real `q_m`, not about a frozen
     injection, so the mask block is split in two and the second half drives `ledger_dp2`.
  5. NEW N13 `K_M_DEPENDS_ON_WATER`, both halves: a source-level half (no `Q`/`V` symbol
     may appear in the code of `q_m_of`/`tau_m_of` or anywhere in `closures_dp2`), and a
     DATA-level half (perturb `V_u` alone and require `T^{mob}` to be BITWISE unchanged
     while the water-driven `F_f` is not).
  6. N12's source list is this round's.  `phase0_gates.py` and `common25.py` are scanned
     too; `level_matched.py` and `shape_diag.py` are scanned when present, because they
     are written after this file and a scan that silently skips a missing file would make
     the assertion weaker in exactly the round that adds the most new files.

N1' IS THE ROUND'S MOST IMPORTANT GATE, AND ITS FAILURE MODE IS SILENT
---------------------------------------------------------------------
The new kernel has a scalar, a second state and a rewritten `pre`.  If any of those were
wrong, every arm would still run, still produce a plausible `169_476`-row frame and still
produce readings -- and the round would report them as if the parent reduction held.  N1'
is the only thing standing between "the new kernel is installed" and "some kernel is
installed".  It is therefore BITWISE (`np.array_equal`), not `allclose`: `allclose` would
pass on a kernel that used `1 - exp(-k_m)` instead of `-expm1(-k_m)`, which is the specific
substitution plan S2.1 item 5 forbids.
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common25 as C
import closures_dp2 as MC
import closures_dp as MC1

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


def claim(name, ok, detail=None):
    """A non-channel assertion.  Recorded with its evidence, and its failure is a STOP."""
    ok = bool(ok)
    node = dict(ok=ok)
    if detail is not None:
        node.update(detail)
    rep.setdefault('claims', {})[name] = node
    if not ok:
        FAIL.append(name)
    return ok


# --------------------------------------------------------------------------
# 1. the two kernels are what they claim to be, and the parent copy is the parent
# --------------------------------------------------------------------------
CLAIM_SRC = {'closures_dp2.py': 'closures_dp2.py', 'closures_dp.py': 'closures_dp.py'}
rep['kernel_sources'] = {k: dict(sha256=C.sha(C.ROUND / 'work' / v),
                                 bytes=int((C.ROUND / 'work' / v).stat().st_size))
                         for k, v in CLAIM_SRC.items()}
rep['parent_copy_is_the_parent'] = dict(
    parent_path=str(C.ROUND_PARENT / 'work' / 'closures_dp.py'),
    parent_round=('MODIFIED OR ABSENT' if C.sha(C.ROUND_PARENT / 'work' / 'closures_dp.py')
                  != rep['kernel_sources']['closures_dp.py']['sha256'] else 'IDENTICAL'),
    how_to_check='diff -q work/closures_dp.py ../20260920_1/work/closures_dp.py')
claim('parent_copy_matches_the_parent',
      C.sha(C.ROUND_PARENT / 'work' / 'closures_dp.py')
      == rep['kernel_sources']['closures_dp.py']['sha256'])

# --------------------------------------------------------------------------
# 2. the frozen parameters and the frozen fraction triple (round 1's own injection)
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
                     lower_release=list(l.shape), inp=list(inp.shape),
                     demand=list(dem.shape))
rep['f_is_not_fast_fraction'] = dict(
    f_max=float(f.max()), fast_fraction_max=float(np.max(model.data.fast_fraction)),
    n_differing=int(np.sum(f != np.asarray(model.data.fast_fraction, np.float64))),
    note='f = aq*ff/(aq*ff+1-ff); the two arrays are different objects by construction')

Ff0, Fs0, a0, p0 = _closed.scan(h, s, f, k, l, inp, dem, model.cap)
P0frozen, S0frozen, A0frozen, PP0frozen = MC1.scan_dp(
    h, s, f, k, l, inp, dem, model.cap, p0, f, l)
rep['parents_agree'] = dict(
    round1_kernel_vs_frozen=dict(
        fast=bool(np.array_equal(P0frozen, Ff0)), slow=bool(np.array_equal(S0frozen, Fs0)),
        available=bool(np.array_equal(A0frozen, a0)),
        prob=bool(np.array_equal(PP0frozen, p0))),
    note='round 1 already proved this; re-asserted here because N1\' compares MC against '
         'MC1 and a broken MC1 would make the gate vacuous')
for nm, aa, bb in (('fast', P0frozen, Ff0), ('slow', S0frozen, Fs0),
                   ('available', A0frozen, a0), ('prob', PP0frozen, p0)):
    claim('parent_kernel_agrees_with_frozen.' + nm, np.array_equal(aa, bb))

mask = np.asarray(model.data.contact, np.float64) <= 0.0
rep['mask'] = dict(
    n_cells=int(mask.sum()), n_total=int(mask.size), frac=float(mask.mean()),
    static_in_time=bool(np.all(mask == mask[:1])),
    n_masked_reaches_day0=int(mask[0].sum()),
    n_cols_masked_every_day=int(np.all(mask, axis=0).sum()),
    n_cols_never_masked=int(np.all(~mask, axis=0).sum()),
    h_exactly_zero_on_mask=bool(np.all(h[mask] == 0.0)),
    prob_exactly_zero_on_mask=bool(np.all(p0[mask] == 0.0)))

# --------------------------------------------------------------------------
# 3. N1' channel group A -- the untagged kernel, 4 channels, at q_m = 1
# --------------------------------------------------------------------------
fa, sa_, aa, pa, prea = MC.scan_dp2(h, s, f, k, l, inp, dem, model.cap, p0, f, l, 1.0)
eq('scan.fast', fa, P0frozen)
eq('scan.slow', sa_, S0frozen)
eq('scan.available', aa, A0frozen)
eq('scan.prob', pa, PP0frozen)
# the FIFTH output is new this round and has no round-1 counterpart to be equal to; at
# q_m = 1 it must be the parent's `A_frozen` exactly, because `N^{M,pre} = A` is what makes
# `E_M = A*g_u` (plan S2.1).  Asserted here rather than assumed.
eq('scan.mobile_pre_mobilisation', prea, A0frozen)

# --------------------------------------------------------------------------
# 4. N1' channel group B -- the tagged kernel, 7 channels, on the SLICED arrays
# --------------------------------------------------------------------------
rr = C.pilot_indices(model)
hs = np.ascontiguousarray(h[:, rr]); ss = np.ascontiguousarray(s[rr])
fs = np.ascontiguousarray(f[:, rr])
tag_inputs = np.ascontiguousarray(np.asarray(model.tag_inputs, np.float64))
tag_demand = np.ascontiguousarray(np.asarray(model.tag_demand, np.float64))
tag_release = np.ascontiguousarray(np.asarray(model.tag_release, np.float64))
assert tag_inputs.shape[1] == len(rr) and tag_release.shape[1] == len(rr)
assert ss.shape == (len(rr),), ss.shape
rep['tag_shapes'] = dict(h=list(hs.shape), s=list(ss.shape), f=list(fs.shape),
                         inputs=list(tag_inputs.shape), demand=list(tag_demand.shape),
                         release=list(tag_release.shape), n_pilot=len(rr))

T0 = _tagged.tag_scan(hs, ss, fs, tag_release, tag_inputs, tag_demand)
gu_s = np.ascontiguousarray(p0[:, rr]); pf_s = np.ascontiguousarray(f[:, rr])
gs_s = np.ascontiguousarray(l[:, rr])
T1 = MC1._tag_dp_nb(hs, ss, fs, tag_release, tag_inputs, tag_demand, gu_s, pf_s, gs_s)
T2 = MC._tag_dp2_nb(hs, ss, fs, tag_release, tag_inputs, tag_demand,
                    gu_s, pf_s, gs_s, 1.0)
TAG_CH = ('fast', 'slow', 'raw', 'mobile_state', 'slow_state', 'uptake', 'mineral_loss')
for i, nm in enumerate(TAG_CH):
    claim('parent_tag.' + nm, np.array_equal(T1[i], T0[i]))
    eq('tag.' + nm, T2[i], T1[i])

# the public wrapper too, so `_check`/`_check_q_m`'s shape and scalar assertions are
# exercised on the tag path
MC1.set_tag_slice(rr)
MC.set_tag_slice(rr)
MC1.set_config(p0, f, l, gu_s, pf_s, gs_s)
MC.set_config(p0, f, l, 1.0, gu_s, pf_s, gs_s)
T3 = MC.tag_scan_dp2(hs, ss, fs, tag_release, tag_inputs, tag_demand)
for i, nm in enumerate(TAG_CH):
    eq('tag_via_wrapper.' + nm, T3[i], T1[i])

# --------------------------------------------------------------------------
# 5. N1' channel group C -- Transport.apply, 2 channels
# --------------------------------------------------------------------------
_saved = getattr(model, 'dp_fractions', None)
model.dp_fractions = dict(gu=p0, phi_f=f, gs=l, guard=np.ones_like(p0), form='injected',
                          q_m=1.0)
C1 = MC.TransportDP2.apply(torch.from_numpy(h), torch.from_numpy(s),
                           torch.from_numpy(f), torch.from_numpy(k), model)
C0 = _closed.Transport.apply(torch.from_numpy(h), torch.from_numpy(s),
                             torch.from_numpy(f), torch.from_numpy(k), model)
eq('transport.fast', C1[0].numpy(), C0[0].numpy())
eq('transport.slow', C1[1].numpy(), C0[1].numpy())
if _saved is None:
    del model.dp_fractions
else:
    model.dp_fractions = _saved

# the injection is NOT the installed triple: if it were, N1' would be vacuous (a kernel
# that ignores its arguments passes every reduction test)
rep['injection_is_not_the_installed_triple'] = dict(
    n_cells_where_prob_differs_from_xu_times_guard=int(
        np.sum(p0 != MC1.CONFIG['gu'] if MC1.CONFIG['gu'] is not None else np.zeros_like(p0))),
    note='p0 is the frozen prob; the installed gu is x_u*guard -- different functions by '
         'design, and different on every live cell')

MC1.set_config(None, None, None)
MC.set_config(None, None, None)
MC1.set_tag_slice([])
MC.set_tag_slice([])

# --------------------------------------------------------------------------
# 6. N2' -- the mask, in two halves
# --------------------------------------------------------------------------
# 6a. the INJECTION half, bitwise at q_m = 1: on the mask the water export is exactly zero
eq('N2p.Eu_on_mask', (prea * p0)[mask], (a0 * p0)[mask])
eq('N2p.J_on_mask', (prea * p0 * (1.0 - f))[mask], (a0 * p0 * (1.0 - f))[mask])
eq('N2p.fast_on_mask', (prea * p0 * f)[mask], (a0 * p0 * f)[mask])

# 6b. the INSTALLED half, at a real two-pool q_m.  THIS IS THE PLAN S0.2 RED LINE.
Q_PRIMARY = C.q_m_of(200.0)
pack1 = C.dp_arrays(model, q_m=Q_PRIMARY, form='x')
g = pack1['guard']
rep['N2p'] = dict(
    guard_zero_set_is_contact_le0=bool(np.array_equal(g == 0.0, mask)),
    gu_exactly_zero_on_mask=bool(np.all(pack1['gu'][mask] == 0.0)),
    gu_equals_guard_times_xu=bool(np.array_equal(pack1['gu'], pack1['xu'] * g)),
    max_gu_off_mask=float(np.max(pack1['gu'][~mask])),
    q_m_used=float(Q_PRIMARY))
MC.set_config(pack1['gu'], pack1['phi_f'], pack1['gs'], pack1['q_m'])
led = MC.ledger_dp2(model, x)
fast1 = led['fast']; tr = led['transfer']
nLb = led['legacy_pool_before_transfer']
rep['N2p'].update(
    max_abs_fast_on_mask=float(np.max(np.abs(fast1[mask]))),
    n_transfer_on_mask=int(np.sum(tr[mask] > 0.0)),
    max_transfer_on_mask=float(np.max(tr[mask])),
    transfer_total_off_mask_REFUSED='not computed: plan S0.1 forbids reporting loads',
    n_legacy_pool_positive_on_mask=int(np.sum(nLb[mask] > 0.0)),
    mask_cells=int(mask.sum()),
    mask_is_daily_not_static=bool(not np.all(mask == mask[:1])))
claim('N2p.mask_forbids_water_export', rep['N2p']['max_abs_fast_on_mask'] == 0.0)
claim('N2p.mask_does_not_forbid_transfer',
      rep['N2p']['n_transfer_on_mask'] > 0 and rep['N2p']['max_transfer_on_mask'] > 0.0)

# --------------------------------------------------------------------------
# 7. N13 -- k_m does not depend on water.  DATA HALF.
# --------------------------------------------------------------------------
# The perturbation must move something on the WATER side or the test is vacuous, and the
# thing it must NOT move is `T^{mob}`.  `dp_alternate_Vu` swaps `V_u` alone (the producer
# DECLARES both `upper` and `soil`), leaving `V_s`, `Q` and `q_m` untouched.
pack_alt = C.dp_alternate_Vu(model, 'soil')
MC.set_config(pack_alt['gu'], pack_alt['phi_f'], pack_alt['gs'], pack_alt['q_m'])
led_alt = MC.ledger_dp2(model, x)
model.dp_fractions = pack1
MC.set_config(pack1['gu'], pack1['phi_f'], pack1['gs'], pack1['q_m'])

TOL2 = float(2.0 * np.spacing(Q_PRIMARY))


def rate_law(ledger):
    """`T^{mob}` against `N-tilde^L`: identical to `q_m*N-tilde^L`, and the recovered
    ratio equal to `q_m` up to the rounding of the division.

    `T / N-tilde^L` must be the SAME constant on every cell.  It is the scalar `q_m` up to
    float64 rounding: the kernel computes `T = fl(q_m*nL)`, and `fl(fl(q_m*nL)/nL)` need
    not return `q_m`.  The bound is stated as 2 ulp rather than hidden behind an inequality
    loose enough to also pass on a wrong constant.
    """
    t = ledger['transfer']; nl = ledger['legacy_pool_before_transfer']
    nz = nl > 0.0
    r = t[nz] / nl[nz]
    return dict(
        denominator='N-tilde^L (legacy_pool_before_transfer), NOT N^{L,*}',
        n_cells_with_positive_legacy_pool=int(nz.sum()),
        transfer_is_exactly_q_m_times_the_legacy_pool=bool(
            np.array_equal(t, Q_PRIMARY * nl)),
        max_abs_dev_of_ratio_from_q_m=float(np.max(np.abs(r - Q_PRIMARY))),
        ratio_tolerance_2ulp=TOL2,
        ratio_is_a_single_constant_within_2ulp=bool(
            np.max(np.abs(r - Q_PRIMARY)) <= TOL2))


rep['N13'] = dict(
    q_m_carried_unchanged_by_the_Vu_swap=bool(pack_alt['q_m'] == pack1['q_m']),
    vu_moves=bool(not np.array_equal(pack_alt['Vu'], pack1['Vu'])),
    gu_moves=bool(not np.array_equal(pack_alt['gu'], pack1['gu'])),
    phi_f_does_not_move=bool(np.array_equal(pack_alt['phi_f'], pack1['phi_f'])),
    fast_moves_on_real_cells=int(np.sum(led_alt['fast'] != fast1)))

# ---------------------------------------------------------------------------
# THE ONE-STEP TEST, AND WHY IT IS THE ONE THAT ANSWERS N13
# ---------------------------------------------------------------------------
# A first version of this block asserted `array_equal(led_alt['transfer'], led['transfer'])`
# and it FAILED -- correctly, because the claim was wrong.  `T` is not a function of water,
# but `T` IS a function of `N-tilde^L`, and `N-tilde^L` is a function of the STATE: the
# mobile pool's carry `N^M = pre*(1-g_u)*s_M` is water-flushed, so perturbing `V_u` changes
# tomorrow's `uP = N^L + inp + N^M`, hence `av`, hence `nL`, hence `T`.  That is the round's
# intended coupling (the whole point is that N is carried at the WATER's pace once it is
# mobile) and reporting it as "k_m depends on water" would be exactly backwards.
#
# The clean statement is therefore about the LAW, on the first day, where no state exists
# yet: on day 0 both pools start at zero, so `uL = uP = inp[0]`, `rL = 1.0`, `nL = av[0]`,
# `T = q_m*av[0]` -- no `g_u` appears anywhere and no state can carry the perturbation.  So
# `transfer[0]` must be BITWISE identical between the two packs while `fast[0]` is not.
one_step = dict(
    transfer_day0_is_bitwise_unchanged=bool(
        np.array_equal(led_alt['transfer'][0], led['transfer'][0])),
    fast_day0_moves=int(np.sum(led_alt['fast'][0] != fast1[0])),
    n_reaches_day0=int(led['transfer'].shape[1]),
    day0_has_no_state='both pools start at zero, so rL == 1 exactly and T == q_m*av')
rep['N13']['one_step'] = one_step
rep['N13']['rate_law_upper_Vu'] = rate_law(led)
rep['N13']['rate_law_soil_Vu'] = rate_law(led_alt)
rep['N13']['trajectory_feedback'] = dict(
    n_cells_where_transfer_differs=int(np.sum(led_alt['transfer'] != led['transfer'])),
    n_cells_where_the_legacy_pool_differs=int(np.sum(
        led_alt['legacy_pool_before_transfer'] != led['legacy_pool_before_transfer'])),
    mechanism='perturbing V_u moves g_u, which moves the mobile pool carry N^M, which '
             'moves tomorrow uP/av/nL, which moves T. State feedback, not a rate that '
             'reads water.',
    registered_as='INTENDED COUPLING, not an N13 violation')
claim('N13.one_step_transfer_is_water_free',
      one_step['transfer_day0_is_bitwise_unchanged'] and one_step['fast_day0_moves'] > 0)
for _nm, _key in (('upper', 'rate_law_upper_Vu'), ('soil', 'rate_law_soil_Vu')):
    claim('N13.rate_law_holds.%s' % _nm,
          rep['N13'][_key]['transfer_is_exactly_q_m_times_the_legacy_pool']
          and rep['N13'][_key]['ratio_is_a_single_constant_within_2ulp'])

# --------------------------------------------------------------------------
# 8. N13 -- k_m does not depend on water.  SOURCE HALF.
# --------------------------------------------------------------------------
# Scanned by AST, not by text: the docstrings in `closures_dp2.py` exist precisely to
# EXPLAIN that `T` carries no `Q` and no `V`, so a text scan would flag the prohibition's
# own statement.  A `Name` or attribute whose identifier is one of the water symbols, in
# non-docstring code, is the real thing and nothing else is.
WATER_SYMBOLS = {'Qf', 'Qp', 'Qs', 'Qu', 'Vu', 'Vs', 'xu', 'xs', 'x_u', 'x_s', 'flow',
                 'volume'}
TARGETS = [('closures_dp2.py', None), ('common25.py', ('q_m_of', 'tau_m_of'))]


def _stripped(path):
    """The module's AST with every docstring removed, so only code is scanned."""
    tree = ast.parse(path.read_text(encoding='utf-8'))
    for fn in ast.walk(tree):
        body = getattr(fn, 'body', None)
        if (isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                and body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            fn.body = body[1:]
    if (tree.body and isinstance(tree.body[0], ast.Expr)
            and isinstance(tree.body[0].value, ast.Constant)
            and isinstance(tree.body[0].value.value, str)):
        tree.body = tree.body[1:]
    return tree


n13_src = {}
for name, fns in TARGETS:
    p = C.ROUND / 'work' / name
    if not p.is_file():
        n13_src[name] = dict(present=False)
        continue
    tree = _stripped(p)
    scope = []
    for node in ast.walk(tree):
        if fns is None:
            scope.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in fns:
            scope.extend(ast.walk(node))
    hits = []
    for node in scope:
        ident = node.id if isinstance(node, ast.Name) else (
            node.attr if isinstance(node, ast.Attribute) else None)
        if ident in WATER_SYMBOLS:
            hits.append(dict(line=getattr(node, 'lineno', None), symbol=ident))
    # de-duplicate (a Name node inside a walked FunctionDef is visited once)
    seen = set(); uniq = []
    for hh in hits:
        key = (hh['line'], hh['symbol'])
        if key not in seen:
            seen.add(key); uniq.append(hh)
    n13_src[name] = dict(present=True, scanned_functions=(list(fns) if fns else 'ALL'),
                         n_water_symbol_references=len(uniq), hits=uniq[:20])
rep['N13_source'] = n13_src
claim('N13.no_water_symbol_in_the_rate_definition',
      all((not v['present']) or v['n_water_symbol_references'] == 0
          for v in n13_src.values()))

# --------------------------------------------------------------------------
# 9. N12 -- the source scan and the underflow measurement
# --------------------------------------------------------------------------
SRC = ['dp_kernel.py', 'closures_dp2.py', 'closures_dp.py', 'common25.py',
       'layers25.py', 'phase1_arms.py', 'level_variance.py', 'shape_diag.py',
       'level_matched.py', 'verdict_dp2.py', 'phase0_gates.py', 'audit_dp2.py']
PERMITTED = {('dp_kernel.py', 'expm1_underflow_probe'),
             ('common25.py', 'expm1_underlying_contrast')}


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
    keep = []
    for hh in naive_exp_sites(p):
        if (name, hh['enclosing']) in PERMITTED:
            permitted.append([name, hh['enclosing'], hh['line'], hh['text']])
        else:
            keep.append(hh)
    bad[name] = keep
rep['N12'] = dict(
    naive_exp_sites=bad, permitted_sites=permitted,
    permitted_registration=('TWO sites may spell `1 - exp(-x)`, and only because measuring '
                            'that the form is wrong REQUIRES writing it: '
                            '`dp_kernel.expm1_underflow_probe` (round 1, at x = 1e-200) and '
                            '`common25.expm1_underlying_contrast` (this round, at '
                            'k_m = 1e-200, per plan S3.5 N12 "本轮同时把它套在 q_m 上"). '
                            'Neither is on a production path: both are called only to '
                            'report the two numbers side by side. The registration is '
                            'listed here so the exemption is legible rather than silent.'),
    files_scanned=sorted(n for n in SRC if (C.ROUND / 'work' / n).is_file()),
    files_not_yet_written=sorted(n for n in SRC if not (C.ROUND / 'work' / n).is_file()),
    probe=C.XI.expm1_underflow_probe(),
    probe_q_m=C.expm1_underlying_contrast(),
    note='the probe is the assertion: at x=1e-200 the subtraction is exactly 0.0 while '
         'expm1 returns the argument.  This round applies the SAME measurement to the new '
         'scalar k_m (plan S3.5 N12), which is why there are two probes.')
n_bad = sum(len(v) for v in bad.values())
rep['N12']['n_unpermitted_sites'] = n_bad
claim('N12.no_naive_exponential_in_source', n_bad == 0)
claim('N12.k_m_underflow_measured',
      C.expm1_underlying_contrast()['naive_is_exactly_zero']
      and C.expm1_underlying_contrast()['stable_equals_k_m'])

# --------------------------------------------------------------------------
# 10. verdict
# --------------------------------------------------------------------------
rep['verdict'] = dict(
    n_channels=int(len(rep.get('channels', {}))),
    n_claims=int(len(rep.get('claims', {}))),
    failed=sorted(set(FAIL)),
    passed=bool(not FAIL),
    verdict='N1p_N2p_N12_N13_PASSED' if not FAIL else 'BLOCKED')
out = C.ROUND / 'reports' / 'phase0_n1.json'
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(rep, indent=2, ensure_ascii=False, default=str), encoding='utf-8')

print(json.dumps({kk: vv for kk, vv in rep.items()
                  if kk not in ('channels', 'N12')}, indent=1, ensure_ascii=False,
                 default=str))
print('\n--- channels (this round vs the PARENT kernel, at q_m = 1) ---')
for kk, vv in rep.get('channels', {}).items():
    print('%-30s bitwise=%-5s n_diff=%-8s maxabs=%s'
          % (kk, vv['bitwise_equal'], vv['n_elements_differing'], vv['max_abs_diff']))
print('\n[phase0 N1] wrote %s' % out, flush=True)
