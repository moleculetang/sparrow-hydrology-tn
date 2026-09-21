"""20260919_5 -- PHASE -1: the seven zero-forward screens, the stop rule, the k field.

WHAT THIS SCRIPT IS FOR
-----------------------
Round 5 asks ONE question -- was round 4's 80.1% event-amplitude gain bought by the
*path-concentration structure* or by *raising the long-run mobilisation level*?  The device
that asks it is `Xi* = k_r(beta) * Xi_raw`, with `k_r` solved from an internal ledger
identity so that the long-run mobilisable MASS is unchanged.  `phase_minus1.py` is the
screen that decides whether that device is even worth a forward, and it FREEZES `k_r`
before any observation is read.

THE EXECUTION ORDER, AND WHY IT IS NOT THE PLAN'S LISTING ORDER
---------------------------------------------------------------
The plan's section 1 lists C1 first and section 3.3 says the `k` field is frozen "BEFORE
reading any observation".  Those two cannot both hold in listing order, because C1 reads
the 4-HOURLY TN PANEL -- it scores a level change against monthly observations.  The plan
is explicit that the ordering rule wins ("先于读任何观测"), so the order actually executed
here is

    C6 -> C5 -> C3/C4 (inside the k solve) -> C2 -> *FREEZE + HASH* -> C1 -> C7

and the reversal is registered as a deviation, not presented as the plan's order.

THIS IS NOT LITERALLY "ZERO FORWARD", AND SAYING SO WOULD BE A FALSE CLAIM
--------------------------------------------------------------------------
The plan's header says Phase -1 is "零前向".  Six of the seven items are: C6 is metadata,
C5 is frozen arrays, C2/C3/C4 are frozen-array sums, C7 is one autograd pass of the frozen
router.  C1 IS NOT.  Two forwards are charged to this script and both are named:

  * ONE baseline forward (`layers23.forward_layers` at beta = 0 with the kernel NOT
    installed) inside C1.  The plan says C1 should use the STORED beta = 0 station-month
    `pL3` series; round 4 stored no such series (`reports/daily_layers.parquet` is a
    round-5 deliverable precisely because its absence was a round-4 gap), so the series is
    reproduced.  It is the SAME forward `phase1_full.py` will compute as its null point, so
    it is SHARED, not a duplicate -- but it is still a forward and is reported as one.
  * The `k` solve itself is NOT a forward: it calls the frozen `closures.scan` on the land
    phase only, never the router and never the observation operator.  That distinction is
    the plan's design-review point (b)/(c) and it is what keeps 4 devices x 19 betas inside
    minutes rather than hours.

C1's ONE FORWARD IS ALSO THE ANCHOR REPLAY: the reproduced baseline is asserted against
the registered `nse`, `median_station_nse` and `mean_concentration` at 1e-12.  If it does
not reproduce, this script stops -- a C1 whose baseline is not the baseline is not a screen
worth reading.
"""
import json
import time

import numpy as np
import pandas as pd

import common23 as C
import eventlib as E
import layers23 as LY
import xi_k as XI

R = C.ROUND
OUT = R / 'reports'

# ---------------------------------------------------------------- tolerances
TOL_LIN = 1e-4          # plan 2.2: rho_lin below this and the closed form IS the product
TOL_K = 1e-8            # plan 2.2, criterion 1: max_r |dk_r|/k_r
TOL_RHO = 1e-10         # plan 2.2, criterion 2: max_r rho_r, the ledger residual at the
                        # state the accepted k actually carries.  The register states BOTH
                        # and they are not redundant: in this model rho/step ~ 1, so this
                        # bound is ~50x the tighter of the pair and an orbit that stops on
                        # the step alone still fails it.
MAX_ITER = 64           # plan 2.2: the fixed-point iteration cap; non-convergence is a
                        # RESULT, never a silent fallback

# ------------------------------------------------- C2: the window-transfer ratio
GAMMA_LO, GAMMA_HI = 0.80, 1.25     # plan section 1 C2, the registered band
GAMMA_MAX_OUT = 4                   # plan section 1 C2, "4 of 15"
GAMMA_N = 15

# ------------------------------------------------- C1: the pure-level stop rule
C1_HIGH, C1_LOW = 0.65, 0.55        # plan section 1 C1
# The sweep is a CROSS-CHECK, so its range must be wide enough to bracket the closed form
# rather than to define the answer.  A sweep that is not wide enough would report a
# boundary point and look like a disagreement; the assertion below is therefore
# one-sided (the grid can never BEAT the exact optimum, it can only fail to reach it).
C1_COARSE = np.round(np.arange(-0.99, 20.0 + 1e-9, 0.05), 10)
C1_FINE_HALF_WIDTH = 0.05
C1_FINE_N = 201
C1_BRACKET_EPS = 1e-6               # the closed form must be a local min to this width

# ------------------------------------------------- C5: the two census floors
# `DEGEN_FLOOR` is registered (plan 3.1: `degen_floor = 0.001`).  The ill-conditioning
# floor is NOT registered anywhere, so it is DECLARED here, at the same magnitude, and
# reusing the registered number rather than inventing a second one is the point.
ILL_CONDITIONED_FLOOR = C.DEGEN_FLOOR


def _p(msg):
    print(msg, flush=True)


# ==========================================================================
# THE SHARDED EXECUTION OF THE k FIELD
# ==========================================================================
# WHY THIS EXISTS.  The registered run is ONE process: `device_state` and `solve_k` are
# pure, `torch.set_num_threads(1)` is pinned by `common23`, and the frozen `closures.scan`
# is a serial `@njit`.  Measured on this machine, one point costs 0.8 s (beta = 0) to
# ~370 s (|beta| = 2), so the 76 points are hours of ONE core while 31 sit idle.
#
# THE POINTS ARE INDEPENDENT, AND THAT IS MEASURED, NOT ASSUMED.  `solve_k` is a pure
# function of `(B, G, device, beta)` -- it takes `B` and `G` as arguments and mutates
# neither (the only process-global state in this round, `install_kernel`, is NOT touched by
# the solve at all; it is `phase1_full.py`'s, and there it forces sequential points WITHIN a
# process).  Nothing in the k solve reads another point's result.  So the registered point
# ORDER is a presentation order, not a dependency order, and any partition of the 76 points
# gives the same 230-vector per point.  `--verify` re-derives a chosen subset in ONE process
# from a FRESHLY BUILT MODEL -- not from the cache -- and compares `k_sha256` per point, so
# the claim "sharding is neutral" ships with a reading rather than an argument.
#
# WHY A SHARED mmap CACHE AND NOT "EACH WORKER BUILDS ITS OWN MODEL".  Measured, same
# machine, one worker each:
#     build model + bootstrap, then `del model`        RSS 2720 MB, peak 3358 MB
#       (`del model` frees NOTHING: `B` pins it through three VIEWS -- `lower_release` is a
#        memmap, `inp`/`demand` are ndarrays with a `.base` -- and `geometry` memoises the
#        model itself.  Measured, not deduced.)
#     detach those views, release the model            RSS 1999 MB, peak 2848 MB
#     load a mmap cache, never build the model         RSS 1126 MB, PEAK COMMIT 2069 MB
# The binding resource on this box is NOT physical RAM but COMMIT CHARGE: `CommitLimit` is
# 63.8 GB with essentially no pagefile, and ~45 GB of it was already held by other
# processes, so only ~19 GB of headroom remained.  `freeCommit` was sampled during a
# 16-worker ramp and fell to 63 MB -- which is what killed 9 of those 16 with
# `Unable to allocate 41.0 MiB` while 13 GB of PHYSICAL memory was still free.  At 2069 MB
# of peak commit per worker, ~9 fit in that headroom; the launcher below is sized from that
# number and staggers the starts so the peaks do not coincide.
#
# READ-ONLY FILE PAGES COST NO COMMIT.  Measured: `commit` was 785.2 MB before
# `np.load(mmap_mode='r')` and 785.2 MB after -- the 538 MB of arrays are page-cache backed
# and shared by every worker, so N workers read ONE copy of the inputs.
B_CACHE_KEYS = ('h', 's', 'f', 'k', 'a0', 'p0', 'inp', 'demand', 'lower_release', 'S2',
                'dates')
G_CACHE_KEYS = ('Qf', 'Qs', 'W', 'Weff', 'active', 'clim', 'u', 'sd_r', 'ref')
CACHE_NPZ = C.ROUND / 'work' / '_k_inputs.npz'
CACHE_MANIFEST = C.ROUND / 'work' / '_k_inputs_manifest.json'
SHARD_DIR = C.ROUND / 'work' / '_k_shards'
VERIFY_POINTS = (('N1', 0.0), ('N1', -0.05), ('N1', 0.05), ('N1', 0.5),
                 ('N2', -0.05), ('N3', 0.05), ('N1e', 0.0), ('N2', 0.0), ('N3', 0.0))


def sha_array(a):
    """sha256 of an array's RAW BYTES, dtype-agnostic.

    `datetime64` has no buffer interface, so it is viewed as `int64` first -- without that
    the manifest cannot cover `B['dates']` and the window mask would be unverified.
    """
    import hashlib
    a = np.ascontiguousarray(a)
    if a.dtype.kind == 'M':
        a = a.view('int64')
    return hashlib.sha256(memoryview(a).cast('B')).hexdigest()


def point_order():
    """THE registered order: device-major, then beta in `C.BETA_GRID` order."""
    return [(d, float(b)) for d in C.DEVICES for b in C.BETA_GRID]


def device_block(device, Wt):
    """The per-device part of `kf` that is NOT a point.  One construction, two callers."""
    return dict(target=C.DEVICE_SPEC[device]['target'],
                window=C.DEVICE_SPEC[device]['window'],
                window_years=list(C.WINDOWS[C.DEVICE_SPEC[device]['window']]),
                n_window_days=int(Wt.sum()),
                points={})


def solve_point(B, G, device, beta):
    """ONE point of the registered k field.  Pure over `(B, G, device, beta)`.

    THIS IS THE ONLY PLACE A POINT IS BUILT, so the single-process path and a shard worker
    cannot drift apart: `main()` calls it, `--shard` calls it, `--verify` calls it.
    """
    ts = time.time()
    res = C.solve_k(B, G, device, beta, tol_lin=TOL_LIN, tol_k=TOL_K,
                    tol_rho=TOL_RHO, max_iter=MAX_ITER)
    keep = {k: v for k, v in res.items()
            if k in ('beta', 'n_reaches', 'k_sha256', 'max_rho_lin',
                     'max_rho_exact', 'adoptable', 'n_blocked_reaches',
                     'adoptable_registered_route_only',
                     'n_blocked_registered_route_only', 'fixed_point',
                     'n_no_finite_fixed_point', 'n_degenerate_no_hazard',
                     'n_degenerate_no_available_n', 'n_neutral_xi',
                     'n_fixed_point_solved', 'n_fixed_point_converged',
                     'status_counts', 'k_quantiles', 'n_k_below_one',
                     'n_k_above_one', 'n_k_exactly_one', 'frac_k_nonunit',
                     'k_uniform', 'headroom_closed_form_agrees',
                     'headroom_naive_excess', 'n_h_clipped_to_700',
                     'orbit')}
    keep['k'] = [float(v) for v in res['k']]
    keep['rho_exact'] = [float(v) for v in res['rho_exact']]
    keep['rho_lin'] = [float(v) for v in res['rho_lin']]
    keep['headroom'] = [float(v) for v in res['headroom']]
    keep['headroom_plan_closed_form'] = [float(v) for v in
                                         res['headroom_plan_closed_form']]
    keep['headroom_telescoped'] = [float(v) for v in res['headroom_telescoped']]
    keep['headroom_naive_plan_F_inf'] = [float(v) for v in res['headroom_naive_plan_F_inf']]
    keep['T_target'] = [float(v) for v in res['T_target']]
    keep['G_at_k'] = [float(v) for v in res['G_at_k']]
    keep['G_at_k1'] = [float(v) for v in res['G_at_k1']]
    keep['k_sign'] = res['k_sign']
    keep['seconds'] = round(time.time() - ts, 3)
    return keep


def _detach(B, G):
    """Replace every array that pins the model with an OWNED copy, and drop the layer
    outputs the identity never reads.  Each replacement is asserted `array_equal` against
    the original: the copy may be differently OWNED, never DIFFERENT."""
    for key in ('inp', 'demand', 'lower_release'):
        orig = B.get(key)
        if isinstance(orig, np.ndarray) and orig.base is not None:
            copy = np.array(orig, copy=True)
            if not np.array_equal(copy, orig):
                raise SystemExit('DETACH_CHANGED_THE_ARRAY %s' % key)
            B[key] = copy
    for key in ('fast', 'slow'):
        B[key] = None
    for key in list(G):
        v = G[key]
        if isinstance(v, np.ndarray) and v.base is not None:
            copy = np.array(v, copy=True)
            if not np.array_equal(copy, v):
                raise SystemExit('DETACH_CHANGED_G %s' % key)
            G[key] = copy
    return B, G


def build_manifest(B, G):
    """The sha/shape/dtype of every array the identity reads.  ONE construction, used by
    the writer (`prepare_inputs`) and by the checker (`load_shard_field`), so the two
    cannot disagree about what is being compared."""
    manifest = {}
    for k in B_CACHE_KEYS:
        a = np.asarray(B[k])
        manifest['B.' + k] = dict(sha256=sha_array(a), shape=list(a.shape),
                                  dtype=str(a.dtype))
    for k in G_CACHE_KEYS:
        a = np.asarray(G[k])
        manifest['G.' + k] = dict(sha256=sha_array(a), shape=list(a.shape),
                                  dtype=str(a.dtype))
    return manifest


def prepare_inputs(model=None):
    """Build once, detach, and write the frozen solve inputs to a shared mmap cache."""
    if model is None:
        model = C.build()
    G = C.geometry(model)
    B = C.bootstrap_arrays(model)
    B, G = _detach(B, G)
    payload = {}
    for k in B_CACHE_KEYS:
        payload['B.' + k] = np.asarray(B[k])
    for k in G_CACHE_KEYS:
        payload['G.' + k] = np.asarray(G[k])
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE_NPZ, **payload)
    manifest = build_manifest(B, G)
    CACHE_MANIFEST.write_text(json.dumps(manifest, indent=1, sort_keys=True),
                              encoding='utf-8')
    return manifest


def load_inputs():
    """Reconstruct `(B, G)` from the mmap, VERIFYING every array against the manifest.

    The sha check is the point of the cache.  A cached input is admissible only if it is
    PROVEN to be the array the model produced, so the manifest is recomputed and compared
    on every load rather than trusted -- and `--merge` re-verifies the manifest against a
    freshly built model, which closes the loop from the model to the shard.
    """
    z = np.load(CACHE_NPZ, mmap_mode='r')
    manifest = json.loads(CACHE_MANIFEST.read_text(encoding='utf-8'))
    B, G = {}, {}
    if sorted(z.files) != sorted(manifest):
        raise SystemExit('CACHE_AND_MANIFEST_DISAGREE_ON_THE_KEY_SET %d vs %d'
                         % (len(z.files), len(manifest)))
    for name in z.files:
        a = z[name]
        if list(a.shape) != manifest[name]['shape'] or str(a.dtype) != manifest[name]['dtype']:
            raise SystemExit('CACHED_ARRAY_SHAPE_OR_DTYPE %s' % name)
        if sha_array(a) != manifest[name]['sha256']:
            raise SystemExit('CACHED_ARRAY_DISAGREES_WITH_THE_MANIFEST %s' % name)
        (B if name.startswith('B.') else G)[name.split('.', 1)[1]] = a
    B['shape'] = list(B['h'].shape)
    B['cap'] = False
    B['k'] = np.asarray(B['k'])
    G['S_u'] = float(np.median(G['sd_r']))
    return B, G, manifest


def manifest_sha(manifest):
    import hashlib
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


def load_shard_field(B, G, n_shards):
    """Assemble the registered k field from the shard files, PROVING three things.

    1. THE SHARDS ALL USED THE SAME INPUTS.  Every shard records the sha256 of the manifest
       it loaded; all of them must agree with each other AND with a manifest rebuilt from
       the `(B, G)` THIS process just built from the model.  That is the loop closure that
       makes the cache admissible: the arrays a shard read are the arrays the model
       produces, checked here, not asserted there.
    2. THE PARTITION IS EXACTLY A PARTITION.  Each of the 76 registered points appears in
       exactly one shard, no point is missing, and no shard carries an unregistered point.
    3. THE ASSEMBLY ORDER IS THE REGISTERED ORDER, not the order the shards finished in.
    """
    order = point_order()
    files = sorted(SHARD_DIR.glob('k_shard_*.json'))
    if len(files) != n_shards:
        raise SystemExit('SHARD_FILE_COUNT_%d_EXPECTED_%d' % (len(files), n_shards))
    mine_sha = manifest_sha(build_manifest(B, G))
    seen, blocks, shards = {}, {}, []
    for path in files:
        rec = json.loads(path.read_text(encoding='utf-8'))
        if int(rec['n_shards']) != int(n_shards):
            raise SystemExit('SHARD_%s_DECLARES_N_%s' % (path.name, rec['n_shards']))
        if rec['inputs_manifest_sha256'] != mine_sha:
            raise SystemExit('SHARD_%s_INPUTS_DIFFER_FROM_THIS_PROCESS_MODEL'
                             % path.name)
        for i in rec['assigned_global_indices']:
            if i in seen:
                raise SystemExit('POINT_%d_IN_TWO_SHARDS' % i)
            seen[i] = path.name
        for device, pts in rec['blocks'].items():
            blocks.setdefault(device, {}).update(pts)
        shards.append(dict(file=path.name, shard_index=int(rec['shard_index']),
                           n_points=int(rec['n_points']),
                           elapsed_seconds=rec['elapsed_seconds']))
    if sorted(seen) != list(range(len(order))):
        raise SystemExit('SHARDS_DO_NOT_COVER_THE_REGISTERED_POINTS %r' % sorted(seen))
    kf = {}
    for device in C.DEVICES:
        Q, Wt = C.device_state(B, G, device)
        kf[device] = device_block(device, Wt)
        for beta in C.BETA_GRID:
            key = '%g' % beta
            if key not in blocks.get(device, {}):
                raise SystemExit('MISSING_POINT_%s_%s' % (device, key))
            kf[device]['points'][key] = blocks[device].pop(key)
    leftover = {d: sorted(p) for d, p in blocks.items() if p}
    if leftover:
        raise SystemExit('SHARDS_CARRY_UNREGISTERED_POINTS %r' % leftover)
    for device in C.DEVICES:                       # registered order, re-stated here
        for beta in C.BETA_GRID:
            keep = kf[device]['points']['%g' % beta]
            _p('    %-4s beta=%-6g %7.2fs  rho_lin=%.3e  fp=%d iters=%d esc=%d un=%d  '
               'adoptable=%s (registered-route-only=%s)  nonunit=%.3f  [%s]'
               % (device, beta, keep['seconds'], keep['max_rho_lin'],
                  keep['n_fixed_point_solved'], keep['fixed_point'].get('n_orbit_iter', 0),
                  keep['fixed_point'].get('n_resolved_after_escalation', 0),
                  keep['fixed_point'].get('n_unresolved', 0), keep['adoptable'],
                  keep['adoptable_registered_route_only'], keep['frac_k_nonunit'],
                  seen[order.index((device, float(beta)))]))
    rep = dict(used=True, n_shards=int(n_shards),
               inputs_manifest_sha256=mine_sha,
               inputs_manifest_verified_against_a_freshly_built_model=True,
               partition_is_exact=True, assembly_order='registered (device-major, beta '
               'in grid order), not completion order',
               sum_of_shard_seconds=round(sum(s['elapsed_seconds'] for s in shards), 2),
               shards=sorted(shards, key=lambda s: s['shard_index']),
               launcher='a fixed number of staggered `--shard i/N` processes; the shard '
                        'count is sized from the MEASURED peak commit charge per worker, '
                        'not from the core count')
    return kf, rep


def run_shard(index, n_shards):
    """Solve the points whose REGISTERED global index is `index (mod n_shards)`."""
    order = point_order()
    mine = [(i, d, b) for i, (d, b) in enumerate(order) if i % n_shards == index]
    B, G, manifest = load_inputs()
    msha = manifest_sha(manifest)
    out = {}
    t0 = time.time()
    for i, device, beta in mine:
        keep = solve_point(B, G, device, beta)
        out.setdefault(device, {})['%g' % beta] = keep
        _p('  [shard %d/%d] %2d/76 %-4s beta=%-6g %7.2fs  sha=%s  nonunit=%.3f'
           % (index, n_shards, i, device, beta, keep['seconds'], keep['k_sha256'][:16],
              keep['frac_k_nonunit']))
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    rec = dict(shard_index=index, n_shards=n_shards, inputs_manifest_sha256=msha,
               registered_point_order=[['%s' % d, float(b)] for d, b in order],
               assigned_global_indices=[i for i, _d, _b in mine],
               blocks=out, elapsed_seconds=round(time.time() - t0, 2),
               n_points=len(mine))
    sha = C.write_json(SHARD_DIR / ('k_shard_%03d.json' % index), rec)
    _p('  [shard %d/%d] wrote %d points in %.1fs  sha=%s'
       % (index, n_shards, len(mine), rec['elapsed_seconds'], sha[:16]))
    return 0


# ==========================================================================
# C6 -- the boundary-code census (run FIRST: everything else depends on it)
# ==========================================================================
def c6_boundary_census(model):
    """Every evaluated station-day must be an ORDINARY INTERNAL reach.

    `routing.py:133-135` overrides the boundary mass with a reservoir release (`code == 1`)
    or an official external series (`code == 2`).  On either of those the device's `k_r`
    does NOT control the mass at all, so a single such station-day would silently remove
    that station from the device's reach.  Round 3 verified there are none; this round
    RE-RUNS the census rather than inheriting the conclusion, because the conclusion is
    load-bearing for the whole design.
    """
    meta = pd.read_parquet(C.PEER / 'data/prediction_calendar.parquet')
    meta = meta[meta.year.le(C.END_YEAR)].copy()
    c, _record, _w = model.daily_metadata(meta)
    code = c['boundary_code'].numpy()
    st = meta.station_type.astype(str)
    uniq_type = sorted(st.unique().tolist())
    out = dict(
        n_calendar_rows=int(len(meta)),
        n_station_days_expanded=int(code.size),
        n_stations=int(meta.station_key.nunique()),
        station_type_counts={k: int(v) for k, v in st.value_counts().items()},
        n_distinct_station_type=len(uniq_type),
        boundary_code_values=[int(v) for v in np.unique(code)],
        boundary_code_counts={str(int(v)): int((code == v).sum())
                              for v in np.unique(code)},
        reservoir_index_distinct=[int(v) for v in np.unique(c['reservoir_index'].numpy())],
        all_ordinary_internal=bool(len(uniq_type) == 1
                                   and uniq_type[0] == 'ordinary_internal'
                                   and np.array_equal(np.unique(code), np.array([0]))),
        consequence='if this is False the device loses control of the affected '
                    'station-days entirely and the round must stop')
    if not out['all_ordinary_internal']:
        raise SystemExit('C6_BOUNDARY_CODE_CENSUS_CHANGED %r' % out)
    return out


# ==========================================================================
# C5 -- the reach census, the degeneracy tags, the pin footprint
# ==========================================================================
def c5_reach_census(B, G, beta_census=(0.5, -0.5)):
    """Per-reach facts about the cells the identity is written on.

    Reported, never dropped: the reaches where the device is DEGENERATE, the reaches where
    it is ill-conditioned, and -- the item the plan says MUST be restated rather than
    silently omitted -- the cells where `active == False` while `h != 0`.  On those cells
    `xi_base.xi_from` pins `Xi` to exactly 1.0, and `k_r` is then multiplied ON TOP, so the
    pin is not inert there.
    """
    h, a0, p0, S2 = B['h'], B['a0'], B['p0'], B['S2']
    active = G['active']
    nr = h.shape[1]
    assert active.ndim == 2 and active.shape == h.shape, active.shape

    pin = ~active
    h_pin = h[pin]
    live = pin & (h != 0.0)
    footprint = dict(
        n_active_cells=int(active.sum()), n_inactive_cells=int(pin.sum()),
        n_inactive_and_h_nonzero=int(live.sum()),
        max_h_on_pinned_cells=float(h_pin.max()) if h_pin.size else 0.0,
        max_h_on_pinned_and_nonzero=float(h[live].max()) if live.any() else 0.0,
        restatement='THE PIN IS NOT INERT ON THE CELLS COUNTED ABOVE: `xi_base.xi_from` '
                    'sets Xi = 1.0 exactly where active is False, but the device then '
                    'MULTIPLIES k_r on top, so on those cells a non-zero hazard is '
                    'rescaled.  Registered in the plan (section 2.1 / C5) and restated '
                    'here rather than dropped.',
        max_h_registered_in_round4=0.000473712904325966,
        max_h_matches_round4=bool(abs(float(h[live].max() if live.any() else 0.0)
                                      - 0.000473712904325966) <= 1e-18),
    )

    # the reference device's window and normaliser, for the mass-share readouts
    Q, Wt = C.device_state(B, G, 'N1')
    win = Wt[:, None]
    mw = win * Q[None, :]
    mass_all = (a0 * p0 * mw).sum(axis=0)
    n = (B['inp'] - B['demand'])
    M = a0 * (1.0 - p0) * S2
    Mprev = np.vstack([np.zeros((1, nr)), M[:-1]])
    u = np.maximum(-(Mprev + n), 0.0)

    per = {}
    for b in beta_census:
        Xr = XI.xi_raw(G, b)
        off = win & (Xr != 1.0)
        mass_off = (a0 * p0 * mw * off).sum(axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            share = np.where(mass_all > 0, mass_off / mass_all, 0.0)
        per['%+.2f' % b] = dict(
            frac_cells_xi_ne_one=float(off.sum() / max(1, win.sum())),
            mass_share_on_xi_ne_one=float((mass_off.sum() / mass_all.sum())
                                          if mass_all.sum() > 0 else 0.0),
            n_reaches_below_small_sample_floor=int((share < C.DEGEN_FLOOR).sum()),
            small_sample_floor=C.DEGEN_FLOOR,
            tag='K_SMALL_SAMPLE where the share is below the floor: a reach whose '
                'Xi != 1 cells carry negligible mass cannot be discriminated by k_r at '
                'that beta, which is a statement about beta, not about the reach')

    out = dict(
        footprint=footprint,
        n_active_cells_per_reach_quantiles={
            str(q): float(v) for q, v in
            zip((0, 1, 50, 99, 100), np.percentile(active.sum(axis=0), (0, 1, 50, 99, 100)))},
        u_abs_max=float(np.abs(u).max()), u_abs_p50=float(np.median(np.abs(u))),
        n_u_nonzero_cells=int((u > 0).sum()),
        xi_off_cells=per,
        note='the census is on the beta = 0 FROZEN arrays: `active`, `h`, `a0`, `p0`, '
             '`inp - demand`.  Nothing here reads an observation.',
    )
    return out


# ==========================================================================
# C2 -- the window-transfer ratio Gamma_r
# ==========================================================================
def _lift(B, G, beta, Q, W):
    """`Lambda_r(W) = sum_W av0 (1 - e^{-h Xi}) / sum_W av0 (1 - e^{-h})  - 1`.

    This is the UN-NEUTRALISED lift at `k = 1`, i.e. WITHOUT the device: the quantity
    `k_r` is asked to cancel.  `Gamma_r` therefore measures whether cancelling it on the
    FULL record also cancels it on the EVALUATION window -- if it does not, no amount of
    accuracy in the solve can make the device neutral on the scored window.

    `W` is a `(nd,)` boolean window mask, or `None` for the full record.  The RATIO is
    computed on the same frozen `av0` for numerator and denominator, so it is a pure
    hazard-shape statement.
    """
    Xr = XI.xi_raw(G, beta)
    # SHAPES ARE ASSERTED HERE, NOT LEFT TO THE BROADCAST.
    # `av0` is PER-CELL `(nd, nr)`; `Q` is PER-REACH `(nr,)`; `W` is PER-DAY `(nd,)`.
    # The weight is therefore `av0 * Q[None, :]`.  This line was first written as
    # `B['a0'][:, None] * Q[None, :]` -- as if `av0` were the per-DAY array `W` is --
    # which broadcasts to `(nd, nr, nr)`: a 936 GiB allocation.  That raises rather than
    # corrupting, which is the only reason it was found; the same class of error in
    # `phase0_freeze.hxi_precondition` was SILENT.  Both were found by running the code.
    a0 = np.asarray(B['a0'], float)
    Q = np.asarray(Q, float)
    if a0.ndim != 2 or Q.shape != (a0.shape[1],):
        raise ValueError('LIFT_SHAPE av0=%r Q=%r' % (a0.shape, Q.shape))
    if W is not None and np.asarray(W).shape != (a0.shape[0],):
        raise ValueError('LIFT_WINDOW_SHAPE %r vs %d days'
                         % (np.asarray(W).shape, a0.shape[0]))
    L = a0 * Q[None, :]
    num = ((L * -np.expm1(-np.minimum(B['h'] * Xr, XI.CLIP)))
           * (W[:, None] if W is not None else 1.0)).sum(axis=0)
    den = ((L * -np.expm1(-np.minimum(B['h'], XI.CLIP)))
           * (W[:, None] if W is not None else 1.0)).sum(axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        lam = np.where(den > 0, num / den - 1.0, 0.0)
    return lam


def c2_window_transfer(B, G):
    """Gamma_r for the 15 scored stations' 13 reaches, for EVERY beta in the grid.

    THE AGGREGATION IS PRE-REGISTERED AND IS SELECTION-FREE.  A station is OUT if Gamma_r
    leaves [0.80, 1.25] at ANY beta in the 19-point grid.  The rule is conservative on
    purpose: any aggregation that first picks a beta would be a window choice made after
    seeing a result, which the plan forbids outright.  The per-beta OUT counts are reported
    beside the aggregate so a reader can see exactly which beta drives the verdict.
    """
    sr, reaches = C.station_reaches()
    assert len(sr) == GAMMA_N, (len(sr), GAMMA_N)
    one = np.ones(int(B['h'].shape[1]))
    W_eval = C.window_mask(B['dates'], C.EVAL_YEARS)
    mass = {}
    for beta in C.BETA_GRID:
        le = _lift(B, G, beta, one, W_eval)
        lf = _lift(B, G, beta, one, None)
        with np.errstate(divide='ignore', invalid='ignore'):
            g = np.where(np.abs(lf) > 0, le / lf, 1.0)
        mass['%g' % beta] = dict(
            gamma=[float(g[r]) for r in reaches],
            lambda_eval=[float(le[r]) for r in reaches],
            lambda_full=[float(lf[r]) for r in reaches],
            n_out=int(sum(1 for r in reaches if not (GAMMA_LO <= g[r] <= GAMMA_HI))),
        )
    # the same quantity weighted the way a CONCENTRATION device weights it
    conc = {}
    for device in C.DEVICES:
        if C.DEVICE_SPEC[device]['target'] != 'conc':
            continue
        Q, _Wt = C.device_state(B, G, device)
        le = _lift(B, G, 0.5, Q, W_eval)
        lf = _lift(B, G, 0.5, Q, None)
        with np.errstate(divide='ignore', invalid='ignore'):
            g = np.where(np.abs(lf) > 0, le / lf, 1.0)
        conc[device] = dict(gamma=[float(g[r]) for r in reaches],
                            n_out=int(sum(1 for r in reaches
                                           if not (GAMMA_LO <= g[r] <= GAMMA_HI))),
                            beta=0.5,
                            note='the plan writes Lambda without the device normaliser; '
                                 'this is the same ratio with 1/W, reported so the two '
                                 'spellings cannot be confused')
    worst = int(max(v['n_out'] for v in mass.values()))
    # the rule is applied to the MASS spelling, which is the plan's literal definition
    triggered = bool(worst > GAMMA_MAX_OUT)
    return dict(
        stations=list(sr), reaches_0based=[int(r) for r in reaches],
        band=[GAMMA_LO, GAMMA_HI], max_out_allowed=GAMMA_MAX_OUT, n_stations=GAMMA_N,
        per_beta=mass, conc_weighted_at_half=conc,
        worst_n_out_across_the_grid=worst,
        rule='a station is OUT if Gamma_r leaves the band at ANY beta in the 19-point '
             'grid; the rule triggers if more than 4 of 15 are OUT',
        triggered=triggered,
        trigger_consequence='the primary device becomes the EVALUATION-WINDOW mass variant '
                            '(N1e), decided here, before any forward',
        chosen_primary_device=('N1e' if triggered else 'N1'),
        chosen_primary_scope=('eval-window solve' if triggered else 'reference-window solve'),
        beta_half_n_out=int(mass['0.5']['n_out']),
        note='this ratio is pure frozen `h`, `av0` and `Xi_raw` -- no forward, no routing',
    )


# ==========================================================================
# C1 -- the pure level control
# ==========================================================================
def c1_pure_level(model, elig, obs_m, A):
    """If a single constant multiplier on the baseline already repairs the monthly NSE,
    then the collapse is a LEVEL effect and the device has a target.  If it does not, the
    round's premise is false and the script stops.

    The plan writes "sweep delta".  A sweep can only report the best point ON its grid, so
    the exact optimum is also derived in closed form (`common23.best_constant_scale`) and
    the sweep is kept as its CROSS-CHECK -- the two must agree to 1e-12, and they are both
    reported.  The decision is taken from the exact value.
    """
    ly0, aux = LY.forward_layers(model, C.TAG)
    m, z = C.monthly_join(ly0, elig, obs_m, 1.0)
    base = C.monthly_stats_from_z(m, z)

    # the reproduced baseline must BE the registered baseline
    checks = {
        'nse': (base['nse'], float(A['nse'])),
        'median_station_nse': (base['median_station_nse'],
                               float(A['median_station_nse'])),
        'mean_concentration': (base['mean_concentration'],
                               float(A['mean_concentration'])),
        'n_station_months': (base['n_station_months'], 580),
        'n_eligible_rows': (base['n_eligible_rows'], 12152),
    }
    bad = {k: v for k, v in checks.items() if not (abs(v[0] - v[1]) <= 1e-12)}
    if bad:
        raise SystemExit('C1_BASELINE_DOES_NOT_REPRODUCE_THE_REGISTERED_ANCHORS %r' % bad)

    opt = C.best_constant_scale(z)
    o = z.obs.to_numpy(float)
    p0 = z.pred.to_numpy(float)
    grid = [dict(delta=float(d), nse=C.nse_of(o, p0 * (1.0 + float(d))))
            for d in C1_COARSE]
    bA = max(grid, key=lambda r: r['nse'])
    fine = np.linspace(bA['delta'] - C1_FINE_HALF_WIDTH, bA['delta'] + C1_FINE_HALF_WIDTH,
                       C1_FINE_N)
    gridB = [dict(delta=float(d), nse=C.nse_of(o, p0 * (1.0 + float(d)))) for d in fine]
    bB = max(gridB, key=lambda r: r['nse'])
    # ONE-SIDED.  The closed form is the exact minimiser of the same sum of squares, so a
    # sweep can only fail to REACH it, never beat it.  A grid value ABOVE the closed form
    # would mean the two are computing different objectives, which is a defect.
    grid_gap = float(bB['nse'] - opt['nse_at_star'])
    if grid_gap > 1e-12:
        raise SystemExit('C1_GRID_BEATS_THE_CLOSED_FORM %.6e' % grid_gap)
    # the closed form must sit at a local minimum, checked without a grid
    e_ = C1_BRACKET_EPS
    lo = C.nse_of(o, p0 * (1.0 + opt['delta_star'] - e_))
    hi = C.nse_of(o, p0 * (1.0 + opt['delta_star'] + e_))
    if not (opt['nse_at_star'] >= lo and opt['nse_at_star'] >= hi):
        raise SystemExit('C1_CLOSED_FORM_IS_NOT_A_LOCAL_MINIMUM %r' % (lo, hi))

    # the two spellings of "pure level control" must be the same number
    daily = C.monthly_stats(ly0, elig, obs_m, 1.0 + opt['delta_star'])
    spell_gap = abs(daily['nse'] - opt['nse_at_star'])

    nse_star = float(opt['nse_at_star'])
    if nse_star >= C1_HIGH:
        decision, action = 'PROCEED', (
            'the monthly collapse is substantially a pure LEVEL effect; the device has a '
            'target in principle.  NSE(delta*) is registered as this round\'s theoretical '
            'recovery ceiling.')
    elif nse_star < C1_LOW:
        decision, action = 'PREMISE_FALSIFIED_LEVEL_IS_NOT_THE_CAUSE', (
            'a constant multiplier CANNOT repair the collapse, so the device cannot '
            'either -- its whole effect is a per-reach level change.  STOP: do not enter '
            'Phase 0.  Per the user rule the action line turns to additional event N '
            'sources; THIS ROUND CHANGES NO SOURCE.')
    else:
        decision, action = 'PROCEED_WITH_LEVEL_CAVEAT', (
            'level explains only part of the collapse; proceed, but the first screen of '
            'the report must say so and the section 2.7 decomposition is MANDATORY.')
    return dict(
        baseline=base,
        registered_anchor_check={k: dict(reproduced=v[0], registered=v[1],
                                        abs_diff=abs(v[0] - v[1])) for k, v in checks.items()},
        optimum=opt, grid_coarse=grid, grid_fine=gridB,
        best_on_coarse_grid=bA, best_on_fine_grid=bB,
        grid_minus_closed_form=grid_gap,
        closed_form_local_minimum=dict(at_minus_eps=float(lo), at_star=opt['nse_at_star'],
                                       at_plus_eps=float(hi), eps=e_),
        delta_star_inside_the_coarse_grid=bool(C1_COARSE[0] <= opt['delta_star']
                                               <= C1_COARSE[-1]),
        station_day_spelling_minus_station_month=float(spell_gap),
        thresholds=dict(proceed_at_or_above=C1_HIGH, falsify_below=C1_LOW),
        nse_at_delta_zero=float(C.nse_of(o, p0)),
        # THE GATE CANNOT FALSIFY.  The sequence C1 scales is the beta = 0 baseline, whose
        # own NSE is already above `proceed_at_or_above`, and delta = 0 sits inside the
        # sweep -- so `max_delta NSE(delta) >= NSE(0) >= C1_HIGH` holds BY CONSTRUCTION and
        # `PREMISE_FALSIFIED_LEVEL_IS_NOT_THE_CAUSE` is unreachable.  Shipped as a boolean
        # rather than left for a reader to notice; the informative reading is the OPTIMUM
        # and the IMPROVEMENT, not the decision.  See the module deviations.
        cannot_falsify_because_base_above_high=bool(float(C.nse_of(o, p0)) >= C1_HIGH),
        improvement_over_delta_zero=float(opt['nse_at_star'] - C.nse_of(o, p0)),
        decision=decision, action_line=action,
        deviation='the plan says C1 uses the STORED beta = 0 station-month pL3 series.  '
                  'Round 4 stored no daily pL3 series, so ONE baseline forward is run here '
                  '(kernel NOT installed) and asserted against the registered nse / '
                  'median_station_nse / mean_concentration at 1e-12.  It is the same '
                  'forward phase1_full computes as its null point.',
        forward_note='this is the ONE forward charged to Phase -1', decided_by='closed form',
        _ly0=ly0, _aux=aux,
    )


# ==========================================================================
# C7 -- the frozen linear routing adjoint
# ==========================================================================
def c7_adjoint(model, B, elig, aux):
    """One autograd pass of the FROZEN router gives `d(mean pL3)/d local[t, r]` for every
    `(t, r)`, valid for EVERY beta and device, because `routing.py:130` is linear in
    `local` with coefficients that do not contain `Xi`.  That is what lets a land-phase
    change be turned into a predicted level change with no forward at all.

    The mask is the ELIGIBLE station-days of 2021-2024 -- the same rows `G5b` scores -- so
    `dCbar_pred` predicts the criterion's own quantity rather than some other mean.
    """
    meta = pd.read_parquet(C.PEER / 'data/prediction_calendar.parquet')
    meta = meta[meta.year.le(C.END_YEAR)].copy()
    c, record, _w = model.daily_metadata(meta)
    full = pd.DataFrame(dict(
        station_key=pd.Series(meta.station_key.to_numpy()[record.numpy()]).astype(str),
        date=pd.Series(model.data.dates[c['ti'].numpy()]).astype('datetime64[ns]')))
    left = full.assign(_one=1)
    right = (elig.assign(_one=1)[['station_key', 'date', '_one']]
             .assign(station_key=lambda d: d.station_key.astype(str),
                     date=lambda d: d.date.astype('datetime64[ns]')))
    mk = left.merge(right, on=['station_key', 'date', '_one'], how='inner')
    # EVERY eligible row must have matched exactly once; if the left frame carried a
    # duplicate (station, date) the mask would still be built correctly but the count
    # below would be the only sign of it, so it is checked rather than assumed.
    if len(mk) != len(elig):
        raise SystemExit('C7_MASK_MATCHES %d OF %d ELIGIBLE ROWS' % (len(mk), len(elig)))
    mask = np.zeros(len(full), bool)
    mask[mk.index.to_numpy()] = True

    adj = C.routing_adjoint(model, B, mask=mask)
    share = adj['share']

    # the adjoint's `local` is `fast + slow` with NO human term; this checks that claim
    # against the forward rather than restating it
    hum = float(np.max(np.abs(aux['local'] - (B['fast'] + B['slow']))))
    return dict(
        share_shape=list(share.shape), n_rows_selected=adj['n_rows_selected'],
        n_rows=int(adj['n_rows']), mask_rows=int(mask.sum()),
        share_abs_max=float(np.abs(share).max()),
        share_l1_max_per_day=float(np.abs(share).sum(axis=1).max()),
        share_per_reach_sum_top10=[float(v) for v in
                                   np.sort(np.abs(share).sum(axis=0))[::-1][:10]],
        operator_id=adj['operator_id'],
        human_term_max_abs_diff=hum,
        human_term_note='routing_adjoint builds `local = fast + slow`; layers23 builds '
                        '`local = fast + slow + model.human_mass(t)`.  The number above is '
                        'the maximum absolute difference over all cells, so the claim '
                        'that the human term is absent is CHECKED, not asserted.',
        objective=adj['objective'],
        note='valid for every beta and every device because the coefficients are Xi-free',
    )


# ==========================================================================
# main
# ==========================================================================
def _rank(x):
    """Average ranks with ties shared -- `scipy.stats.rankdata` without the scipy import.

    Written out rather than imported because this round adds no dependency, and a
    one-line `argsort` is auditable in a way a library call is not.
    """
    order = np.argsort(x, kind='mergesort')
    r = np.empty(x.size, float)
    r[order] = np.arange(x.size, dtype=float)
    sx = x[order]
    i = 0
    while i < sx.size:
        j = i + 1
        while j < sx.size and sx[j] == sx[i]:
            j += 1
        if j - i > 1:
            r[order[i:j]] = 0.5 * (i + j - 1)      # mean of the ranks i .. j-1
        i = j
    return r


def _corr(a, b, rank):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if a.size != b.size or a.size < 3:
        return None
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return None
    if rank:
        a, b = _rank(a), _rank(b)
    if float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def k_field_spatial_structure(B, G, kf, model):
    """Is `k_r` a disguised spatial source?  Plan section 3.3 and risk 9.

    WHY THIS LIVES IN THE FREEZE AND NOT IN `level_variance.py`.  The plan freezes
    the k field BEFORE any observation is read, exactly so that "did the
    neutralisation itself introduce spatial structure" is answered by a quantity
    frozen alongside the field it describes.  A diagnostic computed downstream,
    after the forward, would be indistinguishable from one chosen after seeing
    results.  So it is computed in `main()` and stored inside the frozen block.
    (`--shard` never calls `main()`, so this cannot perturb the bitwise-verified
    per-shard records.)

    WHY BOTH PEARSON AND SPEARMAN.  At the extreme betas `k_r` spans ten orders of
    magnitude -- at (N2, beta=+4) the quantiles run 5.7e-15 .. 0.0615.  Pearson on
    a vector like that is dominated by its few largest entries and answers "do the
    biggest k sit where the biggest Qs/W are", not "do the reaches rank alike".
    Spearman answers the second.  Both ship; a disagreement between them is itself
    a reading.

    WHY THERE IS NO PASS/FAIL HERE.  The plan asks for the coefficients and gives
    NO numeric threshold for "高度相关".  Inventing one would be the unregistered
    retuning this round's red lines forbid, so the coefficients are reported and
    the qualitative sentence is written in the report against their actual values.
    Correlations at beta=0 are `None` BY CONSTRUCTION -- `k == 1` there is
    constant, so the correlation is undefined; `k_std = 0.0` is reported beside it
    so `None` cannot be misread as a failure.

    The covariate `Qs/W` is WINDOW-SUMMED per reach using the same window the
    device solved on, because `k_r` is device- and window-specific; `W = Qf + Qs`,
    so `Qs/W` is the slow-response fraction.
    """
    area = np.asarray(model.data.area_ha, float)
    clim = np.asarray(G['clim'], float)
    out = {}
    for device in C.DEVICES:
        _Q, Wt = C.device_state(B, G, device)
        wb = np.asarray(Wt, bool)[:, None]
        Qs_r = np.where(wb, G['Qs'], 0.0).sum(axis=0)
        W_r = np.where(wb, G['W'], 0.0).sum(axis=0)
        ratio = Qs_r / W_r
        rows = {}
        for beta in C.BETA_GRID:
            k = np.asarray(kf[device]['points']['%g' % beta]['k'], float)
            rows['%g' % beta] = dict(
                n_reaches=int(k.size),
                k_std=float(np.std(k)), k_min=float(k.min()), k_max=float(k.max()),
                pearson=dict(qs_over_w=_corr(k, ratio, False),
                             clim=_corr(k, clim, False),
                             area_ha=_corr(k, area, False)),
                spearman=dict(qs_over_w=_corr(k, ratio, True),
                              clim=_corr(k, clim, True),
                              area_ha=_corr(k, area, True)),
            )
        out[device] = dict(rows=rows,
                           covariate_note='Qs/W is the window-summed slow-response '
                                          'fraction per reach; clim_r is the per-reach '
                                          'mean log-mobile-water over the ACTIVE '
                                          'reference cells; area_ha is the per-reach area')
    return out


def main(n_shards=None):
    t0 = time.time()
    rep = dict(phase=-1, n_fits=0, fit_worker_calls=0, round=str(R),
               n_stations=15, n_reaches=230, beta_grid=[float(b) for b in C.BETA_GRID],
               execution_order=['C6', 'C5', 'k_field(C3,C4)', 'C2', 'FREEZE+HASH',
                                'C1', 'C7'])

    _p('=== round-4 JSONs are read-only ===')
    rep['round4_json_intact'] = C.assert_round4_jsons_intact()
    _p('    ' + ', '.join('%s=%s' % (k[:16], v[:12])
                          for k, v in rep['round4_json_intact']['shas'].items()))

    _p('=== frozen anchors, from their producers ===')
    rep['anchors'] = C.load_anchors()
    A = {k: v['value'] for k, v in rep['anchors'].items()}
    rep['restatement_cross_check'] = C.restatement_cross_check()
    _p('    n anchors=%d  restatement disagreements=%d'
       % (len(A), rep['restatement_cross_check']['n_disagree']))

    _p('=== model, geometry, frozen arrays ===')
    model = C.build()
    G = C.geometry(model)
    B = C.bootstrap_arrays(model)
    elig = C.eligible_grid()
    obs_m = None            # NOT read yet: the k field is frozen first
    rep['model'] = dict(operator_id=model.data.operator_id, calendar=model.calendar,
                        cap=bool(model.cap), type=type(model).__name__,
                        pilot_indices=C.pilot_indices(model),
                        shape=list(B['h'].shape),
                        n_days=int(B['h'].shape[0]), n_reaches=int(B['h'].shape[1]))
    rep['eligible'] = dict(n_rows=int(len(elig)),
                           n_stations=int(elig.station_key.nunique()),
                           mask_sha=C.sha(E.MASK), mask_sha_matches=bool(
                               C.sha(E.MASK) == E.MASK_SHA))
    _p('    %s  %s  shape=%s  pilot=%s'
       % (rep['model']['operator_id'], rep['model']['calendar'], rep['model']['shape'],
          rep['model']['pilot_indices']))

    _p('=== C6: boundary_code census ===')
    rep['C6_boundary_code_census'] = c6_boundary_census(model)
    _p('    all_ordinary_internal=%s  codes=%s'
       % (rep['C6_boundary_code_census']['all_ordinary_internal'],
          rep['C6_boundary_code_census']['boundary_code_values']))

    _p('=== C5: reach census, degeneracy tags, pin footprint ===')
    rep['C5_reach_census'] = c5_reach_census(B, G)
    fp = rep['C5_reach_census']['footprint']
    _p('    active=%d inactive=%d  inactive&h!=0=%d  max_h_on_pin=%.16g (round4=%.16g ok=%s)'
       % (fp['n_active_cells'], fp['n_inactive_cells'], fp['n_inactive_and_h_nonzero'],
          fp['max_h_on_pinned_cells'], fp['max_h_registered_in_round4'],
          fp['max_h_matches_round4']))

    _p('=== k field: 4 devices x 19 betas (zero forward) ===')
    if n_shards is None:
        kf = {}
        for device in C.DEVICES:
            Q, Wt = C.device_state(B, G, device)
            kf[device] = device_block(device, Wt)
            for beta in C.BETA_GRID:
                keep = solve_point(B, G, device, beta)
                kf[device]['points']['%g' % beta] = keep
                _fp = keep['fixed_point']
                _p('    %-4s beta=%-6g %7.2fs  rho_lin=%.3e  fp=%d iters=%d '
                   'esc=%d un=%d  adoptable=%s (registered-route-only=%s)  nonunit=%.3f'
                   % (device, beta, keep['seconds'], keep['max_rho_lin'],
                      keep['n_fixed_point_solved'], _fp.get('n_orbit_iter', 0),
                      _fp.get('n_resolved_after_escalation', 0), _fp.get('n_unresolved', 0),
                      keep['adoptable'], keep['adoptable_registered_route_only'],
                      keep['frac_k_nonunit']))
        rep['sharded_execution'] = dict(
            used=False, note='one process, the registered point order, the registered '
                             'path; `--shard` produces the same 76 records and `--verify` '
                             'checks them against this code path bitwise')
    else:
        kf, rep['sharded_execution'] = load_shard_field(B, G, int(n_shards))
        rep['execution_order'].insert(2, 'k_field(sharded, %d workers)' % int(n_shards))

    # ---- beta = 0 must be bitwise the identity, on EVERY device ----
    one230 = np.ones(230)
    for device in C.DEVICES:
        k0 = np.asarray(kf[device]['points']['0']['k'], float)
        assert np.array_equal(k0, one230), ('K_NOT_UNIT_AT_BETA_ZERO', device)
        xs = XI.xi_star(G, 0.0, k0)
        assert np.array_equal(xs, np.ones_like(xs)), ('XI_STAR_NOT_UNIT', device)
    rep['beta_zero_is_bitwise_identity'] = dict(
        k=np.array_equal(np.asarray(kf['N1']['points']['0']['k'], float), one230),
        xi_star_unit=True, checked_on=list(C.DEVICES),
        note='asserted with array_equal on every device, not allclose')
    _p('=== beta = 0 is bitwise the identity on all four devices ===')

    _p('=== C2: window transfer ratio Gamma_r ===')
    rep['C2_window_transfer'] = c2_window_transfer(B, G)
    c2 = rep['C2_window_transfer']
    _p('    worst out-of-band = %d of %d   triggered=%s   primary device -> %s'
       % (c2['worst_n_out_across_the_grid'], c2['n_stations'], c2['triggered'],
          c2['chosen_primary_device']))

    _p('=== C3: headroom (device-invariant in beta) ===')
    hd = {}
    for device in C.DEVICES:
        pts = kf[device]['points']
        first = pts['0']
        same = all(pts[b]['headroom'] == first['headroom'] for b in pts)
        hd[device] = dict(
            headroom_identical_across_beta=bool(same),
            n_no_finite_fixed_point=first['n_no_finite_fixed_point'],
            n_reaches_headroom_positive=int(sum(1 for v in first['headroom'] if v > 0)),
            n_reaches_headroom_zero=int(sum(1 for v in first['headroom'] if v == 0)),
            n_reaches_headroom_negative=int(sum(1 for v in first['headroom'] if v < 0)),
            headroom_min=float(np.min(first['headroom'])),
            headroom_max=float(np.max(first['headroom'])),
            headroom_quantiles={str(q): float(v) for q, v in
                                zip((0, 1, 50, 99, 100),
                                    np.percentile(first['headroom'], (0, 1, 50, 99, 100)))},
            closed_form_agrees_with_telescoped=first['headroom_closed_form_agrees'],
            naive_plan_F_inf_excess=first['headroom_naive_excess'],
            note='the numeric headroom comes from one `scan` at h_eff = 700*1[h*Xi>0]: at '
                 'that hazard the kernel drives prob = -expm1(-700) == 1.0 exactly, the '
                 'state drains to max(n_t, 0), and the mobilisation saturates everywhere, '
                 'so it is the true supremum INCLUDING the state feedback.  The plan '
                 'writes headroom = F(inf) - T with F(inf) = sum over h>0 of av0; that '
                 'reading is reported too (naive_plan_F_inf_excess) and it is LARGER, '
                 'because it assumes no mass is carried across an h = 0 day.',
        )
    rep['C3_headroom'] = hd
    for device in C.DEVICES:
        _p('    %-4s no_finite_fp=%d  headroom min=%.6g max=%.6g  closed==telescoped=%s'
           % (device, hd[device]['n_no_finite_fixed_point'], hd[device]['headroom_min'],
              hd[device]['headroom_max'],
              hd[device]['closed_form_agrees_with_telescoped']))

    _p('=== C4: closed form k_lin and its exact residual ===')
    _sr, _scored = C.station_reaches()
    _scored_s = [str(int(r)) for r in _scored]
    c4 = {}
    for device in C.DEVICES:
        pts = kf[device]['points']
        c4[device] = dict(
            max_rho_lin_by_beta={b: pts[b]['max_rho_lin'] for b in pts},
            max_rho_exact_by_beta={b: pts[b]['max_rho_exact'] for b in pts},
            n_within_tol_lin_by_beta={b: int(sum(1 for v in pts[b]['rho_lin']
                                                 if v <= TOL_LIN)) for b in pts},
            k_sign_at_half_on_the_scored_reaches={
                r: pts['0.5']['k_sign'][r] for r in _scored_s},
            n_k_below_one_at_half=pts['0.5']['n_k_below_one'],
            n_k_above_one_at_half=pts['0.5']['n_k_above_one'],
            n_k_exactly_one_at_half=pts['0.5']['n_k_exactly_one'],
        )
    rep['C4_closed_form'] = dict(per_device=c4, tol_lin=TOL_LIN, max_iter=MAX_ITER,
                                 registered_rule='rho_lin <= 1e-4 => the closed form IS the '
                                                 'product and the fixed point only confirms')
    for device in C.DEVICES:
        vals = list(c4[device]['max_rho_lin_by_beta'].values())
        _p('    %-4s max rho_lin over grid = %.6g   max rho_exact = %.6g'
           % (device, max(vals),
              max(c4[device]['max_rho_exact_by_beta'].values())))

    _p('=== section 3.3 / risk 9: k vs the frozen spatial covariates ===')
    rep['k_field_spatial_structure'] = k_field_spatial_structure(B, G, kf, model)
    for device in C.DEVICES:
        _rows = rep['k_field_spatial_structure'][device]['rows']
        _bmax = max(_rows, key=lambda s: abs(float(s)))
        _r = _rows[_bmax]
        _fmt = lambda which: ','.join(                      # noqa: E731
            ('None' if _r[which][kk] is None else '%.4f' % _r[which][kk])
            for kk in ('qs_over_w', 'clim', 'area_ha'))
        _p('    %-4s beta=%-6s k_std=%.4g  pearson(Qs/W,clim,area)=(%s)  spearman=(%s)'
           % (device, _bmax, _r['k_std'], _fmt('pearson'), _fmt('spearman')))

    # ---- FREEZE: hash the whole k field before any observation is read ----
    import hashlib
    hsh = hashlib.sha256()
    for device in C.DEVICES:
        for beta in C.BETA_GRID:
            hsh.update(('%s|%g|' % (device, beta)).encode())
            hsh.update(np.ascontiguousarray(
                kf[device]['points']['%g' % beta]['k'], float).tobytes())
    per_point = {d: {b: kf[d]['points'][b]['k_sha256'] for b in kf[d]['points']}
                 for d in C.DEVICES}
    frozen = dict(
        k_field_sha256=hsh.hexdigest(),
        per_device_per_beta_sha256=per_point,
        canonical_order='device-major then beta in the registered grid order; the sha '
                        'covers the ASCII "<device>|<beta>|" tag followed by the 230 '
                        'float64 k values in reach order',
        n_points=len(C.DEVICES) * len(C.BETA_GRID),
        frozen_before_reading_any_observation=True,
    )
    rep['k_field_frozen'] = frozen
    _p('=== K FIELD FROZEN: sha256=%s ===' % frozen['k_field_sha256'])

    _p('=== do the FOUR devices really solve four k fields? ===')
    rep['device_degeneracy'] = device_degeneracy(kf)
    for _pair, _d in rep['device_degeneracy']['pairs'].items():
        _p('    %-9s %-38s max|dk|=%.6g over the grid  sha-equal at %d/%d betas'
           % (_pair, _d['what'], _d['max_abs_diff_over_grid'],
              _d['n_sha_equal'], _d['n_beta']))

    _p('=== k_field.json ===')
    kf_out = dict(round=str(R), phase='k_field', n_fits=0, fit_worker_calls=0,
                  devices=list(C.DEVICES), beta_grid=[float(b) for b in C.BETA_GRID],
                  frozen=frozen, device_spec={d: C.DEVICE_SPEC[d] for d in C.DEVICES},
                  registered_point_order=[['%s' % d, float(b)] for d, b in point_order()],
                  sharded_execution=rep['sharded_execution'],
                  k_field=kf)
    sh = C.write_json(OUT / 'k_field.json', kf_out)
    rep['k_field_json'] = dict(path=str(OUT / 'k_field.json'), sha256=sh,
                               k_field_sha256=frozen['k_field_sha256'])
    _p('    written, sha=%s' % sh[:16])

    _p('=== C1: pure level control (READS OBSERVATIONS; runs after the freeze) ===')
    obs_m = C.obs_monthly()
    c1 = c1_pure_level(model, elig, obs_m, A)
    ly0 = c1.pop('_ly0')
    aux = c1.pop('_aux')
    rep['C1_pure_level'] = c1
    _p('    delta*=%.6g  NSE(delta*)=%.6f  NSE(0)=%.6f  grid gap=%.3e  spell gap=%.3e'
       % (c1['optimum']['delta_star'], c1['optimum']['nse_at_star'],
          c1['nse_at_delta_zero'], c1['grid_minus_closed_form'],
          c1['station_day_spelling_minus_station_month']))
    _p('    DECISION: %s' % c1['decision'])

    _p('=== C7: frozen linear routing adjoint ===')
    rep['C7_adjoint'] = c7_adjoint(model, B, elig, aux)
    _p('    share %s  rows=%d  human_max_abs_diff=%.3e'
       % (rep['C7_adjoint']['share_shape'], rep['C7_adjoint']['n_rows_selected'],
          rep['C7_adjoint']['human_term_max_abs_diff']))

    # ---- the stop rule, adjudicated from the pre-registered thresholds ----
    stop = dict(
        C1_decision=c1['decision'],
        C2_triggered=bool(c2['triggered']),
        enter_phase0=bool(c1['decision'] != 'PREMISE_FALSIFIED_LEVEL_IS_NOT_THE_CAUSE'),
        outcome=('PREMISE_FALSIFIED_LEVEL_IS_NOT_THE_CAUSE'
                 if c1['decision'] == 'PREMISE_FALSIFIED_LEVEL_IS_NOT_THE_CAUSE'
                 else 'ENTER_PHASE_0'),
        rule='C1 first: NSE(delta*) < 0.55 stops the round before Phase 0 and the action '
             'line turns to additional event N sources (no source is changed this round). '
             'C2 does NOT stop the round; it changes WHICH WINDOW the primary device solves '
             'on, and that change is recorded above, before any forward is run.',
        n_fits=0, fit_worker_calls=0,
    )
    rep['stop_rule'] = stop
    rep['elapsed_seconds'] = round(time.time() - t0, 2)
    rep['deviations'] = [
        'C1 needs ONE baseline forward: round 4 stored no daily pL3 series.  It is the '
        'same forward phase1_full computes as its null point.',
        'the execution order is C6,C5,k-field,C2,freeze,C1,C7 -- not the plan listing '
        'order -- because C1 reads observations and the k field must be frozen first.',
        'anchors are read from their ORIGINAL producers (20260919_3 / 20260919_2) with a '
        'round-4 restatement cross-check, not from round 4 alone.',
        'C1\'s optimum is computed in CLOSED FORM; the plan\'s delta sweep is kept only as '
        'a cross-check because a sweep cannot report a point off its grid.',
        'C2 is evaluated with the aggregation fixed before any forward: a station is OUT '
        'if Gamma_r leaves the band at ANY beta.',
        'the ill-conditioning floor is C.DEGEN_FLOOR (1e-3), reused rather than invented.',
        'section 3.3 / risk 9 (k vs the frozen spatial covariates) is computed HERE, inside '
        'the freeze, rather than in level_variance.py as the plan\'s file table implies: the '
        'plan freezes the k field BEFORE any observation is read, so "did the neutralisation '
        'introduce spatial structure" must be frozen with the field it describes.  The plan '
        'registers NO numeric threshold for "highly correlated", so no pass/fail is emitted '
        'here and none is claimed in the report -- the coefficients are reported as read.  '
        'Both Pearson and Spearman ship because at the extreme betas k spans ten orders of '
        'magnitude and Pearson alone would be an outlier statistic.',
        'A BROADCAST DEFECT IN `_lift` (C2) WAS FOUND BY RUNNING IT, NOT BY READING IT.  It '
        'was written as `B["a0"][:, None] * Q[None, :]`, i.e. as if the per-CELL `av0` '
        'were the per-DAY array `W` is; that broadcasts against the per-REACH `Q` into '
        '`(nd, nr, nr)` and asks for 936 GiB.  It RAISES rather than corrupting, which is '
        'the only reason it was found.  This is the SECOND defect of the same class in the '
        'round: `phase0_freeze.hxi_precondition` had the SILENT per-day/per-cell form and '
        'was fixed earlier by executing that block.  `_lift` now asserts its own argument '
        'shapes (plan section 3.2 item 2).  The primary mass reading passes Q = ones, so '
        'there its weight is exactly `av0`.',
        'C1 AS PRE-REGISTERED CANNOT FALSIFY ITS OWN PREMISE, SO THE SHIPPED DECISION MUST '
        'NOT BE READ AS THE PREMISE HOLDING.  The plan fixes the threshold at '
        '`NSE(delta*) >= 0.65` for "the collapse is substantially a level effect", but the '
        'sequence C1 is registered to scale is the BETA = 0 BASELINE -- whose own NSE is '
        '`0.7029748157444711`, ALREADY ABOVE 0.65 -- and delta = 0 is inside the sweep, so '
        '`max_delta NSE(delta) >= NSE(0) >= 0.65` holds BY CONSTRUCTION.  `cannot_falsify_'
        'because_base_above_high = true` ships in the JSON beside the decision rather than '
        'being left for a reader to notice.  The informative readings are therefore the '
        'OPTIMUM and the IMPROVEMENT, not the pass: `delta* = -0.023699266770822125`, '
        '`NSE(delta*) = 0.7049056581993793`, an improvement of `+0.001931` over delta = 0.  '
        'Pure level control buys about 0.2 per cent of NSE on the baseline sequence, and '
        'plan risk 2 ("C1 may kill the round") CANNOT MATERIALISE.  The threshold is NOT '
        're-tuned and the decision is NOT re-derived here; whether 0.65 was chosen with the '
        'candidate\'s collapsed NSE in mind is not recoverable from the plan text, and this '
        'round reports the reading instead of re-interpreting the registration.',
        'THE FOUR DEVICES ARE TWO, AND THAT IS A MEASURED FACT RATHER THAN A DESIGN CHOICE.  '
        'A PER-REACH CONSTANT normaliser cancels from the ledger identity '
        '`sum_t av0 Q_r [1 - e^{-h k Xi}] = sum_t av0 Q_r [1 - e^{-h}]`: `Q_r` is constant '
        'over t, so it divides out of both sides.  The concentration devices use '
        '`Q_r = 1/W_r` and the mass devices use `Q_r = 1`, so each pair solves the SAME '
        '`k_r`.  Measured on the shipped field: `max|k(N1) - k(N3)|` = 6.02141e-12 at '
        'beta = -8 and 1.01807e-13 at beta = +0.5; `max|k(N1e) - k(N2)|` = 2.95763e-13 and '
        '1.56986e-13; both correlations are 1.00000000.  The per-device `k_sha256` differ '
        'ONLY in the last bits, because the two arithmetic paths round differently -- a '
        'different sha is NOT evidence of a different solve.  CONSEQUENCES, reported rather '
        'than smoothed over: (i) the plan\'s "N3 is the device that can really pass G5b" is '
        'NOT a different device from N1 -- same window, same `k` -- so N1 and N3 are ONE '
        'device with two reporting targets; (ii) the outcome '
        '`INVARIANT_IS_CONCENTRATION_NOT_MASS` cannot fire from a TARGET mismatch, because '
        'N1 and N3 produce the same forward; it can now fire only from the WINDOW contrast '
        '(C2 having moved the primary to N1e), which is a DIFFERENT claim from the one the '
        'cell is named for -- if it fires, the report must say so in those words; (iii) the '
        'action line "the N1->N3 difference quantifies the target mismatch" quantifies '
        'round-off; (iv) section 2.3\'s justification for `W` -- that it keeps `k` '
        'per-reach -- is true but VACUOUS with respect to `k`: the red line "do not pin the '
        'level target on the statistic G5b measures" is satisfied trivially, because a '
        'per-reach constant target pins nothing at all.  NONE of this is repaired here: '
        'making `Q` time-varying would pin `k` onto the criterion\'s own statistic, which '
        'the plan forbids outright.',
    ]

    sha = C.write_json(OUT / 'phase_minus1.json', rep)
    _p('=== wrote reports/phase_minus1.json sha=%s in %.1fs ===' % (sha, rep['elapsed_seconds']))
    _p('C1=%s  C2_triggered=%s  primary=%s' % (c1['decision'], c2['triggered'],
                                              c2['chosen_primary_device']))
    return 0


def device_degeneracy(kf):
    """Whether the FOUR devices really are four.  Shipped, not inferred.

    A PER-REACH CONSTANT normaliser cancels from the ledger identity
    `sum_t av0 Q_r [1 - e^{-h k Xi}] = sum_t av0 Q_r [1 - e^{-h}]`: `Q_r` is constant over
    t, so it divides out of BOTH sides.  The concentration devices use `Q_r = 1/W_r` and
    the mass devices use `Q_r = 1`, so each window pair solves the same `k_r` -- to within
    the difference between two arithmetic paths, which is round-off, not physics.

    THIS IS WHY THE NUMBERS SHIP HERE.  The two `k_sha256` differ (they are hashes of raw
    bytes and the last bits differ), so a reader comparing shas would conclude the four
    devices are four.  The elementwise comparison is what settles it, and a claim that
    only exists in prose cannot be checked by anyone.
    """
    pairs = (('N1', 'N3', 'ref window: mass vs concentration target'),
             ('N1e', 'N2', 'eval window: mass vs concentration target'),
             ('N1', 'N1e', 'same target, ref vs eval window'),
             ('N1', 'N2', 'ref mass vs eval concentration'))
    out = {}
    for a, b, what in pairs:
        rows = {}
        for beta in C.BETA_GRID:
            ka = np.asarray(kf[a]['points']['%g' % beta]['k'], float)
            kb = np.asarray(kf[b]['points']['%g' % beta]['k'], float)
            d = np.abs(ka - kb)
            # beta = 0 has k == ones on both sides, so the correlation is 0/0 there and is
            # reported as None rather than as a NaN that a JSON reader would trip on.
            cc = None
            if float(ka.std()) > 0.0 and float(kb.std()) > 0.0:
                cc = float(np.corrcoef(ka, kb)[0, 1])
            rows['%g' % beta] = dict(
                max_abs_diff=float(d.max()), median_abs_diff=float(np.median(d)),
                correlation=cc,
                sha_equal=bool(kf[a]['points']['%g' % beta]['k_sha256']
                               == kf[b]['points']['%g' % beta]['k_sha256']),
            )
        worst = max(rows, key=lambda s: rows[s]['max_abs_diff'])
        out['%s_vs_%s' % (a, b)] = dict(
            what=what, worst_beta=worst,
            max_abs_diff_over_grid=rows[worst]['max_abs_diff'],
            n_sha_equal=sum(1 for s in rows if rows[s]['sha_equal']),
            n_beta=int(len(rows)), rows=rows)
    return dict(
        pairs=out,
        note='A PER-REACH CONSTANT normaliser cancels from the ledger identity, so the '
             'concentration and mass devices in the SAME window solve the same `k`.  The '
             'per-device `k_sha256` still differ, in the last bits only -- a different sha '
             'is NOT evidence of a different solve.  Consequences are registered as module '
             'deviations and must be carried into the verdict narrative: the pre-registered '
             'outcome `INVARIANT_IS_CONCENTRATION_NOT_MASS` cannot fire from a TARGET '
             'mismatch, only from the WINDOW contrast.',
    )


def verify_sharding():
    """Re-derive a chosen subset of the k field in ONE process, FROM A FRESHLY BUILT MODEL,
    and compare it with the shipped `k_field.json` point by point.

    THIS IS THE ADDITIVITY EVIDENCE, and it is deliberately NOT "run it again in the same
    process".  The two sides differ in every way that could matter: different process,
    different memory layout, a model built rather than loaded from the cache, and -- in the
    shipped run -- a different partition of the point list.  If the per-point `k_sha256`
    agrees, the shard assignment cannot have entered the answer.
    """
    model = C.build()
    G = C.geometry(model)
    B = C.bootstrap_arrays(model)
    shipped = json.loads((OUT / 'k_field.json').read_text(encoding='utf-8'))
    rows, n_bad = [], 0
    for device, beta in VERIFY_POINTS:
        keep = solve_point(B, G, device, beta)
        ref = shipped['k_field'][device]['points']['%g' % beta]['k_sha256']
        ok = bool(keep['k_sha256'] == ref)
        n_bad += int(not ok)
        rows.append(dict(device=device, beta=float(beta), seconds=keep['seconds'],
                         k_sha256_reverified=keep['k_sha256'], k_sha256_shipped=ref,
                         bitwise_identical=ok,
                         frac_k_nonunit=keep['frac_k_nonunit'],
                         adoptable=keep['adoptable']))
        _p('  %-4s beta=%-6g %7.2fs  %s  %s' % (device, beta, keep['seconds'],
                                                keep['k_sha256'][:16], 'IDENTICAL' if ok
                                                else 'DISAGREES WITH THE SHIPPED FIELD'))
    order = point_order()
    shipped_order_ok = ([['%s' % d, float(b)] for d, b in order]
                        == shipped.get('registered_point_order',
                                       [['%s' % d, float(b)] for d, b in order]))
    rep = dict(round=str(R), phase='shard_additivity_check', n_fits=0, fit_worker_calls=0,
               what='points re-derived in ONE process from a FRESHLY BUILT MODEL and '
                    'compared bitwise with the sharded k_field.json',
               n_points_checked=len(VERIFY_POINTS), n_disagree=n_bad,
               all_bitwise_identical=bool(n_bad == 0),
               partition_size_checked=int(
                   shipped.get('sharded_execution', {}).get('n_shards', 0) or 0),
               registered_order_preserved=bool(shipped_order_ok),
               points=rows,
               limits=['a SUBSET is re-derived: re-deriving all 76 points in one process '
                       'is the serial run this sharding exists to avoid',
                       'agreement is on `k_sha256`, which covers the 230 float64 `k` '
                       'values only -- the surrounding per-point records are compared in '
                       '`load_shard_field` by construction (one shared `solve_point`), '
                       'not by this check'])
    sha = C.write_json(OUT / 'shard_additivity_check.json', rep)
    _p('=== shard additivity: %d/%d identical  sha=%s ==='
       % (len(VERIFY_POINTS) - n_bad, len(VERIFY_POINTS), sha[:16]))
    return 0 if n_bad == 0 else 1


if __name__ == '__main__':
    import sys
    argv = sys.argv[1:]
    if not argv:
        raise SystemExit(main())
    if argv[0] == '--prepare':
        m = prepare_inputs()
        _p('=== inputs cache written: %d arrays, manifest sha=%s ==='
           % (len(m), manifest_sha(m)[:16]))
        raise SystemExit(0)
    if argv[0] == '--shard':
        i, n = argv[1].split('/')
        raise SystemExit(run_shard(int(i), int(n)))
    if argv[0] == '--merge':
        raise SystemExit(main(n_shards=int(argv[1])))
    if argv[0] == '--verify':
        raise SystemExit(verify_sharding())
    raise SystemExit('USAGE: phase_minus1.py [--prepare | --shard I/N | '
                     '--merge N | --verify]')
