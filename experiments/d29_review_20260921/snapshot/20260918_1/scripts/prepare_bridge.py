"""Build the H1 arm's folds and designs from H0's, changing only what water changed.

The round needs two arms of the *same* grey box to run on identical folds, with
the hydrology product as the single difference.  `fit_worker.py` and
`campaign_model.py` cannot be edited (they are frozen launch-chain hashes), so
the arm is encoded purely in the registry files those modules already read:

    configs/folds.json[<fold>]['domain']     -> which data/domains/<domain> to load
    data/designs/<fold>.json                 -> the basis every job consumes
    data/folds/<fold>/{train.parquet,registry.json}

`campaign_model.for_job` reads exactly those three and nothing else, so a fold
named `T24_L_M_H1` pointing at domain `FULL24C` is a complete arm definition.
Suffixing the fold name works where a subdirectory would not: `native_runtime.RUN`
must stay exactly `5_Test/20260917_5`, because `prepare_global.py` reaches its
inputs through `RUN.parent` and `RUN.parents[1]`.

The label side is copied, never rebuilt: `train.parquet` and `registry.json` are
byte-identical to H0's and the copy is asserted against the original's sha256.
Both arms therefore see the same observations, the same weights, the same
variance floor and the same folds -- only the covariate domain differs.

--- what the design carries, and why all of it must be rebuilt -----------------
`build_design` mixes hydrology-derived and hydrology-free quantities:

    low, high, mean, sd        from `static_raw` and `fit_design`  -- unmoved
    extra_mean, extra_sd       from [q, q^2, T, W, d, dW, qW, TW]  -- MOVES (q, W)
    scientific                 centre/components from static; norms from basis
    hc2_gate                   from d.bfi over 1991-2020            -- MOVES
    endpoint_pi_mean           from d.bfi and hc2_gate              -- MOVES

`scientific` needs care in the other direction.  Its inputs -- `static_raw`,
`low/high/mean/sd` and the reach list -- are byte-identical across arms, so its
true value is arm-invariant and there is nothing for H1 to rebuild.  It is
*carried* from the frozen design rather than recomputed, because the SVD factor
is only defined up to a per-component sign: the rebuild reproduces the frozen
`components` to 5.7e-15 in absolute value with every principal cosine exactly
1.0 and rows 1 and 2 sign-flipped, so the sign is the only freedom and the frozen
sign is the one the mainline's 30 parameters were fitted against.  Carrying it
for both arms keeps H0 a faithful replay; the rebuild is still verified against
it, up to that sign.

`endpoint_pi_mean` is the easy one to miss and the worst to miss: it is the only
one with no producer in this round's own code.  `prepare_hf.py:71` copies it from
a predecessor design rather than deriving it, and its real formula lives upstream
in `20260915_4/scripts/register_temporal.py:18-35`:

    pi = where(bfi_water_positive, (1 + tanh((bfi - gate.mean)/gate.sd))/2, .5)
    endpoint_pi_mean = pi[pilot_reaches].mean()

Both `bfi` and the gate move with the water, so a frozen `endpoint_pi_mean` would
leave `Endpoints.prior` blending beta and beta_b on the *old* hydrology while
`Endpoints.flux_parameters` ran the *new* one -- a half-switched arm, which is
precisely the failure mode Gate 0-5 was about for `capacity`.  It is recomputed
here for both arms so that H0's value is shown to be reproduced rather than
assumed.

H0 is not rebuilt from scratch in `configs/folds.json`: its four folds already
exist and already point at FULL24.  What this script does for H0 is *verify* that
`build_design` reproduces `frozen_design.json` exactly, which is the §6.1
regression check, and then prove the H1 design differs only in the moving keys.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

RUN = Path(__file__).resolve().parents[1]
ROOT = RUN.parents[1]
sys.path.insert(0, str(RUN / 'scripts'))
import native_runtime as rt                                                          # noqa: E402
import numpy as np                                                                   # noqa: E402
import pandas as pd                                                                  # noqa: E402
from campaign_model import build_design, load_data                                   # noqa: E402

# Folds this round runs, and their mode.  Spatial holdouts (S56/S113/S191) and the
# 2025 folds are explicitly out of scope; FULL25 does not exist in this round.
SCOPES = ['T24_L', 'T24_G']
MODES = ['M', 'D']
ARM_SUFFIX = '_H1'

# The keys `build_design` derives from water, established by measurement rather
# than by reading: each was diffed across FULL24 and FULL24C and each moved.
# `dynamic_scales` is easy to mistake for a static normalisation; it is not.
MOVING = ['dynamic_scales', 'extra_mean', 'extra_sd', 'hc2_gate']
# Static-derived.  Asserted identical across arms rather than carried blindly: if
# one of these had drifted, the arm would differ by more than water.
UNMOVED = ['low', 'high', 'mean', 'sd']
# Arm-invariant in value, but not reproducible in sign: carried from the frozen
# design rather than rebuilt.  See the module docstring.
CARRIED = ['scientific']
# Written into the H1 fold designs as provenance.  No fit-path or validation code
# reads any of them; `temporal_model.py` reads the two registry fields, which are
# repointed at this arm's byte-identical copy of H0's registry.
PROVENANCE = ['arm', 'hydro_domain', 'driver_version',
              'observation_registry_file', 'observation_registry_hash']
# `build_design` reproduces the frozen design to this tolerance, not bit-exactly
# (the SVD and the means carry float round-off).  This is the same tolerance
# `prepare_global.py` used when it first asserted the reproduction.
TOL = 1e-12


def close(a, b, tol=TOL):
    """Tolerant structural comparison of design values (floats and nested lists)."""
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(close(a[k], b[k], tol) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(close(x, y, tol) for x, y in zip(a, b))
    try:
        return bool(np.allclose(np.asarray(a, float), np.asarray(b, float),
                                rtol=tol, atol=tol, equal_nan=True))
    except (TypeError, ValueError):
        return a == b


def compare_scientific(a, b):
    """Equal up to the per-component sign the SVD leaves free."""
    if set(a) != set(b):
        return False, dict(key_mismatch=sorted(set(a) ^ set(b)))
    detail, ok = {}, True
    for key in sorted(a):
        if key != 'components':
            same = close(a[key], b[key])
            detail[key] = 'same' if same else 'DIFFERS'
            ok &= same
            continue
        x, y = np.asarray(a[key], float), np.asarray(b[key], float)
        if x.shape != y.shape:
            detail[key] = dict(shape=[list(x.shape), list(y.shape)])
            ok = False
            continue
        # A per-component sign flip changes the projection by a whole component,
        # so the raw difference is O(1); |components| is what is comparable.  The
        # sign product is per row: a row yielding both +1 and -1 would mean the
        # two bases are rotated into each other, not merely flipped, and the
        # |components| agreement would then be a coincidence worth catching.
        products = np.sign(x) * np.sign(y)
        detail[key] = dict(
            equal_up_to_sign=bool(np.allclose(np.abs(x), np.abs(y), rtol=1e-8, atol=1e-8)),
            max_abs_abs=float(np.max(np.abs(np.abs(x) - np.abs(y)))),
            max_abs_raw=float(np.max(np.abs(x - y))),
            sign_products=[sorted({int(s) for s in row}) for row in products])
        ok &= detail[key]['equal_up_to_sign']
    return ok, detail


def compare_gate(a, b):
    """`training_global_reaches` is a set, not a sequence: its order is unstable."""
    if set(a) != set(b):
        return False, dict(key_mismatch=sorted(set(a) ^ set(b)))
    detail, ok = {}, True
    for key in sorted(a):
        if key != 'training_global_reaches':
            same = close(a[key], b[key])
            detail[key] = 'same' if same else 'DIFFERS'
            ok &= same
            continue
        same_set = set(a[key]) == set(b[key])
        detail[key] = dict(same_set=same_set, same_order=list(a[key]) == list(b[key]),
                           n=len(a[key]))
        ok &= same_set
    return ok, detail


def compare_design(key, a, b):
    """Compare one design key, honouring the degeneracy that key carries."""
    if key == 'scientific':
        return compare_scientific(a, b)
    if key == 'hc2_gate':
        return compare_gate(a, b)
    return close(a, b), {}


def arm_design(arm, ref, frozen):
    """Rebuild one arm's design and check it against the frozen baseline."""
    domain = 'FULL24' if arm == 'H0' else 'FULL24C'
    d = load_data(domain)
    rebuilt = build_design(d, ref)

    # `endpoint_pi_mean` has no producer in this round; recompute it from the same
    # formula the upstream registrar used, on THIS arm's water and gate.
    gate = rebuilt['hc2_gate']
    pilot = [int(d.global_reach_ids[i]) for i in np.unique(ref.reach_id).astype(int) - 1]
    rr = [d.global_reach_ids.index(r) for r in gate['training_global_reaches']]
    pi = np.where(d.bfi_water_positive,
                  (1 + np.tanh((d.bfi - gate['mean']) / gate['sd'])) / 2, .5)
    rebuilt['endpoint_pi_mean'] = float(pi[rr].mean())

    design = copy.deepcopy(frozen)
    for key in MOVING:
        design[key] = rebuilt[key]
    design['endpoint_pi_mean'] = rebuilt['endpoint_pi_mean']

    # The unmoved keys must be equal, not merely carried: if `static_raw` or the
    # normalisation window had shifted, the arm would differ by more than water.
    moved = {}
    for key in UNMOVED:
        same = close(rebuilt[key], frozen[key])
        moved[key] = 'identical' if same else 'DIFFERS'
        if not same:
            design[key] = rebuilt[key]

    # `CARRIED` keys are deliberately left at their frozen value in `design`; the
    # rebuild is still checked against them so a genuine change could not hide
    # behind the sign the SVD leaves free.
    carried = {key: dict(zip(('equal', 'detail'), compare_design(key, rebuilt[key], frozen[key])))
               for key in CARRIED}
    return design, rebuilt, moved, dict(domain=domain, pilot_reaches=pilot,
                                        pi_mean=rebuilt['endpoint_pi_mean'],
                                        gate_mean=gate['mean'], gate_sd=gate['sd'],
                                        gate_reaches=gate['training_global_reaches'],
                                        unmoved=moved, carried=carried)


def main():
    assert Path(sys.prefix).name.lower() == 'sparrow', 'conda sparrow required'
    assert not (RUN / 'work/campaign.json').exists(), 'campaign already launched'

    ref = pd.read_parquet(RUN / 'data/common_design_metadata.parquet')
    ref = ref[['observation_id', 'reach_id', 'year']].copy()
    # `common_design_metadata` carries reach ids in the predecessor round's indexing.
    oldtop = rt.read(ROOT / '5_Test/20260915_5/data/domains/NH/topology.json')
    ref['reach_id'] = ref['reach_id'].map(lambda x: oldtop['global_reach_ids'][int(x) - 1])
    frozen = rt.read(RUN / 'data/frozen_design.json')

    report = dict(stage='20260917_5/prepare_bridge', arms={}, folds={})
    arms = {}
    for arm in ['H0', 'H1']:
        design, rebuilt, moved, info = arm_design(arm, ref, frozen)
        arms[arm] = dict(design=design, rebuilt=rebuilt, info=info)
        report['arms'][arm] = info
        print(f'[{arm}] pi_mean {info["pi_mean"]:.6f}  gate {info["gate_mean"]:.6f}'
              f'/{info["gate_sd"]:.6f}  unmoved {set(moved.values())}'
              f'  carried {[k for k, v in info["carried"].items() if not v["equal"]] or "ok"}',
              flush=True)

    # H0 must reproduce the frozen baseline, or the two arms are not the same grey
    # box and the comparison is uninterpretable (§6.1).  A genuine drift here (the
    # frozen design having been built on other water) shows up as a large delta and
    # is a stop; float round-off shows up as ~1e-16 and is recorded, not fatal.
    a, b = arms['H0']['info']['pi_mean'], frozen['endpoint_pi_mean']
    report['pi_mean_reproduction'] = dict(recomputed=a, frozen=b, abs_diff=abs(a - b))
    assert abs(a - b) < 1e-9, f'H0 endpoint_pi_mean drifted: {a} vs {b}'

    # Every key `build_design` produced is checked, not just the ones that moved --
    # including `scientific`, which `prepare_global.py` omitted from its own
    # reproduction loop (its sign is not reproducible; see the docstring).
    repro, drifted = {}, []
    for key in MOVING + CARRIED:
        equal, detail = compare_design(key, arms['H0']['rebuilt'][key], frozen[key])
        repro[key] = dict(equal=equal, detail=detail)
        if not equal:
            drifted.append(key)
    assert not drifted, f'H0 does not reproduce frozen_design: {drifted}'
    report['h0_reproduction'] = dict(tolerance=TOL, drifted_keys=drifted, keys=repro)
    print(f'[H0] design reproduces frozen_design.json to {TOL:g} '
          f'(pi_mean {b:.12f}, delta {abs(a - b):.2e}; '
          f'all {len(repro)} rebuilt keys verified)', flush=True)

    # The arm difference must land in the water-derived keys and nowhere else.
    keys = sorted(set(arms['H0']['design']) | set(arms['H1']['design']))
    diff = {k: compare_design(k, arms['H0']['design'].get(k),
                              arms['H1']['design'].get(k))[0] for k in keys}
    changed = sorted(k for k, v in diff.items() if not v)
    report['design_diff'] = dict(changed=changed,
                                 identical=sorted(k for k, v in diff.items() if v))
    assert set(changed) <= set(MOVING + ['endpoint_pi_mean']), \
        f'arm difference outside the water-derived keys: {sorted(set(changed) - set(MOVING + ["endpoint_pi_mean"]))}'
    print(f'[design] H1 vs H0 differ in: {changed}', flush=True)

    # --- folds: H1 points at FULL24C, H1 folds point at H1 jobs -----------------
    folds = rt.read(RUN / 'configs/folds.json')
    jobs = rt.read(RUN / 'configs/jobs.json')
    for scope in SCOPES:
        for mode in MODES:
            base = f'{scope}_{mode}'
            home = f'{base}{ARM_SUFFIX}'
            assert base in folds, f'missing H0 fold {base}'
            parent = folds[base]
            assert parent['domain'] == 'FULL24', parent
            source = RUN / 'data/folds' / base
            target = RUN / 'data/folds' / home
            target.mkdir(parents=True, exist_ok=True)

            # Labels are copied, never rebuilt, and the copy is proven byte-identical.
            for name in ['train.parquet', 'registry.json']:
                (target / name).write_bytes((source / name).read_bytes())
                assert rt.sha(target / name) == rt.sha(source / name), f'copy drifted: {name}'
            assert rt.sha(target / 'train.parquet') == parent['train_sha256'], 'train sha moved'

            # Inherit the H0 fold's full key set -- including every round-specific
            # field this script does not understand -- and override only the water.
            # `CARRIED` keys are inherited along with everything else, so the H1
            # design is fitted against the mainline's exact PCA basis.
            design = copy.deepcopy(rt.read(RUN / 'data/designs' / f'{base}.json'))
            inherited = {k: compare_design(k, design.get(k), frozen.get(k))[0] for k in CARRIED}
            assert all(inherited.values()), f'{base} does not carry the frozen {CARRIED}'
            for key in MOVING:
                design[key] = arms['H1']['design'][key]
            design['endpoint_pi_mean'] = arms['H1']['design']['endpoint_pi_mean']
            # Provenance only -- no fit-path or validation code reads these four
            # (`temporal_model.py` reads `observation_registry_file`/`_hash`, which
            # is why they are repointed at this arm's byte-identical copy).
            design['arm'] = 'H1'
            design['hydro_domain'] = 'FULL24C'
            design['driver_version'] = 'corrected_2024'
            design['observation_registry_file'] = f'data\\folds\\{home}\\registry.json'
            design['observation_registry_hash'] = rt.sha(target / 'registry.json')
            rt.write(RUN / 'data/designs' / f'{home}.json', design)

            folds[home] = dict(parent, domain='FULL24C', arm='H1',
                               train_sha256=parent['train_sha256'], h0_fold=base)
            report['folds'][home] = dict(
                domain='FULL24C', train_sha256=parent['train_sha256'],
                train_identical=rt.sha(target / 'train.parquet') == rt.sha(source / 'train.parquet'),
                registry_identical=rt.sha(target / 'registry.json') == rt.sha(source / 'registry.json'),
                design_keys_overridden=sorted(MOVING + ['endpoint_pi_mean']))
            print(f'  {home}: labels copied byte-identical -> FULL24C', flush=True)

    # --- the real arm-difference check, on the files the model actually reads ---
    # The check above compared two in-memory designs built from the same frozen
    # copy, so it could not fail.  This one diffs the on-disk H0 fold design
    # against the H1 fold design this script just wrote -- that is what proves the
    # arm difference landed in the water and nowhere else.
    allowed = set(MOVING + ['endpoint_pi_mean'] + PROVENANCE)
    fold_diff = {}
    for home in sorted(report['folds']):
        base = folds[home]['h0_fold']
        left = rt.read(RUN / 'data/designs' / f'{base}.json')
        right = rt.read(RUN / 'data/designs' / f'{home}.json')
        # H1 may *add* provenance keys; it may not drop or rename any of H0's.
        added = set(right) - set(left)
        assert set(left) <= set(right) and added <= set(PROVENANCE), \
            f'{home} key set differs from {base}: added={sorted(added)} dropped={sorted(set(left) - set(right))}'
        changed = sorted(k for k in left
                         if not compare_design(k, left[k], right[k])[0])
        fold_diff[home] = dict(h0=base, changed=changed, added=sorted(added),
                               outside_water=sorted(set(changed) - allowed))
        assert not fold_diff[home]['outside_water'], \
            f'{home} differs from {base} outside the water: {fold_diff[home]["outside_water"]}'
    report['fold_design_diff'] = fold_diff
    for home in sorted(fold_diff):
        print(f'  [design] {home} vs {fold_diff[home]["h0"]}: '
              f'{fold_diff[home]["changed"]}', flush=True)

    # --- jobs: this round's eight folds, both arms, nothing else ----------------
    keep = [f'{scope}_{mode}' for scope in SCOPES for mode in MODES]
    pairs = {base: f'{base}{ARM_SUFFIX}' for base in keep}
    # A job's *dependencies* name a sibling fold, not its own: the D fold is
    # initialised from the M fold's fitted parameters.  So the remap is by fold
    # prefix over the whole round, longest first -- `T24_L_M_s0` belongs to fold
    # `T24_L_M`, not to `T24_L`.  Mapping by the job's own fold (as an earlier
    # revision did) silently left the H1 D jobs pointing at H0's M jobs, which
    # `fit_worker.py:32` would have read as H1's initial parameters.
    bases = sorted(pairs, key=len, reverse=True)

    def to_h1(tag):
        for base in bases:
            if tag == base or tag.startswith(base + '_'):
                return pairs[base] + tag[len(base):]
        return tag

    selected = [j for fold in keep for j in jobs if j['fold'] == fold]
    for home in sorted(report['folds']):
        base = folds[home]['h0_fold']
        for job in [j for j in jobs if j['fold'] == base]:
            clone = copy.deepcopy(job)
            clone['fold'] = home
            clone['tag'] = to_h1(job['tag'])
            clone['arm'] = 'H1'
            clone['dependencies'] = [to_h1(t) for t in job.get('dependencies', [])]
            clone['parent_tags'] = [to_h1(t) for t in job.get('parent_tags', [])]
            selected.append(clone)

    # Arm purity: no H1 job may read an H0 artifact.  This is the assertion that
    # would have caught the bug above -- the arms are separable only if every tag
    # an H1 job names belongs to H1.
    tags = {j['tag'] for j in selected}
    refs = {t for j in selected if j.get('arm') == 'H1'
            for t in list(j.get('dependencies', [])) + list(j.get('parent_tags', []))}
    leaked = sorted(t for t in refs if t not in tags or ARM_SUFFIX not in t)
    assert not leaked, f'H1 jobs reference non-H1 tags: {leaked}'

    # folds.json is pruned to the folds that actually run, so that seal's per-fold
    # loop and any reader cannot mistake a dormant fold for a live one.
    dropped = sorted(set(folds) - set(keep) - set(pairs.values()))
    folds = {k: folds[k] for k in keep + sorted(pairs.values())}
    report['jobs'] = dict(kept=sorted(tags), folds=sorted(folds),
                          dropped_folds=dropped, count=len(selected),
                          h1_dependencies={j['tag']: j['dependencies']
                                           for j in selected if j.get('arm') == 'H1'
                                           and j.get('dependencies')})
    rt.write(RUN / 'configs/folds.json', folds)
    rt.write(RUN / 'configs/jobs.json', selected)
    print(f'\njobs: {len(selected)} ({len(keep)} folds x 2 arms x 2 starts)'
          f'  dropped folds: {dropped}', flush=True)

    rt.write(RUN / 'reports/bridge_manifest.json', report)
    print('PREPARED_BRIDGE', len(selected), flush=True)


if __name__ == '__main__':
    main()
