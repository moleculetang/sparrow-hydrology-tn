"""The round's own pre-launch admission, in place of the five mainline validators.

`seal_global.py` was written for the mainline campaign -- 28 paths over two
domains, all five folds, a 2025 extension -- and its gates cannot pass here even
though nothing is wrong with this round's inputs:

  * `validate_global.py` iterates seven folds (StopIteration on four) and loads
    `FULL25`, which this round deliberately does not build.
  * `validate_common_outputs.py` / `verify_extension_contract.py` both load
    `FULL25` and score the 2025 window.
  * `validate_hf.py` scores the high-frequency panel of the mainline's own folds.

So the five report names `seal_global` demands are replaced by one, and this file
has to be at least as strong as what it replaces -- weaker would be a silent
downgrade of the launch gate, which is the one thing a round may not do to itself.
What it checks, in the order the report lists it:

  V1  inventory        every fold in `jobs.json` has its domain, design, labels
                       and registry, and the registry hash matches the design
  V2  arm separation   on disk: H1 labels are byte-identical to H0's, each H1
                       design differs from its H0 design only in the water keys,
                       and no H1 job names an H0 tag
  V3  switch effective `temperature` is byte-identical across the two domains and
                       `soil_water_mm` is not, so "only the water moved" holds
  V4  source identity  `source`, `crop` and `source_tags` are byte-identical, so
                       the arm difference cannot be nitrogen
  V5  fold admission   every fold of both arms builds a model, takes a finite
                       gradient, and closes its ledger
  V6  jacobian         a full 30-column central-difference Jacobian on the
                       largest fold, finite at every entry

V3 and V6 are the ones the mainline's validators would have covered had the
domains existed; V2 and V4 are this round's own guarantees and have no mainline
equivalent.  The report carries the process peak in both the shapes
`seal_global.py` reads (`process.peak_gib` and `peak.peak_gib`) so that sealing
needs one filename changed rather than a rewrite.
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

RUN = Path(__file__).resolve().parents[1]              # E:\SPARROW\5_Test\20260917_5
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
             'NUMBA_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_key, '1')
os.environ['NUMBA_CACHE_DIR'] = str(RUN / 'work/numba')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.insert(0, str(RUN / 'scripts'))

import native_runtime as rt                                                          # noqa: E402
import numpy as np                                                                   # noqa: E402
from campaign_model import for_job, residual                                         # noqa: E402

ARM_SUFFIX = '_H1'
# What H1 is allowed to move relative to its H0 fold.  Kept as literals rather
# than imported from `prepare_bridge` so that a change there cannot widen the gate
# here without also changing this file.
WATER_KEYS = ['dynamic_scales', 'endpoint_pi_mean', 'extra_mean', 'extra_sd', 'hc2_gate']
PROVENANCE_KEYS = ['arm', 'hydro_domain', 'driver_version',
                   'observation_registry_file', 'observation_registry_hash']
# The hydrology-independent arrays, by the argument in `prepare_bridge`: `static_raw`
# drives the spatial basis, and `source`/`crop`/`source_tags` are the nitrogen side
# that this round does not switch.  `temperature` is the clean control: the corrected
# line reproduces it to 7.1e-15, so it must come through the override byte-identical.
# Asserted as two named claims rather than one list, so that a wrong guess about some
# other array cannot pass for a check on these.
IDENTICAL_CLAIMED = ['temperature', 'source', 'crop', 'source_tags', 'static_raw']
SWITCHED_ARRAYS = ['soil_water_mm', 'soil_wetness']
# Ledger tolerances in kg.  Both are float64 accumulation residuals over a full
# training window, so neither can be exact.  Measured on the largest fold: local
# balance 8.7e-9, source-label sums 4.6e-8 at worst.  The bounds below sit ~20x
# above those so they catch a real imbalance rather than float noise; the measured
# worst is written into the report so the margin stays auditable.
BALANCE_TOL_KG = 1e-6
LABEL_SUM_TOL_KG = 1e-6
# Largest fold of each arm: the Jacobian and the reservation are measured on the
# most expensive one so the peak is not understated.
HEAVIEST = 'T24_G_D'


def v1_inventory(folds, jobs):
    detail, failures = {}, []
    for fold, cfg in sorted(folds.items()):
        row = dict(domain=cfg['domain'])
        folder = RUN / 'data/domains' / cfg['domain']
        row['arrays'] = len(json.loads((folder / 'arrays.json').read_text(encoding='utf-8')))
        design_path = RUN / 'data/designs' / f'{fold}.json'
        row['design'] = design_path.exists()
        label = RUN / 'data/folds' / fold / 'train.parquet'
        registry = RUN / 'data/folds' / fold / 'registry.json'
        row['labels'] = label.exists() and registry.exists()
        if row['design'] and row['labels']:
            design = json.loads(design_path.read_text(encoding='utf-8'))
            row['registry_hash_matches'] = rt.sha(registry) == design['observation_registry_hash']
            row['registry_file_matches'] = Path(str(design['observation_registry_file']).replace('\\', '/')
                                                ).name == 'registry.json'
        else:
            row['registry_hash_matches'] = False
        row['jobs'] = sum(1 for j in jobs if j['fold'] == fold)
        ok = (row['arrays'] == 26 and row['design'] and row['labels']
              and row['registry_hash_matches'] and row['registry_file_matches']
              and row['jobs'] > 0)
        row['passed'] = bool(ok)
        if not ok:
            failures.append(fold)
        detail[fold] = row
    return dict(passed=not failures, folds=detail, failures=failures,
                n_folds=len(folds), n_jobs=len(jobs))


def v2_arm_separation(folds, jobs):
    detail, failures = {}, []
    for fold, cfg in sorted(folds.items()):
        if not fold.endswith(ARM_SUFFIX):
            continue
        base = fold[:-len(ARM_SUFFIX)]
        row = dict(h0=base)
        for name in ['train.parquet', 'registry.json']:
            row[name] = rt.sha(RUN / 'data/folds' / fold / name) == \
                rt.sha(RUN / 'data/folds' / base / name)
        left = json.loads((RUN / 'data/designs' / f'{base}.json').read_text(encoding='utf-8'))
        right = json.loads((RUN / 'data/designs' / f'{fold}.json').read_text(encoding='utf-8'))
        row['keys_dropped'] = sorted(set(left) - set(right))
        changed = []
        for key in left:
            if json.dumps(left[key], sort_keys=True) != json.dumps(right[key], sort_keys=True):
                changed.append(key)
        # `hc2_gate` carries a reach list whose order is unstable; a reordering is
        # not an arm difference.  Compare it as a set, as `prepare_global.py` did.
        row['changed'] = sorted(changed)
        row['outside_registered'] = sorted(set(changed) - set(WATER_KEYS + PROVENANCE_KEYS))
        row['gate_reaches_same_set'] = (
            set(left['hc2_gate']['training_global_reaches'])
            == set(right['hc2_gate']['training_global_reaches']))
        row['passed'] = bool(row['train.parquet'] and row['registry.json']
                             and not row['keys_dropped']
                             and not row['outside_registered']
                             and row['gate_reaches_same_set'])
        if not row['passed']:
            failures.append(fold)
        detail[fold] = row

    # Arm purity of the job graph: the assertion that caught the D-fold dependency
    # leak, where H1's daily arm would have been initialised from H0's parameters.
    tags = {j['tag'] for j in jobs}
    h1 = [j for j in jobs if j['tag'].endswith(ARM_SUFFIX + '_s0') or j['tag'].endswith(ARM_SUFFIX + '_s1')]
    refs = {t for j in h1 for t in list(j.get('dependencies', [])) + list(j.get('parent_tags', []))}
    leaked = sorted(t for t in refs if t not in tags or ARM_SUFFIX not in t)
    return dict(passed=not failures and not leaked, folds=detail, failures=failures,
                h1_jobs=len(h1), cross_arm_references=leaked)


def v3_switch_effective():
    left = json.loads((RUN / 'data/domains/FULL24/arrays.json').read_text(encoding='utf-8'))
    right = json.loads((RUN / 'data/domains/FULL24C/arrays.json').read_text(encoding='utf-8'))
    assert set(left) == set(right), 'the two domains do not carry the same arrays'
    changed = sorted(n for n in left if left[n]['sha256'] != right[n]['sha256'])
    identical = sorted(n for n in left if left[n]['sha256'] == right[n]['sha256'])
    claimed_moved = [n for n in IDENTICAL_CLAIMED if n in changed]
    moved = [n for n in SWITCHED_ARRAYS if n in left]
    detail = dict(n_arrays=len(left), changed=changed, identical=identical,
                  claimed_identical_moved=claimed_moved,
                  switched_moved=[n for n in moved if n in changed],
                  switched_static=[n for n in moved if n not in changed])
    # Both halves are required: `soil_water_mm` identical would mean the override
    # silently failed, and `temperature` different would mean it moved more than water.
    ok = (not claimed_moved and detail['switched_moved'] and not detail['switched_static'])
    return dict(passed=bool(ok), **detail)


def v4_source_identity():
    left = json.loads((RUN / 'data/domains/FULL24/arrays.json').read_text(encoding='utf-8'))
    right = json.loads((RUN / 'data/domains/FULL24C/arrays.json').read_text(encoding='utf-8'))
    names = ['source', 'crop', 'source_tags']
    detail = {n: dict(identical=left[n]['sha256'] == right[n]['sha256'],
                      sha256=left[n]['sha256']) for n in names}
    return dict(passed=all(r['identical'] for r in detail.values()), arrays=detail)


def v5_fold_admission(jobs, cap_gib):
    """Every fold of both arms must build, differentiate and close its ledger."""
    detail, failures = {}, []
    for job in sorted(jobs, key=lambda j: j['tag']):
        started = time.time()
        row = dict(fold=job['fold'], start=job['start'])
        try:
            model = for_job(job)
            x = np.asarray(model.initial(job['start']), float)
            row['parameters'] = int(x.size)
            row['finite_initial'] = bool(np.isfinite(x).all())
            jac, grad = model.value_gradient(x)
            row['finite_gradient'] = bool(np.isfinite(np.asarray(grad)).all()
                                          and np.isfinite(np.asarray(jac)).all())
            row['residual_finite'] = bool(np.isfinite(residual(model, x)).all())
            book = model.ledger(x)
            row['local_balance_max_kg'] = float(book['local_balance_max_kg'])
            row['network_balance_kg'] = float(book['network_balance_kg'])
            row['ledger_finite'] = bool(np.isfinite(row['local_balance_max_kg'])
                                        and np.isfinite(row['network_balance_kg']))
            if 'source_label_sum_errors' in book:
                errors = {k: float(v) for k, v in book['source_label_sum_errors'].items()}
                row['source_label_sum_errors'] = errors
                row['worst_label_sum_kg'] = max(errors.values())
                row['labels_close'] = row['worst_label_sum_kg'] < LABEL_SUM_TOL_KG
            memories = rt.process(os.getpid())
            row['peak_gib'] = float(memories['peak_gib'])
            row['over_cap'] = row['peak_gib'] > cap_gib
            row['passed'] = bool(row['finite_initial'] and row['finite_gradient']
                                 and row['residual_finite'] and row['ledger_finite']
                                 and row.get('labels_close', True)
                                 and row['local_balance_max_kg'] < BALANCE_TOL_KG
                                 and not row['over_cap'])
            del model, jac, grad, book
        except Exception as exc:                                   # noqa: BLE001
            row['passed'] = False
            row['error'] = f'{type(exc).__name__}: {exc}'
        row['seconds'] = time.time() - started
        if not row['passed']:
            failures.append(job['tag'])
        detail[job['tag']] = row
        gc.collect()
        print(f'  [V5] {job["tag"]:<18} {"ok" if row["passed"] else "FAIL"} '
              f'{row["seconds"]:.1f}s', flush=True)
    return dict(passed=not failures, jobs=detail, failures=failures,
                tolerance_kg=float(BALANCE_TOL_KG), label_tolerance_kg=float(LABEL_SUM_TOL_KG),
                worst_local_balance_kg=max(r.get('local_balance_max_kg', float('nan'))
                                           for r in detail.values()),
                worst_label_sum_kg=max(r.get('worst_label_sum_kg', float('nan'))
                                       for r in detail.values()))


def v6_jacobian(jobs, cap_gib):
    """Full central-difference Jacobian on the heaviest fold of each arm."""
    detail, failures = {}, []
    for arm, suffix in [('H0', ''), ('H1', ARM_SUFFIX)]:
        tag = f'{HEAVIEST}{suffix}_s0'
        job = next(j for j in jobs if j['tag'] == tag)
        model = for_job(job)
        x = np.asarray(model.initial(0), float)
        started = time.time()
        columns = []
        for i in range(x.size):
            h = 1e-5 * max(1.0, abs(x[i]))
            up, down = x.copy(), x.copy()
            up[i] += h
            down[i] -= h
            columns.append((residual(model, up) - residual(model, down)) / (2 * h))
        jac = np.column_stack(columns)
        memories = rt.process(os.getpid())
        row = dict(tag=tag, shape=list(jac.shape),
                   finite=bool(np.isfinite(jac).all()),
                   max_abs=float(np.abs(jac).max()),
                   all_zero_columns=int((np.abs(jac).max(axis=0) == 0).sum()),
                   seconds=time.time() - started, peak_gib=float(memories['peak_gib']))
        row['over_cap'] = row['peak_gib'] > cap_gib
        row['passed'] = bool(row['finite'] and not row['over_cap'])
        detail[arm] = row
        if not row['passed']:
            failures.append(tag)
        del model, jac, columns
        gc.collect()
        print(f'  [V6] {tag:<18} {"ok" if row["passed"] else "FAIL"} '
              f'{row["seconds"]:.1f}s  max|J| {row["max_abs"]:.3g}', flush=True)
    return dict(passed=not failures, arms=detail, failures=failures)


def main():
    assert Path(sys.prefix).name.lower() == 'sparrow', 'conda sparrow required'
    assert not (RUN / 'work/campaign.json').exists(), 'campaign already launched'
    started = time.time()
    folds = rt.read(RUN / 'configs/folds.json')
    jobs = rt.read(RUN / 'configs/jobs.json')
    report = dict(stage='20260917_5/validate_bridge',
                  started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    # `rt.process()['peak_gib']` is a memory *amount*, so the ceiling is the headroom
    # one cold worker may take under `native_runtime.admission()`'s own rule -- RAM
    # stays below `dispatch_ram` percent of total -- not that percentage itself.
    campaign = rt.read(RUN / 'configs/campaign.json')
    limit = float(campaign['resources'].get('dispatch_ram', 90)) / 100.0
    now = rt.resources()
    used_gib = now['total_gib'] - now['available_gib']
    cap_gib = limit * now['total_gib'] - used_gib
    report['cap'] = dict(limit_fraction=limit, total_gib=now['total_gib'],
                         used_gib=used_gib, headroom_gib=cap_gib,
                         reservation=campaign['resources'].get('peak_factor'))

    for name, function in [('V1_inventory', lambda: v1_inventory(folds, jobs)),
                           ('V2_arm_separation', lambda: v2_arm_separation(folds, jobs)),
                           ('V3_switch_effective', v3_switch_effective),
                           ('V4_source_identity', v4_source_identity),
                           ('V5_fold_admission', lambda: v5_fold_admission(jobs, cap_gib)),
                           ('V6_jacobian', lambda: v6_jacobian(jobs, cap_gib))]:
        print(f'[{name}]', flush=True)
        report[name] = function()
        print(f'[{name}] {"PASS" if report[name]["passed"] else "FAIL"}', flush=True)

    failures = [k for k, v in report.items()
                if isinstance(v, dict) and v.get('passed') is False]
    peak = float(rt.process(os.getpid())['peak_gib'])
    report['status'] = 'PASS_BRIDGE_VALIDATION' if not failures else 'FAIL_BRIDGE_VALIDATION'
    report['failures'] = failures
    # Both shapes, because `seal_global.py:18` reads `process.peak_gib` from the
    # global validator and `peak.peak_gib` from the boundary validator.
    report['process'] = dict(peak_gib=peak)
    report['peak'] = dict(peak_gib=peak)
    report['seconds'] = time.time() - started
    rt.write(RUN / 'reports/bridge_validation.json', report)
    print(f'\n{report["status"]}  peak {peak:.3f} GiB  {report["seconds"]:.1f}s', flush=True)
    assert not failures, f'bridge validation failed: {failures}'


if __name__ == '__main__':
    main()
