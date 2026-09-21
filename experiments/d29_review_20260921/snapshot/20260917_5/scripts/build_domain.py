"""Derive one domain's arrays from a hydrology product; upstream stays read-only.

This is the round's only new derivation code.  It exists because neither of the
two mainline producers can do H1's job:

  * `gb_model.data_cache('sensitivity')` is a bare `torch.load` of a pre-built
    `.pt`.  It never recomputes anything, so pointing it at a new hydrology
    product would silently return the old arrays.
  * `bootstrap_global.py` does not derive either.  It `np.load`s the registered
    `20260914_1/data/cache/*.npy` and slices them by year, so it can only ever
    reproduce the arrays that cache already holds.

So this script takes over `20260914_1/scripts/prepare.py` lines 22-47 -- the
part that turns a hydrology directory into the 26-array cache -- and calls the
same derivation function for both arms:

    fc_data.load_data('sensitivity', 'S1', verify=False)

`audit_inputs` (imported first, below) decides which directory `'sensitivity'`
resolves to.  With no override that is `20260828_38`; with
`SPARROW_HYDRO_OVERRIDE` it is the corrected line.  The code path is therefore
byte-for-byte identical across arms, which is what makes the arm difference a
single variable rather than two code paths that happen to agree.

`verify=False` is mandatory: `tn_reference.load_data` checks every hydrology path
against `20260905_1/reports/input_manifest.json`, and a corrected directory has no
entry there.  The cost of skipping that check is paid here -- `provenance.json`
and `reports/input_manifest_<arm>.json` record the sha256 of every hydrology file
actually read, so the round carries its own manifest instead of borrowing one.

For arm H0 the resulting arrays are additionally asserted equal, per array, to
the registered `20260914_1/data/cache/arrays.json`.  That assertion is what
proves `fc_data.load_data` reproduces `gb_model.data_cache` exactly, and hence
that H1 differs from H0 only by the hydrology product it names.

--- capacity (Gate 0-5) -------------------------------------------------------
`fc_data.load_data` hardcodes the bucket capacity it divides soil water by:

    soil_wetness = clip((soil_storage_mm + actual_aet_mm_day) / capacity, 0, 1)

at `20260828_9/outputs/parent_preserving_state_consistent_model.pt`.  That is the
*old* line's parameter file.  Each hydrology line's own storage saturates exactly
at its own capacity -- measured: `20260828_38` reaches 465.0671 against capacity
465.0671, and the corrected line reaches 376.8416 against its own 376.8416 -- so
storage is expressed in its own line's units.  Pairing corrected storage with the
frozen capacity would divide by the wrong scale and multiply `soil_wetness` by
0.8103 across the board; `W` enters five of the eight dynamic basis columns
(`W, delta, delta*W, q*W, T*W`), so that artefact would land directly in the
comparison.  Each arm therefore recomputes `soil_wetness` from its own parameter
file, and H0 asserts the recomputation reproduces `fc_data`'s value bit-for-bit
so the substitution is provably faithful rather than merely plausible.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

RUN = Path(__file__).resolve().parents[1]           # E:\SPARROW\5_Test\20260917_5
ROOT = RUN.parents[1]                               # E:\SPARROW

for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
             'NUMBA_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_key, '1')
os.environ['NUMBA_CACHE_DIR'] = str(RUN / 'work/numba')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.insert(0, str(RUN / 'scripts'))

# Must precede every other import: this seeds sys.modules['audit_inputs'].
import audit_inputs                                                                 # noqa: E402
sys.path.insert(0, str(ROOT / '5_Test/20260908_1/scripts'))
from fc_data import load_data as fc_load_data                                       # noqa: E402
import numpy as np                                                                  # noqa: E402
import pandas as pd                                                                 # noqa: E402
import torch                                                                        # noqa: E402

# The 23 arrays `prepare.py` takes off the data object, in its order.
NAMES = ['contact', 'fast_fraction', 'lower_release', 'fast_water', 'slow_water',
         'official_water', 'h_day', 'temperature', 'upper_water', 'percolation',
         'soil_wetness', 'source', 'crop', 'h_month', 'source_tags', 'static_raw',
         'area_ha', 'mid', 'starts', 'stops', 'release_fraction', 'released_water',
         'enabled']
# prepare.py reads `soil_water_mm` straight off the hydrology table, and
# `soil_wetness` is recomputed here from the same two columns.
WATER_COLUMNS = ['date', 'reach_id', 'soil_storage_mm', 'actual_aet_mm_day']

ARMS = {
    'H0': dict(
        domain='FULL24',
        override=False,
        hydrology='20260828_38/outputs',
        capacity='20260828_9/outputs/parent_preserving_state_consistent_model.pt',
        note='frozen sensitivity hydrology with the parameters that produced it'),
    'H1': dict(
        domain='FULL24C',
        override=True,
        hydrology='20260917_3/work/screen/outputs',
        capacity='20260917_3/work/screen/outputs/export/parent_preserving_state_consistent_model.pt',
        note='corrected hydrology with the parameters that produced it'),
}

REFERENCE_CACHE = ROOT / '5_Test/20260914_1/data/cache'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()


def plain(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=plain),
                    encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('arm', choices=sorted(ARMS))
    args = parser.parse_args()
    arm = args.arm
    spec = ARMS[arm]
    domain = spec['domain']

    assert Path(sys.prefix).name.lower() == 'sparrow', 'conda sparrow required'
    override = os.environ.get('SPARROW_HYDRO_OVERRIDE')
    if spec['override']:
        assert override, f'{arm} requires SPARROW_HYDRO_OVERRIDE'
    else:
        assert not override, f'{arm} must run without SPARROW_HYDRO_OVERRIDE'
    hydro_dir = audit_inputs.HYDRO['sensitivity']
    capacity_file = ROOT / '5_Test' / spec['capacity']
    assert hydro_dir.name == Path(spec['hydrology']).name, hydro_dir
    started = time.time()

    # --- derivation ---------------------------------------------------------
    print(f'[{arm}] deriving from {hydro_dir}', flush=True)
    d = fc_load_data('sensitivity', 'S1', verify=False)

    full = {}
    for name in NAMES:
        value = np.asarray(getattr(d, name))
        assert value.shape[0] in (len(d.dates), len(d.months), 230), (name, value.shape)
        full[name] = value
    full['dates'] = d.dates.to_numpy('datetime64[D]')
    full['months'] = d.months.to_numpy('datetime64[D]')

    water = pd.read_parquet(hydro_dir / 'tn_hydrology_reach_daily.parquet',
                            columns=WATER_COLUMNS).sort_values(['date', 'reach_id'])
    assert np.array_equal(
        water.date.to_numpy().reshape(-1, 230)[:, 0].astype('datetime64[D]'),
        d.dates.to_numpy('datetime64[D]')), 'soil water calendar differs from the daily hydrology'
    soil = water.soil_storage_mm.to_numpy(np.float64).reshape(-1, 230)
    assert np.isfinite(soil).all() and soil.min() >= 0, 'negative native soil water'
    full['soil_water_mm'] = soil

    # --- capacity: each arm divides by its own line's parameter -------------
    parameters = torch.load(capacity_file, weights_only=True,
                            map_location='cpu')['physical_parameters'].numpy()
    capacity = (parameters[:, 0] if parameters.ndim == 2
                else np.full(len(d.order), parameters[0]))
    numerator = (water.soil_storage_mm + water.actual_aet_mm_day).to_numpy(np.float64)
    wetness = np.clip(numerator.reshape(-1, 230) / capacity, 0, 1)
    saturation = float(numerator.max() / float(np.max(capacity)))
    if arm == 'H0':
        # Proves the substitution below is faithful, so H1's wetness differs only
        # by the capacity it names and not by a reimplementation of the formula.
        assert np.array_equal(wetness, np.asarray(d.soil_wetness)), \
            'recomputed soil_wetness does not reproduce fc_data'
        print(f'[{arm}] recomputed soil_wetness reproduces fc_data exactly', flush=True)
    full['soil_wetness'] = wetness
    assert len(full) == 26, sorted(full)

    # --- write the full-length cache (prepare.py's contract) ----------------
    cache = RUN / 'data' / f'cache_{arm}'
    cache.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, value in full.items():
        path = cache / f'{name}.npy'
        np.save(path, value, allow_pickle=False)
        reread = np.load(path, mmap_mode='r', allow_pickle=False)
        assert reread.shape == value.shape and reread.dtype == value.dtype, name
        assert np.array_equal(reread, value, equal_nan=value.dtype.kind == 'f'), name
        manifest[name] = dict(file=path.name, sha256=sha(path),
                              shape=list(value.shape), dtype=value.dtype.str)

    # H0 must reproduce the registered cache byte-for-byte, or the two arms are
    # not running the same derivation and the comparison means nothing.
    reference = json.loads((REFERENCE_CACHE / 'arrays.json').read_text(encoding='utf-8'))
    unchanged = sorted(n for n in reference if manifest[n]['sha256'] == reference[n]['sha256'])
    changed = sorted(n for n in reference if manifest[n]['sha256'] != reference[n]['sha256'])
    if arm == 'H0':
        assert not changed, f'H0 does not reproduce the registered cache: {changed}'
        print(f'[{arm}] cache reproduces the registered cache exactly (26/26)', flush=True)
    else:
        print(f'[{arm}] vs registered cache: {len(changed)} changed, {len(unchanged)} identical',
              flush=True)
    put(cache / 'arrays.json', manifest)

    # --- slice to the evaluation domain (bootstrap_global.py's contract) ----
    nd = int((d.dates.year <= 2024).sum())
    nm = int((d.months.year <= 2024).sum())
    folder = RUN / 'data/domains' / domain
    folder.mkdir(parents=True, exist_ok=True)
    specs = {}
    for name, value in full.items():
        if value.ndim and value.shape[0] == len(d.dates):
            value = value[:nd]
        elif value.ndim and value.shape[0] == len(d.months):
            value = value[:nm]
        path = folder / f'{name}.npy'
        np.save(path, value, allow_pickle=False)
        specs[name] = dict(file=path.name, sha256=sha(path),
                           shape=list(value.shape), dtype=value.dtype.str)
    put(folder / 'arrays.json', specs)

    topo = {k: getattr(d, k) for k in ['product', 'forcing', 'order', 'downstream',
                                       'terminal', 'metadata', 'static_fields']}
    topo['global_reach_ids'] = list(range(1, 231))
    topo['global_reservoir_indices'] = list(range(13))
    # Mirrors bootstrap_global.py: the 2024 domain is labelled 'formal' even
    # though its arrays come from the sensitivity line.
    topo['product'] = 'formal'
    put(folder / 'topology.json', topo)

    # --- the other arm's view: what the switch actually moved ---------------
    comparison = None
    baseline = RUN / 'data/cache_H0/arrays.json'
    if arm != 'H0' and baseline.exists():
        base = json.loads(baseline.read_text(encoding='utf-8'))
        detail = {}
        for name in sorted(full):
            same = base[name]['sha256'] == manifest[name]['sha256']
            row = dict(identical=same)
            if not same and full[name].dtype.kind == 'f':
                other = np.load(RUN / 'data/cache_H0' / base[name]['file'],
                                mmap_mode='r', allow_pickle=False)
                if other.shape == full[name].shape:
                    delta = np.abs(full[name] - np.asarray(other, dtype=full[name].dtype))
                    row['max_abs'] = float(np.nanmax(delta))
                    row['max_rel'] = float(np.nanmax(delta / np.maximum(np.abs(other), 1e-12)))
            detail[name] = row
        comparison = dict(baseline=str(baseline),
                          identical=sorted(n for n, r in detail.items() if r['identical']),
                          changed=sorted(n for n, r in detail.items() if not r['identical']),
                          detail=detail)
        print(f'[{arm}] vs H0: {len(comparison["changed"])} changed, '
              f'{len(comparison["identical"])} identical', flush=True)

    # --- our own manifest, in place of the upstream one we cannot reuse -----
    hydrology = ['tn_hydrology_reach_daily.parquet', 'tn_hydrology_reach_monthly.parquet',
                 'tn_hydrology_reservoir_daily.parquet',
                 'tn_hydrology_reservoir_static_metadata.parquet']
    companion = {'source': str(audit_inputs.SOURCE['sensitivity']),
                 'composition': str(ROOT / '5_Test/20260907_3/outputs/source_composition_sensitivity.parquet'),
                 'topology_edges': str(audit_inputs.TOPO),
                 'static_features': str(ROOT / '5_Test/20260905_1/outputs/h7_raw_features.parquet'),
                 'capacity_parameters': str(capacity_file),
                 'fc_data': str(ROOT / '5_Test/20260908_1/scripts/fc_data.py'),
                 'tn_reference': str(ROOT / '5_Test/20260907_3/scripts/tn_reference.py')}
    provenance = dict(
        arm=arm, domain=domain, note=spec['note'],
        hydro_override=override, hydro_directory=str(hydro_dir),
        hydrology={name: sha(hydro_dir / name) for name in hydrology},
        companion={key: sha(Path(path)) for key, path in companion.items()},
        capacity=dict(file=str(capacity_file), value=capacity.tolist(),
                      numerator_max_mm=float(numerator.max()),
                      saturation_ratio=saturation),
        arrays_cache=str(cache / 'arrays.json'), cache_manifest_sha256=sha(cache / 'arrays.json'),
        domain_manifest_sha256=sha(folder / 'arrays.json'),
        days_kept=nd, months_kept=nm,
        reference_comparison=comparison,
        domain_temperature_sha256=specs['temperature']['sha256'],
        domain_soil_water_sha256=specs['soil_water_mm']['sha256'],
        soil_water_mm=dict(minimum=float(soil.min()), mean=float(soil.mean())),
        soil_wetness=dict(maximum=float(wetness.max()), mean=float(wetness.mean()),
                          clipped_fraction=float((numerator.reshape(-1, 230) / capacity >= 1).mean())),
        reservoir_metadata=topo['metadata'],
        seconds=time.time() - started)
    put(folder / 'provenance.json', provenance)
    put(RUN / 'reports' / f'input_manifest_{arm}.json', provenance)
    print(f'[{arm}] capacity {float(np.max(capacity)):.4f}  saturation {saturation:.6f}  '
          f'W max {float(wetness.max()):.4f}', flush=True)
    print(f'[{arm}] wrote {folder} ({len(specs)} arrays, {nd} days x 230 reaches)', flush=True)


if __name__ == '__main__':
    main()
