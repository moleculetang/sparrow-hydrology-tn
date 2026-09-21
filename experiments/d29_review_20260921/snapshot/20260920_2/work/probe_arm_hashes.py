"""Reproduce every hash in the frozen `reports/arms.json` from `common24` + `dp_kernel`.

Zero forwards.  This is the cheapest place to catch a mistake: if the `V` construction, the
`guard`, the lower-store read or the calendar has drifted since the arm table was frozen, it
shows up here as a hash mismatch rather than as a plausible-looking result later.

It also settles ONE naming trap that the arm table carries and the report must not carry
silently:

    `Vu_post` is the PRE-outflow carry, and `Vu_at_outflow = "Vu_post + Qu"`.

The `_post` suffix means "the state the producer exported", and the producer exports the
POST-outflow carry -- so `Vu_post + Qu` is the volume at the instant outflow begins, which
is what `dp_kernel.volumes` returns and what every concentration divides by.  Read as
"the volume after outflow" the name would be wrong by a whole day's flux.  The trap is
registered here and in `实际方法与偏离.md`; the field NAME is frozen and is not renamed.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common24 as C
import dp_kernel as XI

R = C.ROUND
OUT = R / 'reports'


def arr_sha(a):
    return C.hashlib.sha256(np.ascontiguousarray(a, np.float64).tobytes()).hexdigest()


def set_Vu(model, raw):
    """`common24.dp_alternate_Vu`'s body with the source array supplied.

    `dp_alternate_Vu` only accepts the three registered names; the diagnostic arm needs the
    same construction from a fourth array.  Rather than special-case it, this is the same
    three lines and `main` asserts it agrees with `dp_alternate_Vu` on all three names, so
    the shared path is proven rather than assumed.
    """
    pack = model.dp_fractions
    V = XI.volumes(np.asarray(raw, np.float64), pack['Vs'] - pack['Qs'], pack)
    frac = XI.fractions(pack, V, pack['form'], pack['guard'])
    out = dict(pack)
    out.update(Vu=V['Vu'], **frac)
    model.dp_fractions = out
    return out


def main():
    arms = C.read_json(OUT / 'arms.json')
    H = arms['hashes']
    model = C.build()
    out = {'arm_table_sha256': H['table_sha256'], 'checks': {}}

    d = model.data
    dates = np.asarray(d.dates)

    # --- the frozen hydrology and the frozen calendar
    F = XI.flows(np.asarray(d.fast_water, np.float64),
                 np.asarray(d.percolation, np.float64),
                 np.asarray(d.slow_water, np.float64),
                 np.asarray(d.area_ha, np.float64))
    got = dict(Qf=arr_sha(F['Qf']), Qp=arr_sha(F['Qp']), Qs=arr_sha(F['Qs']),
               Qu=arr_sha(F['Qu']),
               dates=arr_sha(dates.astype('datetime64[D]').astype(np.int64)))
    for k, v in got.items():
        out['checks']['hashes.' + k] = dict(got=v, frozen=H[k], ok=bool(v == H[k]))

    lower = C.load_lower_storage(model)
    out['checks']['hashes.lower_storage'] = dict(
        got=arr_sha(lower), frozen=H['lower_storage'],
        ok=bool(arr_sha(lower) == H['lower_storage']))

    guard = XI.guard_from(np.asarray(d.fast_fraction, np.float64))
    out['checks']['hashes.guard'] = dict(got=arr_sha(guard), frozen=H['guard'],
                                         ok=bool(arr_sha(guard) == H['guard']))

    V = XI.volumes(np.asarray(d.upper_water, np.float64), lower, F)
    out['checks']['hashes.Vs'] = dict(got=arr_sha(V['Vs']), frozen=H['Vs'],
                                      ok=bool(arr_sha(V['Vs']) == H['Vs']))
    out['checks']['hashes.Vu_P_upper'] = dict(
        got=arr_sha(V['Vu']), frozen=H['Vu_P_upper'],
        ok=bool(arr_sha(V['Vu']) == H['Vu_P_upper']))
    frac = XI.fractions(F, V, arms['closure_form'], guard)
    out['checks']['hashes.phi_f'] = dict(got=arr_sha(frac['phi_f']),
                                         frozen=H['phi_f'],
                                         ok=bool(arr_sha(frac['phi_f']) == H['phi_f']))
    out['checks']['hashes.gs'] = dict(got=arr_sha(frac['gs']), frozen=H['gs'],
                                      ok=bool(arr_sha(frac['gs']) == H['gs']))

    # --- the four `Vu_post` readings
    refmask = C.window_mask(dates, C.REF_YEARS)
    up = np.asarray(d.upper_water, np.float64)
    raw = {'upper_water': up,
           'soil_water_mm': np.asarray(d.soil_water_mm, np.float64),
           'soil_plus_upper': (np.asarray(d.soil_water_mm, np.float64) + up)}
    d_const = np.broadcast_to(up[refmask].mean(axis=0), up.shape).copy()
    raw['P-upper_reference_mean'] = d_const
    out['d_const'] = dict(
        definition='per-reach mean of `upper_water` over the reference window '
                   '[1961, 2020], broadcast back over all days',
        n_reference_days=int(refmask.sum()),
        per_reach_min=float(d_const[0].min()), per_reach_max=float(d_const[0].max()),
        is_time_invariant=bool(np.all(d_const == d_const[0])),
        n_reaches=int(d_const.shape[1]),
        note='time-structure diagnostic: the V_u TIME variation is removed and only the '
             'saturation nonlinearity survives. It is not a physical store and no '
             'producer declares it.')

    for a in arms['arms']:
        n = a['arm']
        if a.get('Vu_post') is None:
            continue
        got_sha = arr_sha(raw[a['Vu_post']])
        out['checks']['arm.%s.Vu_post' % n] = dict(
            source=a['Vu_post'], got=got_sha, frozen=a['Vu_post_sha'],
            ok=bool(got_sha == a['Vu_post_sha']))

    # --- `set_Vu` must equal `dp_alternate_Vu` on all three registered names
    C.dp_arrays(model, form=arms['closure_form'])
    for name, which in (('upper_water', 'upper'), ('soil_water_mm', 'soil'),
                        ('soil_plus_upper', 'unsat')):
        man = set_Vu(model, raw[name])
        C.dp_arrays(model, form=arms['closure_form'])
        lib = C.dp_alternate_Vu(model, which)
        same = all(np.array_equal(man[k], lib[k]) for k in
                   ('Vu', 'gu', 'phi_f', 'gs', 'xu', 'xs'))
        out['checks']['set_Vu_equals_dp_alternate_Vu.%s' % name] = dict(ok=bool(same))
        if not same:
            raise SystemExit('SET_VU_DISAGREES_WITH_THE_LIBRARY_HELPER %s' % name)

    bad = [k for k, v in out['checks'].items() if not v.get('ok', True)]
    out['n_checks'] = len(out['checks'])
    out['n_failed'] = len(bad)
    out['failed'] = bad
    out['all_reproduce'] = bool(not bad)
    out['naming_trap'] = 'Vu_post is the PRE-outflow carry; Vu_at_outflow = Vu_post + Qu'

    (OUT / 'probe_arm_hashes.json').write_text(
        json.dumps(out, indent=1, sort_keys=True, ensure_ascii=False, default=str),
        encoding='utf-8')
    print(json.dumps(out, indent=1, ensure_ascii=False, default=str))
    if bad:
        raise SystemExit('FROZEN_ARM_TABLE_HASHES_DO_NOT_REPRODUCE %r' % bad)


if __name__ == '__main__':
    main()
