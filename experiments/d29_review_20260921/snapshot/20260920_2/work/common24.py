"""Round `20260920_1` bootstrap: the frozen reference tree, the two water volumes, the
per-reach fraction triple, and the runtime install of the conserving dual-pathway
mobile-water-concentration kernel.

DERIVED FROM `20260919_5/work/common23.py`, WITH THE DELTA EXECUTABLE AND LISTED
-------------------------------------------------------------------------------
This is NOT a byte-identical copy and the paragraph does not claim one.  The deltas are:

  1. `assert ROUND.name == '20260919_5'` -> `'20260920_1'`.
  2. `import xi_k as XI` -> `import dp_kernel as XI`; `import closures_mc as MC` ->
     `import closures_dp as MC`.
  3. DELETED, because this round has no hazard and no fitted multiplier at all:
     `geometry`, `_GEOM`, `bootstrap_arrays`, `device_state`, `solve_k`,
     `routing_adjoint`, `build_xi_star`, `build_xi`, the `BETA_*` grids, `DEVICES`,
     `DEVICE_SPEC`, `WINDOWS`, `SAT_BOUND` and `DEGEN_FLOOR` as criteria.  In round 5
     those existed to serve `Xi_star = k_r * Xi_raw`; here the mobilisation is
     parameter-free, so none of them has an argument.
  4. NEW: `load_lower_storage`, `dp_arrays`, and the `ResearchObjective.ledger` rebind.

  The rebinding census, the install/restore machinery, the replay, the anchor gate and
  the frozen-hash report are round 5's, unchanged apart from the names above.

WHY THE LEDGER IS NOW A FIFTH BINDING AND WAS NOT IN ROUND 5
------------------------------------------------------------
`closures_mc.py:82-84` states that `closures.ResearchObjective.ledger` resolves the BARE
module-global name `scan`, which the round-5 install rebinds to `scan_mc_ledger` -- so
round 5 never needed to touch `ledger` itself.  That works only because round 5's kernel
KEPT the frozen `f` split.  This round's split is `phi_f = Q_f/Q_u`, a water share, so
the frozen line `L = np.cumsum(a*p*(1-f) - slow, axis=0)` would put a DIFFERENT split
into the `L` recurrence than the kernel used, and `local_balance_max_kg` would measure
the split difference instead of the float64 rounding it exists to measure.  Hence
`setattr(_closed.ResearchObjective, 'ledger', MC.ledger_dp)`.

This is a genuine correction to the plan text too: the plan's S10 says `common23.py`
binds five classes; it binds FOUR, and `ledger` is the one it does not, for a reason that
does not carry over.

WHAT THIS ROUND CHANGES, IN ONE SENTENCE
----------------------------------------
The land-phase mobilisation `E = av * (-expm1(-min(h,700)))` is replaced by one upper
mobile-water concentration multiplied by the two water shares of the same store, and a
second concentration reservoir on the slow path.  No parameter of the 30-vector is
touched, and 21 of the 30 fall out of the calculation entirely (plan S2.4).
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROUND = Path(__file__).resolve().parents[1]
PEER = Path(r'E:\SPARROW\5_Test\20260916_2')
R4 = Path(r'E:\SPARROW\5_Test\20260919_4')
assert ROUND.name == '20260920_1', ROUND
assert PEER.is_dir(), PEER
assert R4.is_dir(), R4

# `campaign_model.py` sets NUMBA_CACHE_DIR to PEER/work/numba at import.  Pin it HERE
# first so no compiled entry can be deposited inside a read-only earlier round.  This is
# also why this module may not import `common23.py`: THAT file writes its cache into
# `20260919_5/work/numba`, which is a write to an earlier round.
CACHE = ROUND / 'work' / 'numba'
CACHE.mkdir(parents=True, exist_ok=True)
os.environ['NUMBA_CACHE_DIR'] = str(CACHE)
import numba.core.config as _nbcfg  # noqa: E402

sys.path[:0] = [str(ROUND / 'work'),
                str(PEER / 'scripts'),
                str(PEER / 'vendor/expert/tn_challenge'),
                str(PEER / 'vendor/research'),
                str(PEER / 'vendor/transfer_research')]
# `eventlib.py` is the event/criterion apparatus and lives in the read-only round
# `20260919_2`.  APPENDED, not prepended: this round's own modules must win name lookup,
# and a module the sandbox cannot shadow is one whose hash check means something.
PEER_WORK = PEER.parent / '20260919_2' / 'work'
EVENTLIB_SHA = 'b680b51faedf6e24e38babe9061e468e3588ffc0eb080e316f3111bdef034f1a'
if str(PEER_WORK) not in sys.path:
    sys.path.append(str(PEER_WORK))
import campaign_model as cm  # noqa: E402

assert cm.RUN == PEER, (cm.RUN, PEER)
os.environ['NUMBA_CACHE_DIR'] = str(CACHE)
_nbcfg.CACHE_DIR = str(CACHE)

import torch  # noqa: E402
import closures as _closed  # noqa: E402
import tagged_transport as _tagged  # noqa: E402
import routing as _routing  # noqa: E402
import model as _hcm  # noqa: E402
import dp_kernel as XI  # noqa: E402
import closures_dp as MC  # noqa: E402
import eventlib as EL  # noqa: E402

assert hashlib.sha256((PEER_WORK / 'eventlib.py').read_bytes()).hexdigest() \
    == EVENTLIB_SHA, 'EVENTLIB_CHANGED'
assert Path(EL.__file__).resolve() == (PEER_WORK / 'eventlib.py').resolve(), EL.__file__

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

# --------------------------------------------------------------------------
# scalars
# --------------------------------------------------------------------------
TAG = 'C0_s1'
ANCHOR_TABLE = PEER / 'outputs/C0_s1/daily_station_mass_water.parquet'
ANCHOR_ROWS = 169476
ANCHOR_TOL = 1e-12
END_YEAR = 2024
REF_YEARS = (1961, 2020)          # strictly pre-evaluation
EVAL_YEARS = (2021, 2024)

# the ONLY hydrology producer whose store this round needs but whose value is not in the
# 26-array domain cache
HYDRO_PARQUET = (PEER.parent / '20260828_38' / 'outputs'
                 / 'tn_hydrology_reach_daily.parquet')
HYDRO_SHA = 'PLACEHOLDER--filled from 20260905_1/reports/input_manifest.json at run time'

# --------------------------------------------------------------------------
# FROZEN ANCHORS -- READ from their producers, never retyped (delivery check 1)
# --------------------------------------------------------------------------
R3 = PEER.parent / '20260919_3'
ANCHOR_PATHS = {
    'A_L1': ('20260919_3/reports/phase1_l1.json', ('anchors', 'A_L1'), 1.0400772083480145),
    'A_L2': ('20260919_2/reports/phase0_amplitude_budget.json',
             ('amplitude_budget', 'L2', 'amp_ratio_median'), 1.0217750264713188),
    'A_L3': ('20260919_3/reports/phase1_l1.json', ('anchors', 'A_L3'), 1.023217381861253),
    'A_obs': ('20260919_3/reports/phase1_l1.json', ('anchors', 'A_obs'), 1.2758737517831669),
    'G1_target_50pct': ('20260919_3/reports/phase1_l1.json',
                        ('anchors', 'G1_target_50pct'), 1.1579754800655908),
    'G2_target_50pct': ('20260919_3/reports/phase1_l1.json',
                        ('anchors', 'G2_target_50pct'), 1.1495455668222099),
    'c_base_L1': ('20260919_2/reports/phase0_amplitude_budget.json',
                  ('amplitude_budget', 'L1', 'c_base_median'), 0.6447085818470569),
    'c_peak_L1': ('20260919_2/reports/phase0_amplitude_budget.json',
                  ('amplitude_budget', 'L1', 'c_peak_median'), 0.860765144866467),
    'c_base_L3': ('20260919_2/reports/phase0_amplitude_budget.json',
                  ('amplitude_budget', 'L3', 'c_base_median'), 2.031723094631754),
    'c_peak_L3': ('20260919_2/reports/phase0_amplitude_budget.json',
                  ('amplitude_budget', 'L3', 'c_peak_median'), 2.094720376603729),
    'beta_P': ('20260919_2/reports/phase1_scores.json',
               ('point_estimates', 'beta', 'limit_delta_0.001'), 0.0023311817983205185),
    'beta_obs': ('20260919_2/reports/phase1_scores.json',
                 ('frozen_anchors', 'beta_obs'), 0.004063600875414098),
    'alpha_P': ('20260919_2/reports/phase1_scores.json',
                ('point_estimates', 'alpha', 'limit_delta_0.001'), -0.05034345372182859),
    'alpha_obs': ('20260919_2/reports/phase1_scores.json',
                  ('frozen_anchors', 'alpha_obs'), -0.0704045722214265),
    'nse': ('20260919_2/reports/phase1_scores.json',
            ('null_point', 'monthly', 'nse'), 0.7029748157444711),
    'median_station_nse': ('20260919_2/reports/phase1_scores.json',
                           ('null_point', 'monthly', 'median_station_nse'),
                           -0.1407034380572627),
    'n_events': ('20260919_2/reports/phase0_amplitude_budget.json',
                 ('amplitude_budget', 'L1', 'n_events'), 214),
}
ANCHOR_PATHS_4 = {
    'mean_concentration': ('20260919_4/reports/phase2_full.json',
                           ('baseline', 'mean_concentration'), 2.8234476727379607),
    'D_alpha_base': ('20260919_4/reports/phase2_full.json',
                     ('baseline', 'D_alpha_base'), 0.020061118499597907),
    'D_beta_base': ('20260919_4/reports/phase2_full.json',
                    ('baseline', 'D_beta_base'), 0.0017324190770935791),
    'sd_gate_threshold': ('20260919_4/reports/phase2_full.json',
                          ('baseline', 'sd_gate_threshold'), 0.5559042757541334),
    'sd_L3_ddof0_median_e': ('20260919_4/reports/phase2_full.json',
                             ('baseline', 'sd_L3_ddof0_median_e'), 0.7941489653630477),
    'sd_L3_ddof1_median_e': ('20260919_4/reports/phase2_full.json',
                             ('baseline', 'sd_L3_ddof1_median_e'), 0.7941489653630476),
    'mean_rel_at_plus_half': ('20260919_4/reports/phase2_full.json',
                              ('points', '0.5', 'mean_concentration_relative_change'),
                              0.22381499501517088),
    'alpha_hat_at_plus_half': ('20260919_4/reports/phase1_l1.json',
                               ('points', '0.5', 'alpha_hat'), -0.09534394496011094),
    'beta_hat_at_plus_half': ('20260919_4/reports/phase1_l1.json',
                              ('points', '0.5', 'beta_hat'), 0.00440187504803832),
    'A_L1_at_plus_half': ('20260919_4/reports/phase1_l1.json',
                          ('points', '0.5', 'A_L1'), 1.2289037922001007),
    'A_L3_at_plus_half': ('20260919_4/reports/phase1_l1.json',
                          ('points', '0.5', 'A_L3'), 1.1553373082531406),
    'nse_at_plus_half': ('20260919_4/reports/phase2_full.json',
                         ('points', '0.5', 'nse'), 0.1904071085427308),
    'mnse_at_plus_half': ('20260919_4/reports/phase2_full.json',
                          ('points', '0.5', 'median_station_nse'), -0.8009168003367888),
    'S_u': ('20260919_4/reports/frozen_anchors.json',
            ('geometry', 'S_u'), 14.438101895057091),
}
SD_BASE_DDOF0 = {'L1': 1.3155445644228716, 'L2': 0.5534371191913016,
                 'L3': 0.7941489653630477}
SD_GATE_70PCT = {'L1': 0.9208811950960101, 'L2': 0.3874059834339111,
                 'L3': 0.5559042757541334}
PEER_ROOT = PEER.parent

# The plan (S2.4) said the anchor set would drop every beta / `k_r` anchor.  It does NOT,
# and the deviation is registered in `实际方法与偏离.md`: `beta_P`/`beta_obs`/`alpha_P`/
# `alpha_obs` are ROUND TWO's frozen point estimates, not round 5's intervention grid, and
# B0 must reproduce them.  Deleting them would weaken the anchor gate on the one arm whose
# whole job is to prove the frozen kernel still runs.  `mean_rel_at_plus_half` and the
# other `_at_plus_half` entries are round 4's, kept as read-only historical anchors.

# --------------------------------------------------------------------------
# json / hashing helpers
# --------------------------------------------------------------------------
def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(4 << 20), b''):
            h.update(b)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=False,
                               default=str), encoding='utf-8')
    return sha(path)


def _dig(obj, keys):
    for k in keys:
        if not isinstance(obj, dict) or k not in obj:
            raise SystemExit('ANCHOR_KEY_MISSING %r' % (keys,))
        obj = obj[k]
    return obj


# --------------------------------------------------------------------------
# the frozen model
# --------------------------------------------------------------------------
def model_record(tag):
    return read_json(PEER / 'outputs' / tag / 'model.json')


def parameters(tag):
    """The frozen 30-vector.  This round NEVER alters one element of it."""
    x = np.asarray(model_record(tag)['parameters'], float)
    assert len(x) == 30, (tag, len(x))
    return x


def design_for(tag):
    design = json.loads(json.dumps(model_record(tag)['design']))
    before = dict(design)
    design['observation_registry_file'] = 'data/prediction_registry.json'
    design['observation_registry_hash'] = sha(PEER / 'data/prediction_registry.json')
    changed = {k for k in set(before) | set(design) if before.get(k) != design.get(k)}
    assert changed <= {'observation_registry_file', 'observation_registry_hash'}, changed
    assert design['structure'] == {'human': False, 'calendar': 'monthfirst'}, design['structure']
    return design


def build(tag=TAG):
    design = design_for(tag)
    data = cm.load_data('FULL24')
    model = cm.make_model(data, None, 'D29_BE', design)
    assert model.calendar == 'monthfirst', model.calendar
    assert model.data.operator_id != 'OS_MIX', model.data.operator_id
    assert model.cap is False, 'the reference `scan` mode flag must be False'
    assert type(model).__name__ == 'StructureEndpoints', type(model).__name__
    return model


def pilot_indices(model):
    """READ from the model -- never hard-coded.  0-BASED COLUMN indices."""
    rr = list(model.data.pilot_indices)
    assert len(rr) > 0, 'PILOT_INDICES_EMPTY (operator_id must not be O0)'
    assert all(0 <= int(i) < int(model.data.fast_water.shape[1]) for i in rr), rr
    return [int(i) for i in rr]


def window_mask(dates, window):
    yrs = np.asarray(dates).astype('datetime64[Y]').astype(np.int64) + 1970
    return (yrs >= int(window[0])) & (yrs <= int(window[1]))


# --------------------------------------------------------------------------
# `V_s`: the producer's lower store, which is NOT in the 26-array cache
# --------------------------------------------------------------------------
_LOWER = {}


def load_lower_storage(model=None, tag=TAG):
    """`lower_slow_storage_mm`, `(nd, nr)`, aligned to the model's own day/reach grid.

    ALIGNMENT IS PROVEN, NOT ASSUMED.  The parquet is the producer's own `(date,
    reach_id)` long table; it is sorted and reshaped, and the day axis is required to be
    array-equal to `model.data.dates`.  The reach axis is checked against the producer's
    own `local_slow_response_m3_s` column, which IS bitwise the cache's `slow_water`
    (probe 1: `slow_water == local_slow_s * 86400` with `maxabs = 0.0`).  If either axis
    disagreed, the `V_s` array would be a permutation of the right numbers and every
    downstream concentration would be silently wrong.

    The alignment check is deliberately made in the model's units rather than in mm: a
    mm-based comparison would need the same `area_ha` twice and could agree by accident.
    """
    key = (str(tag), int(len(model.data.dates)))
    if key in _LOWER:
        return _LOWER[key]
    nd = int(model.data.dates.shape[0])
    nr = int(model.data.fast_water.shape[1])
    w = pd.read_parquet(HYDRO_PARQUET,
                        columns=['date', 'reach_id', 'lower_slow_storage_mm',
                                 'local_slow_response_m3_s'])
    w = w.sort_values(['date', 'reach_id'])
    if len(w) % nr:
        raise SystemExit('HYDRO_ROW_COUNT_NOT_A_MULTIPLE_OF_NR %d %d' % (len(w), nr))
    if len(w) // nr < nd:
        raise SystemExit('HYDRO_SHORTER_THAN_THE_MODEL %d %d' % (len(w) // nr, nd))
    d = w.date.to_numpy().reshape(-1, nr)[:, 0].astype('datetime64[D]')
    if not np.array_equal(d[:nd], np.asarray(model.data.dates).astype('datetime64[D]')):
        raise SystemExit('HYDRO_CALENDAR_MISALIGNED')
    slow_s = w.local_slow_response_m3_s.to_numpy(np.float64).reshape(-1, nr)[:nd]
    if not np.array_equal(slow_s * 86400.0, np.asarray(model.data.slow_water, np.float64)[:nd]):
        raise SystemExit('HYDRO_REACH_AXIS_MISALIGNED (local_slow*86400 != slow_water)')
    S = w.lower_slow_storage_mm.to_numpy(np.float64).reshape(-1, nr)[:nd]
    if not (np.all(np.isfinite(S)) and np.all(S > 0.0)):
        raise SystemExit('LOWER_STORAGE_NOT_STRICTLY_POSITIVE min=%r' % float(np.min(S)))
    _LOWER[key] = np.ascontiguousarray(S)
    return _LOWER[key]


# --------------------------------------------------------------------------
# the fraction triple, built once and installed on the model
# --------------------------------------------------------------------------
def dp_arrays(model, form=None, lower=None, tag=TAG):
    """Build `V_u`, `V_s`, `g_u`, `phi_f`, `g_s`, `guard` and attach them to `model`.

    `form` MUST be supplied by the caller from the Phase-0 ruling; it is not defaulted
    here, because a silent default is exactly the "which closure ran?" ambiguity that
    assertion N1 and plan S1.3 exist to remove.  Phase 0 measured `x <= 1` globally at
    the pre-outflow volume, so the ruling is `'x'`.

    `lower` may be injected for the `S-soil` / `V-unsat` sensitivity arms, which vary
    `V_u` and NOT `V_s`.
    """
    if form is None:
        raise ValueError('CLOSURE_FORM_NOT_GIVEN')
    d = model.data
    area_ha = np.asarray(d.area_ha, np.float64)
    F = XI.flows(np.asarray(d.fast_water, np.float64),
                 np.asarray(d.percolation, np.float64),
                 np.asarray(d.slow_water, np.float64), area_ha)
    S_low = load_lower_storage(model, tag) if lower is None else np.asarray(lower, np.float64)
    V = XI.volumes(np.asarray(d.upper_water, np.float64), S_low, F)
    guard = XI.guard_from(np.asarray(d.fast_fraction, np.float64))
    frac = XI.fractions(F, V, form, guard)
    pack = dict(Qf=F['Qf'], Qp=F['Qp'], Qs=F['Qs'], Qu=F['Qu'],
                Vu=V['Vu'], Vs=V['Vs'], guard=guard, form=form, **frac)
    model.dp_fractions = pack
    return pack


def dp_alternate_Vu(model, which):
    """Swap `V_u` only, for the sensitivity and diagnostic arms.  `V_s` never moves.

    `upper` and `soil` are both states the producer DECLARES; `unsat` is their sum and is
    a COMPOSITE reading that no producer declares, which is why plan S2.3-0 keeps it out
    of the primary-arm candidates and uses it as a sensitivity arm only.
    """
    Vu = {'upper': np.asarray(model.data.upper_water, np.float64),
          'soil': np.asarray(model.data.soil_water_mm, np.float64),
          'unsat': (np.asarray(model.data.soil_water_mm, np.float64)
                    + np.asarray(model.data.upper_water, np.float64))}[which]
    pack = model.dp_fractions
    V = XI.volumes(Vu, pack['Vs'] - pack['Qs'], pack)
    frac = XI.fractions(pack, V, pack['form'], pack['guard'])
    out = dict(pack)
    out.update(Vu=V['Vu'], **frac)
    model.dp_fractions = out
    return out


# --------------------------------------------------------------------------
# install / restore
# --------------------------------------------------------------------------
def derive_binding_modules(attr, target, known):
    """Every ALREADY-LOADED peer module whose globals bind `attr` to `target`.

    The `str(PEER) not in str(f)` filter means only modules physically under
    `20260916_2` are candidates; this round's own modules are excluded by construction,
    which is what keeps the `known` name check meaningful.
    """
    found = {}
    for name, m in list(sys.modules.items()):
        if m is None:
            continue
        f = getattr(m, '__file__', None)
        if not f or str(PEER) not in str(f):
            continue
        if getattr(m, attr, None) is target:
            found[name] = m
    keys = set(found)
    missing = set(known) - keys
    if missing:
        raise SystemExit('%s_MODULE_NOT_LOADED %s' % (attr.upper(), sorted(missing)))
    return found, sorted(keys - set(known))


_FROZEN = {'Transport': _closed.Transport, 'scan': _closed.scan,
           'tag_scan': cm.tag_scan}
_KNOWN = {'Transport': ('closures', 'scientific_models', 'temporal_model', 'hf_model',
                        'structure_model', 'campaign_model'),
          'scan': ('closures',),
          'tag_scan': ('campaign_model', 'tagged_transport')}
_BIND = {'Transport': 'TransportDP', 'scan': 'scan_dp_ledger', 'tag_scan': 'tag_scan_dp'}
_FROZEN_LEDGER = _closed.ResearchObjective.ledger


def binding_census():
    """The four rebinds, censused.  `ResearchObjective.ledger` is censused DIFFERENTLY
    from the other three and the difference is not cosmetic: `ledger` is a CLASS
    attribute, not a module global, so `derive_binding_modules`'s
    `getattr(module, attr) is target` never sees it -- the `closures` module has no global
    named `ledger` at all.  Running it through the same machinery raises
    `LEDGER_MODULE_NOT_LOADED` for a module that is in fact loaded, which is how this was
    caught.  The class attribute is therefore checked directly, and its `frontier` is the
    set of subclasses that could carry their own override (none does; asserted).
    """
    out = {}
    for attr in ('Transport', 'scan', 'tag_scan'):
        found, extra = derive_binding_modules(attr, _FROZEN[attr], _KNOWN[attr])
        out[attr] = dict(modules=sorted(found), known=list(_KNOWN[attr]), extra=extra,
                         n_modules=len(found))
    overrides = [c.__name__ for c in _closed.ResearchObjective.__subclasses__()
                 if 'ledger' in getattr(c, '__dict__', {})]
    if overrides:
        raise SystemExit('LEDGER_SUBCLASS_OVERRIDE %s' % overrides)
    out['ResearchObjective.ledger'] = dict(
        modules=['closures.ResearchObjective'], known=['closures.ResearchObjective'],
        extra=[], n_modules=1, kind='class_attribute', subclass_overrides=[])
    return out


_INSTALLED_SETS = {}


def install_kernel(model):
    """Bind the new kernel into this process: THREE module globals plus ONE class attr.

    `closures_mc.py:53-59`'s lesson applies to the fraction arrays exactly as it did to
    round 5's `Xi`: numba treats module globals as compile-time constants, so the arrays
    travel as call arguments and `CONFIG` is read only in the Python wrappers.
    """
    rr = pilot_indices(model)
    d = model.dp_fractions
    MC.set_config(d['gu'], d['phi_f'], d['gs'],
                  np.ascontiguousarray(d['gu'][:, rr]),
                  np.ascontiguousarray(d['phi_f'][:, rr]),
                  np.ascontiguousarray(d['gs'][:, rr]))
    MC.set_tag_slice(rr)
    cen = binding_census()
    _INSTALLED_SETS.clear()
    _INSTALLED_SETS.update({k: list(v['modules']) for k, v in cen.items()})
    for attr in ('Transport', 'scan', 'tag_scan'):
        impl = getattr(MC, _BIND[attr])
        for n in _INSTALLED_SETS[attr]:
            setattr(sys.modules[n], attr, impl)
    # the class attribute, bound directly -- see `binding_census`
    setattr(_closed.ResearchObjective, 'ledger', MC.ledger_dp)
    MC.assert_no_autograd()
    return dict(census=cen, installed_sets=dict(_INSTALLED_SETS), pilot_indices=rr,
                closure_form=d['form'], **is_installed())


def restore_kernel():
    """Restore every binding AND clear `_INSTALLED_SETS`.

    Round 5's version cleared nothing, so a second `install_kernel` in the same process
    restored the SECOND set's names while the first set stayed bound wherever the two
    differed.  With five sequential forwards in one process that is a live hazard, so the
    clear is added -- and it is the only behavioural change to this function.
    """
    for attr, names in _INSTALLED_SETS.items():
        if attr == 'ResearchObjective.ledger':
            # `names` here is ['closures.ResearchObjective'], a LABEL not a module key;
            # the target is the class attribute, so it is set directly.
            setattr(_closed.ResearchObjective, 'ledger', _FROZEN_LEDGER)
            continue
        for n in names:
            setattr(sys.modules[n], attr, _FROZEN[attr])
    _INSTALLED_SETS.clear()
    MC.set_config(None, None, None)
    MC.set_tag_slice([])
    return is_installed()


_ORIG_HAZARD = _hcm.Predictor.hazard


def hazard_is_frozen():
    return bool(_hcm.Predictor.hazard is _ORIG_HAZARD)


def is_installed():
    per = {}
    for attr, names in (_INSTALLED_SETS or {}).items():
        if attr == 'ResearchObjective.ledger':
            per[attr] = dict(modules=['closures'], n_modules=1,
                             live=['closures'] if _ledger_is_dp() else [],
                             all_live=bool(_ledger_is_dp()))
            continue
        impl = getattr(MC, _BIND[attr])
        live = [n for n in names if getattr(sys.modules.get(n), attr, None) is impl]
        per[attr] = dict(modules=names, n_modules=len(names), live=live,
                         all_live=bool(names) and len(live) == len(names))
    return dict(per_name=per, hazard_is_frozen=hazard_is_frozen(),
                ledger_is_frozen=bool(_ledger_is_dp() is False),
                n_transport_modules=len((_INSTALLED_SETS or {}).get('Transport', [])),
                all_bound=bool(per) and all(v['all_live'] for v in per.values())
                and hazard_is_frozen())


def _ledger_is_dp():
    return bool(_closed.ResearchObjective.ledger is MC.ledger_dp)


def _ledger_is_really_the_frozen_one():
    return bool(_closed.ResearchObjective.ledger is _FROZEN_LEDGER)


# --------------------------------------------------------------------------
# the daily replay and the anchor gate
# --------------------------------------------------------------------------
def replay(model, tag, arrays=None):
    if arrays is not None:
        for k, v in arrays.items():
            assert np.asarray(v).flags['C_CONTIGUOUS'], k
            setattr(model, k, v)
    x = parameters(tag)
    bounds = model.bounds
    if not all(a <= v <= bb for v, (a, bb) in zip(x, bounds)):
        raise SystemExit('PARAMETER_OUTSIDE_BOUNDS ' + tag)
    meta = pd.read_parquet(PEER / 'data/prediction_calendar.parquet')
    meta = meta[meta.year.le(END_YEAR)].copy()
    with torch.no_grad():
        d = model.daily_boundary(torch.tensor(x), meta)
    ri = d['record'].numpy(); di = d['day_index'].numpy()
    F = d['mass'].numpy(); V = d['water'].numpy()
    t = pd.DataFrame(dict(station_key=meta.station_key.to_numpy()[ri],
                          date=model.data.dates[di],
                          p=1000 * F / V, water_m3_day=V))
    assert not any('mass' in c or 'load' in c or 'kg' in c for c in t.columns), t.columns
    return t


def anchor_gate(replay_frame):
    """The stored C0_s1 daily table must be reproduced exactly."""
    stored = pd.read_parquet(ANCHOR_TABLE)[['station_key', 'date', 'concentration_mg_l']]
    stored = stored.rename(columns={'concentration_mg_l': 'p_stored'})
    mine = replay_frame.rename(columns={'p': 'p_replay'})
    cmp = stored.merge(mine, on=['station_key', 'date'], validate='one_to_one')
    if len(cmp) != len(stored) or len(stored) != ANCHOR_ROWS:
        raise SystemExit('ANCHOR_ROW_COUNT_CHANGED %d %d %d'
                         % (len(cmp), len(stored), ANCHOR_ROWS))
    d = float(np.max(np.abs(cmp.p_replay.to_numpy() - cmp.p_stored.to_numpy())))
    return dict(n_compared_rows=int(len(cmp)), max_abs_elementwise_concentration=d,
                tolerance=ANCHOR_TOL, expected_rows=ANCHOR_ROWS,
                passed=bool(d <= ANCHOR_TOL),
                source='20260916_2\\outputs\\C0_s1\\daily_station_mass_water.parquet')


def load_anchors():
    """Read every anchor from its producer and assert it equals the registered value."""
    out = {}
    for table, root, origin in ((ANCHOR_PATHS, PEER_ROOT, 'producer'),
                                (ANCHOR_PATHS_4, R4.parent, 'round4')):
        for name, (rel, keys, expect) in table.items():
            p = root / rel
            if not p.is_file():
                raise SystemExit('ANCHOR_FILE_MISSING %s' % p)
            got = _dig(json.loads(p.read_text(encoding='utf-8')), keys)
            ok = (float(got) == float(expect)) if isinstance(expect, float) \
                else (int(got) == int(expect))
            if not ok:
                raise SystemExit('ANCHOR_DRIFTED %s: json=%r registered=%r'
                                 % (name, got, expect))
            out[name] = dict(value=got, source=rel, keys=list(keys), sha256=sha(p),
                             origin=origin, matches_registered=True)
    pf = read_json(R4 / 'reports/phase2_full.json')
    sda = pf['baseline']['sd_all_layers']
    for lay in ('L1', 'L2', 'L3'):
        for d in (0, 1):
            got = float(sda[lay]['ddof%d' % d])
            if d == 0 and got != SD_BASE_DDOF0[lay]:
                raise SystemExit('ANCHOR_DRIFTED sd_%s_ddof0 %r' % (lay, got))
            out['sd_%s_ddof%d' % (lay, d)] = dict(
                value=got, source='20260919_4/reports/phase2_full.json',
                keys=['baseline', 'sd_all_layers', lay, 'ddof%d' % d],
                sha256=sha(R4 / 'reports/phase2_full.json'), origin='round4',
                matches_registered=True)
    gate_layer = pf['baseline']['sd_gate_layer']
    if gate_layer != 'L3':
        raise SystemExit('SD_GATE_LAYER_MOVED %r' % gate_layer)
    out['sd_gate_layer'] = dict(value=gate_layer, source='20260919_4/reports/phase2_full.json',
                                keys=['baseline', 'sd_gate_layer'],
                                sha256=sha(R4 / 'reports/phase2_full.json'),
                                origin='round4', matches_registered=True)
    if not bool(pf['baseline']['sd_ddof_choice_is_inert']):
        raise SystemExit('SD_DDOF_CHOICE_NO_LONGER_INERT')
    return out


# --------------------------------------------------------------------------
# the scores: eligible grid, observed monthly panel, monthly statistics
# --------------------------------------------------------------------------
PANEL_SHA = '7ed9e6179705affc00494fafb2119ea2c656b32e661c44fee7dd5fd162d56a18'
MONTHLY_GATE = 0.005        # phase1_score.py:42, carried over unchanged
SD_GATE_FRACTION = 0.70     # G3: "median_s e_s <= 0.70 x baseline"


def eligible_grid():
    if sha(EL.MASK) != EL.MASK_SHA:
        raise SystemExit('ELIGIBLE_MASK_DRIFTED %s' % sha(EL.MASK))
    m = pd.read_parquet(EL.MASK)
    out = m[m.eligible][['station_key', 'date']].copy()
    out['date'] = EL.as_day(out.date)
    return out


def obs_monthly():
    """Observed monthly station-mean TN, from the READ-ONLY 4h panel.

    This is a LEVEL gate on a concentration -- not an event selection and not a fit
    target -- which is what the panel's registered status permits: a zero-fit round may
    SCORE against it, never calibrate on it.
    """
    if sha(EL.PANEL) != PANEL_SHA:
        raise SystemExit('THE_4h_PANEL_CHANGED %s != %s' % (sha(EL.PANEL), PANEL_SHA))
    p = pd.read_parquet(EL.PANEL)
    p = p.assign(date=p.monitoring_time.dt.tz_localize(None).dt.normalize(),
                 y=p.monitoring_time.dt.year, mo=p.monitoring_time.dt.month)
    d = p.groupby(['station_key', 'date'], as_index=False).TN.mean()
    d = d[(d.date.dt.year >= EL.START_YEAR) & (d.date.dt.year <= EL.END_YEAR)]
    d = d.assign(y=d.date.dt.year, mo=d.date.dt.month)
    return d.groupby(['station_key', 'y', 'mo']).TN.mean().rename('obs')


def monthly_join(ly, elig, obs_m, scale=1.0):
    m = elig.merge(ly[['station_key', 'date', 'pL3']].rename(columns={'pL3': 'p'}),
                   on=['station_key', 'date'], how='left', validate='one_to_one')
    assert int(m.p.isna().sum()) == 0, 'A CANDIDATE DROPPED AN ELIGIBLE DAY'
    if float(scale) != 1.0:
        m = m.assign(p=m.p.to_numpy(float) * float(scale))
    m = m.assign(y=m.date.dt.year, mo=m.date.dt.month)
    pm = m.groupby(['station_key', 'y', 'mo']).p.mean().rename('pred')
    z = obs_m.to_frame().join(pm, how='inner').dropna()
    return m, z


def monthly_stats_from_z(m, z):
    o = z.obs.to_numpy(float)
    pv = z.pred.to_numpy(float)
    e = pv - o
    per = {}
    for kk, g in z.groupby(level=0):
        ee = g.pred.to_numpy(float) - g.obs.to_numpy(float)
        per[kk] = 1.0 - float(np.mean(ee ** 2)) / float(np.var(g.obs.to_numpy(float)))
    return dict(n_station_months=int(len(z)),
                nse=float(1.0 - np.mean(e ** 2) / np.var(o)),
                r2=float(np.corrcoef(o, pv)[0, 1] ** 2),
                median_station_nse=float(np.median(list(per.values()))),
                per_station_nse={str(kk): float(v) for kk, v in per.items()},
                mean_concentration=float(m.p.to_numpy(float).mean()),
                n_eligible_rows=int(len(m)))


def monthly_stats(ly, elig, obs_m, scale=1.0):
    m, z = monthly_join(ly, elig, obs_m, scale)
    return monthly_stats_from_z(m, z)


def nse_of(o, pv):
    o = np.asarray(o, float); pv = np.asarray(pv, float)
    return float(1.0 - np.mean((pv - o) ** 2) / np.var(o))


# --------------------------------------------------------------------------
# the frozen hash map
# --------------------------------------------------------------------------
def frozen_hash_report():
    watched = ('closures.py', 'tagged_transport.py', 'structure_model.py', 'hf_model.py',
               'temporal_model.py', 'campaign_model.py', 'sc_kernel.py', 'routing.py',
               'model.py', 'scientific_models.py')
    out = {}
    for fold in ('C0', 'C2'):
        p = PEER / 'reports/launch_by_fold' / (fold + '.json')
        if not p.is_file():
            out[fold] = dict(status='ABSENT')
            continue
        fh = read_json(p)['frozen_hashes']
        out[fold] = dict(n_frozen=len(fh), watched={})
        for k, v in fh.items():
            base = k.replace('\\', '/').rsplit('/', 1)[-1]
            if base not in watched:
                continue
            q = PEER / k
            now = sha(q) if q.is_file() else None
            out[fold]['watched'][base] = dict(registered=v, on_disk=now,
                                              unchanged=bool(now == v))
    return out
