"""Round 20260919_4 bootstrap: the frozen reference tree, the mobile-water geometry
`Xi`, and the runtime install of the pre-mobilisation pathway-concentration kernel.

NOTHING IS WRITTEN INTO 20260916_2, AND NOTHING COVERED BY `frozen_hashes` IS EDITED
------------------------------------------------------------------------------------
`20260916_2/reports/launch_by_fold/{C0,C2}.json` each carry a 75-entry
`frozen_hashes` map.  The files leaned on here are verified in Phase 0:

    vendor\\research\\closures.py          02433a200de4911f...   the frozen `scan`
    vendor\\research\\tagged_transport.py  b1a7df8a559e00a4...   the frozen `tag_scan`
    vendor\\expert\\tn_challenge\\model.py  ...                    `Predictor.hazard`
    scripts\\structure_model.py            615e87b9fd2ec067...
    scripts\\hf_model.py                   88a85da13541e612...
    scripts\\temporal_model.py             1b3217cbec5260f0...
    scripts\\campaign_model.py             44bdb7c00b9ea572...

THIS ROUND DOES NOT REBIND `Predictor.hazard` -- the explicit opposite of
`20260919_3`, which rebind it and nothing else.  Here the change is one factor inside
the HAZARD-to-`prob` conversion, so it must live in the kernel that performs it.

WHAT THE MODEL ACTUALLY IS (read off the frozen artifacts, not assumed)
-----------------------------------------------------------------------
`outputs/C0_s1/model.json::design` carries `observation_operator='MATCH'`,
`structure={'human':False,'calendar':'monthfirst'}` and `operator_id='OU'`.  Both keys
`make_model` branches on are present, so `kind='D29_BE'` resolves to

    structure_model.StructureEndpoints -> hf_model.HFEndpoints
                                       -> temporal_model.TemporalEndpoints
                                       -> campaign_model.Endpoints -> Matched

`cap` is False, so the frozen `scan` takes `mode=False` and `risk = h[t,r]` with no
`av + k`; `state_extension` is False and `kind` is not `SC*`, so `sc_kernel` is off the
path.  `operator_id='OU'` (not `'O0'`) means `data.support` is loaded and
`data.pilot_indices` is non-empty, so `campaign_model.py:102`'s `if rr:` guard passes and
the `tag_scan` branch is LIVE.  Phase 0 records the measured values, not this paragraph.

FOUR REBINDINGS, NOT THREE
--------------------------
`20260919_2/work/common20.py:163::install_kernel` rebinds three things: the six modules'
`Transport`, `closures.scan`, and `cm.tag_scan`.  Red-team finding 6b: that set is
INCOMPLETE, because `tagged_transport.py:53`'s `TaggedTransport.forward` resolves the
BARE name `tag_scan` through ITS OWN module globals.  Leaving it bound to the frozen
function would let the `OS_MIX` forward path modulate only its scalar channel.  That
path is CURRENTLY DEAD -- it is gated on `operator_id == 'OS_MIX'` and this round's
reference is `'OU'` -- so the omission is invisible here and would be silent in a later
round that switched operators.  This round rebinds it anyway and asserts the operator.

`TRANSPORT_MODULES` IS RE-DERIVED, NOT COPIED
---------------------------------------------
Round 2's tuple is a hard-coded list of six.  A module that did `from closures import
Transport` at import time holds a SNAPSHOT of the class, so rebinding the origin module
does not reach it -- and a copied list cannot discover a seventh.  `derive_transport_
modules()` scans what is ACTUALLY LOADED from the peer tree and returns every module
whose globals bind the frozen `Transport` class, then asserts it covers the known six.
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
assert ROUND.name == '20260919_4', ROUND
assert PEER.is_dir(), PEER

# campaign_model.py sets NUMBA_CACHE_DIR to PEER/work/numba at import; pin it here
# first so no compiled entry can be deposited inside the read-only reference round.
CACHE = ROUND / 'work' / 'numba'
CACHE.mkdir(parents=True, exist_ok=True)
os.environ['NUMBA_CACHE_DIR'] = str(CACHE)
import numba.core.config as _nbcfg  # noqa: E402  (reads the env var above)

sys.path[:0] = [str(ROUND / 'work'),
                str(PEER / 'scripts'),
                str(PEER / 'vendor/expert/tn_challenge'),
                str(PEER / 'vendor/research'),
                str(PEER / 'vendor/transfer_research')]
# `eventlib.py` is the event/criterion apparatus and lives in the PEER round, which is
# read-only.  It is APPENDED, not prepended: this round's own modules must still win any
# name lookup, and a module the sandbox cannot shadow is one whose hash check means
# something.
PEER_WORK = PEER.parent / '20260919_2' / 'work'
EVENTLIB_SHA = 'b680b51faedf6e24e38babe9061e468e3588ffc0eb080e316f3111bdef034f1a'
if str(PEER_WORK) not in sys.path:
    sys.path.append(str(PEER_WORK))
import campaign_model as cm  # noqa: E402
import native_runtime as _rt  # noqa: E402

assert cm.RUN == PEER, (cm.RUN, PEER)
os.environ['NUMBA_CACHE_DIR'] = str(CACHE)
_nbcfg.CACHE_DIR = str(CACHE)

import torch  # noqa: E402
import closures as _closed  # noqa: E402
import tagged_transport as _tagged  # noqa: E402
import scientific_models as _sci  # noqa: E402
import temporal_model as _temporal  # noqa: E402
import hf_model as _hf  # noqa: E402
import structure_model as _structure  # noqa: E402
import model as _hcm  # noqa: E402  (`vendor/expert/tn_challenge/model.py`)
import xi as XI  # noqa: E402
import closures_mc as MC  # noqa: E402
import eventlib as EL  # noqa: E402

assert hashlib.sha256((PEER_WORK / 'eventlib.py').read_bytes()).hexdigest() \
    == EVENTLIB_SHA, 'EVENTLIB_CHANGED'
assert Path(EL.__file__).resolve() == (PEER_WORK / 'eventlib.py').resolve(), EL.__file__

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

TAG = 'C0_s1'
ANCHOR_TABLE = PEER / 'outputs/C0_s1/daily_station_mass_water.parquet'
ANCHOR_ROWS = 169476
ANCHOR_TOL = 1e-12
END_YEAR = 2024
REF_YEARS = (1961, 2020)          # strictly pre-evaluation (EVALUATION is 2021-2024)

# --------------------------------------------------------------------------
# THE BETA GRID, SPLIT INTO THREE GROUPS, REGISTERED BEFORE ANY AMPLITUDE
# --------------------------------------------------------------------------
# Round 3's registered lesson (§C1): a grid that lies entirely inside ONE regime cannot
# distinguish "the mechanism fails" from "the grid never entered the regime".  Round 3
# normalised `gamma` into units of `z`'s own 39-log-unit spread, so every tested point
# sat BELOW proportional enrichment and the hyper-proportional region was never directly
# tested.  Here `beta` is the raw log-log elasticity `d log(C_fast/C_slow) / d log W`, so
# the regime is read directly off the number:
#
#     |beta| = 0        constant contrast == the frozen model (exactly, bitwise)
#     |beta| < 1        sub-proportional
#     |beta| = 1        proportional: the contrast tracks the mobile water
#     |beta| > 1        HYPER-proportional enrichment / dilution
#
# Only the first group carries the verdict.  `BETA_EXTREME` is reported and explicitly
# may NOT be used to claim `QUALIFY`, because red-team measurement shows the response is
# saturation-flattened there (`frac(prob >= 0.99)` reaches 0.73 at beta = 4 without the
# floor; `xi.saturation_census` re-measures it WITH the floor).
BETA_MAIN = (0.0, -0.05, 0.05, -0.1, 0.1, -0.25, 0.25, -0.5, 0.5, -1.0, 1.0)
BETA_HYPER = (-1.5, 1.5, -2.0, 2.0)
BETA_EXTREME = (-4.0, 4.0, -8.0, 8.0)
BETA_GRID = BETA_MAIN + BETA_HYPER + BETA_EXTREME
BETA_GROUPS = {'main': list(BETA_MAIN), 'hyper': list(BETA_HYPER),
               'extreme': list(BETA_EXTREME)}
assert len(BETA_GRID) == 19 and len(set(BETA_GRID)) == 19, BETA_GRID

# the PRE-REGISTERED mechanical availability rule (plan 1.3-3).  It is a rule applied to
# a reported number, chosen before any amplitude existed, and must not be relaxed.
SAT_BOUND = 0.10       # frac(prob >= 0.99) must be <= this
DEGEN_FLOOR = 1e-3     # frac(|Xi - 1| > 1e-6) must be >= this
W_FLOOR = XI.W_FLOOR

# --------------------------------------------------------------------------
# FROZEN ANCHORS -- READ from their producers, never retyped (delivery check ①)
# --------------------------------------------------------------------------
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
PEER_ROOT = PEER.parent


def _dig(obj, keys):
    for k in keys:
        if not isinstance(obj, dict) or k not in obj:
            raise SystemExit('ANCHOR_KEY_MISSING %r' % (keys,))
        obj = obj[k]
    return obj


def load_anchors():
    """Read every anchor from its producer and assert it equals the registered value.

    The delivery discipline forbids retyped literals in reports, so the anchors are
    LOADED and the load is verified against the registered number.  A mismatch is a
    hard stop: an anchor that has drifted makes every gain measured against it
    meaningless.
    """
    out = {}
    for name, (rel, keys, expect) in ANCHOR_PATHS.items():
        p = PEER_ROOT / rel
        if not p.is_file():
            raise SystemExit('ANCHOR_FILE_MISSING %s' % p)
        got = _dig(json.loads(p.read_text(encoding='utf-8')), keys)
        if isinstance(expect, float):
            ok = (float(got) == expect)
        else:
            ok = (int(got) == int(expect))
        if not ok:
            raise SystemExit('ANCHOR_DRIFTED %s: json=%r registered=%r' % (name, got, expect))
        out[name] = dict(value=got, source=rel, keys=list(keys), sha256=sha(p),
                         matches_registered=True)
    return out


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


# --------------------------------------------------------------------------
# the frozen model, verbatim from common21.py
# --------------------------------------------------------------------------
def model_record(tag):
    return read_json(PEER / 'outputs' / tag / 'model.json')


def parameters(tag):
    """The frozen 30-vector.  This round NEVER alters one element of it."""
    rec = model_record(tag)
    x = np.asarray(rec['parameters'], float)
    assert len(x) == 30, (tag, len(x))
    return x


def design_for(tag):
    """The registered design with only the prediction registry substituted."""
    design = json.loads(json.dumps(model_record(tag)['design']))
    before = dict(design)
    design['observation_registry_file'] = 'data/prediction_registry.json'
    design['observation_registry_hash'] = cm.sha(PEER / 'data/prediction_registry.json')
    changed = {k for k in set(before) | set(design) if before.get(k) != design.get(k)}
    allowed = {'observation_registry_file', 'observation_registry_hash'}
    assert changed <= allowed, changed
    assert design['structure'] == {'human': False, 'calendar': 'monthfirst'}, design['structure']
    return design


def build(tag):
    design = design_for(tag)
    data = cm.load_data('FULL24')
    model = cm.make_model(data, None, 'D29_BE', design)
    assert model.calendar == 'monthfirst', model.calendar
    assert getattr(model, 'human_enabled') is False, model.human_enabled
    assert model.cap is False, 'the reference `scan` mode flag must be False'
    assert type(model).__name__ == 'StructureEndpoints', type(model).__name__
    return model


# --------------------------------------------------------------------------
# the mobile-water geometry, computed ONCE and cached
# --------------------------------------------------------------------------
_GEOM = {}


def geometry(model=None, tag=TAG):
    """`xi.geometry` on the model's own data, memoised for the process."""
    if 'G' not in _GEOM:
        if model is None:
            model = build(tag)
        d = model.data
        G = XI.geometry(d.fast_water, d.percolation, d.area_ha, d.dates,
                        ref_years=REF_YEARS, w_floor=W_FLOOR)
        _GEOM['G'] = G
        _GEOM['model'] = model
    return _GEOM['G']


def pilot_indices(model):
    """READ from the model -- never hard-coded.  These are 0-BASED COLUMN indices.

    Red-team finding 2: `campaign_model.py:104` passes `h[:, rr]`, so `Xi` handed to
    the tagged kernel must be sliced by exactly this.  A FULL `Xi` would be read from
    column 0 and silently modulate the wrong reaches.
    """
    rr = list(model.data.pilot_indices)
    assert len(rr) > 0, 'PILOT_INDICES_EMPTY (operator_id must not be O0)'
    assert all(0 <= int(i) < int(model.data.fast_water.shape[1]) for i in rr), rr
    return [int(i) for i in rr]


def build_xi(model, beta):
    """The one `Xi`, plus its tagged slice -- a single source for all three entries."""
    G = geometry(model)
    X = XI.xi_from(G, beta)
    rr = pilot_indices(model)
    return X, XI.tag_slice(X, rr), G


# --------------------------------------------------------------------------
# install / restore -- FOUR rebindings
# --------------------------------------------------------------------------
def derive_binding_modules(attr, target, known):
    """Every ALREADY-LOADED peer module whose globals bind `attr` to `target`.

    Re-derived from `sys.modules` rather than copied from a list, because a copied list
    cannot discover a module that imported the object BY NAME.  Only certain: the
    consequence of a snapshot import is that rebinding the origin module does not reach
    the snapshotting module, and that consequence is invisible until a criterion
    silently reads the frozen object.

    Measured on this tree, `campaign_model.py:13-14` does
    `from closures import Transport` and `from tagged_transport import TaggedTransport,
    tag_scan`, so `campaign_model` holds SNAPSHOTS of two of the three objects this
    round must replace -- which is exactly why a single `closures.Transport = ...`
    rebind is not enough and why this function is general over the name.

    The known set must be a SUBSET of the derivation, not the other way round, and any
    module beyond it is REPORTED rather than ignored.
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
    want = set(known)
    missing = want - keys
    if missing:
        raise SystemExit('%s_MODULE_NOT_LOADED %s' % (attr.upper(), sorted(missing)))
    return found, sorted(keys - want)


# The three objects this round replaces, captured at IMPORT so "which modules hold a
# snapshot" is answerable at any later moment without the answer depending on whether
# the install has already run.
_FROZEN = {'Transport': _closed.Transport, 'scan': _closed.scan,
           'tag_scan': cm.tag_scan, 'tagged_tag_scan': _tagged.tag_scan}
_KNOWN = {'Transport': ('closures', 'scientific_models', 'temporal_model', 'hf_model',
                        'structure_model', 'campaign_model'),
          'scan': ('closures',),
          'tag_scan': ('campaign_model', 'tagged_transport')}


def derive_transport_modules():
    return derive_binding_modules('Transport', _FROZEN['Transport'], _KNOWN['Transport'])


def binding_census():
    """All three names, re-derived.  `extra` is empty or a finding, never ignored."""
    out = {}
    for attr in ('Transport', 'scan', 'tag_scan'):
        found, extra = derive_binding_modules(attr, _FROZEN[attr], _KNOWN[attr])
        out[attr] = dict(modules=sorted(found), known=list(_KNOWN[attr]), extra=extra,
                         n_modules=len(found))
    out['TaggedTransport'] = dict(
        modules=[n for n, m in sys.modules.items() if m is not None
                 and str(getattr(m, '__file__', '')) .startswith(str(PEER))
                 and getattr(m, 'TaggedTransport', None) is _tagged.TaggedTransport],
        known=['tagged_transport'], extra=[], n_modules=0)
    out['TaggedTransport']['n_modules'] = len(out['TaggedTransport']['modules'])
    return out


def snapshot_import_census():
    """Look for `from X import Transport`-style SNAPSHOTS in the peer source tree.

    A snapshot import copies the FUNCTION OBJECT into the importing module's globals at
    import time, so rebinding the origin module does not reach it.  `derive_transport_
    modules` detects the consequence; this census names the cause.
    """
    hits = []
    pats = ('import Transport', 'import Transport,', ' Transport,', ',Transport')
    for base in (PEER / 'scripts', PEER / 'vendor' / 'research',
                 PEER / 'vendor' / 'expert' / 'tn_challenge'):
        if not base.is_dir():
            continue
        for p in sorted(base.rglob('*.py')):
            try:
                txt = p.read_text(encoding='utf-8', errors='replace')
            except OSError:
                continue
            for ln, line in enumerate(txt.splitlines(), 1):
                s = line.strip()
                if s.startswith('#'):
                    continue
                if 'import' in s and 'Transport' in s and not s.startswith('Transport'):
                    hits.append(dict(file=str(p.relative_to(PEER)), line=ln, text=s))
    return dict(n_hits=len(hits), hits=hits)


_INSTALLED_SETS = {}


def install_kernel(model, Xi, Xi_tag):
    """Bind the modulated kernel and the geometry into this process.

    FOUR bindings, re-derived from the loaded modules rather than listed, because the
    land-phase recursion is reached four ways:

      `Transport.apply`      closures.py:61/:212, structure_model.py:39,
                             temporal_model.py:77, campaign_model.py:97,
                             scientific_models.py:80      -- the forward replay
      `closures.scan`        closures.py:220               -- the ledger's SCALAR branch
      `campaign_model.tag_scan`  campaign_model.py:104      -- the ledger's TAGGED branch
      `tagged_transport.tag_scan` tagged_transport.py:53    -- TaggedTransport's own call

    Missing the second or third makes `source_label_sum_errors` measure the modification
    instead of the tagged/scalar rounding it is registered to measure.  Missing the
    fourth is invisible while `operator_id == 'OU'` and silent when it is not.

    `Transport.apply`'s SIGNATURE IS UNCHANGED (all five call sites pass `self`), and
    the bare name `tag_scan` still takes exactly SIX positional arguments.
    """
    X = np.ascontiguousarray(Xi, dtype=np.float64)
    XT = np.ascontiguousarray(Xi_tag, dtype=np.float64)
    rr = pilot_indices(model)
    if XT.shape != (X.shape[0], len(rr)):
        raise ValueError('XI_TAG_SHAPE %r vs %r' % (XT.shape, (X.shape[0], len(rr))))
    if not np.array_equal(XT, X[:, rr]):
        raise ValueError('XI_TAG_NOT_THE_SLICE_OF_XI')
    MC.set_config(X, XT)
    model.Xi = X
    model.Xi_tag = XT
    cen = binding_census()
    _INSTALLED_SETS.clear()
    _INSTALLED_SETS.update({k: list(v['modules']) for k, v in cen.items()
                            if k in ('Transport', 'scan', 'tag_scan')})
    bind = {'Transport': MC.TransportMC, 'scan': MC.scan_mc_ledger,
            'tag_scan': MC.tag_scan_mc}
    for attr, names in _INSTALLED_SETS.items():
        for n in names:
            setattr(sys.modules[n], attr, bind[attr])
    MC.assert_no_autograd()
    return dict(census=cen, installed_sets=dict(_INSTALLED_SETS),
                Xi_shape=list(X.shape), Xi_tag_shape=list(XT.shape), pilot_indices=rr,
                **is_installed())


def restore_kernel():
    for attr, names in _INSTALLED_SETS.items():
        for n in names:
            setattr(sys.modules[n], attr, _FROZEN[attr])
    MC.set_config(None, None)
    return is_installed()


# The frozen hazard method, captured at import so "not rebound" is checkable.  Round 3
# rebound exactly this and nothing else; this round must do the exact opposite, and an
# identity comparison is the only way to say so rather than assert it.
_ORIG_HAZARD = _hcm.Predictor.hazard


def hazard_is_frozen():
    """`Predictor.hazard` must still be the frozen method object.

    `campaign_model.py:130` calls `Predictor.hazard(self, t[:29])` at CLASS level, so a
    class-attribute rebind is what round 3 used and what this round must NOT do.
    """
    return bool(_hcm.Predictor.hazard is _ORIG_HAZARD)


def is_installed():
    """Are all THREE names live in every module that binds them, and `hazard` untouched?"""
    bind = {'Transport': MC.TransportMC, 'scan': MC.scan_mc_ledger,
            'tag_scan': MC.tag_scan_mc}
    per = {}
    for attr, names in (_INSTALLED_SETS or {}).items():
        per[attr] = dict(modules=names,
                         live=[n for n in names
                               if getattr(sys.modules.get(n), attr, None) is bind[attr]],
                         n_modules=len(names))
        per[attr]['all_live'] = bool(names) and len(per[attr]['live']) == len(names)
    hazard_ok = hazard_is_frozen()
    return dict(per_name=per, hazard_is_frozen=hazard_ok,
                n_transport_modules=len((_INSTALLED_SETS or {}).get('Transport', [])),
                all_bound=bool(per and all(v['all_live'] for v in per.values()) and hazard_ok))


# --------------------------------------------------------------------------
# the daily replay, byte-for-byte the arithmetic of common21.py::replay
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


def frozen_hash_report():
    """The 75-entry frozen map, restricted to the files this round leans on."""
    watched = ('closures.py', 'tagged_transport.py', 'structure_model.py', 'hf_model.py',
               'temporal_model.py', 'campaign_model.py', 'sc_kernel.py', 'routing.py',
               'model.py', 'scientific_models.py')
    out = {}
    for fold in ('C0', 'C2'):
        fh = read_json(PEER / 'reports/launch_by_fold' / (fold + '.json'))['frozen_hashes']
        out[fold] = dict(n_frozen=len(fh), watched={})
        for k, v in fh.items():
            base = k.replace('\\', '/').rsplit('/', 1)[-1]
            if base in watched:
                now = sha(PEER / k)
                out[fold]['watched'][base] = dict(registered=v, on_disk=now, path=k,
                                                  unchanged=bool(now == v))
    return out
