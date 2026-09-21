"""Round 20260919_5 bootstrap: the frozen reference tree, the mobile-water geometry
`Xi_raw`, the per-reach mass-neutral normaliser `k_r`, and the runtime install of the
pre-mobilisation pathway-concentration kernel.

CARRIED FROM `20260919_4/work/common22.py`.  THE DELTA IS EXECUTABLE AND IS LISTED
-------------------------------------------------------------------------------
This file is a COPY of `common22.py` with the following changes, and nothing else:

  1. `ROUND.name` assertion `20260919_4` -> `20260919_5`.
  2. Every pre-existing anchor is READ FROM ITS ORIGINAL PRODUCER (the round that first
     produced it), not from round 4's restatement of it; round 4 is additionally
     CROSS-CHECKED, so a drift shows up as `ANCHOR_DRIFTED` / `agree: false` rather than
     as a silently different baseline.  Round-5 additions are read from round 4, which is
     where they were produced.
  3. `import xi as XI` -> `import xi_k as XI`: this round has TWO Xi sources and the one
     that reaches the kernel is `xi_k.xi_star` (see that module's docstring).
  4. New: `bootstrap_arrays`, `device_state`, `solve_k`, `routing_adjoint`.
  5. `build_xi` -> `build_xi_star` (the ONLY construction of the installed `Xi*`).
  6. The dead `TaggedTransport` census entry is dropped; `_BIND` names the four
     replacement objects in one place.

The rebinding census, the install/restore machinery, the replay, the anchor gate and the
frozen-hash report are round 4's, with the one signature change listed above.

WHAT THIS ROUND CHANGES, IN ONE SENTENCE
----------------------------------------
Round 4 installed `Xi_raw(beta)`.  This round installs

    Xi_star(t, r) = k_r(beta) * Xi_raw(t, r)      on ACTIVE cells, and 1.0 elsewhere

where `k_r` is solved -- NOT fitted -- so that the reach's total mobilised mass over the
solving window is UNCHANGED FROM THE beta = 0 BASELINE:

    sum_t av0[t,r] * (1 - exp(-h[t,r]*k_r*Xi_raw[t,r]))
        = sum_t av0[t,r] * (1 - exp(-h[t,r]))                     <-- beta = 0, Xi = 1

                              ^^^ RHS HAS NO Xi ^^^

THAT ASYMMETRY IS THE WHOLE DEVICE AND IS EASY TO GET WRONG
-----------------------------------------------------------
The right-hand side is the mobilisation at `beta = 0`, NOT round 4's mobilisation at this
beta.  The quantity this round must not let beta inflate is the long-run mean relative to
NO beta at all -- "只允许'什么时候浓'，不允许凭空把长期平均N抬高".  Normalising against
round 4's own `sum av0[1-exp(-h*Xi_raw)]` would preserve beta's level shift instead of
removing it, i.e. would make the device a no-op by construction.  `T_target` is therefore
`(Wt * Q * a0 * p0).sum(0)`, read off the FROZEN beta = 0 trajectory.

`av0` is the FROZEN (beta = 0, k = 1) available-N trajectory from `closures.scan`; the
residual `rho` of the identity is reported per reach, and the exact fixed point
(`av` re-solved under `k`) is reported beside it.  NOTHING IS FITTED: `k_r` reads no
observation, has no objective function and no search over data.  See `xi_k.py`.

WHY `k_r` IS A `(nr,)` VECTOR AND NOT A STATION-LEVEL ONE
---------------------------------------------------------
This round has 15 stations on 13 reaches: 九甸大桥 and 永昌桥 share reach 158, and
盘溪大桥 and 禄丰村 share reach 190.  A station-level target would emit TWO different `k`
for one reach, which is not a normalisation of the land phase at all.  So the target is
per-reach and, for a concentration device, the normaliser is the FROZEN
`W = fast_water + percolation * area_ha * 10` (`xi_k.geometry`), never a routed station
quantity and never the statistic G5b measures.

NOTHING IS WRITTEN INTO 20260916_2, AND NOTHING COVERED BY `frozen_hashes` IS EDITED
------------------------------------------------------------------------------------
`closures_mc.py` is carried over UNEDITED from round 4: it already takes `Xi` as a real
call argument under `CONFIG['Xi']`, so "install `Xi_star` instead of `Xi_raw`" is a change
of ARGUMENT, not of kernel source.
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
assert ROUND.name == '20260919_5', ROUND
assert PEER.is_dir(), PEER
assert R4.is_dir(), R4

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

assert cm.RUN == PEER, (cm.RUN, PEER)
os.environ['NUMBA_CACHE_DIR'] = str(CACHE)
_nbcfg.CACHE_DIR = str(CACHE)

import torch  # noqa: E402
import closures as _closed  # noqa: E402
import tagged_transport as _tagged  # noqa: E402
import routing as _routing  # noqa: E402
import model as _hcm  # noqa: E402  (`vendor/expert/tn_challenge/model.py`)
import xi_k as XI  # noqa: E402  (Xi_raw via xi_base, plus the k-solve)
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
REF_YEARS = (1961, 2020)          # strictly pre-evaluation
EVAL_YEARS = (2021, 2024)

# --------------------------------------------------------------------------
# THE BETA GRID -- round 4's, byte-for-byte, in round 4's order
# --------------------------------------------------------------------------
# The order is round 4's `BETA_MAIN + BETA_HYPER + BETA_EXTREME` concatenation, i.e.
# 0, -0.05, +0.05, -0.1, +0.1, ... -- NOT sorted by |beta|.  It is copied rather than
# re-sorted because `phase1_l1.json::grid` is the frozen registry and a re-ordering would
# be a silent re-registration.
BETA_MAIN = (0.0, -0.05, 0.05, -0.1, 0.1, -0.25, 0.25, -0.5, 0.5, -1.0, 1.0)
BETA_HYPER = (-1.5, 1.5, -2.0, 2.0)
BETA_EXTREME = (-4.0, 4.0, -8.0, 8.0)
BETA_GRID = BETA_MAIN + BETA_HYPER + BETA_EXTREME
BETA_GROUPS = {'main': list(BETA_MAIN), 'hyper': list(BETA_HYPER),
               'extreme': list(BETA_EXTREME)}
assert len(BETA_GRID) == 19 and len(set(BETA_GRID)) == 19, BETA_GRID

SAT_BOUND = 0.10       # frac(prob >= 0.99) must be <= this
DEGEN_FLOOR = 1e-3     # frac(|Xi - 1| > 1e-6) must be >= this
W_FLOOR = XI.W_FLOOR

# --------------------------------------------------------------------------
# THE FOUR DEVICES.  All four share ONE kernel change; they differ only in what
# quantity the per-reach constant `k_r` is asked to preserve and on which window.
# --------------------------------------------------------------------------
#  name  target quantity                  window      role
#  N1    mobilised MASS  sum av0*p         1961-2020   PRIMARY, carries the verdict
#  N1e   mobilised MASS  sum av0*p         2021-2024   window-leak diagnostic only
#  N2    CONCENTRATION   (sum av0*p)/W     2021-2024   diagnostic only, weakest G5b
#  N3    CONCENTRATION   (sum av0*p)/W     1961-2020   verdict-capable: its target grid
#                                                      does not intersect the scored grid
DEVICES = ('N1', 'N1e', 'N2', 'N3')
DEVICE_SPEC = {
    'N1':  dict(target='mass', window='ref',   role='primary'),
    'N1e': dict(target='mass', window='eval',  role='diagnostic'),
    'N2':  dict(target='conc', window='eval',  role='diagnostic'),
    'N3':  dict(target='conc', window='ref',   role='verdict_capable'),
}
WINDOWS = {'ref': REF_YEARS, 'eval': EVAL_YEARS}

# --------------------------------------------------------------------------
# FROZEN ANCHORS -- READ from their producers, never retyped (delivery check 1)
# --------------------------------------------------------------------------
# Citing the round that FIRST produced a value rather than a downstream restatement of it
# is the `near-duplicate-products-share-a-name` discipline; `restatement_cross_check`
# re-reads round 4 and asserts agreement, so the restatement is CHECKED, not trusted.
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
# Round-5 additions: quantities round 4 MEASURED but did not register as anchors.  All are
# produced by round 4.  They are the operands of G5b, G3 and the G4 flip threshold, so
# they are frozen here instead of being re-derived from a re-run.
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
    # NOT `('points', '0.5', 'mean_concentration')` -- THAT KEY DOES NOT EXIST, and the
    # first draft of this table registered a value under it anyway (a number that had been
    # BACK-SOLVED from the baseline and the relative change, and which does not even equal
    # that product: 2.8234476727379607 * (1 + 0.22381499501517088) = 3.455377599537403).
    # A back-solved literal is not an anchor: it cannot be read back from the JSON that is
    # supposed to hold it, which is exactly what delivery check 1 exists to catch.  Round 4
    # stored only the RELATIVE change at a candidate point (`phase2_full.py:238-240`,
    # `abs(C_cand - C_null)/C_null`), so that is what is registered here, and no absolute
    # candidate mean concentration is claimed.
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
    # `S_u` IS REGISTERED IN PLAN SECTION 3.1 AND WAS MISSING FROM THIS TABLE.  Every
    # point record reports `g = beta * S_u` beside the raw `beta` as a pure relabelling
    # (`20260919_4/work/phase1_l1.py:203` spells it `float(beta * G['S_u'])`), so the
    # field cannot be produced without it -- `phase1_full.py` reached for `A['S_u']` and
    # the key did not exist.  It is read back from its producer here rather than
    # re-typed, and `phase1_full.py` additionally asserts that the geometry THIS round
    # re-derives reproduces it bitwise, which is a check the round-4 spelling never had.
    'S_u': ('20260919_4/reports/frozen_anchors.json',
            ('geometry', 'S_u'), 14.438101895057091),
}
SD_BASE_DDOF0 = {'L1': 1.3155445644228716, 'L2': 0.5534371191913016,
                 'L3': 0.7941489653630477}
SD_GATE_70PCT = {'L1': 0.9208811950960101, 'L2': 0.3874059834339111,
                 'L3': 0.5559042757541334}
PEER_ROOT = PEER.parent


def _dig(obj, keys):
    for k in keys:
        if not isinstance(obj, dict) or k not in obj:
            raise SystemExit('ANCHOR_KEY_MISSING %r' % (keys,))
        obj = obj[k]
    return obj


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
    # The per-layer SDs live only inside `baseline.sd_all_layers`, a nested map rather than
    # a scalar, so they are read by hand and checked the same way.
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
    out['sd_gate_layer'] = dict(value=gate_layer,
                                source='20260919_4/reports/phase2_full.json',
                                keys=['baseline', 'sd_gate_layer'],
                                sha256=sha(R4 / 'reports/phase2_full.json'),
                                origin='round4', matches_registered=True)
    if not bool(pf['baseline']['sd_ddof_choice_is_inert']):
        raise SystemExit('SD_DDOF_CHOICE_NO_LONGER_INERT')
    return out


# --------------------------------------------------------------------------
# round-4 RESTATEMENT CROSS-CHECK (`near-duplicate-products-share-a-name`)
# --------------------------------------------------------------------------
_RESTATEMENTS = {
    'A_L1': ('phase1_l1.json', ('anchors', 'A_L1')),
    'A_L3': ('phase1_l1.json', ('anchors', 'A_L3')),
    'A_obs': ('phase1_l1.json', ('anchors', 'A_obs')),
    'G1_target_50pct': ('phase1_l1.json', ('anchors', 'G1_target_50pct')),
    'G2_target_50pct': ('phase1_l1.json', ('anchors', 'G2_target_50pct')),
    'alpha_obs': ('phase1_l1.json', ('anchors', 'alpha_obs')),
    'beta_obs': ('phase1_l1.json', ('anchors', 'beta_obs')),
    'nse': ('phase2_full.json', ('baseline', 'monthly', 'nse')),
    'median_station_nse': ('phase2_full.json',
                           ('baseline', 'monthly', 'median_station_nse')),
}


def restatement_cross_check():
    """Producer value vs round-4's restatement of it, both read from disk.

    TWO SHAPES, AND THEY ARE NOT INTERCHANGEABLE.  Round 4's
    `phase1_l1.json::anchors.<name>` is not a bare number: it is round 4's own
    `load_anchors()` record -- a dict carrying `value`, `source`, `keys` and `sha256`.
    Round 4's `phase2_full.json::baseline.monthly.<name>` entries ARE bare numbers.  The
    first version of this function assumed a scalar everywhere and died with
    `TypeError: float() argument must be a string or a real number, not 'dict'` on the
    first nested entry, which is how the shape was found -- the comparison had been aimed
    one level too high.

    BOTH SHAPES ARE NOW HANDLED, and the nested record's `source` is compared as well:
    a restatement that agrees on the NUMBER while citing a DIFFERENT producer is still a
    finding, and it is the exact failure mode the `near-duplicate-products-share-a-name`
    discipline exists to catch.
    """
    out = {}
    prod = {n: (t[n][0], t[n][1])
            for t in (ANCHOR_PATHS, ANCHOR_PATHS_4) for n in t}
    for name, (f, keys) in _RESTATEMENTS.items():
        p = R4 / 'reports' / f
        if not p.is_file():
            out[name] = dict(status='ROUND4_FILE_ABSENT', round4_file=f)
            continue
        got = _dig(read_json(p), keys)
        nested = isinstance(got, dict)
        gv = got.get('value') if nested else got
        cited = got.get('source') if nested else None
        rel, pk = prod.get(name, (None, None))
        root = R4.parent if rel and rel.startswith('20260919_4') else PEER_ROOT
        pv = _dig(read_json(root / rel), pk) if rel else None
        out[name] = dict(round4_restatement_shape=('record' if nested else 'scalar'),
                         round4_restatement=gv, producer=pv,
                         agree=bool(pv is not None and gv is not None
                                    and float(gv) == float(pv)),
                         round4_cites_source=cited,
                         cited_matches_the_producer_path=(None if cited is None
                                                          else bool(cited == rel)),
                         round4_file=f, keys=list(keys))
    return dict(checked=out, n=len(out),
                n_disagree=int(sum(1 for v in out.values() if v.get('agree') is False)),
                n_citing_a_different_producer=int(
                    sum(1 for v in out.values()
                        if v.get('cited_matches_the_producer_path') is False)),
                note='a restatement that disagrees with its producer is a FINDING, not a '
                     'reason to prefer one of the two')


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
# the frozen model, verbatim from common22.py
# --------------------------------------------------------------------------
def model_record(tag):
    return read_json(PEER / 'outputs' / tag / 'model.json')


def parameters(tag):
    """The frozen 30-vector.  This round NEVER alters one element of it."""
    x = np.asarray(model_record(tag)['parameters'], float)
    assert len(x) == 30, (tag, len(x))
    return x


def design_for(tag):
    """The registered design with only the observation-registry path substituted."""
    design = json.loads(json.dumps(model_record(tag)['design']))
    before = dict(design)
    design['observation_registry_file'] = 'data/prediction_registry.json'
    design['observation_registry_hash'] = sha(PEER / 'data/prediction_registry.json')
    changed = {k for k in set(before) | set(design) if before.get(k) != design.get(k)}
    assert changed <= {'observation_registry_file', 'observation_registry_hash'}, changed
    assert design['structure'] == {'human': False, 'calendar': 'monthfirst'}, \
        design['structure']
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


# --------------------------------------------------------------------------
# the mobile-water geometry, computed ONCE and cached
# --------------------------------------------------------------------------
_GEOM = {}


def geometry(model=None, tag=TAG):
    """`xi_k.geometry` on the model's own data, memoised for the process."""
    if 'G' not in _GEOM:
        if model is None:
            model = build(tag)
        d = model.data
        _GEOM['G'] = XI.geometry(d.fast_water, d.percolation, d.area_ha, d.dates,
                                 ref_years=REF_YEARS, w_floor=W_FLOOR)
        _GEOM['model'] = model
    return _GEOM['G']


def pilot_indices(model):
    """READ from the model -- never hard-coded.  These are 0-BASED COLUMN indices."""
    rr = list(model.data.pilot_indices)
    assert len(rr) > 0, 'PILOT_INDICES_EMPTY (operator_id must not be O0)'
    assert all(0 <= int(i) < int(model.data.fast_water.shape[1]) for i in rr), rr
    return [int(i) for i in rr]


# --------------------------------------------------------------------------
# ZERO-FORWARD: the arrays the k-solve needs
# --------------------------------------------------------------------------
def bootstrap_arrays(model, tag=TAG):
    """`h`, `s`, `f`, `k`, `av0`, `p0` and the reach-level inputs.  NO routing, NO
    station aggregation, NO `replay`.

    WHY NOT `replay`: `common22.py::replay` returns only the station-day concentration
    frame; it never returns the per-reach `a`/`p` the identity is written in.
    `closures.ledger` returns `available=a` but NOT `p` (`closures.py:227`), so it cannot
    supply `p0` either.  `closures.scan` returns BOTH as its 3rd and 4th outputs, which is
    why the frozen `scan` is called directly here.
    """
    x = parameters(tag)
    with torch.no_grad():
        h, s, f, kk = model.flux_parameters(torch.tensor(x))
    hh, ss, ff, kk_ = [np.ascontiguousarray(v.detach().numpy()) for v in (h, s, f, kk)]
    # SHAPES DIFFER BY DESIGN, and asserting them equal is a bug (it fired on the first
    # run of this file).  `flux_parameters` returns `h` and `f` as `(nd, nr)`, but `s` as
    # `(nr,)` -- the frozen kernel handles both spellings explicitly:
    # `survival = s[r] if s.ndim == 1 else s[t, r]` (`closures.py:14-29`).  `S2` below is
    # the 2-D spelling of whatever `s` is, which is what the C3 ledger identity needs.
    assert hh.ndim == 2 and ff.ndim == 2, (hh.shape, ff.shape)
    assert hh.shape == ff.shape, (hh.shape, ff.shape)
    assert kk_.shape == (hh.shape[1],), (kk_.shape, hh.shape)
    assert ss.ndim in (1, 2) and ss.size in (1, hh.shape[0], hh.shape[1],
                                            hh.shape[0] * hh.shape[1]), ss.shape
    fast, slow, a0, p0 = _closed.scan(hh, ss, ff, kk_, model.data.lower_release,
                                      model.inp, model.demand, model.cap)
    S2 = np.ascontiguousarray(np.broadcast_to(ss, hh.shape) if ss.ndim == 1
                              else np.asarray(ss))
    return dict(h=hh, s=ss, S2=S2, f=ff, k=kk_,
                a0=np.ascontiguousarray(a0), p0=np.ascontiguousarray(p0),
                fast=np.ascontiguousarray(fast), slow=np.ascontiguousarray(slow),
                inp=np.ascontiguousarray(model.inp),
                demand=np.ascontiguousarray(model.demand),
                lower_release=model.data.lower_release, cap=bool(model.cap),
                dates=model.data.dates, shape=list(hh.shape))


def window_mask(dates, window):
    """Boolean `(nd,)` mask for an inclusive `(start_year, end_year)` window."""
    yrs = np.asarray(dates).astype('datetime64[Y]').astype(np.int64) + 1970
    return (yrs >= int(window[0])) & (yrs <= int(window[1]))


# --------------------------------------------------------------------------
# THE TARGET QUANTITY, per device
# --------------------------------------------------------------------------
def device_state(B, G, device):
    """`(Q, Wt)` for one device, from FROZEN arrays only.

    `Q` is the per-reach normaliser `(nr,)`: `1` for a MASS device, `1/W_r` for a
    CONCENTRATION device.  `Wt` is the window mask `(nd,)`.  Keeping the normaliser
    SEPARATE from the state is what lets the fixed point re-solve `av(k)` while the target
    stays anchored on `av0`.

    WHY `1/W` FOR A CONCENTRATION DEVICE
    ------------------------------------
    The station-level quantity a criterion reads is a ROUTED concentration.  Routing a
    per-reach `k` up to the station would make `k` a station constant, and two reaches in
    this round carry TWO stations each (158: 九甸大桥/永昌桥, 190: 盘溪大桥/禄丰村), so a
    station target would emit two `k` per reach -- not a normalisation at all.  The
    concentration device therefore divides by the FROZEN `W = fast_water + percolation *
    area_ha * 10` (`xi_k.geometry`), which is (a) per-reach, (b) entirely frozen, and (c)
    NOT the statistic G5b measures.  That last point is a registered red line: pinning the
    target on the criterion's own statistic would make G5b true by construction.
    """
    spec = DEVICE_SPEC[device]
    Wt = window_mask(B['dates'], WINDOWS[spec['window']])
    nr = B['shape'][1]
    if spec['target'] == 'mass':
        Q = np.ones(nr)
    else:
        assert spec['target'] == 'conc', spec
        Wr = np.where(Wt[:, None], G['W'], 0.0).sum(axis=0)
        if not np.all(Wr > 0):
            raise SystemExit('DEVICE_WINDOW_WITHOUT_WATER %r' % device)
        Q = 1.0 / Wr
    return np.ascontiguousarray(Q), Wt


def solve_k(B, G, device, beta, **kw):
    """Build this device's `(Q, Wt)` and delegate to `xi_k.solve_k`.

    THE SOLVE LIVES IN `xi_k.py` AND ONLY THERE (plan section 7).  This function is a
    wrapper: it resolves the device into a per-reach normaliser `Q` and a window mask
    `Wt`, hands the FROZEN `closures.scan` in as `scan_fn`, and stamps the device
    identity onto the result.  Any second implementation of the root-find would be a second
    SOURCE OF `k`, which is exactly the defect this round's plan forbids.
    """
    Q, Wt = device_state(B, G, device)
    out = XI.solve_k(B, G, Q, Wt, _closed.scan, beta, **kw)
    out.update(device=device, window=DEVICE_SPEC[device]['window'],
               target=DEVICE_SPEC[device]['target'])
    return out


def build_xi_star(model, B, G, device, beta, **kw):
    """THE ONLY CONSTRUCTION OF THE INSTALLED `Xi*`.  Returns `(Xi*, Xi*_tag, k, res)`.

    Three things are pinned here, and none of them is left to the caller:

    1. `Xi*` comes from `xi_k.xi_star`, which is MULTIPLY THEN PIN.  Pinning first would
       hand all 832,146 inactive cells a `k_r != 1`; the plan makes the order a hard
       constraint, and `xi_k.pin_report` measures the footprint where the pin is NOT inert.
    2. `Xi*_tag` is the SAME `Xi*` sliced at the `pilot_indices` READ FROM THE MODEL, so the
       pilot columns carry `k` too.  If the scalar kernel got `k` and the tagged kernel did
       not, `source_label_sum_errors` would measure that mismatch instead of the roundoff it
       is registered to measure.  `pilot_indices` is never hard-coded -- in round 4 it was
       `[157, 224]`, which is a fact about this model, not a constant of the problem.
    3. `Xi*` is asserted the same shape as `h`.  `closures.scan` is `@njit` with no shape
       checking, so a mismatch here is not an exception: it is a silent wrong-column read or
       a SIGSEGV.  The assert belongs at the boundary where the array is born.
    """
    res = solve_k(B, G, device, beta, **kw)
    k = np.asarray(res['k'], dtype=np.float64)
    Xs = XI.xi_star(G, beta, k)
    XT = XI.xi_star_tag(G, beta, k, pilot_indices(model))
    h = np.asarray(B['h'], dtype=np.float64)
    if Xs.shape != h.shape:
        raise SystemExit('XI_STAR_SHAPE %r != h %r' % (Xs.shape, h.shape))
    if XT.shape[0] != h.shape[0]:
        raise SystemExit('XI_STAR_TAG_ROWS %r != h rows %d' % (XT.shape, h.shape[0]))
    if not np.isfinite(Xs).all() or not (Xs > 0).all():
        raise SystemExit('XI_STAR_NOT_POSITIVE_FINITE')
    res['xi_star_shape'] = list(Xs.shape)
    res['xi_star_tag_shape'] = list(XT.shape)
    res['pin'] = XI.pin_report(G, Xs, h=h, k=k)
    return Xs, XT, k, res


# --------------------------------------------------------------------------
# C7: the frozen linear routing adjoint
# --------------------------------------------------------------------------
def routing_adjoint(model, B, mask=None, tag=TAG):
    """`share[t, r] = d(mean pL3 over the selected rows) / d local[t, r]`.

    WHY THIS EXISTS.  `routing.py:130` is LINEAR in `local` and its coefficients do not
    contain `Xi`:
        mass[i] = inlet[ti,ri]*exp(-vf*h*f) + f*local[ti,ri]*exp(-.5*vf*h*f)
    so ONE adjoint pass of the FROZEN model gives a `(nd, nr)` sensitivity that is valid
    for EVERY beta and every device.  A change in the land phase can then be turned into a
    predicted change in the scored mean concentration with no torch forward and no routing
    re-run:

        dCbar_pred = sum_{t,r} share[t,r] * dlocal[t,r]

    and `dlocal` is available from the land-phase recursion alone.  Reporting `dCbar_pred`
    beside the measured `dCbar` is what separates "the device failed to move the level"
    from "the linearisation broke".

    `local` is recomputed here exactly as `temporal_model.daily_boundary` computes it
    (`local = fast + slow`; the human term is absent because `structure.human is False`).
    `mask` selects rows of the station-day record; `None` means the whole record.
    """
    x = parameters(tag)
    meta = pd.read_parquet(PEER / 'data/prediction_calendar.parquet')
    meta = meta[meta.year.le(END_YEAR)].copy()
    c, record, _w = model.daily_metadata(meta)
    vf = torch.tensor(x)[2]
    local0 = torch.tensor(np.ascontiguousarray(B['fast'] + B['slow']), dtype=torch.float64)
    local0.requires_grad_(True)
    i, o, r = _routing.RiverN.apply(local0, vf, model.daily_data, 'monthly', True)
    mass = _routing.boundary_mass(i, o, r, local0, vf, c)
    p = 1000.0 * mass / c['water']
    if mask is None:
        obj, n_sel = p.mean(), int(p.numel())
    else:
        m = torch.as_tensor(np.asarray(mask, bool), dtype=torch.bool)
        if m.numel() != p.numel():
            raise ValueError('ADJOINT_MASK_LENGTH %d vs %d' % (m.numel(), p.numel()))
        obj, n_sel = p[m].mean(), int(m.sum())
    obj.backward()
    gl = local0.grad.detach().numpy()
    return dict(share=np.ascontiguousarray(gl), shape=list(gl.shape),
                objective='mean(1000*mass/water) over the selected record rows',
                n_rows_selected=n_sel, n_rows=int(p.numel()),
                local_shape=list(np.asarray(B['fast']).shape),
                operator_id=model.data.operator_id,
                note='coefficients are Xi-free, so this adjoint is valid for every beta '
                     'and every device; `local` here is fast+slow with no human term, '
                     'matching temporal_model.daily_boundary')


# --------------------------------------------------------------------------
# install / restore -- FOUR rebindings, from round 4
# --------------------------------------------------------------------------
def derive_binding_modules(attr, target, known):
    """Every ALREADY-LOADED peer module whose globals bind `attr` to `target`."""
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
           'tag_scan': cm.tag_scan, 'tagged_tag_scan': _tagged.tag_scan}
_KNOWN = {'Transport': ('closures', 'scientific_models', 'temporal_model', 'hf_model',
                        'structure_model', 'campaign_model'),
          'scan': ('closures',),
          'tag_scan': ('campaign_model', 'tagged_transport')}
_BIND = {'Transport': 'TransportMC', 'scan': 'scan_mc_ledger', 'tag_scan': 'tag_scan_mc'}


def binding_census():
    out = {}
    for attr in ('Transport', 'scan', 'tag_scan'):
        found, extra = derive_binding_modules(attr, _FROZEN[attr], _KNOWN[attr])
        out[attr] = dict(modules=sorted(found), known=list(_KNOWN[attr]), extra=extra,
                         n_modules=len(found))
    return out


def snapshot_import_census():
    """Every import line naming `Transport`, so a missed rebinding site is visible."""
    hits = []
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
                if s.startswith('#') or s.startswith('Transport'):
                    continue
                if 'import' in s and 'Transport' in s:
                    hits.append(dict(file=str(p.relative_to(PEER)), line=ln, text=s))
    return dict(n_hits=len(hits), hits=hits)


_INSTALLED_SETS = {}


def install_kernel(model, Xi_star, Xi_star_tag):
    """Bind the modulated kernel and the geometry into this process.

    FOUR bindings: `Transport` across the six modules that bind it, `closures.scan`,
    `campaign_model.tag_scan` and `tagged_transport.tag_scan`; plus one CLASS-LEVEL
    non-binding (`Predictor.hazard` must stay the frozen method).  The sites are
    re-derived from the loaded modules rather than listed.  `Xi_star` is the ONLY thing
    this round puts in: `closures_mc.py` is carried over UNEDITED, so the kernel
    arithmetic is round 4's byte for byte and the difference is entirely in the array
    handed to it.
    """
    X = np.ascontiguousarray(Xi_star, dtype=np.float64)
    XT = np.ascontiguousarray(Xi_star_tag, dtype=np.float64)
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
    _INSTALLED_SETS.update({k: list(v['modules']) for k, v in cen.items()})
    for attr, names in _INSTALLED_SETS.items():
        impl = getattr(MC, _BIND[attr])
        for n in names:
            setattr(sys.modules[n], attr, impl)
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


_ORIG_HAZARD = _hcm.Predictor.hazard


def hazard_is_frozen():
    """`Predictor.hazard` must still be the frozen method object."""
    return bool(_hcm.Predictor.hazard is _ORIG_HAZARD)


def is_installed():
    per = {}
    for attr, names in (_INSTALLED_SETS or {}).items():
        impl = getattr(MC, _BIND[attr])
        live = [n for n in names if getattr(sys.modules.get(n), attr, None) is impl]
        per[attr] = dict(modules=names, n_modules=len(names), live=live,
                         all_live=bool(names) and len(live) == len(names))
    hazard_ok = hazard_is_frozen()
    return dict(per_name=per, hazard_is_frozen=hazard_ok,
                n_transport_modules=len((_INSTALLED_SETS or {}).get('Transport', [])),
                all_bound=bool(per) and all(v['all_live'] for v in per.values())
                and hazard_ok)


# --------------------------------------------------------------------------
# the daily replay, byte-for-byte the arithmetic of common22.py::replay
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
    """The frozen map, restricted to the files this round leans on."""
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
            if base in watched:
                now = sha(PEER / k)
                out[fold]['watched'][base] = dict(registered=v, on_disk=now, path=k,
                                                  unchanged=bool(now == v))
    return out


ROUND4_JSON_SHA = {
    'verdict.json': '898382605b1d019ddc880c9391c7d7c9a0760a42337d8ab5218f106c51712387',
    'phase2_full.json': '8910569f717de60c2f46fae2039f3f0ed5899bb641bc0d2ddcc901ae6b80cda4',
    'phase1_l1.json': 'f9b60441efafa07d4f9901e423d6785acdad9dfa610cc23e6e6f32e242aa31e5',
}


def round4_json_shas():
    """The three round-4 JSONs whose sha round-4's reports quote.  MUST NOT CHANGE."""
    return {f: sha(R4 / 'reports' / f) for f in ROUND4_JSON_SHA}


def assert_round4_jsons_intact():
    got = round4_json_shas()
    bad = {k: (got[k], v) for k, v in ROUND4_JSON_SHA.items() if got[k] != v}
    if bad:
        raise SystemExit('ROUND4_JSON_CHANGED %r' % bad)
    return dict(shas=got, all_intact=True,
                note='round-4 JSONs are read-only for this round; only round-4 PROSE may '
                     'be annotated (plan section 5)')


# --------------------------------------------------------------------------
# THE MONTHLY APPARATUS -- ONE code path, shared by C1 and by G5
# --------------------------------------------------------------------------
# WHY IT LIVES HERE AND NOT IN A PHASE SCRIPT.  Round 4 kept `obs_monthly` and
# `monthly_stats` inside `phase2_full.py`, which was fine while exactly one script needed
# them.  Round 5 needs them in TWO scripts -- `phase_minus1.py` (C1's pure-level control)
# and `phase1_full.py` (G5/G5b) -- and a second copy of a criterion's code path is a second
# SOURCE of that criterion, which is a defect this round's plan names explicitly.  It is
# therefore defined ONCE, here, and both scripts import it.
#
# The arithmetic is round 4's, verbatim, with ONE parameter added: a multiplicative
# `scale` on the station-day concentration column.  `scale == 1.0` reproduces round 4's
# code path bitwise (the branch is not taken), and it is what G5 uses.  `scale = 1+delta`
# is C1's pure-level control -- "multiply the baseline by a constant and see how much of
# the monthly degradation that alone repairs".  Scaling the STATION-DAY column and then
# grouping is the same monthly quantity as scaling the grouped series, and it is the
# literal reading of C1, so there is exactly one place where the level moves.
PANEL_SHA = '7ed9e6179705affc00494fafb2119ea2c656b32e661c44fee7dd5fd162d56a18'
MONTHLY_GATE = 0.005        # phase1_score.py:42, carried over unchanged
SD_GATE_FRACTION = 0.70     # G3: "median_s e_s <= 0.70 x baseline"


def eligible_grid():
    """The frozen eligible station-day grid, with the mask's sha asserted."""
    if sha(EL.MASK) != EL.MASK_SHA:
        raise SystemExit('ELIGIBLE_MASK_DRIFTED %s' % sha(EL.MASK))
    m = pd.read_parquet(EL.MASK)
    out = m[m.eligible][['station_key', 'date']].copy()
    out['date'] = EL.as_day(out.date)
    return out


def obs_monthly():
    """Observed monthly station-mean TN, from the READ-ONLY 4h panel.

    Verbatim from `20260919_4/work/phase2_full.py::obs_monthly` (itself from
    `phase1_score.py:66-79`) with the path taken from `eventlib.PANEL`.  This is a LEVEL
    gate on a concentration -- not an event selection and not a fit target -- which is
    exactly what the panel's registered status permits: a zero-fit round may SCORE
    against it, never calibrate on it.
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
    """`(m, z)`: the eligible-row frame and the joined station-month frame.

    `m` is the eligible grid with the candidate's `pL3` attached (a dropped eligible day is
    a loud stop, never a silent intersect); `z` is `m` collapsed to station-months and
    inner-joined onto the observation index, which is what every NSE below is computed on.
    """
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
    """Round 4's `monthly_stats` return block, on an already-joined `z`."""
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
    """One call: join, then score.  `scale = 1.0` IS round 4's code path."""
    m, z = monthly_join(ly, elig, obs_m, scale)
    return monthly_stats_from_z(m, z)


def nse_of(o, pv):
    """The pooled monthly NSE, exactly as `monthly_stats` computes it."""
    o = np.asarray(o, float)
    pv = np.asarray(pv, float)
    return float(1.0 - np.mean((pv - o) ** 2) / np.var(o))


# --------------------------------------------------------------------------
# TOPOLOGY -- which reaches drain into which (read from the frozen network)
# --------------------------------------------------------------------------
def catchment_map(model=None, tag=TAG):
    """Per-reach CATCHMENT: the reach itself plus every reach that drains into it.

    `campaign_model.load_data` exposes `d.downstream = {reach: its downstream reach}`.  The
    upstream relation is that dict INVERTED, and the catchment is the closure under the
    walk from the reach upward.  A reach is never visited twice, so a cycle in the topology
    would terminate rather than hang -- and the resulting set would then be wrong, which is
    why the graph is summarised and reported instead of trusted silently.

    This exists because section 2.8 of the plan needs an unmeasured quantity: the
    SPATIAL CONSISTENCY of `k_r` over the reaches that supply a station's water.  L3 at a
    station is mostly upstream mass (`c_base_L1` is about 32% of `c_base_L3` at the event
    base medians), so a per-reach constant is only a per-STATION constant to the extent
    that the catchment's `k_r` agree.
    """
    if model is None:
        model = build(tag)
    d = model.data
    nr = int(d.fast_water.shape[1])
    up = {r: [] for r in range(nr)}
    for a, b in d.downstream.items():
        up[int(b)].append(int(a))
    out = {}
    for r in range(nr):
        seen = {r}
        stack = [r]
        while stack:
            c = stack.pop()
            for s in up.get(c, ()):
                if s not in seen:
                    seen.add(s)
                    stack.append(s)
        out[r] = sorted(seen)
    return dict(upstream={int(k): [int(v) for v in vs] for k, vs in up.items()},
                catchment={int(k): v for k, v in out.items()},
                n_reaches=nr, n_downstream_edges=len(d.downstream),
                terminal=[int(v) for v in np.atleast_1d(d.terminal)],
                max_catchment_size=int(max(len(v) for v in out.values())),
                note='d.downstream is read from the frozen topology; the catchment is its '
                     'inverted closure, so it is downstream-inclusive of the reach itself')


def station_reaches(model=None, tag=TAG):
    """The eligible station_key -> 0-BASED reach column, DERIVED from the calendar.

    The plan registers the 15-station / 13-reach table in prose.  Deriving it here from
    `prediction_calendar.parquet` and ASSERTING the derived reach set equals the
    registered one is what turns that prose into a checked claim; two stations sharing one
    reach (158: 九甸大桥/永昌桥, 190: 盘溪大桥/禄丰村) is the reason a per-reach `k` cannot
    be pushed to the station level without becoming two different constants for one reach.

    SCOPED TO THE ELIGIBLE STATIONS.  The calendar carries every one of the 116 stations
    with a reach assignment; the first version of this function returned all of them (101
    distinct reaches) and therefore could not have matched the registered 13.  The scored
    set is the eligible mask's, so the eligible mask is what selects.
    """
    meta = pd.read_parquet(PEER / 'data/prediction_calendar.parquet')
    meta = meta[meta.year.le(END_YEAR)]
    elig = eligible_grid()
    keys = set(elig.station_key.astype(str))
    meta = meta[meta.station_key.astype(str).isin(keys)]
    if meta.station_key.astype(str).nunique() != len(keys):
        raise SystemExit('ELIGIBLE_STATION_MISSING_FROM_THE_CALENDAR %d vs %d'
                         % (int(meta.station_key.astype(str).nunique()), len(keys)))
    tab = (meta.assign(sk=meta.station_key.astype(str))
           .groupby('sk', as_index=False).reach_id
           .agg(lambda s: sorted({int(v) for v in s})))
    if any(len(v) != 1 for v in tab.reach_id):
        raise SystemExit('STATION_WITH_MULTIPLE_REACHES %r'
                         % tab[tab.reach_id.map(len) != 1].to_dict('records'))
    sr = {str(k): int(v[0]) - 1 for k, v in zip(tab.sk, tab.reach_id)}
    return sr, sorted(set(sr.values()))


STATION_REACHES_1BASED = (7, 10, 56, 59, 81, 103, 113, 120, 144, 158, 190, 210, 213)


def best_constant_scale(z):
    """C1's EXACT optimum, in closed form, with no grid and no search tolerance.

    Scaling the candidate station-month series by `c` and minimising
    `sum_i (c*p_i - o_i)^2` over `c` gives `c* = sum(p o) / sum(p p)`; the residual at
    `c*` is `sum(o^2) - (sum(o p))^2 / sum(p^2)`.  So the BEST achievable pooled monthly
    NSE under a pure level change is available in one line -- the plan writes "sweep
    delta", and a sweep is how that is usually done, but a sweep can only ever report the
    best point ON its grid, so the sweep is kept only as a cross-check of this value.
    `delta* = c* - 1`.
    """
    o = z.obs.to_numpy(float)
    p = z.pred.to_numpy(float)
    den = float(np.dot(p, p))
    if den <= 0:
        raise SystemExit('C1_ZERO_PREDICTION_POWER')
    c = float(np.dot(p, o)) / den
    resid = float(np.dot(o, o)) - (float(np.dot(o, p)) ** 2) / den
    nse = 1.0 - (resid / len(o)) / float(np.var(o))
    return dict(c_star=c, delta_star=c - 1.0, nse_at_star=float(nse),
                n_station_months=int(len(z)),
                definition='c* = argmin_c sum(c*pred - obs)^2 = <pred,obs>/<pred,pred>; '
                           'nse_at_star uses the same denominator (np.var of obs) as '
                           'monthly_stats, so the two are directly comparable')
