"""Round 20260919_3 bootstrap: the frozen reference tree, the z climate, and the
runtime install of the pathway-selective mobilisation modulation.

NOTHING IS WRITTEN INTO 20260916_2, AND NOTHING COVERED BY `frozen_hashes` IS EDITED
------------------------------------------------------------------------------------
`20260916_2/reports/launch_by_fold/{C0,C2}.json` each carry a 75-entry
`frozen_hashes` map.  Six of those are leaned on here and verified in Phase 0:

    vendor\\research\\closures.py          02433a200de4911f...   the frozen `scan`
    vendor\\research\\tagged_transport.py  b1a7df8a559e00a4...   the frozen `tag_scan`
    scripts\\structure_model.py            615e87b9fd2ec067...
    scripts\\hf_model.py                   88a85da13541e612...
    scripts\\temporal_model.py             1b3217cbec5260f0...
    scripts\\campaign_model.py             44bdb7c00b9ea572...

THIS ROUND DOES NOT COPY OR REBIND EITHER LAND-PHASE KERNEL.  `20260919_2` had to
copy `scan`/`tag_scan` because it changed the recurrence itself; here the change is
UPSTREAM of both, in the `f` they are handed.  `work/hazard_g.py` rebinds exactly
one function, `Predictor.hazard`, and `closures.scan` / `tagged_transport.tag_scan`
run byte-identical.  See hazard_g's docstring for why the binding is class-level.

WHAT THE MODEL ACTUALLY IS (read off the frozen artifacts, not assumed)
-----------------------------------------------------------------------
`outputs/C0_s1/model.json::design` carries `observation_operator='MATCH'`,
`structure={'human':False,'calendar':'monthfirst'}` and `operator_id='OU'`.  Both
keys `make_model` branches on are present, so `kind='D29_BE'` resolves to

    structure_model.StructureEndpoints -> hf_model.HFEndpoints
                                       -> temporal_model.TemporalEndpoints
                                       -> campaign_model.Endpoints -> Matched

No class in that chain overrides `flux_parameters`, so `f` comes from
`campaign_model.py:129-134::Endpoints.flux_parameters`, which calls
`Predictor.hazard(self, t[:29])` -- a CLASS-LEVEL explicit call.  That is the fact
`hazard_g.install` is built around.

`operator_id='OU'` (not `'O0'`) means `data.support` is loaded, so
`pilot_indices` is non-empty and `campaign_model.py:102`'s `if rr:` guard passes:
the `tag_scan` branch is LIVE in this round and conjunct 5 is not vacuous.
Phase 0 records the measured count rather than trusting this paragraph.
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
assert ROUND.name == '20260919_3', ROUND
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
# `eventlib.py` is the round's event/criterion apparatus and lives in the PEER round,
# which is read-only.  It is APPENDED, not prepended: this round's own modules must
# still win any name lookup, and a module the sandbox cannot shadow is one whose
# hash check below means something.
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
import hazard_g as HG  # noqa: E402
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

# the frozen A-arm reference window.  Strictly pre-evaluation: EVALUATION is 2021-2024.
REF_YEARS = (1961, 2020)

# --------------------------------------------------------------------------
# THE GAMMA GRID IS IN UNITS OF THE DRIVER'S OWN REFERENCE SD, AND THAT IS AN
# AMENDMENT MADE BEFORE ANY AMPLITUDE EXISTED
# --------------------------------------------------------------------------
# The grid was registered as `gamma in {0, 0.25, 0.5, 1, 2, 4}` with the reading
# "gamma = 1 <=> the N log-odds track the water log-odds".  Phase 0 measured that
# reading to be false by ~39x, because `z` is not O(1): `fast_water` bottoms out at
# 4.18447e-200, so `log(fast_water)` reaches -459 while remaining finite and strictly
# positive, and the per-reach sd of `z` over 1961-2020 is 38.925686592399551.
#
# In the ORIGINAL units the top of the grid is not an enrichment test at all:
#
#     gamma=2   pins f^N == 1 on 22.98% of station x evaluation cells
#     gamma=4   pins f^N == 1 on 59.85% of them
#
# -- a saturated binary selector, not pathway-selective mobilisation.  Rescaling
# `gamma` into units of `SIGMA_Z` puts all six points back in the modulating regime
# (`f^N == 1` is exactly 0.0000 at every point on the whole record and on the
# station x evaluation subset alike) and restores the registered reading: `g = 1` is
# one reference sd of the water log-odds, so `g in {2, 4}` is the enrichment
# hypothesis and `g <= 1` is the "at most proportional" control region.
#
# ONLY gamma's UNIT changed.  `z` is exactly as specified, the functional form is
# exactly as specified, and the gamma = 0 no-op is untouched because `0 * sigma` is
# exactly 0.0.  The amendment is registered in `reports/预注册_判据与门槛.md` and
# `reports/实际方法与偏离.md`, and it was made with no amplitude computed.
GAMMA_GRID = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
GAMMA_UNITS = 'sigma_z'
SIGMA_Z_RULE = ('median over all 230 reaches of the per-reach sd of z over 1961-2020 '
                '(the pre-evaluation reference window); the 13 station reaches alone '
                'would give 39.637173980835684, a 1.8% difference, and choosing them '
                'would be selecting the scale on the evaluation set')
SIGMA_Z_PUBLISHED = 38.925686592399551
SIGMA_Z_ALT_STATION_REACHES = 39.637173980835684


def gamma_effective(g, sigma):
    """`gamma` in the units `hazard_g` wants, from `g` in reference-sd units.

    The standardized driver is `z~/= z/sigma`, so `logit(f^N) = logit(f) + g*(z/sigma)`
    and therefore `gamma = g / sigma`.  (`g * sigma` would be the inverse rescaling and
    would put `g = 4` at `gamma = 155.7`, i.e. back inside the saturated regime the
    amendment exists to leave -- measured at `f^N == 1` on 59.85% of station x
    evaluation cells.)

    The option text shown when this amendment was chosen carried the formula line
    `gamma = g * sigma`; the table directly beneath it, which is what the choice was
    made on, used `g = 1 -> gamma = 0.025690 = 1/38.925668...`.  The table is correct
    and is what is implemented; the mis-stated line is recorded in
    `reports/实际方法与偏离.md`.

    `g = 0` maps to exactly `0.0`, so the no-op stays bitwise.
    """
    return float(g) / float(sigma)


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
# the frozen model, verbatim from 20260919_2/work/common20.py
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
# the reach-climatology log-odds driver z  (plan section 1.3)
# --------------------------------------------------------------------------
def build_z(model, ref_years=REF_YEARS):
    """z[t,r] = log(fast_water/slow_water) - reach mean over `ref_years`.

    LOCAL water, not routed: `f` is a function of local quantities and the N split
    happens in the land phase, so the routed ratio would import the in-channel
    layer this round deliberately does not touch.  The local pair is also strictly
    positive everywhere (asserted), so z needs NO epsilon and is therefore exact
    and free of a tuning constant; `routed_total_m3_s` has nonpositive cells in the
    reference window and would have needed one.
    """
    d = model.data
    fw = np.asarray(d.fast_water, dtype=np.float64)
    sw = np.asarray(d.slow_water, dtype=np.float64)
    assert fw.shape == sw.shape, (fw.shape, sw.shape)
    n_fw_nonpos = int((fw <= 0).sum())
    n_sw_nonpos = int((sw <= 0).sum())
    assert n_fw_nonpos == 0, 'FAST_WATER_NOT_STRICTLY_POSITIVE %d' % n_fw_nonpos
    assert n_sw_nonpos == 0, 'SLOW_WATER_NOT_STRICTLY_POSITIVE %d' % n_sw_nonpos
    lr = np.log(fw) - np.log(sw)
    years = np.asarray(d.dates.year, dtype=np.int64)
    m = (years >= ref_years[0]) & (years <= ref_years[1])
    n_ref_days = int(m.sum())
    clim = lr[m].mean(axis=0)
    z = lr - clim[None, :]
    assert np.isfinite(z).all(), 'Z_NOT_FINITE'
    # ---- SCALE DIAGNOSTIC.  `strictly positive` is NOT `usably non-zero` ----------
    # `fast_water` bottoms out at 4.18447e-200, so `log(fast_water)` reaches -459
    # while remaining finite and strictly positive.  The plan's precondition ②
    # (finite, > 0) therefore PASSES on a driver whose typical spread is ~39 log
    # units, i.e. ~250x the spread the registered reading `gamma = 1 <=> "N log-odds
    # track water log-odds"` presupposes.  Measured and reported here rather than
    # discovered later as an unexplained `f^N` step function.
    zr = z[m]
    sd_r = zr.std(axis=0)
    sigma_z = float(np.median(sd_r))
    return dict(sigma_z=sigma_z, sigma_z_rule=SIGMA_Z_RULE,
                sigma_z_matches_published=bool(sigma_z == SIGMA_Z_PUBLISHED),
                gamma_units=GAMMA_UNITS,
                gamma_grid_effective=[gamma_effective(g, sigma_z) for g in GAMMA_GRID],
                z=z.astype(np.float64), clim=clim, n_ref_days=n_ref_days,
                ref_years=list(ref_years),
                fast_water_min=float(fw.min()), slow_water_min=float(sw.min()),
                n_fast_water_nonpositive=n_fw_nonpos, n_slow_water_nonpositive=n_sw_nonpos,
                z_min=float(z.min()), z_max=float(z.max()), z_mean=float(z.mean()),
                z_sd=float(z.std()),
                log_ratio_clim_min=float(clim.min()), log_ratio_clim_max=float(clim.max()),
                z_ref_per_reach_sd_min=float(sd_r.min()),
                z_ref_per_reach_sd_median=float(np.median(sd_r)),
                z_ref_per_reach_sd_max=float(sd_r.max()),
                z_ref_per_reach_mean_abs_max=float(np.abs(zr.mean(axis=0)).max()),
                z_ref_frac_abs_gt_5=float((np.abs(zr) > 5).mean()),
                z_ref_frac_abs_gt_20=float((np.abs(zr) > 20).mean()),
                fast_water_frac_below_1e9=float((fw < 1e-9).mean()),
                fast_water_frac_exactly_log_floor=float((fw <= 1e-16).mean()))


def f0_census(model):
    """How often the carrier fraction touches an endpoint -- the reason the design
    multiplies `aq` instead of taking `logit(f0)` (logit(0) is -inf)."""
    f0 = np.asarray(model.data.fast_fraction, dtype=np.float64)
    return dict(shape=list(f0.shape), min=float(f0.min()), max=float(f0.max()),
                mean=float(f0.mean()), frac_exactly_zero=float((f0 == 0.0).mean()),
                frac_exactly_one=float((f0 == 1.0).mean()),
                n_nonfinite=int((~np.isfinite(f0)).sum()))


# --------------------------------------------------------------------------
# the daily replay, byte-for-byte the arithmetic of
# 20260919_2/work/common20.py::replay
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
    """The stored C0_s1 daily table must be reproduced exactly.

    A replay is only readable if it lands on the reference round's own stored
    artifact; otherwise the kernel install, not the science, is the finding.
    """
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


# --------------------------------------------------------------------------
# the one install this round performs
# --------------------------------------------------------------------------
def install_hazard(model, z, g, sigma):
    """Bind `hazard_g` on the CLASS and freeze (z, gamma) for it.

    `g` is in REFERENCE-SD units (`GAMMA_UNITS`); the value handed to `hazard_g` is
    `g / sigma` (see `gamma_effective`).  Both are recorded so no reported number can
    be read in the wrong unit.  `z` rides in hazard_g's module config rather than on the model, because the
    frozen `hazard` signature `(self, t)` is called from a frozen call site
    (`campaign_model.py:130`) that this round may not edit.
    """
    gamma_eff = gamma_effective(g, sigma)
    shape = HG.set_config(z, gamma_eff)
    assert HG.assert_identity()
    HG.install()
    model.hazard_g = float(g)
    model.hazard_sigma = float(sigma)
    model.hazard_gamma_effective = gamma_eff
    return dict(g=float(g), sigma=float(sigma), gamma_effective=gamma_eff,
                gamma_units=GAMMA_UNITS, z_shape=list(shape),
                installed=HG.is_installed())


def restore_hazard():
    HG.restore()
    return HG.is_installed()


def frozen_hash_report():
    """The 75-entry frozen map, restricted to the six files this round leans on."""
    watched = ('closures.py', 'tagged_transport.py', 'structure_model.py',
               'hf_model.py', 'temporal_model.py', 'campaign_model.py')
    out = {}
    for fold in ('C0', 'C2'):
        fh = read_json(PEER / 'reports/launch_by_fold' / (fold + '.json'))['frozen_hashes']
        out[fold] = dict(n_frozen=len(fh), watched={})
        for k, v in fh.items():
            base = k.replace('\\', '/').rsplit('/', 1)[-1]
            if base in watched:
                out[fold]['watched'][base] = dict(
                    registered=v, on_disk=sha(PEER / k), path=k,
                    unchanged=bool(sha(PEER / k) == v))
    return out
