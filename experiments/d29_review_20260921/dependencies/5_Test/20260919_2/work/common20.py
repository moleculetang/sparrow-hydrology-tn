"""Round 20260919_2 bootstrap: the frozen reference tree, and the runtime install
of the minimal inter-event availability kernel.

NOTHING IS WRITTEN INTO 20260916_2, AND NOTHING COVERED BY `frozen_hashes` IS EDITED
------------------------------------------------------------------------------------
`20260916_2/reports/launch_by_fold/{C0,C2}.json` each carry a 75-entry
`frozen_hashes` map.  Six of those entries are read and relied on here, with the
hashes verified in this round's Phase 0:

    vendor\\research\\closures.py          02433a200de4911f...   the frozen `scan`
    vendor\\research\\tagged_transport.py  b1a7df8a559e00a4...   the frozen `tag_scan`
    scripts\\structure_model.py            615e87b9fd2ec067...
    scripts\\hf_model.py                   88a85da13541e612...
    scripts\\temporal_model.py             1b3217cbec5260f0...
    scripts\\campaign_model.py             44bdb7c00b9ea572...

`work\\closures_r.py` is a NEW module holding a COPY of `scan`/`tag_scan` with the
one permitted change.  It is installed over the frozen classes by REBINDING module
attributes at runtime (the `apply_arrays` precedent in `20260919_1/work/common19.py`
-- a runtime override of attributes, with `scripts/structure_model.py`'s bytes
untouched), never by editing a file.  `identity()` reads bytes on disk, so a
process-local rebinding cannot reach it.

WHERE THE TWO CALL SITES ARE, AND WHY BOTH MUST BE COVERED
----------------------------------------------------------
The land-phase kernel is reached from exactly two places, and BOTH must see the
same kernel or the tagged source ledger stops matching the scalar one:

  forward   scripts\\structure_model.py:39   `Transport.apply(h,s,f,k,self)`
            -> closures.py:57-63 -> scan(h,s,f,k,owner.data.lower_release,
                                          owner.inp, owner.demand, owner.cap)
  ledger    scripts\\campaign_model.py:104   `tag_scan(h[:,rr], s[rr], f[rr], ...)`

The frozen call is `Transport.apply(h,s,f,k,self)`, so `forward` receives the MODEL
as `owner`; the new parameters therefore ride on the model object and no call site
has to change its signature.
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
assert ROUND.name == '20260919_2', ROUND
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
import closures_r as KR  # noqa: E402

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

# every module that binds the name `Transport` in its own globals.  `tagged_transport`
# is not among them -- it binds `TaggedTransport`, which this round does not touch.
TRANSPORT_MODULES = (_closed, _sci, _temporal, _hf, _structure, cm)

ANCHOR_TABLE = PEER / 'outputs/C0_s1/daily_station_mass_water.parquet'
ANCHOR_ROWS = 169476
ANCHOR_TOL = 1e-12
END_YEAR = 2024


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
# the frozen model, verbatim from 20260919_1/work/common19.py
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
    """The registered design with only the prediction registry substituted.

    The calendar is left at the registered `monthfirst`: this round holds the
    source calendar fixed and changes the land-phase kernel instead.
    """
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
    return model


# --------------------------------------------------------------------------
# install / restore the new kernel
# --------------------------------------------------------------------------
_ORIG = {m: m.Transport for m in TRANSPORT_MODULES}
_ORIG_TAG = cm.tag_scan
_ORIG_SCAN = _closed.scan


def install_kernel(model, delta, r_init=0.0):
    """Bind the new kernel into this process, and the parameters onto the model.

    THREE bindings, not one, because the land-phase recursion is reached three ways:

      `Transport.apply`   structure_model.py:39 / temporal_model.py:77   the forward replay
      `closures.scan`     closures.py:220                                the ledger's SCALAR branch
      `tag_scan`          campaign_model.py:104                          the ledger's TAGGED branch

    Missing the second would make `source_label_sum_errors` measure the cap instead
    of the tagged/scalar rounding it is registered to measure.  `delta = 1.0` is the
    bitwise no-op: closures_r documents it and this round's gates assert it with
    np.array_equal, not allclose.
    """
    delta = float(delta)
    r_init = float(r_init)
    KR.set_config(delta, r_init)
    model.R_delta = delta
    model.R_init = r_init
    for m in TRANSPORT_MODULES:
        m.Transport = KR.TransportR
    _closed.scan = KR.scan_r_ledger
    cm.tag_scan = KR.tag_scan_r
    return model


def restore_kernel():
    for m, T in _ORIG.items():
        m.Transport = T
    _closed.scan = _ORIG_SCAN
    cm.tag_scan = _ORIG_TAG


def ledger_with(model, x, delta, r_init=0.0):
    """`model.ledger(x)` under one (delta, r_init), with the config asserted.

    `campaign_model.Matched.ledger:104` calls the bare name `tag_scan`, which
    resolves through that module's globals, so the tagged ledger reads its
    parameters from closures_r's module config rather than from a call argument.
    The assertion below is what keeps that single source of truth honest.
    """
    install_kernel(model, delta, r_init)
    assert KR.CONFIG['delta'] == float(delta) and KR.CONFIG['r_init'] == float(r_init)
    return model.ledger(x)


# --------------------------------------------------------------------------
# the daily replay, byte-for-byte the arithmetic of
# 20260919_1/work/phase1_replay_calendar.py::forward
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
