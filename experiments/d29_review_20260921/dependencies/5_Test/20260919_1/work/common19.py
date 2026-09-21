"""Round 20260919_1 shared bootstrap and the two calendar constructions.

Nothing in this module writes into 20260916_2.  The reference round is imported,
never copied, so its registered model objects (and therefore its verified array
hashes and its ANCHOR table) are provably the same bytes this round replays
against.  A copy would silently re-open the question of which array set was used.

TWO CALENDAR CONSTRUCTIONS
--------------------------
The registered contrast already exists as two files:

  monthfirst      vendor/research/closures.py:138,142   self.inp[data.starts]=data.source
                                                        self.demand[data.starts]=data.crop
                  campaign_model.py:72                   tag_inputs[data.starts]=source_tags
  uniform_daily   scripts/structure_model.py:9,12,14,15  data.source[data.mid]/nd   (etc.)

They are the same formula written twice in two places, and the daily values enter
the land-phase recursion NONLINEARLY:

  vendor/research/closures.py:15-27 (scan, @njit cache=True)
      av = max(M[r] + inp[t,r] - demand[t,r], 0.)        <- clamp
      risk = h[t,r]/(av+k[r]); prob = -expm1(-min(risk,700.))
      E = av*prob; fast = E*f; pre = L + E*(1-f); slow = pre*l
      M = av*(1-prob)*survival; L = pre-slow

so `inp`/`demand` are not a linear injection that a monthly total would
determine.  That is the whole scientific content of the contrast, and it is also
why the two arms must be compared at IDENTICAL monthly totals (see below).

LAST-DAY RESIDUAL COMPENSATION  (plan S3.5, technical lock 1)
------------------------------------------------------------
uniform_daily as registered computes I_d = I_m/n_d for every day, and the
float64 sum of n_d identical copies is not I_m.  The compensated construction
puts the rounding residue on the last day:

  I_d = I_m/n_d (d < d_last),   I_{d_last} = I_m - sum_{d<d_last} I_d

BITWISE CLOSURE IS PROVABLE, NOT HOPED FOR.  Required: acc + (I_m - acc) == I_m
exactly, where acc is the ASCENDING SEQUENTIAL partial sum of the identical
copies.  With I_m = n*per and acc = fl((n-1)*per) computed by repeated addition,

    |I_m - acc| = |n*per - fl((n-1)*per)| <= approx per = I_m/n

so acc/I_m lies in [1/2, 1) for n >= 2 and Sterbenz' lemma makes fl(I_m - acc)
EXACT.  The final addition then returns the exact value I_m, which is
representable, so the sum closes exactly.  n == 1 has no "days before last" and
closes trivially.  The accumulation is done with np.add.accumulate (left to
right, byte-identical to a Python += loop) and the gate asserts with
np.array_equal, never allclose.

The residue itself is ~1e-15 relative; the reason to demand bitwise closure is
that the acceptance criterion is otherwise a function of the calendar, which is
the exact failure this round exists to remove.
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
assert ROUND.name == '20260919_1', ROUND
assert PEER.is_dir(), PEER

# numba must not deposit compiled-cache entries inside the read-only reference
# round.  campaign_model.py sets NUMBA_CACHE_DIR to PEER/work/numba when it is
# imported, so the config is pinned here before that import and re-pinned after.
CACHE = ROUND / 'work' / 'numba'
CACHE.mkdir(parents=True, exist_ok=True)
os.environ['NUMBA_CACHE_DIR'] = str(CACHE)
import numba.core.config as _nbcfg  # noqa: E402  (initialises config from the env above)

sys.path[:0] = [str(PEER / 'scripts'),
                str(PEER / 'vendor/expert/tn_challenge'),
                str(PEER / 'vendor/research'),
                str(PEER / 'vendor/transfer_research')]
import campaign_model as cm  # noqa: E402
import native_runtime as _rt  # noqa: E402

assert cm.RUN == PEER, (cm.RUN, PEER)
os.environ['NUMBA_CACHE_DIR'] = str(CACHE)
_nbcfg.CACHE_DIR = str(CACHE)

import torch  # noqa: E402

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

ARMS = ('C0_s0', 'C0_s1')
CALENDARS = ('monthfirst', 'uniform_daily')


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


def model_record(tag):
    return read_json(PEER / 'outputs' / tag / 'model.json')


def parameters(tag):
    rec = model_record(tag)
    x = np.asarray(rec['parameters'], float)
    assert len(x) == 30, (tag, len(x))
    return x


def design_for(tag, calendar=None):
    """The registered design verbatim, with only the named keys replaced.

    The prediction pass must read the prediction registry rather than the fold
    registry, which is what both reference replay templates do; that is the only
    edit besides the calendar, and it is asserted below.
    """
    design = json.loads(json.dumps(model_record(tag)['design']))
    before = dict(design)
    design['observation_registry_file'] = 'data/prediction_registry.json'
    design['observation_registry_hash'] = cm.sha(PEER / 'data/prediction_registry.json')
    if calendar is not None:
        design['structure'] = dict(design['structure'])
        design['structure']['calendar'] = calendar
    changed = {k for k in set(before) | set(design) if before.get(k) != design.get(k)}
    allowed = {'observation_registry_file', 'observation_registry_hash', 'structure'}
    assert changed <= allowed, changed
    if calendar is not None:
        assert design['structure']['calendar'] == calendar
        assert before['structure']['calendar'] == 'monthfirst', before['structure']
    return design


def build(tag, calendar='monthfirst'):
    """Load FULL24 and construct the registered model for one calendar."""
    design = design_for(tag, calendar)
    data = cm.load_data('FULL24')
    model = cm.make_model(data, None, 'D29_BE', design)
    assert model.calendar == calendar, model.calendar
    assert getattr(model, 'human_enabled') is False, 'human channel is not part of this contrast'
    return model


# --------------------------------------------------------------------------
# the two calendar constructions
# --------------------------------------------------------------------------

def _last_day_residual_per_month(monthly, mid, starts, stops):
    """Spread each month's total over its days, residue on the last day.

    Returns (nday, *trailing) float64, ascending-sequentially summing to
    `monthly` bitwise for every (month, trailing index).  See the module
    docstring for the Sterbenz argument.
    """
    monthly = np.asarray(monthly, dtype=np.float64)
    nday = len(mid)
    out = np.zeros((nday,) + monthly.shape[1:], dtype=np.float64)
    per = np.empty_like(monthly)
    n = (np.asarray(stops) - np.asarray(starts)).astype(np.int64)
    assert (n > 0).all()
    np.divide(monthly, n.reshape((-1,) + (1,) * (monthly.ndim - 1)), out=per)
    out[:] = per[mid]
    for m in range(len(starts)):
        k = int(n[m])
        if k == 1:
            continue
        # ascending sequential partial sum of the k-1 identical copies, exactly
        # the order np.add.accumulate uses and a Python += loop would use.
        block = np.broadcast_to(per[m], (k - 1,) + monthly.shape[1:])
        acc = np.add.accumulate(block, axis=0)[-1]
        out[int(stops[m]) - 1] = monthly[m] - acc
    return out


def uniform_daily_arrays(data, rr, compensated=True):
    """The four land-input arrays for uniform_daily.

    `rr` is the pilot-index vector; it is set on the DATA object by
    Matched.__init__ (campaign_model.py:71), not by load_data, so it must be
    taken from a constructed model rather than from the raw domain.

    compensated=False reproduces the registered arithmetic (I_d = I_m/n_d on
    every day); compensated=True is the construction this round adjudicates.
    Both are built here, in one place, so the two differ only in the last day.
    """
    mid = np.asarray(data.mid)
    starts = np.asarray(data.starts)
    stops = np.asarray(data.stops)
    rr = np.asarray(rr)
    if compensated:
        spread = _last_day_residual_per_month
    else:
        def spread(monthly, *a):
            per = np.empty_like(np.asarray(monthly, float))
            n = (np.asarray(stops) - np.asarray(starts)).astype(np.int64)
            np.divide(monthly, n.reshape((-1,) + (1,) * (np.asarray(monthly).ndim - 1)), out=per)
            return per[mid]
    inp = spread(np.asarray(data.source, float), mid, starts, stops)
    demand = spread(np.asarray(data.crop, float), mid, starts, stops)
    # tag_inputs/tag_demand are PILOT-RESTRICTED in BOTH registered
    # implementations -- Matched.__init__ (campaign_model.py:72) uses
    # data.source_tags[:, rr, :] and structure_model.py:14 uses
    # data.source_tags[data.mid][:, rr, :] -- so the trailing axis is len(rr),
    # not n_reach.  Building them full-width is not a different convention, it is
    # a shape error that surfaces inside tag_scan.
    tag_inputs = spread(np.asarray(data.source_tags, float)[:, rr, :], mid, starts, stops)
    tag_demand = np.ascontiguousarray(demand[:, rr])
    return dict(inp=np.ascontiguousarray(inp),
                demand=np.ascontiguousarray(demand),
                tag_inputs=np.ascontiguousarray(tag_inputs),
                tag_demand=tag_demand)


def apply_arrays(model, arrays):
    """Runtime override of the four land-input attributes, after construction.

    The registered files are NOT edited -- scripts/structure_model.py is in the
    75-entry frozen_hashes list of 20260916_2/reports/launch_by_fold/*.json and
    the reference round is read-only for this one.
    """
    for k, v in arrays.items():
        assert np.asarray(v).flags['C_CONTIGUOUS'], k
        setattr(model, k, v)
    return model


def monthly_reference(data, key, rr):
    """The monthly array a spread construction must reproduce, per key.

    `rr` is required because tag_inputs is pilot-restricted in the registered
    code; passing the full-width array here would compare a (230,4) block against
    a (2,4) one and only appears to work because both are 2-D.
    """
    rr = np.asarray(rr)
    return {'inp': np.asarray(data.source),
            'demand': np.asarray(data.crop),
            'tag_inputs': np.asarray(data.source_tags)[:, rr, :],
            'tag_demand': np.asarray(data.crop)[:, rr]}[key]


def monthly_closure(data, arrays, key, rr):
    """Ascending sequential day-sum of one spread array, per month.

    Returns (worst_abs_error, n_months_not_bitwise_equal) against the monthly
    reference, using np.array_equal -- the plan's requirement, not allclose.
    """
    starts = np.asarray(data.starts)
    stops = np.asarray(data.stops)
    v = np.asarray(arrays[key], float)
    ref_all = monthly_reference(data, key, rr)
    if v.shape[1:] != ref_all.shape[1:]:
        raise ValueError('SHAPE_MISMATCH %s %s %s' % (key, v.shape, ref_all.shape))
    worst = 0.0
    bad = 0
    for m in range(len(starts)):
        acc = np.add.accumulate(v[int(starts[m]):int(stops[m])], axis=0)[-1]
        ref = ref_all[m]
        if not np.array_equal(acc, ref):
            bad += 1
            worst = max(worst, float(np.max(np.abs(acc - ref))))
    return worst, bad
