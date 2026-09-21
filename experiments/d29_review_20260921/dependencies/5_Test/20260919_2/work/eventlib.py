"""The calendar-neutral event/fingerprint library, COPIED from `20260919_1`.

SOURCE: `20260919_1/work/phase1_fingerprints.py`, lines 84-353 -- the block that
carries no calendar dependence.  The approved plan (section 10) names exactly this
range as reusable.  It cannot be imported: that module does `import common19 as C`
at line 60 and `common19` asserts `ROUND.name == '20260919_1'`, so importing it
from this round aborts on the assertion.

WHAT CHANGED IN THE COPY -- labels only, never arithmetic
--------------------------------------------------------
  * the import is `common20`, and the two frozen input paths are re-rooted;
  * `paired_cluster_boot` / `paired_event_boot` are byte-for-byte the same bodies,
    but their docstrings said "the calendar CONTRAST" and "BOTH ARMS".  This round
    contrasts **baseline vs candidate**, so `dfP`/`dfU` are read as (baseline,
    candidate).  The resamplers were written to take an arbitrary pair of frames
    on one common station multiset, which is exactly why no arithmetic had to move.

Everything else -- `base_peak`'s four pinned `side` arguments, `f1_coef`'s
within-station demeaning, `f3_index`'s consecutive-pair construction, the two
resampling units, `sign_flip_exact`'s enumerated null -- is unchanged.

WINDOW LENGTH IS A SPECIFICATION ASSERTION, NOT A NUMERICAL ONE
---------------------------------------------------------------
`build_event_table` asserts `n_peak == (t_end - t_start).days + 2` and
`n_base == TN_PRE_DAYS`.  `20260919_1`'s B1 defect (a `searchsorted` defaulting to
`side='left'` swallowed the `+1` day, searching 6 days where the observation
searched 7) was caught by NOTHING ELSE: both implementations made the same error,
so no consistency check could see it.  The assertion is kept.
"""
from pathlib import Path

import numpy as np
import pandas as pd

UP = Path(r'E:\SPARROW\5_Test')
R = UP / '20260919_2'
D4 = UP / '20260918_4'
EVENTS = D4 / 'data/stage_a_events.parquet'
PANEL = UP / '20260918_1/reports/ammonia_TN_quality_only.parquet'
MASK = UP / '20260919_1/data/phase1_eligible_mask.parquet'
ANCHOR_TABLE = UP / '20260916_2/outputs/C0_s1/daily_station_mass_water.parquet'

MASK_SHA = '872280128ed6261ca936544379d54e63800497e0ee66625bf1cf5b83c310af41'

PRIMARY_ARM = 'A:routed:D3'
TN_PRE_DAYS = 7                     # `_5`'s constant, unchanged
F3_GAPS = (15, 30, 60)
F3_PRIMARY_GAP = 30
MIN_UNITS = 12
N_BOOT = 2000
SEED = 20260917
NOISE_BAND = 0.0529                 # 2 x the `_5` amplitude-ratio null SD (0.02647)
J5_MAX_DEGRADATION = 0.005
START_YEAR, END_YEAR = 2021, 2024
FORBIDDEN = ('mass', 'load', 'kg', 'yield', 'flux', 'tonne')


def assert_no_mass_columns(frame):
    hits = sorted({c for c in frame.columns for t in FORBIDDEN if t in c.lower()})
    if hits:
        raise SystemExit('MASS_COLUMN_IN_OUTPUT ' + str(hits))
    return True


def as_day(s):
    """One datetime resolution for every join.  The mask, the panel and the
    replays are written by three producers and carry ns / ms / us."""
    return pd.to_datetime(s).dt.normalize().astype('datetime64[ns]')


def eligible_events():
    """The frozen eligible event set: arm A:routed:D3 AND obs_status == OK."""
    ev = pd.read_parquet(EVENTS)
    ev = ev[(ev.arm == PRIMARY_ARM) & (ev.obs_status == 'OK')].copy()
    ev = ev.sort_values(['station_key', 'event_id']).reset_index(drop=True)
    if len(ev) != 214 or ev.station_key.nunique() != 15:
        raise SystemExit('ELIGIBLE_EVENT_COUNT_CHANGED %d %d'
                         % (len(ev), ev.station_key.nunique()))
    return ev


# --------------------------------------------------------------------------
# the lineage's two estimators, copied verbatim except for the column names
# --------------------------------------------------------------------------
def f1_coef(d, xcol):
    """within-station  dC ~ xcol + C_base.  Returns the xcol coefficient.

    Byte-for-byte the arithmetic of 20260918_4/work/stage_a_fingerprints.py:180
    with `obs_delta_tn -> dc` and `obs_tn_base -> c_base`.
    """
    d = d[['station_key', 'dc', xcol, 'c_base']].dropna().copy()
    if len(d) < MIN_UNITS:
        return None
    cnt = d.station_key.map(d.station_key.value_counts())
    d = d[cnt >= 2]
    if d.station_key.nunique() < 3 or len(d) < MIN_UNITS:
        return None
    for c in ('dc', xcol, 'c_base'):
        d[c] = d[c] - d.groupby('station_key')[c].transform('mean')
    y = d.dc.to_numpy(float)
    X = np.column_stack([np.ones(len(d)), d[xcol].to_numpy(float),
                         d.c_base.to_numpy(float)])
    if np.linalg.matrix_rank(X) < X.shape[1]:
        return None
    return float(np.linalg.lstsq(X, y, rcond=None)[0][1])


def f3_index(d, maxgap):
    """consecutive-event pairs within `maxgap` days, per station (lineage:403)."""
    d = d[['station_key', 'event_rank', 'c_peak', 'c_base',
           'q_peak', 'T_interevent']].dropna()
    d = d[(d.c_base > 0) & (d.c_peak > 0) & (d.q_peak > 0)]
    ys, xs = [], []
    for _st, sub in d.sort_values(['station_key', 'event_rank']).groupby('station_key'):
        prev = None
        for r in sub.itertuples():
            if prev is not None and r.T_interevent <= maxgap:
                ys.append(np.log(r.c_peak / r.c_base) - prev[0])
                xs.append(np.log(r.q_peak / prev[1]))
            prev = (np.log(r.c_peak / r.c_base), r.q_peak)
    return np.asarray(ys, float), np.asarray(xs, float)


def f3_intercept(d, maxgap):
    y, x = f3_index(d, maxgap)
    if len(y) < MIN_UNITS:
        return None
    X = np.column_stack([np.ones(len(y)), x])
    return float(np.linalg.lstsq(X, y, rcond=None)[0][0])


# --------------------------------------------------------------------------
# the observation recipe, applied to the model's daily series
# --------------------------------------------------------------------------
def event_windows(ev):
    """The frozen recipe's two date windows, per event."""
    ts = pd.to_datetime(ev.t_start).dt.normalize()
    te = pd.to_datetime(ev.t_end).dt.normalize()
    return ts, te


def base_peak(series, dates, ts, te):
    """C_base / C_peak for one event on one (station, day) series.

    `series` is indexed by `dates` (datetime64[D], sorted, unique).  The windows
    are the frozen recipe's:

        base  [t_start - 7d, t_start)      half-open
        peak  [t_start, t_end + 1d]        BOTH ENDS INCLUSIVE

    The peak window's upper bound is INCLUSIVE and is one day past `t_end`; the
    observation recipe says so explicitly (`stage_a_events.py:145`,
    `win = (tnd >= t_start) & (tnd <= np.datetime64(t_end) + np.timedelta64(1,'D'))`)
    because the 4h record lags the daily hydrology window.  A searchsorted with
    the default side='left' on `t_end + 1d` returns the first day AT OR AFTER that
    date, so slicing `[j0:j1]` silently DROPS the lag day and searches a 6-day
    window where the observation searched 7 -- which lowers C_peak, lowers the
    amplitude ratio, and biases every event statistic downward.  Hence side is
    pinned on all four bounds, and `n_peak` is asserted against window_days + 1
    by the caller.
    """
    lo = np.datetime64(ts - pd.Timedelta(days=TN_PRE_DAYS), 'D')
    t0 = np.datetime64(ts, 'D')
    hi = np.datetime64(te + pd.Timedelta(days=1), 'D')
    b = series[np.searchsorted(dates, lo, side='left'):
               np.searchsorted(dates, t0, side='left')]
    p = series[np.searchsorted(dates, t0, side='left'):
               np.searchsorted(dates, hi, side='right')]
    b = b[np.isfinite(b)]
    p = p[np.isfinite(p)]
    c_base = float(np.median(b)) if len(b) else np.nan
    c_peak = float(np.max(p)) if len(p) else np.nan
    return c_base, c_peak, int(len(b)), int(len(p))


def build_event_table(replay, ev):
    """One arm's per-event C_base / C_peak / dC / ratio, on the frozen events.

    `replay` is any frame with `station_key`, `date` and `p`.
    """
    piv = replay.pivot_table(index='date', columns='station_key', values='p')
    piv = piv.sort_index()
    dates = piv.index.values.astype('datetime64[D]')
    ts, te = event_windows(ev)
    rows = []
    for i, r in enumerate(ev.itertuples()):
        if r.station_key not in piv.columns:
            raise SystemExit('REPLAY_MISSING_STATION ' + str(r.station_key))
        v = piv[r.station_key].to_numpy(float)
        cb, cp, nb, npk = base_peak(v, dates, ts.iloc[i], te.iloc[i])
        rows.append(dict(station_key=r.station_key, event_id=r.event_id,
                         event_rank=int(r.event_rank), t_start=r.t_start, t_end=r.t_end,
                         T_interevent=float(r.T_interevent), q_peak=float(r.q_peak),
                         c_base=cb, c_peak=cp, dc=cp - cb,
                         amp_ratio=cp / cb if cb > 0 else np.nan,
                         obs_amp_ratio=float(r.obs_ratio), n_base=nb, n_peak=npk))
    t = pd.DataFrame(rows).sort_values(['station_key', 'event_id']).reset_index(drop=True)
    assert_no_mass_columns(t)
    # The recipe's peak window is [t_start, t_end + 1d] INCLUSIVE, so it is exactly
    # one day longer than the event window; the base window is exactly TN_PRE_DAYS.
    # Every day of 2021-2024 is present in the replay, so the counts are exact.
    want_peak = ((pd.to_datetime(ev.t_end).dt.normalize()
                  - pd.to_datetime(ev.t_start).dt.normalize()).dt.days + 2).to_numpy()
    got = t.set_index('event_id').n_peak.sort_index()
    exp = pd.Series(want_peak, index=ev.event_id.to_numpy()).sort_index()
    if not np.array_equal(got.to_numpy(), exp.to_numpy()):
        raise SystemExit('PEAK_WINDOW_LENGTH_WRONG %s'
                         % [int(got.iloc[i]) for i in range(min(5, len(got)))])
    gotb = t.set_index('event_id').n_base.sort_index()
    if not np.array_equal(gotb.to_numpy(), np.full(len(ev), TN_PRE_DAYS)):
        raise SystemExit('BASE_WINDOW_LENGTH_WRONG')
    return t


def station_daily_sd(replay, mask):
    """Per-station sd of the daily concentration over the eligible days.

    Only used as a LEVEL diagnostic (the plan's 2.1 L1/L2/L3 layer comparison and
    the `14/15 stations below the observation` reading); it is never a criterion.
    """
    m = mask[mask.eligible][['station_key', 'date']].copy()
    m['date'] = as_day(m.date)
    r = replay.copy()
    r['date'] = as_day(r.date)
    j = r.merge(m, on=['station_key', 'date'], how='inner')
    g = j.groupby('station_key').p
    return pd.DataFrame(dict(sd=g.std(), mean=g.mean(), n=g.size())).reset_index()


def obs_recipe_reproduce(ev):
    """Recompute obs_ratio from the RAW 4h panel with the frozen windows.

    This is the check that the model-side mirror below is the same recipe: if the
    date windows and the median/max convention are reproduced here against the
    lineage's own column, the mirror is the same arithmetic on a different series.
    """
    p = pd.read_parquet(PANEL)
    p = p.assign(d=p.monitoring_time.dt.tz_localize(None).values.astype('datetime64[D]'))
    ts, te = event_windows(ev)
    out = np.full(len(ev), np.nan)
    for st, sub in p.groupby('station_key'):
        s = sub.sort_values('d')
        d = np.asarray(s.d.values, dtype='datetime64[D]')
        v = np.asarray(s.TN, float)
        m = np.where(ev.station_key.values == st)[0]
        for i in m:
            lo = np.datetime64(ts.iloc[i] - pd.Timedelta(days=TN_PRE_DAYS), 'D')
            pre = (d >= lo) & (d < np.datetime64(ts.iloc[i], 'D'))
            win = (d >= np.datetime64(ts.iloc[i], 'D')) & \
                  (d <= np.datetime64(te.iloc[i] + pd.Timedelta(days=1), 'D'))
            if not pre.sum():
                continue
            b = float(np.median(v[pre]))
            if win.sum() and b > 0:
                out[i] = float(np.max(v[win]) / b)
    ref = ev.obs_ratio.to_numpy(float)
    ok = np.isfinite(out) & np.isfinite(ref)
    return dict(n=int(ok.sum()), n_expected=int(np.isfinite(ref).sum()),
                max_abs_diff=float(np.max(np.abs(out[ok] - ref[ok]))) if ok.sum() else None,
                note=('recomputed from the raw 4h panel, compared with the lineage column '
                      'stage_a_events.parquet::obs_ratio'))


# --------------------------------------------------------------------------
# resampling -- NEVER by row
# --------------------------------------------------------------------------
def _codes(df, key='station_key'):
    codes, uniq = pd.factorize(df[key].values)
    return codes, [np.where(codes == i)[0] for i in range(len(uniq))]


def paired_cluster_boot(dfP, dfU, fn, n_boot=N_BOOT, seed=SEED):
    """Resample STATIONS with replacement, take all their events, in BOTH frames.

    The same drawn station multiset is applied to the baseline and the candidate,
    so the two are paired through the resampling and the resulting distribution is
    of the CONTRAST, not of two independent arms.
    """
    _, idx = _codes(dfP)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(idx), len(idx))
        rows = np.concatenate([idx[i] for i in pick])
        v = fn(dfP.iloc[rows], dfU.iloc[rows])
        if v is not None and np.isfinite(v):
            vals.append(float(v))
    return np.asarray(vals, float), len(idx)


def paired_event_boot(dfP, dfU, fn, n_boot=N_BOOT, seed=SEED + 1):
    """Sensitivity arm: resample EVENTS with replacement (not by row-of-a-frame)."""
    rng = np.random.default_rng(seed)
    n = len(dfP)
    vals = []
    for _ in range(n_boot):
        rows = rng.integers(0, n, n)
        v = fn(dfP.iloc[rows], dfU.iloc[rows])
        if v is not None and np.isfinite(v):
            vals.append(float(v))
    return np.asarray(vals, float), n


def boot_summary(vals, point, want):
    """want: +1 the statistic must be positive, -1 negative, for `holds`."""
    if len(vals) < 100:
        return dict(status='BOOTSTRAP_INSUFFICIENT', n_boot=int(len(vals)))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return dict(status='OK', n_boot=int(len(vals)), point=float(point),
                mean=float(vals.mean()), ci95=[float(lo), float(hi)],
                excludes_zero=bool(lo > 0 or hi < 0),
                p_two_sided=float(min(max(2.0 * min(float(np.mean(vals <= 0)),
                                                     float(np.mean(vals >= 0))),
                                         1.0 / len(vals)), 1.0)),
                holds=bool((lo > 0 or hi < 0) and np.sign(point) == want))


def two_units_agree(boot_entry, want):
    """Pre-registration S3: the verdict takes the CONSERVATIVE combination of the
    two resampling units, and they must agree.  Conservative = both the
    station-cluster reading and the event-level sensitivity must show the effect
    with the predicted sign and an interval off zero."""
    sc = boot_entry.get('station_cluster', {})
    el = boot_entry.get('event_level', {})
    ok = (sc.get('status') == 'OK' and el.get('status') == 'OK'
          and bool(sc.get('holds')) and bool(el.get('holds')))
    return dict(station_cluster_holds=bool(sc.get('holds')),
                event_level_holds=bool(el.get('holds')),
                agree=bool(ok), want_sign=int(want))


# --------------------------------------------------------------------------
# J2's exact paired sign-flip permutation
# --------------------------------------------------------------------------
def sign_flip_exact(delta):
    """All 2**n sign flips of the paired per-station differences.

    n = 15 stations, so 32768 assignments: the null is enumerated, not sampled.
    """
    d = np.asarray(delta, float)
    n = len(d)
    if n > 20:
        raise SystemExit('SIGN_FLIP_TOO_MANY_STATIONS %d' % n)
    stat = float(np.mean(d))
    signs = ((np.arange(1 << n)[:, None] >> np.arange(n)) & 1) * 2 - 1
    null = (signs * d).mean(axis=1)
    p = float(np.mean(np.abs(null) >= abs(stat) - 1e-15))
    return dict(n_stations=int(n), statistic=stat, n_assignments=int(1 << n),
                p_two_sided=p, null_sd=float(null.std()),
                null_ci95=[float(v) for v in np.percentile(null, [2.5, 97.5])])
