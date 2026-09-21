"""Freeze the observation-side and baseline-side anchors BEFORE the envelope.

WHY THIS RUNS BEFORE THE PRE-REGISTRATION IS HASHED
---------------------------------------------------
`reports/预注册_判据与门槛.md` must be frozen before any envelope point is read
(the approved plan, section 6).  Two of its thresholds, however, cannot be quoted
from an earlier round because no earlier round stored them:

    C_base,obs  and  C_peak,obs   -- the observation's OWN baseline and peak
    concentration levels, per event, on the frozen windows

`20260919_1` stored only the RATIO (`obs_ratio` / `obs_amp_ratio`).  A criterion
of the form "|C_peak - C_peak,obs| must not increase" needs both LEVELS, so they
are computed here, from the frozen raw panel and the frozen event set, and then
carried into the pre-registration as frozen numbers.

WHAT THIS SCRIPT DOES *NOT* DO
------------------------------
It reads no envelope point.  `delta = 1` is the null point of the new kernel, and
the new kernel at `delta = 1` was proven BITWISE equal to the frozen kernel by
`work/gate_noop.py` gates G2/G3/G4 (`np.array_equal`, not `allclose`).  The
baseline model anchors below are therefore taken from `20260919_1`'s PUBLISHED
event table -- they are prior-round constants, not new readings.

The 4h TN panel is used here strictly as the registered read-only magnitude /
independent-check object.  It is never a fit target; this round has `n_fits = 0`.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

UP = Path(r'E:\SPARROW\5_Test')
R = UP / '20260919_2'
D4 = UP / '20260918_4'
EVENTS = D4 / 'data/stage_a_events.parquet'
PANEL = UP / '20260918_1/reports/ammonia_TN_quality_only.parquet'
MASK = UP / '20260919_1/data/phase1_eligible_mask.parquet'
BASE_TABLE = UP / '20260919_1/data/phase1_event_table.parquet'
FINGERPRINTS = D4 / 'reports/stage_a_fingerprints.json'

PRIMARY_ARM = 'A:routed:D3'
TN_PRE_DAYS = 7
MASK_SHA_EXPECTED = '872280128ed6261ca936544379d54e63800497e0ee66625bf1cf5b83c310af41'
EVENT_DEF_SHA_EXPECTED = '7a1b173086012366'


def sha(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(4 << 20), b''):
            h.update(b)
    return h.hexdigest()


def obs_levels(ev):
    """Per event, the observation's own C_base (median of the pre-window) and
    C_peak (max of the peak window), on the FROZEN windows.

    This is `20260919_1/work/phase1_fingerprints.py::obs_recipe_reproduce` with
    the two levels kept instead of collapsed into their ratio.  The ratio it
    returns must still reproduce the lineage's `obs_ratio` column exactly; that
    identity is the check that the levels come from the same recipe.
    """
    p = pd.read_parquet(PANEL)
    p = p.assign(d=p.monitoring_time.dt.tz_localize(None).values.astype('datetime64[D]'))
    ts = pd.to_datetime(ev.t_start).dt.normalize()
    te = pd.to_datetime(ev.t_end).dt.normalize()
    n = len(ev)
    cb = np.full(n, np.nan)
    cp = np.full(n, np.nan)
    nb = np.zeros(n, int)
    npk = np.zeros(n, int)
    for st, sub in p.groupby('station_key'):
        s = sub.sort_values('d')
        d = np.asarray(s.d.values, dtype='datetime64[D]')
        v = np.asarray(s.TN, float)
        for i in np.where(ev.station_key.values == st)[0]:
            lo = np.datetime64(ts.iloc[i] - pd.Timedelta(days=TN_PRE_DAYS), 'D')
            t0 = np.datetime64(ts.iloc[i], 'D')
            hi = np.datetime64(te.iloc[i] + pd.Timedelta(days=1), 'D')
            pre = (d >= lo) & (d < t0)
            win = (d >= t0) & (d <= hi)
            if not pre.sum():
                continue
            cb[i] = float(np.median(v[pre]))
            nb[i] = int(pre.sum())
            if win.sum():
                cp[i] = float(np.max(v[win]))
                npk[i] = int(win.sum())
    ratio = np.where(cb > 0, cp / cb, np.nan)
    ref = ev.obs_ratio.to_numpy(float)
    ok = np.isfinite(ratio) & np.isfinite(ref)
    return dict(c_base=cb, c_peak=cp, n_base=nb, n_peak=npk, ratio=ratio,
                reproduce=dict(n=int(ok.sum()), n_expected=int(np.isfinite(ref).sum()),
                               max_abs_diff=float(np.max(np.abs(ratio[ok] - ref[ok])))
                               if ok.sum() else None))


def main():
    S = dict(zero_fits=True, n_fits=0, fit_worker_calls=0, not_a_fit=True,
             purpose='frozen anchors for reports/预注册_判据与门槛.md')
    S['input_shas'] = dict(events=sha(EVENTS), panel=sha(PANEL), mask=sha(MASK),
                           base_event_table=sha(BASE_TABLE))
    if S['input_shas']['mask'] != MASK_SHA_EXPECTED:
        raise SystemExit('ELIGIBLE_MASK_CHANGED ' + S['input_shas']['mask'])

    ev_all = pd.read_parquet(EVENTS)
    ev = ev_all[(ev_all.arm == PRIMARY_ARM) & (ev_all.obs_status == 'OK')].copy()
    ev = ev.sort_values(['station_key', 'event_id']).reset_index(drop=True)
    S['eligible_set'] = dict(n_events=int(len(ev)), n_stations=int(ev.station_key.nunique()),
                             rule="arm == 'A:routed:D3' and obs_status == 'OK'",
                             definition_sha=EVENT_DEF_SHA_EXPECTED)
    if len(ev) != 214 or ev.station_key.nunique() != 15:
        raise SystemExit('ELIGIBLE_EVENT_COUNT_CHANGED %d %d'
                         % (len(ev), ev.station_key.nunique()))

    # ---- the observation anchor ------------------------------------------
    ol = obs_levels(ev)
    rep = ol.pop('reproduce')
    S['obs_recipe_reproduce'] = rep
    if rep['max_abs_diff'] is None or rep['max_abs_diff'] > 0.0:
        raise SystemExit('OBS_RECIPE_DOES_NOT_REPRODUCE ' + json.dumps(rep))
    fin = np.isfinite(ol['c_base']) & np.isfinite(ol['c_peak'])
    S['obs_levels'] = dict(
        n_events_with_levels=int(fin.sum()),
        c_base_median=float(np.median(ol['c_base'][fin])),
        c_base_mean=float(np.mean(ol['c_base'][fin])),
        c_base_q25=float(np.percentile(ol['c_base'][fin], 25)),
        c_base_q75=float(np.percentile(ol['c_base'][fin], 75)),
        c_peak_median=float(np.median(ol['c_peak'][fin])),
        c_peak_mean=float(np.mean(ol['c_peak'][fin])),
        c_peak_q25=float(np.percentile(ol['c_peak'][fin], 25)),
        c_peak_q75=float(np.percentile(ol['c_peak'][fin], 75)),
        median_ratio=float(np.median(ol['ratio'][fin])),
        units='mg/L, the panel is read-only and is never a fit target')
    S['obs_levels']['obs_amp_ratio_column_median'] = float(np.median(ev.obs_ratio.to_numpy(float)))

    # ---- the baseline (delta = 1) model anchor, quoted from the prior round --
    bt = pd.read_parquet(BASE_TABLE)
    b = bt[(bt.start == 'C0_s1') & (bt.calendar == 'monthfirst')].copy()
    b = b.sort_values(['station_key', 'event_id']).reset_index(drop=True)
    if len(b) != len(ev) or not np.array_equal(b.event_id.to_numpy(), ev.event_id.to_numpy()):
        raise SystemExit('BASE_TABLE_NOT_ALIGNED_WITH_THE_FROZEN_EVENT_SET')
    S['baseline_model_C0_s1_monthfirst'] = dict(
        source='20260919_1\\data\\phase1_event_table.parquet',
        n_events=int(len(b)),
        c_base_median=float(b.c_base.median()), c_peak_median=float(b.c_peak.median()),
        dc_median=float(b.dc.median()),
        amp_ratio_median=float(b.event_amp_ratio_sim.median()),
        obs_ratio_median=float(b.obs_amp_ratio.median()))
    S['anchors_used_by_the_criteria'] = dict(
        beta_obs=0.004063600875414098,
        alpha_obs=-0.0704045722214265,
        median_obs_ratio=float(np.median(ev.obs_ratio.to_numpy(float))),
        median_obs_ratio_note='frozen value 1.2758737517831669')
    print(json.dumps({k: v for k, v in S.items()
                      if k in ('eligible_set', 'obs_levels', 'baseline_model_C0_s1_monthfirst',
                               'anchors_used_by_the_criteria', 'obs_recipe_reproduce')},
                     indent=1, ensure_ascii=False), flush=True)
    out = R / 'reports/frozen_anchors.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(S, indent=1, sort_keys=True, ensure_ascii=False, default=str),
                   encoding='utf-8')
    print('ANCHORS_FROZEN', sha(out), flush=True)


if __name__ == '__main__':
    main()
