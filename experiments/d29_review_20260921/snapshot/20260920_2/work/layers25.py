"""The three-layer forward and the per-layer budget, derived from
`20260920_1/work/layers24.py`.

DERIVED FROM `20260920_1\\work\\layers24.py`, WITH THE DELTA EXECUTABLE AND LISTED
------------------------------------------------------------------------------------
This is NOT a byte-identical copy and the paragraph does not claim one.  The deltas are:

  1. `import common24 as C` -> `import common25 as C`.
  2. `C.MC.scan_dp_full` -> `C.MC.scan_dp2_full`, and the unpack goes from four values to
     FIVE: `fast, slow, a, p` -> `fast, slow, a, p, pre`.  The fifth is `N^{M,pre}`, the
     mobile pool the mobilisation actually draws from; `closures_dp2.scan_dp2` returns it
     because no second spelling of the recursion may be introduced to recover it.
  3. `cu_cs`'s upper-store `X` changes from `a` to `pre`.  This is the ONE substantive
     change in this file and it is forced, not cosmetic: round 1 mobilised `E = a*g_u`,
     so `C_u` was built on `a`; this round mobilises `E_M = N^{M,pre}*g_u` with
     `N^{M,pre} <= a`, so leaving `X = a` would (i) report a concentration of a pool that
     is not the one mobilisation draws from, and (ii) trip this file's own bitwise pin
     `Eu*phi_f == fast` (that assertion is kept exactly as it was -- it is the check that
     caught this, and removing it would remove the reason the change is trustworthy).
     `C_s` is untouched: the slow store is unchanged.

  The plan (`20260920_2` S6, 交付物 table) described this file's delta as "only the
  provenance of `local = F_f + F_s + human_mass(t)`".  Reading `layers24.py` shows that is
  WRONG in two directions, and both are registered in `实际方法与偏离.md` rather than
  quietly fixed here: (a) `local`'s provenance does NOT change at all -- `SM.Transport`
  still resolves to this round's kernel and still returns `(F_f, F_s)`, so line
  `local = F_f_t + F_s_t + model.human_mass(t)` is copied verbatim; (b) the delta that DOES
  exist is in `cu_cs`, which the plan did not mention.

THIS PARAGRAPH QUOTES NO NUMBER ABOUT THIS FILE -- not its sha256, not its line count --
because every one of those changes the moment this paragraph changes.  The `diff` command
is named instead, because re-running it is how the claim above is checked.

WHAT THE FIVE-OUTPUT CALL DOES AND DOES NOT PROVE
-------------------------------------------------
Unchanged from round 1, and the warning still applies verbatim: `forward_layers` calls both
`SM.Transport.apply` and `MC.scan_dp2_full` on the same arguments and asserts the two pairs
of fluxes are `array_equal`.  Both enter the SAME `scan_dp2`, so this is NOT evidence that
anything is unmodified -- a same-process nesting check moves both sides together, and no
live comparison can prove additivity.  What it DOES prove is narrower and still worth
having: `Transport.apply`'s `torch.from_numpy` round-trip is lossless, the rebind really did
reach `structure_model`'s `SM.Transport`, and `a`/`p`/`N^{M,pre}` -- which the two-output
entry does not return -- come from the same call that produced `local`.

`L1` READS THE NEW KERNEL; `L3` ALSO READS A SECOND, FROZEN HAZARD
-----------------------------------------------------------------
`L1 = F_f + F_s + human_mass(t)` comes straight from `SM.Transport.apply`.  `L3` goes on
through `RiverN.apply` and `boundary_mass`, and `routing.py:130` computes
`mass = inlet*exp(-vf*h*f) + f*local*exp(-.5*vf*h*f)` with `c['h'] =
data.h_month[mid[ti],ri]` -- a river-phase hazard taken from the frozen monthly table.  So
`L3`'s concentration depends on BOTH the replaced land-phase mobilisation AND an untouched
monthly river hazard.  A flat `L3` response is therefore NOT evidence that the land-phase
replacement did nothing, and the report must say so.

THE TWO-POOL STATES ARE NOT READ HERE
-------------------------------------
`N^L`, `N^M` and `T^{mob}` are produced by `closures_dp2.ledger_dp2` and read by
`phase1_arms.py`.  This module does not touch them: `forward_layers` returns the
CONCENTRATION layers only, and adding a state trace here would put a second spelling of the
two-pool recurrence next to the kernel's.

NOTHING FROZEN IS EDITED.  Every primitive is CALLED through the module that owns it
(`SM.Transport.apply`, `SM.RiverN.apply`, `SM.boundary_mass`), so the frozen kernels run
exactly as registered.
"""
import numpy as np
import pandas as pd
import torch

import common25 as C
import eventlib as E
import structure_model as SM


# The exact column list of the DELIVERED per-arm table (`reports/daily_arms.parquet`),
# separated from the frame `forward_layers` returns.  Round 5's copy claimed the `*_kg`
# columns were delivered and they were not; see `layers24.py`'s module docstring.
DELIVERED_COLUMNS = ('station_key', 'date', 'water_m3_day', 'pL1', 'pL2', 'pL3')


def deliverable(df):
    """Drop the ledger columns and assert the result is criterion-clean.

    `eventlib.FORBIDDEN` contains `'kg'`, `'mass'` and `'load'`, so the drop is also
    checked by the same guard that protects every other criterion table rather than by a
    column list this module maintains by hand.
    """
    out = df[list(DELIVERED_COLUMNS)].copy()
    E.assert_no_mass_columns(out)
    return out


def forward_layers(model, tag, meta=None, installed=True):
    """Mirror of `StructureEndpoints.daily_boundary`, keeping the three layers.

    `installed=True` (the arms) requires `model.dp_fractions` to be installed and the
    kernel to be bound -- i.e. this must run under `common25.install_kernel`, which is what
    makes `SM.Transport` resolve to `TransportDP2`.  Asserted here rather than trusted,
    because running it with the frozen kernel still bound would produce a complete,
    plausible, WRONG table.

    `installed=False` (the `B0` anchor arm) runs the FROZEN kernel.

    Returns a frame of `169_476` station-days (the whole `prediction_calendar` up to
    `END_YEAR`), NOT the eligible grid.  `phase1_arms.py` writes that frame, concentration
    columns only, as `daily_dense_pL.parquet` -- which is what makes `A_L1` independently
    recomputable this round (round 1 registered it as a scope limit that it was not).
    """
    x = C.parameters(tag)
    if meta is None:
        meta = pd.read_parquet(C.PEER / 'data/prediction_calendar.parquet')
        meta = meta[meta.year.le(C.END_YEAR)].copy()
    if installed and not bool(getattr(model, 'dp_fractions', None)):
        raise SystemExit('DP_FRACTIONS_NOT_INSTALLED')
    extra = {}
    with torch.no_grad():
        t = torch.tensor(x)
        c, record, weights = model.daily_metadata(meta)
        h_, s_, f_, k_ = model.flux_parameters(t[:30])
        F_f_t, F_s_t = SM.Transport.apply(h_, s_, f_, k_, model)
        if installed:
            # the same call through the five-output entry, for `a`/`p`/`N^{M,pre}` and for
            # the round-trip check documented in the module docstring (NOT an additivity
            # proof)
            fast, slow, a, p, pre = C.MC.scan_dp2_full(h_, s_, f_, k_, model)
            if not (np.array_equal(fast, F_f_t.numpy())
                    and np.array_equal(slow, F_s_t.numpy())):
                raise SystemExit('TRANSPORT_AND_SCAN_DISAGREE')
            extra = dict(F_f=np.asarray(fast, float), F_s=np.asarray(slow, float),
                         a=np.asarray(a, float), gu=np.asarray(p, float),
                         pre=np.asarray(pre, float),
                         **cu_cs(model, fast, slow, pre, p))
        local = F_f_t + F_s_t + model.human_mass(t)
        inlet, official, releases = SM.RiverN.apply(local, t[2], model.daily_data,
                                                    'monthly', True)
        mass = SM.boundary_mass(inlet, official, releases, local, t[2], c)
    ti = c['ti'].numpy().astype(np.int64)
    ri = c['ri'].numpy().astype(np.int64)
    L = np.asarray(local.numpy(), float)
    I = np.asarray(inlet.numpy(), float)
    M = np.asarray(mass.numpy(), float)
    V = np.asarray(c['water'].numpy(), float)
    df = pd.DataFrame(dict(
        station_key=meta.station_key.to_numpy()[record.numpy()],
        date=model.data.dates[ti],
        water_m3_day=V,
        L1_kg=L[ti, ri],
        L2_kg=I[ti, ri] + L[ti, ri],
        L3_kg=M))
    for lay in ('L1', 'L2', 'L3'):
        df['p' + lay] = 1000.0 * df[lay + '_kg'] / df.water_m3_day
    # The `pL*` are the only columns a criterion reads, and they are checked here; the
    # `*_kg` columns stay on the RETURNED frame (the budget is a mass ledger, which the
    # round discipline exempts explicitly) and are removed by `deliverable()` before
    # anything is written out.
    E.assert_no_mass_columns(df[['station_key', 'date', 'pL1', 'pL2', 'pL3']])
    return df, dict(c=c, ti=ti, ri=ri, local=L, inlet=I, mass=M, water=V,
                    record=record, meta=meta, x=x, installed=bool(installed), **extra)


def slow_track(J, gs):
    """The slow store's `L_pre` / `L` tracks, in the KERNEL'S OWN ARITHMETIC ORDER.

    Day-sequential, reach-vectorised: `pre = L + J[t]`, `F_s = pre*g_s[t]`, `L = pre -
    F_s`.  numpy's elementwise ops are IEEE-deterministic, so looping over the day axis
    while vectorising the reach axis reproduces the njit body bitwise, at 23,376 numpy
    calls on 230-wide vectors rather than 5.4 million scalar steps.

    WHY NOT `np.cumsum(J - F_s) + F_s`, WHICH IS THE LEDGER'S INDEPENDENT SPELLING
    ----------------------------------------------------------------------------
    Because it is only ALGEBRAICALLY equal, not equal in float64.  Measured on the real
    arrays (`20260920_1/work/probe_lpre.py`): the cumsum form reproduces `L_pre*g_s`
    bitwise on 92,000/92,000 elements of a 400-day prefix and exactly nowhere else --
    `max relative residual 1.9e-13`, `median 2.9e-15`, `frac(residual == 0) = 0.021`.  The
    kernel does `L - L*g_s`; the cumsum re-associates it.  That difference is irrelevant to
    the N3' ledger (tolerance 1e-6 kg, and re-association is exactly what gives it teeth)
    and irrelevant physically, but it must not be silently baked into a reported
    concentration, so the reported `C_s` uses the exact track and the residual of the
    cumsum form is REPORTED alongside as a measured quantity.
    """
    nd = J.shape[0]
    Lpre = np.empty_like(J)
    Lrun = np.empty_like(J)
    L = np.zeros(J.shape[1])
    for t in range(nd):
        pre = L + J[t]
        F = pre * gs[t]
        Lpre[t] = pre
        L = pre - F
        Lrun[t] = L
    return Lpre, Lrun


def cu_cs(model, fast, slow, pre, p):
    """`C_u` and `C_s` in mg/L, per reach-day, in the STABLE form of plan S1.2.

        C = 1000 * X / ((area_ha * 10) * V) * phi(x)

    The denominator contains no `Q`, and `phi` comes from `dp_kernel.phi_of` rather than
    from a second spelling of the closure.  `X` is `N^{M,pre}` (the pool mobilisation draws
    from) for the upper store and `L_pre` (the pre-release slow-store content) for the slow
    store.

    BOTH `X` VALUES ARE PINNED TO THE KERNEL BITWISE before either reaches a report:
    `N^{M,pre}*g_u*phi_f` must equal `F_f` and `L_pre*g_s` must equal `F_s`, `array_equal`.
    A wrong `L_pre` would otherwise produce a plausible concentration that nothing checks,
    since `C_s` is a reported distribution and not a criterion.  The upper-store pin is the
    assertion that forced `X`'s change from round 1's `a`; see the module docstring.

    These are per REACH-day, not per station-day: the criteria read `pL1/pL2/pL3`, and this
    pair is the S4.2 distribution reading only.
    """
    d = model.data
    frac = model.dp_fractions
    Eu = np.asarray(pre, float) * np.asarray(p, float)
    slow = np.asarray(slow, float)
    if not np.array_equal(Eu * frac['phi_f'], np.asarray(fast, float)):
        raise SystemExit('PRE_GU_PHI_F_DOES_NOT_REPRODUCE_FAST')
    J = Eu * (1.0 - frac['phi_f'])
    Lpre, _ = slow_track(J, frac['gs'])
    if not np.array_equal(Lpre * frac['gs'], slow):
        raise SystemExit('L_PRE_GS_DOES_NOT_REPRODUCE_SLOW')
    area = np.asarray(d.area_ha, np.float64)[None, :] * C.XI.MM_PER_HA
    phi_u = C.XI.phi_of(frac['xu'], frac['form'])
    phi_s = C.XI.phi_of(frac['xs'], frac['form'])
    cum = np.cumsum(J - slow, axis=0) + slow
    resid = np.abs(cum * frac['gs'] - slow) / np.maximum(np.abs(slow), 1e-300)
    return dict(C_u=1000.0 * np.asarray(pre, float) / (area * frac['Vu']) * phi_u,
                C_s=1000.0 * Lpre / (area * frac['Vs']) * phi_s,
                phi_u=phi_u, phi_s=phi_s, Lpre=Lpre,
                cumsum_reassociation_max_rel=float(resid.max()),
                cumsum_reassociation_median_rel=float(np.median(resid)),
                cumsum_exact_frac=float(np.mean(resid == 0.0)))


def amp_of(layers, ev, lay):
    tbl = E.build_event_table(
        layers[['station_key', 'date', 'p' + lay]].rename(columns={'p' + lay: 'p'}), ev)
    fin = np.isfinite(tbl.c_base) & np.isfinite(tbl.c_peak)
    r = (tbl.c_peak / tbl.c_base)[fin]
    return tbl, fin, r


def layer_budget(layers, ev):
    """Per-layer C_base / C_peak / ratio on the frozen events."""
    out = {}
    for lay in ('L1', 'L2', 'L3'):
        tbl, fin, r = amp_of(layers, ev, lay)
        out[lay] = dict(
            n_events=int(fin.sum()),
            c_base_median=float(tbl.c_base[fin].median()),
            c_peak_median=float(tbl.c_peak[fin].median()),
            amp_ratio_median=float(r.median()),
            amp_ratio_mean=float(r.mean()),
            amp_ratio_sd=float(r.std()),
            n_peak_min=int(tbl.n_peak.min()), n_peak_max=int(tbl.n_peak.max()),
            n_base_unique=[int(v) for v in sorted(tbl.n_base.unique())])
    return out


def station_sd_gate(layers, mask, lays=('L1', 'L2', 'L3')):
    """|log(SD_pred_s / SD_obs_s)| per station, at BOTH ddof conventions.

    WHICH LAYER CARRIES THE GATE
    ---------------------------
    **L3.**  The registered statistic (`phase1_fingerprints.py:518`) takes `p` from an arm
    replay, and arm `P` is `monthfirst` -- the frozen calendar.  Checked rather than
    assumed: `phase1_replay_s1_monthfirst.parquet::p` against the frozen anchor
    `20260916_2/outputs/C0_s1/daily_station_mass_water.parquet::concentration_mg_l` agrees
    at `max|d| = 0` over all 169,476 rows, i.e. arm P IS the frozen kernel.  The model's `p`
    is the observation-operator output, which is this module's `pL3` (`pL1` is the station
    reach's own land-phase concentration, upstream of routing).  The first version of this
    function defaulted to `L1` and produced 1.3155 instead of the published 0.7941; all
    three layers are reported so that choice is visible.

    THE `ddof` INCONSISTENCY DISSOLVES HERE
    ---------------------------------------
    `eventlib.station_daily_sd:215` uses pandas' default `ddof=1`; the J2 criterion path in
    `20260919_1/work/phase1_fingerprints.py:518` uses `ddof=0`.  For THIS statistic the
    choice cannot matter: `sd_pred` and `sd_obs` are grouped over the same eligible days, so
    both carry the same `sqrt(n_s/(n_s-1))` factor and it cancels in `sd_pred/sd_obs`.  `e_s`
    is therefore ddof-invariant, which the two reported columns confirm by agreeing to all
    16 digits.  Both are still emitted, so the claim is checkable rather than asserted.

    THE REGISTERED BASIS
    --------------------
    `20260919_1/work/phase1_fingerprints.py:486-520` is the definition the stored
    `0.7941489653630477` was computed under, and it is explicit at :500:

        "per-station sd of the station-day MEAN TN, ddof=0, over the ELIGIBLE days"

    The panel is 4-HOURLY (57,284 rows over 12,152 eligible station-days), so the
    observations must be COLLAPSED TO STATION-DAY MEANS (:488) before the sd is taken.
    """
    m = mask[mask.eligible][['station_key', 'date']].copy()
    m['date'] = m.date.astype('datetime64[ns]')
    panel = pd.read_parquet(E.PANEL)
    panel = panel.assign(date=E.as_day(panel.monitoring_time.dt.tz_localize(None)))
    daily = (panel.groupby(['station_key', 'date'], as_index=False)['TN'].mean()
             .rename(columns={'TN': 'obs_tn'}))
    obs_e = m.merge(daily, on=['station_key', 'date'], how='left')
    n_missing = int(obs_e.obs_tn.isna().sum())
    if n_missing:
        raise SystemExit('ELIGIBLE_DAY_WITHOUT_OBSERVATION %d' % n_missing)
    out = dict(gating_layer='L3', n_obs_rows=int(len(obs_e)),
               n_panel_rows=int(len(panel)),
               panel_rows_per_station_day=float(len(panel) / max(1, len(daily))),
               obs_definition='per-station sd of the station-day MEAN TN, ddof=0, '
                              'over the ELIGIBLE days (phase1_fingerprints.py:500)',
               sd_obs={'ddof0': {str(k): float(v) for k, v in
                                 obs_e.groupby('station_key').obs_tn.std(ddof=0).items()},
                       'ddof1': {str(k): float(v) for k, v in
                                 obs_e.groupby('station_key').obs_tn.std(ddof=1).items()}})
    for lay in lays:
        cmp = m.merge(layers[['station_key', 'date', 'p' + lay]],
                      on=['station_key', 'date'], how='left', validate='one_to_one')
        if int(cmp['p' + lay].isna().sum()):
            raise SystemExit('ARM_DROPPED_AN_ELIGIBLE_DAY %s' % lay)
        node = {}
        for d in (0, 1):
            sm = pd.DataFrame(dict(
                obs=obs_e.groupby('station_key').obs_tn.std(ddof=d),
                mdl=cmp.groupby('station_key')['p' + lay].std(ddof=d))).dropna()
            sm = sm[(sm.obs > 0) & (sm.mdl > 0)]
            if len(sm) == 0:
                # The statistic is UNDEFINED here, not violated.  `Q0-zero` (q_m = 0) has no
                # legacy->mobile transfer at all, so the land-side output is identically zero
                # and every modelled sd is 0 -- `sm.mdl > 0` filters out all 15 stations.  The
                # boundary arm is a REGISTERED arm and its readings must land, so this emits a
                # typed empty node with the reason; it does not lower any threshold, and the
                # gate below reads `median_e is None` as NOT passed.
                node['ddof%d' % d] = dict(
                    n_stations=0, median_e=None, mean_e=None, stations={},
                    sd_pred={}, sd_obs={}, ratio_mdl_over_obs_median=None,
                    undefined_because='no station has a positive modelled sd on the eligible '
                                      'days (the land-side output is identically zero)')
                continue
            e = np.abs(np.log(sm.mdl / sm.obs))
            node['ddof%d' % d] = dict(
                n_stations=int(len(e)),
                median_e=float(e.median()), mean_e=float(e.mean()),
                stations={str(k): float(v) for k, v in e.items()},
                sd_pred={str(k): float(v) for k, v in sm.mdl.items()},
                sd_obs={str(k): float(v) for k, v in sm.obs.items()},
                ratio_mdl_over_obs_median=float((sm.mdl / sm.obs).median()))
        d0 = node['ddof0']
        d0['gate_threshold_70pct'] = (None if d0['median_e'] is None
                                      else 0.70 * d0['median_e'])
        kk = sorted(set(node['ddof0']['stations']) & set(node['ddof1']['stations']))
        # `sd_pred` must MOVE with ddof, otherwise the factor was never applied and the
        # invariance below would be a statement about nothing.
        dev = (max(abs(node['ddof0']['stations'][k] - node['ddof1']['stations'][k])
                   for k in kk) if kk else None)
        moved = (max(abs(node['ddof0']['sd_pred'][k] - node['ddof1']['sd_pred'][k])
                     for k in kk) if kk else None)
        node['ddof_note'] = ('sd_pred and sd_obs are grouped over the SAME eligible days, '
                            'so both carry sqrt(n_s/(n_s-1)) and it cancels in their '
                            'ratio; e_s is therefore ddof-invariant, up to rounding')
        node['ddof0_vs_ddof1_max_abs_diff'] = None if dev is None else float(dev)
        node['sd_pred_moves_with_ddof_max_abs_diff'] = None if moved is None else float(moved)
        node['ddof_invariant'] = None if dev is None else bool(dev <= 1e-12)
        node['ddof_choice_is_inert'] = (None if dev is None
                                        else bool(dev <= 1e-12 and moved > 0.0))
        node['statistic_is_defined'] = bool(node['ddof0']['n_stations'] > 0)
        out[lay] = node
    out['gating'] = out['L3']
    return out


def median_of_stored_e_s():
    """The SAME quantity as recomputed above, read from the artifact that recorded it.

    `20260919_1/reports/phase1_fingerprints.json` stores 15 per-station values under
    `J2_station_sd_log_distance.per_start.C0_s1.e_s_P`.  NO stored scalar holds their
    median, so the plan's registered `0.7941489653630477` is a DERIVED number; Phase 0
    re-derives it and reports both rather than trusting the prose.
    """
    fp = C.read_json(C.PEER_ROOT / '20260919_1/reports/phase1_fingerprints.json')
    node = fp['J2_station_sd_log_distance']['per_start']['C0_s1']['e_s_P']
    v = np.asarray([float(x) for x in node.values()], float) if isinstance(node, dict) \
        else np.asarray(node, float)
    return dict(source='20260919_1\\reports\\phase1_fingerprints.json'
                       '::J2_station_sd_log_distance.per_start.C0_s1.e_s_P',
                n=len(v), median=float(np.median(v)), mean=float(np.mean(v)),
                values=sorted(float(x) for x in v))
