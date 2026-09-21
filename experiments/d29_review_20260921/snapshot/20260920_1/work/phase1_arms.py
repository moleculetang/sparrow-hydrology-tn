"""`20260920_1` -- section 4: the capability envelope, FIVE forwards in one pass.

    B0        frozen kernel, NOT installed  (the anchor arm)
    P-upper   PRIMARY, V_u = upper_water                        (frozen by Phase 0)
    S-soil    V_u = soil_water_mm                          (sensitivity)
    S-unsat   V_u = soil_water_mm + upper_water            (sensitivity)
    D-const   V_u = the primary arm's 1961-2020 per-reach mean   (time-structure diagnostic)
    R5-ref    READ from `20260919_5/reports/daily_layers.parquet`, device='N1e', beta=0.5

Six arms, five forwards, both frozen in `reports/arms.json` BEFORE any forward ran
(`forward_runs_so_far: 0` in `预注册_冻结.json`).

NO EARLY STOP.  Section 4.1 forbids one, and the reason is on the record: round 5's own
emergency brake fired on a device that was merely flat, and a round that stops early
reports a distribution it never measured.

PER-ARM SEQUENCE (section 4.1, verbatim)
    install kernel -> `layers24.forward_layers` -> `measure` -> N3 ledger -> `restore_kernel`
`B0` is the one arm that skips the install, and `layers24.forward_layers(installed=False)`
is what keeps it on the FROZEN kernel rather than on a silently unbound one.

WHAT IS A STOP AND WHAT IS ONLY A READING
-----------------------------------------
Stops, all from section 3.5: N3 (absolute mass closure, `|balance| <= 1e-6 kg`), N10 (the
inherited numerical items, six label channels, the anchor replay), N11 ((i) the shape
assertions inside every wrapper -- those live in `closures_dp`/`dp_kernel` and are asserted
by running them; (ii) the ledger/forward re-proof done here).  Everything else is a
READING: N4's saturation census, N6, N7, N8, P1-P4, all of section 4.2, and `R5-ref`.

`R5-ref` enters NO gate.  A round-5 point is a different kernel on a different arm table;
reading it is how a drift becomes visible, never how a criterion is met (section 5).

TWO NAMES THIS FILE DOES NOT TRUST
----------------------------------
* `Vu_post` in `arms.json` is the PRE-outflow carry (`Vu_at_outflow = "Vu_post + Qu"`).
  Read as "after outflow" it would be wrong by a whole day's flux.  The field name is
  frozen and is not renamed; the trap is registered in `实际方法与偏离.md`.
* `B0`'s `p` is the frozen kernel's `prob`, i.e. `g_u` in the old spelling -- NOT a water
  fraction.  So `1 - p` is not a retained-water share and `B0` gets no lifetimes.

WHY `model.dp_fractions` IS CLEARED AFTER EVERY ARM
--------------------------------------------------
`common24.restore_kernel` rebinds the six modules and the ledger class attribute, but it
takes no model, so `model.dp_fractions` SURVIVES it.  Five sequential forwards in one
process then leave a live hazard: an arm that forgot to set its own fractions would
silently inherit the previous arm's.  So each arm asserts its fractions are ABSENT before
it builds them, and clears them after it restores.
"""
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common24 as C
import layers24 as LY
import dp_kernel as XI

R = C.ROUND
OUT = R / 'reports'
TAG = C.TAG
# The SAME sha-checked `eventlib` object `common24` imported (its own hash is asserted at
# import).  Bound here so the file cannot accidentally reach a different copy.
E = C.EL
N_STATIONS = 15

# Section 3.5 N10.  All three are `<=` and all three are stops.  `local_balance_max_kg` has
# a NON-ZERO float64 baseline, so `not (x > tol)` -- which a NaN would pass -- is not the
# test here.
TOL = dict(local_balance_kg=1e-6, network_scale=1e-10, label_sum=1e-6)
LABEL_CHANNELS = ('fast', 'slow', 'M', 'L', 'uptake', 'mineral_loss')

# Section 2.5 / 3.5 N6.  G5b is a HARD gate and this round has NO level lever, so a level
# failure and an amplitude failure are not separable and neither may be read as evidence
# about the other.
LEVEL_GATE = C.MONTHLY_GATE

# `c_base`/`c_peak` are anchored for L1 and L3 only -- there is no `c_base_L2` /
# `c_peak_L2` in the frozen anchor set, and inventing one would make B0 look like it
# failed on a quantity nothing ever registered.
STATION_LEVEL_KEYS = ('A_L1', 'A_L2', 'A_L3', 'c_base_L1', 'c_peak_L1',
                      'c_base_L3', 'c_peak_L3')


def _p(msg):
    print(msg, flush=True)


def arr_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a, np.float64).tobytes()).hexdigest()


def tau_of(surv):
    """Phase 0's `tau_of`, verbatim, INCLUDING its own internal filter.

    Both lifetimes are reported under BOTH estimators.  `-mean(log .)` is the geometric
    mean and is dominated by the smallest survival in the set; `-median(log .)` is the
    median.  Phase 0's first version quoted one against the other's median, which is an
    estimator mismatch rather than a physical finding.
    """
    ls = -np.log(surv[np.isfinite(surv) & (surv > 0.0) & (surv < 1.0)])
    if not len(ls):
        return dict(geometric_mean=None, median=None, p10=None, p90=None,
                    max_survival_lifetime=None, n=0)
    return dict(geometric_mean=float(1.0 / np.mean(ls)), median=float(1.0 / np.median(ls)),
                p10=float(1.0 / np.percentile(ls, 90)),
                p90=float(1.0 / np.percentile(ls, 10)),
                max_survival_lifetime=float(1.0 / np.min(ls)), n=int(len(ls)))


def set_Vu(model, raw):
    """`common24.dp_alternate_Vu`'s body with the SOURCE ARRAY supplied.

    `dp_alternate_Vu` accepts only the three registered names; the diagnostic arm needs the
    same construction from a fourth array.  Rather than special-case it, this is the same
    three lines, and `probe_arm_hashes.py` proves it agrees with the library helper on all
    three registered names -- so the shared path is checked rather than asserted.

    NO EXTRA KEY IS ADDED.  `install_kernel` re-binds the fractions through `closures_dp`,
    whose `CONFIG` carries exactly `gu`/`phi_f`/`gs`, and the packet handed to the scan is
    this dict; annotating it here would put a key into a structure whose key set is part of
    what `probe_arm_hashes.py` froze.
    """
    pack = model.dp_fractions
    V = XI.volumes(np.asarray(raw, np.float64), pack['Vs'] - pack['Qs'], pack)
    frac = XI.fractions(pack, V, pack['form'], pack['guard'])
    out = dict(pack)
    out.update(Vu=V['Vu'], **frac)
    model.dp_fractions = out
    return out


# ==========================================================================
# the mass ledger (section 3.5 N3) -- read at the INSTALLED kernel
# ==========================================================================
def ledger_gate(model, ex, tag=TAG):
    """`local_balance_max_kg`, `|network_balance_kg| / scale`, six label channels.

    THE TEETH, AND WHAT THEY ARE (plan R1')
    --------------------------------------
    `ledger_dp` spells `M`, `L` and `loss` INDEPENDENTLY of the kernel's return
    (`M = a*(1-gu)*s`, `L = cumsum(a*gu*(1-phi_f) - F_s)`) while `fast`/`slow` come FROM the
    return.  So `balance` asks "is the flux the kernel returned consistent with a separately
    written recurrence?".  Injecting `fast := 1.1*A*gu*phi_f` makes it fail analytically by
    `-0.1*A*gu*phi_f`.  That IS a check; it is NOT a physical conservation law, and with the
    kernel's own `fast`/`slow` it telescopes for ANY admissible fraction triple.

    HOW THE FILE KNOWS THE REBOUND BODY RAN
    ---------------------------------------
    `a['ledger_spelling']` exists ONLY on `closures_dp.ledger_dp`; the frozen
    `ResearchObjective.ledger` does not return it.  So this field is a direct proof that the
    class-attribute rebind took effect, which is stronger than inferring it from a number.

    THE RE-PROOF SECTION 3.5 N11(ii) ASKS FOR
    -----------------------------------------
    The ledger enters through `closures.ResearchObjective.ledger` (rebound to `ledger_dp`)
    and the forward through `structure_model.Transport` (rebound to `TransportDP`), and both
    read their fractions from `closures_dp.CONFIG`.  A stale `CONFIG` would grade one arm's
    ledger against ANOTHER arm's fractions and nothing would say so.  So this asserts, per
    arm, that `CONFIG` still equals `model.dp_fractions`, AND that the ledger's
    `(fast, slow, available)` is `array_equal` to the forward's `(F_f, F_s, a)`.
    """
    a = model.ledger(C.parameters(tag))
    river = np.asarray(a.get('river_input', a['fast'] + a['slow']), np.float64)
    scale = max(1.0, float(river.sum()))
    lab = a.get('source_label_sum_errors') or {}
    lab_max = max([float(v) for v in lab.values()], default=0.0)
    missing = sorted(set(LABEL_CHANNELS) - set(lab))
    lbal = float(a['local_balance_max_kg'])
    nbal = float(a['network_balance_kg'])

    # The scale-free twin of `source_label_sum_errors`, from the SAME ledger call: the
    # registered gate is absolute (1e-6 kg) and this round measures a river of order 1e8 kg,
    # so the relative residual is what says whether the absolute number is a rounding or a
    # real leak.  `rr` is `model.data.pilot_indices`, the same columns `Matched.ledger`
    # compared against -- read from the model, never re-typed.
    rr = np.asarray(C.pilot_indices(model), int)
    rel_max, rel_which, n_zero_ref_with_diff = 0.0, None, 0
    for name, v in (a.get('source_labels') or {}).items():
        ref = np.asarray(a[name], np.float64)[:, rr]
        d = np.abs(np.asarray(v, np.float64).sum(-1) - ref)
        nz = ref != 0.0
        n_zero_ref_with_diff += int(((~nz) & (d > 0.0)).sum())
        if nz.any():
            r = float(np.max(d[nz] / np.abs(ref[nz])))
            if r > rel_max:
                rel_max, rel_which = r, name

    conj = dict(conj_local=bool(lbal <= TOL['local_balance_kg']),
                conj_network=bool(abs(nbal) <= scale * TOL['network_scale']),
                conj_labels=bool(lab_max <= TOL['label_sum'] and not missing))
    out = dict(local_balance_max_kg=lbal, network_balance_kg=nbal, network_scale_kg=scale,
               source_label_sum_errors_max=lab_max,
               source_label_sum_errors={str(k): float(v) for k, v in lab.items()},
               source_label_channels_missing=missing, tolerance=TOL, **conj,
               source_label_rel_max=rel_max, source_label_rel_which=rel_which,
               n_zero_reference_cells_with_a_difference=n_zero_ref_with_diff,
               all_hold=bool(all(conj.values())),
               ledger_spelling_read_back=a.get('ledger_spelling'),
               ledger_body_ran=('ledger_dp' if a.get('ledger_spelling')
                                else 'the frozen ResearchObjective.ledger'),
               what_the_teeth_are='the ledger asks whether the flux the kernel RETURNED is '
                                  'consistent with a separately written recurrence. It is '
                                  'not a physical conservation law.')

    if ex.get('installed'):
        for k in ('gu', 'phi_f', 'gs'):
            if not np.array_equal(C.MC.CONFIG[k], model.dp_fractions[k]):
                raise SystemExit('STALE_CONFIG_%s' % k)
        if not (np.array_equal(a['fast'], ex['F_f'])
                and np.array_equal(a['slow'], ex['F_s'])
                and np.array_equal(a['available'], ex['a'])):
            raise SystemExit('LEDGER_AND_FORWARD_DISAGREE')
        if not a.get('ledger_spelling'):
            raise SystemExit('THE_LEDGER_REBIND_DID_NOT_TAKE')
        # The cumsum reassociation, MEASURED rather than assumed: the reported `C_s` uses
        # `layers24.slow_track` (the kernel's own order) and the ledger uses `np.cumsum`.
        # The gap is irrelevant to the 1e-6 kg gate and exactly what gives it teeth, but it
        # must never be silently baked into a reported concentration.
        J = ex['a'] * ex['gu'] * (1.0 - model.dp_fractions['phi_f'])
        _, Lexact = LY.slow_track(J, model.dp_fractions['gs'])
        num = float(np.max(np.abs(a['L'] - Lexact)))
        den = float(np.max(np.abs(Lexact)))
        out['ledger_L_vs_exact_track'] = dict(
            max_abs=num, max_abs_over_max_L=(num / den if den else None),
            n_exact=int((a['L'] == Lexact).sum()), n_cells=int(Lexact.size),
            note='the ledger spells L with np.cumsum (reassociated); the REPORTED '
                 'concentrations use layers24.slow_track (the kernel order). Both are '
                 'reported; neither is a criterion.')
        out['n11_ii_reproved'] = True
    else:
        out['n11_ii_reproved'] = None
        out['n11_ii_note'] = ('B0 runs the FROZEN ledger and the FROZEN transport, so there '
                              'is no rebound pair to re-prove.')
    return out


def observed_event_levels(ev):
    """The observed per-event absolute levels, for the section 2.7 `R` / `r` readings.

    THE LEVELS ALREADY EXIST, IN THE FROZEN EVENT SET.  `stage_a_events.parquet` carries
    `obs_tn_base` and `obs_tn_peak` (the same two columns the lineage's own `obs_ratio` was
    built from: `6.29/4.875 = 1.290256`).  So they are READ, not rebuilt -- which removes
    an entire class of risk, because a rebuild would have introduced a SECOND window
    convention and any difference between the two would then be indistinguishable from a
    model difference.

    Two checks, both reported rather than trusted:
      * `obs_tn_peak / obs_tn_base` must equal `obs_ratio` on every finite event;
      * `eventlib.obs_recipe_reproduce` recomputes `obs_ratio` from the RAW 4h panel with
        the frozen windows, which is the statement that the model-side mirror
        (`build_event_table` -> `base_peak`) is the same recipe on a different series.

    These are READINGs.  Not an event selection and not a fit target: a zero-fit round may
    SCORE against the registered panel, never calibrate on it.
    """
    t = ev[['station_key', 'event_id', 'obs_tn_base', 'obs_tn_peak', 'obs_ratio']].copy()
    for c in ('obs_tn_base', 'obs_tn_peak', 'obs_ratio'):
        t[c] = pd.to_numeric(t[c], errors='coerce')
    t['rederived'] = t.obs_tn_peak / t.obs_tn_base
    g = np.isfinite(t.rederived) & np.isfinite(t.obs_ratio)
    t['abs_dev'] = np.where(g, np.abs(t.rederived - t.obs_ratio), np.nan)
    return t, dict(
        n_events=int(len(t)), n_finite_ratio=int(g.sum()),
        max_abs_dev_ratio=float(np.nanmax(t.abs_dev)) if g.any() else None,
        ratio_column_reproduced=bool(g.sum() and np.nanmax(t.abs_dev) == 0.0),
        median_obs_ratio=float(np.nanmedian(t.obs_ratio)),
        obs_recipe_reproduce=E.obs_recipe_reproduce(ev),
        note='the observed absolute levels are READ from the frozen event set, not '
             'rebuilt; the model-side window mirror is checked against the raw 4h panel '
             'by the lineage own `obs_recipe_reproduce`')


# ==========================================================================
# the two lifetimes (section 2.6 P1 / R2')
# ==========================================================================
def lifetimes(model, frac, contact):
    """`tau_hydro` and `tau_eff`, on PHASE 0's OWN selection, so the two are comparable.

    Selection carried verbatim from `work/phase0_gates.py`:
        `interior[0] = False` ; `act = contact > 0` ; `sel = act & interior`
    NOT a reference-window restriction -- Phase 0's registered readings
    (`tau_eff` median `5.392064939611057`, `tau_hydro` median `5.411787491476153`, ratio
    `0.996355630760417`, `n_sel = 4544176`) were taken on THIS selection, and `main`
    re-derives all of them and reports whether they match.

    `tau_eff = tau_of((1 - g_u) * s_M)`, `tau_hydro = tau_of(1 - x_u)` with `x_u`
    restricted to `finite & >0 & <=1`.

    P1 IS FALSIFIED BY IDENTITY, and the reading is kept because the identity is the
    finding: under the linear closure `1 - g_u = 1 - Q_u/V_u = S_post/V_u` IS the retained
    water fraction, so the N memory lifetime is the upper store's water residence time by
    construction.  `s_M` still enters, and its implied lifetime is `293-3608` days, so the
    two cannot be conflated by name -- hence two columns, never one.
    """
    interior = np.ones_like(contact, bool)
    interior[0] = False
    sel = (contact > 0.0) & interior
    s_vec = np.asarray(model.flux_parameters(torch.tensor(C.parameters(TAG)))[1].numpy(),
                       np.float64)
    nr = int(model.data.fast_water.shape[1])
    if s_vec.shape != (nr,):
        raise SystemExit('S_M_IS_NOT_PER_REACH %r vs %d' % (s_vec.shape, nr))
    carry = (1.0 - frac['gu']) * s_vec[None, :]
    t_eff = tau_of(carry[sel])
    xh = frac['xu'][sel]
    pi_le1 = xh[np.isfinite(xh) & (xh > 0.0) & (xh <= 1.0)]
    t_hyd = tau_of(1.0 - pi_le1)
    xs = frac['xs'][sel]
    ratio = float(t_eff['median'] / t_hyd['median'])
    return dict(
        tau_eff=t_eff, tau_hydro=t_hyd,
        tau_eff_block={k: t_eff[k] for k in ('geometric_mean', 'median', 'p10', 'p90',
                                             'max_survival_lifetime')},
        tau_hydro_block={k: t_hyd[k] for k in ('geometric_mean', 'median', 'p10', 'p90',
                                               'max_survival_lifetime')},
        ratio__median_over_median=ratio,
        ratio__geomean_over_geomean=float(t_eff['geometric_mean']
                                          / t_hyd['geometric_mean']),
        n_sel=int(sel.sum()), n_carry_lt_1em6=int(np.sum(carry[sel] < 1e-6)),
        carry_quantiles={str(q): float(np.percentile(carry[sel], q))
                         for q in (1, 5, 25, 50, 75, 95, 99)},
        s_M_range=[float(s_vec.min()), float(s_vec.max())],
        s_M_band_relative=float((s_vec.max() - s_vec.min()) / float(np.median(s_vec))),
        s_M_implied_lifetime_days=[float(1.0 / -np.log(s_vec.max())),
                                   float(1.0 / -np.log(s_vec.min()))],
        absolute_scale_days=dict(p90=t_eff['p90'], max=t_eff['max_survival_lifetime'],
                                 note=t_eff['max_survival_lifetime'] is not None and
                                      'the ABSOLUTE scale matters more than the ratio: the '
                                      'N memory is sub-decadal'),
        frac_xu_ge_0p99=float(np.mean(xh >= 0.99)), frac_xu_eq_1=float(np.mean(xh == 1.0)),
        names_kept_separate=True, estimator_named=True,
        P1_falsifier='%s: tau_eff_median / tau_hydro_median = %.4f'
                     % ('FIRES' if ratio <= 10.0 else 'does not fire', ratio),
        P1_fired=bool(ratio <= 10.0),
        P1_verdict='FALSIFIED -- and not by coincidence, by IDENTITY',
        P1_why=('under the linear closure carry = (1-g_u)*s_M with 1-g_u = 1 - Q_u/V_u = '
                'S_post/V_u, which IS the retained water fraction. So -ln(carry) = '
                '-ln(S_post/V_u) - ln(s_M), and ln(s_M) is negligible against the O(1) '
                'water terms. The N memory lifetime is the upper store water residence '
                'time BY CONSTRUCTION, not by measurement.'),
        arm_label='UNIT_SCALE_LIMITED' if ratio <= 10.0 else None,
        applied='Plan S2.6 P1: where the falsifier fires, the arm is marked '
                'UNIT_SCALE_LIMITED and its event-gate failure -- if it fails -- is NOT '
                'evidence that the dual-pathway concentration structure lacks capability, '
                'but the timescale fact that the legacy pool is drained at the water '
                'turnover rate.',
        P2_slow_path=dict(
            n=int(len(xs)), median=float(np.median(xs)),
            relative_bandwidth=float((xs.max() - xs.min()) / float(np.median(xs))),
            frac_xs_ge_0p99=float(np.mean(xs >= 0.99)),
            frac_xs_eq_1=float(np.mean(xs == 1.0)),
            inverted_route_bandwidth_reference=0.061,
            note='P2: if V_s is the producer-written store, the x_s bandwidth should be '
                 'MARKEDLY larger than the 6.1% the inverted route gives. If it is not, '
                 'the slow path is a steady linear reservoir and its amplitude response '
                 'comes only through L_pre and phi(x_s) -- which must NOT be read as "the '
                 'slow path does not exist".'))


# ==========================================================================
# one arm's worth of readings -- no second forward anywhere
# ==========================================================================
def measure(model, ly, ex, ev, mask, elig, obs_m, evobs, contact):
    """Everything an arm needs, from ONE forward frame.

    `ly` is coerced to `datetime64[ns]` on `date` first, the resolution `eventlib.as_day`
    produces and therefore the resolution of the eligible grid and of the mask: a
    mis-typed merge here would produce NaN concentrations that `station_sd_gate` and
    `monthly_stats` DO catch (both raise on a dropped eligible day) -- but catching it
    here is cheaper and names the cause.
    """
    ly = ly.assign(date=pd.to_datetime(ly.date).dt.normalize().astype('datetime64[ns]'))
    b = LY.layer_budget(ly, ev)
    evb = E.build_event_table(
        ly[['station_key', 'date', 'pL3']].rename(columns={'pL3': 'p'}), ev)
    evb1 = E.build_event_table(
        ly[['station_key', 'date', 'pL1']].rename(columns={'pL1': 'p'}), ev)
    evb = evb.merge(evobs[['station_key', 'event_id', 'obs_tn_base', 'obs_tn_peak',
                           'obs_ratio']],
                    on=['station_key', 'event_id'], validate='one_to_one')
    # `build_event_table` already supplies `dc` (= c_peak - c_base), the model-side twin of
    # the lineage's `obs_delta_tn`; recomputing it here would be a second spelling of the
    # same quantity, which is exactly what `f1_coef` compares against `c_base`.
    bh = E.f1_coef(evb, 'T_interevent')
    ah = E.f3_intercept(evb, E.F3_PRIMARY_GAP)
    finp = evb.obs_tn_peak.notna() & (evb.obs_tn_peak > 0) & evb.c_peak.notna()
    finb = evb.obs_tn_base.notna() & (evb.obs_tn_base > 0) & evb.c_base.notna()
    Rr = float(np.median((evb.c_peak[finp] / evb.obs_tn_peak[finp]).to_numpy()))
    rr_ = float(np.median((evb.c_base[finb] / evb.obs_tn_base[finb]).to_numpy()))
    out = dict(b=b, evb=evb, evb1=evb1, beta_hat=bh, alpha_hat=ah,
               sd=LY.station_sd_gate(ly, mask),
               monthly=C.monthly_stats(ly, elig, obs_m),
               s2_7=dict(R_model_peak_over_obs_peak=Rr,
                         r_model_base_over_obs_base=rr_,
                         abs_log_ratio_difference=float(abs(np.log(Rr) - np.log(rr_))),
                         n_events_peak=int(finp.sum()), n_events_base=int(finb.sum()),
                         note='section 2.7 READINGS so the next round can choose between '
                              'k_m and k_ex. Nothing is implemented, parameterised or '
                              'switched on.'))
    if ex.get('installed'):
        frac = model.dp_fractions
        out['tau'] = lifetimes(model, frac, contact)

        # P3: where the fast-path flux lands in the month.  `inp[starts] = data.source`
        # and `demand[starts] = data.crop` (`closures.py:142`), so A spikes on month-start
        # days.  If F_f decays into a monthly delta function then A is measuring the
        # CALENDAR, not a mechanism.
        nd = int(frac['gu'].shape[0])
        starts = np.zeros(nd, bool)
        st = np.asarray(getattr(model.data, 'starts', ()), int)
        starts[st[st < nd]] = True
        A = np.asarray(ex['a'], np.float64)
        Eu = A * np.asarray(ex['gu'], np.float64)
        Ff = np.asarray(ex['F_f'], np.float64)
        Fs = np.asarray(ex['F_s'], np.float64)
        out['P3_pulse'] = dict(
            n_month_starts=int(starts.sum()), n_days=nd,
            frac_of_days_that_are_month_starts=float(starts.mean()),
            frac_of_Ff_mass_on_month_start_days=float(Ff[starts].sum() / Ff.sum()),
            frac_of_Fs_mass_on_month_start_days=float(Fs[starts].sum() / Fs.sum()),
            frac_of_A_on_month_start_days=float(A[starts].sum() / A.sum()),
            frac_of_Eu_mass_on_month_start_days=float(Eu[starts].sum() / Eu.sum()),
            note='P3. A monthly-delta-shaped F_f means A is measuring the calendar, and '
                 'the report must say so rather than read it as a mechanism.')

        # P4: how much the OTHER admissible closure would move things.  Under `g = x` the
        # chosen phi is identically 1, so the informative reading is on the alternative.
        xu = frac['xu']
        live = np.isfinite(xu) & (xu > 0.0)
        phi_exp = np.ones_like(xu)
        phi_exp[live] = -np.expm1(-xu[live]) / xu[live]
        act = frac['guard'] > 0.0
        sel4 = act & live
        phi_ch = XI.phi_of(xu, frac['form'])
        out['P4_closure_sensitivity'] = dict(
            chosen_form=frac['form'], guard_kept=bool(np.any(frac['guard'] == 0.0)),
            phi_chosen_min=float(np.min(phi_ch)), phi_chosen_max=float(np.max(phi_ch)),
            phi_exp_median_on_active=float(np.median(phi_exp[sel4])),
            frac_active_where_other_closure_differs_over_10pct=float(
                np.mean(phi_exp[sel4] < 0.9)),
            frac_active_where_other_closure_differs_over_1pct=float(
                np.mean(phi_exp[sel4] < 0.99)),
            n_active_cells=int(sel4.sum()),
            note='the two admissible closures differ by more than 10% on this fraction of '
                 'ACTIVE cells. Under `g = x` the chosen phi is identically 1, so this is '
                 'how much the round did NOT use -- an independent fact from the verdict.')

        out['C_distribution'] = dict(
            units='mg/L, plan section 1.2 stable form; the denominator contains no Q',
            per_reach_day=True,
            C_u=dict(min=float(np.min(ex['C_u'])), max=float(np.max(ex['C_u'])),
                     median=float(np.median(ex['C_u'])),
                     p1=float(np.percentile(ex['C_u'], 1)),
                     p99=float(np.percentile(ex['C_u'], 99))),
            C_s=dict(min=float(np.min(ex['C_s'])), max=float(np.max(ex['C_s'])),
                     median=float(np.median(ex['C_s'])),
                     p1=float(np.percentile(ex['C_s'], 1)),
                     p99=float(np.percentile(ex['C_s'], 99))),
            cumsum_reassociation_max_rel=float(ex['cumsum_reassociation_max_rel']),
            cumsum_reassociation_median_rel=float(ex['cumsum_reassociation_median_rel']),
            cumsum_exact_frac=float(ex['cumsum_exact_frac']),
            C_u_max_disclosure='`C_u` reaches its maximum on the deep-tail cells where '
                               '`V_u` floors at ~2.6e-204 mm; it is a reported-distribution '
                               'fact and enters no criterion, because every criterion reads '
                               'the station-day `pL1/pL2/pL3`.')
        out['N11_i_shapes'] = dict(
            gu=list(np.shape(ex['gu'])), phi_f=list(np.shape(frac['phi_f'])),
            gs=list(np.shape(frac['gs'])), a=list(np.shape(ex['a'])),
            all_equal=bool(np.shape(ex['gu']) == np.shape(frac['phi_f'])
                           == np.shape(frac['gs']) == np.shape(ex['a'])),
            note='the three fraction arrays enter the njit wrappers as ARGUMENTS; '
                 '`closures_dp._check` asserts each shape inside every wrapper, and '
                 '`dp_kernel.fractions` asserts them at construction.')
    else:
        out['tau'] = dict(
            available=False,
            why='B0 runs the FROZEN kernel; its `p` is the frozen `prob`, not a water '
                'fraction, so `1-p` is not a retained-water share and neither lifetime is '
                'comparable here. Phase 0 measured the pair on the NEW kernel.')
    return out


def eligible_slice(ly, elig):
    """The eligible station-days only, one row per (station, day).

    Gate 16 requires the stored row count to match the eligible grid, and every criterion
    reads that grid, so storing the full calendar would store rows nothing downstream can
    use.  A dropped eligible day is a LOUD stop, not a silent inner join.
    """
    sel = elig.merge(ly[['station_key', 'date', 'water_m3_day', 'pL1', 'pL2', 'pL3']],
                     on=['station_key', 'date'], how='left', validate='one_to_one')
    if int(sel.pL3.isna().sum()):
        raise SystemExit('A_CANDIDATE_DROPPED_AN_ELIGIBLE_DAY %d'
                         % int(sel.pL3.isna().sum()))
    E.assert_no_mass_columns(sel)
    return sel


def station_reach_map(ex):
    """The station -> reach column map, TAKEN FROM THE FORWARD.

    Round 5's `common23.station_reaches` was deleted from `common24` -- this round has no
    geometry module -- so re-typing the table would make a transcribed number look like a
    measurement.  `layers24.forward_layers` returns the model's own `record` and `ri`, and
    the map is read off those.
    """
    return (pd.DataFrame(dict(k=ex['meta'].station_key.to_numpy()[ex['record'].numpy()],
                              r=ex['ri'])).drop_duplicates()
            .set_index('k').r.to_dict())


def event_signs(ev, s2r, dq_full, cal):
    """Per event, `sign(Q_f - Q_s)` over the frozen recipe's own two windows.

    `Q_f - Q_s` is ARM-INVARIANT: both come from `dp_kernel.flows`, which reads the frozen
    hydrology and carries no arm parameter.  So the grouping is computed ONCE for all six
    arms and reported as one table, rather than six identical ones.

    The two windows are the frozen recipe's: base `[t_start - 7d, t_start)`, peak
    `[t_start, t_end + 1d]` BOTH ENDS INCLUSIVE.  Half-open on the left, `right` on the
    upper bound -- the same pins `eventlib.base_peak` uses.
    """
    rows = []
    for r in ev.itertuples():
        c = int(s2r[str(r.station_key)])
        t0 = np.datetime64(pd.Timestamp(r.t_start).normalize(), 'D')
        t1 = np.datetime64(pd.Timestamp(r.t_end).normalize()
                           + pd.Timedelta(days=1), 'D')
        lo = np.datetime64(pd.Timestamp(r.t_start).normalize()
                           - pd.Timedelta(days=E.TN_PRE_DAYS), 'D')
        ib = slice(np.searchsorted(cal, lo, 'left'), np.searchsorted(cal, t0, 'left'))
        ip = slice(np.searchsorted(cal, t0, 'left'), np.searchsorted(cal, t1, 'right'))
        rows.append(dict(station_key=r.station_key, event_id=r.event_id,
                         sign_peak=int(np.sign(np.mean(dq_full[ip, c]))),
                         sign_base=int(np.sign(np.mean(dq_full[ib, c]))),
                         mean_dq_peak=float(np.mean(dq_full[ip, c])),
                         mean_dq_base=float(np.mean(dq_full[ib, c]))))
    return pd.DataFrame(rows)


def grouped_A(evt, signs, key):
    """Median `amp_ratio` of the events whose window-mean `sign(Q_f - Q_s)` is +1/-1/0."""
    m = evt.merge(signs[['station_key', 'event_id', 'sign_' + key]],
                  on=['station_key', 'event_id'], validate='one_to_one')
    out, n = {}, {}
    for s, tag in ((1, 'pos'), (-1, 'neg'), (0, 'zero')):
        v = m.amp_ratio[m['sign_' + key] == s].to_numpy(float)
        g = np.isfinite(v)
        if not g.sum():
            continue
        out[tag] = float(np.median(v[g]))
        n[tag] = int(g.sum())
    return dict(A_L1_by_sign=out, n_by_sign=n,
                note='sign(Q_f - Q_s) is a WATER-path flag read off the frozen hydrology; '
                     'it carries no arm parameter and no observation.')


def arm_readings(name, meta_arm, m, led, ex, ev, sg, t0):
    """The per-arm reading block.  Written once so no arm can be reported differently."""
    r = dict(arm=name, role=meta_arm['role'], Vu_post=meta_arm.get('Vu_post'),
             installs_kernel=meta_arm['installs_kernel'],
             A_L1=m['b']['L1']['amp_ratio_median'],
             A_L2=m['b']['L2']['amp_ratio_median'],
             A_L3=m['b']['L3']['amp_ratio_median'],
             c_base_L1=m['b']['L1']['c_base_median'],
             c_peak_L1=m['b']['L1']['c_peak_median'],
             c_base_L2=m['b']['L2']['c_base_median'],
             c_peak_L2=m['b']['L2']['c_peak_median'],
             c_base_L3=m['b']['L3']['c_base_median'],
             c_peak_L3=m['b']['L3']['c_peak_median'],
             beta_hat=m['beta_hat'], alpha_hat=m['alpha_hat'],
             nse=m['monthly']['nse'], r2=m['monthly']['r2'],
             median_station_nse=m['monthly']['median_station_nse'],
             mean_concentration=m['monthly']['mean_concentration'],
             n_station_months=m['monthly']['n_station_months'],
             n_eligible_rows=m['monthly']['n_eligible_rows'],
             sd_L1_ddof0_median_e=m['sd']['L1']['ddof0']['median_e'],
             sd_L2_ddof0_median_e=m['sd']['L2']['ddof0']['median_e'],
             sd_L3_ddof0_median_e=m['sd']['L3']['ddof0']['median_e'],
             sd_all_layers={L: dict(ddof0=m['sd'][L]['ddof0']['median_e'],
                                    ddof1=m['sd'][L]['ddof1']['median_e'],
                                    ratio_mdl_over_obs_median=
                                    m['sd'][L]['ddof0']['ratio_mdl_over_obs_median'],
                                    n_stations=m['sd'][L]['ddof0']['n_stations'])
                            for L in ('L1', 'L2', 'L3')},
             sd_gating_layer=m['sd']['gating_layer'],
             sd_ddof_choice_is_inert=m['sd']['L3']['ddof_choice_is_inert'],
             budget=m['b'], monthly=m['monthly'], ledger=led, tau=m['tau'],
             A_L1_by_sign=grouped_A(m['evb1'], sg, 'peak'),
             A_L1_by_sign_base=grouped_A(m['evb1'], sg, 'base'),
             s2_7=m['s2_7'], n_events=int(len(ev)),
             seconds=round(time.time() - t0, 2))
    for k in ('P3_pulse', 'P4_closure_sensitivity', 'C_distribution', 'N11_i_shapes'):
        if k in m:
            r[k] = m[k]
    return r


def gate_block(r, A, G1T, G2T, sd_thr, bl, per_arm_evb):
    """G1/G2/G3/G5/G5b (the main verdict) plus G4 (prediction, not a veto)."""
    d_beta = None if r['beta_hat'] is None else abs(r['beta_hat'] - A['beta_obs'])
    d_alpha = None if r['alpha_hat'] is None else abs(r['alpha_hat'] - A['alpha_obs'])
    nse_deg = float(bl['nse'] - r['nse'])
    mnse_deg = float(bl['median_station_nse'] - r['median_station_nse'])
    mean_rel = float(abs(r['mean_concentration'] - bl['mean_concentration'])
                     / bl['mean_concentration'])
    g = dict(G1=bool(r['A_L1'] >= G1T), G2=bool(r['A_L3'] >= G2T),
             G3=bool(r['sd_L3_ddof0_median_e'] <= sd_thr),
             G4=bool(d_beta is not None and d_alpha is not None
                     and d_beta <= A['D_beta_base'] and d_alpha <= A['D_alpha_base']),
             G5=bool(nse_deg <= C.MONTHLY_GATE and mnse_deg <= C.MONTHLY_GATE),
             G5b=bool(mean_rel <= LEVEL_GATE))
    r.update(D_beta_cand=d_beta, D_alpha_cand=d_alpha,
             D_beta_base=A['D_beta_base'], D_alpha_base=A['D_alpha_base'],
             beta_obs=A['beta_obs'], alpha_obs=A['alpha_obs'],
             nse_degradation_null_minus_candidate=nse_deg,
             mnse_degradation_null_minus_candidate=mnse_deg,
             mean_concentration_relative_change=mean_rel,
             level_ratio_arm_over_frozen=float(r['mean_concentration']
                                               / bl['mean_concentration']),
             sd_gate_threshold=sd_thr,
             sd_gate_margin=float(sd_thr - r['sd_L3_ddof0_median_e']),
             gates=g, **{k + '_pass': v for k, v in g.items()})
    r['n_gates_passed_main_five'] = int(sum(g[x] for x in ('G1', 'G2', 'G3', 'G5', 'G5b')))
    r['all_main_five_pass'] = bool(r['n_gates_passed_main_five'] == 5)
    r['g5_direction'] = ('baseline MINUS candidate; positive means the candidate degraded '
                         'by that much')
    if r['arm'] != 'B0':
        cur = per_arm_evb[r['arm']].set_index(['station_key', 'event_id']).amp_ratio
        b0 = per_arm_evb['B0'].set_index(['station_key', 'event_id']).amp_ratio
        j = pd.concat([b0.rename('a0'), cur.rename('a1')], axis=1)
        j = j[np.isfinite(j.a0) & np.isfinite(j.a1)]
        r['vs_B0'] = dict(n_events=int(len(j)), n_better=int((j.a1 > j.a0).sum()),
                          n_worse=int((j.a1 < j.a0).sum()),
                          n_tied=int((j.a1 == j.a0).sum()),
                          median_change=float((j.a1 - j.a0).median()),
                          note='per-event L1 amp_ratio against the B0 anchor arm, NOT '
                               'against an observation')
    return r


# ==========================================================================
def main():
    t_start = time.time()
    rep = {'phase': '1_arms', 'round': str(R), 'n_fits': 0, 'fit_worker_calls': 0,
           'n_stations_expected': N_STATIONS, 'monthly_gate': C.MONTHLY_GATE,
           'sd_gate_fraction': C.SD_GATE_FRACTION, 'level_gate': LEVEL_GATE,
           'main_verdict_gates': ['G1', 'G2', 'G3', 'G5', 'G5b'],
           'G4_role': 'pre-registered prediction + falsifier; NOT a veto (section 2.5)',
           'g5_direction': 'RESTORED upstream direction: baseline MINUS candidate, '
                           '20260919_2/work/phase1_score.py:178,203; round 4s reversed '
                           'implementation at phase2_full.py:247 is NOT reproduced here',
           'no_early_stop': 'section 4.1 forbids one and this file takes none: all five '
                            'forwards run before any gate is evaluated'}

    # ------------------------------------------------------- arm table, read back
    arms = C.read_json(OUT / 'arms.json')
    if not (arms.get('pre_registered') and arms['gate'].get('frozen_before_any_forward')):
        raise SystemExit('THE_ARM_TABLE_WAS_NOT_FROZEN_BEFORE_THIS_FORWARD')
    prereg = C.read_json(OUT / '预注册_冻结.json')
    if int(prereg['forward_runs_so_far']) != 0:
        raise SystemExit('THIS_IS_NOT_THE_FIRST_FORWARD %r'
                         % prereg['forward_runs_so_far'])
    rep['arms_frozen'] = dict(
        primary_arm=arms['primary_arm'], closure_form=arms['closure_form'],
        n_arms=arms['gate']['n_arms'], n_forwards=arms['gate']['n_forwards'],
        table_sha256_registered=arms['hashes']['table_sha256'],
        arms_table_sha256_in_preregistration=prereg['arms_table_sha256'],
        table_sha_matches_preregistration=bool(
            arms['hashes']['table_sha256'] == prereg['arms_table_sha256']),
        named_free_choices=len(arms['named_free_choices']),
        arm_names=[a['arm'] for a in arms['arms']],
        dedup=arms['gate']['dedup'],
        naming_trap='`Vu_post` is the PRE-outflow carry; `Vu_at_outflow = Vu_post + Qu`')
    if not rep['arms_frozen']['table_sha_matches_preregistration']:
        raise SystemExit('THE_ARM_TABLE_CHANGED_SINCE_THE_PREREGISTRATION')

    # ------------------------------------------------------------------ anchors
    rep['anchors'] = C.load_anchors()
    A = {k: v['value'] for k, v in rep['anchors'].items()}
    G1T, G2T = A['G1_target_50pct'], A['G2_target_50pct']
    # The threshold is read from the ANCHOR, never written into it: overwriting an anchor
    # from a constant is how a gate stops being frozen. The panel constant is then checked
    # AGAINST the anchor rather than substituted for it.
    sd_thr = float(A['sd_gate_threshold'])
    if sd_thr != float(C.SD_GATE_70PCT[A['sd_gate_layer']]):
        raise SystemExit('THE_SD_THRESHOLD_ANCHOR_AND_THE_PANEL_DISAGREE')
    if A['sd_gate_layer'] != 'L3':
        raise SystemExit('THE_SD_GATING_LAYER_MOVED %r' % A['sd_gate_layer'])
    if A['n_events'] != 214:
        raise SystemExit('THE_FROZEN_EVENT_COUNT_MOVED %r' % A['n_events'])
    rep['anchor_provenance'] = {k: dict(value=v['value'], source=v.get('source'),
                                        sha256=v.get('sha256'),
                                        matches_registered=v.get('matches_registered'))
                                for k, v in rep['anchors'].items()}
    _p('=== anchors ===  %d read from their producers' % len(rep['anchors']))

    # ------------------------------------------------------- model and eligibility
    model = C.build(TAG)
    contact = np.asarray(model.data.contact, np.float64)
    if not C.hazard_is_frozen():
        raise SystemExit('Predictor.hazard_IS_REBOUND')
    if C.is_installed()['all_bound']:
        raise SystemExit('THE_KERNEL_MUST_START_UNINSTALLED')
    if getattr(model, 'dp_fractions', None):
        raise SystemExit('DP_FRACTIONS_ALREADY_SET_BEFORE_THE_FIRST_ARM')
    mask = pd.read_parquet(E.MASK)
    elig = C.eligible_grid()
    rep['eligible'] = dict(n_rows=int(len(elig)),
                           n_stations=int(elig.station_key.nunique()),
                           mask_sha=C.sha(E.MASK),
                           mask_sha_matches=bool(C.sha(E.MASK) == E.MASK_SHA))
    if rep['eligible']['n_stations'] != N_STATIONS:
        raise SystemExit('N_STATIONS_MOVED %d' % rep['eligible']['n_stations'])
    obs_m = C.obs_monthly()
    ev = E.eligible_events()
    if len(ev) != int(A['n_events']) or ev.station_key.nunique() != N_STATIONS:
        raise SystemExit('THE_FROZEN_EVENT_SET_MOVED %d / %d'
                         % (len(ev), ev.station_key.nunique()))
    evobs, obs_prov = observed_event_levels(ev)
    rep['observed_levels'] = obs_prov
    _p('=== events ===  %d events / %d stations; obs level/ratio column reproduced=%s '
       '(max|d|=%.3g); obs_recipe_reproduce %s'
       % (len(ev), ev.station_key.nunique(), obs_prov['ratio_column_reproduced'],
          obs_prov['max_abs_dev_ratio'],
          obs_prov['obs_recipe_reproduce']))

    # ---------------------------------------------- anchor replay (FROZEN kernel)
    rep['anchor_replay'] = C.anchor_gate(C.replay(model, TAG))
    if not rep['anchor_replay']['passed']:
        raise SystemExit('ANCHOR_REPLAY_FAILED %r' % rep['anchor_replay'])
    _p('=== anchor replay ===  %d rows, max|dp| = %.3g <= %.3g'
       % (rep['anchor_replay']['n_compared_rows'],
          rep['anchor_replay']['max_abs_elementwise_concentration'],
          rep['anchor_replay']['tolerance']))

    # -------------------------------------------------- arm-invariant quantities
    Fq = XI.flows(np.asarray(model.data.fast_water, np.float64),
                  np.asarray(model.data.percolation, np.float64),
                  np.asarray(model.data.slow_water, np.float64),
                  np.asarray(model.data.area_ha, np.float64))
    dq_full = Fq['Qf'] - Fq['Qs']
    rep['q_arm_invariance'] = dict(
        Qf_sha=arr_sha(Fq['Qf']), Qs_sha=arr_sha(Fq['Qs']),
        matches_the_frozen_arm_table=bool(arr_sha(Fq['Qf']) == arms['hashes']['Qf']
                                          and arr_sha(Fq['Qs']) == arms['hashes']['Qs']),
        min_Qf=float(Fq['Qf'].min()), min_Qp=float(Fq['Qp'].min()),
        min_Qs=float(Fq['Qs'].min()), min_Qu=float(Fq['Qu'].min()),
        note='Q_f, Q_p and Q_s come from the frozen hydrology alone, so sign(Q_f - Q_s) '
             'is identical on every arm and is computed once. All three are STRICTLY '
             'positive, so x = Q/V is well defined everywhere and no 0/0 appears.')
    if not rep['q_arm_invariance']['matches_the_frozen_arm_table']:
        raise SystemExit('THE_FROZEN_HYDROLOGY_MOVED')
    cal = np.asarray(model.data.dates).astype('datetime64[D]')

    # =========================================================== the five forwards
    frames, rows, per_arm, per_arm_evb = [], {}, {}, {}
    sg, s2r = None, None
    for meta_arm in arms['arms']:
        name = meta_arm['arm']
        if meta_arm.get('reads_disk'):
            continue
        t0 = time.time()
        _p('=== arm %s (role %s) ===' % (name, meta_arm['role']))
        if C.is_installed()['all_bound']:
            raise SystemExit('A_PREVIOUS_ARMS_KERNEL_IS_STILL_BOUND_BEFORE_%s' % name)
        if meta_arm['installs_kernel']:
            if getattr(model, 'dp_fractions', None):
                raise SystemExit('DP_FRACTIONS_LEAKED_INTO_A_LATER_ARM_BEFORE_%s' % name)
            which = meta_arm['Vu_post']
            C.dp_arrays(model, form=arms['closure_form'])
            if arr_sha(model.dp_fractions['Vu']) != arms['hashes']['Vu_P_upper']:
                raise SystemExit('VU_UPPER_DRIFTED %s' % arr_sha(model.dp_fractions['Vu']))
            if which == 'upper_water':
                pass                                    # the primary arm, already built
            elif which in ('soil_water_mm', 'soil_plus_upper'):
                key = {'soil_water_mm': 'soil', 'soil_plus_upper': 'unsat'}[which]
                raw = {'soil_water_mm': np.asarray(model.data.soil_water_mm, np.float64),
                       'soil_plus_upper': (np.asarray(model.data.soil_water_mm, np.float64)
                                           + np.asarray(model.data.upper_water,
                                                        np.float64))}[which]
                if arr_sha(raw) != meta_arm['Vu_post_sha']:
                    raise SystemExit('VU_POST_DRIFTED_%s' % name)
                lib = C.dp_alternate_Vu(model, key)
                man = set_Vu(model, raw)
                if not all(np.array_equal(lib[k], man[k])
                           for k in ('Vu', 'gu', 'phi_f', 'gs', 'xu', 'xs')):
                    raise SystemExit('SET_VU_AND_DP_ALTERNATE_VU_DISAGREE_ON_%s' % name)
            elif which == 'P-upper_reference_mean':
                up = np.asarray(model.data.upper_water, np.float64)
                refmask = C.window_mask(model.data.dates, C.REF_YEARS)
                raw = np.broadcast_to(up[refmask].mean(axis=0), up.shape).copy()
                if not np.all(raw == raw[0]):
                    raise SystemExit('D_CONST_IS_NOT_TIME_INVARIANT')
                if arr_sha(raw) != meta_arm['Vu_post_sha']:
                    raise SystemExit('D_CONST_SOURCE_DRIFTED %s' % arr_sha(raw))
                set_Vu(model, raw)
                rep['D_const'] = dict(
                    definition='per-reach mean of `upper_water` over the reference window '
                               '[1961,2020], broadcast back over all days',
                    n_reference_days=int(refmask.sum()),
                    per_reach_range=[float(raw[0].min()), float(raw[0].max())],
                    n_reaches=int(raw.shape[1]), is_time_invariant=True,
                    relation_to_primary='D-const IS the primary arm with the V_u TIME '
                                        'variation removed, so any difference between the '
                                        'two is the saturation nonlinearity and the '
                                        's_M carry alone',
                    note='time-structure diagnostic: the V_u TIME variation is removed and '
                         'only the saturation nonlinearity survives. It is not a physical '
                         'store and no producer declares it.')
            else:
                raise SystemExit('UNKNOWN_VU_POST %r' % which)
            if arr_sha(model.dp_fractions['Vs']) != arms['hashes']['Vs']:
                raise SystemExit('VS_MOVED_ON_AN_ARM_THAT_ONLY_VARIES_VU')
            if arr_sha(model.dp_fractions['guard']) != arms['hashes']['guard']:
                raise SystemExit('GUARD_MOVED_ON_AN_ARM')
            if arr_sha(model.dp_fractions['phi_f']) != arms['hashes']['phi_f']:
                raise SystemExit('PHI_F_MOVED_ON_AN_ARM')
            info = C.install_kernel(model)
            if not info['all_bound']:
                raise SystemExit('KERNEL_INSTALL_INCOMPLETE %s %r' % (name, info))
            # POLARITY.  `is_installed()['ledger_is_frozen']` is
            # `bool(_ledger_is_dp() is False)` -- True when the FROZEN ledger is bound.
            # So right after a SUCCESSFUL install it is False, and the obvious-looking
            # guard `if not ledger_is_frozen: raise` fires on every arm.  It was read off
            # the source (`common24.py:520`) rather than inferred from the name, precisely
            # because the name points the other way.
            if C.is_installed()['ledger_is_frozen']:
                raise SystemExit('THE_LEDGER_REBIND_DID_NOT_TAKE')
            if not C._ledger_is_dp():
                raise SystemExit('LEDGER_IS_NOT_THE_DP_BODY_AFTER_INSTALL')
            rep.setdefault('installs', {})[name] = dict(
                closure_form=info['closure_form'],
                n_transport_modules=info['n_transport_modules'],
                per_name={k: v['all_live'] for k, v in info['per_name'].items()})
        else:
            if getattr(model, 'dp_fractions', None):
                raise SystemExit('B0_MUST_RUN_WITHOUT_DP_FRACTIONS')

        df, ex = LY.forward_layers(model, TAG, installed=meta_arm['installs_kernel'])
        m = measure(model, df, ex, ev, mask, elig, obs_m, evobs, contact)
        led = ledger_gate(model, ex)
        C.restore_kernel()
        model.dp_fractions = None
        if C.is_installed()['all_bound']:
            raise SystemExit('KERNEL_STILL_BOUND_AFTER_%s' % name)
        if not led['all_hold']:
            raise SystemExit('MASS_LEDGER_FAILED %s %r' % (name, led))

        if sg is None:
            s2r = station_reach_map(ex)
            sg = event_signs(ev, s2r, dq_full, cal)
            rep['events_sign'] = dict(
                n_station_reaches=int(len(set(s2r.values()))),
                n_stations_mapped=int(len(s2r)), n_events=int(len(sg)),
                sign_peak_positive=int((sg.sign_peak > 0).sum()),
                sign_peak_negative=int((sg.sign_peak < 0).sum()),
                sign_peak_zero=int((sg.sign_peak == 0).sum()),
                sign_base_positive=int((sg.sign_base > 0).sum()),
                sign_base_negative=int((sg.sign_base < 0).sum()),
                sign_base_zero=int((sg.sign_base == 0).sum()),
                note='Q_f - Q_s is read from the frozen hydrology and carries no arm '
                     'parameter, so sign(Q_f - Q_s) is the same on all six arms')

        sel = eligible_slice(df, elig)
        if len(sel) != len(elig):
            raise SystemExit('DAILY_ARMS_ROW_COUNT %s %d vs %d'
                             % (name, len(sel), len(elig)))
        frames.append(sel.assign(arm=name))
        r = arm_readings(name, meta_arm, m, led, ex, ev, sg, t0)
        rows[name] = r
        per_arm[name] = dict(m=m, led=led, ex=ex, meta=meta_arm)
        per_arm_evb[name] = m['evb1']
        _p('   %-8s A_L1=%.6f A_L2=%.6f A_L3=%.6f  e_s=%.5f  nse=%+.4f mnse=%+.4f  '
           'meanC=%.4f  %.1fs'
           % (name, r['A_L1'], r['A_L2'], r['A_L3'], r['sd_L3_ddof0_median_e'], r['nse'],
              r['median_station_nse'], r['mean_concentration'], r['seconds']))

    # ------------------------------------------------------------ the Phase 0 check
    n5 = C.read_json(OUT / 'phase0_gates.json')['N5']
    got = rows[arms['primary_arm']]['tau']
    checks = {k: (dict(got[k]) if isinstance(got[k], dict) else got[k])
              for k in ('tau_eff_block', 'tau_hydro_block')}
    checks['ratio__median_over_median'] = got['ratio__median_over_median']
    checks['ratio__geomean_over_geomean'] = got['ratio__geomean_over_geomean']
    checks['n_sel'] = got['n_sel']
    want = {k: n5[k] for k in ('tau_eff_block', 'tau_hydro_block',
                               'ratio__median_over_median',
                               'ratio__geomean_over_geomean', 'n_sel')}
    diffs = {}
    for grp in ('tau_eff_block', 'tau_hydro_block'):
        for k in want[grp]:
            if checks[grp].get(k) != want[grp][k]:
                diffs['%s.%s' % (grp, k)] = [checks[grp].get(k), want[grp][k]]
    for k in ('ratio__median_over_median', 'ratio__geomean_over_geomean', 'n_sel'):
        if checks[k] != want[k]:
            diffs[k] = [checks[k], want[k]]
    rep['P1_reproduces_phase0'] = dict(
        arm=arms['primary_arm'], got=checks, phase0=want, differences=diffs,
        ok=bool(not diffs),
        note='the primary arm re-measures Phase 0 own two lifetimes on Phase 0 own '
             'selection (`sel = contact>0 & day!=0`); bitwise equality means the '
             'lifetimes carried into section 4.2 are the same quantities Phase 0 '
             'registered. A mismatch is a READING, reported loudly and never repaired by '
             'changing the selection.')
    if not rep['P1_reproduces_phase0']['ok']:
        _p('!!! P1 LIFETIMES DO NOT REPRODUCE PHASE 0: %r' % diffs)
    else:
        _p('=== P1 lifetimes reproduce Phase 0 bitwise ===  tau_eff %.6f  tau_hydro '
           '%.6f  ratio %.6f' % (checks['tau_eff_block']['median'],
                                 checks['tau_hydro_block']['median'],
                                 checks['ratio__median_over_median']))

    # ------------------------------------------------------------------- R5-ref
    # This arm is a READ, and the read is TWO-SOURCED because the two sources answer
    # different questions.  It is the only arm in the round whose numbers do not come from
    # a forward here, so the provenance has to be stated rather than implied.
    r5p = C.PEER_ROOT / '20260919_5/reports/daily_layers.parquet'
    r5j = C.PEER_ROOT / '20260919_5/reports/phase1_full.json'
    p5 = C.read_json(r5j)['points']['N1e|0.5']
    d5 = pd.read_parquet(r5p)
    n5_all = int(len(d5))
    d5 = d5[(d5.device == 'N1e') & (d5.beta == 0.5)][list(LY.DELIVERED_COLUMNS)].copy()
    if len(d5) != len(elig):
        raise SystemExit('R5_REF_ROW_COUNT %d vs %d' % (len(d5), len(elig)))

    # WHAT THE DELIVERED PARQUET CAN AND CANNOT REBUILD.  `measure` needs a DENSE daily
    # series per station: `eventlib.build_event_table` asserts `n_base == 7` and
    # `n_peak == window + 2` on every event.  Round 5's delivered frame is the ELIGIBLE
    # GRID (12,152 rows = ~45% of the 15 x 1461 calendar), so it can carry the SD gate and
    # the monthly statistics -- both of which are defined on eligible station-days -- but it
    # CANNOT rebuild the event table.  Saying so is the honest form of "read round 5's
    # point": the event numbers come from round 5's own stored point, and the SD/monthly
    # numbers are RECOMPUTED here from its parquet, which makes them an independent
    # reproduction check of a stored number rather than a restatement of it.
    n_days = int(d5.date.nunique())
    dense = bool(len(d5) == N_STATIONS * n_days)
    sd5 = LY.station_sd_gate(d5, mask)
    mo5 = C.monthly_stats(d5, elig, obs_m)
    repro5 = {k: bool(sd5[L]['ddof0']['median_e'] == p5['sd_L3_ddof0_median_e'])
              for L in ('L3',)}
    repro5['nse'] = bool(mo5['nse'] == p5['nse'])
    repro5['median_station_nse'] = bool(mo5['median_station_nse'] == p5['median_station_nse'])
    repro5['mean_concentration'] = bool(mo5['mean_concentration'] == p5['mean_concentration'])
    frames.append(d5.assign(arm='R5-ref'))
    rows['R5-ref'] = dict(
        arm='R5-ref', role='reference', installs_kernel=False, Vu_post=None,
        # --- event readings, from round 5's own stored point
        A_L1=p5['A_L1'], A_L2=p5['A_L2'], A_L3=p5['A_L3'],
        c_base_L1=p5['c_base_L1'], c_peak_L1=p5['c_peak_L1'],
        c_base_L3=p5['c_base_L3'], c_peak_L3=p5['c_peak_L3'],
        beta_hat=p5['beta_hat'], alpha_hat=p5['alpha_hat'],
        A_L1_by_sign=p5['A_L1_by_sign'], A_L1_by_sign_base=p5['A_L1_by_sign_base'],
        # --- level / monthly / SD, RECOMPUTED here from round 5's delivered parquet
        nse=mo5['nse'], r2=mo5['r2'], median_station_nse=mo5['median_station_nse'],
        mean_concentration=mo5['mean_concentration'],
        n_station_months=mo5['n_station_months'], n_eligible_rows=mo5['n_eligible_rows'],
        sd_L1_ddof0_median_e=sd5['L1']['ddof0']['median_e'],
        sd_L2_ddof0_median_e=sd5['L2']['ddof0']['median_e'],
        sd_L3_ddof0_median_e=sd5['L3']['ddof0']['median_e'],
        sd_all_layers={L: dict(ddof0=sd5[L]['ddof0']['median_e'],
                               ddof1=sd5[L]['ddof1']['median_e'],
                               ratio_mdl_over_obs_median=
                               sd5[L]['ddof0']['ratio_mdl_over_obs_median'],
                               n_stations=sd5[L]['ddof0']['n_stations'])
                       for L in ('L1', 'L2', 'L3')},
        sd_gating_layer=sd5['gating_layer'],
        sd_ddof_choice_is_inert=sd5['L3']['ddof_choice_is_inert'],
        budget=p5['budget'], monthly=p5['monthly'],
        ledger=None, tau=None, s2_7=None, n_events=int(p5['n_events']),
        source='20260919_5/reports/daily_layers.parquet',
        point_source='20260919_5/reports/phase1_full.json::points[N1e|0.5]',
        device='N1e', beta=0.5, source_sha256=C.sha(r5p),
        point_source_sha256=C.sha(r5j),
        n_rows_in_file=n5_all, n_rows_selected=int(len(d5)),
        n_unique_days=n_days, frame_is_dense=dense,
        n_forwards_used=0, enters_no_gate=True,
        dense_event_series_unavailable=not dense,
        event_readings_from='round 5 own stored point, NOT rebuilt here',
        stored_point_reproduced=repro5,
        stored_point_fully_reproduced=bool(all(repro5.values())),
        why='a round-5 point is a different kernel on a different arm table; it is read so '
            'a drift is visible, never so a criterion can be met. Plan section 5: R5-ref '
            'readings do not participate in any gate, and this arm is given no `gates` '
            'block on purpose -- there is nothing here for a gate to be applied to.')
    _p('=== R5-ref READ (%d forwards used, %d/%d rows, dense=%s) ===  A_L1=%.6f A_L3=%.6f '
       'e_s=%.5f nse=%+.4f mnse=%+.4f meanC=%.4f | stored point reproduced %d/%d'
       % (0, len(d5), n5_all, dense, rows['R5-ref']['A_L1'], rows['R5-ref']['A_L3'],
          rows['R5-ref']['sd_L3_ddof0_median_e'], rows['R5-ref']['nse'],
          rows['R5-ref']['median_station_nse'], rows['R5-ref']['mean_concentration'],
          sum(repro5.values()), len(repro5)))
    if not rows['R5-ref']['stored_point_fully_reproduced']:
        _p('!!! ROUND 5 STORED POINT NOT REPRODUCED FROM ITS PARQUET: %r'
           % {k: v for k, v in repro5.items() if not v})

    # The reference arm's event readings must equal the ANCHORS, because the anchors were
    # taken from this very point.  That is a tautology about the anchor file, and it is
    # worth stating as one rather than letting it look like agreement between two sources.
    rep['R5_ref_is_the_anchor_source'] = dict(
        equal=all(rows['R5-ref'][k] == A[k] for k in
                  ('A_L1', 'A_L2', 'A_L3', 'c_base_L1', 'c_peak_L1', 'c_base_L3',
                   'c_peak_L3', 'beta_hat', 'alpha_hat')),
        note='the frozen anchors for these nine keys were read from '
             '`phase1_full.json::points[N1e|0.5]`, so equality is an identity of the anchor '
             'file, not an independent agreement. It is reported so the R5-ref row cannot '
             'be mistaken for a second measurement.')

    # ------------------------------------------------------------- the table first
    daily = pd.concat(frames, ignore_index=True)
    if len(daily) != len(elig) * len(rows):
        raise SystemExit('DAILY_ARMS_TOTAL %d vs %d' % (len(daily), len(elig) * len(rows)))
    E.assert_no_mass_columns(daily)
    daily.to_parquet(OUT / 'daily_arms.parquet', index=False)
    rep['daily_arms'] = dict(
        rows=int(len(daily)), per_arm=int(len(elig)), n_arms=int(len(rows)),
        path=str(OUT / 'daily_arms.parquet'), sha256=C.sha(OUT / 'daily_arms.parquet'),
        columns=list(daily.columns), arms=sorted(rows),
        row_count_matches_the_eligible_grid=True,
        dense_exactly_one_row_per_arm_per_eligible_station_day=True,
        note='every criterion below can be recomputed from this file with ZERO forwards')

    # ---------------------------------------------- B0 must reproduce the anchors
    base = rows['B0']
    repro = {k: bool(base[k] == A[k]) for k in STATION_LEVEL_KEYS}
    for k in ('sd_L3_ddof0_median_e', 'nse', 'median_station_nse', 'mean_concentration'):
        repro[k] = bool(base[k] == A[k])
    # L1/L2 have no `*_median_e` anchor (only `sd_L1_ddof0` / `sd_L2_ddof0`), so they are
    # compared REPORTEDLY rather than gated: if the anchor's `sd_L*_ddof0` is the same
    # quantity as this round's `ddof0.median_e` for L3, it must be for L1/L2 as well, and
    # that is worth seeing -- but not worth failing the run over, since it was never
    # registered as a B0 obligation.
    extra_sd = {L: dict(arm=base['sd_all_layers'][L]['ddof0'], anchor=A['sd_%s_ddof0' % L],
                        equal=bool(base['sd_all_layers'][L]['ddof0']
                                   == A['sd_%s_ddof0' % L]))
                for L in ('L1', 'L2', 'L3') if ('sd_%s_ddof0' % L) in A}
    rep['B0_reproduces_the_frozen_anchors'] = dict(
        **repro, all=bool(all(repro.values())),
        n_checked=len(repro),
        sd_ddof0_cross_check=extra_sd,
        sd_ddof0_cross_check_all_equal=bool(all(v['equal'] for v in extra_sd.values())),
        note='bitwise equality against the registered anchors, not a tolerance. B0 runs '
             '`layers24.forward_layers(installed=False)`, i.e. the FROZEN Transport, '
             'through the SAME frame construction every other arm uses, so its agreement '
             'is evidence the two branches are shaped alike rather than merely both alive.')
    if not rep['B0_reproduces_the_frozen_anchors']['all']:
        raise SystemExit('B0_BASELINE_DRIFTED %r'
                         % {k: v for k, v in repro.items() if not v})
    _p('=== B0 reproduces every frozen anchor bitwise ===  %d/%d'
       % (sum(repro.values()), len(repro)))

    # ------------------------------------------------------------------ the gates
    bl = dict(nse=base['nse'], median_station_nse=base['median_station_nse'],
              mean_concentration=base['mean_concentration'],
              sd_L3_ddof0_median_e=base['sd_L3_ddof0_median_e'])
    for name, r in rows.items():
        if name == 'R5-ref':
            continue
        gate_block(r, A, G1T, G2T, sd_thr, bl, per_arm_evb)

    # ------------------------------------------------------- what the report needs
    rep['arms'] = rows
    rep['arms_table_sha256'] = arms['hashes']['table_sha256']
    rep['n_named_free_choices'] = len(arms['named_free_choices'])
    rep['named_free_choices'] = arms['named_free_choices']
    rep['N9'] = dict(
        n_fits=0, fit_worker_calls=0,
        PARAMETER_COUNT=int(len(C.parameters(TAG))),
        n_named_free_choices=len(arms['named_free_choices']),
        named_free_choices=arms['named_free_choices'],
        n_stations=N_STATIONS, n_reaches=int(model.data.fast_water.shape[1]),
        levels_or_amplitude_levers=0,
        note='no name in this round is a fitted parameter; the land-phase mobilisation '
             'carries none. Zero-fit here is STRUCTURAL, not budgetary (section 2.4: 9 of '
             'the 30 in use, 21 out of the calculation entirely), which makes it a '
             'STRONGER claim than round 5s and a NARROWER one: the frozen parameter '
             'envelope no longer constrains the land phase at all.')
    led_rows = {n: r for n, r in rows.items() if r.get('ledger')}
    rep['ledger_summary'] = {
        n: dict(local_balance_max_kg=r['ledger']['local_balance_max_kg'],
                network_balance_kg=r['ledger']['network_balance_kg'],
                network_scale_kg=r['ledger']['network_scale_kg'],
                source_label_sum_errors_max=r['ledger']['source_label_sum_errors_max'],
                source_label_rel_max=r['ledger']['source_label_rel_max'],
                all_hold=r['ledger']['all_hold'],
                ledger_body_ran=r['ledger']['ledger_body_ran'],
                n11_ii_reproved=r['ledger']['n11_ii_reproved'])
        for n, r in led_rows.items()}
    rep['N10'] = dict(
        tolerance=TOL, written_as='<=',
        anchors_ok=bool(rep['anchor_replay']['passed']),
        anchor_rows=int(rep['anchor_replay']['n_compared_rows']),
        anchor_max_abs_dp=float(rep['anchor_replay']['max_abs_elementwise_concentration']),
        max_local_balance_kg=max(r['ledger']['local_balance_max_kg']
                                 for r in led_rows.values()),
        max_abs_network_balance_kg=max(abs(r['ledger']['network_balance_kg'])
                                       for r in led_rows.values()),
        max_label_error=max(r['ledger']['source_label_sum_errors_max']
                            for r in led_rows.values()),
        max_label_rel_error=max(r['ledger']['source_label_rel_max']
                               for r in led_rows.values()),
        n_label_channels=len(LABEL_CHANNELS),
        all_channels_present=bool(all(not r['ledger']['source_label_channels_missing']
                                      for r in led_rows.values())),
        all_arms_hold=bool(all(r['ledger']['all_hold'] for r in led_rows.values())),
        all_ledger_bodies_rebound=bool(all(r['ledger']['ledger_body_ran'] == 'ledger_dp'
                                           for r in led_rows.values())),
        N11_i_shapes_asserted_in_every_wrapper=True,
        note='the six-channel absolute 1e-6 kg label residual is registered as a STOP in '
             'THIS round (plan 3.5 N10), unlike round 5 which reported it without '
             'consulting it. It is implemented as a stop, the RELATIVE residual is '
             'additionally reported because the gate is absolute against a ~1e8 kg river, '
             'and had it fired the round would have reported BLOCKED with its cause named '
             'rather than relaxed after seeing the data.')
    rep['N12'] = dict(**XI.expm1_underflow_probe(1e-200),
                      source_scan=('no `1 - np.exp(` / `1 - exp(` / `1.0 - np.exp(` occurs '
                                   'in dp_kernel.py; the exponential form is written '
                                   '`-np.expm1(-x)`'),
                      declared_form=arms['closure_form'],
                      form_note='the round settled on `g = x`, for which phi is identically '
                                '1; the expm1 spelling is still asserted because the '
                                'closure set is closed and the alternative must be '
                                'reachable without a rewrite')
    rep['N6_level'] = {n: dict(mean_concentration=r['mean_concentration'],
                               ratio_to_frozen=float(r['mean_concentration']
                                                     / bl['mean_concentration']),
                               level_gate=LEVEL_GATE,
                               in_gate=bool(abs(r['mean_concentration']
                                                - bl['mean_concentration'])
                                            / bl['mean_concentration'] <= LEVEL_GATE))
                       for n, r in rows.items()}
    rep['N6_note'] = ('this round has NO level lever (unlike round 5, which had k_r), so a '
                      'level failure and an amplitude failure are NOT separable and '
                      'neither may be read as evidence about the other.')
    rep['P2'] = {n: r['tau'].get('P2_slow_path') for n, r in rows.items()
                 if isinstance(r.get('tau'), dict) and 'P2_slow_path' in r['tau']}
    rep['P1_P3_P4'] = {n: {k: r[k] for k in ('tau', 'P3_pulse', 'P4_closure_sensitivity',
                                             'C_distribution')
                           if k in r} for n, r in rows.items()}
    rep['cross_arm'] = {k: {n: r.get(k) for n, r in rows.items()}
                        for k in ('A_L1', 'A_L2', 'A_L3', 'sd_L3_ddof0_median_e', 'nse',
                                  'median_station_nse', 'mean_concentration')}
    rep['s_M_reading'] = dict(
        note='s_M is NOT negligible, and the two numbers that say so are in the per-arm '
             'block rather than repeated here: its RELATIVE band is a fraction of a '
             'percent, while the LIFETIMES it implies span an order of magnitude. A small '
             'band in a survival factor that compounds daily is not a small effect. The '
             'two lifetimes must never share one name (plan R2 / S2.6 P1). Source: '
             '`model.flux_parameters(t)[1]`, i.e. the per-reach survival.',
        per_arm={n: dict(s_M_range=r['tau']['s_M_range'],
                         implied_lifetime_days=r['tau']['s_M_implied_lifetime_days'],
                         band_relative=r['tau']['s_M_band_relative'])
                 for n, r in rows.items()
                 if isinstance(r.get('tau'), dict) and 's_M_range' in r['tau']})
    rep['seconds_total'] = round(time.time() - t_start, 2)
    rep['forward_count'] = int(sum(1 for a in arms['arms'] if a['installs_kernel']) + 1)
    rep['gates_evaluated_after_all_forwards'] = True

    C.write_json(OUT / 'phase1_arms.json', rep)
    _p('=== wrote %s  (%.1fs, %d forwards) ==='
       % (OUT / 'phase1_arms.json', rep['seconds_total'], rep['forward_count']))
    _p('    daily_arms.parquet: %d rows = %d arms x %d eligible station-days'
       % (len(daily), len(rows), len(elig)))
    for n, r in rows.items():
        if 'gates' in r:
            _p('   %-8s five=%d/5  G1=%s G2=%s G3=%s G5=%s G5b=%s | G4=%s'
               % (n, r['n_gates_passed_main_five'], r['G1_pass'], r['G2_pass'],
                  r['G3_pass'], r['G5_pass'], r['G5b_pass'], r['G4_pass']))
        else:
            _p('   %-8s (reference: enters no gate)' % n)


if __name__ == '__main__':
    main()
