"""F0.2b -- the within-month axis the 116-station monthly panel cannot see.

`work/f0_phenotype.py` classifies 116 stations on monthly sample-months. FCT8's
claim, though, is about *within-month* timing: it splits the latent score into a
month-constant part and a within-month de-meaned part. A monthly panel has one
number per (station, month) and therefore cannot observe that axis at all -- two
stations with identical monthly series can differ completely in how the days
inside each month are allocated.

The 15 high-frequency stations are the only stations with more than one sample per
month, so they are the only place the within-month axis is observable. This script
reads their 2024 evaluated days and decomposes each station's variance into the
two parts FCT8 is built to separate:

    SS_total  = SS_between_month + SS_within_month

giving three ratios that answer one question directly: **is the model's daily
amplitude loss in the month-to-month component or inside the month?** If the
within-month ratio is as damped as the total, the loss lives precisely in the
channel FCT8 reshapes; if the within-month ratio is near 1 and only the
month-to-month component collapsed, FCT8's channel is not where the loss is and
no amount of level/timing decoupling inside a month can recover it.

This pairs with work/f0_saturation.py, which measured the *multiplier's* own
within-month daily spread as healthy (sd 0.432, 0% railed). Together they localize
the damping: healthy multiplier + collapsed end-to-end within-month spread means
the loss is downstream of the multiplier (the M/L reservoir and the routing
operators), not in the soft saturation FCT8 edits.

Zero fits: reads a published evaluated parquet and nothing else. Thresholds are
frozen here in code before any FCT8 result exists and are copied into the output
json so they cannot be re-chosen afterwards.

Output: reports/f0_phenotype_hf.parquet, reports/f0_phenotype_hf.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RUN = Path(__file__).resolve().parents[1]
PARENT = RUN.parent / '20260917_5'
EVAL_YEAR = 2024

TAGS = {'D': 'T24_G_D_H1_s0', 'M': 'T24_G_M_H1_s0'}

#: frozen before any FCT8 result exists; the same 0.50 family the monthly table uses
T_HF_LEVEL, T_HF_AMP, T_HF_TIMING = 0.50, 0.50, 0.50
#: a station is only interpretable on the within-month axis if it has enough
#: sample days inside months to estimate a within-month variance at all
T_HF_MIN_DAYS, T_HF_MIN_MONTHS = 60, 6
#: months contributing to the within-month pool need at least this many sample days
T_HF_MIN_DAYS_PER_MONTH = 3


def station_axes(g):
    """Total / between-month / within-month decomposition for one station."""
    y = g.y.to_numpy(float)
    p = g.p.to_numpy(float)
    m = g.month.to_numpy()
    ok = np.isfinite(y) & np.isfinite(p)
    y, p, m = y[ok], p[ok], m[ok]
    out = dict(n_days=int(len(y)), n_months=int(len(np.unique(m))),
               months_usable=0, usable=False)
    if len(y) < T_HF_MIN_DAYS or len(np.unique(m)) < T_HF_MIN_MONTHS:
        return out

    # only months with enough sample days can carry a within-month variance
    keep = np.zeros(len(y), bool)
    for mo in np.unique(m):
        sel = m == mo
        keep |= sel if sel.sum() >= T_HF_MIN_DAYS_PER_MONTH else False
    y, p, m = y[keep], p[keep], m[keep]
    out['n_days'] = int(len(y))
    out['n_months'] = int(len(np.unique(m)))
    out['months_usable'] = int(len(np.unique(m)))
    if len(np.unique(m)) < T_HF_MIN_MONTHS:
        return out

    my, mp = y.mean(), p.mean()
    ss_tot_o = float(((y - my) ** 2).sum())
    ss_tot_p = float(((p - mp) ** 2).sum())
    ss_win_o = ss_win_p = 0.0
    ya, pa = np.zeros_like(y), np.zeros_like(p)          # within-month anomalies
    for mo in np.unique(m):
        sel = m == mo
        ym, pm = y[sel].mean(), p[sel].mean()
        ya[sel] = y[sel] - ym
        pa[sel] = p[sel] - pm
        ss_win_o += float(((y[sel] - ym) ** 2).sum())
        ss_win_p += float(((p[sel] - pm) ** 2).sum())
    ss_btw_o, ss_btw_p = ss_tot_o - ss_win_o, ss_tot_p - ss_win_p

    def ratio(sp, so):
        return float(np.sqrt(sp / so)) if so > 0 and sp > 0 else np.nan

    def corr(a, b):
        if a.std() <= 0 or b.std() <= 0:
            return np.nan
        return float(np.corrcoef(a, b)[0, 1])

    out.update(
        sd_obs=float(y.std(ddof=1)), sd_pred=float(p.std(ddof=1)),
        bias=float(mp - my),
        level=float((mp - my) / y.std(ddof=1)) if y.std(ddof=1) > 0 else np.nan,
        r=corr(y, p),
        NSE=(float(1 - ss_tot_p / ss_tot_o) if ss_tot_o > 0 else np.nan),
        # the three ratios: total daily, between-month, within-month
        sd_ratio_total=ratio(ss_tot_p, ss_tot_o),
        sd_ratio_between=ratio(ss_btw_p, ss_btw_o),
        sd_ratio_within=ratio(ss_win_p, ss_win_o),
        log_sd_ratio_total=float(np.log(ratio(ss_tot_p, ss_tot_o))),
        log_sd_ratio_between=float(np.log(ratio(ss_btw_p, ss_btw_o))),
        log_sd_ratio_within=float(np.log(ratio(ss_win_p, ss_win_o))),
        r_between=corr(np.array([y[m == mo].mean() for mo in np.unique(m)]),
                       np.array([p[m == mo].mean() for mo in np.unique(m)])),
        r_within=corr(ya, pa),
        # what share of the observed daily signal lives inside months at all
        within_share_obs=float(ss_win_o / ss_tot_o) if ss_tot_o > 0 else np.nan,
        within_share_pred=float(ss_win_p / ss_tot_p) if ss_tot_p > 0 else np.nan,
        usable=True)
    return out


def classify(tab):
    """Same rule family as the monthly table, applied to the within-month axes."""
    z_level = tab['level'].abs() / T_HF_LEVEL
    z_amp = tab['log_sd_ratio_total'].abs() / T_HF_AMP
    z_time = (T_HF_TIMING - tab['r_within']).clip(lower=0) / T_HF_TIMING
    stacked = np.vstack([z_level.fillna(0), z_amp.fillna(0), z_time.fillna(0)])
    pick = np.array(['LEVEL', 'AMPLITUDE', 'TIMING'])[np.argmax(stacked, axis=0)]
    tab['excess_level'], tab['excess_amplitude'] = z_level, z_amp
    tab['excess_timing_within'] = z_time
    tab['axes_over_threshold'] = [
        '+'.join(a for a, v in (('LEVEL', l), ('AMPLITUDE', m), ('TIMING', t))
                 if np.isfinite(v) and v >= 1.0) or 'none'
        for l, m, t in zip(z_level, z_amp, z_time)]
    # A station with n_months < T_HF_MIN_MONTHS has no axes at all: every excess is
    # NaN, `fillna(0)` makes them all zero, and argmax then returns index 0, i.e. the
    # station is silently labelled LEVEL for having no data. Name that case instead.
    tab['phenotype_hf'] = np.where(tab.usable, pick, 'UNUSABLE')
    return tab


def main():
    tables, out = {}, {}
    for arm, tag in TAGS.items():
        src = PARENT / 'reports' / ('%s_HF_evaluated.parquet' % tag)
        frame = pd.read_parquet(src)
        frame = frame[frame.year == EVAL_YEAR]
        rows = []
        for key, g in frame.groupby('station_key'):
            r = station_axes(g)
            r['station_key'] = key
            r['cohort'] = g.cohort.iloc[0]
            r['hf_group'] = g.hf_group.iloc[0]
            r['has_daily_training'] = bool(g.has_daily_training.iloc[0])
            r['training_location'] = bool(g.training_location.iloc[0])
            rows.append(r)
        tab = classify(pd.DataFrame(rows))
        tab['tag'], tab['arm'] = tag, arm
        tables[arm] = tab
        u = tab[tab.usable]
        within_damped = int((u.log_sd_ratio_within <= -T_HF_AMP).sum())
        out[arm] = dict(
            tag=tag, source=str(src.relative_to(RUN.parent)),
            stations=int(len(tab)), usable=int(len(u)),
            days_total=int(tab.n_days.sum()),
            # the headline reading of this file: how much of the daily amplitude
            # loss sits INSIDE months, which is the only part FCT8 can reach
            median_sd_ratio_total=float(np.exp(u.log_sd_ratio_total.median())),
            median_sd_ratio_between=float(np.exp(u.log_sd_ratio_between.median())),
            median_sd_ratio_within=float(np.exp(u.log_sd_ratio_within.median())),
            median_log_sd_ratio_total=float(u.log_sd_ratio_total.median()),
            median_log_sd_ratio_between=float(u.log_sd_ratio_between.median()),
            median_log_sd_ratio_within=float(u.log_sd_ratio_within.median()),
            stations_within_damped=within_damped,
            stations_between_damped=int((u.log_sd_ratio_between <= -T_HF_AMP).sum()),
            median_r_total=float(u.r.median()), median_r_between=float(u.r_between.median()),
            median_r_within=float(u.r_within.median()),
            median_within_share_obs=float(u.within_share_obs.median()),
            median_within_share_pred=float(u.within_share_pred.median()),
            median_NSE=float(u.NSE.median()),
            phenotype_hf_counts=u.phenotype_hf.value_counts().to_dict(),
        )
        print('\n=== HF %s (%s): %d stations, %d usable ===' % (arm, tag, len(tab), len(u)))
        print('  median sd ratio: total %.3f   between-month %.3f   within-month %.3f'
              % (out[arm]['median_sd_ratio_total'], out[arm]['median_sd_ratio_between'],
                 out[arm]['median_sd_ratio_within']))
        print('  median r       : total %.3f   between-month %.3f   within-month %.3f'
              % (out[arm]['median_r_total'], out[arm]['median_r_between'],
                 out[arm]['median_r_within']))
        print('  stations damped: within %.1f/15   between %.1f/15'
              % (within_damped, out[arm]['stations_between_damped']))
        print('  median within-month share of observed variance: %.3f'
              % out[arm]['median_within_share_obs'])
        print(u.phenotype_hf.value_counts().to_string())

    d = tables['D']
    d.to_parquet(RUN / 'reports/f0_phenotype_hf.parquet', index=False)
    payload = dict(
        status='F0_HF_WITHIN_MONTH_DIAGNOSIS_ONLY', evaluation_year=EVAL_YEAR, zero_fits=True,
        note=('The 116-station monthly panel has one value per (station, month) and cannot '
              'observe the within-month axis FCT8 acts on. These 15 stations have several '
              'sample days per month, so SS_total = SS_between_month + SS_within_month is '
              'estimable for them and the three ratios localize the amplitude loss.'),
        thresholds=dict(T_HF_LEVEL=T_HF_LEVEL, T_HF_AMP=T_HF_AMP, T_HF_TIMING=T_HF_TIMING,
                        T_HF_MIN_DAYS=T_HF_MIN_DAYS, T_HF_MIN_MONTHS=T_HF_MIN_MONTHS,
                        T_HF_MIN_DAYS_PER_MONTH=T_HF_MIN_DAYS_PER_MONTH),
        classification_rule=('largest threshold excess among |level|/T_HF_LEVEL, '
                             '|log sd ratio total|/T_HF_AMP, (T_HF_TIMING - r_within)+/T_HF_TIMING'),
        sampling_note=('Sample days are grab-sample days, not a complete daily series: the '
                       'observed gaps run 1..16 days. `y` is the day-level composite of `n` '
                       'records and `alpha` is the evaluator\'s week weight. Values are analysed '
                       'unweighted, matching work/f0_phenotype.py, so the two tables are '
                       'comparable; the weight is not used anywhere in this round.'),
        residual_red_herring=('A first glance at consecutive days with identical `p` suggested '
                              'the daily prediction was broadcast from a coarser unit, which would '
                              'have made the amplitude collapse an evaluator artefact. Checked: '
                              '`p` takes ~as many distinct values as there are sample days '
                              '(e.g. 220 days / 220 values at one station), so the prediction is '
                              'resolved per day and the collapse is not an aggregation artefact. '
                              'Recorded because the suspicion would otherwise recur.'),
        pairs_with=('work/f0_saturation.py measured the multiplier\'s own within-month spread as '
                    'healthy (sd 0.43213, 0 percent at a rail). Read together: the soft '
                    'saturation is not the constraint, so if the end-to-end within-month ratio '
                    'below is damped, the loss is downstream of the multiplier in the M/L '
                    'reservoir and routing operators -- which FCT8 does not touch.'),
        combined_reading=(
            'THREE numbers, one location. (1) 43.1 percent of the observed daily variance at '
            'these stations is within-month (median within_share_obs 0.4311), but the model '
            'places only 15.4 percent of its own variance there (median within_share_pred '
            '0.1536): it converts a within-month-varying signal into a mostly between-month '
            'one. (2) The within-month component is far worse than the between-month one -- '
            'median sd ratio 0.254 within versus 0.512 between, and median r 0.082 within '
            'versus 0.597 between. The model reproduces month means and misses days. (3) The '
            'multiplier is NOT the constraint: its own within-month daily spread is healthy '
            '(sd 0.43213, 0.00 percent railed, 92.37 percent in the tanh linear band). So the '
            'within-month freedom exists at the hazard and only about a third of it survives to '
            'the prediction. The loss is therefore downstream of the point FCT8 edits, in the '
            'M/L reservoir and routing operators, which FCT8 leaves byte-identical (V6).'),
        registered_prediction=(
            'Written before F2, from the three numbers above, and hashed by gate0. FCT8 can only '
            'reallocate which part of the latent score is month-constant; it cannot add a path '
            'for within-month movement to survive the downstream chain. So the falsifiable '
            'split: (a) IF the level/timing conflict is the real defect, FCT8 raises r_within '
            'and within_share_pred -- an allocation gain; (b) FCT8 should NOT recover '
            'sd_ratio_within (0.254), because the multiplier is not amplitude-constrained. '
            'Consequence to state in advance: the seven promotion thresholds are dominated by '
            'RMSE and amplitude terms, so on this prediction FCT8 can confirm its mechanism and '
            'still fail promotion. Should that happen, both readings must be reported together '
            '-- mechanism confirmed, promotion not met -- and neither may be dropped. '
            'Falsifier: FCT8 drives sd_ratio_within toward 1. That would mean the multiplier\'s '
            'within-month SHAPE, not merely its allocation, was the binding constraint after '
            'all, and would contradict the saturation audit.'),
        arms=out)
    (RUN / 'reports/f0_phenotype_hf.json').write_text(
        json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True, default=str),
        encoding='utf-8')
    print('\nwrote reports/f0_phenotype_hf.parquet (%d rows) and f0_phenotype_hf.json' % len(d))


if __name__ == '__main__':
    sys.exit(main())
