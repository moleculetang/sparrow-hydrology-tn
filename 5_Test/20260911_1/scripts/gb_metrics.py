"""Registered station-wise decisions; heldout labels are used only for scoring."""
import numpy as np
import pandas as pd
from fc_io import RUN, TEST, read, write, local, sha
from gb_metrics_base import stable_metrics

ORDER = ('M0', 'MF', 'MG', 'MGF')
TOL = 1e-12


def metrics(frame):
    if frame.loc[frame.primary_gate].empty:
        return pd.DataFrame(), {'rows': 0, 'stations': 0, 'status': 'NO_ELIGIBLE_OBSERVATIONS'}
    table, summary = stable_metrics(frame)
    table['alpha'] = table.sd_ratio
    table['delta'] = np.nan
    table['nse_identity'] = np.nan
    for station, group in frame.loc[frame.primary_gate].groupby('station_key'):
        pos = table.station_key.eq(station)
        if not bool(table.loc[pos, 'dynamic_eligible'].iloc[0]):
            continue
        y = group.tn_mg_l.to_numpy(float)
        p = group.prediction_mg_l.to_numpy(float)
        delta = (p.mean() - y.mean()) / y.std()
        alpha = float(table.loc[pos, 'alpha'].iloc[0])
        # A constant prediction has undefined Pearson r. The cross-covariance
        # term is still exactly zero; retain the undefined-correlation flag.
        r = float(table.loc[pos, 'time_r'].iloc[0]) if np.ptp(p) else 0.
        identity = r*r - (alpha-r)**2 - delta*delta
        np.testing.assert_allclose(identity, table.loc[pos, 'nse'].iloc[0], rtol=1e-10, atol=1e-10)
        table.loc[pos, ['delta', 'nse_identity']] = [delta, identity]
    summary.update(mean_station_raw_rmse=float(table.rmse_mg_l.mean()),
                   mean_station_absolute_bias=float(table.bias_mg_l.abs().mean()),
                   undefined_r_fraction=summary['undefined_r']/summary['dynamic_eligible'] if summary['dynamic_eligible'] else None,
                   nse_identity_verified=True)
    return table, summary


def anomaly_metrics(training, evaluation):
    """Separate observed/predicted training station-month climatologies.

    Two TRAINING records per station/month and eight EVALUATION anomalies are
    mandatory. Never center by evaluation labels or predictions.
    """
    if set(training.observation_id) & set(evaluation.observation_id):
        raise ValueError('ANOMALY_TRAIN_EVALUATION_OVERLAP')
    train = training.loc[training.primary_gate]
    ref = train.groupby(['station_key', 'month']).agg(
        count=('tn_mg_l', 'size'), observed_mean=('tn_mg_l', 'mean'), predicted_mean=('prediction_mg_l', 'mean'))
    ref = ref.loc[ref['count'] >= 2].reset_index()
    rows = evaluation.loc[evaluation.primary_gate].merge(ref, on=['station_key', 'month'], how='left', validate='many_to_one')
    rows['observed_anomaly'] = rows.tn_mg_l - rows.observed_mean
    rows['predicted_anomaly'] = rows.prediction_mg_l - rows.predicted_mean
    result = []
    for station, group in rows.groupby('station_key', sort=True):
        valid = group.dropna(subset=['observed_anomaly', 'predicted_anomaly'])
        y = valid.observed_anomaly.to_numpy(float); p = valid.predicted_anomaly.to_numpy(float)
        eligible = len(y) >= 8 and np.ptp(y) > 0
        corr = float(np.corrcoef(y, p)[0, 1]) if eligible and np.ptp(p) > 0 else np.nan
        result.append({'station_key': station, 'valid_anomaly_rows': len(valid),
                       'anomaly_eligible': eligible, 'anomaly_r': corr})
    table = pd.DataFrame(result, columns=['station_key', 'valid_anomaly_rows', 'anomaly_eligible', 'anomaly_r'])
    defined = table.anomaly_r.dropna()
    summary = {'stations': len(table), 'eligible_stations': int(table.anomaly_eligible.sum()),
               'defined_stations': len(defined), 'median_anomaly_r': float(defined.median()) if len(defined) else None,
               'valid_rows': int(table.valid_anomaly_rows.sum()), 'climatology': 'separate_training_observed_and_predicted_station_month_means'}
    return table, rows, summary


def same_evaluation(a, b):
    cols = ['observation_id', 'station_key', 'year', 'month', 'reach_id', 'tn_mg_l', 'primary_gate']
    aa = a[cols].sort_values('observation_id').reset_index(drop=True)
    bb = b[cols].sort_values('observation_id').reset_index(drop=True)
    if aa.observation_id.duplicated().any() or bb.observation_id.duplicated().any():
        raise ValueError('DUPLICATE_EVALUATION_IDS')
    pd.testing.assert_frame_equal(aa, bb, check_dtype=False)


def finite_values(*values):
    return all(v is not None and np.isfinite(v) for v in values)


def information_gate(baseline, candidate, kind, base_anomaly=None, candidate_anomaly=None, design_usable=True, distinct=True):
    """G1 evidence permits a physical experiment, never mechanism identification."""
    checks = {'usable_training_design': bool(design_usable), 'distinct_from_old_code': bool(distinct)}
    lr0, lr1 = baseline.get('mean_station_log_rmse'), candidate.get('mean_station_log_rmse')
    n0, n1 = baseline.get('median_nse'), candidate.get('median_nse')
    checks['log_rmse_improvement_2_percent'] = finite_values(lr0, lr1) and lr0 > 0 and lr1 <= .98*lr0 + TOL
    if kind == 'MG':
        r0, r1 = baseline.get('median_time_r'), candidate.get('median_time_r')
        checks['median_nse_gain_005'] = finite_values(n0, n1) and n1-n0 >= .05-TOL
        checks['median_r_nondecreasing'] = finite_values(r0, r1) and r1 >= r0-TOL
    elif kind == 'MF':
        r0 = (base_anomaly or {}).get('median_anomaly_r')
        r1 = (candidate_anomaly or {}).get('median_anomaly_r')
        checks['median_nse_nondecreasing'] = finite_values(n0, n1) and n1 >= n0-TOL
        checks['anomaly_median_r_gain_005'] = finite_values(r0, r1) and r1-r0 >= .05-TOL
    else:
        raise ValueError(kind)
    return {'kind': kind, 'passed': all(checks.values()), 'checks': checks,
            'interpretation': 'permission_for_registered_physical_test_only'}


def research_gate(baseline, candidate, base_annual, candidate_annual, years):
    b, c = baseline, candidate
    checks = {}
    def delta(key, threshold):
        return finite_values(b.get(key), c.get(key)) and c[key]-b[key] >= threshold-TOL
    checks['median_nse_gain_010'] = delta('median_nse', .10)
    checks['median_r_gain_005'] = delta('median_time_r', .05)
    checks['positive_median_r'] = finite_values(c.get('median_time_r')) and c['median_time_r'] > 0
    checks['q25_nse_decrease_at_most_005'] = delta('q25_nse', -.05)
    for key in ['negative_r_fraction', 'undefined_r_fraction']:
        checks[key+'_not_increased'] = finite_values(b.get(key), c.get(key)) and c[key] <= b[key]+TOL
    for key in ['mean_station_log_rmse', 'mean_station_raw_rmse', 'mean_station_absolute_bias']:
        checks[key+'_at_most_105_percent'] = finite_values(b.get(key), c.get(key)) and c[key] <= 1.05*b[key]+TOL
    annual0 = {int(r['year']): r for r in base_annual}
    annual1 = {int(r['year']): r for r in candidate_annual}
    differences = {}
    for year in sorted(set(map(int, years))):
        r0 = annual0.get(year, {}).get('uptake_demand_ratio')
        r1 = annual1.get(year, {}).get('uptake_demand_ratio')
        differences[str(year)] = r1-r0 if finite_values(r0, r1) else None
    checks['annual_uptake_ratio_drop_at_most_010'] = bool(differences) and all(
        v is not None and v >= -.10-TOL for v in differences.values())
    return {'passed': all(checks.values()), 'checks': checks, 'uptake_ratio_changes': differences,
            'amplitude_reported_not_a_sole_stop': {'baseline': b.get('d_sigma'), 'candidate': c.get('d_sigma'),
                                                 'baseline_infinite': b.get('d_sigma_infinite'), 'candidate_infinite': c.get('d_sigma_infinite')},
            'formal_promotion': False}


def select_candidate(candidates):
    """Development scoring selects configuration, never optimizer start.

    Anchor the .01 tie set to the greatest median NSE to avoid non-transitive
    pairwise ties. Parameter count breaks ties, then the registered order.
    """
    passed = [r for r in candidates if r['gate']['passed']]
    if not passed:
        return None
    best = max(r['metrics']['median_nse'] for r in passed)
    tied = [r for r in passed if best-r['metrics']['median_nse'] <= .01+TOL]
    return min(tied, key=lambda r: (r['new_parameters'], ORDER.index(r['variant'])))


def report_predictions(tag, evaluation, training=None):
    outputs = {}; table, summary = metrics(evaluation)
    path = local(RUN/'outputs/metrics'/f'{tag}_stations.parquet'); table.to_parquet(path, index=False)
    outputs[str(path)] = sha(path)
    cohorts = read(TEST/'20260908_1/configs/cohorts.json')
    groups = {}
    for name in ['common', 'late_only', 'old_only']:
        # Cohort membership is fixed by the prior coverage audit, not fit results.
        if name not in cohorts:
            raise KeyError('UNREGISTERED_COHORT '+name)
        selected = evaluation.loc[evaluation.station_key.isin(cohorts[name])]
        groups[name] = metrics(selected)[1]
    years = {str(int(y)): metrics(g)[1] for y, g in evaluation.groupby('year')}
    result = {'metrics': summary, 'cohorts': groups, 'by_year': years, 'artifacts_sha256': outputs}
    if training is not None:
        stations, rows, anomaly = anomaly_metrics(training, evaluation)
        for suffix, frame in [('anomaly_stations', stations), ('anomaly_rows', rows)]:
            path = local(RUN/'outputs/metrics'/f'{tag}_{suffix}.parquet'); frame.to_parquet(path, index=False)
            outputs[str(path)] = sha(path)
        result['anomaly'] = anomaly
    write(RUN/'reports/metrics'/f'{tag}.json', result)
    return result
