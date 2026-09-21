"""Gate 0: everything that must hold before a single parameter is fitted.

Nothing here fits anything.  Each check answers a question that, if answered
wrongly *after* 41 hours of fitting, would make the answer uninterpretable:
whether the corrected hydrology is internally consistent, whether the topology
moved with the water, whether the soil-wetness denominator belongs to the line
it divides, whether the headline cohort actually exists, and which stations carry
the daily criterion.

The stop thresholds of the round are written into this report, with the sha256 of
their verbatim text, so that the pre-registration can be checked afterwards
rather than remembered.

`20260905_1.audit_inputs.hydro_audit` is deliberately NOT called.  It writes
`{product}_zero_flow_months.parquet` into its own round's `outputs/`, which this
round must not touch; its assertions are reimplemented here read-only instead.
"""
from __future__ import annotations

import gc
import hashlib
import json
import re
import sys
import time
from pathlib import Path

RUN = Path(__file__).resolve().parents[1]
ROOT = RUN.parents[1]
import numpy as np                                                                  # noqa: E402
import pandas as pd                                                                 # noqa: E402
import pyarrow.parquet as pq                                                        # noqa: E402

CORRECTED = ROOT / '5_Test/20260917_3/work/screen/outputs'
FROZEN = ROOT / '5_Test/20260828_38/outputs'
SCREEN = ROOT / '5_Test/20260917_3/work/screen'

# The daily fields the 26-array derivation actually reads, named rather than
# implied: a silently missing one becomes NaN in a basis column, not an error.
REQUIRED_DAILY = ['date', 'reach_id', 'local_fast_response_m3_s', 'local_slow_response_m3_s',
                  'routed_total_m3_s', 'routed_fast_response_m3_s', 'routed_slow_response_m3_s',
                  'routed_direct_response_m3_s', 'percolation_to_lower_mm_day',
                  'soil_storage_mm', 'actual_aet_mm_day', 'upper_response_storage_mm',
                  'lower_slow_storage_mm', 'tmean_c', 'catchment_area_km2', 'bankfull_depth_m']
DAILY = REQUIRED_DAILY
FLOW = ['routed_total_m3_s', 'routed_fast_response_m3_s', 'routed_slow_response_m3_s',
        'routed_direct_response_m3_s', 'local_fast_response_m3_s', 'local_slow_response_m3_s',
        'percolation_to_lower_mm_day', 'upper_response_storage_mm', 'lower_slow_storage_mm']
MONTHLY = ['month', 'reach_id', 'routed_total_m3_s', 'routed_fast_response_m3_s',
           'routed_slow_response_m3_s', 'routed_direct_response_m3_s']
FILES = ['tn_hydrology_reach_daily.parquet', 'tn_hydrology_reach_monthly.parquet',
         'tn_hydrology_reservoir_daily.parquet',
         'tn_hydrology_reservoir_static_metadata.parquet']

PREREGISTRATION = """\
Delta = medNSE(H1) - medNSE(H0), per-station concentration NSE, evaluation year 2024.
Layer 1 -- worth keeping:      |Delta| < 0.05 end; 0.05 <= Delta < 0.10 downgrade but keep;
                               Delta >= 0.10 keep and enter Gate 2.
Negative tier, not folded into 'very small': Delta <= -0.05 is a substantive result and
                               must be explained; Delta <= -0.10 additionally requires
                               checking the capacity-scale confound (G0-5) first.
Layer 2 -- main bottleneck, all four simultaneously and separately from layer 1:
                               Delta >= 0.3;  Delta r >= 0.05;  RMSE(H1)/RMSE(H0) <= 0.9;
                               share of stations with Delta_i > 0 >= 60 percent; and the
                               1000-draw paired bootstrap 95 percent lower bound of Delta > 0
                               (seed 1729, bootstrap 1000, from configs/protocol.json).
Layers 1 and 2 are independent and must be reported side by side. A Delta of +0.25 is
'worth verifying further' AND 'not the main bottleneck' at the same time; writing it as
'hydrology is irrelevant' is the interpretation error this round guards against.
Load is not reported, not computed and not used as evidence anywhere in this round.
"""


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()


def g0_1_columns():
    """The corrected line must expose the same columns the derivation reads."""
    result = {}
    for name in FILES:
        new, old = pq.ParquetFile(CORRECTED / name), pq.ParquetFile(FROZEN / name)
        cn, co = new.schema.names, old.schema.names
        result[name] = dict(identical=cn == co, order_identical=cn == co,
                            columns=len(cn), columns_frozen=len(co),
                            rows=new.metadata.num_rows, rows_frozen=old.metadata.num_rows,
                            rows_identical=new.metadata.num_rows == old.metadata.num_rows,
                            missing=sorted(set(co) - set(cn)), extra=sorted(set(cn) - set(co)))
    absent = sorted(set(REQUIRED_DAILY) - set(pq.ParquetFile(
        CORRECTED / 'tn_hydrology_reach_daily.parquet').schema.names))
    mismatched = sorted(n for n, r in result.items() if not r['identical'])
    # The files must be the ones the producer locked, or H1 is derived from a
    # draft that happens to sit beside the attested artifact.
    lock = json.loads((ROOT / '5_Test/20260917_3/locks/product_lock.json').read_text(encoding='utf-8'))
    locked = {n: lock['products'][f'outputs/{n}']['sha256'] for n in FILES}
    provenance = {}
    for name in FILES:
        here = sha(CORRECTED / name)
        published = ROOT / '5_Test/20260917_3/outputs' / name
        provenance[name] = dict(read_by_this_round=here, locked=locked[name],
                                matches_lock=here == locked[name],
                                published_copy_matches_lock=(
                                    sha(published) == locked[name] if published.exists() else None))
    return dict(passed=not absent and not mismatched
                and all(v['matches_lock'] for v in provenance.values()),
                required_daily_fields=len(REQUIRED_DAILY), required_fields_absent=absent,
                failing_files=mismatched, files=result, lock_agreement=provenance)


def g0_2_consistency():
    """The hydrology product's own internal identities, reimplemented read-only.

    One arm at a time, each array freed before the next is read, so peak memory
    stays near a single copy of the daily table rather than two.
    """
    result = {}
    expected = pd.date_range('1961-01-01', '2025-12-31')
    for tag, directory in [('frozen', FROZEN), ('corrected', CORRECTED)]:
        d = pd.read_parquet(directory / 'tn_hydrology_reach_daily.parquet',
                            columns=DAILY).sort_values(['date', 'reach_id'])
        dates = pd.DatetimeIndex(pd.to_datetime(d.date.drop_duplicates()))
        days = len(dates)
        reach_ok = np.array_equal(d.reach_id.to_numpy().reshape(days, 230),
                                  np.broadcast_to(np.arange(1, 231), (days, 230)))
        routed = d.routed_total_m3_s.to_numpy()
        component = float(np.max(np.abs(
            routed - d[['routed_fast_response_m3_s', 'routed_slow_response_m3_s',
                        'routed_direct_response_m3_s']].sum(axis=1).to_numpy())))
        block = d[FLOW].to_numpy()
        flow_min, finite = float(block.min()), bool(np.isfinite(block).all())
        del block
        gc.collect()
        # Lower-store continuity: storage may only change by what percolates in
        # and what the local slow response removes.  86.4 converts m3/s to mm/day
        # over the reach's own catchment area.
        area = d.catchment_area_km2.to_numpy().reshape(days, 230)
        slow = d.local_slow_response_m3_s.to_numpy().reshape(days, 230) * 86.4 / area
        lower = d.lower_slow_storage_mm.to_numpy().reshape(days, 230)
        perc = d.percolation_to_lower_mm_day.to_numpy().reshape(days, 230)
        continuity = float(np.max(np.abs(lower[1:] - lower[:-1] - perc[1:] + slow[1:])))
        del d, slow, lower, perc, area
        gc.collect()
        m = pd.read_parquet(directory / 'tn_hydrology_reach_monthly.parquet',
                            columns=MONTHLY).sort_values(['month', 'reach_id'])
        monthly_component = float(np.max(np.abs(
            m.routed_total_m3_s.to_numpy()
            - m[['routed_fast_response_m3_s', 'routed_slow_response_m3_s',
                 'routed_direct_response_m3_s']].sum(axis=1).to_numpy())))
        del m
        gc.collect()
        r = pd.read_parquet(directory / 'tn_hydrology_reservoir_daily.parquet',
                            columns=['date', 'reservoir_entity_id', 'enabled', 'storage_m3',
                                     'total_release_m3', 'mass_balance_error_m3'])
        den = r.storage_m3.abs() + r.total_release_m3.abs()
        frac = np.divide(r.total_release_m3.to_numpy(), den.to_numpy(),
                         out=np.zeros(len(r)), where=den.to_numpy() > 1e-12)
        result[tag] = dict(
            directory=str(directory), days=days, reaches=230, rows=days * 230,
            calendar_exact=bool(dates.equals(expected)), reaches_tiled=bool(reach_ok),
            routed_component_max_error=component, monthly_component_max_error=monthly_component,
            lower_water_continuity_max_abs_mm=continuity,
            flow_minimum=flow_min, finite=finite, nonnegative=bool(flow_min >= -1e-10),
            reservoir_rows=len(r), reservoir_entities=int(r.reservoir_entity_id.nunique()),
            reservoir_mass_balance_max_abs_m3=float(r.mass_balance_error_m3.abs().max()),
            reservoir_release_fraction_min=float(frac.min()),
            reservoir_release_fraction_max=float(frac.max()),
            reservoir_duplicate_keys=int(r.duplicated(['date', 'reservoir_entity_id']).sum()),
            reservoir_negative=bool((r.storage_m3.min() < 0) or (r.total_release_m3.min() < 0)))
        del r, den, frac
        gc.collect()
    checks = []
    for tag, r in result.items():
        checks.extend([
            (f'{tag}.calendar_exact', r['calendar_exact']),
            (f'{tag}.reaches_tiled', r['reaches_tiled']),
            (f'{tag}.routed_component', r['routed_component_max_error'] < 1e-9),
            (f'{tag}.monthly_component', r['monthly_component_max_error'] < 1e-9),
            (f'{tag}.lower_continuity', r['lower_water_continuity_max_abs_mm'] < 1e-7),
            (f'{tag}.finite', r['finite']),
            (f'{tag}.nonnegative', r['nonnegative']),
            (f'{tag}.reservoir_fraction_range',
             0 <= r['reservoir_release_fraction_min'] and r['reservoir_release_fraction_max'] <= 1 + 1e-12),
            (f'{tag}.reservoir_duplicates', r['reservoir_duplicate_keys'] == 0),
            (f'{tag}.reservoir_nonnegative', not r['reservoir_negative']),
            (f'{tag}.reservoir_mass_balance', r['reservoir_mass_balance_max_abs_m3'] < 1e-6),
        ])
    return dict(passed=all(ok for _, ok in checks),
                failed=[n for n, ok in checks if not ok], arms=result)


def g0_3_coverage_declaration():
    """What the producer claims, versus what this round asks of the product.

    Quoted from `locks/product_lock.json` and its round report rather than
    paraphrased, because the whole point is to compare against the producer's
    own wording.  The claim is narrower than "the hydrology is validated", and
    this round steps outside it on three axes at once.
    """
    lock = json.loads((ROOT / '5_Test/20260917_3/locks/product_lock.json').read_text(encoding='utf-8'))
    claim, screen = lock['claim'], lock['screen']
    return dict(
        passed=True, blocking=False,
        conclusion_period=claim.get('zero_negative_applies_to')
        and 'the 74 retained stations, over 2019-01-01..2022-12-31' or None,
        declared_period=lock['carried_facts'][0],
        pet_bridge_flag=[f for f in lock['carried_facts'] if 'PET_BRIDGE_UNVALIDATED_2025' in f],
        not_a_replacement_for=lock['not_a_replacement_for'],
        selection=dict(combination=screen['combination'], origin=screen['origin'],
                       is_post_hoc=screen['is_post_hoc'], base=screen['base'],
                       retained=screen['retained'], excluded=screen['excluded'],
                       worst_retained_nse=screen['gate']['worst_retained_nse'],
                       why=screen['how_the_configuration_was_chosen']),
        same_population_protection=lock['same_population_protection'],
        # What this round asks of the product instead.
        this_round=dict(variable='TN concentration (the producer validated discharge)',
                        cohort='T24_G 116 / T24_L 34 TN stations (the producer validated 74 '
                               'post-hoc-selected discharge stations)',
                        period='evaluation year 2024 (the producer concluded 2006-2022, and its '
                               'zero-negative claim spans 2019-2022)'),
        statement=('The corrected hydrology is validated on discharge, on a post-hoc-selected set of 74 '
                   'stations, over 2019-2022 with a declared conclusion period of 2006-2022. This round '
                   'uses it to drive TN concentration in 2024 on a different and larger station panel. '
                   'It therefore leaves the producer\'s claim on three axes at once, and none of the '
                   'three is repaired here. This is the round\'s largest external-validity limitation '
                   'and must travel with every result. It is also not a fair burden: the product never '
                   'claimed to support this use, and a weak TN result here is not evidence against a '
                   'hydrology the producer never validated for TN.'),
        note='Recorded, never blocking: a FAIL would not change what is measurable.',
        supporting_quote='结论期是 2006-2022（观测档案所及）；1961-2005 无实测流量，「无站 NSE 为负」因此是 2006-2022 的陈述。')


def g0_4_topology():
    """Did the river network move with the water?  It must not have."""
    a = json.loads((RUN / 'data/domains/FULL24/topology.json').read_text(encoding='utf-8'))
    b = json.loads((RUN / 'data/domains/FULL24C/topology.json').read_text(encoding='utf-8'))
    detail = {k: a[k] == b[k] for k in a}
    return dict(passed=all(detail.values()), keys=detail,
                statement=('Topology, including all 13 reservoir entries, is identical between arms, '
                           'so the hydrology swap is not confounded by a network change.'))


def g0_5_capacity():
    """Each line must divide soil water by the capacity that produced it."""
    arms = {}
    for arm in ['H0', 'H1']:
        p = json.loads((RUN / 'data/domains' / ('FULL24' if arm == 'H0' else 'FULL24C')
                        / 'provenance.json').read_text(encoding='utf-8'))
        arms[arm] = dict(capacity=float(np.max(p['capacity']['value'])),
                         capacity_file=p['capacity']['file'],
                         numerator_max_mm=p['capacity']['numerator_max_mm'],
                         saturation_ratio=p['capacity']['saturation_ratio'],
                         wetness_max=p['soil_wetness']['maximum'])
    # Storage saturating at its own capacity is the evidence that the pairing is
    # right; a mismatch means one line is being divided by the other's scale.
    exact = all(abs(v['saturation_ratio'] - 1.0) < 1e-9 for v in arms.values())
    paired_ratio = arms['H1']['capacity'] / arms['H0']['capacity']
    return dict(passed=exact, arms=arms, saturation_exact=exact,
                paired_ratio=paired_ratio,
                statement=('soil_storage_mm + actual_aet_mm_day saturates at exactly the capacity of its '
                           'own line in both arms (ratio 1.0), so each arm is divided by the parameter '
                           'that produced it. Dividing the corrected storage by the frozen capacity '
                           f'would scale every soil_wetness by {paired_ratio:.4f}, and W enters five of '
                           'the eight dynamic basis columns. The concern is settled by measurement, '
                           'not by preference, and arm H1c is therefore not required.'))


def g0_6_cohort():
    """The 74-station screening panel is a discharge cohort, not a TN one."""
    excluded = json.loads((SCREEN / 'excluded.json').read_text(encoding='utf-8'))['excluded']
    registry = pd.read_parquet(RUN / 'data/station_registry.parquet')
    months = {}
    for scope in ['T24_L', 'T24_G']:
        c = pd.read_parquet(RUN / 'data/cohorts'/scope/'station_months.parquet')
        months[scope] = c.station_key.nunique()
    norm = lambda s: re.sub(r'\(.*?\)', '', str(s)).strip()
    tn = {norm(s) for s in registry.station.dropna()}
    ex = {norm(s) for s in excluded}
    return dict(passed=True, blocking=False,
                hydrology_excluded_n=len(excluded),
                hydrology_retained_n=74,
                tn_registry_n=int(len(registry)),
                exact_overlap=len({s for s in excluded} & set(registry.station.dropna())),
                loose_overlap=len(ex & tn),
                cohorts=months,
                statement=('The 74-station panel is the discharge/hydrology development panel (102 base '
                           'minus 28 masked, and post-hoc by its own declaration), not a TN cohort: its '
                           'names barely intersect the TN registry (1 exact, 2 loose, of 28 excluded). '
                           'It therefore cannot head this round\'s report. The headline cohort must be '
                           'one the round actually registers: T24_G as the full observed panel and '
                           'T24_L as the local subset. Both arms are evaluated on identical folds, so '
                           'this changes labelling only, not the comparison; and because the plan\'s '
                           'original headline named this panel, the substitution is recorded here at '
                           'Gate 0 rather than discovered in the results.'))


def g0_7_daily_stations():
    """Does the daily criterion have stations to run on?"""
    prep = json.loads((RUN / 'reports/preparation.json').read_text(encoding='utf-8'))
    cohorts = {c['scope']: c for c in prep['cohorts']}
    g = cohorts['T24_G']
    l = cohorts['T24_L']
    return dict(passed=g['hf_stations'] > 0, blocking=False,
                T24_G=dict(stations=g['stations'], hf_stations=g['hf_stations'],
                           hf_months=g['hf_months'], hf_days=g['hf_days']),
                T24_L=dict(stations=l['stations'], hf_stations=l['hf_stations'],
                           hf_months=l['hf_months'], hf_days=l['hf_days']),
                statement=('The daily criterion is carried by the high-frequency stations already '
                           'registered inside each fold, not by a separate list. T24_G holds '
                           f"{g['hf_stations']} such stations over {g['hf_days']} station-days, "
                           f"T24_L holds {l['hf_stations']}. The count is reported before fitting "
                           'rather than after.'))


def main():
    assert Path(sys.prefix).name.lower() == 'sparrow', 'conda sparrow required'
    started = time.time()
    report = dict(
        stage='20260917_5/gate0', started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        corrected_hydrology=str(CORRECTED), frozen_hydrology=str(FROZEN),
        preregistration_sha256=hashlib.sha256(PREREGISTRATION.encode('utf-8')).hexdigest(),
        preregistration=PREREGISTRATION)
    for name, fn in [('G0-1_columns', g0_1_columns), ('G0-2_consistency', g0_2_consistency),
                     ('G0-3_coverage_declaration', g0_3_coverage_declaration),
                     ('G0-4_topology', g0_4_topology), ('G0-5_capacity', g0_5_capacity),
                     ('G0-6_cohort', g0_6_cohort), ('G0-7_daily_stations', g0_7_daily_stations)]:
        report[name] = fn()
        print(f'{name}: {"PASS" if report[name]["passed"] else "FAIL"}', flush=True)
    blocking = [k for k, v in report.items()
                if isinstance(v, dict) and v.get('passed') is False and v.get('blocking', True)]
    report['blocking_failures'] = blocking
    report['status'] = 'PASS' if not blocking else 'FAIL'
    report['seconds'] = time.time() - started
    out = RUN / 'reports/gate0_preflight.json'
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    print('\n'.join([''] + [f'  {k}' for k in report if k.startswith('G0')] + ['', report['status']]))
    if blocking:
        raise SystemExit(f'GATE0 BLOCKING FAILURES: {blocking}')


if __name__ == '__main__':
    main()
