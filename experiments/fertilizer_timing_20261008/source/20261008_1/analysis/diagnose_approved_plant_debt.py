"""Post-freeze output-debt guard for all 44 approved paths; no TN label reads.

Uses mutually exclusive planned outputs, never repeated uptake opportunities or
stock-day deficits, as the denominator. Each candidate compares with the C0 of
its own plant version and training support. Annual warnings are retained even
when pooled-year summaries dilute them. No training or dispatch is performed.
"""
import csv
import calendar
from decimal import Decimal
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'analysis')]
from d29_platform.runtime import sha, write_json

GROUPS = {'cropland': tuple(range(5, 10)),
          'vegetated_noncrop': (0, 1, 2, 3, 10, 11),
          'passive_surface_ledger': (4, 12)}
DESTINATIONS = ('all_mutually_exclusive_outputs', 'external_removal', 'internal_return')
FIELDS = ('planned_output_by_land_destination_kg',
          'realized_output_by_land_destination_kg',
          'unmet_output_by_land_destination_kg')


def accounting_rows(physical, folder=None):
    """Validate the real annual output accounting before computing any ratio."""
    years = physical.get('year_ledger', [])
    if physical.get('passed') is not True or len(years) != 64:
        raise RuntimeError('PASSED_COMPLETE64YEAR_LEDGER_REQUIRED')
    if sorted(row['year'] for row in years) != list(range(1961, 2025)):
        raise RuntimeError('UNIQUE_EXACT_HISTORY_YEARS_REQUIRED')
    rows = []
    for annual in sorted(years, key=lambda row: row['year']):
        activity = annual['plant_activity']
        arrays = [activity[key] for key in FIELDS]
        if any(len(a) != 13 or any(len(cell) != 3 for cell in a) for a in arrays):
            raise RuntimeError('PHYSICAL13LAND_THREE_DESTINATIONS_REQUIRED')
        if any(not math.isfinite(x) or x < 0 for a in arrays for cell in a for x in cell):
            raise RuntimeError('FINITE_NONNEGATIVE_OUTPUT_ACCOUNTING_REQUIRED')
        summary_difference = 0.
        if folder is not None:
            # These annual domain totals were summed independently in the
            # original diagnostic. Their roundoff is not a daily local balance.
            # Read the bound reach/land artifacts and sum accurately instead.
            import numpy as np
            folder = Path(folder).resolve()
            path = (folder / activity['reach_land_file']).resolve()
            if not path.is_relative_to(folder) or sha(path) != activity['reach_land_sha256']:
                raise RuntimeError('BOUND_ANNUAL_REACH_LAND_ARRAYS_REQUIRED')
            with np.load(path, allow_pickle=False) as source:
                reach_arrays = [source[k].copy() for k in
                                ('planned_output_kg', 'realized_output_kg', 'unmet_output_kg')]
            if any(a.shape != (230, 13, 3) or not np.isfinite(a).all() or (a < 0).any()
                   for a in reach_arrays):
                raise RuntimeError('FINITE230REACH_PHYSICAL_LAND_ACCOUNTING_REQUIRED')
            if np.max(abs(reach_arrays[0] - reach_arrays[1] - reach_arrays[2])) > 1e-6:
                raise RuntimeError('ANNUAL_REACH_OUTPUT_CLOSURE_REQUIRED')
            accurate = [[[math.fsum(a[:, i, j]) for j in range(3)] for i in range(13)]
                        for a in reach_arrays]
            # Conservative forward-error bound for the original nonnegative
            # day/reach sums; not a replacement for the frozen physical gate.
            terms = (366 if calendar.isleap(annual['year']) else 365) * 230
            unit_roundoff = 2. ** -53
            gamma = terms * unit_roundoff / (1. - terms * unit_roundoff)
            for old, new in zip(arrays, accurate):
                for i in range(13):
                    for j in range(3):
                        difference = abs(old[i][j] - new[i][j])
                        if difference > gamma * (abs(old[i][j]) + abs(new[i][j])):
                            raise RuntimeError('DECLARED_SUMMARY_OUTSIDE_SUMMATION_ROUNDOFF_BOUND')
                        summary_difference = max(summary_difference, difference)
            arrays = accurate
        for land in range(13):
            for destination in range(3):
                p, r, u = [a[land][destination] for a in arrays]
                if abs(p - r - u) > 1e-6:
                    raise RuntimeError('PLANNED_REALIZED_UNMET_CLOSURE_REQUIRED')
        for group, indices in GROUPS.items():
            for destination, columns in zip(DESTINATIONS, ((0, 1, 2), (0,), (1, 2))):
                totals = [math.fsum(a[i][j] for i in indices for j in columns) for a in arrays]
                p, r, u = totals
                rows.append(dict(year=annual['year'], land_group=group, destination=destination,
                                 planned_kg=p, realized_kg=r, unmet_kg=u,
                                 unmet_fraction=u / p if p > 0 else None,
                                 ratio_status='defined' if p > 0 else 'zero_planned_output_not_applicable',
                                 annual_reach_arrays_independently_read=folder is not None,
                                 max_original_summary_roundoff_kg=summary_difference,
                                 NSE_status='not_applicable_plant_output_accounting'))
    return rows


def compare_rows(baseline, candidate):
    def indexed(rows):
        out = {(r['year'], r['land_group'], r['destination']): r for r in rows}
        if len(out) != len(rows) or not out:
            raise RuntimeError('NONEMPTY_UNIQUE_DEBT_SUPPORT_REQUIRED')
        return out
    left, right = indexed(baseline), indexed(candidate)
    if set(left) != set(right):
        raise RuntimeError('EXACT_COMMON_PLANT_ACCOUNTING_SUPPORT_REQUIRED')
    comparisons = []
    for key in sorted(left):
        a, b = left[key], right[key]
        same = abs(a['planned_kg'] - b['planned_kg']) <= 1e-6
        defined = a['planned_kg'] > 0 and b['planned_kg'] > 0
        delta = None
        if defined:
            # Decimal comparisons avoid declaring exactly one percentage point
            # above the gate due solely to binary subtraction of fractions.
            delta_decimal = 100 * (Decimal(str(b['unmet_kg'])) / Decimal(str(b['planned_kg']))
                                   - Decimal(str(a['unmet_kg'])) / Decimal(str(a['planned_kg'])))
            delta = float(delta_decimal)
            warning = bool(delta_decimal > Decimal(1)) if same else None
        else:
            warning = None
        status = ('different_planned_output_no_attribution' if not same else
                  'defined' if defined else 'zero_planned_output_not_applicable')
        comparisons.append(dict(year=key[0], land_group=key[1], destination=key[2],
            planned_baseline_kg=a['planned_kg'], planned_candidate_kg=b['planned_kg'],
            realized_baseline_kg=a['realized_kg'], realized_candidate_kg=b['realized_kg'],
            unmet_baseline_kg=a['unmet_kg'], unmet_candidate_kg=b['unmet_kg'],
            unmet_fraction_baseline=a['unmet_fraction'], unmet_fraction_candidate=b['unmet_fraction'],
            same_planned_output=same, comparison_status=status,
            unmet_difference_percentage_points=delta, exceeds_one_percentage_point=warning,
            NSE_status='not_applicable_plant_output_accounting'))
    return comparisons


def pooled(rows, first, last):
    out = []
    for group in GROUPS:
        for destination in DESTINATIONS:
            subset = [r for r in rows if first <= r['year'] <= last and
                      r['land_group'] == group and r['destination'] == destination]
            if len(subset) != last - first + 1:
                raise RuntimeError('EXACT_PERIOD_SUPPORT_REQUIRED')
            p, r, u = [math.fsum(row[k] for row in subset)
                       for k in ('planned_kg', 'realized_kg', 'unmet_kg')]
            out.append(dict(year=f'{first}-{last}', land_group=group, destination=destination,
                            planned_kg=p, realized_kg=r, unmet_kg=u,
                            unmet_fraction=u / p if p > 0 else None))
    return out


def upgrade_guard(rows):
    # External-removal and internal-return rows diagnose which destination
    # contributes. The prescribed guard uses their mutually exclusive total.
    primary = [r for r in rows if r['destination'] == DESTINATIONS[0] and
               r['land_group'] in ('cropland', 'vegetated_noncrop')]
    if not primary:
        raise RuntimeError('NONEMPTY_PRIMARY_PLANT_GUARD_REQUIRED')
    comparable = [r for r in primary if r['comparison_status'] == 'defined']
    uncomparable = [r for r in primary if r['comparison_status'] == 'different_planned_output_no_attribution']
    warnings = [r for r in comparable if r['exceeds_one_percentage_point']]
    return dict(upgrade_recommendation_blocked=bool(warnings or uncomparable),
                guard_status='warning' if warnings else 'incomparable_plan' if uncomparable else
                             'passed' if comparable else 'not_applicable_no_planned_output',
                exceeds_one_percentage_point_rows=len(warnings),
                incomparable_plan_rows=len(uncomparable), comparable_rows=len(comparable),
                guard_pass_is_not_evidence_of_TN_improvement=True)


def write_table(path, rows):
    if not rows:
        raise RuntimeError('EMPTY_PLANT_DELIVERY_FORBIDDEN')
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    from approved_epoch_delivery import admitted_folders
    from assemble_approved_delivery import validate_frozen
    from evaluate_approved_campaign import select, pair_definitions
    folders = admitted_folders(ROOT)  # Fail closed before any result processing.
    jobs = json.loads((ROOT / 'config/jobs.json').read_text(encoding='utf8'))
    support = json.loads((ROOT / 'evidence/expected_station_support.json').read_text(encoding='utf8'))
    byid = {j['id']: j for j in jobs}; records = {}; accounting = {}; dependencies = {}
    for job in jobs:
        jid = job['id']; folder = folders[jid]
        audit = json.loads((folder / 'independent_audit.json').read_text(encoding='utf8'))
        files = validate_frozen(folder, job, audit['formal_gate_sha256'], support)
        physical = json.loads((folder / 'physical_ledger.json').read_text(encoding='utf8'))
        accounting[jid] = accounting_rows(physical, folder)
        records[jid] = dict(objective=audit['objective'], objective_passed=True, physical_passed=True,
                            numerically_sufficient=True, freeze_verified=True)
        dependencies[jid] = {k: files[k] for k in ('independent_audit.json', 'prediction_freeze.json', 'physical_ledger.json')}
    selected = select(jobs, records)  # Training objective only; no TN outcomes.
    table = []; guards = []; annual = []
    for jid in sorted(accounting):
        annual.extend(dict(job=jid, **row) for row in accounting[jid])
    for baseid, candidateid, contrast in pair_definitions(selected):
        if contrast != 'complete_timing_configuration_vs_C0':
            continue
        base, candidate = accounting[baseid], accounting[candidateid]
        rows = compare_rows(base, candidate)
        intervention = [r for r in rows if 2016 <= r['year'] <= 2024]
        history = [r for r in rows if r['year'] <= 2015]
        summaries = compare_rows(pooled(base, 2016, 2024), pooled(candidate, 2016, 2024))
        guard = upgrade_guard(intervention)
        # Calibrated process parameters differ across full histories; these
        # historical diagnostics do not assert identical pre-2016 states.
        identity = dict(baseline=baseid, candidate=candidateid, plant_version=byid[candidateid]['plant_version'],
                        fold=byid[candidateid]['fold'], space_block=byid[candidateid]['space_block'])
        for r in rows + summaries:
            table.append(dict(**identity, **r))
        guards.append(dict(**identity, **guard, annual_intervention_rows=len(intervention),
                           descriptive_historical_rows=len(history), pooled_guard=upgrade_guard(summaries),
                           spatial_scope='complete230reach_physical_land_domain_not_held_station_support'))
    output = ROOT / 'outputs/approved_plant_debt'
    if (output / 'complete.json').exists():
        raise RuntimeError('EXISTING_DELIVERY_INSPECT_BEFORE_RERUN')
    output.mkdir(exist_ok=True)
    write_table(output / 'all44_annual_plant_outputs.csv', annual)
    write_table(output / 'paired_annual_and_pooled_output_debt.csv', table)
    write_json(output / 'upgrade_guards.json', guards)
    files = {p.name: sha(p) for p in output.iterdir() if p.is_file()}
    write_json(output / 'complete.json', dict(passed=True, accounted_paths=len(jobs),
        comparisons=len(guards), selected_by_training_objective=True, evaluation_labels_read=False,
        source_sha256=sha(Path(__file__)), approved_delivery_sha256=sha(ROOT / 'evidence/approved_epoch_delivery.json'),
        dependencies=dependencies, files=files, threshold_percentage_points=1,
        guard_aggregation='each intervention year by cropland/noncrop mutually exclusive total; pooled period also reported',
        no_epsilon=True, physical_oracle_calls=0, no_TN_benefit_claim=True))
    print(json.dumps(dict(passed=True, accounted_paths=len(jobs), comparisons=len(guards),
                         blocked_recommendations=sum(g['upgrade_recommendation_blocked'] for g in guards))))


if __name__ == '__main__':
    main()
