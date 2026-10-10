"""Known erroneous debt accounting controls plus one actual frozen ledger read."""
import copy
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'analysis')]
from diagnose_approved_plant_debt import accounting_rows, compare_rows, pooled, upgrade_guard
from d29_platform.runtime import sha, write_json


def fixture():
    rows = []
    for year in range(1961, 2025):
        planned = [[0., 0., 0.] for _ in range(13)]
        realized = copy.deepcopy(planned); unmet = copy.deepcopy(planned)
        planned[5] = [1000., 400., 600.]
        realized[5] = [900., 380., 570.]; unmet[5] = [100., 20., 30.]
        rows.append(dict(year=year, plant_activity=dict(
            planned_output_by_land_destination_kg=planned,
            realized_output_by_land_destination_kg=realized,
            unmet_output_by_land_destination_kg=unmet,
            # Deliberately irrelevant quantities must not form output debt.
            target_deficit_stock_days_by_land_kg_day=[1e30] * 13,
            potential_daily_demand_with_capacity_by_land_kg=[1e25] * 13)))
    return dict(passed=True, year_ledger=rows)


def main():
    controls = []; original = fixture(); rows = accounting_rows(original)
    pair = compare_rows(rows, rows)
    assert upgrade_guard([r for r in pair if r['year'] >= 2016])['guard_status'] == 'passed'
    crop = next(r for r in rows if r['year'] == 2016 and r['land_group'] == 'cropland' and
                r['destination'] == 'all_mutually_exclusive_outputs')
    assert crop['planned_kg'] == 2000. and crop['unmet_kg'] == 150.
    controls.append('mutually_exclusive_outputs_without_stock_days_or_repeated_demand')
    passive = next(r for r in rows if r['land_group'] == 'passive_surface_ledger')
    assert passive['unmet_fraction'] is None
    controls.append('zero_denominator_not_epsilon_or_zero_fraction')
    changed = copy.deepcopy(original)
    for annual in changed['year_ledger']:
        if annual['year'] == 2016:
            annual['plant_activity']['unmet_output_by_land_destination_kg'][5][0] += 30.
            annual['plant_activity']['realized_output_by_land_destination_kg'][5][0] -= 30.
    candidate = accounting_rows(changed)
    annual = compare_rows(rows, candidate)
    assert upgrade_guard([r for r in annual if r['year'] >= 2016])['upgrade_recommendation_blocked']
    assert not upgrade_guard(compare_rows(pooled(rows, 2016, 2024), pooled(candidate, 2016, 2024)))['upgrade_recommendation_blocked']
    controls.append('annual_warning_not_hidden_by_pooled_dilution')
    exact = copy.deepcopy(rows)
    for r in exact:
        if r['land_group'] == 'cropland' and r['destination'] == 'all_mutually_exclusive_outputs':
            r.update(unmet_kg=170., realized_kg=1830., unmet_fraction=.085)
    assert not upgrade_guard(compare_rows(rows, exact))['upgrade_recommendation_blocked']
    controls.append('exact_one_percentage_point_not_above_threshold')
    incompatible = copy.deepcopy(rows)
    incompatible[0]['planned_kg'] += 1.
    # An incompatible plan must not be accepted as a timing-only contrast.
    assert upgrade_guard(compare_rows(rows, incompatible))['upgrade_recommendation_blocked']
    controls.append('different_planned_output_blocks_upgrade_attribution')
    negative_cases = (
        ('missing_year', lambda p: p['year_ledger'].pop()),
        ('duplicate_year', lambda p: p['year_ledger'][1].update(year=1961)),
        ('unpassed_ledger', lambda p: p.update(passed=False)),
        ('missing_output_field', lambda p: p['year_ledger'][0]['plant_activity'].pop('unmet_output_by_land_destination_kg')),
        ('nonfinite_output', lambda p: p['year_ledger'][0]['plant_activity']['unmet_output_by_land_destination_kg'][5].__setitem__(0, float('nan'))),
        ('negative_output', lambda p: p['year_ledger'][0]['plant_activity']['unmet_output_by_land_destination_kg'][5].__setitem__(0, -1.)),
        ('double_counted_return', lambda p: p['year_ledger'][0]['plant_activity']['realized_output_by_land_destination_kg'][5].__setitem__(1, 780.)),
        ('crop_replicated_land_shape', lambda p: p['year_ledger'][0]['plant_activity']['planned_output_by_land_destination_kg'].append([0., 0., 0.])),
    )
    for name, alter in negative_cases:
        p = copy.deepcopy(original); alter(p)
        try:
            accounting_rows(p)
        except (RuntimeError, KeyError):
            controls.append(name)
        else:
            raise AssertionError(name)
    for name, action in (
        ('empty_support', lambda: compare_rows([], [])),
        ('duplicate_support', lambda: compare_rows(rows + rows[:1], rows)),
        ('missing_common_support', lambda: compare_rows(rows[:-1], rows)),
        ('empty_primary_guard', lambda: upgrade_guard([])),
    ):
        try:
            action()
        except RuntimeError:
            controls.append(name)
        else:
            raise AssertionError(name)
    # Strictly validate this actual one-path freeze; no fabricated all44 proof.
    from assemble_approved_delivery import validate_frozen
    jid = 'TIM_F24_C0_V0_s0'; folder = ROOT / 'outputs/jobs' / jid
    job = next(j for j in json.loads((ROOT / 'config/jobs.json').read_text()) if j['id'] == jid)
    support = json.loads((ROOT / 'evidence/expected_station_support.json').read_text())
    files = validate_frozen(folder, job, sha(ROOT / 'evidence/formal_gate.json'), support)
    physical = json.loads((folder / 'physical_ledger.json').read_text())
    before = sha(folder / 'physical_ledger.json'); real = accounting_rows(physical, folder)
    assert len(real) == 64 * 3 * 3
    assert not upgrade_guard(compare_rows(real, real))['upgrade_recommendation_blocked']
    assert sha(folder / 'physical_ledger.json') == before
    controls.append('actual64year_frozen_ledger_read_only_self_comparison')
    write_json(ROOT / 'evidence/approved_plant_debt_controls.json', dict(passed=True,
        controls=controls, source_sha256=sha(ROOT / 'analysis/diagnose_approved_plant_debt.py'),
        test_sha256=sha(Path(__file__)), actual_benchmark_job=jid,
        actual_bound_files=len(files), actual_rows=len(real),
        max_original_summary_roundoff_kg=max(r['max_original_summary_roundoff_kg'] for r in real),
        annual_reach_arrays_independently_read=True,
        physical_ledger_sha256=before, physical_oracle_calls=0, evaluation_labels_read=False,
        all44_completed=False, final_debt_guard_not_yet_computed=True))
    print(json.dumps(dict(passed=True, controls=len(controls), actual_frozen_paths=1, actual_rows=len(real))))


if __name__ == '__main__':
    main()
