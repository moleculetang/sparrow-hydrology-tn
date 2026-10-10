"""Independent read-back of frozen annual -> month -> day agricultural artifacts.

No predictions, observed TN, optimizer state or calendar generator are read.
Stored daily profiles are checked against the original frozen input budgets.
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from d29_platform.runtime import configure, write_json, sha
configure()
import numpy as np
import pandas as pd


def check_year(budgets, profiles, origin, year):
    keys = ['reach_id', 'crop', 'source']
    if budgets.duplicated(keys).any() or profiles.profile_id.duplicated().any():
        raise RuntimeError('DUPLICATE_ALLOCATION_IDENTITY')
    reference = origin.rename(columns={'field': 'source', 'mass_kg': 'original_budget_kg'})
    joined = budgets.merge(reference, on=keys, how='outer', suffixes=('', '_original'),
                           validate='one_to_one', indicator=True)
    if not joined['_merge'].eq('both').all():
        raise RuntimeError('EXACT_ALL_CROP_SOURCE_PRODUCT_SUPPORT_REQUIRED')
    if not np.array_equal(joined.budget_kg.to_numpy(), joined.original_budget_kg.to_numpy()):
        raise RuntimeError('ANNUAL_INPUT_MASS_CHANGED')
    for column in ['original_area_year', 'rate_year', 'rate_donor_identity']:
        if not np.array_equal(joined[column].to_numpy(), joined[column+'_original'].to_numpy()):
            raise RuntimeError('ORIGINAL_INPUT_YEAR_OR_DONOR_CHANGED ' + column)
    dates = pd.date_range(f'{year}-01-01', f'{year}-12-31')
    months = dates.month.to_numpy()
    weights = {}
    for p in profiles.itertuples():
        w = np.asarray(p.daily_weights, dtype=np.float64)
        if w.shape != (len(dates),) or not np.isfinite(w).all() or (w < 0).any():
            raise RuntimeError('FINITE_NONNEGATIVE_CALENDAR_DAY_SUPPORT_REQUIRED')
        if abs(float(w.sum()) - 1.) > 1e-12:
            raise RuntimeError('DAILY_PROFILE_NOT_NORMALIZED')
        if hashlib.sha256(w.astype('<f8', copy=False).tobytes()).hexdigest() != p.profile_id:
            raise RuntimeError('DAILY_PROFILE_ID_CHANGED')
        by_month = np.array([w[months == m].sum() for m in range(1, 13)])
        if not np.array_equal(by_month, np.asarray(p.monthly_calendar_weights, dtype=float)):
            raise RuntimeError('MONTHLY_PROFILE_NOT_SAME_DAY_INTEGRAL')
        weights[p.profile_id] = w
    maxima = dict(annual=0., month_sum=0.)
    used = set()
    for row in budgets.itertuples():
        if row.year != year or row.annual_identity != 'frozen_input_product_no_source_multiplier':
            raise RuntimeError('ALLOCATION_YEAR_OR_FIXED_MASS_IDENTITY_CHANGED')
        B = float(row.budget_kg)
        if not np.isfinite(B) or B < 0 or row.profile_id not in weights:
            raise RuntimeError('FINITE_BUDGET_AND_REGISTERED_PROFILE_REQUIRED')
        if (not isinstance(row.residual_day_index, (int, np.integer)) or
                isinstance(row.residual_day_index, (bool, np.bool_)) or
                not np.isfinite(row.residual_kg) or not 0 <= row.residual_day_index < len(dates)):
            raise RuntimeError('EXACT_RECONSTRUCTION_RESIDUAL_REQUIRED')
        used.add(row.profile_id)
        daily = B * weights[row.profile_id]
        daily[int(row.residual_day_index)] += float(row.residual_kg)
        if not np.isfinite(daily).all() or (daily < 0).any():
            raise RuntimeError('INVALID_RECONSTRUCTED_DAILY_MASS')
        if hashlib.sha256(daily.astype('<f8', copy=False).tobytes()).hexdigest() != row.daily_mass_sha256:
            raise RuntimeError('EXACT_DAILY_MASS_DIGEST_CHANGED')
        monthly = np.array([daily[months == m].sum() for m in range(1, 13)])
        if not np.array_equal(monthly, np.asarray(row.monthly_kg, dtype=float)):
            raise RuntimeError('MONTH_MASS_NOT_EXACT_DAY_RECONSTRUCTION')
        if list(row.month_mass_status) != ['positive_mass' if x > 0 else 'not_applicable_zero_mass' for x in monthly]:
            raise RuntimeError('ZERO_MONTH_STATUS_CHANGED')
        if B == 0:
            if row.pi is not None or np.any(daily != 0):
                raise RuntimeError('ZERO_ANNUAL_MASS_MUST_HAVE_NO_PI')
        elif not np.array_equal(np.asarray(row.pi, dtype=float), monthly/B):
            raise RuntimeError('MONTH_PROPORTIONS_NOT_FIXED_ANNUAL_BUDGET')
        maxima['annual'] = max(maxima['annual'], abs(float(daily.sum())-B))
        maxima['month_sum'] = max(maxima['month_sum'], abs(float(monthly.sum())-B))
    if used != set(weights):
        raise RuntimeError('UNUSED_OR_MISSING_PROFILE_ACCOUNTING')
    if max(maxima.values()) > 1e-6:
        raise RuntimeError('ORIGINAL_ABSOLUTE_MASS_TOLERANCE_FAILED')
    return dict(year=year, rows=len(budgets), profiles=len(profiles),
                days=len(dates), maxima_kg=maxima, annual_input_kg=float(budgets.budget_kg.sum()))


def main(jid):
    from train_job import validate_admission
    from job_lease import require
    _, gate_hash = validate_admission()
    require(jid, 'allocation_review')
    folder = ROOT / 'outputs/jobs' / jid
    directory = folder / 'input_allocation'
    receipt_path = directory / 'complete.json'
    sha_before = sha(receipt_path)
    receipt = json.loads(receipt_path.read_text())
    freeze_path = folder / 'prediction_freeze.json'
    freeze_before = sha(freeze_path)
    freeze = json.loads(freeze_path.read_text())
    point_hash = sha(folder / 'frozen_parameters.npy')
    if (receipt.get('passed') is not True or receipt.get('source_multiplier') != 1. or
            receipt.get('job') != jid or receipt.get('formal_gate_sha256') != gate_hash or
            receipt.get('parameter_sha256') != point_hash or
            freeze.get('parameter_sha256') != point_hash or
            freeze.get('numerically_sufficient') is not True):
        raise RuntimeError('ACTUAL_SUFFICIENT_FROZEN_ALLOCATION_REQUIRED')
    for relative, digest in receipt['identities'].items():
        if sha(ROOT / relative) != digest:
            raise RuntimeError('ALLOCATION_GENERATOR_IDENTITY_CHANGED')
    expected = {f'{kind}_{y}.parquet' for kind in ['budgets', 'profiles'] for y in range(1961, 2025)}
    if set(receipt['artifact_sha256']) != expected:
        raise RuntimeError('EXACT_128_FULL_HISTORY_ARTIFACTS_REQUIRED')
    for name, digest in receipt['artifact_sha256'].items():
        if sha(directory / name) != digest:
            raise RuntimeError('EXPORTED_ALLOCATION_BYTES_CHANGED ' + name)
    source = ROOT / 'inputs/reference/agriculture_reference/annual_crop_source_reference.parquet'
    origin = pd.read_parquet(source)
    records = []
    for year in range(1961, 2025):
        reference = origin[origin.year == year].drop(columns='year')
        if (len(reference) != 230*21*2 or reference.crop.nunique() != 21 or
                set(reference.reach_id) != set(range(1, 231)) or
                set(reference.field) != {'Nfer', 'Nmanure'} or
                reference.duplicated(['reach_id', 'crop', 'field']).any()):
            raise RuntimeError('FULL_21_CROP_230_REACH_INPUT_REQUIRED')
        records.append(check_year(pd.read_parquet(directory/f'budgets_{year}.parquet'),
                                  pd.read_parquet(directory/f'profiles_{year}.parquet'), reference, year))
        write_json(directory / 'independent_review_progress.json', dict(records=records, final=False))
    if sha(receipt_path) != sha_before or sha(freeze_path) != freeze_before:
        raise RuntimeError('ALLOCATION_OR_FREEZE_CHANGED_DURING_REVIEW')
    result = dict(passed=True, job=jid, years=64, reaches=230, crops=21,
                  scope='independent_serialized_artifact_readback_against_frozen_annual_product',
                  records=records, no_TN_or_prediction_values_read=True,
                  annual_source_sha256=sha(source), allocation_receipt_sha256=sha(receipt_path),
                  parameter_sha256=point_hash, formal_gate_sha256=gate_hash,
                  verifier_sha256=sha(Path(__file__)))
    write_json(directory / 'independent_artifact_review.json', result)
    print(json.dumps(dict(passed=True, years=64, rows=sum(r['rows'] for r in records))))


if __name__ == '__main__':
    main(sys.argv[1])
