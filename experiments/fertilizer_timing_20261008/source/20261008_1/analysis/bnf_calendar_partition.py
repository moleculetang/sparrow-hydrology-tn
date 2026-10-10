"""Partition unchanged BNF by configured crop growth support, for diagnostics.

This module never changes a physical source, target or state. Calendar absence
is an unknown classification, not evidence of continuous growth or fallow.
The V0 classification uses the V1 calendar as a descriptive reference only.
"""
import numpy as np

LABELS = ('other_sources_and_initial', 'BNF_before_2016',
          'BNF_within_configured_growth', 'BNF_outside_configured_growth',
          'BNF_calendar_unknown')


def reconcile(parts, total):
    """Restore the given float64 mass by its largest existing recipient.

    No epsilon, clipping, new source or tolerance enlargement. Only a bounded
    floating-point partition residual may be assigned; all outputs stay >=0.
    """
    parts = np.asarray(parts, dtype=np.float64).copy()
    total = np.asarray(total, dtype=np.float64)
    if (parts.shape[:-1] != total.shape or not np.isfinite(parts).all()
            or not np.isfinite(total).all() or (parts < 0).any() or (total < 0).any()):
        raise ValueError('FINITE_NONNEGATIVE_MASS_PARTITION_REQUIRED')
    residual = total-parts.sum(axis=-1)
    bound = 16*np.spacing(total)
    if np.any(abs(residual) > bound):
        raise ValueError('PARTITION_IS_NOT_A_FLOAT64_RESIDUAL')
    index = np.argmax(parts, axis=-1)[..., None]
    value = np.take_along_axis(parts, index, axis=-1)[..., 0]+residual
    if (value < 0).any():
        raise ValueError('PARTITION_RESIDUAL_WOULD_CREATE_NEGATIVE_MASS')
    np.put_along_axis(parts, index, value[..., None], axis=-1)
    return parts, float(np.max(abs(residual), initial=0.))


def partition(year, tagged_sources, labels, activity, profiles, growth_profile,
              returned_fraction, exported_fraction):
    """Return five diagnostic labels and per-crop temporal accounting.

    activity is a DataFrame for one year with unique (reach_id,crop) rows.
    growth_profile(year, seasons) returns the exact configured growth weights.
    The two BNF entrances (plant and available N) are classified separately.
    """
    tags = np.asarray(tagged_sources, dtype=np.float64)
    labels = list(labels)
    if (tags.ndim != 5 or tags.shape[-1] != 4 or tags.shape[3] != len(labels)
            or labels.count('crop_BNF') != 1 or not np.isfinite(tags).all()
            or (tags < 0).any()):
        raise ValueError('EXPLICIT_NONNEGATIVE_BNF_SOURCE_TAGS_REQUIRED')
    nt, nr, nl, _, _ = tags.shape
    if min(nt,nr,nl) == 0 or not 1961 <= year <= 2024:
        raise ValueError('REGISTERED_NONEMPTY_YEAR_REQUIRED')
    bnf = tags[..., labels.index('crop_BNF'), :]
    if np.any(bnf[..., 1:3] != 0):
        raise ValueError('BNF_ENTRY_SCHEMA_CHANGED')
    total = tags.sum(axis=3)
    result = np.zeros((nt, nr, nl, len(LABELS), 4), dtype=np.float64)
    result[..., 0, :] = total-bnf
    if year < 2016:
        result[..., 1, :] = bnf
        return result, [], dict(labels=list(LABELS), preintervention=True,
            source_sum_max_error_kg=float(np.max(abs(result.sum(axis=3)-total), initial=0.)),
            physical_source_changed=False, temporal_calendar_inferred_before2016=False)
    required = {'year','reach_id','crop','bnf_plant_kg','bnf_soil_kg',
                'harvest_n_kg','residue_n_kg'}
    if (not required <= set(activity.columns) or activity.empty or activity.duplicated(['reach_id','crop']).any()
            or not activity.year.eq(year).all()):
        raise ValueError('UNIQUE_SAME_YEAR_CROP_BNF_ACTIVITY_REQUIRED')
    if set(activity.reach_id) != set(range(1,nr+1)):
        raise ValueError('ALL_REACH_ACTIVITY_RECORDS_REQUIRED_NO_DEFAULT_ZERO')
    mass_fields = ['bnf_plant_kg','bnf_soil_kg','harvest_n_kg','residue_n_kg']
    masses = activity[mass_fields].to_numpy(dtype=np.float64)
    if not np.isfinite(masses).all() or (masses < 0).any():
        raise ValueError('UNKNOWN_OR_NEGATIVE_BNF_ACTIVITY')
    if (not np.isfinite([returned_fraction, exported_fraction]).all()
            or min(returned_fraction, exported_fraction) < 0
            or abs(returned_fraction+exported_fraction-1.) > 1e-12):
        raise ValueError('EXCLUSIVE_CLOSED_RESIDUE_FATES_REQUIRED')
    expected = activity.groupby('reach_id')[['bnf_plant_kg','bnf_soil_kg']].sum().reindex(
        range(1,nr+1)).to_numpy(dtype=np.float64)
    actual = bnf[..., [0,3]].sum(axis=(0,2))
    budget_error = float(np.max(abs(actual-expected),initial=0.))
    if budget_error > 1e-6:
        raise ValueError('INDEPENDENT_ANNUAL_BNF_BUDGET_MISMATCH')
    weight = np.zeros((nt, nr, 3, 2), dtype=np.float64)
    crop_rows = []
    for row in activity.itertuples():
        if (int(row.reach_id) != row.reach_id or not 1 <= row.reach_id <= nr):
            raise ValueError('BNF_REACH_OUTSIDE_PHYSICAL_SUPPORT')
        r = int(row.reach_id)-1
        budget = np.array([row.bnf_plant_kg,row.bnf_soil_kg], dtype=np.float64)
        plan = row.harvest_n_kg+row.residue_n_kg*(returned_fraction+exported_fraction)
        reference = profiles.get(row.crop, {}).get(str(r+1))
        known = reference is not None
        if known:
            growth = np.asarray(growth_profile(year, reference['seasons']), dtype=np.float64)
            if (growth.shape != (nt,) or not np.isfinite(growth).all()
                    or (growth < 0).any() or abs(growth.sum()-1.) > 1e-12):
                raise ValueError('EXACT_NONNEGATIVE_GROWTH_PROFILE_REQUIRED')
            mask = (growth > 0) & (plan > 0)
            weight[:,r,0] += mask[:,None]*budget[None,:]
            weight[:,r,1] += (~mask)[:,None]*budget[None,:]
            nongrowing_days = int((~mask).sum())
        else:
            weight[:,r,2] += budget[None,:]
            nongrowing_days = None
        fraction = nongrowing_days/nt if known else None
        crop_rows.append(dict(year=year,reach_id=r+1,crop=row.crop,
            bnf_plant_budget_kg=float(budget[0]),bnf_soil_budget_kg=float(budget[1]),
            configured_calendar_available=known,nongrowing_days=nongrowing_days,
            configured_growth_plan_kg=float(plan),zero_growth_plan=bool(plan==0),
            plant_BNF_outside_growth_kg=None if fraction is None else float(budget[0]*fraction),
            soil_BNF_outside_growth_kg=None if fraction is None else float(budget[1]*fraction),
            BNF_input_timing='unchanged_uniform_research_scenario',
            calendar_classification='configured_growth_support' if known else 'unknown',
            observations_used=False,NSE='not_applicable'))
    denominator = weight.sum(axis=2)
    fractions = np.divide(weight,denominator[:,:,None,:],
        out=np.zeros_like(weight),where=denominator[:,:,None,:]!=0)
    correction = 0.
    for channel, entry in enumerate((0,3)):
        target = bnf[...,entry]
        if np.any((denominator[:,:,channel]==0) & (target.sum(axis=2)!=0)):
            raise ValueError('BNF_MASS_WITHOUT_INDEPENDENT_CROP_BUDGET')
        proposed = target[...,None]*fractions[:,:,None,:,channel]
        proposed, error = reconcile(proposed, target)
        correction = max(correction,error)
        result[...,2:5,entry] = proposed
    # The total physical source is the authority; compensate only the final
    # partition summation residual, never invent a missing crop budget.
    ordered = result.transpose(0,1,2,4,3)
    ordered, error = reconcile(ordered,total)
    result = ordered.transpose(0,1,2,4,3)
    correction = max(correction,error)
    summed = result.sum(axis=3)
    max_error = float(np.max(abs(summed-total), initial=0.))
    if max_error > 1e-6 or not np.isfinite(result).all() or (result < 0).any():
        raise ValueError('UNCHANGED_PHYSICAL_SOURCE_PARTITION_REQUIRED')
    return result,crop_rows,dict(labels=list(LABELS),preintervention=False,
        source_sum_max_error_kg=max_error,max_float64_partition_residual_kg=correction,
        independent_BNF_budget_max_error_kg=budget_error,
        physical_source_changed=False,crop_rows=len(crop_rows),
        unknown_calendars_are_not_zero_nongrowth=True,
        V0_window_is_reference_not_a_capacity_constraint=True,
        soil_BNF_outside_growth_is_not_by_itself_a_biological_violation=True)
