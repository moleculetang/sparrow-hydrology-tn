"""Independent NumPy grey-box decoding, full-history stores and mass audit.

This module does not call GreyBox, its process function, or its adjoint. Saved
training transforms suffice for future inference; no TN labels are accepted.
M is mineral nitrogen with the existing lifetime/legacy memory; L is slow-water
nitrogen. The old independent_process key 'legacy' aliases L, NOT an extra store.
"""
import argparse
import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from fc_io import RUN, TEST, read, write, local, sha, guard, require_environment, now
from fc_data import prediction_metadata
from fc_legacy import route
from fc_physical_audit import observe


def independent_decode(data, artifact):
    design = artifact['design']; grey = design['grey']
    parameters = dict(zip(artifact['parameter_names'], np.asarray(artifact['theta'], float)))
    raw = pd.read_parquet(TEST/'20260905_1/outputs/h7_raw_features.parquet').sort_values('reach_id')
    x = (np.clip(raw[design['fields']].to_numpy(float), design['low'], design['high'])-design['mean'])/design['sd']
    def latent(name, prefix, lo, hi):
        fraction = (parameters[name]-lo)/(hi-lo)
        if not 0 < fraction < 1:
            raise ValueError('NONFINITE_REGIONAL_LOGIT '+name)
        return np.log(fraction/(1-fraction)) + x @ np.array([parameters[f'{prefix}_{i}'] for i in range(7)])
    la = latent('log_alpha_contact', 'gamma_contact', -9.21, 4.605170186)
    if grey['variant'] in ('MG', 'MGF'):
        old = np.column_stack([np.ones(len(x)), x])
        for i, item in enumerate(grey['gam']):
            a, b, c = item['knots']; xx = x[:, item['field_index']]
            # Independently written truncated-power spline and saved projection.
            val = np.maximum(xx-a, 0.)**3
            val -= ((c-a)*np.maximum(xx-b, 0.)**3 - (b-a)*np.maximum(xx-c, 0.)**3)/(c-b)
            val /= (c-a)**2
            val = (val-old@np.asarray(item['projection']))/item['scale']
            la += parameters[f'gamma_spline_{i}']*val
    alpha = -9.21+(4.605170186+9.21)*expit(la)
    lt = latent('log_tau_mineral_days', 'gamma_lifetime', np.log(182.625), np.log(3652.5))
    logtau = np.log(182.625)+(np.log(3652.5)-np.log(182.625))*expit(lt)
    positive = data.contact > 0
    loghazard = alpha[None, :] + parameters['beta_contact']*np.log(np.where(positive, data.contact, 1.))
    for field, arr, parameter in [('upper_water_mm', data.upper_water, 'eta_upper'),
                                   ('percolation_mm_day', data.percolation, 'eta_percolation')]:
        sc = next(item for item in design['dynamic_scales'] if item['field'] == field)
        loghazard += parameters[parameter]*(np.log1p(arr)-sc['mean'])/sc['sd']
    multiplier_min = multiplier_max = 1.
    if grey['variant'] in ('MF', 'MGF'):
        delta = np.zeros_like(data.upper_water)
        delta[1:] = np.log1p(data.upper_water[1:])-np.log1p(data.upper_water[:-1])
        driver = np.tanh(delta/grey['dw_scale'])
        modification = np.log(4.)*np.tanh(parameters['eta_rising']*driver/np.log(4.))
        multiplier_min = float(np.exp(modification.min())); multiplier_max = float(np.exp(modification.max()))
        loghazard += modification
    if 'internal_response' in design:
        contract=design['internal_response']
        if contract['antecedent_days']!=30 or contract['include_current_day']:
            raise ValueError('INDEPENDENT_INTERNAL_CONTRACT_CHANGED')
        # Independent lag summation, not the training cumulative-sum routine.
        w=np.asarray(data.soil_wetness,float); antecedent=np.zeros_like(w)
        for lag in range(1,31):
            antecedent[:lag]+=w[0]
            antecedent[lag:]+=w[:-lag]
        antecedent/=30
        x=2*w-1; y=2*antecedent-1
        px=[np.ones_like(x),x,(3*x*x-1)/2]; py=[np.ones_like(y),y,(3*y*y-1)/2]
        eta=np.zeros_like(w)
        for i,j in [(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)]:
            eta+=parameters[f'wet_response_{i}{j}']*px[i]*py[j]
        modification=np.log(4.)*np.tanh(eta/np.log(4.))
        loghazard+=modification
        multiplier_min=float(np.exp(modification.min()));multiplier_max=float(np.exp(modification.max()))
    # Capping h at 700 gives exactly the registered saturated probability and
    # avoids overflowing before expm1. A zero contact stays zero regardless of d.
    hazard = np.where(positive, np.exp(np.minimum(loghazard, np.log(700.))), 0.)
    survival = np.exp(-np.exp(-logtau))
    aq = np.exp(parameters['log_aq'])
    fraction = aq*data.fast_fraction/(aq*data.fast_fraction+1-data.fast_fraction)
    diagnostics = {'contact_mapping_saturation_fraction': float(np.mean((expit(la)<1e-6)|(expit(la)>1-1e-6))),
                   'lifetime_mapping_saturation_fraction': float(np.mean((expit(lt)<1e-6)|(expit(lt)>1-1e-6))),
                   'hazard_cap_fraction': float(np.mean(positive & (loghazard>=np.log(700.)))),
                   'modifier_minimum': multiplier_min, 'modifier_maximum': multiplier_max,
                   'log_alpha_minimum': float(alpha.min()), 'log_alpha_maximum': float(alpha.max()),
                   'lifetime_days_minimum': float(np.exp(logtau).min()), 'lifetime_days_maximum': float(np.exp(logtau).max())}
    return hazard, survival, fraction, parameters['v_f'], diagnostics


def replay_stores(hazard, survival, fraction, release, source, demand, month_index, check=lambda: None):
    """Independent daily equations, retaining every mass and ending state."""
    nd, nr = hazard.shape
    if not (np.isfinite(hazard).all() and np.min(hazard)>=0):
        raise ValueError('INVALID_HAZARD')
    for arr in [survival, fraction, release]:
        if not (np.isfinite(arr).all() and np.min(arr)>=0 and np.max(arr)<=1):
            raise ValueError('INVALID_PARTITION_OR_SURVIVAL')
    if min(float(source.min()), float(demand.min())) < 0:
        raise ValueError('NEGATIVE_INPUT_OR_DEMAND')
    keys = ('source', 'demand', 'uptake', 'available', 'probability', 'fast', 'slow', 'injected', 'loss', 'mineral_M', 'slow_water_L', 'balance')
    out = {key: np.zeros((nd, nr)) for key in keys}
    mineral = np.zeros(nr); slow = np.zeros(nr)
    max_relative = 0.
    for t in range(nd):
        m = month_index[t]; first = t == 0 or month_index[t-1] != m
        if first: check()
        inp = source[m] if first else np.zeros(nr)
        cap = demand[m] if first else np.zeros(nr)
        before = mineral+inp; uptake = np.minimum(before, cap); available = before-uptake
        # expm1 keeps precision at very small contact.
        probability = -np.expm1(-np.minimum(hazard[t], 700.))
        mobile = probability*available; fast = fraction[t]*mobile; injected = mobile-fast
        slow_before = slow+injected; slow_out = release[t]*slow_before
        remaining = available-mobile
        loss = remaining*(1-survival); next_mineral = remaining-loss; next_slow = slow_before-slow_out
        balance = mineral+slow+inp-uptake-fast-slow_out-loss-next_mineral-next_slow
        scale = np.maximum(1., mineral+slow+inp)
        max_relative = max(max_relative, float(np.max(np.abs(balance)/scale)))
        values = (inp, cap, uptake, available, probability, fast, slow_out, injected, loss, next_mineral, next_slow, balance)
        for key, value in zip(keys, values): out[key][t] = value
        mineral, slow = next_mineral, next_slow
    return out, max_relative


def replay_artifact(data, artifact, check=lambda: None):
    h, s, f, vf, diagnostic = independent_decode(data, artifact)
    daily, relative = replay_stores(h, s, f, data.lower_release, data.source, data.crop, data.mid, check)
    lateral = daily['fast']+daily['slow']
    check(); routed = route(data, lateral, vf=vf, exposure='monthly')
    check(); water = route(data, data.fast_water+data.slow_water, vf=0., water_replay=True)
    total_input = float(daily['source'].sum())
    total_balance = total_input-sum(float(daily[k].sum()) for k in ['uptake', 'fast', 'slow', 'loss'])
    total_balance -= float(daily['mineral_M'][-1].sum()+daily['slow_water_L'][-1].sum())
    river_balance = float(lateral.sum()-routed['channel_removed'].sum()-routed['terminal'].sum()-routed['stocks'][-1].sum())
    minimum = min(float(daily[k].min()) for k in ['uptake', 'fast', 'slow', 'loss', 'mineral_M', 'slow_water_L'])
    excess = float(np.max(daily['uptake']-daily['demand']))
    zero_contact = float(np.max(daily['probability'][data.contact == 0])) if np.any(data.contact == 0) else 0.
    checks = {'daily_conservation': relative < 1e-10,
              'full_store_conservation': abs(total_balance)/max(1., total_input)<1e-10,
              'river_reservoir_conservation': abs(river_balance)/max(1., float(lateral.sum()))<1e-10,
              'nonnegative_stocks_fluxes': minimum >= -1e-8, 'uptake_within_demand': excess <= 1e-6,
              'zero_contact_zero_mobilization': zero_contact == 0.,
              'modifier_bound': diagnostic['modifier_minimum'] >= .25-1e-12 and diagnostic['modifier_maximum'] <= 4.+1e-12}
    annual = []
    for year in sorted(set(data.dates.year)):
        mask = data.dates.year == year
        row = {k: float(daily[k][mask].sum()) for k in ['source', 'demand', 'uptake', 'fast', 'slow', 'loss']}
        row.update(year=int(year), uptake_demand_ratio=row['uptake']/row['demand'] if row['demand'] else 0.)
        annual.append(row)
    diagnostic.update(mineral_empty_fraction=float(np.mean(daily['mineral_M'] == 0)),
                      slow_empty_fraction=float(np.mean(daily['slow_water_L'] == 0)),
                      uptake_clears_supply_fraction=float(np.mean((daily['demand'] > 0)&(daily['available'] == 0))))
    summary = {'physical_pass': all(checks.values()), 'checks': checks, 'annual': annual,
               'maximum_relative_daily_balance': relative, 'full_balance_kg_n': total_balance,
               'river_balance_kg_n': river_balance, 'minimum_stock_or_flux': minimum,
               'maximum_uptake_excess_kg_n': excess, 'diagnostics': diagnostic,
               'state_definitions': {'mineral_M': 'existing mineral nitrogen legacy with lifetime loss',
                                     'slow_water_L': 'slow-water nitrogen; old replay legacy alias',
                                     'new_stores': 0}, 'simulation_start': str(data.dates[0]),
               'simulation_end': str(data.dates[-1])}
    return summary, {'daily': daily, 'local': lateral, 'routed': routed, 'water': water, 'vf': vf}


def prediction(data, replay, metadata):
    return observe(data, replay['local'], replay['routed'], replay['water'], metadata, replay['vf'])


def audit_fit(tag, product=None, evaluation=None, suffix=''):
    require_environment(); guard(6)
    fitpath = RUN/'reports/fits'/f'{tag}.json'; fit = read(fitpath)
    if fit['status'] != 'COMPLETE' or sha(fit['model_path']) != fit['model_sha256']:
        raise RuntimeError('AUDIT_FIT_OR_ARTIFACT_IDENTITY')
    artifact = torch.load(fit['model_path'], weights_only=False, map_location='cpu')
    if artifact['identity'] != fit['identity']:
        raise RuntimeError('AUDIT_FIT_IDENTITY_MISMATCH')
    spec = fit['identity']['spec']; product = product or spec['product']
    cached = read(RUN/'reports/cache_manifest.json')['products'][product]['data']
    data = torch.load(cached['path'], weights_only=False, map_location='cpu')
    result, values = replay_artifact(data, artifact, guard)
    artifacts = {}; name = tag+suffix
    if True:
        comparisons = []
        for phase, path in fit['predictions'].items():
            if sha(path) != fit['artifacts_sha256'][str(path)]: raise RuntimeError('PREDICTION_ARTIFACT_CHANGED')
            expected = pd.read_parquet(path)
            actual = prediction(data, values, prediction_metadata(expected))
            error = float(np.max(np.abs(actual-expected.prediction_mg_l.to_numpy())))
            np.testing.assert_allclose(actual, expected.prediction_mg_l.to_numpy(), rtol=3e-8, atol=3e-8)
            comparisons.append({'phase': phase, 'rows': len(expected), 'maximum_error_mg_l': error})
        result['prediction_checks'] = comparisons
    if evaluation is not None:
        evaluation = evaluation.copy()
        evaluation['prediction_mg_l'] = prediction(data, values, prediction_metadata(evaluation))
        train_ids = set(artifact['training_ids'])
        if train_ids & set(evaluation.observation_id): raise RuntimeError('EXTRAPOLATION_OVERLAPS_TRAINING')
        path = local(RUN/'outputs/predictions'/f'{name}_evaluation.parquet')
        evaluation.to_parquet(path, index=False); artifacts[str(path)] = sha(path)
        result['prediction_path'] = str(path)
    nr = len(data.area_ha)
    monthly = pd.DataFrame({'month': np.repeat(data.months, nr), 'reach_id': np.tile(np.arange(1,nr+1),len(data.months))})
    for key in ['source', 'demand', 'uptake', 'fast', 'slow', 'injected', 'loss', 'balance']:
        monthly[key+'_kg_n'] = data.monthly_sum(values['daily'][key]).ravel()
    for key in ['mineral_M', 'slow_water_L']:
        monthly[key+'_ending_kg_n'] = values['daily'][key][data.stops-1].ravel()
    monthly['available_zero_fraction']=data.monthly_sum((values['daily']['available']==0).astype(float)).ravel()/np.repeat(data.stops-data.starts,nr)
    monthly['available_mean_kg_n']=data.monthly_sum(values['daily']['available']).ravel()/np.repeat(data.stops-data.starts,nr)
    # Frozen hydrological drivers and actual effective response, at the same
    # month/reach grain as the process ledger. No TN labels enter these fields.
    ww=np.asarray(data.soil_wetness,float);aa=np.zeros_like(ww)
    for lag in range(1,31):aa[:lag]+=ww[0];aa[lag:]+=ww[:-lag]
    aa/=30
    vv=dict(zip(artifact['parameter_names'],artifact['theta']))
    xx=2*ww-1;yy=2*aa-1;px=[np.ones_like(xx),xx,(3*xx*xx-1)/2];py=[np.ones_like(yy),yy,(3*yy*yy-1)/2]
    ee=np.zeros_like(ww)
    for i,j in [(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)]:ee+=vv.get(f'wet_response_{i}{j}',0.)*px[i]*py[j]
    lm=np.log(4.)*np.tanh(ee/np.log(4.))
    for key,array in [('frozen_wetness',ww),('antecedent_30d_wetness',aa),('wetness_minus_antecedent',ww-aa),
                      ('log_mobilization_multiplier',lm),('mobilization_probability',values['daily']['probability'])]:
        monthly[key+'_mean']=data.monthly_sum(array).ravel()/np.repeat(data.stops-data.starts,nr)
    del ww,aa,xx,yy,px,py,ee,lm
    for key in ['inlet', 'official', 'channel_removed']:
        monthly['river_'+key+'_kg_n'] = data.monthly_sum(values['routed'][key]).ravel()
    path = local(RUN/'outputs/physical'/f'{name}_monthly_ledger.parquet')
    monthly.to_parquet(path, index=False); artifacts[str(path)] = sha(path)
    terminal = pd.DataFrame({'month': data.months,
                             'all_terminal_export_kg_n': data.monthly_sum(values['routed']['terminal'][:,None]).ravel()})
    path = local(RUN/'outputs/physical'/f'{name}_terminal_export.parquet')
    terminal.to_parquet(path,index=False); artifacts[str(path)] = sha(path)
    path = local(RUN/'outputs/physical'/f'{name}_daily_stores_ledger.npz')
    np.savez_compressed(path, **values['daily']); artifacts[str(path)] = sha(path)
    reservoir = pd.DataFrame(values['routed']['stocks'][data.stops-1])
    reservoir.insert(0, 'month', data.months)
    path = local(RUN/'outputs/physical'/f'{name}_reservoir_ending_stocks.parquet')
    reservoir.columns = list(map(str, reservoir.columns)); reservoir.to_parquet(path, index=False); artifacts[str(path)] = sha(path)
    from sl_tags import tag_paths,mixing_changes
    def tagged_metadata(frame):
        metadata=prediction_metadata(frame)
        metadata['station_key']=frame.station_key.to_numpy()
        return metadata
    metadata_frames=[tagged_metadata(pd.read_parquet(p)) for p in fit['predictions'].values()]
    if evaluation is not None:metadata_frames.append(tagged_metadata(evaluation))
    all_meta=pd.concat(metadata_frames,ignore_index=True).drop_duplicates('observation_id')
    tags,tag_audit=tag_paths(data,values,all_meta)
    for kind,table in [('path_tags',tags),('mixing_changes',mixing_changes(tags))]:
        tp=local(RUN/'outputs/physical'/f'{name}_{kind}.parquet');table.to_parquet(tp,index=False);artifacts[str(tp)]=sha(tp)
    result['tag_audit']=tag_audit
    result.update(status='PASS_PHYSICAL_REPLAY' if result['physical_pass'] else 'FAILED_PHYSICAL_REPLAY',
                  utc=now(), identity={'fit_sha256': sha(fitpath), 'model_sha256': fit['model_sha256'],
                                      'product': product, 'data_sha256': cached['sha256']}, artifacts_sha256=artifacts)
    write(RUN/'reports/physical'/f'{name}.json', result)
    if not result['physical_pass']: raise RuntimeError('FAILED_INDEPENDENT_PHYSICAL_REPLAY '+name)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--tag', required=True)
    audit_fit(parser.parse_args().tag)
