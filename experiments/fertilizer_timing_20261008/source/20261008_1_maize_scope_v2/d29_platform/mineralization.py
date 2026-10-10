"""First-order SON release with explicit rate identity and analytic pullback.

No automatic air-temperature substitution, partial-history switch or gap fill.
Temperature mode is conditional on a complete, independently prepared soil-T
array. Selection of this mode is outside the physical kernel.
"""
import numpy as np

def first_order_probability(k_per_day, *, dt_days=1., mode='fixed', soil_temperature_c=None,
                            q10=None, reference_temperature_c=None):
    k=np.asarray(k_per_day,dtype=np.float64);dt=np.asarray(dt_days,dtype=np.float64)
    if not np.isfinite(k).all() or not np.isfinite(dt).all() or np.any(k<0) or np.any(dt<=0):
        raise ValueError('INVALID_MINERALIZATION_RATE_OR_TIMESTEP')
    if mode=='fixed':
        if any(x is not None for x in (soil_temperature_c,q10,reference_temperature_c)):
            raise ValueError('FIXED_RATE_MUST_NOT_SILENTLY_IGNORE_TEMPERATURE')
        factor=1.;temp=None
    elif mode=='soil_temperature_q10':
        if any(x is None for x in (soil_temperature_c,q10,reference_temperature_c)):
            raise ValueError('COMPLETE_SOIL_TEMPERATURE_CONFIGURATION_REQUIRED')
        temp=np.asarray(soil_temperature_c,dtype=np.float64)
        if not np.isfinite(temp).all() or not np.isfinite(q10) or q10<=0 or not np.isfinite(reference_temperature_c):
            raise ValueError('MISSING_OR_INVALID_SOIL_TEMPERATURE_CONFIGURATION')
        factor=np.exp(np.log(q10)*(temp-reference_temperature_c)/10.)
    else:raise ValueError('UNKNOWN_MINERALIZATION_MODE')
    h=k*dt*factor
    if not np.isfinite(h).all():raise ValueError('NONFINITE_MINERALIZATION_HAZARD')
    # Exact first-order integration; no linear k*S Euler step and no stock clip.
    p=-np.expm1(-h);survival=np.exp(-h)
    derivatives={'k_per_day':survival*dt*factor,'dt_days':survival*k*factor}
    if temp is not None:
        derivatives.update(soil_temperature_c=survival*h*np.log(q10)/10.,
            q10=survival*h*(temp-reference_temperature_c)/(10.*q10),
            reference_temperature_c=-survival*h*np.log(q10)/10.)
    return p,derivatives

def land1_mineralization_fields(k_active_per_day,k_protected_per_day,**kwargs):
    if np.any(np.asarray(k_protected_per_day)>np.asarray(k_active_per_day)):
        raise ValueError('PROTECTED_RATE_EXCEEDS_ACTIVE_RATE')
    a,da=first_order_probability(k_active_per_day,**kwargs)
    p,dp=first_order_probability(k_protected_per_day,**kwargs)
    return {'mineralize_active':a,'mineralize_protected':p},{'active':da,'protected':dp}

def mineralization_pullback(land1_gradient,derivatives):
    """Chain full-history LAND1 adjoint to two globally shared release rates.

    Returns elementwise T gradient if applicable; summation over land classes
    to a shared reach temperature remains an explicit caller responsibility.
    """
    ga=np.asarray(land1_gradient['mineralize_active'],dtype=np.float64)
    gp=np.asarray(land1_gradient['mineralize_protected'],dtype=np.float64)
    out={'k_active_per_day':float(np.sum(ga*derivatives['active']['k_per_day'])),
         'k_protected_per_day':float(np.sum(gp*derivatives['protected']['k_per_day']))}
    if 'soil_temperature_c' in derivatives['active']:
        out['soil_temperature_c']=ga*derivatives['active']['soil_temperature_c']+gp*derivatives['protected']['soil_temperature_c']
        for name in ('q10','reference_temperature_c'):
            out[name]=float(np.sum(ga*derivatives['active'][name]+gp*derivatives['protected'][name]))
    return out
