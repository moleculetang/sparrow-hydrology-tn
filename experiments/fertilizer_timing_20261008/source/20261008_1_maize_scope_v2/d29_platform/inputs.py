"""External-only source correction and explicit entry allocation."""
import math
import numpy as np

def corrected_external_inputs(raw_kg_day, eta, *, role, already_corrected=False):
    if role!='external_land_input' or already_corrected:
        raise ValueError('SOURCE_MULTIPLIER_REQUIRES_UNCORRECTED_EXTERNAL_INPUT')
    if not np.isfinite(eta) or abs(eta)>math.log(4):raise ValueError('SOURCE_MULTIPLIER_BOUND')
    raw=np.asarray(raw_kg_day,dtype=np.float64)
    if not np.isfinite(raw).all() or np.any(raw<0):raise ValueError('MISSING_OR_NEGATIVE_RAW_SOURCE')
    value=raw*np.exp(eta)
    return value,{'c':float(np.exp(eta)),'already_corrected':True,'role':role,'derivative_wrt_eta':value}

def allocate_external_entries(raw_tagged, fractions):
    """[T,R,L,K] kg/day × explicit [K,4] fractions -> tagged LAND1 entries."""
    a=np.asarray(raw_tagged,dtype=np.float64);f=np.asarray(fractions,dtype=np.float64)
    if a.ndim!=4 or f.shape!=(a.shape[-1],4):raise ValueError('ENTRY_ALLOCATION_SHAPE')
    if not np.isfinite(a).all() or not np.isfinite(f).all() or np.any(a<0) or np.any(f<0):raise ValueError('MISSING_ENTRY_ALLOCATION')
    if not np.allclose(f.sum(1),1,rtol=0,atol=1e-12):raise ValueError('ENTRY_FRACTIONS_MUST_CLOSE')
    return a[...,None]*f
