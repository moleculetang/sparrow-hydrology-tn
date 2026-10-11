"""Explicit published source conversions; no TN, fitted coefficients or gap filling.

BNF: Ludemann et al. 2024, DOI 10.5194/essd-16-525-2024,
supplement S2 equations 2–4 / Table S2, Asia soybean Ndfa.
SoilGrids: official layers mapped-unit conversion table. Total nitrogen is NOT
an observation of organic nitrogen or a historical 1961 stock.
"""
import numpy as np

LEGUME_ASIA = {
    'soybean': (.2775,.1178,.03913,-.00118,1.4,.61),
    'groundnut': (.2614,.1343,.027,0.,1.4,.62),
    'common_bean': (.2839,.0804,.025,0.,1.4,.38),
    'chickpea': (.2839,.0804,.019,0.,2.,.62),
    'pigeon_pea': (.1647,.0517,.019,0.,2.,.74),
    'faba_bean': (.2839,.0804,.025,0.,1.4,.74),
    'lupin': (.2839,.0804,.027,0.,1.4,.74),
    'other_grain_legumes': (.2839,.0804,.025,0.,1.4,.62),
}

def _finite_nonnegative(value, name):
    a=np.asarray(value,dtype=np.float64)
    if not np.isfinite(a).all() or np.any(a<0):
        raise ValueError('UNKNOWN_OR_NEGATIVE_'+name)
    return a

def crop_bnf_kg(crop, harvested_area_ha, commodity_tonnes):
    """Commodity moisture as FAO yield, not pre-converted to dry mass.

    Rice and sugarcane use harvested hectares, not physical hectares with an
    additional multiple-cropping factor. Unsupported crops remain unsupported.
    Positive harvested area with zero yield is outside the legume log model;
    reject rather than silently insert epsilon or return zero.
    """
    area,yield_t=np.broadcast_arrays(_finite_nonnegative(harvested_area_ha,'AREA'),
                                    _finite_nonnegative(commodity_tonnes,'PRODUCTION'))
    if np.any((area==0)&(yield_t>0)):raise ValueError('PRODUCTION_WITHOUT_AREA')
    if crop in ('rice','sugarcane'):return area*25.
    if crop not in LEGUME_ASIA:raise ValueError('UNSUPPORTED_BNF_CROP')
    if np.any((area>0)&(yield_t==0)):raise ValueError('ZERO_YIELD_LOG_BNF_UNDEFINED')
    active=area>0
    Y=np.divide(yield_t,area,out=np.ones_like(area),where=active)
    a,b,c,d,bgf,ndfa=LEGUME_ASIA[crop]
    hi=a+b*np.log(Y)
    if np.any(active&((hi<=0)|(hi>1))):raise ValueError('BNF_HARVEST_INDEX_OUTSIDE_PHYSICAL_RANGE')
    nshoot=c+d*Y/hi
    if np.any(active&((nshoot<=0)|(nshoot>1))):raise ValueError('BNF_SHOOT_N_OUTSIDE_PHYSICAL_RANGE')
    return np.where(active,yield_t*1000*nshoot/hi*bgf*ndfa,0.)

def soil_total_n_stock_kg_m2(nitrogen_cg_kg, bdod_cg_cm3, cfvo_permille, thickness_m):
    """Fine-earth mass × total-N fraction; no SOC C:N conversion or SON claim."""
    n=_finite_nonnegative(nitrogen_cg_kg,'TOTAL_N')
    bd=_finite_nonnegative(bdod_cg_cm3,'BULK_DENSITY')
    cf=_finite_nonnegative(cfvo_permille,'COARSE_FRAGMENTS')
    depth=_finite_nonnegative(thickness_m,'THICKNESS')
    if np.any(cf>1000) or np.any(depth<=0):raise ValueError('SOIL_UNIT_RANGE')
    # cg/kg -> kg/kg = 1e-5; cg/cm3 -> kg/m3 = 10.
    return n*1e-5*(bd*10.)*depth*(1-cf/1000.)

def mutually_exclusive_residue_fates(total_n, *, returned, removed, burned, grazed, other):
    total=_finite_nonnegative(total_n,'RESIDUE_N')
    f=np.asarray([returned,removed,burned,grazed,other],dtype=np.float64)
    if not np.isfinite(f).all() or np.any(f<0) or not np.isclose(f.sum(),1,atol=1e-12,rtol=0):
        raise ValueError('RESIDUE_FATES_NOT_EXCLUSIVE_AND_COMPLETE')
    return total[...,None]*f

def apply_national_ratio_once(previous_basin_total, national_current, national_previous, allocation_weights):
    """National trend is a scenario; normalized local weights only redistribute."""
    old=_finite_nonnegative(previous_basin_total,'BASIN_TOTAL')
    current=_finite_nonnegative(national_current,'NATIONAL_CURRENT')
    prior=_finite_nonnegative(national_previous,'NATIONAL_PREVIOUS')
    w=_finite_nonnegative(allocation_weights,'ALLOCATION_WEIGHTS')
    if np.any(prior<=0) or w.sum()<=0:raise ValueError('UNDEFINED_DOWNSCALING_RATIO')
    return old*(current/prior)*w/w.sum()

def crop_mass_from_rates(area_ha,rate,*,unit,known_no_crop=None,known_zero_rate=None):
    """Prevent legacy invalid-rate -> zero conversion in the new source path.

    Neither a missing rate nor an area NaN alone proves known absence. Explicit
    masks require provenance supplied by the adapter. Invalid positive-area
    rate support blocks a complete mass field rather than hiding its deficit.
    """
    if unit!='kg N ha-1 yr-1':raise ValueError('UNVERIFIED_RATE_OR_MASS_UNIT')
    a,r=np.broadcast_arrays(np.asarray(area_ha,dtype=np.float64),np.asarray(rate,dtype=np.float64))
    no=np.zeros(a.shape,bool) if known_no_crop is None else np.broadcast_to(np.asarray(known_no_crop,bool),a.shape)
    zero=np.zeros(a.shape,bool) if known_zero_rate is None else np.broadcast_to(np.asarray(known_zero_rate,bool),a.shape)
    if np.any(no&np.isfinite(a)&(a>0)):raise ValueError('NO_CROP_CONTRADICTS_POSITIVE_AREA')
    if np.any(zero&np.isfinite(r)&(r>0)):raise ValueError('ZERO_RATE_CONTRADICTS_POSITIVE_RATE')
    if np.any((~np.isfinite(a)|(a<0))&~no):raise ValueError('UNKNOWN_HARVEST_AREA_NOT_ZERO')
    av=np.where(no,0.,a)
    if np.any((av>0)&(~np.isfinite(r)|(r<0))&~zero):raise ValueError('POSITIVE_AREA_RATE_UNKNOWN_NOT_ZERO')
    rr=np.where((av==0)|zero,0.,r)
    return av*rr
