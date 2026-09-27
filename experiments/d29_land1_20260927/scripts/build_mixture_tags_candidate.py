"""Use the actual tracer mixture for proportional plant attribution.

The denominator is independently summed from labelled pre-outflow masses,
not the separately rounded physical total. This is a mixing definition, not
the addition of a mass-balance residual to a chosen label. Local label budgets
and label sums are still separately checked at the unchanged absolute limit.
"""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
s=(ROOT/'d29_platform/precision_tags_candidate.py').read_text(encoding='utf-8')
needle='            flux_sum = np.zeros(4, np.float64)'
s=s.replace(needle,'''            mixture=np.empty(nk,np.float64)
            for label in range(nk):
                p0,a0,b0,n0,l0=current[u,label]
                x0=n0+source[t,u,label,3]+pa[t,u]*a0+pp[t,u]*b0
                mixture[label]=p0+source[t,u,label,0]+uptake_fraction*x0
            mixture_total=_accurate_sum(mixture)
''' + needle,1)
s=s.replace('share = p_pre / plant_pre if plant_pre > 0 else 0.0','share = p_pre / mixture_total if mixture_total > 0 else 0.0',1)
(ROOT/'d29_platform/mixture_tags_candidate.py').write_text(s,encoding='utf-8')
