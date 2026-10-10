"""Adapt the accepted compensated mixture labels to the new uptake capacity."""
from pathlib import Path
root=Path(__file__).resolve().parents[1]
s=(root/'d29_platform/mixture_tags_candidate.py').read_text(encoding='utf8')
s=s.replace('from .precision_candidate import','from .timing_kernel import')
s=s.replace('initial_compensation, physical_correction):','initial_compensation, physical_correction, cap):')
s=s.replace('uptake = min(X, need)','uptake = min(X, need, cap[t,u])')
s=s.replace('raw_need > 0 and X >= need else','raw_need > 0 and X >= need and cap[t,u]>=need else')
s=s.replace("corr, i['compensation_history'])", "corr, i['compensation_history'],i['plant_growth_capacity'])")
(root/'d29_platform/timing_labels.py').write_text(s,encoding='utf8')
print('New passive tags replay planned demand and capped physical uptake.')
