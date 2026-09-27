"""Full-history candidate derivative gate, with forward-only difference calls.

Separate checkpoint and identities from v2; no reuse of old coordinate passes.
The observation objective is identical in analytic and forward-only execution.
"""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
s=(ROOT/'scripts/verify_land1_gradient.py').read_text(encoding='utf-8')
s=s.replace('from d29_training.land1_adapter import Land1Training',
'''import d29_training.annual_chain as annual_module
from d29_platform import precision_candidate as precise
annual_module.run_land1=precise.run_land1
annual_module.land1_adjoint=precise.land1_adjoint
from d29_training.land1_adapter import Land1Training''')
s=s.replace('land1_full_history_gradient_v2.json','land1_full_history_gradient_v3.json')
s=s.replace('land1_gradient_worker.json','land1_precision_gradient_worker.json')
s=s.replace('land1_gradient_evaluations.jsonl','land1_precision_gradient_evaluations.jsonl')
s=s.replace('stop_land1_gradient.flag','stop_land1_gradient_v3.flag')
s=s.replace("[ROOT/'d29_platform/land1.py',", "[ROOT/'d29_platform/precision_candidate.py',ROOT/'d29_platform/precision_transfer.py',ROOT/'d29_platform/land1.py',")
s=s.replace('[1e-4,3e-5,1e-5]','[1e-5,3e-6,1e-6]')
s=s.replace('value,_=a.value_gradient(point)','value,_=a.value_gradient(point,forward_only=True)')
exec(compile(s,str(ROOT/'scripts/verify_land1_precision_gradient.py'),'exec'))
