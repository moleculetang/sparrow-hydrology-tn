"""Freeze exact files used by every job, audit and prediction readout."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from d29_platform.runtime import sha,write_json

target=ROOT/'config/experiment_context.json'
if target.exists():raise RuntimeError('CONTEXT_ALREADY_FROZEN')
files=['config/jobs.json','config/conditional_reference.json','config/mineralization_reference.json',
       'd29_training/objective.py','d29_training/observations.py','d29_training/u_adapter.py',
       'd29_training/land1_adapter.py','d29_training/annual_chain.py',
       'd29_platform/precision_candidate.py','d29_platform/precision_transfer.py',
       'evidence/runtime_input_identity_review.json','data/monthly_tn.parquet','data/hf_daily.parquet']
write_json(target,{'root':str(ROOT.resolve()),'experiment_id':'20260928_1_review_v1','namespace':'jobs',
                   'hashes':{p:sha(ROOT/p) for p in files},'no_namespace_fallback':True,
                   'NH4_DO':'excluded_from_training_and_routine_diagnostics'})
