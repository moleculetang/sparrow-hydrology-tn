"""Issue the new training gate from direct and inherited evidence."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from d29_platform.runtime import sha,write_json
from d29_training.experiment_context import ExperimentContext

context=ExperimentContext.load()
old=ROOT.parent/'20260927_2'
physical=['d29_platform/precision_candidate.py','d29_platform/precision_transfer.py',
          'd29_platform/mixture_tags_candidate.py','d29_platform/land1.py',
          'd29_platform/conditional_inputs.py','d29_platform/coupling.py',
          'config/conditional_reference.json','config/mineralization_reference.json',
          'outputs/agriculture_reference/annual_crop_source_reference.parquet',
          'outputs/agriculture_reference/annual_crop_activity_reference.parquet',
          'outputs/soil_reference/son_soc14_reference.csv']
errors=[];inherit=[]
for name in physical:
    if sha(ROOT/name)!=sha(old/name):errors.append('PHYSICAL_IMPLEMENTATION_CHANGED '+name)
    inherit.append({'path':name,'old_sha256':sha(old/name),'new_sha256':sha(ROOT/name)})
old_gate=json.loads((old/'outputs/land1_formal_gate_numerical_v2.json').read_text(encoding='utf-8'))
mass=json.loads((old/'outputs/authoritative_mixture_candidate/receipt.json').read_text(encoding='utf-8'))
safety=json.loads((old/'outputs/authoritative_safety_history/receipt.json').read_text(encoding='utf-8'))
if not old_gate['passed'] or not mass['strict_mass_pass'] or not safety['strict_mass_pass'] or not safety['nonnegative_and_plant_budget_pass']:
    errors.append('INHERITED_PHYSICAL_GATE_NOT_PASSED')
if mass['days']!=23376 or mass['reaches']!=230 or mass['local_max_balance_kg']>1e-6 or mass['label_max_difference_kg']>1e-6:
    errors.append('INHERITED_MASS_OR_LABEL_TOLERANCE')
required={
    'u_gradient':ROOT/'outputs/u_full_history_gradient.json',
    'u_mass':ROOT/'outputs/u_initial_ledger.json',
    'land1_gradient':ROOT/'outputs/land1_review_gradient.json',
    'land1_objective':ROOT/'outputs/land1_independent_objective_review.json',
    'u_label_isolation':ROOT/'outputs/label_isolation_real_support.json',
    'land1_label_isolation':ROOT/'outputs/land1_label_isolation_review.json',
    'engineering_tests':ROOT/'outputs/review_engineering_tests.json',
    'observation_support':ROOT/'evidence/observation_support_review/support_audit.json',
    'runtime_inputs':ROOT/'evidence/runtime_input_identity_review.json'}
data={}
for name,path in required.items():
    if not path.exists():errors.append('MISSING_'+name)
    else:data[name]=json.loads(path.read_text(encoding='utf-8'))
if 'u_gradient' in data and (data['u_gradient']['status']!='passed' or len(data['u_gradient']['coordinates'])!=31):errors.append('U_GRADIENT_NOT_PASSED')
if 'u_gradient' in data:
    for path,digest in data['u_gradient'].get('identity',{}).items():
        if sha(path)!=digest:errors.append('U_GRADIENT_IMPLEMENTATION_CHANGED '+path)
    if len(data['u_gradient'].get('identity',{}))!=3:errors.append('U_GRADIENT_IDENTITY_INCOMPLETE')
if 'u_mass' in data and not data['u_mass']['passed']:errors.append('U_LEDGER_NOT_PASSED')
if 'land1_gradient' in data and (data['land1_gradient']['status'] not in ('passed','passed_with_recorded_refinement') or len(data['land1_gradient']['coordinates'])!=23):errors.append('LAND1_GRADIENT_NOT_PASSED')
if 'land1_objective' in data:
    if not data['land1_objective']['passed'] or data['land1_objective']['executed_kernel_sha256']!=sha(ROOT/'d29_platform/precision_candidate.py'):
        errors.append('LAND1_OBJECTIVE_OR_KERNEL_NOT_PASSED')
for name in ('u_label_isolation','land1_label_isolation'):
    if name in data and not data[name]['passed']:errors.append('LABEL_ISOLATION_FAILED '+name)
if 'engineering_tests' in data and not data['engineering_tests']['passed']:errors.append('ENGINEERING_TESTS_FAILED')
if 'runtime_inputs' in data:
    if len(data['runtime_inputs']['training_contracts'])!=32 or not all(r['training_contract_unchanged'] for r in data['runtime_inputs']['training_contracts']):
        errors.append('TRAINING_CONTRACT_MISMATCH')
    for path,digest in data['runtime_inputs']['files'].items():
        if sha(path)!=digest:errors.append('RUNTIME_INPUT_CHANGED '+path)
gate=ROOT/'outputs/land1_formal_gate.json'
write_json(gate,{'passed':not errors,'blocked_by':errors,'experiment_id':context.experiment_id,
    'identities':{str(ROOT/name):sha(ROOT/name) for name in physical+[
        'd29_training/land1_adapter.py','d29_training/annual_chain.py','d29_training/u_adapter.py',
        'd29_training/objective.py','d29_training/observations.py','d29_training/experiment_context.py','config/experiment_context.json','config/study.json']},
    'inherited_physics':{'old_gate_sha256':sha(old/'outputs/land1_formal_gate_numerical_v2.json'),
        'mass_receipt_sha256':sha(old/'outputs/authoritative_mixture_candidate/receipt.json'),
        'safety_receipt_sha256':sha(old/'outputs/authoritative_safety_history/receipt.json'),
        'identical_code_and_configuration':inherit},
    'new_evidence':{name:{'path':str(path),'sha256':sha(path)} for name,path in required.items() if path.exists()},
    'scope':'Inherited full-history physical/source tests for byte-identical core; new objective and 23-coordinate full-history gradient recomputed in this experiment.',
    'NSE':'not applicable to implementation gate'})
print('passed',not errors,'errors',errors)
if errors:raise RuntimeError('REVIEW_GATE_BLOCKED')
