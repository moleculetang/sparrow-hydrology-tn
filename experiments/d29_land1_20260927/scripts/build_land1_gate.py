"""Issue only an evidence-backed implementation gate; never starts a fit."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from d29_platform.runtime import write_json,sha
from d29_training.derivative_gate import refinement_passes

def main():
    required={
        'gradient':ROOT/'outputs/land1_full_history_gradient_v5.json',
        'mass':ROOT/'outputs/mixture_precision_candidate/receipt.json',
        'safety':ROOT/'outputs/precision_safety_history/receipt.json',
        'objective':ROOT/'outputs/land1_independent_objective.json',
        'training_isolation':ROOT/'outputs/label_isolation_real_support.json',
        'land1_isolation':ROOT/'outputs/land1_label_isolation.json',
        'inputs':ROOT/'evidence/runtime_input_identity.json',
        'reference_tests':ROOT/'outputs/precision_candidate_test_receipt.json',
    }
    errors=[];data={};gradient_review=[]
    for key,path in required.items():
        if not path.exists():errors.append('missing:'+key)
        else:data[key]=json.loads(path.read_text(encoding='utf-8'))
    if 'gradient' in data:
        g=data['gradient']
        review_path=ROOT/'outputs/land1_gradient_v5_refinement_reviews.json'
        reviews=json.loads(review_path.read_text(encoding='utf-8'))['reviews'] if review_path.exists() else []
        accepted=[]
        for c in g.get('coordinates',[]):
            good=c['passed'];entry={'coordinate':c['index'],'original_scan_passed':good,'accepted_refinement':False}
            if not good:
                for review in reviews:
                    if review['coordinate']!=c['index'] or not review.get('accepted_as_local_smooth_derivative_evidence'):continue
                    if sha(review['path'])!=review['sha256']:errors.append('refinement_receipt_changed');continue
                    diagnostic=json.loads(Path(review['path']).read_text(encoding='utf-8'))
                    if refinement_passes(g,c['index'],diagnostic):
                        good=True;entry.update(accepted_refinement=True,receipt=review)
            accepted.append(good);gradient_review.append(entry)
        if g.get('status') not in ('passed','requires_branch_review') or [c['index'] for c in g.get('coordinates',[])]!=list(range(23)) or not all(accepted):errors.append('complete_23_coordinate_gradient_not_passed')
        for path,digest in g['identity'].items():
            if sha(path)!=digest:errors.append('gradient_implementation_changed:'+path)
    for key in ('mass','safety'):
        if key not in data:continue
        d=data[key]
        if not d.get('strict_mass_pass') or not d.get('full_history_continuity') or not d.get('implementation_unchanged'):errors.append(key+'_not_passed')
        if d.get('reaches')!=230 or d.get('days')!=23376:errors.append(key+'_incomplete_history')
        configuration=required[key].parent/'configuration_frozen.json'
        if not configuration.exists():errors.append(key+'_configuration_missing')
        else:
            frozen=json.loads(configuration.read_text(encoding='utf-8'))
            for path,digest in frozen['implementation_hashes'].items():
                if sha(path)!=digest:errors.append(key+'_implementation_changed:'+path)
            if required[key].stat().st_mtime < configuration.stat().st_mtime:
                errors.append(key+'_receipt_predates_configuration')
    if 'safety' in data and not data['safety'].get('nonnegative_and_plant_budget_pass'):errors.append('nonnegative_or_plant_budget_failed')
    if 'objective' in data:
        if not data['objective'].get('passed'):errors.append('independent_objective_failed')
        if data['objective'].get('candidate_sha256')!=sha(ROOT/'d29_platform/precision_candidate.py'):
            errors.append('independent_objective_candidate_changed')
        if data['objective'].get('adapter_sha256')!=sha(ROOT/'d29_training/land1_adapter.py'):
            errors.append('independent_objective_adapter_changed')
    if 'training_isolation' in data and not data['training_isolation'].get('passed'):errors.append('training_isolation_failed')
    if 'land1_isolation' in data:
        if not data['land1_isolation'].get('passed'):errors.append('land1_label_isolation_failed')
        for name,key in [('precision_candidate','candidate_sha256'),('land1_adapter','adapter_sha256')]:
            base='d29_platform' if name=='precision_candidate' else 'd29_training'
            if sha(ROOT/base/(name+'.py'))!=data['land1_isolation'][key]:errors.append('land1_isolation_implementation_changed')
    if 'reference_tests' in data:
        if not data['reference_tests'].get('passed'):errors.append('reference_tests_failed')
        for path,digest in data['reference_tests']['identities'].items():
            if sha(path)!=digest:errors.append('reference_test_implementation_changed:'+path)
    if 'inputs' in data:
        if len(data['inputs']['training_contracts'])!=32 or not all(r['training_contract_unchanged'] for r in data['inputs']['training_contracts']):errors.append('training_contract_mismatch')
        for path,digest in data['inputs']['files'].items():
            if sha(path)!=digest:errors.append('runtime_input_changed:'+path)
    tests=ROOT/'outputs/precision_candidate_annual_tests.log'
    if not tests.exists() or 'Ran 9 tests' not in tests.read_text(encoding='utf-8') or not tests.read_text(encoding='utf-8').rstrip().endswith('OK'):errors.append('candidate_reference_and_annual_chain_tests_not_passed')
    files=[ROOT/'d29_platform/precision_candidate.py',ROOT/'d29_platform/precision_transfer.py',ROOT/'d29_platform/mixture_tags_candidate.py',ROOT/'d29_platform/land1.py',ROOT/'d29_training/land1_adapter.py',ROOT/'d29_training/annual_chain.py',ROOT/'d29_training/u_adapter.py',ROOT/'d29_training/objective.py',ROOT/'d29_platform/conditional_inputs.py',ROOT/'d29_platform/coupling.py',ROOT/'evidence/runtime_input_identity.json']
    files.extend([ROOT/'d29_training/derivative_gate.py',Path(__file__)])
    write_json(ROOT/'outputs/land1_formal_gate.json',{'passed':not errors,'blocked_by':errors,'gradient_review':gradient_review,'identities':{str(p):sha(p) for p in files},'evidence':{k:{'path':str(p),'sha256':sha(p)} for k,p in required.items() if p.exists()},'model':'LAND1 23-parameter conditional reference configuration','does_not_certify_real_source_or_water_accuracy':True,'numerical_stock_representation':'nonplant high-minus-compensation, four low parts retained per land unit','NSE':'not applicable to implementation acceptance'})
    print('passed',not errors,'blocked_by',errors)

if __name__=='__main__':main()
