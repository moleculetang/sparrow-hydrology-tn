"""Current adjoints against preserved FD slopes of byte-identical forwards.

No blanket stale-hash exemption. Every reused artifact, scientific identity,
coordinate, feasible side and adjacent branch-consistent pair is registered.
"""
import ast
import hashlib
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from d29_platform.runtime import sha,write_json
from d29_platform.luh3_adapter import load_area_transfers
from timing.derivative_gate import check


def forward_ast(path):
    return [hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest()
        for n in ast.parse(path.read_text(encoding='utf8')).body
        if not isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) or n.name not in ('_reverse','land1_adjoint')]


def main():
    artifacts={};pending=[];errors=[]
    def load(rel):
        p=ROOT/rel
        if not p.exists():pending.append(rel);return None
        artifacts[rel]=sha(p);return json.loads(p.read_text(encoding='utf8'))
    continuity=load('evidence/transition_reverse_replay_consistency.json')
    if not continuity or not continuity['passed']:raise RuntimeError('SMALL_REPAIR_PROOF_REQUIRED')
    oldhash=continuity['old_source_sha256'];newhash=sha(ROOT/'d29_platform/timing_kernel.py')
    snapshot=ROOT/continuity['preserved_snapshot']
    if continuity['new_source_sha256']!=newhash or sha(snapshot)!=oldhash or forward_ast(snapshot)!=forward_ast(ROOT/'d29_platform/timing_kernel.py'):
        raise RuntimeError('COMPLETE_FORWARD_AST_CONTINUITY_REQUIRED')
    artifacts[continuity['preserved_snapshot']]=sha(snapshot)
    probe_continuity=load('evidence/coordinate_probe_implementation_continuity.json')
    def validate(rel,record,old=True):
        if record is None:return
        for path,digest in record.get('identities',{}).items():
            if path=='config/workstation_hold.json':continue
            actual=sha(ROOT/path)
            if actual==digest:continue
            if path=='d29_platform/timing_kernel.py' and old and digest==oldhash:continue
            if path=='scripts/audit_solution.py' and probe_continuity and probe_continuity['passed']:
                c=probe_continuity
                if c['old_script_sha256']==digest and c['current_script_sha256']==actual and sha(ROOT/c['preserved_snapshot'])==digest and c['old_functions']==c['current_functions']:continue
            errors.append('UNJUSTIFIED_IDENTITY_CHANGE '+rel+' '+path)
    currents={}
    for jid in ('TIM_F23_C0_V0_s0','TIM_F23_K_V1_s0'):
        rel='outputs/jobs/transition_requalification_'+jid+'/transition_requalification.json'
        r=load(rel);validate(rel,r,False)
        if r and r['passed'] and r['old_objective_equal'] and all(x['bitwise_equal'] for x in r['forward_arrays'].values()):
            folder=ROOT/rel;folder=folder.parent
            for name,key in (('gradient.npy','gradient_sha256'),('initial_cotangent.npy','initial_cotangent_sha256'),('point.npy','point_sha256')):
                if sha(folder/name)!=r[key]:raise RuntimeError('CURRENT_ARRAY_HASH_MISMATCH')
                artifacts[(folder/name).relative_to(ROOT).as_posix()]=sha(folder/name)
            currents[jid]=(r,folder)
        elif r:errors.append('CURRENT_FORWARD_REGRESSION_FAILED '+jid)
    c0rel='outputs/jobs/TIM_F23_C0_V0_s0/prerequisite_audit.json'
    c0=load(c0rel);validate(c0rel,c0)
    krecords=[]
    for i in range(6):
        rel=f'outputs/imported_workstation_coordinates/outputs/jobs/prerequisite_K_coordinates_{i}/coordinate_slice.json'
        r=load(rel);validate(rel,r)
        if r and r.get('complete') and r.get('passed'):krecords.append((rel,r))
        elif r:pending.append('incomplete:'+rel)
    coordinate_cases=[]
    for jid,records in [('TIM_F23_C0_V0_s0',[(c0rel,c0)] if c0 else []),('TIM_F23_K_V1_s0',krecords)]:
        if jid not in currents:continue
        current,folder=currents[jid];gradient=np.load(folder/'gradient.npy',allow_pickle=False);point=np.load(folder/'point.npy',allow_pickle=False)
        cases=[];seen=set()
        for rel,r in records:
            if not np.array_equal(point,np.asarray(r['point'])):raise RuntimeError('PRESERVED_FD_POINT_CHANGED')
            if abs(current['objective']-r['objective'])>1e-8*(1+abs(r['objective'])):raise RuntimeError('PRESERVED_FD_OBJECTIVE_CHANGED')
            for ordinal,c in enumerate(r['checks']):
                index=c.get('coordinate_index',ordinal)
                if index in seen:raise RuntimeError('DUPLICATE_COORDINATE')
                seen.add(index)
                if c['coordinate']!=current['names'][index]:raise RuntimeError('COORDINATE_NAME_CHANGED')
                side=c['side'];sign=1.
                if side=='minus':sign=-1.
                elif side=='feasible_one_sided':
                    lower,upper=current['bounds'][index]
                    if upper is not None and abs(point[index]-upper)<1e-4:sign=-1.
                    elif lower is not None and abs(point[index]-lower)<1e-4:sign=1.
                    else:raise RuntimeError('AMBIGUOUS_FEASIBLE_DIRECTION')
                elif side not in ('central','plus'):raise RuntimeError('UNKNOWN_FEASIBLE_SIDE')
                v=check(float(sign*gradient[index]),c['slopes'],c['steps'],coordinate=c['coordinate'],side=side)
                stable=[]
                for b in c['branch_records']:
                    if 'first' in b:stable.append(b['first']['consistent'] and b['second']['consistent'])
                    else:stable.append(b['plus']['consistent'] and b['other']['consistent'])
                if len(stable)!=len(c['steps']):raise RuntimeError('BRANCH_COVERAGE_MISMATCH')
                pairs=[i for i in v.get('passing_adjacent_pairs',[]) if stable[i] and stable[i+1]]
                v.update(coordinate_index=index,passed=bool(pairs),branch_consistent_pairs=pairs,
                    previous_analytic=c['analytic'],analytic_change=v.get('analytic',float('nan'))-c['analytic'],preserved_FD_receipt=rel)
                cases.append(v)
        complete=sorted(seen)==list(range(len(current['names'])))
        coordinate_cases.append(dict(job=jid,passed=complete and all(c['passed'] for c in cases),complete=complete,checks=cases))
    initialrel='evidence/admissible_initial_feedback.json';initialrecord=load(initialrel);validate(initialrel,initialrecord)
    zero_support=load('evidence/supported_zero_initial_feedback.json');validate('evidence/supported_zero_initial_feedback.json',zero_support)
    if initialrecord and zero_support and initialrecord['zero_pool_proof_sha256']!=artifacts['evidence/supported_zero_initial_feedback.json']:
        raise RuntimeError('PRESERVED_INITIAL_SUPPORT_PROOF_CHANGED')
    if zero_support and not zero_support['passed']:errors.append('SUPPORTED_INITIAL_PROOF_FAILED')
    initialcases=[];kid='TIM_F23_K_V1_s0'
    if initialrecord and kid in currents:
        _,folder=currents[kid];initial=np.load(folder/'initial_point.npy',allow_pickle=False);adj=np.load(folder/'initial_cotangent.npy',allow_pickle=False)
        parent=ROOT/'inputs/land_support/outputs/luh3_reach'
        area,_=load_area_transfers(parent/'luh3_reach_state_1961_2024.csv',parent/'luh3_land_area_transfer_operator_1961_2023.csv',pd.date_range('1961-01-01','2024-12-31'))
        active=json.loads((ROOT/'config/conditional_reference.json').read_text(encoding='utf8'))['land1_integration']['active_land_slots']
        supported=np.zeros(initial.shape[:2],bool);supported[:,:12]=area[0]>0
        mask=np.zeros(13,bool);mask[active]=True;supported &= mask[None,:]
        byname={c['state']:c for c in initialrecord['cases']}
        if set(byname)!=set(('P','Sa','Sp','N_available','L')):raise RuntimeError('INITIAL_FAMILY_COVERAGE')
        for slot,name in enumerate(('P','Sa','Sp','N_available','L')):
            c=byname[name];previous=c['result'];direction=np.zeros_like(initial)
            if slot<3:direction[:,:,slot]=initial[:,:,slot]
            else:
                if np.any(initial[:,:,slot]!=0):raise RuntimeError('ZERO_INITIAL_CHANGED')
                direction[:,:,slot]=supported
            count=int(np.count_nonzero(direction));expected=c.get('direction_nonzero_cells',c.get('supported_cells'))
            if count!=expected:raise RuntimeError('INITIAL_SUPPORT_CHANGED')
            analytic=float(np.sum(adj*direction));v=check(analytic,previous['slopes'],previous['steps'],coordinate=name,side=previous['side'])
            stable=[b['plus']['consistent'] and b['other']['consistent'] for b in previous['branch_records']]
            pairs=[i for i in v.get('passing_adjacent_pairs',[]) if stable[i] and stable[i+1]]
            v.update(passed=bool(pairs),branch_consistent_pairs=pairs,previous_analytic=previous['analytic'],
                analytic_change=analytic-previous['analytic'],direction_nonzero_cells=count)
            initialcases.append(v)
    # Exact forward-only receipts reused under this specific repair. Derivative
    # receipts above are superseded by current analytic checks, never relabelled.
    forward_receipts=['evidence/full_prediction_isolation.json','evidence/full_uptake_support.json',
        'outputs/jobs/diagnostic_TIM_F23_K_V1_s0/physical_ledger.json']
    forward_receipts += ['outputs/imported_workstation_diagnostics/outputs/jobs/diagnostic_'+j+'/physical_ledger.json' for j in ('TIM_F23_C0_V0_s0','TIM_F23_K_V1_s0')]
    for rel in forward_receipts:
        r=load(rel);validate(rel,r)
        if r and not r.get('passed'):errors.append('FORWARD_PROOF_FAILED '+rel)
    passed=not pending and not errors and len(coordinate_cases)==2 and all(c['passed'] for c in coordinate_cases) and len(initialcases)==5 and all(c['passed'] for c in initialcases)
    receipt=dict(passed=passed,scope='current_full1961_2024_adjoints_requalified_against_preserved_bitwise_identical_forward_FD_not_fit',
        old_source_sha256=oldhash,new_source_sha256=newhash,complete_forward_AST_continuity=True,
        coordinate_cases=coordinate_cases,initial_cases=initialcases,pending=pending,errors=errors,
        artifact_sha256=artifacts,forward_only_reused_receipts=forward_receipts,
        old_receipts_overwritten=False,all_current_gradients_recomputed=True,
        initial_scope='five_physically_supported_family_directions_not_every_cell',
        formal_gate_issued=False,source_multiplier=1.)
    write_json(ROOT/'evidence/full_transition_derivative_requalification.json',receipt)
    print(json.dumps(dict(passed=passed,pending=pending,errors=errors,coordinates=[dict(job=c['job'],covered=len(c['checks']),passed=c['passed']) for c in coordinate_cases],
        initial=[dict(state=c['coordinate'],passed=c['passed'],analytic_change=c['analytic_change']) for c in initialcases])))
    return 0 if passed else 2


if __name__=='__main__':sys.exit(main())
