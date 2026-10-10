"""Qualification for a reviewed same-point continuation, never convergence.

This module neither changes checkpoints nor starts a solver. A terminal point
needs full coordinate coverage, the existing objective/mass/recovery gates,
and an actual feasible descent witness before a new canonical solver is
considered. Repeating an unchanged failed point is explicitly rejected.
"""
import math


def finite(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)


def descent_witness(witness, point_sha256, gate_sha256, objective):
    if not isinstance(witness,dict):return False
    if witness.get('parameter_sha256')!=point_sha256 or witness.get('formal_gate_sha256')!=gate_sha256:
        return False
    if witness.get('source_multiplier')!=1. or witness.get('evaluation_scores_used') is not False:return False
    if witness.get('base_objective')!=objective:return False
    direction=witness.get('direction',[])
    if not direction or not all(finite(x) for x in direction) or not any(x!=0 for x in direction):return False
    pairs=witness.get('probes',[])
    if len(pairs)<2:return False
    for first,second in zip(pairs,pairs[1:]):
        if any(not all(finite(p.get(k)) for k in ('step','slope','objective','analytic')) for p in (first,second)):
            continue
        if not (0<second['step']<first['step']):continue
        if any(p.get('feasible') is not True or p.get('branch_consistent') is not True for p in (first,second)):
            continue
        signatures=[p.get('branch_signature') for p in (first,second)]
        if not signatures[0] or signatures[0]!=signatures[1]:continue
        if first['analytic']!=second['analytic']:continue
        tolerance=1e-6*(1+abs(first['analytic']))
        if any(abs(p['slope']-p['analytic'])>tolerance or p['slope']>=-1e-5 for p in (first,second)):
            continue
        if abs(first['slope']-second['slope'])>2*tolerance:continue
        if any(p['objective']>=objective-32*math.ulp(objective) for p in (first,second)):continue
        return True
    return False


def review(audit, contract, witness, previous_attempts=()):
    """Return explicit reasons; a successful result only qualifies re-entry."""
    reasons=[]
    names=contract.get('names',[])
    if not names or len(names)!=len(set(names)):reasons.append('invalid_coordinate_contract')
    if contract.get('source_multiplier')!=1. or audit.get('source_multiplier')!=1.:
        reasons.append('source_multiplier_not_one')
    if contract.get('formal_gate_sha256')!=audit.get('formal_gate_sha256'):
        reasons.append('gate_identity_mismatch')
    if not audit.get('parameter_sha256'):reasons.append('missing_exact_point_identity')
    if audit.get('full_history')!=[1961,2024] or audit.get('reaches')!=230:
        reasons.append('incomplete_physical_scope')
    for flag in ('objective_passed','physical_passed','gradient_recovery_passed','derivative_passed'):
        if audit.get(flag) is not True:reasons.append(flag+'_not_passed')
    checks=audit.get('checks',[])
    if {c.get('coordinate') for c in checks}!=set(names) or len(checks)!=len(names):
        reasons.append('incomplete_or_duplicate_coordinate_coverage')
    if any(c.get('derivative_qualified') is not True for c in checks):reasons.append('unqualified_derivative')
    for name in ('objective','projected_gradient','scaled_projected_gradient'):
        if not finite(audit.get(name)):reasons.append('invalid_'+name)
    if any(finite(audit.get(name)) and audit[name]<0 for name in ('projected_gradient','scaled_projected_gradient')):
        reasons.append('negative_gradient_norm')
    if audit.get('numerically_sufficient') is True:reasons.append('already_sufficient_do_not_retrain')
    if finite(audit.get('objective')) and not descent_witness(witness,audit.get('parameter_sha256'),
            audit.get('formal_gate_sha256'),audit['objective']):
        reasons.append('qualified_actual_feasible_descent_required')
    if witness and len(witness.get('direction',[]))!=len(names):reasons.append('descent_direction_coordinate_mismatch')
    if any(p.get('parameter_sha256')==audit.get('parameter_sha256') and p.get('formal_gate_sha256')==audit.get('formal_gate_sha256')
           and (p.get('continuation_started') is True or p.get('continuation_prepared') is True) for p in previous_attempts):
        reasons.append('same_failed_point_already_reentered_requires_new_diagnosis')
    return dict(qualified_for_reviewed_same_point_reentry=not reasons,reasons=reasons,
                numerically_sufficient=False,automatic_solver_restart=False,
                checkpoint_mutated=False,evaluation_scores_used=False)
