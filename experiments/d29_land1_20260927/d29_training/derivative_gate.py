"""Strict derivative acceptance with preserved, branch-recorded refinement evidence."""
import math


def refinement_passes(scan, coordinate, diagnostic):
    """Refinement is additional evidence, never a minimum-error substitution.

    Require a complete fixed finer grid, two adjacent matches and no recorded
    physical branch change for either sign at those two steps.
    """
    if diagnostic.get('identity') != scan['identity']:
        return False
    if diagnostic.get('coordinate') != coordinate:
        return False
    if abs(diagnostic.get('base_value',float('inf'))-scan['objective'])>1e-8*(1+abs(scan['objective'])):
        return False
    checks=diagnostic.get('checks',[])
    if [r.get('step') for r in checks] != [1e-6,3e-7,1e-7]:
        return False
    expected=scan['analytic_gradient'][coordinate]
    if diagnostic.get('analytic') != expected:return False
    valid=[]
    for r in checks:
        plus=r.get('plus',{});minus=r.get('minus',{})
        if not all(math.isfinite(float(x)) for x in [plus.get('value',float('nan')),minus.get('value',float('nan'))]):
            return False
        fd=(plus['value']-minus['value'])/(2*r['step'])
        valid.append(abs(fd-expected)<=1e-6*(1+abs(expected)) and
                     plus.get('branch_switch_count')==0 and minus.get('branch_switch_count')==0 and
                     plus.get('switches')==[] and minus.get('switches')==[])
    return any(valid[i] and valid[i+1] for i in range(len(valid)-1))
