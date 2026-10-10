"""Resolve an additive, verified same-contract dispatcher replacement.

Keep the original pre-approved legacy observer identity for historical checks;
the current scientific owner comes only from an actual native handoff receipt.
No generation is inferred from a newer PID or a status-file timestamp.
"""
import copy
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def project(base, transition, original_sha256, ownership_sha256):
    if base.get('passed') is not True or base.get('scientific_signals')!=0:
        raise RuntimeError('ACTUAL_ORIGINAL_APPROVED_HANDOFF_REQUIRED')
    scope=base.get('scope',{});epochs=scope.get('epochs',{});gates=scope.get('gates',{})
    if (len(epochs)!=44 or set(epochs)!=set(gates)
            or list(epochs.values()).count('original_nonK')!=30
            or list(epochs.values()).count('approved_maize_K')!=14
            or any(not isinstance(v,str) or not v for v in gates.values())):
        raise RuntimeError('EXACT44_TWO_GATE_SCOPE_REQUIRED')
    if transition is None:return copy.deepcopy(base)
    if (transition.get('passed') is not True or transition.get('version')!=1
            or transition.get('old_owner')!=base['new_owner']
            or transition.get('scope')!=base['scope']
            or transition.get('original_handoff_sha256')!=original_sha256
            or transition.get('ownership_sha256')!=ownership_sha256
            or transition.get('scientific_signals')!=0
            or transition.get('science_gate_bytes_unchanged') is not True
            or transition.get('native_adoption_passed') is not True
            or transition.get('startup_reviews_cleared')!=['TIM_F23_K_V0_s0','TIM_F24_K_V0_s0']):
        raise RuntimeError('ACTUAL_SAME_CONTRACT_NATIVE_PARENT_TRANSITION_REQUIRED')
    owner=transition.get('new_owner',{})
    if (owner==base['new_owner'] or type(owner.get('pid')) is not int or owner['pid']<=0
            or not owner.get('created') or owner.get('host')!='pc' or owner.get('role')!='workstation'):
        raise RuntimeError('EXACT_NEW_NATIVE_WORKSTATION_OWNER_REQUIRED')
    result=copy.deepcopy(base)
    result.update(new_owner=owner,latest_parent_transition=copy.deepcopy(transition),
                  original_legacy_observer_parent_preserved=True)
    return result


def current(root):
    root=Path(root)
    original=root/'outputs/approved_scope_operations_transport/actual_remote_handoff_return.json'
    replacement=root/'outputs/approved_scope_operations_transport/startup_recovery_handoff_v1/complete.json'
    base=json.loads(original.read_text(encoding='utf8'))
    transition=json.loads(replacement.read_text(encoding='utf8')) if replacement.exists() else None
    return project(base,transition,digest(original),digest(root/'deployment/path_ownership.json'))
