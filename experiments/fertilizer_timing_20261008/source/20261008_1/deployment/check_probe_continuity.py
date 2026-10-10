"""Preserve the qualified probe snapshot; compare the functions actually used."""
import ast
import hashlib
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from d29_platform.runtime import sha,write_json


def functions(path):
    tree=ast.parse(path.read_text(encoding='utf8'));result={}
    for node in tree.body:
        if isinstance(node,ast.FunctionDef) and node.name in ('probe','branch_signature'):
            result[node.name]=hashlib.sha256(ast.dump(node,include_attributes=False).encode('utf8')).hexdigest()
    return result


def main():
    prior=ROOT/'evidence/source_snapshots/audit_solution_coordinate_qualification.py'
    current=ROOT/'scripts/audit_solution.py'
    old_hash=sha(prior)
    if old_hash!='50943c26921fd0dee6ec0a345dafe3bb8db0eb1f7c6f6b1ab9d06c2cc0789f5c':
        raise RuntimeError('WRONG_ORIGINAL_COORDINATE_PROBE_SNAPSHOT')
    old_functions=functions(prior);current_functions=functions(current)
    same=old_functions==current_functions and set(old_functions)=={'probe','branch_signature'}
    receipt=dict(passed=same,scope='coordinate_probe_functions_unchanged_after_fit_stopping_criterion_repair',
        old_script_sha256=old_hash,current_script_sha256=sha(current),
        preserved_snapshot='evidence/source_snapshots/audit_solution_coordinate_qualification.py',
        old_functions=old_functions,current_functions=current_functions,
        allowed_scope='full1961_2024_K_V1_coordinate_subset_not_fit',
        changed_responsibility='fit_point_projected_gradient_and_stage_lease_admission_not_coordinate_probe',
        original_qualification_receipts_rewritten=False,formal_gate_issued=False)
    write_json(ROOT/'evidence/coordinate_probe_implementation_continuity.json',receipt)
    print(json.dumps(receipt));return 0 if same else 2


if __name__=='__main__':sys.exit(main())
