"""Compile the exact interpolated SSH programs without executing operations."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def template(path, name, env):
    tree = ast.parse(path.read_text(encoding='utf8'))
    expressions = [node.value for node in ast.walk(tree) if isinstance(node, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)]
    if len(expressions) != 1:
        raise AssertionError('UNIQUE_REMOTE_TEMPLATE_REQUIRED')
    value = eval(compile(ast.Expression(expressions[0]), str(path), 'eval'), {}, env)
    parsed = ast.parse(value)
    compile(parsed, str(path)+'::remote', 'exec')
    return parsed


def main():
    handoff = template(ROOT/'deployment/run_verified_scope_handoff.py', 'code',
                       dict(REMOTE_ROOT='/fixture', stage=dict(files={}), intent={}, allow_launch=False))
    separators = [node.args[0].value for node in ast.walk(handoff) if isinstance(node, ast.Call)
                  and isinstance(node.func, ast.Attribute) and node.func.attr == 'split'
                  and node.args and isinstance(node.args[0], ast.Constant)]
    assert b'\0' in separators, 'native /proc command must split on NUL'
    template(ROOT/'deployment/await_approved_campaign_return.py', 'code',
             dict(REMOTE_ROOT='/fixture', scope={}, remote_owner={}, ownership_hash='x', source='', owner_key='0'*16))
    template(ROOT/'deployment/await_approved_campaign_return.py', 'ack_code',
             dict(REMOTE_ROOT='/fixture', ack={}, owner_key='0'*16))
    result = dict(passed=True, exact_remote_templates_compiled=3, native_command_NUL_separator_correct=True,
                  source_sha256={name:hashlib.sha256((ROOT/'deployment'/name).read_bytes()).hexdigest()
                                 for name in ['run_verified_scope_handoff.py','await_approved_campaign_return.py']},
                  remote_operations_executed=False, scientific_signals=0)
    path = ROOT/'evidence/approved_scope_remote_template_controls_v2.json'
    if path.exists() and json.loads(path.read_text(encoding='utf8')) != result:
        raise RuntimeError('EXISTING_TEMPLATE_RECEIPT_CHANGED')
    path.write_text(json.dumps(result, indent=2), encoding='utf8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
