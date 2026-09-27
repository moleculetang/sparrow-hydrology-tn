"""Verify the finite LAND1 queue's Windows process-memory reservation helper."""
import ast
import ctypes
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from d29_platform.runtime import write_json


def main():
    source = ROOT / 'scripts/run_land1_queue.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    wanted = {'MAX_WORKERS', 'PEAK_RESERVE_BYTES'}
    selected = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in wanted
                                                 for t in node.targets):
            selected.append(node)
        elif isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in {
                '_ProcessMemoryCountersEx', '_private_bytes', '_projected_reserve'}:
            selected.append(node)
    namespace = {'ctypes': ctypes, 'children': {}}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), 'exec'), namespace)
    process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(10)'],
                               creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        namespace['children']['synthetic'] = (process, None)
        used = namespace['_private_bytes'](process)
        reserved, active = namespace['_projected_reserve']()
        expected = 9_000_000_000 + max(0, 9_000_000_000 - used)
        passed_test = (namespace['MAX_WORKERS'] == 6 and used > 0 and
                       active == {'synthetic': used} and reserved == expected)
        receipt = dict(passed=passed_test, maximum_workers=namespace['MAX_WORKERS'],
                       active_private_bytes=used, projected_reserve_bytes=reserved,
                       expected_reserve_bytes=expected,
                       role='resource helper verification; no model fit or TN evaluation')
        write_json(ROOT / 'outputs/queue_6_resource_validation.json', receipt)
        if not passed_test:
            raise RuntimeError('QUEUE_RESOURCE_PROJECTION_FAILED')
    finally:
        process.terminate()
        process.wait()


if __name__ == '__main__':
    main()
