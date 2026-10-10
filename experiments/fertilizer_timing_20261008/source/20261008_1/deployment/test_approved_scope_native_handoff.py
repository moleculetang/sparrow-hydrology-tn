"""Tiny Linux parent/child fixture; no fit, no scientific process signals."""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def main(root):
    root = Path(root).resolve()
    sys.path[:0] = [str(root), str(root/'scripts'), str(root/'deployment')]
    from host_resources import registry, registry_root
    from d29_platform.runtime import write_json, sha
    from controller_lock import acquire
    from handoff_approved_scope import stopped
    if os.name != 'posix':
        raise RuntimeError('NATIVE_LINUX_CONTROL_ONLY')
    folder = root/'outputs/approved_scope_native_controls_v1'
    folder.mkdir(exist_ok=True)
    lock = acquire(folder/'lifetime.lock')
    if (folder/'owner.json').exists():
        raise RuntimeError('NATIVE_FIXTURE_ALREADY_STARTED_REVIEW')
    write_json(folder/'owner.json', dict(pid=os.getpid(), created=registry.identity(os.getpid())))
    toyfile = folder/'toy.json'
    code = '''import json,os,pathlib,subprocess,sys,time
sys.path.insert(0,sys.argv[1]);from host_resources import registry
p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(120)'])
pathlib.Path(sys.argv[2]).write_text(json.dumps(dict(parent=dict(pid=os.getpid(),created=registry.identity(os.getpid())),child=dict(pid=p.pid,created=registry.identity(p.pid)))))
time.sleep(120)
'''
    parent = None
    toy = None
    try:
        parent = subprocess.Popen([sys.executable, '-B', '-c', code, str(root/'scripts'), str(toyfile)])
        for _ in range(100):
            if toyfile.exists():
                break
            if parent.poll() is not None:
                raise RuntimeError('TOY_PARENT_EARLY_EXIT')
            time.sleep(.03)
        toy = json.loads(toyfile.read_text())
        if registry.identity(parent.pid) != toy['parent']['created']:
            raise RuntimeError('TOY_PARENT_CHANGED')
        with registry.transaction(registry_root()):
            os.kill(parent.pid, signal.SIGSTOP)
            for _ in range(100):
                if stopped(parent.pid):
                    break
                time.sleep(.01)
            else:
                raise RuntimeError('TOY_PARENT_STOP_NOT_ACKNOWLEDGED')
            assert registry.identity(toy['child']['pid']) == toy['child']['created']
            assert not stopped(toy['child']['pid'])
        os.kill(parent.pid, signal.SIGTERM)
        os.kill(parent.pid, signal.SIGCONT)
        parent.wait(timeout=5)
        assert registry.identity(toy['child']['pid']) == toy['child']['created']
        assert not stopped(toy['child']['pid'])
        result = dict(passed=True, toy=toy, registry_lock_precedes_stop=True,
                      child_identity_preserved=True, scientific_signals=0,
                      fixture_sha256=sha(Path(__file__)),
                      handoff_sha256=sha(root/'deployment/handoff_approved_scope.py'),
                      scope='native_parent_only_signal_semantics_not_scientific_adoption')
        write_json(folder/'complete.json', result)
        write_json(root/'evidence/approved_scope_native_controls.json', result)
        print(json.dumps(result))
    finally:
        if parent is not None and parent.poll() is None:
            os.kill(parent.pid, signal.SIGCONT)
            parent.terminate()
            parent.wait(timeout=5)
        if toy and registry.identity(toy['child']['pid']) == toy['child']['created']:
            os.kill(toy['child']['pid'], signal.SIGTERM)


if __name__ == '__main__':
    main(sys.argv[1])
