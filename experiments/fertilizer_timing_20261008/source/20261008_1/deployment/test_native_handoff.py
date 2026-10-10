"""Linux dispatcher-only signal control with its own sleeping toy child."""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def main(root):
    root=Path(root).resolve();sys.path[:0]=[str(root),str(root/'scripts'),str(root/'deployment')]
    from host_resources import registry,registry_root
    from d29_platform.runtime import write_json,sha
    from handoff_extended_controller import process_stopped
    from controller_lock import acquire
    if os.name!='posix':raise RuntimeError('LINUX_NATIVE_CONTROL_ONLY')
    assert process_stopped('State:\tT (stopped)') and process_stopped('State:\tt (tracing stop)')
    assert not process_stopped('State:\tS (sleeping)') and not process_stopped('State:\tR (running)')
    folder=root/'outputs/handoff_native_controls_v1';folder.mkdir(exist_ok=True);lock=acquire(folder/'lifetime.lock')
    if (folder/'complete.json').exists():raise RuntimeError('CONTROL_ALREADY_COMPLETED_NO_RESTART')
    ownerfile=folder/'toy_owner.json';parent=None;child=None
    code="""import json,pathlib,subprocess,sys,time
sys.path.insert(0,sys.argv[1]);from host_resources import registry
p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(300)'])
pathlib.Path(sys.argv[2]).write_text(json.dumps(dict(parent_pid=__import__('os').getpid(),parent_created=registry.identity(__import__('os').getpid()),child_pid=p.pid,child_created=registry.identity(p.pid))))
time.sleep(300)
"""
    try:
        with (folder/'toy.log').open('ab') as log:
            parent=subprocess.Popen([sys.executable,'-c',code,str(root/'scripts'),str(ownerfile)],stdout=log,stderr=subprocess.STDOUT)
            for _ in range(100):
                if ownerfile.exists():break
                if parent.poll() is not None:raise RuntimeError('TOY_PARENT_FAILED')
                time.sleep(.02)
            toy=json.loads(ownerfile.read_text());child=toy
            if registry.identity(parent.pid)!=toy['parent_created']:raise RuntimeError('TOY_PARENT_IDENTITY_MISMATCH')
            with registry.transaction(registry_root()):
                os.kill(parent.pid,signal.SIGSTOP)
                for _ in range(100):
                    if process_stopped((Path('/proc')/str(parent.pid)/'status').read_text()):break
                    time.sleep(.01)
                else:raise RuntimeError('TOY_PARENT_NOT_STOPPED')
                assert registry.identity(toy['child_pid'])==toy['child_created']
                assert not process_stopped((Path('/proc')/str(toy['child_pid'])/'status').read_text())
            os.kill(parent.pid,signal.SIGTERM);os.kill(parent.pid,signal.SIGCONT);parent.wait(timeout=5)
            time.sleep(.1)
            preserved=registry.identity(toy['child_pid'])==toy['child_created']
            assert preserved and not process_stopped((Path('/proc')/str(toy['child_pid'])/'status').read_text())
        result=dict(passed=True,control='registry_lock_then_dispatcher_only_stop_and_exit',
            sleeping_toy_child_native_identity_preserved=True,signals_sent_to_scientific_processes=0,
            controls_scope='native_process_semantics_not_model_convergence',toy=toy,
            source_sha256=sha(root/'deployment/handoff_extended_controller.py'))
        write_json(folder/'complete.json',result);write_json(root/'evidence/handoff_native_controls.json',result)
        print(json.dumps(result));return 0
    finally:
        if parent is not None and parent.poll() is None:
            os.kill(parent.pid,signal.SIGCONT);parent.terminate();parent.wait(timeout=5)
        if child is not None and registry.identity(child['child_pid'])==child['child_created']:
            os.kill(child['child_pid'],signal.SIGTERM)


if __name__=='__main__':sys.exit(main(sys.argv[1]))
