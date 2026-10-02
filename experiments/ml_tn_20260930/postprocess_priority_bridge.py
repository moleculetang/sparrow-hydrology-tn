"""One-shot priority for the existing postprocess lease, after training completes.

No new computation is dispatched and no existing process receives a signal.
Never overwrite the training controller's priority while it is still active.
"""
import os
import time
from pathlib import Path
from mltn.common import ROOT, read, write
from mltn.resources import registry


def main():
    assert os.name == 'posix'
    mod, root = registry()
    assert mod is not None
    while not (ROOT/'outputs/all_training_done.json').exists():
        time.sleep(10)
    owner_file = ROOT/'outputs/postprocess_owner.json'
    while not owner_file.exists():
        assert not (ROOT/'outputs/postprocess_done.json').exists()
        time.sleep(1)
    pid = int(read(owner_file)['pid'])
    created = mod.identity(pid)
    assert created is not None, 'POSTPROCESS_OWNER_NOT_LIVE'
    assert Path(f'/proc/{pid}/cwd').resolve() == ROOT.resolve()
    args = [x.decode() for x in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0') if x]
    scripts = [Path(x) for x in args if x.endswith('.py')]
    assert len(scripts) == 1 and (ROOT/scripts[0]).resolve() == (ROOT/'postprocess.py').resolve(), args
    claimed = False
    try:
        while True:
            if mod.identity(pid) != created:
                assert (ROOT/'outputs/postprocess_done.json').exists(), 'POSTPROCESS_STOPPED_BEFORE_SUCCESS'
                reason = 'already completed'
                break
            leased = [v for v in mod.status(root)['leases'].values()
                      if v['project'] == '20260930_1' and v['job'] == 'completion_postprocess'
                      and v['pid'] == pid]
            if leased:
                reason = 'existing postprocess acquired its lease'
                break
            # Its owner marker is written immediately before the existing lease
            # request. This is one real ready CPU+GPU request, not a future queue.
            mod.pending('20260930_1', 1, root)
            claimed = True
            write(ROOT/'outputs/postprocess_priority_bridge.json', dict(
                pid=pid, created=created, priority_count=1, training_complete=True,
                future_jobs_counted=False, signals_sent=False))
            time.sleep(5)
    finally:
        if claimed:
            mod.pending('20260930_1', 0, root)
    write(ROOT/'evidence/postprocess_priority_bridge.json', dict(
        passed=True, pid=pid, created=created, claimed_immediate_priority=claimed,
        result=reason, scientific_changes=False, signals_sent=False,
        scope='one existing completion postprocess request only; no training priority changes'))


if __name__ == '__main__':
    main()
