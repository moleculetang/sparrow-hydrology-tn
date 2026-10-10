"""Transient readers cannot create a terminal scientific event."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'deployment')]
from await_local_campaign_event import read_status
from d29_platform.runtime import write_json,sha


def main():
    answers=iter([PermissionError('race'),FileNotFoundError('replace'),'{' ,'{"phase":"running"}'])
    def reader(path):
        value=next(answers)
        if isinstance(value,Exception):raise value
        return value
    state,errors=read_status(None,reader,lambda t:None)
    assert state=={'phase':'running'} and errors==['PermissionError','FileNotFoundError','JSONDecodeError']
    def unavailable(path):raise PermissionError('race')
    state,errors=read_status(None,unavailable,lambda t:None)
    assert state is None and len(errors)==5
    write_json(ROOT/'evidence/local_event_reader_transient_controls.json',dict(passed=True,
        controls=['same_reader_recovers_permission_replace_and_partial_json','unavailable_is_not_terminal'],
        no_scientific_signal_or_redispatch=True,source_sha256=sha(ROOT/'deployment/await_local_campaign_event.py')))
    print('passed: transient observation failures never classify training as terminal')


if __name__=='__main__':main()
