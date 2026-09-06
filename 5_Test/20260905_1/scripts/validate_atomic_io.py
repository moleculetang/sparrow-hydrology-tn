"""Check atomic persistence under injected transient and persistent locks."""
import common
from common import ROOT,RUNTIME,atomic_json,sha256,utc_now
import json


def main():
    directory=ROOT/'5_Test/20260905_1/work/atomic_io_validation'
    target=directory/'probe.json';atomic_json({'version':1},target)
    replace=common.os.replace;attempts=[]
    def intermittent(src,dst):
        attempts.append(1)
        if len(attempts)<=2:raise PermissionError(13,'injected transient sharing lock',str(src),32)
        return replace(src,dst)
    try:
        common.os.replace=intermittent
        atomic_json({'version':2},target)
    finally:common.os.replace=replace
    assert len(attempts)==3 and json.loads(target.read_text())=={'version':2}
    count=[]
    def persistent(src,dst):
        count.append(1)
        raise PermissionError(13,'injected persistent access lock',str(src),5)
    propagated=False
    try:
        common.os.replace=persistent
        try:atomic_json({'version':3},target)
        except PermissionError:propagated=True
    finally:common.os.replace=replace
    assert propagated and len(count)==10 and json.loads(target.read_text())=={'version':2}
    atomic_json({'version':4},target)
    assert json.loads(target.read_text())=={'version':4}
    atomic_json(dict(status='PASS_ATOMIC_IO_RETRY',runtime=RUNTIME,created_utc=utc_now(),
        transient_attempts=len(attempts),persistent_attempts=len(count),old_target_preserved_on_failure=True,
        no_permission_changes=True,common_sha256=sha256(common.__file__),
        change_scope='atomic persistence retry only; objective and process code are unchanged'),ROOT/'5_Test/20260905_1/reports/atomic_io_validation.json')
    print('ATOMIC_IO_VALIDATED',len(attempts),len(count),flush=True)


if __name__=='__main__':main()
