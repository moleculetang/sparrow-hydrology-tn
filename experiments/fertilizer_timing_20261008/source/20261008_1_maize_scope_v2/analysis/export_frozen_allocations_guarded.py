"""Resource-bound wrapper around the sealed, prediction-frozen exporter."""
import runpy
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]

if __name__=='__main__':
    from job_lease import require
    jid=sys.argv[1]
    require(jid,'allocation_export')
    target=ROOT/'scripts/export_timing_allocations.py'
    sys.argv=[str(target),jid]
    runpy.run_path(str(target),run_name='__main__')
