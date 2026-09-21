"""Actual scoring functions; runtime logging is replaced, never the arithmetic."""
import sys,types,tempfile,runpy
from pathlib import Path
P=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(P/'snapshot/20260920_6/scripts'))
with tempfile.TemporaryDirectory() as tmp:
    runtime=types.ModuleType('native_runtime');runtime.RUN=Path(tmp)
    runtime.write=lambda path,value:None
    sys.modules['native_runtime']=runtime
    runpy.run_path(str(P/'snapshot/20260920_6/scripts/test_evaluation.py'),run_name='__main__')
