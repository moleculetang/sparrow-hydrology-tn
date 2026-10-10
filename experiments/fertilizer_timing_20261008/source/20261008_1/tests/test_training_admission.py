"""Absence of the formal prerequisite seal cannot initiate a training call."""
import sys,json,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('timing_training_entry',ROOT/'scripts/train_job.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
failed=False
try:m.validate_admission()
except RuntimeError as e:
    assert 'PREREQUISITES_NOT_SEALED' in str(e);failed=True
assert failed and not (ROOT/'evidence/formal_gate.json').exists()
r=dict(passed=True,scope='training_entry_fail_closed_without_actual_model_oracle',training_calls=0,
       incomplete_prerequisite_gate_blocks_training=True,workstation_hold_retained=True)
(ROOT/'evidence/training_admission_block.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
