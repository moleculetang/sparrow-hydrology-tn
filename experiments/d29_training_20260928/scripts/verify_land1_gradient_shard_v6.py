"""Independent coordinate partition of the frozen v6 derivative audit.

Each shard evaluates a complete analytic reference at the same frozen point.
The consolidation tool must check reference and implementation identity before
accepting rows; this script does not write the authoritative v6 receipt.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
start, end = map(int, sys.argv[1:3])
if not (0 <= start < end <= 23):
    raise ValueError("INVALID_SHARD_RANGE")
tag = f"{start:02d}_{end:02d}"
s = (ROOT / "scripts/verify_land1_gradient.py").read_text(encoding="utf-8")
s = s.replace("from d29_training.land1_adapter import Land1Training",
'''import d29_training.annual_chain as annual_module
from d29_platform import precision_candidate as precise
annual_module.run_land1=precise.run_land1
annual_module.land1_adjoint=precise.land1_adjoint
from d29_training.land1_adapter import Land1Training''')
s = s.replace("land1_full_history_gradient_v2.json", f"land1_gradient_v6_shard_{tag}.json")
s = s.replace("land1_gradient_worker.json", f"land1_gradient_v6_shard_{tag}_worker.json")
s = s.replace("land1_gradient_evaluations.jsonl", f"land1_gradient_v6_shard_{tag}_evaluations.jsonl")
s = s.replace("stop_land1_gradient.flag", f"stop_land1_gradient_v6_shard_{tag}.flag")
s = s.replace("[ROOT/'d29_platform/land1.py',", "[ROOT/'d29_platform/precision_candidate.py',ROOT/'d29_platform/precision_transfer.py',ROOT/'d29_platform/land1.py',")
s = s.replace("[1e-4,3e-5,1e-5]", "[1e-5,3e-6,1e-6]")
s = s.replace("value,_=a.value_gradient(point)", "value,_=a.value_gradient(point,forward_only=True)")
s = s.replace("ROOT/'d29_training/objective.py']",
              "ROOT/'d29_training/objective.py',ROOT/'d29_platform/coupling.py',"
              "ROOT/'d29_platform/conditional_inputs.py',ROOT/'config/conditional_reference.json',"
              "ROOT/'config/mineralization_reference.json']")
s = s.replace("for i in range(len(rows),len(x)):", f"for i in range({start},{end}):")
assert f"for i in range({start},{end}):" in s
exec(compile(s, str(Path(__file__)), "exec"))
