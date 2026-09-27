"""Request three running audit workers to stop after their assigned prefix.

The workers themselves save at their next safe evaluation boundary. This only
creates their own registered stop flags; it does not signal unrelated processes.
"""
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
targets = [
    ("land1_full_history_gradient_v6.json", 6, "stop_land1_gradient_v6.flag"),
    ("land1_gradient_v6_shard_11_17.json", 3, "stop_land1_gradient_v6_shard_11_17.flag"),
    ("land1_gradient_v6_shard_17_23.json", 3, "stop_land1_gradient_v6_shard_17_23.flag"),
]
done = set()
while len(done) < len(targets):
    for name, count, flag_name in targets:
        if name in done:
            continue
        path = ROOT / "outputs" / name
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if len(data.get("coordinates", [])) >= count:
                (ROOT / "work" / flag_name).write_text("partition complete\n", encoding="utf-8")
                print(f"requested_safe_stop {name} {count}", flush=True)
                done.add(name)
    time.sleep(10)
