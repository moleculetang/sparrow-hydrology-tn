from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

from runtime_guard import assert_sparrow_runtime


RUNTIME = assert_sparrow_runtime()
RUN = Path(__file__).resolve().parents[1]
STEPS = [
    ("build_local_input", "build_q72_baseline_input.py"),
    ("run_three_blocked_folds", "run_q72_baseline.py"),
    ("build_baseline_audit", "build_q72_baseline_audit.py"),
    ("independent_validation", "validate_q72_baseline.py"),
]


def main() -> None:
    logs = RUN / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, script_name in STEPS:
        script = RUN / "scripts" / script_name
        started = perf_counter()
        completed = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(RUN),
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            check=False,
        )
        elapsed = perf_counter() - started
        (logs / f"{name}.stdout.log").write_text(
            completed.stdout, encoding="utf-8"
        )
        (logs / f"{name}.stderr.log").write_text(
            completed.stderr, encoding="utf-8"
        )
        rows.append(
            {
                "step": name,
                "script": str(script.relative_to(RUN)),
                "returncode": completed.returncode,
                "elapsed_seconds": elapsed,
            }
        )
        if completed.returncode != 0:
            (logs / "run_all_manifest.json").write_text(
                json.dumps(
                    {"runtime": RUNTIME, "steps": rows},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            raise RuntimeError(
                f"{name} failed; inspect logs/{name}.stderr.log"
            )
    manifest = {"runtime": RUNTIME, "steps": rows, "completed": True}
    (logs / "run_all_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
