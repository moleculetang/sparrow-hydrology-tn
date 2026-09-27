"""Register only complete smooth-side refinements of failed v6 coordinates."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from d29_platform.runtime import sha, write_json
from d29_training.derivative_gate import refinement_passes

scan = json.loads((ROOT / "outputs/land1_full_history_gradient_v6.json").read_text(encoding="utf-8"))
reviews = []
for row in scan["coordinates"]:
    if row["passed"]:
        continue
    coordinate = row["index"]
    path = ROOT / f"outputs/land1_coordinate_v6_{coordinate}_branch_refinement.json"
    if not path.exists():
        raise RuntimeError(f"REFINEMENT_MISSING:{coordinate}")
    diagnostic = json.loads(path.read_text(encoding="utf-8"))
    if not refinement_passes(scan, coordinate, diagnostic):
        raise RuntimeError(f"REFINEMENT_NOT_ACCEPTED:{coordinate}")
    reviews.append(dict(coordinate=coordinate, path=str(path), sha256=sha(path),
                        accepted_as_local_smooth_derivative_evidence=True,
                        interpretation="Original coarse-grid failure retained; two adjacent finer steps match the analytic gradient without branch changes. This does not establish optimizer convergence."))
write_json(ROOT / "outputs/land1_gradient_v6_refinement_reviews.json",
           dict(reviews=reviews, original_failed_scan_unchanged=True))
print(json.dumps({"accepted_coordinates": [r["coordinate"] for r in reviews]}))
