"""Branch-recorded refinement bound to the consolidated v6 numerical scan."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
source = (ROOT / "scripts/diagnose_precision_coordinate.py").read_text(encoding="utf-8")
source = source.replace("land1_full_history_gradient_v3.json",
                        "land1_full_history_gradient_v6.json")
source = source.replace("land1_coordinate_{coordinate}_branch_refinement.json",
                        "land1_coordinate_v6_{coordinate}_branch_refinement.json")
exec(compile(source, str(Path(__file__)), "exec"))
