"""Refine coordinate 6 from its matching full-history partition receipt."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
source = (ROOT / "scripts/diagnose_precision_coordinate.py").read_text(encoding="utf-8")
source = source.replace("land1_full_history_gradient_v3.json",
                        "land1_gradient_v6_shard_06_11.json")
source = source.replace("scan['coordinates'][coordinate]['checks']",
                        "next(r for r in scan['coordinates'] if r['index']==coordinate)['checks']")
source = source.replace("land1_coordinate_{coordinate}_branch_refinement.json",
                        "land1_coordinate_v6_{coordinate}_branch_refinement.json")
exec(compile(source, str(Path(__file__)), "exec"))
