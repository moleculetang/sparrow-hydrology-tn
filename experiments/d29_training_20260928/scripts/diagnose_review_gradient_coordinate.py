"""Branch-recorded fine-step check for a failed new full-history coordinate."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
coordinate=int(sys.argv[1])
source=(ROOT/'scripts/diagnose_precision_coordinate.py').read_text(encoding='utf-8')
tag='00_12' if coordinate<12 else '12_23'
source=source.replace('land1_full_history_gradient_v3.json',f'land1_gradient_v6_shard_{tag}.json')
source=source.replace("scan['coordinates'][coordinate]['checks']",
    "next(r for r in scan['coordinates'] if r['index']==coordinate)['checks']")
source=source.replace('land1_coordinate_{coordinate}_branch_refinement.json',
    'land1_review_coordinate_{coordinate}_branch_refinement.json')
exec(compile(source,str(Path(__file__)),'exec'))
