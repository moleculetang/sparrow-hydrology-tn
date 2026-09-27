"""Three frozen annual states, candidate only; not a full-history gate."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
source=(ROOT/'scripts/diagnose_land1_precision.py').read_text(encoding='utf-8')
source=source.replace('from d29_platform.land1 import run_land1,hazard_to_probability','from d29_platform.precision_candidate import run_land1,hazard_to_probability')
source=source.replace('outputs/land1_precision_diagnosis.json','outputs/precision_candidate_diagnosis.json')
exec(compile(source,str(ROOT/'scripts/diagnose_land1_precision.py'),'exec'))
