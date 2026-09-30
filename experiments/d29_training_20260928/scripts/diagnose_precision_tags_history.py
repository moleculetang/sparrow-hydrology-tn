"""Independent candidate full-history label diagnostic, not a fitting worker."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
s=(ROOT/'scripts/diagnose_tagged_full_history.py').read_text(encoding='utf-8')
s=s.replace('d29_platform.land1','d29_platform.precision_candidate')
s=s.replace('d29_platform.tagged_candidate','d29_platform.precision_tags_candidate')
s=s.replace('d29_platform/land1.py','d29_platform/precision_candidate.py')
s=s.replace('d29_platform/tagged_candidate.py','d29_platform/precision_tags_candidate.py')
s=s.replace('outputs/tag_precision_candidate','outputs/represented_precision_candidate_v2')
if '--mixture' in sys.argv:
    s=s.replace('precision_tags_candidate','mixture_tags_candidate').replace('represented_precision_candidate_v2','mixture_precision_candidate')
s=s.replace("paths=[Path(__file__),", "paths=[Path(__file__),ROOT/'d29_platform/precision_transfer.py',")
# Retain rounded-total annual residuals as diagnostics; acceptance uses local
# represented-state and source balances with the unchanged absolute tolerance.
s=s.replace("'mass_residual_kg'", "'rounded_total_mass_residual_kg_not_acceptance'")
s=s.replace('existing full-history kernel validated; annual streaming here is forward only, no truncated-gradient training','forward/source audit only; candidate full-history derivative acceptance is separate and remains required')
exec(compile(s,str(ROOT/'scripts/diagnose_precision_tags_history.py'),'exec'))
