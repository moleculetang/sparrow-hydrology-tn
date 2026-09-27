"""Full coordinate check after roundoff and generic-chain interface repair."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
source = (ROOT / 'scripts/verify_land1_precision_gradient.py').read_text(encoding='utf-8')
source = source.replace('land1_full_history_gradient_v3.json',
                        'land1_full_history_gradient_v5.json')
source = source.replace('land1_precision_gradient_worker.json',
                        'land1_precision_gradient_v5_worker.json')
source = source.replace('land1_precision_gradient_evaluations.jsonl',
                        'land1_precision_gradient_v5_evaluations.jsonl')
source = source.replace('stop_land1_gradient_v3.flag',
                        'stop_land1_gradient_v5.flag')
exec(compile(source, str(Path(__file__)), 'exec'))
