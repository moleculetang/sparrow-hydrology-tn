"""Fresh 23-coordinate full-history derivative scan after numerical revision.

Uses the existing frozen scientific point and its complete history. Results
have a new identity and never reuse v5 coordinate passes.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
source = (ROOT / 'scripts/verify_land1_precision_gradient.py').read_text(encoding='utf-8')
source = source.replace('land1_full_history_gradient_v3.json',
                        'land1_full_history_gradient_v6.json')
source = source.replace('land1_precision_gradient_worker.json',
                        'land1_precision_gradient_v6_worker.json')
source = source.replace('land1_precision_gradient_evaluations.jsonl',
                        'land1_precision_gradient_v6_evaluations.jsonl')
source = source.replace('stop_land1_gradient_v3.flag',
                        'stop_land1_gradient_v6.flag')
source = source.replace("ROOT/'d29_training/objective.py']",
    "ROOT/'d29_training/objective.py',ROOT/'d29_platform/coupling.py',"
    "ROOT/'d29_platform/conditional_inputs.py',"
    "ROOT/'config/conditional_reference.json',"
    "ROOT/'config/mineralization_reference.json']")
exec(compile(source, str(Path(__file__)), 'exec'))
