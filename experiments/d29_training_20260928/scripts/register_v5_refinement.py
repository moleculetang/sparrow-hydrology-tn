"""Register only a complete, independent v5 branch refinement that passes."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from d29_platform.runtime import sha, write_json
from d29_training.derivative_gate import refinement_passes


def main():
    scan = json.loads((ROOT / 'outputs/land1_full_history_gradient_v5.json').read_text(encoding='utf-8'))
    path = ROOT / 'outputs/land1_coordinate_v5_6_branch_refinement.json'
    diagnostic = json.loads(path.read_text(encoding='utf-8'))
    if not refinement_passes(scan, 6, diagnostic):
        raise RuntimeError('V5_COORDINATE_6_REFINEMENT_NOT_ACCEPTED')
    checks = diagnostic['checks']
    accepted = [c for c in checks if c['passed'] and
                c['plus']['branch_switch_count'] == 0 and
                c['minus']['branch_switch_count'] == 0]
    if [c['step'] for c in accepted] != [3e-7, 1e-7]:
        raise RuntimeError('UNEXPECTED_REFINED_STEP_PAIR')
    interpretation = ('The original v5 scan failure remains recorded. '
                      'The fixed 1e-6 step has a supply-limited uptake branch change; '
                      'adjacent 3e-7 and 1e-7 steps show no branch changes on either side '
                      'and both meet the unchanged derivative tolerance. This certifies '
                      'the local smooth derivative, not optimizer convergence.')
    review = dict(coordinate=6, path=str(path), sha256=sha(path),
                  accepted_as_local_smooth_derivative_evidence=True,
                  interpretation=interpretation)
    write_json(ROOT / 'outputs/land1_gradient_v5_refinement_reviews.json',
               dict(reviews=[review], original_failed_scan_unchanged=True))


if __name__ == '__main__':
    main()
