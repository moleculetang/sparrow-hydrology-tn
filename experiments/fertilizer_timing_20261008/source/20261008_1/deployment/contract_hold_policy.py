"""Hold the old narrow-range K epoch; unrelated paths keep their own identity."""
import json
from pathlib import Path


def held_jobs(root, gate_hash):
    root=Path(root)
    from d29_platform.runtime import sha
    revision=json.loads((root/'deployment/maize_range_revision_v2.json').read_text(encoding='utf8'))
    if revision['original_gate_sha256']!=gate_hash:
        raise RuntimeError('CONTRACT_HOLD_GATE_CHANGED')
    if sha(root/'config/timing_evidence.json')!=revision['original_config_sha256']:
        raise RuntimeError('CONTRACT_HOLD_REQUIRES_ORIGINAL_SCIENTIFIC_EPOCH')
    jobs=json.loads((root/'config/jobs.json').read_text(encoding='utf8'))
    identifiers=[j['id'] for j in jobs]
    k={j['id'] for j in jobs if j['calendar']=='K'}
    affected=revision['affected_paths']
    if len(jobs)!=44 or len(set(identifiers))!=44 or len(k)!=14 or len(affected)!=14 or set(affected)!=k:
        raise RuntimeError('EXACT14K_OUT_OF44_REQUIRED')
    if revision['planned_delta_bounds']!=[-.1,0.] or revision['current_delta_bounds']!=[-.05,0.]:
        raise RuntimeError('EXPLICIT_APPROVED_RANGE_REVISION_REQUIRED')
    if revision['current_normalized_prior_strength_relative_to_planned']!=4.:
        raise RuntimeError('EXPLICIT_PRIOR_REVISION_REQUIRED')
    return sorted(k)


def append_holds(reviews, jobs):
    result=[dict(r) for r in reviews]
    existing={r['job'] for r in result}
    for job in jobs:
        if job not in existing:
            result.append(dict(job=job,reason='narrow_K_epoch_requires_approved_range_revision',
                               not_convergence_failure=True,original_checkpoints_retained=True))
    return result
