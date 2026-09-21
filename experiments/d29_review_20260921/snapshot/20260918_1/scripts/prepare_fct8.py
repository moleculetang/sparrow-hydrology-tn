"""Assemble the FCT8 folds, designs, frozen weights and job graph.

Non-destructive by construction: the baseline folds/jobs are read from the
frozen parent round `../20260917_5/configs/` (read-only) and the copies in this
round are asserted byte-identical to them first. Nothing already published in
`outputs/` is touched. Re-running is idempotent -- the two `*_FCT8` folds are
rebuilt from scratch each time and the four FCT8 jobs are regenerated.

`configs/folds.json` keeps all 8 baseline folds and appends 2 FCT8 folds.
`configs/jobs.json` holds **only** the 4 FCT8 jobs: the baseline is reused as
published artefacts (proved by the F1-7 replay), never refitted here, so putting
the 16 baseline jobs back in would only invite a 25-hour re-run.
"""
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

RUN = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(RUN / 'scripts')]
import campaign_model as CM
from fct8_model import build_phi_bar, DELTA_NAMES

PARENT = RUN.parent / '20260917_5'
#: (baseline fold, baseline published tag, new FCT8 fold, objective id)
PAIRS = [('T24_G_D_H1', 'T24_G_D_H1_s0', 'T24_G_D_H1_FCT8', 'D'),
         ('T24_G_M_H1', 'T24_G_M_H1_s0', 'T24_G_M_H1_FCT8', 'M')]


def phi_bar_for(base_fold, base_tag):
    """Freeze w = h_preD29 at the published theta_ref and reduce it to phi_bar."""
    folds = json.loads((PARENT / 'configs/folds.json').read_text(encoding='utf-8'))
    fold = folds[base_fold]
    data = CM.load_data(fold['domain'], verify=True)
    train = __import__('pandas').read_parquet(RUN / 'data/folds' / base_fold / 'train.parquet')
    design = json.loads((RUN / 'data/designs' / (base_fold + '.json')).read_text(encoding='utf-8'))
    model = CM.make_model(data, train, 'D29_BE', design)
    ref = json.loads((RUN / 'outputs' / base_tag / 'model.json').read_text(encoding='utf-8'))
    t = torch.tensor(np.asarray(ref['parameters'], float))
    if len(t) != 30:
        raise ValueError('BASELINE_NOT_30 %s %d' % (base_tag, len(t)))
    # Endpoints.flux_parameters, minus its last line (the tanh factor): this is
    # exactly h_preD29, verified bitwise against the published model.
    h, _, _ = CM.Predictor.hazard(model, t[:29])
    h = h * torch.exp((model.pi * (t[29] - t[1]))[None, :] * model.logcontact)
    h_pub, _, _, _ = model.flux_parameters(t)
    mult = torch.exp(math.log(10) * torch.tanh(
        torch.einsum('trj,j->tr', model.dynamic_basis, t[21:29]) / math.log(10)))
    if float((h * mult - h_pub).abs().max()) != 0.0:
        raise ValueError('H_PRED29_NOT_BITWISE ' + base_tag)

    w = h.detach().numpy().astype(np.float64)
    basis = model.dynamic_basis.numpy().astype(np.float64)
    starts = np.asarray(data.starts)
    count = np.bincount(np.asarray(data.mid), minlength=len(data.months)).astype(float)
    phibar, mass, degenerate = build_phi_bar(w, basis, starts, len(data.months),
                                             w.shape[1], count)
    if not np.isfinite(phibar).all():
        raise ValueError('NONFINITE_PHI_BAR ' + base_fold)
    # the weighted within-month mean of (phi_bar - phi) must vanish identically
    resid = phibar[np.asarray(data.mid)] - basis
    num = np.add.reduceat(w[:, :, None] * resid, starts, axis=0)
    orth = float(np.abs(np.where(~degenerate[:, :, None],
                                 num / np.where(mass == 0, 1., mass)[:, :, None], 0.)).max())
    stats = dict(reference_tag=base_tag, reference_pg=ref['pg'], reference_objective=ref['objective'],
                 weight_fraction_exactly_zero=float((w == 0).mean()),
                 cells_with_zero_weight_mass=int(degenerate.sum()),
                 reaches_with_a_zero_mass_month=int(degenerate.any(0).sum()),
                 min_positive_monthly_weight_mass=float(mass[~degenerate].min()),
                 uniform_fallback_cells=int(degenerate.sum()),
                 max_abs_phi_bar=float(np.abs(phibar).max()),
                 orthogonality_residual=orth)
    return w, phibar, stats


def main():
    # This round is initialised as a byte copy of the parent, so the FIRST run proves
    # that by comparing the two config files directly. Later runs have already
    # overwritten this round's copies with the FCT8 versions, so they instead check
    # that the parent has not changed since -- against the hashes recorded by the
    # first run. Without this branch the function is not re-runnable at all, which
    # the docstring used to claim it was.
    record = RUN / 'work/fct8_prepare.json'
    for name in ('folds', 'jobs'):
        a = PARENT / 'configs' / (name + '.json')
        if record.exists():
            recorded = json.loads(record.read_text(encoding='utf-8'))['parent_config_sha256']
            if CM.sha(a) != recorded[name]:
                raise ValueError('PARENT_CONFIG_CHANGED ' + name)
        elif CM.sha(a) != CM.sha(RUN / 'configs' / (name + '.json')):
            raise ValueError('ROUND_NOT_A_COPY_OF_PARENT ' + name)
    folds = json.loads((PARENT / 'configs/folds.json').read_text(encoding='utf-8'))
    print('parent configs verified byte-identical; %d baseline folds' % len(folds))

    report = {}
    for base_fold, base_tag, new_fold, objective in PAIRS:
        base = folds[base_fold]
        if base['arm'] != 'H1' or base['domain'] != 'FULL24C':
            raise ValueError('UNEXPECTED_BASELINE ' + base_fold)
        w, phibar, stats = phi_bar_for(base_fold, base_tag)
        print('\n%s -> %s' % (base_fold, new_fold))
        for k in ('weight_fraction_exactly_zero', 'cells_with_zero_weight_mass',
                  'reaches_with_a_zero_mass_month', 'min_positive_monthly_weight_mass',
                  'max_abs_phi_bar', 'orthogonality_residual'):
            print('  %-38s %s' % (k, stats[k]))

        dst = RUN / 'data/folds' / new_fold
        dst.mkdir(parents=True, exist_ok=True)
        # training labels and the observation registry must be byte-identical
        for f in ('train.parquet', 'registry.json'):
            shutil.copyfile(RUN / 'data/folds' / base_fold / f, dst / f)
        train_sha = CM.sha(dst / 'train.parquet')
        registry_sha = CM.sha(dst / 'registry.json')
        if train_sha != base['train_sha256']:
            raise ValueError('TRAIN_NOT_IDENTICAL ' + new_fold)
        if registry_sha != CM.sha(RUN / 'data/folds' / base_fold / 'registry.json'):
            raise ValueError('REGISTRY_NOT_IDENTICAL ' + new_fold)
        np.save(dst / 'phi_bar.npy', phibar)
        np.save(dst / 'w_preD29.npy', w)

        design = json.loads((RUN / 'data/designs' / (base_fold + '.json')).read_text(encoding='utf-8'))
        if design['observation_registry_hash'] != registry_sha:
            raise ValueError('REGISTRY_HASH_MISMATCH ' + base_fold)
        design['observation_registry_file'] = r'data\folds\%s\registry.json' % new_fold
        design['fct8'] = dict(phi_bar=r'data\folds\%s\phi_bar.npy' % new_fold,
                              phi_bar_sha256=CM.sha(dst / 'phi_bar.npy'),
                              weight=r'data\folds\%s\w_preD29.npy' % new_fold,
                              weight_sha256=CM.sha(dst / 'w_preD29.npy'),
                              base_fold=base_fold, delta_names=DELTA_NAMES, **stats)
        (RUN / 'data/designs' / (new_fold + '.json')).write_text(
            json.dumps(design, ensure_ascii=False, sort_keys=True), encoding='utf-8')

        folds[new_fold] = dict(base, arm='H1', mode=base['mode'], fct8_of=base_fold,
                               train_sha256=train_sha)
        report[new_fold] = dict(base_fold=base_fold, base_tag=base_tag,
                                train_sha256=train_sha, registry_sha256=registry_sha,
                                objective=objective, **stats)

    # rebuild the three config files that record jobs rather than delete-by-hand
    for name in ('configs/folds.json',):
        (RUN / name).write_text(json.dumps(folds, ensure_ascii=False, sort_keys=True, indent=1),
                                encoding='utf-8')
    jobs = []
    # Mirror `prepare_global.py:99` exactly: a D arm's `s0` warm-starts from the TWO
    # starts of **the M arm**, and `s1` from nothing. The M arm of this round is the
    # FCT8 M fold -- the D29_BE M arm is reused as published artefacts, not refitted,
    # so its tags are not in this job graph and naming them would deadlock the child
    # on a dependency that is never produced.
    m_fold = [p[2] for p in PAIRS if p[3] == 'M'][0]
    for new_fold, objective in [(p[2], p[3]) for p in PAIRS]:
        for start in (0, 1):
            parents = (['%s_s0' % m_fold, '%s_s1' % m_fold]
                       if objective == 'D' and start == 0 else [])
            jobs.append(dict(tag='%s_s%d' % (new_fold, start), fold=new_fold, kind='FCT8',
                             start=start, operator_id='OU', mapping_id='G1',
                             observation_operator='MATCH', objective_id=objective,
                             dependencies=list(parents), parent_tags=list(parents),
                             priority=0, arm='H1'))
    # A dependency that names the job itself can never be satisfied: the controller
    # waits for `outputs/<dep>/audit.json`, which only that job would write. The child
    # would then sit until the campaign deadline, losing the arm silently. Guard both
    # that and any parent outside the graph.
    tags = {j['tag'] for j in jobs}
    for j in jobs:
        if j['tag'] in j['parent_tags']:
            raise ValueError('SELF_PARENT %s' % j['tag'])
        if not set(j['parent_tags']) <= tags:
            raise ValueError('UNKNOWN_PARENT %s %s' % (j['tag'], j['parent_tags']))
    (RUN / 'configs/jobs.json').write_text(json.dumps(jobs, ensure_ascii=False, indent=1),
                                           encoding='utf-8')
    (RUN / 'work/fct8_prepare.json').write_text(
        json.dumps(dict(parent_round=str(PARENT),
                        parent_config_sha256=dict(folds=CM.sha(PARENT / 'configs/folds.json'),
                                                  jobs=CM.sha(PARENT / 'configs/jobs.json')),
                        new_folds=report, jobs=[j['tag'] for j in jobs]),
                   ensure_ascii=False, indent=1, sort_keys=True), encoding='utf-8')
    print('\nwrote %d folds, %d jobs, %d designs' % (len(folds), len(jobs), len(PAIRS)))


if __name__ == '__main__':
    main()
