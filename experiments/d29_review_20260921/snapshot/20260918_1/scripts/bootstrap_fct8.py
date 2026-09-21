"""Establish this round's own launch state, over inherited parent-round stamps.

`20260918_1` was created by copying `20260917_5` wholesale, so `work/` arrived
carrying the parent run's records: its launch stamp `work/campaign.json`, its
`work/controller_status.json` (which reads `FINAL_AUDIT_FAILED`), and its
`work/experiment_clock.json` whose deadlines are the parent's 72-hour window. This
script does not assume that -- it *proves* it, by asserting each file is still
byte-identical to the parent's copy -- then archives them and writes this round's
own clock. The parent keeps its originals, so nothing is destroyed.

Two of the inherited stamps are actively hostile to a fresh launch:

* `seal_global.py` asserts `work/campaign.json` does NOT exist ('Already
  launched'), so the inherited stamp would stop the seal outright.
* `campaign_controller.py` raises `LAUNCH_CHANGED` when the stamp's
  `launch_sha256` disagrees with `reports/launch_validation.json`. The inherited
  stamp was sealed against the parent's launch and its 28-path job graph, so a new
  seal can never match it.

`configs/campaign.json` is also inherited verbatim and describes the parent: 28
paths, `models: ['D29_BE']`, a 72-hour budget, and `training_years` including 2024.
Its descriptive fields are corrected here. The `solver` and `resources` blocks are
deliberately left byte-identical and asserted unchanged, because `fit_worker.py`
reads `cfg['solver']` at line 53 to drive its escalation schedule -- editing it
would silently give the FCT8 fits a different call budget from the baseline they
are compared against, which is the one thing this round must not do.

The training window recorded is 2021-2023, read from the fold itself rather than
from any design field: `data/folds/<fold>/train.parquet` holds 2021, 2022 and 2023,
while `design['training_years']` says [2021, 2022] for every fold, baseline and
FCT8 alike. The design field is a stale annotation and is not quoted anywhere.

Re-running is idempotent: it refuses to touch a clock this round already wrote.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

RUN = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(RUN / 'scripts')]

import native_runtime as rt                                                      # noqa: E402

PARENT = RUN.parent / '20260917_5'
ARCHIVE = RUN / 'work/inherited_from_20260917_5'
#: parent files this round inherited, and which must therefore be provably the
#: parent's identical bytes before being replaced
INHERITED = ['work/campaign.json', 'work/controller_status.json',
             'work/experiment_clock.json', 'configs/campaign.json']
#: this round's window: the plan's ~48 h small experiment, plus a 4 h reporting tail
PREPARATION_HOURS, TRAINING_HOURS, DELIVERY_HOURS = 2, 48, 52


def main():
    assert Path(sys.prefix).name.lower() == 'sparrow', 'conda sparrow required'
    clock_path = RUN / 'work/experiment_clock.json'

    # Idempotence: only a clock whose bytes differ from the parent's is this
    # round's own, so anything else means the work has already been done once.
    if clock_path.exists():
        if rt.sha(clock_path) != rt.sha(PARENT / 'work/experiment_clock.json'):
            print('clock is already this round\'s own; nothing to do')
            return
    # Proof of inheritance: hash each file against the parent's copy BEFORE it is
    # moved, so the claim 'this was inherited, not produced here' is checked rather
    # than asserted in prose.
    proof = {}
    for rel in INHERITED:
        here, there = RUN / rel, PARENT / rel
        if not here.exists():
            proof[rel] = 'ABSENT_HERE'
            continue
        if not there.exists():
            raise SystemExit('PARENT_ARTEFACT_ABSENT %s' % rel)
        same = rt.sha(here) == rt.sha(there)
        proof[rel] = dict(sha256=rt.sha(here), identical_to_parent=same)
        if not same:
            raise SystemExit('NOT_INHERITED_SHAPE %s' % rel)

    ARCHIVE.mkdir(parents=True, exist_ok=True)
    for rel in INHERITED:
        src = RUN / rel
        if not src.exists():
            continue
        dst = ARCHIVE / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())

    # the launch stamp must go: seal_global.py asserts its absence and the
    # controller can never match an inherited launch_sha256
    (RUN / 'work/campaign.json').unlink(missing_ok=True)

    now = time.time()
    rt.write(clock_path, dict(
        started=now,
        preparation_deadline=now + 3600 * PREPARATION_HOURS,
        training_deadline=now + 3600 * TRAINING_HOURS,
        delivery_deadline=now + 3600 * DELIVERY_HOURS))

    # descriptive fields only; solver/resources are asserted untouched below
    parent_cfg = json.loads((PARENT / 'configs/campaign.json').read_text(encoding='utf-8'))
    cfg = json.loads((RUN / 'configs/campaign.json').read_text(encoding='utf-8'))
    before = dict(solver=json.dumps(cfg.get('solver'), sort_keys=True),
                  resources=json.dumps(cfg.get('resources'), sort_keys=True))
    cfg.update(paths=4, models=['D29_BE', 'FCT8'],
               training_years=[2021, 2022, 2023], evaluation_years=[2024],
               budget=dict(campaign_hours=TRAINING_HOURS, report_hours=4))
    after = dict(solver=json.dumps(cfg.get('solver'), sort_keys=True),
                 resources=json.dumps(cfg.get('resources'), sort_keys=True))
    assert before == after, 'SOLVER_OR_RESOURCES_CHANGED'
    assert after['solver'] == json.dumps(parent_cfg.get('solver'), sort_keys=True), \
        'SOLVER_NOT_PARENT_BUDGET'
    rt.write(RUN / 'configs/campaign.json', cfg)

    rt.write(RUN / 'work/bootstrap_fct8.json', dict(
        status='ROUND_LAUNCH_STATE_ESTABLISHED', inherited=proof,
        archived_to=str(ARCHIVE.relative_to(RUN)),
        removed=['work/campaign.json'],
        clock=dict(started=now, preparation_hours=PREPARATION_HOURS,
                   training_hours=TRAINING_HOURS, delivery_hours=DELIVERY_HOURS),
        campaign_descriptor=dict(paths=4, models=['D29_BE', 'FCT8'],
                                 training_years=[2021, 2022, 2023],
                                 evaluation_years=[2024],
                                 budget_hours=TRAINING_HOURS),
        solver_unchanged_from_parent=True,
        training_window_note=('2021-2023, read from data/folds/<fold>/train.parquet. '
                              'design[training_years] says [2021, 2022] for every fold, '
                              'baseline and FCT8 alike; that field is stale and is not '
                              'quoted. Evaluation year 2024.'),
        statement=('The inherited stamps were proven byte-identical to 20260917_5\'s own '
                   'before being replaced, so nothing belonging to this round was '
                   'discarded and nothing belonging to the parent was destroyed.')))

    print('inherited and archived, then replaced:')
    for rel, p in proof.items():
        print('   %-34s identical_to_parent=%s' % (rel, p.get('identical_to_parent')))
    print('clock: training %.0f h, delivery %.0f h from now' % (TRAINING_HOURS, DELIVERY_HOURS))
    print('campaign descriptor: paths=4 models=%s training=%s eval=%s, solver untouched'
          % (cfg['models'], cfg['training_years'], cfg['evaluation_years']))
    print('wrote work/bootstrap_fct8.json')


if __name__ == '__main__':
    sys.exit(main())
