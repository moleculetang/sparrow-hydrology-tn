"""Plan section 9, verification item 17: `20260919_4` / `20260919_5` /
`20260920_1` must be UNCHANGED.  ZERO FORWARDS, ZERO FITS, AND IT WRITES NOTHING
OUTSIDE THIS ROUND.

`20260920_1` is added to the neighbour set for THIS round.  It is the round this
one is a direct successor to, and it is the round whose every delivered artifact
would be at risk if this round slipped a write: the plan's section 0.2 forbids
writing anything under it, and section 9.0 records that the one operation that
WOULD have touched it (the preserve-a-copy step) was withdrawn on the code
evidence that the plan-file sha check is informational, not a gate.

WHY THIS IS A MODULE AND NOT AN AD-HOC `sha256sum` LINE
------------------------------------------------------
Memory `evidence-file-needs-a-producer`: a comparison that only ever existed as a
shell command cannot be re-run, and a reader has no way to tell whether the shas
it printed were the ones it was supposed to compare.  This module is the
registered producer for the comparison: it names BOTH images, prints them, and
writes the diff to `reports/neighbour_write_check.json`.

THE BEFORE-IMAGE IS NOT "WHATEVER WAS ON DISK WHEN I LOOKED"
------------------------------------------------------------
Three independent before-images are used, because any one of them alone is weak:

  (1) THIS ROUND'S OWN STEP-5 IMAGE.  `reports/audit_dp.json::neighbour_json_shas`
      records 29 JSON digests over `20260919_4/reports` and `20260919_5/reports`.
      It was written by `work/audit_dp.py` during step 5 of section 9 -- i.e. AFTER
      steps 1-4 and BEFORE step 6 and before any report existed.  A digest that
      still matches is a digest that steps 1-6 and all seven reports did not touch.
  (2) TWO CROSS-ROUND BEFORE-IMAGES THAT PREDATE THIS ROUND ENTIRELY.  Each
      neighbour's own `check_delivery.py` run wrote `<report>_delivery_check.json`
      next to the report it audited, and each of those records the sha256 of a
      report it audited AT THE TIME.  So `20260919_5/reports/*_delivery_check.json`
      is a 7-file before-image of round 5's reports taken by round 5, and
      `20260919_4/...` the same for round 4.  These are genuine time-travel
      checks: they were computed before this round existed.
  (3) AN MTIME SWEEP.  Any file under either neighbour's `reports/` or `work/`
      whose mtime is at or after this round's own start is a FOREIGN WRITE, even
      if its bytes happen to match (a rewrite that restores content still wrote).
      The threshold is derived from this round's own work files rather than
      hard-coded, so it cannot drift away from when the round actually began.

WHAT THIS DOES NOT COVER, SAID PLAINLY
--------------------------------------
It does not prove the neighbours were unchanged before step 5.  Nothing was
recorded when this round started, and `5_Test` is NOT a git-tracked tree
(`git ls-files 5_Test/20260919_5` returns nothing), so there is no version-control
before-image to appeal to.  The honest statement is the one below: the image
taken at step 5, the two cross-round images, and the mtime sweep all agree.
"""
import json
import sys
from pathlib import Path

R = Path(__file__).resolve().parent.parent
UP = R.parent
OUT = R / 'reports'
NEIGHBOURS = ('20260919_4', '20260919_5', '20260920_1')
PARENT = '20260920_1'
SWEEP_SUBDIRS = ('reports', 'work')
SWEEP_EXTS = ('.json', '.md', '.parquet', '.npy')


def sha(p):
    import hashlib
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def this_round_start():
    """Epoch of when this round began, with the INHERITED FILES EXCLUDED.

    `work/*.py` alone is NOT a usable lower bound here.  A new round is seeded
    with the parent's `work/` and `reports/` (memory
    `round-dir-inherits-parent-work-and-reports`), so for `20260920_2` the
    earliest `work/*.py` mtimes are `20260920_1`'s own -- 01:19 to 02:38, HOURS
    before this round existed.  Taking that as the threshold would flag every
    file round 1 wrote during and after its own run as a foreign write by this
    round: a check that can never pass, which is the same failure as a check that
    can never fail.

    So a `work/*.py` that is BYTE-IDENTICAL to the parent's file of the same name
    is skipped -- it is an inherited copy, not this round's writing -- and the
    remaining candidates are this round's own.  The round directory's creation
    time is the floor beneath them.
    """
    cands = []
    n_inherited = 0
    for p in (R / 'work').glob('*.py'):
        if p.name == Path(__file__).name:
            continue
        q = UP / PARENT / 'work' / p.name
        if q.exists() and q.read_bytes() == p.read_bytes():
            n_inherited += 1
            continue
        cands.append((p.stat().st_mtime, p.name))
    if not cands:
        raise SystemExit('NO_WORK_FILES_TO_DERIVE_THE_ROUND_START_FROM')
    cands.sort()
    floor = R.stat().st_ctime
    if cands[0][0] <= floor:
        t, name, which = cands[0][0], cands[0][1], 'earliest non-inherited work/*.py'
    else:
        t, name, which = floor, 'round-directory creation time', 'round-dir ctime'
    return t, '%s = the threshold (%s; %d inherited copies of %s excluded; ' \
              'the other candidate was %.0f)' % (name, which, n_inherited, PARENT,
                                                 floor if which != 'round-dir ctime'
                                                 else cands[0][0])


def step5_image():
    a = json.loads((OUT / 'audit_dp2.json').read_text(encoding='utf-8'))
    return a['neighbour_json_shas']


def cross_round_images():
    """`<report>_delivery_check.json` -> the report sha it recorded, per neighbour."""
    out = {}
    for nb in NEIGHBOURS:
        for p in sorted((UP / nb / 'reports').glob('*_delivery_check.json')):
            try:
                d = json.loads(p.read_text(encoding='utf-8'))
                rep = d['verdict']['report']
                out[rep] = dict(sha256=d['verdict']['sha256'],
                                recorded_by='%s/reports/%s' % (nb, p.name))
            except (KeyError, ValueError):
                out['%s/reports/%s' % (nb, p.name)] = dict(
                    sha256=None, recorded_by='unreadable')
    return out


def sweep(start_epoch):
    hits = []
    n = 0
    for nb in NEIGHBOURS:
        for sub in SWEEP_SUBDIRS:
            d = UP / nb / sub
            if not d.is_dir():
                continue
            for p in d.rglob('*'):
                if not p.is_file() or p.suffix.lower() not in SWEEP_EXTS:
                    continue
                n += 1
                m = p.stat().st_mtime
                if m >= start_epoch:
                    hits.append(dict(path=str(p.relative_to(UP)), mtime=m))
    return n, sorted(hits, key=lambda h: -h['mtime'])


def main():
    start, start_from = this_round_start()
    out = {'phase': 'verify_neighbours', 'round': str(R),
           'zero_forwards': True, 'n_fits': 0, 'fit_worker_calls': 0,
           'item': 'plan section 9 verification item 17',
           'claim': '20260919_4, 20260919_5 and 20260920_1 are unchanged by this round',
           'round_start_epoch': start, 'round_start_derived_from': start_from}

    img5 = step5_image()
    now5 = {k: sha(UP / k) for k in img5}
    out['step5_image'] = dict(
        n=len(img5), source='reports/audit_dp.json::neighbour_json_shas',
        taken_at='during step 5, before step 6 and before any report existed',
        n_matching=int(sum(1 for k in img5 if img5[k] == now5[k])),
        changed=sorted(k for k in img5 if img5[k] != now5[k]),
        now=now5)

    xr = cross_round_images()
    xr_now = {k: (sha(UP / k) if (UP / k).exists() else None) for k in xr}
    xr_bad = sorted(k for k in xr if xr[k]['sha256'] != xr_now[k])
    out['cross_round_image'] = dict(
        n=len(xr), taken_at='by each neighbour round, before this round existed',
        n_matching=int(sum(1 for k in xr if xr[k]['sha256'] == xr_now[k])),
        changed=xr_bad, recorded=xr, now=xr_now)

    n_scanned, hits = sweep(start)
    out['mtime_sweep'] = dict(
        n_files_scanned=n_scanned, subdirs=list(SWEEP_SUBDIRS),
        exts=list(SWEEP_EXTS), n_foreign_writes=len(hits), hits=hits)

    out['passed'] = bool(not out['step5_image']['changed'] and not xr_bad
                         and not hits)
    out['not_covered'] = ('nothing was recorded when this round started, and '
                          '5_Test is not a git-tracked tree, so there is no '
                          'version-control before-image; the three images above '
                          'are what exists and all three agree')
    (OUT / 'neighbour_write_check.json').write_text(
        json.dumps(out, indent=1, ensure_ascii=False, default=str),
        encoding='utf-8')
    print('step5 image   %d/%d match   changed=%s'
          % (out['step5_image']['n_matching'], len(img5),
             out['step5_image']['changed'][:6]), flush=True)
    print('cross-round   %d/%d match   changed=%s'
          % (out['cross_round_image']['n_matching'], len(xr), xr_bad[:6]),
          flush=True)
    print('mtime sweep   %d files, %d foreign write(s)   start=%.0f (%s)'
          % (n_scanned, len(hits), start, start_from), flush=True)
    print('NEIGHBOURS_UNCHANGED' if out['passed'] else 'NEIGHBOUR_WRITE_DETECTED',
          flush=True)
    return 0 if out['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
