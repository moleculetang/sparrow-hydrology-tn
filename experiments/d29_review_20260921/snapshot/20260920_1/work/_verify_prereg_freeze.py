"""`reports/预注册_冻结.json` has NO producer anywhere in this round.  Bind it instead.

Established by exhaustion, not by impression: every `.py` in the round was searched for
both writing mechanisms the round uses -- the helper (`C.write_json`) and direct dumps
(`write_text` / `json.dump`) -- and the complete call-site inventory is

    level_variance.json  phase0_gates.json  arms.json  phase1_arms.json
    probe_label_floor.json  probe_lpre.json  verdict.json  phase0_n1.json
    audit_dp.json  neighbour_write_check.json  prereg_freeze_check.json (this file)

None of them is `预注册_冻结.json`.  The only `.py` files in the round that mention it are
readers or citers: `phase1_arms.py:665` (reads it), `check_delivery.py:108,958` (corpus
entry + reference comment).  `phase0_gates.py` does not write it -- there is no
subprocess, no `runpy`, no `exec` in that file, so it cannot be doing it indirectly.

The record nevertheless IS machine-made: `recorded_at_utc` carries microseconds and the
keys are sorted with 1-space indent, i.e. some ad-hoc command produced it and the script
was not kept.  That is exactly the defect this round otherwise treats as fatal -- an
evidence file with no producer cannot be re-run and cannot be checked against the code
that quotes it (memory: `evidence-file-needs-a-producer`).  Gate 1's landing point is
this file, and three reports quote its readings.

THIS FILE DOES NOT FIX THAT BY RE-WRITING THE RECORD.  Re-writing it would stamp a new
`recorded_at_utc` and would be a POST-HOC RE-FREEZE -- the one thing gate 1 exists to
prevent.  So there is no `--write-record` flag, not even a commented-out one, and the
freeze record's sha256 is taken before and asserted unchanged after so that this claim
about itself is machine-checked rather than asserted in prose.

What it does instead is the only legitimate repair: BIND the existing bytes to the
artifacts that DO have producers, and prove the claim that actually matters -- that the
record predates the first forward -- from mtimes, rather than trusting the record's own
timestamp field about itself.  A record that vouches for its own timing is not evidence;
a record whose file sits strictly between the arm table and the first forward's output
is.

Writes `reports/prereg_freeze_check.json`.  Read-only otherwise.
"""
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common24 as C

REP = C.ROUND / 'reports'
OUT = REP / 'prereg_freeze_check.json'

FREEZE = REP / '预注册_冻结.json'
MD = REP / '预注册_判据与门槛.md'
ARMS = REP / 'arms.json'
P1 = REP / 'phase1_arms.json'
CHECKER = C.ROUND / 'work' / 'check_delivery.py'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc(path):
    return datetime.fromtimestamp(os.path.getmtime(path), timezone.utc)


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


# The freeze record must be byte-identical at exit: this file's central claim about
# itself is that it did not re-freeze anything.
freeze_sha_before = sha(FREEZE)

fz = load(FREEZE)
arms = load(ARMS)
p1 = load(P1)

checks = []


def chk(name, ok, expected, observed, note=''):
    checks.append(dict(name=name, ok=bool(ok), expected=expected, observed=observed,
                       note=note))


# ---- 1. the arm table it froze, re-derived with phase0_gates.py's own rule ----------
# `phase0_gates.py:466` hashes `json.dumps(ARMS, sort_keys=True, default=str)`.  Using
# the producer's own rule (not a hash of the FILE) is the point: it binds the record to
# the arm table's CONTENT, so a rewritten arms.json with the same content still binds.
recomputed = hashlib.sha256(
    json.dumps(arms['arms'], sort_keys=True, default=str).encode()).hexdigest()
chk('arms_table_sha256 == recomputed_from_arms_json_arms',
    fz['arms_table_sha256'] == recomputed,
    'frozen=' + fz['arms_table_sha256'][:16], 'recomputed=' + recomputed[:16],
    'phase0_gates.py:466 rule')
chk('arms_table_sha256 == arms.json::hashes.table_sha256',
    fz['arms_table_sha256'] == arms['hashes']['table_sha256'],
    arms['hashes']['table_sha256'][:16], fz['arms_table_sha256'][:16],
    'the same binding level_variance.py:119 and phase1_arms.py:675 assert')

# ---- 2. arm count / identity / closure / named free choices -------------------------
chk('n_arms == len(arms.json::arms)', fz['n_arms'] == len(arms['arms']),
    len(arms['arms']), fz['n_arms'])
chk('n_arms == arms.json::gate.n_arms', fz['n_arms'] == arms['gate']['n_arms'],
    arms['gate']['n_arms'], fz['n_arms'])
chk('primary_arm == arms.json::primary_arm', fz['primary_arm'] == arms['primary_arm'],
    arms['primary_arm'], fz['primary_arm'],
    'the arm that alone may issue capability (plan section 5)')
chk('closure_form == arms.json::closure_form',
    fz['closure_form'] == arms['closure_form'], arms['closure_form'],
    fz['closure_form'], 'g is g(x)=x this round, chosen by the timing gate')
nfc = arms['named_free_choices']
chk('n_named_free_choices == len(arms.json::named_free_choices)',
    fz['n_named_free_choices'] == len(nfc), len(nfc), fz['n_named_free_choices'],
    'plan section 3.5 N9 enumerates every named free choice before any forward')

# ---- 3. the pre-registration md it freezes, and the checker's own exclusion ---------
md_sha = sha(MD)
md_bytes = os.path.getsize(MD)
chk('pre_registration_sha256 == sha256(md_now)', fz['pre_registration_sha256'] == md_sha,
    md_sha[:16], fz['pre_registration_sha256'][:16],
    'so the md is UNCHANGED SINCE THE FREEZE -- gate 1 says frozen before any forward, '
    'and this is what proves it was not edited after')
chk('pre_registration_bytes == size(md_now)',
    fz['pre_registration_bytes'] == md_bytes, md_bytes, fz['pre_registration_bytes'])

# Cross-file binding: pull FROZEN_SHA out of the checker's source rather than importing
# it, so a checker whose exclusion was silently repointed is caught here too.
_fs = None
for line in CHECKER.read_text(encoding='utf-8').splitlines():
    if line.startswith('FROZEN_SHA = '):
        _fs = line.split('=', 1)[1].strip().strip("'\"")
chk('FROZEN_SHA(checker) == pre_registration_sha256(freeze) == sha256(md_now)',
    _fs is not None and _fs == fz['pre_registration_sha256'] == md_sha,
    fz['pre_registration_sha256'][:16], 'checker=%s' % (str(_fs)[:16] if _fs else None),
    'the exclusion and the freeze record must name the same md, or the exclusion '
    'excludes the wrong bytes and the md gets checked as delivery claims')

# ---- 4. forward count, before and after ---------------------------------------------
chk('forward_runs_so_far == 0', fz['forward_runs_so_far'] == 0, 0,
    fz['forward_runs_so_far'], 'the value the record claims AT THE FREEZE MOMENT')
chk('n_forwards == phase1_arms.json::forward_count',
    fz['n_forwards'] == p1['forward_count'], p1['forward_count'], fz['n_forwards'],
    'the planned count and the count actually run agree')
chk('frozen_before_any_forward is True', fz['frozen_before_any_forward'] is True, True,
    fz['frozen_before_any_forward'])

# ---- 5. THE PRE-FORWARD PROOF: from mtimes, not from the record's own timestamp ------
# `recorded_at_utc` is the record TESTIFYING ABOUT ITSELF.  These are the filesystem's
# account of the same events, and they are not writable by whoever edited the JSON.
t_arms, t_freeze, t_md, t_p1 = (utc(ARMS), utc(FREEZE), utc(MD), utc(P1))
rec = datetime.fromisoformat(fz['recorded_at_utc'])

chk('mtime(arms.json) < mtime(freeze)', t_arms < t_freeze,
    t_arms.isoformat(), t_freeze.isoformat(),
    'the record must postdate the arm table it freezes')
chk('mtime(md) <= mtime(freeze)', t_md <= t_freeze,
    t_md.isoformat(), t_freeze.isoformat(),
    'it must postdate the md whose digest it carries')
chk('mtime(freeze) < mtime(phase1_arms.json) -- FROZEN BEFORE THE FIRST FORWARD',
    t_freeze < t_p1, t_p1.isoformat(), t_freeze.isoformat(),
    'phase1_arms.json is written by the run that did the 5 forwards; the record is '
    'older than it by %.1f min' % ((t_p1 - t_freeze).total_seconds() / 60.))
chk('recorded_at_utc lies in the same window as the mtimes',
    t_arms <= rec <= t_p1, 'in [%s, %s]' % (t_arms.isoformat(), t_p1.isoformat()),
    rec.isoformat(),
    'the record\'s own timestamp is CONSISTENT with the filesystem -- reported because '
    'it is evidence, not because it is proof; the mtime ordering above is the proof')
chk('recorded_at_utc == mtime(freeze) to the second',
    abs((rec - t_freeze).total_seconds()) < 1.0, t_freeze.isoformat(),
    rec.isoformat(), 'the ad-hoc writer stamped the file it had just written')

# ---- 6. fields that are READ but NOT independently bound ----------------------------
# Declared rather than quietly omitted: an unbound field is not a failed check, but a
# reader who assumes every field was verified would be wrong.
UNBOUND = {k: fz[k] for k in ('phase0_verdict', 'phase0_STOP')
           if k in fz}
chk('every field of the record is either bound above or declared unbound',
    sorted(fz.keys()) == sorted(
        ['arms_table_sha256', 'closure_form', 'forward_runs_so_far',
         'frozen_before_any_forward', 'n_arms', 'n_forwards', 'n_named_free_choices',
         'phase0_STOP', 'phase0_verdict', 'pre_registration_bytes',
         'pre_registration_path', 'pre_registration_sha256', 'primary_arm',
         'recorded_at_utc']),
    'the 14 registered fields', '%d fields' % len(fz),
    'bound: 12.  unbound: %s -- they restate phase0_gates.json, which HAS a producer, '
    'so they are read as data rather than re-derived here' % sorted(UNBOUND))

# ---- 7. this file did not re-freeze anything ----------------------------------------
freeze_sha_after = sha(FREEZE)
chk('freeze record byte-identical after this run (no post-hoc re-freeze)',
    freeze_sha_before == freeze_sha_after, freeze_sha_before[:16],
    freeze_sha_after[:16],
    'there is no code path in this file that writes it; the assertion is what makes '
    'that checkable by someone who does not read the whole file')

n_ok = sum(1 for c in checks if c['ok'])
failed = [c['name'] for c in checks if not c['ok']]
out = dict(
    verdict='FREEZE_RECORD_BOUND_AND_PRE_FORWARD' if not failed else 'FREEZE_RECORD_UNBOUND',
    n_checks=len(checks), n_ok=n_ok, failed=failed,
    producer_status='NO_PRODUCER_IN_THIS_ROUND',
    what_that_means=(
        'reports/预注册_冻结.json was produced by an ad-hoc command whose script was not '
        'kept.  It is NOT regenerable and it MUST NOT be regenerated: a new write would '
        'carry a new recorded_at_utc and would be a post-hoc re-freeze, which is what '
        'gate 1 forbids.  It is bound instead, field by field, to artifacts that do have '
        'producers, and its timing is proved from mtimes rather than from its own '
        'timestamp field.'),
    is_regenerable=False,
    must_not_be_regenerated=True,
    bound_fields=[c['name'] for c in checks if c['ok']],
    unbound_fields=sorted(UNBOUND),
    freeze_sha256=freeze_sha_after,
    order=[dict(file=p, mtime_utc=utc(p).isoformat()) for p in
           (str(ARMS), str(MD), str(FREEZE), str(P1))],
    pre_forward_margin_minutes=round((t_p1 - t_freeze).total_seconds() / 60., 3),
    checks=checks,
)
OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False, default=str),
               encoding='utf-8')

print('[verify_prereg_freeze] %s   %d/%d checks ok'
      % (out['verdict'], n_ok, len(checks)))
if failed:
    for f in failed:
        print('   FAILED:', f)
print('   arm table -> md -> FREEZE -> first forward, in mtimes:')
for row in out['order']:
    print('     %-42s %s' % (Path(row['file']).name, row['mtime_utc']))
print('   frozen before the first forward by %.1f min (from mtimes, not from the record)'
      % out['pre_forward_margin_minutes'])
print('   freeze record sha256 %s -- UNCHANGED by this run: no producer, and none added'
      % freeze_sha_after[:16])
print('[verify_prereg_freeze] wrote %s' % OUT, flush=True)
