"""Bind `reports/预注册_冻结.json` (round 20260920_2) field by field.  NEVER writes it.

DIFFERENCE FROM THE PARENT ROUND, AND WHY IT MATTERS
----------------------------------------------------
Round `20260920_1`'s record had NO producer: its writer was an ad-hoc command that
was not kept, so the record was not regenerable and, as round 1's own checker put
it, the record "is NOT regenerable and it MUST NOT be regenerated" -- a new write
would carry a new `recorded_at_utc` and would be a post-hoc re-freeze.

This round's record **HAS a producer**: `work/_freeze_prereg.py`.  That is a real
improvement in one direction (the record can be re-derived and compared against the
code that quotes it) and a new hazard in the other: a producer makes it *possible*
to re-run the freeze, and re-running it after the forwards ran would be exactly the
post-hoc re-freeze gate 1 forbids.  The producer therefore REFUSES to overwrite an
existing record (`FREEZE_RECORD_ALREADY_EXISTS`) and refuses to write at all once
`reports/phase1_arms.json` exists (`FORWARDS_HAVE_ALREADY_RUN`).  This verifier
checks that those two guards are present in the producer's source text -- a textual
check, labelled as such -- and then does the thing that is actually load-bearing:

  * re-derives every binding from the artifacts that have producers;
  * proves the claim that matters -- **the record predates the first forward** --
    from **mtimes**, not from `recorded_at_utc`.  A record that vouches for its own
    timing is not evidence; a record whose file sits strictly between the arm table
    and the first forward's output is.

Writes `reports/prereg_freeze_check.json`.  Read-only otherwise: there is no
`--write-record` flag, not even a commented-out one, and the freeze record's sha256
is taken before and asserted unchanged after, so this claim about itself is
machine-checked rather than asserted in prose.

Runs before the forwards give a PARTIAL verdict (`..._PENDING`): the pre-forward
proof needs `phase1_arms.json` to exist.  Re-run it as step 17 of section 9.1.
"""
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common25 as C

REP = C.ROUND / 'reports'
OUT = REP / 'prereg_freeze_check.json'

FREEZE = REP / '预注册_冻结.json'
MD = REP / '预注册_判据与门槛.md'
ARMS = REP / 'arms.json'
GATES = REP / 'phase0_gates.json'
P1 = REP / 'phase1_arms.json'
CHECKER = C.ROUND / 'work' / 'check_delivery.py'
PRODUCER = C.ROUND / 'work' / '_freeze_prereg.py'

ROUND = '20260920_2'
REC_REF = '预注册_冻结.json'

REGISTERED_FIELDS = sorted([
    'arms_table_sha256', 'closure_form', 'forward_runs_so_far',
    'frozen_before_any_forward', 'n_arms', 'n_candidate_arms', 'n_forwards',
    'n_gate_rows', 'n_named_free_choices', 'phase0_STOP', 'phase0_verdict',
    'pre_registration_bytes', 'pre_registration_path', 'pre_registration_sha256',
    'primary_arm', 'primary_tau_m_days', 'recorded_at_utc', 'round',
])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc(path):
    return datetime.fromtimestamp(os.path.getmtime(path), timezone.utc)


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


freeze_sha_before = sha(FREEZE)

fz = load(FREEZE)
arms = load(ARMS)
gates = load(GATES)
p1 = load(P1) if P1.exists() else None

checks = []


def chk(name, ok, expected, observed, note=''):
    checks.append(dict(name=name, ok=None if ok is None else bool(ok),
                       expected=expected, observed=observed, note=note))


def chk_pending(name, reason, observed, note=''):
    chk(name, None, 'checked once the artifact exists', observed,
        ('PENDING: ' + reason + ('  ' + note if note else '')))


# ---- 1. the arm table it froze, re-derived with the producer's own rule --------------
# `_freeze_prereg.py` hashes `json.dumps(arms['arms'], sort_keys=True, default=str)`.
# Using the producer's own rule (not a hash of the FILE) is the point: it binds the
# record to the arm table's CONTENT, so a rewritten arms.json with the same content
# still binds.
recomputed = hashlib.sha256(
    json.dumps(arms['arms'], sort_keys=True, default=str).encode()).hexdigest()
chk('arms_table_sha256 == recomputed_from_arms.json::arms',
    fz['arms_table_sha256'] == recomputed,
    'frozen=' + fz['arms_table_sha256'][:16], 'recomputed=' + recomputed[:16],
    "_freeze_prereg.py's own rule")
chk('arms_table_sha256 == arms.json::hashes.table_sha256',
    fz['arms_table_sha256'] == arms['hashes']['table_sha256'],
    arms['hashes']['table_sha256'][:16], fz['arms_table_sha256'][:16],
    'the binding phase1_arms.py asserts before it will run any arm')
chk('arms_table_sha256 == arms.json::gate.table_sha256',
    fz['arms_table_sha256'] == arms['gate']['table_sha256'],
    arms['gate']['table_sha256'][:16], fz['arms_table_sha256'][:16])

# ---- 2. arm count / identity / closure / named free choices --------------------------
chk('n_arms == len(arms.json::arms)', fz['n_arms'] == len(arms['arms']),
    len(arms['arms']), fz['n_arms'])
chk('n_arms == len(arms.json::gate.arms)', fz['n_arms'] == len(arms['gate']['arms']),
    len(arms['gate']['arms']), fz['n_arms'])
chk('n_arms == arms.json::gate.n_arms == phase0_gates.json::3.4.n_arms',
    fz['n_arms'] == arms['gate']['n_arms'] == gates['3.4']['n_arms'],
    '%s / %s' % (arms['gate']['n_arms'], gates['3.4']['n_arms']), fz['n_arms'])
n_cand = sum(1 for a in arms['arms'] if a.get('is_candidate'))
chk('n_candidate_arms == count(is_candidate) == gate.n_candidates',
    fz['n_candidate_arms'] == n_cand == arms['gate']['n_candidates'],
    '%d / %d' % (n_cand, arms['gate']['n_candidates']), fz['n_candidate_arms'],
    'the six candidates that may sign a verdict; the rest may not (plan section 5)')
chk('n_forwards == arms.json::gate.n_forwards == phase0_gates.json::3.4.n_forwards',
    fz['n_forwards'] == arms['gate']['n_forwards'] == gates['3.4']['n_forwards'],
    '%s / %s' % (arms['gate']['n_forwards'], gates['3.4']['n_forwards']),
    fz['n_forwards'], 'K-inf is a gate read from disk at zero cost; TSTAR is budgeted '
    'under MAX_ROOT_FORWARDS, so neither is an arm forward')
chk('n_gate_rows == len(arms.json::gate_rows)', fz['n_gate_rows'] == len(arms['gate_rows']),
    len(arms['gate_rows']), fz['n_gate_rows'])
chk('primary_arm == arms.json::primary_arm',
    fz['primary_arm'] == arms['primary_arm'], arms['primary_arm'], fz['primary_arm'],
    'the arm that alone may issue capability (plan section 5)')
chk('primary_arm == arms.json::gate.primary_arm',
    fz['primary_arm'] == arms['gate']['primary_arm'], arms['gate']['primary_arm'],
    fz['primary_arm'])
chk('primary_arm is the literal P-1e2 (N16)',
    fz['primary_arm'] == 'P-1e2' and arms['gate']['primary_is_a_literal'] is True,
    'P-1e2', fz['primary_arm'],
    'so FULL_CAPABILITY cannot arise on the TSTAR code path')
chk('primary_tau_m_days == float(arms.json::primary_tau_m)',
    abs(float(fz['primary_tau_m_days']) - float(arms['primary_tau_m'])) == 0.0,
    arms['primary_tau_m'], fz['primary_tau_m_days'])
chk('primary_tau_m_days == the blind rule\'s selected grid point',
    float(fz['primary_tau_m_days'])
    == float(arms['blind_rule']['selected_grid_point_days']),
    arms['blind_rule']['selected_grid_point_days'], fz['primary_tau_m_days'],
    'tau_m = sqrt(tau_water * tau_frozen) = %.6f d, nearest grid point in log space'
    % arms['blind_rule']['raw_days'])
chk('closure_form == arms.json::closure_form',
    fz['closure_form'] == arms['closure_form'], arms['closure_form'],
    fz['closure_form'], 'g is g(x)=x this round, chosen by the timing gate')
chk('closure_form == phase0_gates.json::3.3_closure.closure_choice',
    fz['closure_form'] == gates['3.3_closure']['closure_choice'],
    gates['3.3_closure']['closure_choice'], fz['closure_form'])
chk('round == %s' % ROUND, fz['round'] == ROUND, ROUND, fz['round'])
nfc = arms['named_free_choices']
chk('n_named_free_choices == len(arms.json::named_free_choices)',
    fz['n_named_free_choices'] == len(nfc), len(nfc), fz['n_named_free_choices'],
    'N9-prime enumerates every named free choice before any forward')

# ---- 3. the pre-registration md it freezes, and the checker's own exclusion ----------
md_sha = sha(MD)
md_bytes = os.path.getsize(MD)
chk('pre_registration_sha256 == sha256(md_now)', fz['pre_registration_sha256'] == md_sha,
    md_sha[:16], fz['pre_registration_sha256'][:16],
    'so the md is UNCHANGED SINCE THE FREEZE -- gate 1 says frozen before any forward, '
    'and this is what proves it was not edited after')
chk('pre_registration_bytes == size(md_now)',
    fz['pre_registration_bytes'] == md_bytes, md_bytes, fz['pre_registration_bytes'])
rel = fz['pre_registration_path'].replace('\\', '/')
chk('pre_registration_path resolves to the file just hashed',
    (C.ROUND / rel).resolve() == MD.resolve(), str(MD), str(C.ROUND / rel))

# Cross-file binding: pull FROZEN_SHA out of the checker's source rather than importing
# it, so a checker whose exclusion was silently repointed is caught here too.  NOTE the
# inherited copy still names the PARENT round's md; this check is expected to fail until
# step 6 of section 9.1 repoints it, and it is reported separately from the freeze
# record's own binding so that "the checker is stale" is never read as "the freeze
# record is unbound".
CHK_BASELINE = '62734f98795cdce6d61e71588209db23b2e3fb00cd5f6a150032af0d4d79d4ff'
_fs = None
if CHECKER.exists():
    for line in CHECKER.read_text(encoding='utf-8').splitlines():
        if line.startswith('FROZEN_SHA = '):
            _fs = line.split('=', 1)[1].strip().strip("'\"")
checker_ok = (_fs is not None and _fs == fz['pre_registration_sha256'] == md_sha)
checker_status = ('CHECKER_EXCLUSION_REPOINTED' if checker_ok else
                  'CHECKER_EXCLUSION_STILL_POINTS_AT_THE_PARENT_ROUND'
                  if _fs == CHK_BASELINE else 'CHECKER_EXCLUSION_UNRESOLVED')

# ---- 4. forward count, before and after ---------------------------------------------
chk('forward_runs_so_far == 0', fz['forward_runs_so_far'] == 0, 0,
    fz['forward_runs_so_far'], 'the value the record claims AT THE FREEZE MOMENT')
if p1 is None:
    chk_pending('n_forwards == phase1_arms.json::forward_count',
                'reports/phase1_arms.json does not exist yet, so no forward has run',
                'phase1_arms.json absent')
    chk_pending('mtime(freeze) < mtime(phase1_arms.json) -- FROZEN BEFORE THE FIRST FORWARD',
                'needs the first forward\'s output; this is the run to re-check after '
                'section 9.1 step 2', 'phase1_arms.json absent')
else:
    chk('n_forwards == phase1_arms.json::forward_count',
        fz['n_forwards'] == p1['forward_count'], p1['forward_count'], fz['n_forwards'],
        'the planned count and the count actually run agree')
chk('frozen_before_any_forward is True', fz['frozen_before_any_forward'] is True, True,
    fz['frozen_before_any_forward'])

# ---- 5. THE PRE-FORWARD PROOF: from mtimes, not from the record's own timestamp ------
# `recorded_at_utc` is the record TESTIFYING ABOUT ITSELF.  These are the filesystem's
# account of the same events, and they are not writable by whoever edited the JSON.
t_arms, t_freeze, t_md = utc(ARMS), utc(FREEZE), utc(MD)
rec = datetime.fromisoformat(fz['recorded_at_utc'])
t_p1 = utc(P1) if P1.exists() else None

chk('mtime(arms.json) < mtime(freeze)', t_arms < t_freeze,
    t_arms.isoformat(), t_freeze.isoformat(),
    'the record must postdate the arm table it freezes')
chk('mtime(md) <= mtime(freeze)', t_md <= t_freeze,
    t_md.isoformat(), t_freeze.isoformat(),
    'it must postdate the md whose digest it carries')
chk('recorded_at_utc == mtime(freeze) to the second',
    abs((rec - t_freeze).total_seconds()) < 1.0, t_freeze.isoformat(),
    rec.isoformat(),
    'the producer stamps the file it has just written; the mtime ordering below is '
    'the actual proof, this is reported for consistency')
chk('recorded_at_utc postdates every input it binds',
    rec >= max(t_arms, t_md), max(t_arms, t_md).isoformat(), rec.isoformat(),
    'a record whose stamp predates its own inputs would mean the inputs moved after '
    'the freeze')
if t_p1 is not None:
    chk('mtime(freeze) < mtime(phase1_arms.json) -- FROZEN BEFORE THE FIRST FORWARD',
        t_freeze < t_p1, t_p1.isoformat(), t_freeze.isoformat(),
        'phase1_arms.json is written by the run that does the forwards; the record is '
        'older than it by %.1f min' % ((t_p1 - t_freeze).total_seconds() / 60.))
    chk('recorded_at_utc lies in the same window as the mtimes',
        t_arms <= rec <= t_p1, 'in [%s, %s]' % (t_arms.isoformat(), t_p1.isoformat()),
        rec.isoformat(),
        'the record\'s own timestamp is CONSISTENT with the filesystem -- reported '
        'because it is evidence, not because it is proof; the mtime ordering is the proof')

# ---- 6. the producer exists, and its two refusal guards are in its own source --------
# A TEXTUAL check, and labelled as one: it shows the guard is written down; it does not
# execute it (executing it would try to write the record, which is the thing this file
# must never do).
chk('the record HAS a producer this round (work/_freeze_prereg.py)',
    PRODUCER.exists(), str(PRODUCER), 'exists' if PRODUCER.exists() else 'MISSING',
    'the parent round had none; this round can re-derive the record instead of only '
    'binding it')
if PRODUCER.exists():
    src = PRODUCER.read_text(encoding='utf-8')
    chk('the producer refuses to overwrite an existing record (textual)',
        'FREEZE_RECORD_ALREADY_EXISTS' in src and REC_REF in src,
        'FREEZE_RECORD_ALREADY_EXISTS on %s' % REC_REF,
        'present' if 'FREEZE_RECORD_ALREADY_EXISTS' in src else 'ABSENT',
        'without this guard a second run would stamp a new recorded_at_utc -- a '
        'post-hoc re-freeze')
    chk('the producer refuses to write once a forward has run (textual)',
        'FORWARDS_HAVE_ALREADY_RUN' in src and 'phase1_arms.json' in src,
        'FORWARDS_HAVE_ALREADY_RUN when reports/phase1_arms.json exists',
        'present' if 'FORWARDS_HAVE_ALREADY_RUN' in src else 'ABSENT')

# ---- 7. fields that are READ but NOT independently bound ----------------------------
# Declared rather than quietly omitted: an unbound field is not a failed check, but a
# reader who assumes every field was verified would be wrong.
UNBOUND = {k: fz[k] for k in ('phase0_verdict', 'phase0_STOP') if k in fz}
chk('every field of the record is either bound above or declared unbound',
    sorted(fz.keys()) == REGISTERED_FIELDS,
    'the %d registered fields' % len(REGISTERED_FIELDS), '%d fields' % len(fz),
    'unbound: %s -- they restate phase0_gates.json, which HAS its own producer, so they '
    'are read as data rather than re-derived here' % sorted(UNBOUND))

# ---- 8. this file did not re-freeze anything ----------------------------------------
freeze_sha_after = sha(FREEZE)
chk('freeze record byte-identical after this run (no post-hoc re-freeze)',
    freeze_sha_before == freeze_sha_after, freeze_sha_before[:16],
    freeze_sha_after[:16],
    'there is no code path in this file that writes it; the assertion is what makes '
    'that checkable by someone who does not read the whole file')

n_ok = sum(1 for c in checks if c['ok'] is True)
pending = [c['name'] for c in checks if c['ok'] is None]
failed = [c['name'] for c in checks if c['ok'] is False]
verdict = ('FREEZE_RECORD_UNBOUND' if failed else
           'FREEZE_RECORD_BOUND__PENDING' if pending else
           'FREEZE_RECORD_BOUND_AND_PRE_FORWARD')
out = dict(
    round=ROUND,
    verdict=verdict,
    n_checks=len(checks), n_ok=n_ok, n_pending=len(pending), n_failed=len(failed),
    failed=failed, pending=pending,
    producer_status='HAS_PRODUCER: work/_freeze_prereg.py',
    what_that_means=(
        'reports/预注册_冻结.json IS regenerable this round, unlike the parent round. '
        'It MUST NOT be regenerated: the producer refuses to overwrite an existing '
        'record and refuses to write once reports/phase1_arms.json exists, because a '
        're-stamped recorded_at_utc after the forwards would be a post-hoc re-freeze. '
        'This verifier binds the record field by field to artifacts that have their own '
        'producers, and proves its timing from mtimes rather than from its own '
        'timestamp field.  It never writes the record.'),
    is_regenerable=True,
    must_not_be_regenerated=True,
    checker_exclusion_status=checker_status,
    checker_FROZEN_SHA_seen=_fs,
    freeze_sha256=freeze_sha_after,
    order=[dict(file=p, mtime_utc=utc(p).isoformat())
           for p in (str(ARMS), str(MD), str(FREEZE))],
    pre_forward_margin_minutes=(round((t_p1 - t_freeze).total_seconds() / 60., 3)
                                if t_p1 is not None else None),
    bound_fields=[c['name'] for c in checks if c['ok'] is True],
    unbound_fields=sorted(UNBOUND),
    checks=checks,
)
OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False, default=str),
               encoding='utf-8')

print('[verify_prereg_freeze] %s   %d ok / %d pending / %d failed'
      % (verdict, n_ok, len(pending), len(failed)))
for f in failed:
    print('   FAILED :', f)
for p in pending:
    print('   PENDING:', p)
print('   arm table -> md -> FREEZE, in mtimes:')
for row in out['order']:
    print('     %-28s %s' % (Path(row['file']).name, row['mtime_utc']))
if t_p1 is not None:
    print('   frozen before the first forward by %.1f min '
          '(from mtimes, not from the record)' % out['pre_forward_margin_minutes'])
else:
    print('   first forward has NOT run yet -- the pre-forward mtime proof is pending')
print('   checker exclusion: %s (FROZEN_SHA=%s)'
      % (checker_status, str(_fs)[:16] if _fs else None))
print('   freeze record sha256 %s -- UNCHANGED by this run: it binds, it never writes'
      % freeze_sha_after[:16])
print('[verify_prereg_freeze] wrote %s' % OUT, flush=True)
