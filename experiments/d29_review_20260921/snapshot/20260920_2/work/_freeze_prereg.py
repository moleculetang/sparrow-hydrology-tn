"""Producer for `reports\\预注册_冻结.json` (round 20260920_2).

WHY THIS FILE EXISTS
--------------------
Round `20260920_1` left a freeze record with **no producer**: its writer was an
ad-hoc shell command that was never kept, so the record could not be re-derived
and could not be shown to be reproducible.  Round 1's own
`work/_verify_prereg_freeze.py` documents that by exhaustion (every
`C.write_json` / `write_text` / `json.dump` call site in the round is
enumerated; none of them writes the record) and then deliberately **refuses to
rewrite it** -- a new `recorded_at_utc` would be a post-hoc re-freeze.

This round gives the record a producer, and the producer's first act is to
**REFUSE to overwrite an existing record**.  That refusal is the whole point:

  * the record is written exactly once, BEFORE any registered forward;
  * a second run is an error, not a refresh, because re-stamping
    `recorded_at_utc` after the forwards ran is precisely the failure mode
    `_verify_prereg_freeze.py` exists to detect;
  * therefore the record's content can only be re-derived and compared, never
    regenerated -- which is what the verifier does.

The producer also refuses to write if `reports/phase1_arms.json` exists, i.e.
if any forward has already run.

Run:  PYTHONIOENCODING=utf-8 .../python.exe -B work/_freeze_prereg.py
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

WORK = Path(__file__).resolve().parent
ROOT = WORK.parent
REPORTS = ROOT / 'reports'

REC_PATH = REPORTS / '预注册_冻结.json'
ARMS_PATH = REPORTS / 'arms.json'
GATES_PATH = REPORTS / 'phase0_gates.json'
MD_PATH = REPORTS / '预注册_判据与门槛.md'
MD_REL = 'reports\\预注册_判据与门槛.md'
PHASE1_PATH = REPORTS / 'phase1_arms.json'

ROUND = '20260920_2'


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def load(p: Path):
    return json.loads(p.read_text(encoding='utf-8'))


def main() -> int:
    # ---- the two refusals -------------------------------------------------
    if REC_PATH.exists():
        raise SystemExit(
            'FREEZE_RECORD_ALREADY_EXISTS %s -- this producer writes the record '
            'exactly once, before any forward. Re-running it after the forwards '
            'would be a post-hoc re-freeze (a new recorded_at_utc). Use '
            'work/_verify_prereg_freeze.py to BIND and COMPARE the record '
            'instead; it deliberately never writes it.' % REC_PATH.name)
    for p in (ARMS_PATH, GATES_PATH, MD_PATH):
        if not p.exists():
            raise SystemExit('FREEZE_INPUT_MISSING %s' % p)
    if PHASE1_PATH.exists():
        raise SystemExit(
            'FORWARDS_HAVE_ALREADY_RUN %s exists -- the freeze record must be '
            'written BEFORE any registered forward.' % PHASE1_PATH.name)

    # ---- inputs, re-derived rather than trusted ---------------------------
    arms = load(ARMS_PATH)
    gates = load(GATES_PATH)

    # the producer's own rule for the arms-table digest: the table is the
    # `arms` list, canonicalised. Re-derived here so the record does not simply
    # copy a number out of the file it is meant to bind.
    table_sha = sha256_bytes(
        json.dumps(arms['arms'], sort_keys=True, default=str).encode('utf-8'))
    recorded = arms['hashes']['table_sha256']
    if table_sha != recorded:
        raise SystemExit('ARMS_TABLE_SHA_DISAGREES re-derived=%s recorded=%s'
                         % (table_sha, recorded))

    md_bytes = MD_PATH.read_bytes()
    md_sha = sha256_bytes(md_bytes)

    # ---- the gate state the record certifies ------------------------------
    if gates.get('STOP'):
        raise SystemExit('PHASE0_STOP_IS_NOT_EMPTY %r' % (gates['STOP'],))
    if gates.get('verdict') != 'PHASE0_GATES_PASSED':
        raise SystemExit('PHASE0_VERDICT_IS_NOT_PASSED %r' % (gates.get('verdict'),))

    for key, want, where in (
            ('k_ex', 0.0, 'arms.json'),
            ('n_fits', 0, 'arms.json'),
            ('fit_worker_calls', 0, 'arms.json'),
            ('closure_form', gates['3.3_closure']['closure_choice'], 'arms.json'),
    ):
        got = arms[key]
        if got != want:
            raise SystemExit('FREEZE_INPUT_%s_IN_%s %r != %r' % (key, where, got, want))
    if arms['pre_registered'] is not True:
        raise SystemExit('PRE_REGISTERED_IS_NOT_TRUE')
    if arms['frozen_before_any_forward'] is not True:
        raise SystemExit('FROZEN_BEFORE_ANY_FORWARD_IS_NOT_TRUE')
    if arms['round'] != ROUND:
        raise SystemExit('ROUND_MISMATCH %r' % (arms['round'],))

    n_arms = len(arms['arms'])
    if n_arms != gates['3.4']['n_arms']:
        raise SystemExit('N_ARMS_DISAGREES %d vs %d' % (n_arms, gates['3.4']['n_arms']))
    n_cand = sum(1 for a in arms['arms'] if a.get('is_candidate'))
    if n_cand != gates['3.4']['n_candidates']:
        raise SystemExit('N_CANDIDATES_DISAGREES %d vs %d'
                         % (n_cand, gates['3.4']['n_candidates']))
    n_forwards = gates['3.4']['n_forwards']
    if n_forwards != n_arms:
        raise SystemExit('N_FORWARDS_DISAGREES_WITH_THE_TABLE %d vs %d'
                         % (n_forwards, n_arms))
    n_named = len(arms['named_free_choices'])
    if n_named != 13:
        raise SystemExit('N_NAMED_FREE_CHOICES %d != 13' % n_named)

    primary = gates['3.4']['primary_arm']
    if arms['primary_arm'] != primary:
        raise SystemExit('PRIMARY_ARM_DISAGREES %r vs %r'
                         % (arms['primary_arm'], primary))

    record = {
        'round': ROUND,
        'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
        'forward_runs_so_far': 0,
        'frozen_before_any_forward': True,
        'phase0_verdict': gates['verdict'],
        'phase0_STOP': list(gates['STOP']),
        'pre_registration_path': MD_REL,
        'pre_registration_bytes': len(md_bytes),
        'pre_registration_sha256': md_sha,
        'arms_table_sha256': table_sha,
        'closure_form': arms['closure_form'],
        'n_arms': n_arms,
        'n_candidate_arms': n_cand,
        'n_forwards': n_forwards,
        'n_gate_rows': len(arms.get('gate_rows', [])),
        'n_named_free_choices': n_named,
        'primary_arm': primary,
        'primary_tau_m_days': float(arms['primary_tau_m']),
    }

    REC_PATH.write_text(
        json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False) + '\n',
        encoding='utf-8')

    # self-check: read it back and re-derive both digests from the file bytes
    back = load(REC_PATH)
    if back != record:
        raise SystemExit('FREEZE_RECORD_DID_NOT_ROUNDTRIP')
    if (sha256_bytes(MD_PATH.read_bytes()) != md_sha
            or sha256_bytes(json.dumps(arms['arms'], sort_keys=True,
                                       default=str).encode('utf-8')) != table_sha):
        raise SystemExit('FREEZE_RECORD_INPUTS_MOVED_WHILE_WRITING')

    print('[freeze] wrote %s' % REC_PATH)
    print(json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
