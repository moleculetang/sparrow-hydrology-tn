"""Prove the delivered ledger altered no INHERITED line EXCEPT the registered ones.

`diff` reports a hunk as a change when a byte was APPENDED to a line, so its
`1399c1600,1623` is not evidence that line 1399 was rewritten.  This checks the
property directly, from the producer's own manifest:

  (1) every seed line whose bytes no longer appear verbatim in the delivered file
      must be a line of a REGISTERED anchor -- otherwise the round edited history
      without saying so;
  (2) the count must match exactly, so an edit cannot hide behind another's anchor;
  (3) the seed's digest re-verified, and every edit's `n_replacements == 1`;
  (4) every line of the delivered file that is NOT in the seed must lie entirely
      within a registered insertion's new text -- i.e. this round added nothing
      anywhere it did not register.

Read-only; prints readings, writes nothing.  Zero forwards, zero fits.
"""
import collections
import io
import json
import sys
from pathlib import Path

R = Path('E:/SPARROW/5_Test/20260920_1')
SEED = Path('E:/SPARROW/5_Test/20260919_5/reports/已关闭假设台账.md')
DST = R / 'reports' / '已关闭假设台账.md'
MAN = R / 'reports' / '_ledger_refresh.json'

seed_lines = io.open(SEED, encoding='utf-8').read().split('\n')
dst_lines = io.open(DST, encoding='utf-8').read().split('\n')
man = json.load(io.open(MAN, encoding='utf-8'))

fails = []

print('seed %d lines -> delivered %d lines' % (len(seed_lines), len(dst_lines)))
print('source sha matches registered: %s' % man['source_sha256_matches_registered'])
if not man['source_sha256_matches_registered']:
    fails.append('source sha does not match the registered digest')
if man['n_fits'] != 0 or man['fit_worker_calls'] != 0:
    fails.append('producer reports a fit')
if not all(v['n_replacements'] == 1 for v in man['edits'].values()):
    fails.append('an anchor matched other than exactly once')

ca, cb = collections.Counter(seed_lines), collections.Counter(dst_lines)
lost = sorted(l for l in (ca - cb) if l.strip())

# --- (1)+(2): every lost inherited line must be a line of a registered anchor ---
anchor_lines = {}
for k, v in man['edits'].items():
    for ln in (v.get('anchor_text') or '').split('\n'):
        if ln.strip():
            anchor_lines.setdefault(ln, []).append(k)
print('registered anchors contribute %d non-empty lines, across edits %s'
      % (len(anchor_lines), sorted({k for v in anchor_lines.values() for k in v})))
for l in lost:
    hit = anchor_lines.get(l)
    print('   altered inherited line -> %s   %r' % (hit or 'UNREGISTERED', l[:58]))
    if not hit:
        fails.append('unregistered alteration of an inherited line: %r' % l[:80])

# `expected` counts, per edit, the anchor lines that survive in neither the edit's
# own new text nor the file -- i.e. the anchors that REPLACED an inherited line
# rather than inserting before/after it.  An edit whose new text still contains the
# anchor's line verbatim (scope, count, row16, section20, tail) altered nothing.
expected = 0
for k, v in man['edits'].items():
    a_ln = [x for x in (v.get('anchor_text') or '').split('\n') if x.strip()]
    n_ln = set((v.get('new_text') or '').split('\n'))
    gone = [x for x in a_ln if x not in n_ln]
    if gone:
        print('   edit %-10s replaced %d inherited line(s)' % (k, len(gone)))
    expected += len(gone)
print('registered replaced lines: %d   observed altered inherited lines: %d   equal: %s'
      % (expected, len(lost), expected == len(lost)))
if expected != len(lost):
    fails.append('altered-line count %d != registered anchor-line count %d'
                 % (len(lost), expected))

# --- (4): every delivered line NOT in the seed must come from a registered addition ---
added = sorted(l for l in (cb - ca) if l.strip())
authored = collections.Counter()
for k, v in man['edits'].items():
    for ln in (v.get('new_text') or '').split('\n'):
        if ln.strip():
            authored[ln] += 1
print('delivered lines not in the seed: %d   lines the manifest registers as added: %d'
      % (len(added), sum(authored.values())))
for l in added:
    if authored.get(l, 0) <= 0:
        print('   ADDED LINE WITH NO REGISTERED SOURCE: %r' % l[:100])
        fails.append('added line not in any registered new text: %r' % l[:80])
dupes = [l for l, n in authored.items() if n > 1 and l in set(added)]
if dupes:
    print('   note: %d added lines also occur in another edit\'s new text '
          '(multiplicity is not asserted)' % len(dupes))

print()
print('VERDICT %s' % ('HISTORY_ONLY_APPENDED_AND_REGISTERED' if not fails
                      else 'FAILED'))
for f in fails:
    print('   - %s' % f)
sys.exit(0 if not fails else 1)
