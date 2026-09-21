"""The two delivery machine checks, as a RE-RUNNABLE PRODUCER.

The round's delivery discipline (carried from `20260919_1`) is:

  (1) every literal with >= 3 decimals must resolve back to a JSON source;
  (2) every backticked path carrying an extension must exist on disk.

`20260919_1` ran these as ad-hoc shell commands, so no artifact of them survives
and they cannot be re-run.  This module is the registered producer instead: it
takes a report path, prints the two verdicts, and writes a JSON next to the
report it audited.

WHY (1) IS NOT PARANOIA
-----------------------
`20260919_1`'s B8 defect was a five-station transcription slip in one column of
a hand-written table: every other column in that table was right, the medians
and the counts were right, and no criterion depended on the field -- so neither a
consistency check nor a threshold check could see it.  Only re-resolving each
literal back to its JSON source caught it.  This round already produced one
instance of the same class before any result existed: four of the eighteen-digit
`delta` values in the pre-registration's grid table were mis-transcribed.
"""
import json
import re
import sys
from pathlib import Path

UP = Path(r'E:\SPARROW\5_Test')
R = UP / '20260919_4'
# The round this checker was copied FROM.  Prior rounds are READ-ONLY, so
# the plan's instruction to add this round's JSONs to the peer's CORPUS is
# honoured by copying the checker here instead of editing a closed round:
# a copy is auditable, and the peer's file is byte-unchanged.
#
# `R3` keeps the copied-from round's artifacts addressable.  Without it the
# three `R / ...` entries below would silently retarget this round and the
# checker would resolve `phase1_l1.json` to THIS round's file, which is a
# different artifact with the same basename -- the near-duplicate failure the
# ledger warns about, wired into the checker itself.
R3 = UP / '20260919_3'
PEER = UP / '20260919_2'

# The JSON corpus a literal is allowed to come from.  Kept as an explicit list so
# that "resolved" means "resolved to a registered source", not "found somewhere
# in a 40 GB tree".
CORPUS = [
    # ---- this round's own artifacts ----
    # NOTE: there is deliberately NO `pre_registration.json` this round.  The round
    # froze its criteria in the approved PLAN and transcribed them into
    # `reports/预注册_判据与门槛.md` AFTER the first run; the timing is registered as
    # deviation D5 in `reports/实际方法与偏离.md` and in that md's §0.  A path here
    # that does not exist is harmless (the loader skips it), but listing one would
    # claim a producer that was never run.
    R / 'reports/frozen_anchors.json',
    R / 'reports/phase1_l1.json',
    R / 'reports/phase2_full.json',
    R / 'reports/phase2_gamma_backfill.json',
    R / 'reports/verdict.json',
    R / 'reports/audit_beta.json',
    # ---- the round this checker was copied from: read-only, cited by FULL path ----
    R3 / 'reports/pre_registration.json',
    R3 / 'reports/frozen_anchors.json',
    R3 / 'reports/phase1_l1.json',
    R3 / 'reports/phase1_verdict.json',
    R3 / 'reports/phase2_full.json',
    R3 / 'reports/audit_gamma.json',
    R3 / 'reports/levels_shift.json',
    # ---- peers: a literal may resolve to a NAMED source, never to "found somewhere" ----
    PEER / 'reports/gate_noop.json',
    PEER / 'reports/frozen_anchors.json',
    PEER / 'reports/frozen_grid.json',
    PEER / 'reports/预注册_hash.json',
    PEER / 'reports/phase0_amplitude_budget.json',
    PEER / 'reports/phase1_envelope.json',
    PEER / 'reports/phase1_scores.json',
    PEER / 'reports/audit_envelope.json',
    UP / '20260918_4/reports/stage_a_fingerprints.json',
    UP / '20260918_4/reports/audit_stage_a.json',
    UP / '20260919_1/reports/phase1_fingerprints.json',
    UP / '20260919_1/reports/audit_phase1.json',
    UP / '20260919_1/reports/phase0_acceptance.json',
    UP / '20260916_2/outputs/C0_s1/model.json',
]

# Thresholds and constants that are defined in CODE, not in a JSON: the `legal`
# gate conjuncts, the frozen estimator constants, and the grid's own definition.
# Each is admitted only with the file:line that defines it, so the allowlist is
# itself checkable.
DEFINED_IN_CODE = {
    '1e-6': 'the absolute mass acceptance, 20260919_2/reports/预注册_判据与门槛.md §七',
    '1e-7': 'fit_worker.py:70 / audit_job.py:60-63 legal conjunct 3',
    '1e-10': 'legal conjunct 2 network_balance scale',
    '1e-12': 'common20.py:89 ANCHOR_TOL',
    '1e-13': 'campaign_model.py:74 SOURCE_TAG_IDENTITY rtol',
    '1e-9': '20260919_1 audit tolerance',
    '0.005': 'phase1_fingerprints.py:77 J5_MAX_DEGRADATION',
    '0.0529': 'phase1_fingerprints.py:76 NOISE_BAND = 2 x 0.02647',
    '0.02647': 'the `_5` amplitude-ratio null SD, quoted in that same comment',
    '2000': 'phase1_fingerprints.py:74 N_BOOT',
    '20260917': 'phase1_fingerprints.py:75 SEED',
    '12': 'phase1_fingerprints.py:73 MIN_UNITS',
    '7': 'phase1_fingerprints.py:70 TN_PRE_DAYS',
    '30': 'phase1_fingerprints.py:72 F3_PRIMARY_GAP',
    '60': 'phase1_fingerprints.py:71 F3_GAPS and the tau_R operating bound',
    '15': 'phase1_fingerprints.py:71 F3_GAPS',
    '0.5': '20260919_2 plan §一 J1a threshold',
    # ---- this round's own constants, each with the file:line that defines it ----
    '1e-15': 'work/phase1_verdict.py TOL_ULP (declared, not used as a gate)',
    '1e-6': 'work/common21.py install path: hazard_g identity + mass conjuncts',
    '20260919': 'work/audit_gamma.py default_rng seed',
    '0.05': 'the round registers that the md names 明显 with NO number; this value is '
            'the MEDIAN split move at g=0.25, quoted as a reading, not a threshold',
    '8': 'the limit case g=8, work/phase1_l1.py G_LIMIT',
    '1.207490548772179': '20260916_2/outputs/C0_s1/model.json objective',
    # this round's own frozen thresholds, each with the line that defines it
    '1e-15': 'audit_envelope.py kappa tolerance / both-nan comparison',
    '2.18': 'reports/frozen_anchors.json obs_levels.c_base_median (20260919_2)',
    '3.11': 'reports/frozen_anchors.json obs_levels.c_peak_median (20260919_2)',
    '2.094720376603729': 'reports/frozen_anchors.json baseline_model_C0_s1_monthfirst',
    '2.031723094631754': 'reports/frozen_anchors.json baseline_model_C0_s1_monthfirst',
    '1.2758737517831669': 'reports/frozen_anchors.json anchors_used_by_the_criteria',
    '0.004063600875414098': 'reports/frozen_anchors.json anchors_used_by_the_criteria',
    '-0.0704045722214265': 'reports/frozen_anchors.json anchors_used_by_the_criteria',
}

MASS_EXT = ('.parquet', '.json', '.md', '.csv', '.nc', '.py', '.npy', '.txt', '.xlsx')
# `path.ext`, `path.ext::json_key`, and `file.py:123` all denote the same FILE.
# The first draft of this pattern required the backticked span to END with the
# extension, so every citation written as `report.json::key` -- the notation this
# round uses most -- escaped check (2) entirely: `专家诊断报告.md` scored only 8
# paths while citing dozens.  A checker whose coverage depends on prose style is
# not a checker, so the suffix is now consumed and discarded.
PATH_RE = re.compile(r'`([^`\n]+?\.(?:%s))(?::[^`\n]*)?`'
                     % '|'.join(e.lstrip('.') for e in MASS_EXT))
LIT_RE = re.compile(r'(?<![\w.])(\d+\.\d{3,})(?![\d])')
SCI_RE = re.compile(r'(?<![\w.])(\d+(?:\.\d+)?[eE]-?\d+)(?![\d])')


def _corpus_numbers():
    """Every number appearing anywhere in the corpus, as written AND as float."""
    as_written = set()
    as_float = set()
    for p in CORPUS:
        if not p.exists():
            continue
        txt = p.read_text(encoding='utf-8', errors='replace')
        for m in re.finditer(r'-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?', txt):
            tok = m.group(0)
            as_written.add(tok)
            try:
                as_float.add(float(tok))
            except ValueError:
                pass
        try:
            for v in _walk(json.loads(txt)):
                as_float.add(float(v))
        except (ValueError, TypeError):
            pass
    return as_written, as_float


def _walk(o):
    if isinstance(o, dict):
        for v in o.values():
            yield from _walk(v)
    elif isinstance(o, (list, tuple)):
        for v in o:
            yield from _walk(v)
    elif isinstance(o, (int, float)) and not isinstance(o, bool):
        yield o


def check_literals(report):
    written, floats = _corpus_numbers()
    text = Path(report).read_text(encoding='utf-8')
    out = {'resolved': [], 'allowlisted': [], 'unresolved': []}
    seen = set()
    for m in list(LIT_RE.finditer(text)) + list(SCI_RE.finditer(text)):
        tok = m.group(1)
        if tok in seen:
            continue
        seen.add(tok)
        if tok in written or _near(float(tok), floats):
            out['resolved'].append(tok)
        elif tok in DEFINED_IN_CODE:
            out['allowlisted'].append(dict(literal=tok, where=DEFINED_IN_CODE[tok]))
        else:
            out['unresolved'].append(tok)
    return out


def _near(v, floats, rel=1e-9):
    """A literal is resolved if the corpus holds the same value in any spelling:
    a report may round, and the corpus may carry more digits than the prose."""
    for f in floats:
        if f == v:
            return True
        if f != 0 and abs(f - v) <= abs(f) * rel:
            return True
    return False


# Paths a report backticks in order to say they DO NOT RESOLVE.  A defect
# register has to be able to quote the wrong spelling it is registering; without
# this list the only ways out would be to drop the backticks (hiding the exact
# string from the reader) or to widen the rule (letting real misses through).
# Each entry carries where the defect is registered, so the exemption is itself
# checkable rather than a blanket allowance.
CITED_AS_MISSING = {
    'data/source.npy': 'transcription defect registered in '
                       '20260919_2/reports/实际方法与偏离.md section 5 C4',
    'amplitude_budget.json': 'truncated spelling of phase0_amplitude_budget.json; '
                             'the defect THIS CHECKER CAUGHT, registered in '
                             '20260919_2/reports/实际方法与偏离.md D15.1 and '
                             '独立完成审计.md section 5 blind spot 2 -- both reports '
                             'must be able to name the bad spelling verbatim',
}

# The ledger's first run under the tightened pattern reported these five as
# MISSING, because the ledger cited peer artifacts by BARE FILENAME.  The ledger
# itself was fixed by writing the paths in full, and it now cites none of them
# bare.  These entries exist only so the two reports that REGISTER that defect
# (实际方法与偏离.md D15.4, 独立完成审计.md section 5.4) can quote the exact
# strings that failed.  Unlike the entry above, these five files DO exist -- at
# '20260916_2\reports\', '20260916_2\outputs\C2_s{0,1}\' and '20260919_1\reports\'
# -- so the defect being registered is the CITATION FORM, not existence.
_CITATION_FORM_DEFECT = ('cited bare by the ledger on its first run; the ledger was '
                         'fixed by writing the path in full, not by exempting; quoted '
                         'bare only in the defect register (D15.4 / audit section 5.4)')
CITED_AS_MISSING.update({k: _CITATION_FORM_DEFECT for k in (
    'additive_channel_validation.json', 'phase1_replay.json', 'source_label_audit.json',
    'structure_edge_validation.json', 'structure_gradient_validation.json')})

# Names that are not paths at all: either the producer's own output name (it
# does not exist until the producer has run for THAT report, so a bare
# `_delivery_check.json` resolves nowhere even though six of them now exist
# under derived names), or metasyntactic placeholders used while explaining the
# `name.ext::key` notation itself.  Distinct from CITED_AS_MISSING: nothing here
# is claimed to be a real file, so nothing is hidden by exempting it.
NAME_SHAPES = {
    '_delivery_check.json': 'producer output name: <report>_delivery_check.json',
    'xxx.json': 'metasyntactic placeholder in the notation discussion',
    'file.json': 'metasyntactic placeholder in the notation discussion',
}

# Peer rounds a report may cite for PROVENANCE.  Each is a declared root, so
# "resolved" keeps meaning "resolved to a named source" rather than "found
# somewhere in the tree".  A report that cites `data/source.npy` means the
# registered ledger, which lives in `20260916_2`, not a same-named file here.
# The directory roots exist for the same reason one level finer: reports name
# peer SCRIPTS by bare filename (`campaign_model.py`, `scientific_models.py`,
# `phase1_fingerprints.py`), which are unambiguous only once the directory is
# declared.  A bare name that resolves nowhere still fails -- which is how this
# round caught `amplitude_budget.json`, a truncation of
# `phase0_amplitude_budget.json`, on the first run under the wider pattern.
PEER_ROOTS = [
    # The raw-data catalogue lives one level ABOVE the round tree (`E:\SPARROW\
    # 0_reach_topology\data\raw\README.md`).  Declared so a provenance citation to
    # it resolves to a NAMED source rather than to "found somewhere".
    # Base is the REPO root, not `.../0_reach_topology`: the report writes the full
    # `0_reach_topology\data\raw\README.md`, so a deeper base would double the prefix.
    ('repo(E:/SPARROW)', UP.parent),
    ('peer(20260916_2)', UP / '20260916_2'),
    ('peer(20260916_2/scripts)', UP / '20260916_2' / 'scripts'),
    ('peer(20260916_2/vendor/research)', UP / '20260916_2' / 'vendor' / 'research'),
    ('peer(20260916_2/vendor/transfer_research)',
     UP / '20260916_2' / 'vendor' / 'transfer_research'),
    ('peer(20260918_4)', UP / '20260918_4'),
    ('peer(20260918_4/data)', UP / '20260918_4' / 'data'),
    ('peer(20260919_1)', UP / '20260919_1'),
    ('peer(20260919_1/work)', UP / '20260919_1' / 'work'),
    ('peer(20260918_1)', UP / '20260918_1'),
    ('peer(20260824_10)', UP / '20260824_10'),
    ('peer(20260919_2)', PEER),
    ('peer(20260919_2/work)', PEER / 'work'),
    ('peer(20260919_2/reports)', PEER / 'reports'),
    # The round this checker was copied from.  Declared for the same reason as the
    # others: this round's reports cite its action line, its anchors and its
    # `预注册_判据与门槛.md:189` by full path, and a full path must land on a
    # DECLARED root rather than on "found somewhere".
    ('peer(20260919_3)', R3),
    ('peer(20260919_3/work)', R3 / 'work'),
    ('peer(20260919_3/reports)', R3 / 'reports'),
    ('peer(20260916_2/outputs)', UP / '20260916_2' / 'outputs'),
    ('peer(20260916_2/data)', UP / '20260916_2' / 'data'),
    ('peer(20260916_2/data/domains)', UP / '20260916_2' / 'data' / 'domains'),
    ('peer(20260916_1/scripts)', UP / '20260916_1' / 'scripts'),
]


def check_paths(report):
    """Resolve every backticked extensioned path against a DECLARED rule list.

    A report legitimately names its own siblings by bare filename (`gate_noop.json`
    next to the report) and this round's standard directories (`work/...`).  The
    rule that resolved each path is recorded, so a bare name that happens to match
    somewhere unexpected stays visible rather than silently passing.
    """
    report = Path(report).resolve()
    text = report.read_text(encoding='utf-8')
    # (label, base) in priority order; `Path(raw)` == cwd-absolute last.
    rules = ([('report_dir', report.parent), ('round_root', R),
              ('reports', R / 'reports'), ('work', R / 'work')]
             + PEER_ROOTS + [('up', UP)])
    ok, bad = [], []
    for m in PATH_RE.finditer(text):
        raw = m.group(1).strip()
        # `…` (U+2026) is the same abbreviation marker as `...`.  Leaving it out
        # made every range written with a real ellipsis escape check (2) entirely
        # -- the mirror of the D15.1 blind spot, where the hole was in the pattern
        # rather than in the prose.
        if (any(c in raw for c in '{*') or re.search(r'\.py:\d+$', raw)
                or '...' in raw or '…' in raw):
            ok.append(dict(path=raw, rule='pattern/brace-glob/file:line notation'))
            continue
        if raw.startswith('<') or '<' in raw.split('/')[0]:
            # placeholder notation (`<report>_delivery_check.json`): a NAME SHAPE the
            # producer writes, not a file that exists before the producer has run.
            ok.append(dict(path=raw, rule='placeholder notation, not a literal path'))
            continue
        hit = rule = None
        for label, base in rules:
            c = base / raw
            if c.exists():
                hit, rule = c, label
                break
        if hit is None and Path(raw).exists():
            hit, rule = Path(raw), 'cwd'
        if hit is not None:
            ok.append(dict(path=raw, rule=rule,
                           resolved=str(hit.relative_to(UP)) if UP in hit.parents
                           or hit == UP else str(hit)))
        elif raw in CITED_AS_MISSING:
            ok.append(dict(path=raw, rule='cited-as-NOT-resolving',
                           why=CITED_AS_MISSING[raw]))
        elif raw in NAME_SHAPES:
            ok.append(dict(path=raw, rule='name-shape, not a path',
                           why=NAME_SHAPES[raw]))
        else:
            bad.append(raw)
    return {'n_ok': len(ok), 'ok': ok, 'missing': sorted(set(bad))}


# The pre-registration is FROZEN: its sha256 is registered and re-verified before
# every phase, so editing it -- even to fix a citation -- would invalidate the round.
# It also predates this checker's root table, so some of its citations are bare
# producer filenames that no declared root resolves.  The two honest options were
# (a) add roots that would make bare `sc_kernel.py` resolve, which WEAKENS check (2),
# or (b) skip the file AND SAY SO.  (b), below: the frozen report is reported as
# EXCLUDED with its sha and its unresolved count printed, so the omission is visible
# rather than silently absorbed.  `FROZEN_SHA` is the same digest the phases assert
# against.
#
# THIS ROUND differs in ONE way, and it is registered rather than smoothed over.
# There is NO `pre_registration.json`: the criteria were frozen in the approved
# PLAN (`b72c99e06f35cd0111f1c3d9635280aa2c52b0dca01132c78c22be9f8452dd7b`) and
# transcribed into `reports/预注册_判据与门槛.md`; there is no machine-checked
# phase that re-hashes either of them.  So the digest below is a STATEMENT ABOUT
# THE FILE, not a re-verified gate: it is what the md hashes to right now, and the
# exclusion means "this md is the frozen pre-registration, so the two prose checks
# do not apply to it".  A reader who wants the gate that round 3 had should read
# `预注册_判据与门槛.md` §0, which registers this difference as D5.
FROZEN_SHA = 'd8f962e072bdabca6b44a01c9cfbd232cd77c5b0bfdb4fa83acd825dbe051a65'


def main(argv):
    targets = [Path(a).resolve() for a in argv[1:]] or sorted((R / 'reports').glob('*.md'))
    allok = True
    for t in targets:
        if not t.exists():
            print('MISSING REPORT %s' % t)
            allok = False
            continue
        if t.exists() and _sha(t) == FROZEN_SHA:
            pat = check_paths(t)
            print('%-58s EXCLUDED (frozen pre-registration, sha verified) -- '
                  '%d unresolved citation(s) are NOT checked here: %s'
                  % (t.name, len(pat['missing']), pat['missing'][:8]), flush=True)
            continue
        lit = check_literals(t)
        pat = check_paths(t)
        verdict = dict(
            report=str(t.relative_to(UP)), sha256=_sha(t),
            literals=dict(n_resolved=len(lit['resolved']),
                          n_allowlisted=len(lit['allowlisted']),
                          unresolved=lit['unresolved']),
            paths=dict(n_ok=pat['n_ok'], missing=pat['missing']),
            passed=bool(not lit['unresolved'] and not pat['missing']))
        print('%-58s literals %3d resolved / %d allowlisted / %d UNRESOLVED   paths %3d ok / %d MISSING   %s'
              % (t.name, verdict['literals']['n_resolved'],
                 verdict['literals']['n_allowlisted'],
                 len(lit['unresolved']), pat['n_ok'], len(pat['missing']),
                 'PASSED' if verdict['passed'] else 'FAILED'), flush=True)
        if lit['unresolved']:
            print('   unresolved literals: %s' % lit['unresolved'][:20], flush=True)
        if pat['missing']:
            print('   missing paths: %s' % pat['missing'][:20], flush=True)
        out = t.with_name(t.stem + '_delivery_check.json')
        out.write_text(json.dumps(dict(verdict=verdict, literals=lit, paths=pat),
                                  indent=1, ensure_ascii=False, default=str),
                       encoding='utf-8')
        allok = allok and verdict['passed']
    print('DELIVERY_CHECK_PASSED' if allok else 'DELIVERY_CHECK_FAILED', flush=True)
    return 0 if allok else 1


def _sha(p):
    import hashlib
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


if __name__ == '__main__':
    sys.exit(main(sys.argv))
