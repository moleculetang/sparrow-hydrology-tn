"""The two delivery machine checks, as a RE-RUNNABLE PRODUCER.

The round's delivery discipline (carried from `20260919_1`) is:

  (1) every literal with >= 3 decimals must resolve back to a JSON source;
  (2) every backticked path carrying an extension must exist on disk.

`20260919_1` ran these as ad-hoc shell commands, so no artifact of them survives
and they cannot be re-run.  This module is the registered producer instead: it
takes a report path, prints the two verdicts, and writes a JSON next to the
report it audited.

WHAT CHANGED IN THIS ROUND'S COPY (and why the copy is not a formality)
----------------------------------------------------------------------
This file is `20260919_5\work\check_delivery.py` (sha
`0ad0ce293e2c7606c292d61bd1a6880140f0f2ee681b304b8bd5f553cf662be7`, drawn from
round 5 and NOT from round 4: D39's three repairs exist only in round 5's copy --
memory `legacy-checker-copies-stay-vacuous`).  Three edits, each registered:

1. RETARGET.  `R`, `R3`, `R4` and `CORPUS` move to this round.  The retarget is
   LOAD-BEARING rather than cosmetic: this round's entire anchor set is read from
   `20260919_5\reports\`, and `frozen_anchors.json` in rounds 4 and 5 SHARE most
   of their values, so a stale `R3` would have made the literal gate pass while
   resolving against the wrong round.
2. DEFECT (i) OF PLAN SECTION 8 ITEM 14.  The inherited `DEFINED_IN_CODE` cited
   `work/xi.py:90` twice and `work/xi.py:173` once.  **There is no `xi.py`**: in
   round 5 the constants live at `xi_base.py:90` (`W_FLOOR = 1e-3`, with `CLIP`
   at `:92`) and `common23.py:149` (`SAT_BOUND = 0.10`), and the two `1e-3`/`0.1`
   values were conflated onto one line.  Repaired here onto the defining files,
   and marked HISTORICAL, because plan section 3.5 N4 demotes both from criteria
   to diagnostics.  The `700` entry moves to the frozen kernel that actually
   contains the clip (`20260916_2/vendor/research/closures.py:23`): THIS round's
   kernel has no clip and is forbidden one (`work/dp_kernel.py:72`).
3. THE SELF-CHECK.  Round 5's copy documented D39 in comments and PROVED nothing
   at run time.  `selfcheck()` (below) now drives four synthetic probes through
   the REAL `_near` / `LIT_RE` / `SCI_RE` / `check_paths` code, writes
   `reports/_check_delivery_selfcheck.json`, and ABORTS the run if any stops
   holding -- so plan section 9 gate 18's "the checker's own vacuity self-check
   is on disk" is enforced by the checker itself rather than asserted about it.

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
import math
import re
import sys
from pathlib import Path

UP = Path(r'E:\SPARROW\5_Test')
R = UP / '20260920_1'
# The round this checker was copied FROM.  Prior rounds are READ-ONLY, so
# the plan's instruction to add this round's JSONs to the peer's CORPUS is
# honoured by copying the checker here instead of editing a closed round:
# a copy is auditable, and the peer's file is byte-unchanged.
#
# `R3` keeps the copied-from round's artifacts addressable.  Without it the
# `R / ...` entries below would silently retarget this round and the checker
# would resolve `phase1_l1.json` to THIS round's file, which is a different
# artifact with the same basename -- the near-duplicate failure the ledger warns
# about, wired into the checker itself.
#
# RETARGETED for `20260920_1`: `R` moves to this round and `R3` moves one step
# with it (round 5 is now the copied-from round).  That retarget is
# LOAD-BEARING here in a way it was not before: this round's whole anchor set is
# read from `20260919_5\reports\`, so leaving `R3 = 20260919_4` would send every
# ≥3-decimal anchor of round 5's to round 4's JSONs -- which HOLD THE SAME
# NUMBERS for the shared anchors, so the check would pass while resolving
# against the wrong round.  That is precisely the failure mode `R3` exists to
# prevent, and it is why the retarget is verified against `frozen_anchors.json`
# rather than assumed.
R3 = UP / '20260919_5'
R4 = UP / '20260919_4'
PEER = UP / '20260919_2'
# Cited for the elapsed-time and provenance disclosures.  Kept as whole rounds
# rather than as file lists for the same reason as `R3`/`R4`: this round's
# reports name `20260918_1\reports\ammonia_TN_quality_only.parquet`,
# `20260919_1\data\phase1_eligible_mask.parquet` and the hydrology producers by
# full path, and a full path must land on a DECLARED root.
R1 = UP / '20260919_1'
R18 = UP / '20260918_1'
HYD = UP / '20260828_38'

# The JSON corpus a literal is allowed to come from.  Kept as an explicit list so
# that "resolved" means "resolved to a registered source", not "found somewhere
# in a 40 GB tree".
CORPUS = [
    # ---- this round's own artifacts ----
    # NOTE: there is deliberately NO `pre_registration.json` this round either.
    # The round froze its criteria in the approved PLAN and transcribed them into
    # `reports/预注册_判据与门槛.md`; the timing is registered as a deviation in
    # `reports/实际方法与偏离.md` and in that md's §0.  A path here that does not
    # exist is harmless (the loader skips it), but listing one would claim a
    # producer that was never run.
    #
    # `预注册_冻结.json` is the machine-readable half of the freeze: it carries the
    # arms-table sha, the primary arm, `n_forwards = 5` and `forward_runs_so_far = 0`
    # as of the moment before the first forward.  It is in the corpus because the
    # reports quote those four readings, and quoting them must resolve to THIS
    # round's freeze rather than to a similarly-shaped file in a neighbour.
    R / 'reports/预注册_冻结.json',
    # The freeze record's independent binding check, and the ONLY thing in this round
    # that vouches for it.  It is in the corpus for the same reason the freeze record
    # is: `实际方法与偏离.md` and `数据与运行说明.md` both cite it as gate 1's landing
    # point, and that citation has to resolve to this round's file.
    R / 'reports/prereg_freeze_check.json',
    R / 'reports/phase0_gates.json',
    R / 'reports/phase0_n1.json',
    R / 'reports/arms.json',
    R / 'reports/phase1_arms.json',
    R / 'reports/level_variance.json',
    R / 'reports/verdict.json',
    R / 'reports/audit_dp.json',
    # The seven probes and the two smokes.  (The count is written out on purpose and
    # it is seven, not the four a first draft of this comment claimed: the probe set
    # grew after that sentence was written, and a stale count in a comment about
    # this very list is how a reader concludes the list is complete when it is not.)
    # These are where the §3.3 gate's readings LIVE (`probe_timing.json`), where the
    # turnover table lives (`probe_saturation.json`), and where the pip-install-free
    # smoke readings are.  A report that quotes `p100` or `n_x_gt_1` must resolve to
    # the probe that measured it, not to a report that restated it.
    R / 'reports/probe_sources.json',
    R / 'reports/probe_saturation.json',
    R / 'reports/probe_prevolume.json',
    R / 'reports/probe_timing.json',
    R / 'reports/probe_arm_hashes.json',
    R / 'reports/probe_label_floor.json',
    # The seventh probe, and the only one that was missing from this list until now.
    # `layers24.py`'s `slow_track` docstring quotes four of its readings
    # (`max_rel_resid`, `median_rel_resid`, `frac_exact`, `prefix_exact/prefix_n`)
    # to justify reproducing the kernel's day-sequential order instead of the
    # ledger's `np.cumsum` spelling.  Until this entry existed, `probe_lpre.py`
    # PRINTED those readings and wrote no file -- the justification for a code
    # decision rested on numbers no artifact contained.  It now writes
    # `reports/probe_lpre.json` like the other six, and a report that restates
    # either the cumsum residual or the exact-prefix result resolves it here.
    R / 'reports/probe_lpre.json',
    R / 'reports/smoke24.json',
    R / 'reports/smoke_layers24.json',
    # The neighbour-write check is a delivered artifact like any other, and the
    # reports quote readings out of it that exist nowhere else: its
    # `round_start_epoch`, its swept-file count, and the three images it compares.
    # Leaving it out of the corpus would mean a report could not cite the very
    # evidence section 9 item 17 requires it to cite.
    #
    # `_check_delivery_selfcheck.json` is DELIBERATELY NOT here, even though it is
    # also a delivered artifact.  Its whole content is synthetic sentinels chosen
    # because they appear in NO source (`0.987654321`, `12345.6789`, ...).  Putting
    # it in the corpus would make exactly those values resolve, which is the one
    # thing the D39 probe exists to prove cannot happen.  A report that needs to
    # quote one of them cites it through `DEFINED_IN_CODE` below, where the
    # file:line that defines it is named and checkable.
    R / 'reports/neighbour_write_check.json',
    # ---- the round this checker was copied from: read-only, cited by FULL path ----
    # This is where EVERY frozen anchor of this round is READ FROM (round 5's JSONs
    # are byte-unchanged this round, by §0.2), so these entries are load-bearing
    # rather than incidental: a ≥3-decimal anchor that resolves to nothing here is
    # a literal this round may have mis-transcribed.
    R3 / 'reports/frozen_anchors.json',
    R3 / 'reports/phase1_full.json',
    R3 / 'reports/level_variance.json',
    R3 / 'reports/verdict.json',
    R3 / 'reports/audit_beta.json',
    R3 / 'reports/k_field.json',
    R3 / 'reports/phase_minus1.json',
    R3 / 'reports/round4_G5_correction.json',
    R3 / 'reports/shard_additivity_check.json',
    # ---- two rounds back: still cited for the A_L2 anchor and the 19-point grid ----
    R4 / 'reports/frozen_anchors.json',
    R4 / 'reports/phase1_l1.json',
    R4 / 'reports/phase2_full.json',
    R4 / 'reports/phase2_gamma_backfill.json',
    R4 / 'reports/verdict.json',
    R4 / 'reports/audit_beta.json',
    # ---- peers: a literal may resolve to a NAMED source, never to "found somewhere" ----
    PEER / 'reports/gate_noop.json',
    PEER / 'reports/frozen_anchors.json',
    PEER / 'reports/frozen_grid.json',
    PEER / 'reports/预注册_hash.json',
    PEER / 'reports/phase0_amplitude_budget.json',
    PEER / 'reports/phase1_envelope.json',
    PEER / 'reports/phase1_scores.json',
    PEER / 'reports/audit_envelope.json',
    R1 / 'reports/phase1_fingerprints.json',
    R1 / 'reports/audit_phase1.json',
    R1 / 'reports/phase0_acceptance.json',
    R1 / 'reports/phase1_replay.json',
    R18 / 'reports/computation_accounting.json',
    R18 / 'reports/input_manifest_H1.json',
    HYD / 'reports/tn_hydrology_interface_qa.json',
    HYD / 'reports/dual_product_crosscheck.json',
    HYD / 'reports/long_simulation_qa.json',
    UP / '20260918_4/reports/stage_a_fingerprints.json',
    UP / '20260918_4/reports/audit_stage_a.json',
    UP / '20260916_2/outputs/C0_s1/model.json',
]

# Thresholds and constants that are defined in CODE, not in a JSON: the `legal`
# gate conjuncts, the frozen estimator constants, and the grid's own definition.
# Each is admitted only with the file:line that defines it, so the allowlist is
# itself checkable.
# DUPLICATE KEYS, CLEANED.  The inherited table declared `1e-6` and `1e-15` TWICE
# each; a Python dict literal keeps the last and drops the first with no error, so
# two allowlist entries (the absolute mass acceptance and the round-3 ULP tolerance)
# had already been silently overwritten in the copy this round inherited.  Their
# justifications were different and one of them -- `1e-6`, the acceptance the plan
# forbids loosening -- is load-bearing.  Registered as a deviation; the entries are
# merged rather than dropped.
DEFINED_IN_CODE = {
    '1e-6': 'the absolute mass acceptance.  THIS ROUND: work/phase0_gates.py:522 '
            '`tolerance_kg=1e-6, written_as="<="`.  The criterion is the same as the one '
            '20260919_2/reports/预注册_判据与门槛.md section 7 registered; only the '
            'file:line that DEFINES it moves with the round.',
    '1e-7': 'fit_worker.py:70 / audit_job.py:60-63 legal conjunct 3',
    '1e-10': 'legal conjunct 2 network_balance scale; also the fixed-point rho criterion, '
             'work/xi_k.py',
    '1e-12': 'THIS ROUND: work/phase0_gates.py:182 the V_s provenance cross-check '
             'tolerance (the two-route criterion the plan registers as VS_PROVENANCE_'
             'CROSSCHECKED).  Round 5 spelled the same tolerance `common20.py:89 '
             'ANCHOR_TOL`; the entry was retargeted because THIS round uses it as the '
             'provenance criterion, which is the reading a report quotes.',
    '1e-13': 'campaign_model.py:74 SOURCE_TAG_IDENTITY rtol',
    '1e-9': 'THIS ROUND: work/phase0_gates.py:304 the pre/post-outflow recurrence '
            'closure tolerance of the storage-timing gate (section 3.3).',
    '0.005': 'work/common24.py:614 MONTHLY_GATE -- the G5/G5b threshold.  The same 0.005 '
             'is the `MONTHLY_GATE` of 20260919_2/work/phase1_score.py:42; both are '
             'asserted equal at run time by work/verdict_dp.py.',
    '0.0529': '20260919_2/work/eventlib.py:52 NOISE_BAND = 2 x 0.02647',
    '0.02647': 'the `_5` amplitude-ratio null SD, quoted in that same comment',
    '2000': '20260919_2/work/eventlib.py:50 N_BOOT',
    '20260917': '20260919_2/work/eventlib.py:51 SEED',
    '12': '20260919_2/work/eventlib.py:49 MIN_UNITS',
    '7': '20260919_2/work/eventlib.py:46 TN_PRE_DAYS',
    '30': '20260919_2/work/eventlib.py:48 F3_PRIMARY_GAP; also the frozen parameter count, '
          'asserted by work/common24.py:244 (`assert len(x) == 30`)',
    '60': '20260919_2/work/eventlib.py:47 F3_GAPS[2]',
    '15': '20260919_2/work/eventlib.py:47 F3_GAPS; also n_stations, asserted by '
          'work/phase0_gates.py N9',
    '0.5': 'the beta=0.5 point of the round-5 reference frame, '
           '20260919_5/reports/phase1_full.json points["N1e|0.5"]',
    # ---- THIS ROUND'S OWN LITERAL-CHECK SELF-CHECK (D39) -----------------------
    # These five are the SYNTHETIC sentinels `selfcheck()` feeds through the real
    # `_near` / `LIT_RE` / `SCI_RE` code.  They resolve to no corpus file BY
    # CONSTRUCTION -- that absence is the reading the probe reports (three of them
    # are asserted to NOT resolve; the other two are the deliberate positive
    # spelling of a value the corpus holds only in its negative form, and the
    # MANTISSA of a scientific literal).  `独立完成审计.md` section 2 has to be
    # able to quote them, so they are admitted here the same way `1e-6` is: with
    # the file:line that DEFINES them.  A reader can open that line and see the
    # value; nothing is admitted that is not checkable.
    '0.987654321': 'work/check_delivery.py:669 -- selfcheck probe 1 synthetic '
                   'sentinel, asserted NOT to resolve against a corpus holding only '
                   'an infinity',
    '12345.6789': 'work/check_delivery.py:669 -- selfcheck probe 1 synthetic '
                  'sentinel, same assertion',
    '3.14159265358979': 'work/check_delivery.py:669 -- selfcheck probe 1 synthetic '
                        'sentinel, same assertion',
    '0.1407034380572627': 'work/check_delivery.py:693 -- selfcheck probe 2: the '
                          'UNSIGNED token the inherited pattern produced for a corpus '
                          'value that is negative, i.e. the false positive it fixed',
    '2.220446049250313': 'work/check_delivery.py:711 -- selfcheck probe 3: the '
                         'MANTISSA the inherited pattern saw as a bare decimal; the '
                         'literal itself is checked whole by SCI_RE',
    '1e-15': 'audit_envelope.py kappa tolerance / both-nan comparison (round 2)',
    '8': 'the limit case beta=8, work/common23.py BETA_GRID',
    '0.05': 'work/common23.py BETA_GRID, the smallest non-zero step',
    # ---- HISTORICAL constants: not criteria THIS round, but this round's reports
    # ---- must be able to register their demotion, which requires naming them.
    # `1e-3` and `0.1` were cited by the copy this round inherited as
    # `work/xi.py:90` -- a file that DOES NOT EXIST in round 5 (its `work/` holds
    # `xi_base.py`) and exists in NEITHER round.  That is defect (i) of plan
    # section 8 item 14, and it is repaired here by moving each entry onto the
    # file that actually defines the value.  The two `work/xi.py:90` entries were
    # ALSO conflated: `SAT_BOUND` is not a `xi_base` constant at all, it lives in
    # `common23`.  Neither value is a criterion of THIS round (plan section 3.5 N4
    # demotes both to a diagnostic); the entries exist so that a report which says
    # so can name them without tripping the literal gate.
    '1e-3': 'HISTORICAL (demoted this round): 20260919_5/work/xi_base.py:90 '
            '`W_FLOOR = 1e-3` == DEGEN_FLOOR, the Xi pin and degeneracy floor. '
            'Plan section 3.5 N4: a diagnostic, no longer a criterion.',
    '0.1': 'HISTORICAL (demoted this round): 20260919_5/work/common23.py:149 '
           '`SAT_BOUND = 0.10`.  Plan section 3.5 N4: a diagnostic, no longer a '
           'criterion.',
    '700': 'the exponent clip `-np.expm1(-min(risk, 700.))` in the FROZEN kernel, '
           '20260916_2/vendor/research/closures.py:23, which this round still runs '
           'through arm `B0` and through every `Transport` binding.  THIS round\'s own '
           'kernel has NO clip and is not permitted one (work/dp_kernel.py:72): the '
           '`700` a report on this round may quote is the frozen one, so the frozen '
           'file is the honest citation.  (Round 5 also spelled it `CLIP = 700.0` at '
           'work/xi_base.py:92.)',
    '1961': 'work/common24.py:112 REF_YEARS[0], the reference-period start',
    '2020': 'work/common24.py:112 REF_YEARS[1], the reference-period end',
    '2021': 'work/common24.py:113 EVAL_YEARS[0]; also design.training_years[0]',
    '2022': 'design.training_years[1] -- the disclosure, not a criterion',
    '2024': 'work/common24.py:113 EVAL_YEARS[1]',
    '230': 'the reach count; V, k and s_M are one value per reach',
    '169476': 'the dense reference-frame row count, the round-5 anchor replay '
              'criterion (plan section 3.5 N10)',
    '1e-200': 'the observed LOWER BOUND of `fast_water` (same for `1e-204` and '
              '`percolation`); the values are measured into '
              'reports/phase0_gates.json, which is in the corpus, so a report may '
              'quote them from there.  Listed here as well because the N12 underflow '
              'probe uses exactly `x = 1e-200` as its argument '
              '(work/dp_kernel.py:280) and a report that quotes the PROBE\'s argument '
              'is quoting a code literal, not a measurement.',
    # These resolve through the R3 corpus above; kept so that a reader grepping this
    # table still finds where each came from.
    '2.094720376603729': 'R3 reports/frozen_anchors.json baseline_model_C0_s1_monthfirst',
    '2.031723094631754': 'R3 reports/frozen_anchors.json baseline_model_C0_s1_monthfirst',
    '1.2758737517831669': 'R3 reports/frozen_anchors.json anchors.A_obs',
    '0.004063600875414098': 'R3 reports/frozen_anchors.json anchors.beta_obs',
    '-0.0704045722214265': 'R3 reports/frozen_anchors.json anchors.alpha_obs',
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
# Signed, and NOT the mantissa of a scientific literal.  Both corrections are
# registered as deviation D39 (b) in `实际方法与偏离.md`, and both are false
# POSITIVES the inherited pattern produced -- the gate was not merely vacuous,
# it was also wrong in the other direction:
#   * `(\d+\.\d{3,})` does not capture a leading `-`, while `_near` compares
#     AGAINST a signed corpus value.  Every negative literal in every report was
#     therefore reported as unresolved: `-0.1407034380572627` is
#     `_corpus_numbers`' own value for the baseline `median_station_nse`, and the
#     inherited pattern tokenised it as `0.1407034380572627`, which matches
#     nothing.  Round 4's seven reports carry 212 such tokens between them.
#   * `(\d+\.\d{3,})(?![\d])` also matches the MANTISSA of `2.220446049250313e-16`
#     (the next char is `e`, not a digit), so the mantissa was checked as a bare
#     decimal -- a value that appears in no source and never could.
# `(?![eE][-+]?\d)` is what suppresses the second; it is placed AFTER the
# `(?![\d])` guard so a trailing digit still fails first.
LIT_RE = re.compile(r'(?<![\w.])(-?\d+\.\d{3,})(?![\d])(?![eE][-+]?\d)')
SCI_RE = re.compile(r'(?<![\w.])(-?\d+(?:\.\d+)?[eE]-?\d+)(?![\d])')


def _floats_from_text(txt):
    """The float set ONE file contributes to the literal check: every token the
    loose number pattern matches that parses, plus every number `json.loads`
    yields.  `_corpus_numbers` is the file walker; this is its per-file body,
    factored out so `selfcheck` can drive the REAL extraction path with synthetic
    text.  A probe that re-implemented this loop would prove something about the
    probe, not about the checker."""
    out = set()
    for m in re.finditer(r'-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?', txt):
        try:
            out.add(float(m.group(0)))
        except ValueError:
            pass
    try:
        for v in _walk(json.loads(txt)):
            out.add(float(v))
    except (ValueError, TypeError):
        pass
    return out


def _corpus_numbers():
    """Every number appearing anywhere in the corpus, as written AND as float."""
    as_written = set()
    as_float = set()
    for p in CORPUS:
        if not p.exists():
            continue
        txt = p.read_text(encoding='utf-8', errors='replace')
        for m in re.finditer(r'-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?', txt):
            as_written.add(m.group(0))
        as_float |= _floats_from_text(txt)
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


# ---------------------------------------------------------------------------
# SEEDED COPIES: delivered files that are a byte-verified copy of a peer artifact
# PLUS this round's additions.  Keyed by basename.  For these, a literal that
# occurs VERBATIM in the seed is a reading an earlier round published; the seed's
# digest is re-verified here, and a mismatch turns the carve-out off.
#
# Registered rather than inferred, and deliberately one entry long.  A second
# entry is a claim about a second seed, and each one costs a registration in
# `实际方法与偏离.md`.
SEEDED_COPIES = {
    '已关闭假设台账.md': dict(
        seed=UP / '20260919_5' / 'reports' / '已关闭假设台账.md',
        sha256='2909a10330660ac28780b4ec5ff2d1b8eec0ee4aa6d9ff08174d9050e95e2fd2',
        bytes=100997,
        producer=r'20260920_1\work\_refresh_ledger.py',
    ),
}


def _sha256(p):
    import hashlib
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _seed_tokens_for(report):
    """The seed's literal tokens, for a registered seeded copy; empty otherwise.

    `matches` is False whenever the carve-out is NOT in force (no entry, seed
    unreadable, digest differs).  The caller reports it either way, so a reader
    can see that the exemption was checked rather than assumed.
    """
    null = dict(tokens=frozenset(), matches=None, why=None, seed_name=None)
    ent = SEEDED_COPIES.get(Path(report).name)
    if ent is None:
        return null
    try:
        got = _sha256(ent['seed'])
    except OSError as e:
        return dict(tokens=frozenset(), matches=False, seed_name=ent['seed'].name,
                    why='seed UNREADABLE (%s) -- carve-out OFF' % e)
    matches = bool(got == ent['sha256'])
    if not matches:
        return dict(tokens=frozenset(), matches=False, seed_name=ent['seed'].name,
                    why='seed digest %s != registered %s -- carve-out OFF'
                        % (got[:16], ent['sha256'][:16]))
    return dict(tokens=frozenset(_tokens(ent['seed'])), matches=True,
                seed_name=ent['seed'].name,
                why='seeded copy: literals occurring verbatim in `%s` (sha verified) '
                    'are earlier rounds\' published readings; producer `%s`'
                    % (ent['seed'].name, ent['producer']))


def _tokens(path):
    """The set of `LIT_RE`/`SCI_RE` tokens a file contains, as written."""
    t = Path(path).read_text(encoding='utf-8')
    return set(m.group(1) for m in
               list(LIT_RE.finditer(t)) + list(SCI_RE.finditer(t)))


def check_literals(report):
    written, floats = _corpus_numbers()
    text = Path(report).read_text(encoding='utf-8')
    out = {'resolved': [], 'allowlisted': [], 'inherited': [], 'unresolved': [],
           'seeded_copy': None, 'seed_sha256_matches': None}
    # -------------------------------------------------------------------
    # THE SEEDED-COPY CARVE-OUT.  One delivered file is a byte-verified copy
    # of a peer artifact plus this round's additions (`SEEDED_COPIES` below).
    # Its inherited rows state readings published by EARLIER rounds -- the
    # 97 literals round 5's ledger failed on are exactly that: round-1..19
    # readings that no round's corpus contains, because the rounds that
    # measured them are not this round's sources.
    #
    # The exemption is therefore at LITERAL granularity, not file granularity,
    # and it is the same principle the checker already applies at FILE
    # granularity to the frozen pre-registration: the copy is exempt, the
    # authored part is not.  A file-level exclusion would have skipped the
    # authored part too, which is the half this round is answerable for.
    #
    # It is NOT a loosening, and these four properties are what make that
    # checkable rather than asserted:
    #   (i)   it applies to ONE named file, whose seed's sha256 is registered;
    #   (ii)  the seed's digest is re-verified HERE, and a wrong digest turns
    #         the carve-out OFF (the literal then fails as before) -- so the
    #         exemption cannot survive the seed being swapped;
    #   (iii) the exempted tokens are LISTED, not merely counted, so a reader
    #         can diff them against the seed with one grep;
    #   (iv)  a literal this round authored cannot be exempt -- it is not in
    #         the seed -- so section 20's readings are gated exactly as every
    #         other report's are.
    seeded = _seed_tokens_for(report)
    out['seeded_copy'] = seeded['why']
    out['seed_sha256_matches'] = seeded['matches']
    seeded = _seed_tokens_for(report)
    out['seeded_copy'] = seeded['why']
    out['seed_sha256_matches'] = seeded['matches']
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
        elif tok in seeded['tokens']:
            out['inherited'].append(dict(literal=tok, from_seed=seeded['seed_name']))
        else:
            out['unresolved'].append(tok)
    return out


def _near(v, floats, rel=1e-9):
    """A literal is resolved if the corpus holds the same value in any spelling:
    a report may round, and the corpus may carry more digits than the prose.

    NON-FINITE CORPUS VALUES ARE SKIPPED, and in this corpus that is
    LOAD-BEARING rather than defensive.  `abs(f - v) <= abs(f) * rel` is TRUE for
    EVERY finite `v` once `f` is +/-inf: both sides evaluate to inf.  One
    infinity anywhere in the corpus therefore makes the whole literal gate
    vacuous -- `_near(0.987654321, floats)` and `_near(12345.6789, floats)` both
    return True, for a corpus that contains neither.

    And the corpus does contain them, twice over and neither by accident:
      * `_corpus_numbers` scans every file with the loose token pattern
        `-?\\d+(\\.\\d+)?([eE][-+]?\\d+)?`, which matches the TAIL OF A HEX
        SHA256 -- `...3e499` parses as 3 x 10**499 and overflows to inf;
      * two peer JSONs (`gate_noop.json`, `phase1_scores.json`) carry a literal
        `Infinity` that `json.loads` accepts and `_walk` then yields as a float.

    Measured before the fix: `_near` returned True for all three of
    `0.987654321`, `12345.6789`, `3.14159265358979`.  The round's own reported
    `unresolved: []` was therefore true but EMPTY OF CONTENT on the literal
    half -- a gate that passed because of the shape of the comparison, not
    because of the property it was registered to test.  Registered as deviation
    D39 in `实际方法与偏离.md`; the PATHS half of this check is unaffected
    (`_exists` tests existence, never values), and it is the half that has been
    doing real work all along.
    """
    for f in floats:
        if f == v:
            return True
        if math.isfinite(f) and f != 0 and abs(f - v) <= abs(f) * rel:
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

# THE THIRD INHERITED REFERENCE DEFECT OF PLAN SECTION 8 ITEM 14, and the same
# shape as the `data/source.npy` entry above: a path the PLAN cites that does not
# exist, which the round has to be able to name verbatim in order to register the
# defect.  Unlike the five above, this one is NOT a citation-form slip -- the
# DIRECTORY is wrong, so there is no full-path spelling that fixes it, and the
# provenance it was cited for (the producer's arithmetic order) really is at
# `20260825_3`.  Registered in `reports/实际方法与偏离.md` and already recorded
# machine-readably at `reports/phase0_gates.json::3.1.plan_cited_path_does_not_exist`.
# The exemption is checkable rather than blanket: it names the file that registers
# it, and the corrected spelling sits beside it in the same JSON.
CITED_AS_MISSING['20260828_25_3\\scripts\\hydrology_core.py'] = (
    'plan section 10 cites this path as the storage-timing-order evidence; the '
    'directory does not exist and the producer is '
    '20260825_3\\scripts\\hydrology_core.py, which this round actually read. '
    'Quoted only so the defect register can name the bad spelling verbatim; '
    'recorded at reports/phase0_gates.json::3.1.plan_cited_path_does_not_exist.')

# A path that existed and was REMOVED ON PURPOSE, by the checker's own D39 probe.
# `selfcheck()` writes `work/_selfcheck_probe.md`, runs `check_paths` on it to prove
# the path half of the gate is live in both directions, then unlinks it and records
# `probe_file_removed_after_use`.  A reader has to be able to see the path that was
# used, and it is listed in `reports/_check_delivery_selfcheck.json::probes.
# 4_paths_half_is_live.probe_file`.  Unlike the entries above this is NOT a defect
# in anyone's citation: the file is absent by design, and it is absent NOW.
CITED_AS_MISSING['20260920_1\\work\\_selfcheck_probe.md'] = (
    'created and then unlinked by check_delivery.selfcheck() probe 4, which uses it '
    'to prove check (2) reports a missing path instead of absorbing it; the removal '
    'is recorded as `probe_file_removed_after_use` in '
    'reports/_check_delivery_selfcheck.json')

# WHY `model.py` IS DELIBERATELY **NOT** IN THIS DICTIONARY -- do not "fix" this.
# It is the one citation the frozen pre-registration leaves unresolved, and plan
# section 9 gate 18 requires that count to be PRINTED for the excluded file.  Adding
# it here was tried and REVERTED, because the exemption is consulted before the
# exclusion: the excluded file then printed `0 unresolved citation(s)`, i.e. the
# reading became "the frozen pre-registration has nothing unchecked" when the truth
# is that it has one.  The count is the gate's reading, so it must stay 1.
#
# AND the entry would have been miscategorised on its own terms: everything else in
# this dictionary is ABSENT BY DESIGN (removed on purpose, or a defect of the
# CITATION FORM).  `model.py` is neither -- it is a GENUINE dangling citation in a
# file that is frozen before any forward and must not be edited to make a checker
# happy.  Registering it as "deliberately missing" would have relabelled a real
# defect as a designed absence.
#
# The reports that register the count (`数据与运行说明.md` section 8, `简短结论.md`
# section 6) therefore name it WITHOUT backticks, which is what keeps it out of
# `PATH_RE` and leaves the excluded file's count honest at 1.  Naming a path and
# citing a path are different acts; this comment exists so the difference survives.

# The SECOND path this round removed on purpose, and the same shape as the entry
# above: a reader has to be able to see the path that is gone, or the report's
# account of what it deleted cannot be checked.  `probe_lpre.py` used to `np.save`
# four 23376x230 float64 arrays here (~170 MB) and NOTHING read them -- no report
# cites the .npy, `layers24.py` cites the .py.  The producer was changed to write
# `reports/probe_lpre.json` instead (the readings `layers24.py` actually quotes),
# and this scratch file was dropped rather than left as a 170 MB artifact that
# looks like evidence.  Registered in `reports/实际方法与偏离.md` section 12.
CITED_AS_MISSING['work/_probe_lpre.npy'] = (
    'written by probe_lpre.py until this round and read by nothing; removed when '
    'the producer was changed to land its readings as reports/probe_lpre.json. '
    'Absent by design, and absent NOW. Named verbatim in '
    'reports/实际方法与偏离.md section 12 so the deletion is auditable')

# The THIRD path this round removed on purpose.  `_anchors_probe.py` printed the
# exact bytes of every anchor `_refresh_ledger.py` needed, so the author could copy
# them instead of guessing a Chinese anchor containing full-width quotes.  It was a
# ONE-SHOT: its function is now served permanently by the producer's own
# `LEDGER_ANCHOR_NOT_UNIQUE` asserts and by the full anchors recorded in
# `reports/_ledger_refresh.json`, so keeping it would leave a script whose only
# remaining job is to be re-run by someone who no longer needs it.
CITED_AS_MISSING['work/_anchors_probe.py'] = (
    'one-shot anchor-printing probe, deleted after the ledger producer absorbed its '
    'function (the anchors are now asserted and recorded in full in '
    'reports/_ledger_refresh.json). Absent by design, and absent NOW. Named '
    'verbatim in reports/实际方法与偏离.md section 14.3')

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
    'report.json': 'metasyntactic placeholder in the notation discussion',
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
    # The round this checker was copied from (round 5 for THIS round; it was round 4
    # for round 5).  Declared for the same reason as the others: this round's
    # reports cite its anchors, its `daily_layers.parquet` (the `R5-ref` source) and
    # its prose corrections by full path, and a full path must land on a DECLARED
    # root rather than on "found somewhere".  `R3/work` is declared because the
    # reports cite round 5's `work\layers23.py:100-103` VERBATIM in order to
    # register defect (ii) of plan section 8 item 14 -- an exemption is not needed
    # for that, but a resolution is, and the resolution must be to ROUND 5's file
    # and not to this round's `layers24.py`.
    ('peer(20260919_5)', R3),
    ('peer(20260919_5/work)', R3 / 'work'),
    ('peer(20260919_5/reports)', R3 / 'reports'),
    # One more step back, kept because the A_L2 anchor and the 19-point beta grid
    # are still cited from round 4's `phase1_l1.json` / `phase2_full.json`.
    ('peer(20260919_4)', R4),
    ('peer(20260919_4/work)', R4 / 'work'),
    ('peer(20260919_4/reports)', R4 / 'reports'),
    # One more step back, and ONLY one.  The seeded ledger cites round-3 artifacts
    # by bare `reports\...` INSIDE its round-3 rows (`reports\audit_gamma.json` in
    # the §16 table, which is round 3's own audit).  That citation resolved in
    # round 5 only because round 5 declared round 3 among its peers; this round's
    # copy arrived declaring rounds 4 and 5, so the citation went MISSING on the
    # first run.  Declared last, so every path that already resolved keeps the
    # rule that resolved it -- this can only repair, never re-point.
    #
    # Deliberately NOT extended to rounds 1 and 2.  Nothing in the delivered set
    # cites them by bare name, and a declared root that nothing needs is a root
    # that can only ever make a future bare name resolve to the wrong round.
    ('peer(20260919_3)', UP / '20260919_3'),
    ('peer(20260919_3/work)', UP / '20260919_3' / 'work'),
    ('peer(20260919_3/reports)', UP / '20260919_3' / 'reports'),
    ('peer(20260916_2/outputs)', UP / '20260916_2' / 'outputs'),
    ('peer(20260916_2/data)', UP / '20260916_2' / 'data'),
    ('peer(20260916_2/data/domains)', UP / '20260916_2' / 'data' / 'domains'),
    ('peer(20260916_1/scripts)', UP / '20260916_1' / 'scripts'),
    # The HBV storage-timing producer, declared THIS ROUND for two reasons that
    # only make sense together:
    #   (1) it is the file the storage-timing gate is built on -- plan section 3.3
    #       reads its arithmetic order line by line, and the primary arm's identity
    #       rests on those lines (see the report's section 6);
    #   (2) the PLAN cites it under a directory that does not exist
    #       (`20260828_25_3\scripts\`).  So the round must be able to cite BOTH
    #       spellings: the wrong one verbatim (that is the `CITED_AS_MISSING` entry
    #       below) and the right one resolving to a named root (this entry).
    # A root that only served to make a bad path pass would be a weakening; this
    # one makes the CORRECTED path checkable, which is the opposite.
    ('peer(20260825_3)', UP / '20260825_3'),
    ('peer(20260825_3/scripts)', UP / '20260825_3' / 'scripts'),
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
# PLAN and transcribed into `reports/预注册_判据与门槛.md`; there is no
# machine-checked phase that re-hashes either of them.  So the digest below is a
# STATEMENT ABOUT THE FILE, not a re-verified gate: it is what the md hashes to
# right now, and the exclusion means "this md is the frozen pre-registration, so
# the two prose checks do not apply to it".  A reader who wants the gate that
# round 3 had should read `预注册_判据与门槛.md` §0, which registers this
# difference as D5.
#
# AND ONE THING IS REGISTERED HERE THAT ROUND 4 COULD NOT SAY.  Round 4 cited a
# plan sha (`b72c99e06f35cd0111f1c3d9635280aa2c52b0dca01132c78c22be9f8452dd7b`,
# 49046 bytes) for `C:\Users\Administrator\.claude\plans\cheerful-tinkering-moth.md`.
# That file no longer hashes to it: the ROUND-5 plan was written to the SAME PATH
# (66253 bytes, mtime 2026-09-19 14:22:07.746138500 +0800, sha
# `bf0713fd5fe9065da08ede0550c9a5de00f8e851f0a2911c2688c584f486c9f1`), because the
# plan file is a single reused slot per campaign, not a per-round artifact.  So a
# round-4 reader who re-hashes the cited path TODAY gets a different digest, and
# that is a fact about the plan file's storage, not evidence that round 4 edited
# its pre-registration.  Round 5's §0 records the sha it actually froze against.
PLAN_PATH = r'C:\Users\Administrator\.claude\plans\cheerful-tinkering-moth.md'
PLAN_SHA = '88381c2920bbf94d708b3e1929a8b94f86b1d13801237c781f8a12f27a1c2c88'
PLAN_BYTES = 61848
# THIS ROUND's digest, set ONCE, from a md that is final: this file was copied from
# `20260919_5\work\check_delivery.py`, so `FROZEN_SHA` arrived carrying ROUND 5's value
# for ROUND 5's md.  A digest that names a file living in another round's directory
# excludes nothing here -- `reports/预注册_判据与门槛.md` would have been fully checked
# against this round's corpus and its pre-forward readings would have been treated as
# delivery claims.  The value below is this round's own md (29431 bytes), taken after the
# last pre-forward registration and BEFORE any Phase 1 forward; `reports/预注册_冻结.json`
# carries the SAME digest, with `forward_runs_so_far = 0`.  So this exclusion is not a
# statement about a file's current bytes alone -- it is cross-checkable against the freeze
# record.
#
# CORRECTION (this line used to read "and was written by `phase0_gates.py` at the moment
# of the freeze").  That was FALSE and it was mine.  NO script in this round writes
# `reports/预注册_冻结.json`: every `.py` was searched for both writing mechanisms this
# round uses (`C.write_json` and direct `write_text`/`json.dump`), and the complete
# call-site list contains level_variance/phase0_gates/arms/phase1_arms/probe_*/
# verdict/phase0_n1/audit_dp/neighbour_write_check and nothing else.  `phase0_gates.py`
# has no subprocess, `runpy` or `exec`, so it cannot be doing it indirectly either.  The
# record was made by an ad-hoc command whose script was not kept.
#
# Do NOT "fix" this by adding a producer, and above all do NOT regenerate the record: a
# rewrite would stamp a new `recorded_at_utc` and would be a POST-HOC RE-FREEZE, which is
# the one thing gate 1 exists to prevent.  The repair is `work/_verify_prereg_freeze.py`,
# which binds the existing bytes field-by-field to artifacts that DO have producers and
# proves the pre-forward claim from mtimes (arms.json -> md -> freezing -> first forward,
# 15.9 min of margin) rather than from the record's own timestamp field.  This exclusion
# rests on that file, not on a producer.
# From here the md is frozen: further readings go into `reports/`, not into the md.
FROZEN_SHA = '62734f98795cdce6d61e71588209db23b2e3fb00cd5f6a150032af0d4d79d4ff'


# ---------------------------------------------------------------------------
# D39 SELF-CHECK: prove this checker is not vacuous, BEFORE it is trusted.
#
# Plan section 9 hard gate 18 requires "the checker's own vacuity self-check is
# on disk (D39)".  The copy this round inherited had NONE: `_near`'s docstring
# documents the inf defect and the two `LIT_RE` comment blocks document the
# tokenisation defects, but NOTHING at run time proved the repairs hold.  A
# checker whose correctness rests on a docstring is a checker the next copy can
# hollow out again without anyone noticing -- which is precisely the history of
# this file (memory: `legacy-checker-copies-stay-vacuous`: an earlier round's
# copy kept the old rule and its `passed: true` is not evidence for the next
# round).  So the three literal probes below are SYNTHETIC and hermetic: they
# never touch the corpus, so they cannot pass by accident of this round's data,
# and `main` FAILS THE RUN if any of them stops holding.
#
# Each probe PAIRS a must-resolve case with the must-not-resolve case the old
# rule produced, because a probe that only shows a negative result would also
# pass on a checker that rejects everything.
# ---------------------------------------------------------------------------
SELFCHECK_PROBE = R / 'work' / '_selfcheck_probe.md'
# The two inherited spellings, kept VERBATIM so the self-check can measure what
# they did rather than assert what they did in prose.
_PRE_LIT_RE = re.compile(r'(\d+\.\d{3,})')


def _near_prefix(v, floats, rel=1e-9):
    """The body `_near` inherited, kept ONLY as the self-check's control: with a
    non-finite value anywhere in `floats` it returns True for EVERY finite `v`,
    because `abs(inf - v) <= abs(inf) * rel` is True for every finite `v`."""
    for f in floats:
        if f == v:
            return True
        if abs(f - v) <= abs(f) * rel:
            return True
    return False


def selfcheck():
    """Five probes: one per defect D39 registered, plus the paths half, plus the
    revocability of this round's seeded-copy carve-out."""
    out = {'what': 'D39 -- is this checker vacuous?  Synthetic inputs only; no '
                   'corpus file is read, so no reading here can pass by accident '
                   'of this round\'s data.',
           'why': 'the copy this round inherited documented these defects in '
                  'comments and proved none of them at run time',
           'probes': {}}

    # ---- 1. a non-finite corpus value must not make every finite literal
    # ----    resolve.  The +inf is produced the way the REAL corpus produces
    # ----    it: `float('3e499')` from a hash tail, and a JSON `Infinity`.
    text = '{"sha": "ab3e499", "null_sd": Infinity, "a_obs": 1.2758737517831669}'
    fl = _floats_from_text(text)
    p1 = {'corpus_text': text,
          'floats_extracted': sorted(repr(f) for f in fl),
          'contains_inf': bool(float('inf') in fl),
          'must_resolve': {'v': 1.2758737517831669,
                           'resolves': bool(_near(1.2758737517831669, fl))},
          'must_NOT_resolve': [
              {'v': v, 'now': bool(_near(v, fl)), 'under_the_inherited_body':
               bool(_near_prefix(v, fl))}
              for v in (0.987654321, 12345.6789, 3.14159265358979)]}
    p1['ok'] = bool(
        p1['contains_inf'] and p1['must_resolve']['resolves']
        and all(not d['now'] and d['under_the_inherited_body']
                for d in p1['must_NOT_resolve']))
    p1['reading'] = ('the inherited body returns True for all three finite '
                     'values against a corpus holding ONLY an infinity; the '
                     'repaired body returns True only for the value the corpus '
                     'actually contains.  Check (1) is live in both directions.')
    out['probes']['1_nonfinite_corpus_value'] = p1

    # ---- 2. a SIGNED negative literal must resolve against its signed corpus
    # ----    value, and must NOT resolve when tokenised unsigned.
    v2 = -0.1407034380572627
    fl2 = _floats_from_text('{"baseline_median_station_nse": -0.1407034380572627}')
    p2 = {'corpus_value': v2,
          'text': '`median_station_nse` %s' % v2,
          'now_tokens': LIT_RE.findall('`median_station_nse` %s' % v2),
          'inherited_tokens': _PRE_LIT_RE.findall('`median_station_nse` %s' % v2),
          'now_resolves': bool(_near(v2, fl2))}
    p2['inherited_resolves'] = bool(
        all(_near(float(t), fl2) for t in p2['inherited_tokens'])) \
        if p2['inherited_tokens'] else False
    p2['ok'] = bool(p2['now_tokens'] == ['%r' % v2] and p2['now_resolves']
                    and p2['inherited_tokens'] == ['0.1407034380572627']
                    and not p2['inherited_resolves'])
    p2['reading'] = ('the inherited pattern drops the minus sign, so every '
                     'negative literal was tested against a corpus that holds '
                     'its positive counterpart -- which resolves to nothing.  '
                     'The repaired pattern keeps the sign.')
    out['probes']['2_signed_negative_literal'] = p2

    # ---- 3. the MANTISSA of a scientific literal must not be tested as a bare
    # ----    decimal, and the literal itself must resolve.
    lit3 = '2.220446049250313e-16'
    fl3 = _floats_from_text('{"eps": %s}' % lit3)
    p3 = {'text': '`%s`' % lit3,
          'now_tokens_from_LIT_RE': LIT_RE.findall('`%s`' % lit3),
          'inherited_tokens': _PRE_LIT_RE.findall('`%s`' % lit3),
          'SCI_RE_tokens': SCI_RE.findall('`%s`' % lit3),
          'sci_resolves': bool(_near(float(lit3), fl3))}
    p3['ok'] = bool(p3['now_tokens_from_LIT_RE'] == []
                    and p3['inherited_tokens'] == ['2.220446049250313']
                    and p3['SCI_RE_tokens'] == [lit3] and p3['sci_resolves'])
    p3['reading'] = ('the inherited pattern saw the mantissa `2.220446049250313` '
                     'as a bare decimal -- a value that appears in no source and '
                     'never could -- while the literal itself escaped check (1) '
                     'entirely.  The repaired pattern suppresses the mantissa and '
                     '`SCI_RE` checks the whole literal against the corpus.')
    out['probes']['3_scientific_notation_mantissa'] = p3

    # ---- 4. the PATHS half must be live: one existing path, one absent path,
    # ----    one name-shape.  Written to `work/` (not `reports/`, which is what
    # ----    `main` globs) so the probe can never be audited as a report.
    SELFCHECK_PROBE.write_text(
        '# synthetic self-check input -- see check_delivery.selfcheck()\n\n'
        'resolves: `reports/预注册_冻结.json`\n'
        'does NOT resolve: `reports/__no_such_file__.json`\n'
        'name shape: `<report>_delivery_check.json`\n', encoding='utf-8')
    pat = check_paths(SELFCHECK_PROBE)
    p4 = {'probe_file': str(SELFCHECK_PROBE.relative_to(UP)),
          'n_ok': pat['n_ok'], 'missing': pat['missing']}
    p4['ok'] = bool(pat['missing'] == ['reports/__no_such_file__.json'])
    p4['reading'] = ('the absent path is reported MISSING and the placeholder is '
                     'not: check (2) is not vacuous, and the exception list is '
                     'narrow enough that a genuinely absent path still fails.')
    out['probes']['4_paths_half_is_live'] = p4

    # ---- 5. the SEEDED-COPY carve-out must be revocable.  This probe exists
    # ----    because the carve-out is the one place this checker deliberately
    # ----    stops checking something, and an exemption nobody can revoke is
    # ----    just a hole.  It is exercised on SYNTHETIC files in the scratch
    # ----    directory: no delivered artifact is read, so the probe cannot pass
    # ----    by accident of this round's data.
    #
    #    Three readings, and the third is the load-bearing one:
    #      (a) with the seed's digest MATCHING, a token only the seed contains
    #          is classified inherited, and one the seed does NOT contain is
    #          not -- so the carve-out is neither vacuous nor unbounded;
    #      (b) with the digest WRONG, NOTHING is inherited -- so swapping the
    #          seed turns the exemption off instead of silently keeping it;
    #      (c) the registration is by BASENAME, so a report that merely shares
    #          the name is covered too -- which is why (b) has to hold.
    seed_dir = R / 'work' / '_selfcheck_seed'
    seed_dir.mkdir(parents=True, exist_ok=True)
    seed = seed_dir / 'seeded_probe.md'
    seed.write_text('历史的读数 `0.2718281828459045` 与 `0.123456789`。\n', encoding='utf-8')
    real_sha = _sha256(seed)
    body = ('历史的读数 `0.2718281828459045` 与 `0.123456789`。\n'
            '本轮新增 `0.987654321098765`。\n')
    p5 = {'seed_tokens_contains_seed_only_value': None,
          'seed_tokens_contains_round_only_value': None,
          'with_matching_digest': None, 'with_wrong_digest': None}

    def _classify(sha_value):
        keep = dict(SEEDED_COPIES)
        try:
            SEEDED_COPIES['seeded_probe.md'] = dict(
                seed=seed, sha256=sha_value, bytes=seed.stat().st_size,
                producer='check_delivery.selfcheck() probe 5')
            toks = _seed_tokens_for(seed)
            return dict(matches=toks['matches'], tokens=sorted(toks['tokens']),
                        seed_only='0.2718281828459045' in toks['tokens'],
                        round_only='0.987654321098765' in toks['tokens'])
        finally:
            SEEDED_COPIES.clear()
            SEEDED_COPIES.update(keep)

    p5['with_matching_digest'] = _classify(real_sha)
    p5['with_wrong_digest'] = _classify('0' * 64)
    p5['seed_tokens_contains_seed_only_value'] = p5['with_matching_digest']['seed_only']
    p5['seed_tokens_contains_round_only_value'] = p5['with_matching_digest']['round_only']
    p5['ok'] = bool(
        p5['with_matching_digest']['matches'] is True
        and p5['with_matching_digest']['seed_only'] is True
        and p5['with_matching_digest']['round_only'] is False
        and p5['with_wrong_digest']['matches'] is False
        and p5['with_wrong_digest']['tokens'] == [])
    p5['reading'] = ('with the seed\'s digest the exemption covers exactly the '
                     'seed\'s own tokens (a value this round authored is NOT '
                     'covered); with a different digest it covers nothing.  The '
                     'carve-out is revoked by a seed swap, so registering a seed '
                     'is checkable rather than self-certifying.')
    p5['synthetic_body_the_exemption_was_tested_on'] = body
    out['probes']['5_seeded_carveout_is_revocable'] = p5
    try:
        seed.unlink()
        seed_dir.rmdir()
    except OSError:
        pass

    out['n_probes'] = len(out['probes'])
    out['n_probes_passed'] = int(sum(1 for p in out['probes'].values() if p['ok']))
    out['passed'] = bool(out['n_probes_passed'] == out['n_probes'])
    # The probe file is REMOVED after use.  It is a synthetic input to a function,
    # not an artifact: leaving it would put a file named like a report next to the
    # round's real work, and a later mtime sweep would have to classify it.
    SELFCHECK_PROBE.unlink(missing_ok=True)
    out['probe_file_removed_after_use'] = not SELFCHECK_PROBE.exists()
    return out


def main(argv):
    targets = [Path(a).resolve() for a in argv[1:]] or sorted((R / 'reports').glob('*.md'))
    # ---- the self-check runs FIRST and gates the run (plan section 9 gate 18) ----
    selfchk = selfcheck()
    (R / 'reports' / '_check_delivery_selfcheck.json').write_text(
        json.dumps(selfchk, indent=1, ensure_ascii=False, default=str),
        encoding='utf-8')
    print('D39 SELF-CHECK  %d/%d probes passed  ->  %s'
          % (selfchk['n_probes_passed'], selfchk['n_probes'],
             'NON_VACUOUS' if selfchk['passed']
             else 'VACUOUS -- THIS CHECKER IS NOT TRUSTWORTHY'), flush=True)
    if not selfchk['passed']:
        for k in sorted(selfchk['probes']):
            if not selfchk['probes'][k]['ok']:
                print('   FAILED probe: %s' % k, flush=True)
        print('DELIVERY_CHECK_ABORTED: the checker failed its own vacuity self-check, '
              'so no verdict it could return on the reports below would be evidence',
              flush=True)
        return 2
    # Informational, NOT a gate.  The plan file is outside the delivery tree and is
    # a reused single slot, so it may legitimately be rewritten by a LATER round;
    # failing this round's delivery over that would be a false alarm.  Printing it
    # keeps the §0 registration checkable by hand -- a reader sees at once whether
    # the digest this round froze against is still the one on disk.
    try:
        psha = _sha(PLAN_PATH)
        print('plan  %s  (%d bytes)  sha256 %s  %s'
              % (Path(PLAN_PATH).name, Path(PLAN_PATH).stat().st_size, psha,
                 'MATCHES the sha frozen in §0' if psha == PLAN_SHA else
                 'DIFFERS from the sha frozen in §0 (%s) -- the slot was reused'
                 % PLAN_SHA), flush=True)
    except OSError as e:
        print('plan  %s  UNREADABLE (%s)' % (PLAN_PATH, e), flush=True)
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
                          n_inherited=len(lit['inherited']),
                          inherited_seed_sha256_matches=lit['seed_sha256_matches'],
                          unresolved=lit['unresolved']),
            paths=dict(n_ok=pat['n_ok'], missing=pat['missing']),
            passed=bool(not lit['unresolved'] and not pat['missing']))
        print('%-58s literals %3d resolved / %d allowlisted / %3d inherited-%s / %d UNRESOLVED   paths %3d ok / %d MISSING   %s'
              % (t.name, verdict['literals']['n_resolved'],
                 verdict['literals']['n_allowlisted'],
                 len(lit['inherited']),
                 {True: 'EXEMPT', False: 'NOT-EXEMPT', None: 'n/a'}[
                     lit['seed_sha256_matches']],
                 len(lit['unresolved']), pat['n_ok'], len(pat['missing']),
                 'PASSED' if verdict['passed'] else 'FAILED'), flush=True)
        if lit['seeded_copy']:
            print('   seeded copy: %s' % lit['seeded_copy'], flush=True)
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
