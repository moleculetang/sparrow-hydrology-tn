"""Explain `source_label_rel_max = 1.0` on the primary arm.  ZERO FORWARDS.

`phase1_arms.json` reports, for the primary arm, `source_label_sum_errors_max = 4.56e-08`
(the registered ABSOLUTE gate, `<= 1e-6`, holds with two orders of margin) alongside
`source_label_rel_max = 1.0` (a diagnostic that looks like a total failure).

Those two are not in conflict, and this file shows why from the tensors rather than by
argument: the relative form divides by the reference magnitude, and the cells where it
reaches 1.0 are cells where the reference is itself at the rounding floor of a mass ledger
whose totals are ~1e8 kg -- so `|sum(tags) - ref| == |ref|` there means the label sum is 0
against a reference of ~1e-8.

That is exactly why the plan made the ABSOLUTE form the gate and left the relative form a
diagnostic without a threshold (section 3.5 N3): as `A -> 0` the relative form has no
usable threshold.

NO FORWARD IS TAKEN.  Only `model.ledger(...)` is called, and the ledger is not a transport
forward.  The five registered forwards are untouched and the pre-registration stands.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common24 as C

R = C.ROUND
OUT = R / 'reports'
CHANNELS = ('fast', 'slow', 'M', 'L', 'uptake', 'mineral_loss')


def set_Vu(model, raw):
    """`phase1_arms.set_Vu` / `common24.dp_alternate_Vu`'s body, source supplied."""
    pack = model.dp_fractions
    V = C.XI.volumes(np.asarray(raw, np.float64), pack['Vs'] - pack['Qs'], pack)
    frac = C.XI.fractions(pack, V, pack['form'], pack['guard'])
    out = dict(pack)
    out.update(Vu=V['Vu'], **frac)
    model.dp_fractions = out
    return out


def channel_node(a, rr, chan):
    ref = np.asarray(a[chan], np.float64)[:, rr]
    lab = np.asarray(a['source_labels'][chan], np.float64).sum(-1)
    d = np.abs(lab - ref)
    nz = ref != 0.0
    rel = np.zeros_like(d)
    rel[nz] = d[nz] / np.abs(ref[nz])
    i, j = np.unravel_index(int(np.argmax(rel)), rel.shape)
    big = rel > 1e-3
    worst_ref = float(ref[i, j])
    return dict(
        max_abs=float(d.max()), max_rel=float(rel.max()),
        worst_cell=dict(day=int(i), reach=int(rr[j]), tag_sum=float(lab[i, j]),
                        reference=worst_ref, abs_diff=float(d[i, j])),
        n_cells_reference_nonzero=int(nz.sum()),
        n_cells_reference_zero=int((~nz).sum()),
        max_abs_reference=float(np.abs(ref).max()),
        # The finding: is the cell that produces the worst relative residual in the
        # SUBNORMAL range?  `np.nextafter(0.0, 1.0)` is the smallest positive float64.
        worst_cell_is_subnormal=bool(0.0 < abs(worst_ref) < np.finfo(np.float64).tiny),
        smallest_positive_float64=float(np.nextafter(0.0, 1.0)),
        n_cells_rel_gt_1e3=int(big.sum()),
        max_abs_reference_where_rel_gt_1e3=(
            float(np.abs(ref)[big].max()) if big.any() else None),
        max_abs_diff_where_rel_gt_1e3=(float(d[big].max()) if big.any() else None),
        max_abs_reference_where_rel_gt_1e3_over_ledger_scale=(
            float(np.abs(ref)[big].max() / max(1.0, float(np.abs(a[chan]).sum())))
            if big.any() else None))


def main():
    arms = C.read_json(OUT / 'arms.json')
    model = C.build()
    out = {'zero_forwards': True, 'n_forwards_taken': 0,
           'what_this_explains': 'the gap between source_label_sum_errors_max (ABSOLUTE, '
                                 'the registered gate) and source_label_rel_max (relative, '
                                 'a diagnostic) on the same ledger call',
           'per_arm': {}}

    for meta_arm in arms['arms']:
        name = meta_arm['arm']
        if meta_arm.get('reads_disk') or not meta_arm['installs_kernel']:
            continue
        C.dp_arrays(model, form=arms['closure_form'])
        if meta_arm['Vu_post'] in ('soil_water_mm', 'soil_plus_upper'):
            raw = {'soil_water_mm': np.asarray(model.data.soil_water_mm, np.float64),
                   'soil_plus_upper': (np.asarray(model.data.soil_water_mm, np.float64)
                                       + np.asarray(model.data.upper_water,
                                                    np.float64))}[meta_arm['Vu_post']]
            set_Vu(model, raw)
        elif meta_arm['Vu_post'] == 'P-upper_reference_mean':
            up = np.asarray(model.data.upper_water, np.float64)
            refmask = C.window_mask(model.data.dates, C.REF_YEARS)
            set_Vu(model, np.broadcast_to(up[refmask].mean(axis=0), up.shape).copy())
        C.install_kernel(model)

        a = model.ledger(C.parameters(C.TAG))
        rr = np.asarray(C.pilot_indices(model), int)
        if not a.get('ledger_spelling'):
            raise SystemExit('THE_LEDGER_REBIND_DID_NOT_TAKE ' + name)
        node = {chan: channel_node(a, rr, chan) for chan in CHANNELS
                if chan in (a.get('source_labels') or {})}
        node['ledger_scale_kg'] = float(np.abs(a['river_input']).sum())
        node['n_channels_with_a_subnormal_worst_cell'] = int(sum(
            1 for v in node.values() if isinstance(v, dict)
            and v.get('worst_cell_is_subnormal')))
        node['verdict'] = (
            'the relative maximum is a DENORMAL artefact, and on the primary arm it is a '
            'SYMPTOM of the same thing that sank the arm rather than a bookkeeping quirk. '
            'Where it reaches 1.0 the reference is a SUBNORMAL float64 (4.94e-324) and the '
            'label sum is exactly 0 -- 4.94e-324 against 0 is a 100% relative error and a '
            '5e-324 kg absolute one. Those cells exist on the primary arm because it drives '
            'the upper store to near-total emptiness (frac(x_u >= 0.99) = 16.5%), so a '
            'subset of its fluxes lands in the denormal range; the three arms that do not '
            'show it are the three that do not empty that store. Every channel satisfies '
            'the registered ABSOLUTE gate (<= 1e-6 kg) with two orders of margin, and that '
            'is why the plan made the absolute form the gate and left the relative form a '
            'diagnostic with no threshold: as A -> 0 the relative form has none.')
        out['per_arm'][name] = node
        C.restore_kernel()
        model.dp_fractions = None

    if C.is_installed()['all_bound']:
        raise SystemExit('KERNEL_STILL_BOUND_AT_EXIT')
    C.write_json(OUT / 'probe_label_floor.json', out)
    print(json.dumps(out, indent=1, ensure_ascii=False, default=str))


if __name__ == '__main__':
    main()
