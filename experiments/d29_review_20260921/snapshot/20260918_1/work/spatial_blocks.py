"""Two-level hydrologic dependence blocks -- deterministic, topology-only, zero-fit.

Frozen BEFORE any FCT8 result is seen. Reads no TN label and no prediction:
only `data/domains/<domain>/topology.json` and the station -> reach map from the
frozen training panel.

The definition is `hydrologic dependence block`, not independent catchment. The
user's picture: tributaries A and B are relatively independent of each other, but
the station X below their confluence receives both, so {A, B, X} are not three
independent clusters.

Level 1 -- terminal-connected components. Every reach drains to exactly one
terminal; reaches sharing a terminal are connected through channel routing and
are the closest thing to genuinely independent clusters available.

Level 2 -- inside one Level-1 component, blocks are cut by Strahler order, purely
for sensitivity, leave-one-tributary-out and block contribution. These are NOT
claimed to be independent.

Every station lands in exactly one block at each level, as a pure function of its
reach, so the partition is reproducible from the topology alone.

Output: reports/spatial_blocks.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

RUN = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(RUN / 'scripts')]

import numpy as np                                                            # noqa: E402
import pandas as pd                                                           # noqa: E402

DOMAIN = 'FULL24C'
#: admission gate, frozen before FCT8 results (user's ruling, plan 7.3)
MIN_COMPONENTS = 8
MAX_LARGEST_SHARE = 0.4


def load_topology(domain):
    """`downstream`/`terminal`/`order` are 0-based model reach indices.

    The station table's `reach_id`/`global_reach_id` is the 1-based global id, so
    the two id spaces must be bridged through `global_reach_ids` (index -> id).
    """
    top = json.loads((RUN / 'data/domains' / domain / 'topology.json').read_text(encoding='utf-8'))
    down = {int(k): int(v) for k, v in top['downstream'].items()}
    gid = [int(v) for v in top['global_reach_ids']]
    return top, down, {g: i for i, g in enumerate(gid)}


def terminal_of(down, r):
    seen = set()
    while r in down:
        if r in seen:
            raise ValueError('CYCLE ' + str(r))
        seen.add(r)
        r = down[r]
    return r


def strahler(down, n):
    """Strahler order from the downstream map, processed in topological order.

    `n` is the full reach count: `downstream` only has entries for reaches that
    have a downstream reach, so terminals -- including isolated single-reach
    sinks with neither an upstream nor a downstream entry -- are absent from both
    its key and its value set and must be seeded explicitly.
    """
    up = {}
    for a, b in down.items():
        up.setdefault(b, []).append(a)
    order, pend = {}, {}
    for r in range(n):
        pend[r] = 0
    for r in pend:
        if r not in up:
            order[r] = 1
    changing = True
    guard = 0
    while changing and guard < 10 * len(pend) + 10:
        guard += 1
        changing = False
        for r in list(pend):
            if r in order:
                continue
            kids = up.get(r, [])
            if not kids or any(k not in order for k in kids):
                continue
            o = [order[k] for k in kids]
            m = max(o)
            order[r] = m + 1 if o.count(m) >= 2 else m
            changing = True
    if len(order) != len(pend):
        raise ValueError('STRAHLER_INCOMPLETE %d/%d' % (len(order), len(pend)))
    return order, up


def upstream_closure(r, up):
    out, stack = set(), [r]
    while stack:
        x = stack.pop()
        if x in out:
            continue
        out.add(x)
        stack.extend(up.get(x, []))
    return out


def level2_key(r, down, order, up):
    """The first downstream reach with strictly higher Strahler order is the anchor."""
    cur = r
    seen = set()
    while cur in down:
        if cur in seen:
            raise ValueError('CYCLE ' + str(cur))
        seen.add(cur)
        nxt = down[cur]
        if order[nxt] > order[r]:
            return 'ANCHOR_%d' % nxt
        cur = nxt
    return 'TERMINAL_%d' % cur


def main():
    top, down, model_of = load_topology(DOMAIN)
    order, up = strahler(down, len(top['global_reach_ids']))
    reaches = sorted(order)
    terminal = {r: terminal_of(down, r) for r in reaches}
    l2 = {r: level2_key(r, down, order, up) for r in reaches}

    # stations: the reach of each station, from the frozen training panel
    jobs = json.loads((RUN / 'configs/jobs.json').read_text(encoding='utf-8'))
    fold = jobs[0]['fold']
    train = pd.read_parquet(RUN / 'data/folds' / fold / 'train.parquet')
    gcol = 'global_reach_id' if 'global_reach_id' in train.columns else 'reach_id'
    reach_by_station = (train.groupby('station_key')[gcol]
                        .agg(lambda s: int(s.mode().iloc[0])))
    stations = sorted(reach_by_station.index)
    unknown = sorted({int(reach_by_station[s]) for s in stations} - set(model_of))
    if unknown:
        raise ValueError('STATION_ON_UNKNOWN_REACH %s' % unknown)
    keys = {s: model_of[int(reach_by_station[s])] for s in stations}
    l1_of, l2_of = {}, {}
    for s, r in keys.items():
        l1_of[s] = 'TERMINAL_%d' % terminal[r]
        l2_of[s] = l2[r]

    def summarise(group_of):
        g = {}
        for s in stations:
            g.setdefault(group_of[s], []).append(s)
        counts = {k: len(v) for k, v in sorted(g.items())}
        n = len(stations)
        return g, counts, (max(counts.values()) / n if counts else float('nan'))

    g1, c1, share1 = summarise(l1_of)
    g2, c2, share2 = summarise(l2_of)
    reach_per_comp = {}
    for r in reaches:
        reach_per_comp.setdefault('TERMINAL_%d' % terminal[r], []).append(r)
    n_comp_total = len(reach_per_comp)
    no_station = sorted(set(reach_per_comp) - set(c1))

    admitted = bool(len(c1) >= MIN_COMPONENTS and share1 < MAX_LARGEST_SHARE)
    report = dict(
        domain=DOMAIN, station_panel_fold=fold, n_reaches=len(reaches),
        n_stations=len(stations),
        strahler_order_range=[int(min(order.values())), int(max(order.values()))],
        level1=dict(definition='terminal-connected component (every reach drains to one terminal)',
                    n_components=len(c1), n_nonempty_components=len(c1),
                    n_components_total=n_comp_total,
                    components_without_any_station=no_station,
                    stations_per_component=c1, reaches_per_component={k: len(v) for k, v in sorted(reach_per_comp.items())},
                    largest_share_of_stations=share1,
                    largest_share_of_reaches=len(reach_per_comp[max(reach_per_comp, key=lambda k: len(reach_per_comp[k]))]) / len(reaches),
                    members={k: v for k, v in sorted(g1.items())}),
        level2=dict(definition='first downstream reach of strictly higher Strahler order',
                    n_blocks=len(c2), stations_per_block=c2, largest_share_of_stations=share2,
                    members={k: v for k, v in sorted(g2.items())}),
        global_reach_ids=[int(v) for v in top['global_reach_ids']],
        station_global_reach={s: int(reach_by_station[s]) for s in stations},
        admission=dict(rule='n_nonempty_components >= %d AND largest share < %.2f'
                            % (MIN_COMPONENTS, MAX_LARGEST_SHARE),
                       n_nonempty_components=len(c1), largest_share=share1,
                       component_cluster_bootstrap_admitted=admitted,
                       fallback='leave-one-component-out jackknife + per-block effects'
                                if not admitted else None),
        thresholds_frozen_before_fct8_results=True,
    )
    (RUN / 'reports').mkdir(exist_ok=True)
    (RUN / 'reports/spatial_blocks.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=1, sort_keys=True), encoding='utf-8')

    print('reaches %d  stations %d  Strahler %s' % (len(reaches), len(stations),
                                                    report['strahler_order_range']))
    print('Level 1: %d terminal components over all reaches, %d carry stations, '
          'largest share of stations %.3f' % (n_comp_total, len(c1), share1))
    for k, v in sorted(c1.items(), key=lambda kv: -kv[1]):
        print('   %-16s stations %3d  reaches %3d' % (k, v, len(reach_per_comp[k])))
    print('Level 2 blocks %d   largest share %.3f' % (len(c2), share2))
    for k, v in sorted(c2.items(), key=lambda kv: -kv[1])[:12]:
        print('   %-16s stations %3d' % (k, v))
    print('admission: component-cluster bootstrap %s'
          % ('ADMITTED' if admitted else 'NOT ADMITTED -> use jackknife + per-block effects'))
    return report


if __name__ == '__main__':
    main()
