# -*- coding: utf-8 -*-
"""Seed this round's 已关闭假设台账.md from 20260919_2's copy.

The ledger is the CUMULATIVE decision record: rows 1-12 and the sections keyed to
them must survive verbatim, so the file is seeded by copying the peer's copy and
then edited in place -- never retyped.  Retyping a 42 KB historical register is
how digits change without anyone noticing.

Run:  python -B work/_refresh_ledger.py
"""
import pathlib

ROUND = pathlib.Path(__file__).resolve().parent.parent
PEER = ROUND.parent / '20260919_2' / 'reports'
dst = ROUND / 'reports'

srcs = [p for p in PEER.iterdir() if p.suffix == '.md' and '台账' in p.name]
assert len(srcs) == 1, srcs
src = srcs[0]
target = dst / src.name
target.write_text(src.read_text(encoding='utf-8'), encoding='utf-8')
print('seeded %s  <-  %s' % (target.name, src))
print('bytes = %d, lines = %d'
      % (target.stat().st_size,
         target.read_text(encoding='utf-8').count('\n') + 1))
