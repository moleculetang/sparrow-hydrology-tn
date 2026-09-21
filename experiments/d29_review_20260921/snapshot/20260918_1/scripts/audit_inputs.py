"""Arm switch for the hydrology product. Upstream stays read-only.

Every consumer downstream of us (`tn_reference.py`, `fc_legacy.py`, `fc_data.py`)
does `from audit_inputs import HYDRO, SOURCE, TOPO`.  This module loads the
upstream 20260905_1 registry under a private name, copies its three mappings,
and then applies exactly one environment-controlled override:

    SPARROW_HYDRO_OVERRIDE=<dir>   ->   HYDRO['sensitivity'] = <dir>

Nothing else is altered, and the product key stays `'sensitivity'`.  That is the
semantically correct label rather than a convenience: the corrected line *is* the
sensitivity line (1961-2025, 230 reaches, identical column set, same driving
model), so `tn_reference.load_data`'s hardcoded
`source_{product}.parquet` and `fc_data`'s `source_composition_{product}.parquet`
both remain the right nitrogen files.  Nitrogen is deliberately not switched in
this round.

The override is read once, at import.  Because Python consults `sys.modules`
before any path finder, seeding `sys.modules['audit_inputs']` here defeats the
`sys.path.insert(0, ...)` calls inside `fc_legacy.py` and `tn_reference.py`
without editing either file.
"""
from __future__ import annotations

import os
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]          # E:\SPARROW
_UPSTREAM = ROOT / '5_Test/20260905_1/scripts/audit_inputs.py'

# The upstream module's own `from common import ...` needs its sibling on the path.
if str(_UPSTREAM.parent) not in sys.path:
    sys.path.insert(0, str(_UPSTREAM.parent))

_spec = spec_from_file_location('_upstream_audit_inputs', _UPSTREAM)
_upstream = module_from_spec(_spec)
_spec.loader.exec_module(_upstream)

HYDRO = dict(_upstream.HYDRO)
SOURCE = dict(_upstream.SOURCE)
TOPO = _upstream.TOPO
OBS, POS, OBS25 = _upstream.OBS, _upstream.POS, _upstream.OBS25
FEATURE = _upstream.FEATURE

# --- the single scientific switch of this round ------------------------------
OVERRIDE = os.environ.get('SPARROW_HYDRO_OVERRIDE')
if OVERRIDE:
    HYDRO['sensitivity'] = Path(OVERRIDE)

# Make this module the one every downstream `from audit_inputs import ...` finds,
# regardless of what sys.path manipulation happens later.
sys.modules['audit_inputs'] = sys.modules[__name__]
