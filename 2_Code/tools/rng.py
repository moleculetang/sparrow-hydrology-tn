"""Deterministic RNG helpers used by SPARROW runtime.

The implementation uses a SAS-style linear congruential generator (LCG) and a
Box-Muller transform for normals, so call-order and seed progression are fully
deterministic and closer to historical SPARROW behavior than NumPy RNG streams.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import cos, log, pi, sin, sqrt

import numpy as np


# Historical SAS LCG constants used by legacy random functions.
_LCG_MODULUS = 2_147_483_647
_LCG_MULTIPLIER = 397_204_094


def _normalize_seed(seed: int) -> int:
    """Map input seed to a deterministic positive LCG state."""
    val = int(seed)
    # SAS seed semantics are not deterministic for <=0, so force a stable mapping.
    if val <= 0:
        val = 1
    return ((val - 1) % (_LCG_MODULUS - 1)) + 1


def _lcg_next(state: int) -> int:
    return (_LCG_MULTIPLIER * state) % _LCG_MODULUS


@dataclass
class _SeededStreams:
    uniform_state: dict[int, int] = field(default_factory=dict)
    normal_state: dict[int, int] = field(default_factory=dict)
    normal_spare: dict[int, float] = field(default_factory=dict)

    def _get_uniform_state(self, seed: int) -> int:
        key = _normalize_seed(seed)
        if key not in self.uniform_state:
            self.uniform_state[key] = key
        return key

    def _get_normal_state(self, seed: int) -> int:
        key = _normalize_seed(seed)
        if key not in self.normal_state:
            self.normal_state[key] = key
        return key


_STREAMS = _SeededStreams()


def ranuni(seed: int, n: int) -> np.ndarray:
    """Return n deterministic U(0,1) values for a seed using stateful SAS-style LCG."""
    count = max(int(n), 0)
    if count == 0:
        return np.empty((0,), dtype=float)
    key = _STREAMS._get_uniform_state(seed)
    state = _STREAMS.uniform_state[key]
    out = np.empty((count,), dtype=float)
    for i in range(count):
        state = _lcg_next(state)
        out[i] = state / _LCG_MODULUS
    _STREAMS.uniform_state[key] = state
    return out


def rannor(seed: int, n: int) -> np.ndarray:
    """Return n deterministic N(0,1) values for a seed using Box-Muller on LCG U(0,1)."""
    count = max(int(n), 0)
    if count == 0:
        return np.empty((0,), dtype=float)
    key = _STREAMS._get_normal_state(seed)
    state = _STREAMS.normal_state[key]
    out = np.empty((count,), dtype=float)
    i = 0
    if key in _STREAMS.normal_spare:
        out[i] = _STREAMS.normal_spare.pop(key)
        i += 1
    while i < count:
        state = _lcg_next(state)
        u1 = state / _LCG_MODULUS
        state = _lcg_next(state)
        u2 = state / _LCG_MODULUS
        # Guard against log(0) while preserving deterministic progression.
        u1 = max(u1, np.finfo(float).tiny)
        r = sqrt(-2.0 * log(u1))
        theta = 2.0 * pi * u2
        z0 = r * cos(theta)
        z1 = r * sin(theta)
        out[i] = z0
        i += 1
        if i < count:
            out[i] = z1
            i += 1
        else:
            _STREAMS.normal_spare[key] = z1
    _STREAMS.normal_state[key] = state
    return out
