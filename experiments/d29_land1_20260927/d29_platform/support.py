"""Full-network, input-independent topological source/observation support.

Reachability is potential influence, not a certificate of positive sensitivity.
Within-reach station fractions remain the responsibility of the OU operator.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class NetworkSupport:
    reach_count: int
    downstream: dict[int, int]

    def __post_init__(self):
        self.downstream = {int(k): int(v) for k, v in self.downstream.items()}
        if self.reach_count < 1:
            raise ValueError("empty network")
        for source, target in self.downstream.items():
            if not 0 <= source < self.reach_count or not 0 <= target < self.reach_count:
                raise ValueError("topology node out of range")
        for reach in range(self.reach_count):
            self.chain(reach)

    def chain(self, reach: int) -> list[int]:
        if not 0 <= reach < self.reach_count:
            raise ValueError("reach out of range")
        result, seen = [reach], {reach}
        while reach in self.downstream:
            reach = self.downstream[reach]
            if reach in seen:
                raise ValueError("cycle in frozen topology")
            seen.add(reach)
            result.append(reach)
        return result

    def influence_matrix(self) -> np.ndarray:
        result = np.zeros((self.reach_count, self.reach_count), dtype=bool)
        for source in range(self.reach_count):
            result[source, self.chain(source)] = True
        return result

    def nearest_observation_zones(self, observation_reaches) -> np.ndarray:
        targets = sorted(set(int(x) for x in observation_reaches))
        labels = {reach: i for i, reach in enumerate(targets)}
        result = np.full(self.reach_count, -1, dtype=int)
        for source in range(self.reach_count):
            for target in self.chain(source):
                if target in labels:
                    result[source] = labels[target]
                    break
        return result

    def station_pairs(self, station_to_reach):
        influence = self.influence_matrix()
        for source in range(self.reach_count):
            for station, target in station_to_reach.items():
                target = int(target)
                if source == target:
                    relation = "local_reach"
                elif influence[source, target]:
                    relation = "upstream_source"
                elif influence[target, source]:
                    relation = "downstream_source_no_upstream_influence"
                else:
                    relation = "unconnected"
                yield {"source_reach_id": source + 1, "station_key": station,
                       "station_reach_id": target + 1, "relation": relation,
                       "topological_influence": bool(influence[source, target]),
                       "OU_fraction_required": source == target,
                       "positive_sensitivity_certified": False}

    def route_conservative_fixture(self, local_mass):
        """Independent no-loss topology fixture, not the scientific routing model."""
        mass = np.asarray(local_mass, dtype=np.float64)
        if mass.shape[-1] != self.reach_count or np.any(~np.isfinite(mass)) or np.any(mass < 0):
            raise ValueError("bad fixture mass")
        return mass @ self.influence_matrix().astype(np.float64)
