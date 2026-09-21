"""Frozen water, causal upstream support and fold-local TN loss.

Loading labels is separate from prediction. Scalers carry training identities;
geometry and water volumes never use concentration observations.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
from fc_io import RUN, TEST, guard, read, sha
from fc_legacy import baseline_data, route, HYDRO

SOURCE_FIELDS = ['fertilizer_kg_n', 'manure_kg_n', 'cropland_bnf_kg_n', 'atmospheric_deposition_kg_n']
GEOMETRY_FIELDS = ['reach_id', 'downstream_fraction_on_reach', 'station_type', 'reservoir_index']


def observations():
    return pd.read_parquet(TEST / '20260906_1/outputs/observations.parquet')


def load_data(product='formal', forcing='S1', verify=True):
    guard(2)
    data = baseline_data(product, verify_hashes=verify)
    source_path = TEST / ('20260907_3/outputs/source_composition_' if forcing == 'S1' else '20260906_1/outputs/source_')
    source_path = source_path.with_name(source_path.name + product + '.parquet')
    if verify:
        expected = read(RUN / 'reports/input_registry.json')['inputs'][str(source_path)]['sha256']
        if sha(source_path) != expected:
            raise RuntimeError('Frozen source identity mismatch')
    source = pd.read_parquet(source_path).sort_values(['year', 'month', 'reach_id'])
    nr = data.source.shape[1]
    data.source_tags = source[SOURCE_FIELDS].to_numpy(float).reshape(-1, nr, 4).copy()
    data.source = data.source_tags.sum(axis=-1)
    np.testing.assert_array_equal(source.crop_demand_kg_n.to_numpy().reshape(-1, nr), data.crop)
    water = pd.read_parquet(HYDRO[product] / 'tn_hydrology_reach_daily.parquet', columns=[
        'date', 'reach_id', 'soil_storage_mm', 'actual_aet_mm_day', 'catchment_area_km2']).sort_values(['date', 'reach_id'])
    parameters = torch.load(TEST / '20260828_9/outputs/parent_preserving_state_consistent_model.pt',
                            weights_only=True, map_location='cpu')['physical_parameters'].numpy()
    capacity = parameters[:, 0] if parameters.ndim == 2 else np.full(nr, parameters[0])
    data.soil_wetness = np.clip((water.soil_storage_mm + water.actual_aet_mm_day).to_numpy().reshape(-1, nr) / capacity, 0, 1)
    data.area_ha = water.catchment_area_km2.to_numpy().reshape(-1, nr)[0] * 100
    raw = pd.read_parquet(TEST / '20260905_1/outputs/h7_raw_features.parquet').sort_values('reach_id')
    data.static_fields = [c for c in raw if c != 'reach_id']
    data.static_raw = raw[data.static_fields].to_numpy(float)
    data.support = upstream_support(data.order, data.downstream, nr)
    data.forcing = forcing
    return data


def upstream_support(order, downstream, nr):
    """Row r contains r and every strict upstream contributor, never TN."""
    support = np.eye(nr, dtype=bool)
    for r in order:
        if r in downstream:
            support[downstream[r]] |= support[r]
    return support


def training_support(data, train):
    return np.flatnonzero(data.support[train.reach_id.to_numpy(int) - 1].any(axis=0))


@dataclass
class Scaler:
    mean: np.ndarray
    sd: np.ndarray
    low: np.ndarray
    high: np.ndarray

    @classmethod
    def fit(cls, values, clip=False):
        x = np.asarray(values, dtype=np.float64).reshape(-1, values.shape[-1])
        if not len(x) or not np.isfinite(x).all():
            raise ValueError('Invalid training-only scaler input')
        low = np.quantile(x, .01, axis=0) if clip else np.min(x, axis=0)
        high = np.quantile(x, .99, axis=0) if clip else np.max(x, axis=0)
        ref = np.clip(x, low, high) if clip else x
        sd = ref.std(axis=0)
        sd = np.where(sd > 1e-12, sd, 1.)
        # No clipping for dynamic extrapolation. Persist flag as finite bounds
        # separately, instead of serializing JSON Infinity.
        return cls(ref.mean(axis=0), sd, low if clip else np.array([]), high if clip else np.array([]))

    def transform(self, values):
        x = np.clip(values, self.low, self.high) if self.low.size else values
        return (x - self.mean) / self.sd

    def as_dict(self):
        return {name: getattr(self, name).tolist() for name in ['mean', 'sd', 'low', 'high']}

    @classmethod
    def from_dict(cls, value):
        return cls(**{k: np.array(v, dtype=float) for k, v in value.items()})


def static_scaler(data, train):
    reaches = training_support(data, train)
    scale = Scaler.fit(data.static_raw[reaches], clip=True)
    return scale, {'training_support_reach_ids': (reaches + 1).tolist(), 'fields': data.static_fields,
                   'scaler': scale.as_dict(), 'training_observation_ids': train.observation_id.tolist()}


class StationLoss:
    """Same 0.5 station-normalized MSE as the stationary S1P0 baseline.

    The variance floor is estimated from training stations with >=12 rows.
    Heldout new stations get a heldout scoring variance using the frozen floor;
    these scoring weights cannot enter fitted parameters or prediction/scalers.
    """
    def __init__(self, train, floor=None):
        stats = train.groupby('station_key').tn_mg_l.agg(['size', 'var'])
        # pandas var is ddof=1; parent uses population variance, including n=1.
        stats['variance'] = train.groupby('station_key').tn_mg_l.apply(lambda y: np.var(y.to_numpy(float)))
        eligible = stats.loc[(stats['size'] >= 12) & (stats.variance > 0), 'variance']
        if floor is None and eligible.empty:
            raise ValueError('No training station supports variance floor')
        self.floor = float(np.quantile(eligible, .1)) if floor is None else float(floor)
        if not self.floor > 0:
            raise ValueError('Variance floor must be positive')
        variance = stats.variance.where(stats['size'] >= 2, self.floor).clip(lower=self.floor)
        self.nstation = len(stats)
        self.weight = (1 / (self.nstation * train.station_key.map(stats['size']).to_numpy(float)
                           * train.station_key.map(variance).to_numpy(float)))
        self.y = train.tn_mg_l.to_numpy(float).copy()
        self.ids = train.observation_id.tolist()

    def value(self, prediction):
        p = np.asarray(prediction, dtype=np.float64)
        if p.shape != self.y.shape or not np.isfinite(p).all():
            raise ValueError('Invalid predictions')
        return float(.5 * np.dot(self.weight, (p - self.y)**2))

    def torch_value(self, prediction):
        return .5 * (torch.as_tensor(self.weight, device=prediction.device) *
                     (prediction.double() - torch.as_tensor(self.y, device=prediction.device))**2).sum()


class WaterBoundary:
    """Label-free exact old observation boundary, including reservoirs."""
    def __init__(self, data):
        self.data = data
        water = route(data, data.fast_water + data.slow_water, vf=0., water_replay=True)
        self.inlet = data.monthly_sum(water['inlet'])
        self.official = data.monthly_sum(water['official'])
        self.release = data.monthly_sum(water['releases'])
        self.local = data.monthly_sum(data.fast_water + data.slow_water)

    def map(self, metadata):
        if 'tn_mg_l' in metadata:
            raise ValueError('Prediction boundary accepts metadata only')
        ti = ((metadata.year.to_numpy() - 1961) * 12 + metadata.month.to_numpy() - 1).astype(np.int64)
        if np.any(ti < 0) or np.any(ti >= len(self.data.months)):
            raise ValueError(f'Observation month outside frozen {self.data.product} hydrology coverage')
        ri = metadata.reach_id.to_numpy(np.int64) - 1
        types = metadata.station_type.to_numpy()
        f = np.where(types == 'predam', 1., metadata.downstream_fraction_on_reach.to_numpy(float))
        codes = np.where(types == 'dam_outlet', 1, np.where(types == 'postdam_mixed', 2, 0))
        ridx = metadata.reservoir_index.to_numpy(int)
        if np.any((codes == 1) & (ridx < 0)):
            raise ValueError('Unregistered dam outlet')
        ridx = np.maximum(ridx, 0)
        water = self.inlet[ti, ri] + f * self.local[ti, ri]
        water = np.where(codes == 1, self.release[ti, ridx], np.where(codes == 2, self.official[ti, ri], water))
        if np.any(water <= 0):
            raise ValueError('Zero water at observed boundary')
        return {key: torch.from_numpy(np.asarray(value)) for key, value in {
            'ti': ti, 'ri': ri, 'f': f, 'water': water, 'h': self.data.h_month[ti, ri],
            'boundary_code': codes, 'reservoir_index': ridx}.items()}


def prediction_metadata(obs):
    return obs[['observation_id', 'year', 'month', *GEOMETRY_FIELDS]].copy()


def daily_drivers(data):
    """Local causal forcing, no TN values or fitted history curves.

    Nitrogen is kg N/ha on the original first-of-month pulse. Frozen response
    volumes convert to depth via local physical area, never harvested area.
    """
    nd, nr = data.contact.shape
    first = np.r_[True, np.diff(data.mid) != 0]
    tags = data.source_tags[data.mid] / data.area_ha[None, :, None]
    tags *= first[:, None, None]
    demand = data.crop[data.mid] / data.area_ha[None, :] * first[:, None]
    arrays = [data.fast_water / (10 * data.area_ha), data.slow_water / (10 * data.area_ha),
              data.percolation, data.upper_water, data.soil_wetness, data.contact, data.lower_release]
    names = ['fast_mm', 'slow_mm', 'percolation_mm', 'upper_mm', 'soil_wetness', 'contact', 'lower_release']
    values = [np.log1p(x) for x in arrays]
    values.append(data.temperature / 10)
    names.append('temperature_10C')
    values.extend([np.log1p(tags[:, :, k]) for k in range(4)] + [np.log1p(demand)])
    names.extend(SOURCE_FIELDS + ['crop_demand_kg_n_ha'])
    # Day-of-year is a known calendar input; leap years retain their actual day.
    angle = 2 * np.pi * (data.dates.dayofyear.to_numpy() - 1) / np.where(data.dates.is_leap_year, 366, 365)
    values.extend([np.broadcast_to(v[:, None], (nd, nr)) for v in [np.sin(angle), np.cos(angle)]])
    names.extend(['sin_day', 'cos_day'])
    return np.stack(values, axis=-1), names


def causal_rolling_mean(values, width, include_current=True):
    """Month t never accesses month t+1; initial windows use available history."""
    x = np.asarray(values, dtype=np.float64)
    cs = np.concatenate([np.zeros_like(x[:1]), np.cumsum(x, axis=0)], axis=0)
    ends = np.arange(len(x)) + int(include_current)
    starts = np.maximum(0, ends - width)
    count = np.maximum(ends - starts, 1)
    result = (cs[ends] - cs[starts]) / count.reshape((-1,) + (1,) * (x.ndim - 1))
    return result
