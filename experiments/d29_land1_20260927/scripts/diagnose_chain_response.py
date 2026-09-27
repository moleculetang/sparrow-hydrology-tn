"""Same-mass pulse/30-day perturbations through LAND1/H1/OU. No TN is read."""
from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
import sys, json, hashlib, gc, time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from d29_platform.runtime import configure, write_json, sha, dispatch_allowed
configure()
import numpy as np
import pandas as pd
from d29_platform.legacy import build_legacy, SNAP
from d29_platform.coupling import response_mapping, route_and_sample, sampling_from_inference_model
from d29_platform.land1 import run_land1, hazard_to_probability

OUT = ROOT / "outputs" / "chain_response"
HORIZONS = (1, 7, 30, 90, 365)
EVENT_DATE = pd.Timestamp("2021-06-15")


def digest(a):
    return hashlib.sha256(memoryview(np.ascontiguousarray(a))).hexdigest()


def source_field(dates, area):
    # Preserve the parent's exact summation order, not just its mathematical
    # annual mass, so the frozen source SHA256 is identical.
    smooth = np.empty((len(dates), *area.shape, 4), np.float64)
    for k,v in enumerate((.0001,.0015,.0005,.002)):
        smooth[...,k]=area[None]*v
    result = smooth.copy()
    result[...,1]=0.;result[...,3]=0.
    for year in sorted(set(dates.year)):
        year_days = np.flatnonzero(dates.year == year)
        doy = dates[year_days].dayofyear.to_numpy()
        for entry, intensity, days in ((1, .0015, (95, 235)), (3, .002, (75, 145, 225))):
            use = year_days[np.isin(doy, days)]
            result[use, :, :, entry] = smooth[year_days,:,:,entry].sum(0)/len(use)
    return result


def representative_headwater(data):
    downstream = {int(k): int(v) for k, v in data.downstream.items()}
    upstream_targets = set(downstream.values())
    candidates = []
    for reach in range(len(data.area_ha)):
        if reach in upstream_targets:
            continue
        path, seen, node = [], set(), reach
        while node not in seen:
            path.append(node)
            seen.add(node)
            if node not in downstream:
                break
            node = downstream[node]
        else:
            raise ValueError("CYCLIC_TOPOLOGY")
        candidates.append((len(path), int(data.global_reach_ids[reach]), reach, path))
    if not candidates:
        raise ValueError("NO_HEADWATER")
    selected = sorted(candidates, key=lambda x: (-x[0], x[1]))[0]
    return selected[2], [int(data.global_reach_ids[k]) for k in selected[3]]


def stable_increment(base, spec, addition):
    """Exact paired difference for unchanged coefficients/outflows, no linearization.

    The two min branches are compared explicitly. This avoids subtracting two
    large stocks to measure a 1 kg input and remains valid across uptake switches.
    Only this fixture's mineral entry changes; its transitions are empty.
    """
    nt, nr, nl, _ = spec["sources"].shape
    state = np.zeros((nt + 1, nr, nl, 5))
    flux = np.zeros((nt, nr, nl, 4))
    uptake = np.zeros((nt, nr, nl))
    probs = {k: np.broadcast_to(v, (nt, nr, nl)) for k,v in spec["probabilities"].items()}
    target = np.broadcast_to(spec["plant_target"], (nt, nr, nl))
    outgoing = np.broadcast_to(spec["plant_outflows"], (nt, nr, nl, 3)).sum(-1)
    for t in range(nt):
        old = base.states[t]
        dp, da, db, dn, dl = np.moveaxis(state[t], -1, 0)
        pa, pp, pm, pl, f, lr = [probs[k][t] for k in ("mineralize_active","mineralize_protected","mobilize","available_loss","fast_fraction","lower_release")]
        raw = target[t] + outgoing[t] - old[...,0] - spec["sources"][t,...,0]
        shifted = raw - dp
        dneed = np.where((raw>0)&(shifted>0), -dp,
                 np.where((raw<=0)&(shifted<=0), 0., np.maximum(shifted,0.)-np.maximum(raw,0.)))
        need = np.maximum(raw,0.)
        x = old[...,3] + spec["sources"][t,...,3] + pa*old[...,1] + pp*old[...,2]
        dx = dn + pa*da + pp*db + addition[t]
        before_supply = x<=need
        after_supply = (x+dx)<=(need+dneed)
        du = np.where(before_supply & after_supply, dx,
             np.where(~before_supply & ~after_supply, dneed,
             np.where(before_supply, dneed+(need-x), dx+(x-need))))
        av = dx-du
        e = pm*av
        low = dl+(1-f)*e
        rem = (1-pm)*av
        state[t+1,...,0] = dp+du
        state[t+1,...,1] = (1-pa)*da
        state[t+1,...,2] = (1-pp)*db
        state[t+1,...,3] = (1-pl)*rem
        state[t+1,...,4] = (1-lr)*low
        flux[t,...,0] = f*e
        flux[t,...,1] = lr*low
        flux[t,...,2] = pl*rem
        uptake[t] = du
    return state,flux,uptake


def waveform(values, input_centroid, origin, baseline_scale, unit):
    """Never use a signed first moment as if it were a transport delay."""
    a = np.asarray(values, float)
    tolerance = 64 * np.finfo(float).eps * max(1., float(baseline_scale))
    positive = np.maximum(a, 0.)
    negative = np.maximum(-a, 0.)
    ptotal, ntotal = float(positive.sum()), float(negative.sum())
    signed = float(a.sum())
    time_axis = np.arange(len(a), dtype=float)
    nonnegative = bool(np.all(a >= -tolerance))
    defined = nonnegative and ptotal > tolerance
    centroid = float(np.dot(time_axis, positive) / ptotal) if defined else None
    pos_centroid = float(np.dot(time_axis, positive) / ptotal) if ptotal > tolerance else None
    neg_centroid = float(np.dot(time_axis, negative) / ntotal) if ntotal > tolerance else None
    peak = float(a.max()) if len(a) else 0.
    peak_index = int(np.argmax(a)) if peak > tolerance else None
    half_indices = np.flatnonzero(a >= peak / 2) if peak > tolerance else np.empty(0, int)
    if defined:
        cdf = np.cumsum(positive) / ptotal
        width = int(np.searchsorted(cdf, .9) - np.searchsorted(cdf, .1))
    else:
        width = None
    return dict(unit=unit, window_days=len(a), increment_signed_sum=signed,
        positive_sum=ptotal, negative_sum=ntotal,
        negative_absolute_fraction=ntotal / (ptotal + ntotal) if ptotal + ntotal > tolerance else None,
        numerical_sign_tolerance=tolerance,
        centroid_day=centroid, centroid_delay_from_source_days=centroid - input_centroid if centroid is not None else None,
        positive_centroid_day=pos_centroid, negative_centroid_day=neg_centroid,
        centroid_status="nonnegative_response" if defined else "signed_response" if not nonnegative else "zero_response",
        increment_peak=peak, increment_minimum=float(a.min()),
        peak_date=str((origin + pd.Timedelta(days=peak_index)).date()) if peak_index is not None else None,
        peak_day=peak_index,
        halfmax_span_days=int(half_indices[-1] - half_indices[0] + 1) if len(half_indices) else None,
        halfmax_occupied_days=int(len(half_indices)), positive_10_90_width_days=width,
        NSE=None, NSE_status="not_applicable_response_descriptor_without_observed_truth")


def reservoir_fixture():
    from routing import route
    n = 6
    data = SimpleNamespace(metadata=[dict(controls=[0, 1], target=2, fraction=1.)],
        order=[0, 1, 2], downstream={0: 2, 1: 2}, terminal=[2],
        mid=np.zeros(n, int), h_month=np.zeros((1, 3)), release_fraction=np.full((n, 1), .25),
        enabled=np.ones((n, 1), bool), operator_id="O0")
    local = np.zeros((n, 3)); local[0, :2] = (10., 6.)
    actual = route(data, local, vf=0.)
    expected, state = [], 0.
    buggy, bad_state = [], 0.
    for day in range(n):
        state += local[day, :2].sum()
        release = .25 * state
        expected.append(release); state -= release
        released_twice = 0.
        for reach in (0, 1):
            bad_state += local[day, reach]
            r = .25 * bad_state
            bad_state -= r; released_twice += r
        buggy.append(released_twice)
    error = float(np.max(abs(actual["releases"][:, 0] - expected)))
    negative_error = float(np.max(abs(np.array(buggy) - expected)))
    proof = dict(kind="synthetic_shared_reservoir_software_fixture", controls=[0, 1],
        target=2, expected_releases_kg=expected, actual_releases_kg=actual["releases"][:, 0].tolist(),
        actual_max_error_kg=error, double_release_negative_control_error_kg=negative_error,
        closure_kg=float(local.sum() - actual["terminal"].sum() - actual["stocks"][-1].sum()),
        passed=bool(error <= 1e-12 and negative_error > 1 and abs(local.sum() - actual["terminal"].sum() - actual["stocks"][-1].sum()) <= 1e-12))
    write_json(OUT / "shared_reservoir_fixture.json", proof)
    return proof


def concentration_comparison(pulse, spread, dates, geometry):
    rows = []
    for station in pulse:
        truth, pred = pulse[station], spread[station]
        den = float(np.sum((truth - truth.mean()) ** 2))
        errs, variances = [], []
        months = dates.to_period("M")
        for month in months.unique():
            use = months == month
            y, p = truth[use], pred[use]
            errs.append(float(np.mean(((p - p.mean()) - (y - y.mean())) ** 2)))
            variances.append(float(np.mean((y - y.mean()) ** 2)))
        nonzero = bool(np.any(truth != 0))
        rows.append(dict(geometry=geometry, station_key=station, days=len(truth),
            comparison="30_day_vs_single_day_incremental_synthetic_response",
            NSE=1 - float(np.sum((pred - truth) ** 2)) / den if den > 0 else None,
            monthly_centered_NSE=1 - sum(errs) / sum(variances) if sum(variances) > 0 else None,
            RMSE=float(np.sqrt(np.mean((pred - truth) ** 2))), bias=float((pred - truth).mean()),
            correlation=float(np.corrcoef(truth, pred)[0, 1]) if min(truth.std(), pred.std()) > 0 else None,
            amplitude_ratio=float(pred.std() / truth.std()) if truth.std() > 0 else None,
            active_response=nonzero, uses_real_TN=False))
    return rows


def operator_reachable_stations(data, source_reaches, meta):
    """Reachability of the implemented routing edges and OU reservoir readout."""
    controls={int(control):k for k,reservoir in enumerate(data.metadata) for control in reservoir["controls"]}
    reachable=set()
    for source in source_reaches:
        node=int(source);seen=set()
        while node not in seen:
            seen.add(node);reachable.add(node)
            if node in controls:
                node=int(data.metadata[controls[node]]["target"])
            elif node in data.downstream:
                node=int(data.downstream[node])
            else:break
        else:raise ValueError("CYCLIC_EFFECTIVE_ROUTING")
    output={}
    for row in meta.drop_duplicates("station_key").itertuples():
        if row.station_type=="dam_outlet":
            reservoir=data.metadata[int(row.reservoir_index)]
            output[row.station_key]=any(int(control) in reachable for control in reservoir["controls"])
        else:
            output[row.station_key]=(int(row.reach_id)-1) in reachable
    return output


def main():
    start = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    dependencies = [Path(__file__), ROOT / "d29_platform" / "land1.py", ROOT / "d29_platform" / "coupling.py",
                    ROOT / "d29_platform" / "legacy.py", ROOT / "d29_platform" / "runtime.py",
                    SNAP / "vendor" / "expert" / "tn_challenge" / "routing.py",
                    SNAP / "vendor" / "expert" / "tn_challenge" / "support_integral.py",
                    ROOT/"outputs"/"support"/"all_230_source_station_influence.csv"]
    starting_hashes = {str(path): sha(path) for path in dependencies}
    allowed, resources = dispatch_allowed(reserve_bytes=9_000_000_000)
    if not allowed:
        raise RuntimeError("RESOURCE_GATE_PAUSED " + str(resources))
    m, x, anchor = build_legacy("F23", "U", inference_only=True)
    retired = {"log_tau_mineral_days"} | {f"gamma_lifetime_{k}" for k in range(7)}
    named = {n: float(v) for n, v in zip(m.names, x) if n not in retired | {"log_source_correction_0"}}
    h, f = response_mapping(m, named)
    pmob, _ = hazard_to_probability(h)
    del h
    dates = m.data.dates
    cut = int(np.flatnonzero(dates.year == 2021)[0])
    modern = dates[cut:]
    t0 = int(np.flatnonzero(modern == EVENT_DATE)[0])
    nr, nl = len(m.data.area_ha), 2
    fractions = np.array([.45, .55])
    area = np.asarray(m.data.area_ha)[:, None] * fractions[None]
    initial = np.array([.05, .4, 1., .02, .03])[None, None, :] * area[..., None]
    all_sources = source_field(dates, area)
    parent_fixture=json.loads((ROOT/"outputs"/"chain"/"synthetic_identity.json").read_text(encoding="utf-8"))
    if digest(all_sources)!=parent_fixture["source_hashes"]["pulse"]:
        raise ValueError("PARENT_SYNTHETIC_SOURCE_BYTE_IDENTITY_FAILURE")
    probabilities = dict(mineralize_active=.008, mineralize_protected=.0001,
        mobilize=pmob[:, :, None], available_loss=.001, fast_fraction=f[:, :, None],
        lower_release=m.data.lower_release[:, :, None])
    target = area[None] * .05
    outflows = area[None, :, :, None] * np.array([.00015, .00005, .00002])
    def prob_slice(begin, end):
        return {k: (v[begin:end] if isinstance(v, np.ndarray) else v) for k, v in probabilities.items()}
    warm = run_land1(initial=initial, sources=all_sources[:cut], plant_target=target,
        plant_outflows=outflows, probabilities=prob_slice(0, cut), keep_history=False)
    warm_local = warm.fluxes[..., :2].sum(axis=(2, 3))
    modern_spec = dict(initial=warm.final, sources=all_sources[cut:], plant_target=target,
        plant_outflows=outflows, probabilities=prob_slice(cut, len(dates)))
    base = run_land1(**modern_spec)
    base_local = base.fluxes[..., :2].sum(axis=(2, 3))
    del warm.fluxes
    meta = pd.read_parquet(SNAP / "data/prediction_calendar.parquet")
    meta = meta[meta.year.between(2021, 2024)].reset_index(drop=True)
    m.registry = json.loads((SNAP / "data/prediction_registry.json").read_text(encoding="utf-8"))["records"]
    m._daily_meta_cache = {}
    sampling = sampling_from_inference_model(m, meta)
    coupled_base = route_and_sample(m.data, np.concatenate((warm_local, base_local)), x[2], sampling)
    c, rec, weight, n, operator = sampling
    station_keys = meta.station_key.to_numpy()[rec.numpy()]
    sample_dates = dates[c["ti"].numpy()]
    selected, path = representative_headwater(m.data)
    identity = dict(uses_TN=False, anchor_parameters_sha256=digest(x), selected_reach_id=int(m.data.global_reach_ids[selected]),
        selection="headwater with longest downstream path; ties use smallest global reach ID", downstream_path=path,
        event_date=str(EVENT_DATE.date()), reference="same synthetic assumptions as scripts/validate_chain.py",
        fixture=dict(land_fractions=fractions.tolist(), initial_intensity=[.05,.4,1.,.02,.03],
                     source_daily_intensities=[.0001,.0015,.0005,.002], active_organic_days=[95,235], mineral_days=[75,145,225],
                     mineralize_active=.008, mineralize_protected=.0001, available_loss=.001,
                     target_plant_intensity=.05, outflow_intensities=[.00015,.00005,.00002]),
        source_hash=digest(all_sources), historical_checkpoint_hash=digest(warm.final),
        source_byte_identical_to_frozen_chain_fixture=True,
        warmup_dates=[str(dates[0].date()), str(dates[cut-1].date())], warmup_local_balance_kg=warm.max_local_balance_kg,
        prediction_calendar_sha256=sha(SNAP / "data/prediction_calendar.parquet"),
        prediction_registry_sha256=sha(SNAP / "data/prediction_registry.json"),
        parent_fixture_script_sha256=sha(ROOT / "scripts" / "validate_chain.py"))
    np.savez_compressed(OUT / "1961_2020_warmup_checkpoint.npz", states=warm.final,
                        date=np.array(str(dates[cut - 1].date())), state_names=np.array(["plant","son_active","son_protected","available","slow"]))
    write_json(OUT / "identity.json", identity)
    summaries, horizons, receipts, station_comparison, diagram_curves, influence_checks = [], [], [], [], [], []
    registered_influence=pd.read_csv(ROOT/"outputs"/"support"/"all_230_source_station_influence.csv")
    base_river = coupled_base["routing"]["official"][cut:].copy()
    base_terminal = coupled_base["routing"]["terminal"][cut:].copy()
    base_reservoir = coupled_base["routing"]["stocks"][cut:].copy()
    base_channel_loss = coupled_base["routing"]["channel_removed"][cut:].copy()
    base_sample_mass = coupled_base["sample_mass_kg"].copy()
    water = coupled_base["sample_water_m3"].copy()
    del coupled_base
    gc.collect()
    for geometry in ("representative_headwater", "all_reaches"):
        input_kg = np.zeros(nr)
        if geometry == "all_reaches": input_kg[:] = 1.
        else: input_kg[selected] = 1.
        source_indices=np.flatnonzero(input_kg>0)
        official_support=operator_reachable_stations(m.data,source_indices,meta)
        registered=registered_influence[registered_influence.source_reach_id.isin(source_indices+1)].groupby("station_key").topological_influence.any().to_dict()
        comparative = {}
        for duration in (1, 30):
            name = f"{geometry}_{duration}day"
            source = all_sources[cut:].copy()
            addition = np.zeros((len(modern), nr))
            addition[t0:t0+duration] = input_kg / duration
            source[..., 3] += addition[:, :, None] * fractions[None, None, :]
            result = run_land1(**{**modern_spec, "sources": source})
            delta_stocks,delta_flux,uptake = stable_increment(base,modern_spec,addition[:,:,None]*fractions[None,None,:])
            state_difference_error=float(np.max(abs((result.states-base.states)-delta_stocks)))
            flux_difference_error=float(np.max(abs((result.fluxes-base.fluxes)-delta_flux)))
            delta_uptake = uptake.sum(axis=2)
            delta_land_stock = delta_stocks[1:].sum(axis=(2, 3))
            local = result.fluxes[..., :2].sum(axis=(2, 3))
            coupled = route_and_sample(m.data, np.concatenate((warm_local, local)), x[2], sampling)
            delta_fast, delta_slow, delta_loss, delta_export = [delta_flux[..., j].sum(axis=2) for j in range(4)]
            increment_local=np.zeros((len(dates),nr));increment_local[cut:]=delta_fast+delta_slow
            inc=route_and_sample(m.data,increment_local,x[2],sampling)
            delta_river = inc["routing"]["official"][cut:]
            delta_terminal = inc["routing"]["terminal"][cut:]
            delta_reservoir = inc["routing"]["stocks"][cut:]
            delta_channel_loss = inc["routing"]["channel_removed"][cut:]
            sample_mass = inc["sample_mass_kg"]
            station_superposition_error=float(np.max(abs((coupled["sample_mass_kg"]-base_sample_mass)-sample_mass)))
            sample_conc = 1000 * sample_mass / water
            station_table = pd.DataFrame(dict(station_key=station_keys, date=sample_dates,
                incremental_mass_kg=sample_mass, incremental_concentration_mg_l=sample_conc,
                baseline_mass_kg=base_sample_mass,baseline_concentration_mg_l=1000*base_sample_mass/water,
                water_m3=water, calendar_weight=weight.numpy()))
            station_table.to_parquet(OUT / f"{name}_station_daily.parquet", index=False)
            actual_maximum=station_table.groupby("station_key").incremental_mass_kg.apply(lambda values:float(np.max(abs(values)))).to_dict()
            influence_rows=[]
            for station,value in actual_maximum.items():
                row=dict(scenario=name,station_key=station,max_abs_increment_mass_kg=value,
                    changed_above_tolerance=value>1e-12,influence_zero_tolerance_kg=1e-12,registered_topology_potential=bool(registered.get(station,False)),
                    implemented_routing_and_OU_potential=bool(official_support[station]),
                    support_extension_by_reservoir_or_readout=bool(official_support[station] and not registered.get(station,False)))
                influence_rows.append(row);influence_checks.append(row)
            unreachable=[row["max_abs_increment_mass_kg"] for row in influence_rows if not row["implemented_routing_and_OU_potential"]]
            registered_unreachable=[row["max_abs_increment_mass_kg"] for row in influence_rows if not row["registered_topology_potential"]]
            support_proof=dict(changed_station_count=sum(row["changed_above_tolerance"] for row in influence_rows),influence_zero_tolerance_kg=1e-12,
                station_count=len(influence_rows),implemented_reachable_station_count=sum(row["implemented_routing_and_OU_potential"] for row in influence_rows),
                registered_reachable_station_count=sum(row["registered_topology_potential"] for row in influence_rows),
                reservoir_or_readout_extension_count=sum(row["support_extension_by_reservoir_or_readout"] for row in influence_rows),
                nonreachable_max_increment_kg=max(unreachable,default=0.),registered_nonreachable_max_increment_kg=max(registered_unreachable,default=0.),
                unreachable_zero_passed=max(unreachable,default=0.)<=1e-12)
            arrays = dict(source=addition, local_fast=delta_fast, local_slow=delta_slow,
                land_loss=delta_loss, plant_uptake=delta_uptake, remaining_land_stock=delta_land_stock,
                river_official=delta_river, terminal=delta_terminal, reservoir_stock=delta_reservoir,
                channel_loss=delta_channel_loss, state_change=delta_stocks[1:].sum(axis=2))
            np.savez_compressed(OUT / f"{name}_increments.npz", dates=modern.to_numpy(), **arrays)
            input_centroid = (duration - 1) / 2
            layer_fields = [("source", addition, all_sources[cut:].sum(axis=(2,3))),
                ("local_fast", delta_fast, base.fluxes[...,0].sum(axis=2)),
                ("local_slow", delta_slow, base.fluxes[...,1].sum(axis=2)),
                ("local_total", delta_fast+delta_slow, base_local),
                ("river_official", delta_river, base_river)]
            for layer, delta, reference in layer_fields:
                for support, values, baseline in (("selected_reach",delta[:,selected],reference[:,selected]),
                    ("sum_all_reaches",delta.sum(axis=1),reference.sum(axis=1))):
                    # Sum of official reach exports repeats upstream material;
                    # it is a propagation descriptor, never basin mass closure.
                    curve = values[t0:t0+365]
                    summaries.append(dict(scenario=name,geometry=geometry,duration_days=duration,layer=layer,support=support,
                        **waveform(curve,input_centroid,EVENT_DATE,np.max(abs(baseline)),"kg/day")))
                    diagram_curves.append(pd.DataFrame(dict(scenario=name,layer=layer,support=support,
                        day=np.arange(len(curve)),increment=curve)))
            terminal_curve = delta_terminal[t0:t0+365]
            summaries.append(dict(scenario=name,geometry=geometry,duration_days=duration,layer="basin_terminal",support="basin_outlets",
                **waveform(terminal_curve,input_centroid,EVENT_DATE,np.max(abs(base_terminal)),"kg/day")))
            station_curves = {}
            for station, group in station_table.groupby("station_key"):
                group = group[(group.date >= EVENT_DATE) & (group.date < EVENT_DATE + pd.Timedelta(days=365))].sort_values("date")
                if len(group) != 365 or group.date.duplicated().any():
                    raise ValueError("COMPLETE_SYNTHETIC_DAILY_STATION_SUPPORT_REQUIRED")
                for layer,column,unit in (("station_mass","incremental_mass_kg","kg/day"),("station_concentration","incremental_concentration_mg_l","mg/L")):
                    summaries.append(dict(scenario=name,geometry=geometry,duration_days=duration,layer=layer,support=station,
                        **waveform(group[column].to_numpy(),input_centroid,EVENT_DATE,
                                   group.baseline_mass_kg.abs().max() if layer=="station_mass" else group.baseline_concentration_mg_l.abs().max(),unit)))
                station_curves[station] = group.incremental_concentration_mg_l.to_numpy()
            comparative[duration] = station_curves
            for days in HORIZONS:
                use=slice(t0,t0+days); end=t0+days-1
                vals={"added_source_kg":float(addition[use].sum()),"local_fast_kg":float(delta_fast[use].sum()),
                    "local_slow_kg":float(delta_slow[use].sum()),"land_loss_kg":float(delta_loss[use].sum()),
                    "uptake_internal_transfer_kg":float(delta_uptake[use].sum()),"remaining_land_inventory_kg":float(delta_land_stock[end].sum()),
                    "terminal_export_kg":float(delta_terminal[use].sum()),"river_loss_kg":float(delta_channel_loss[use].sum()),
                    "remaining_reservoir_kg":float(delta_reservoir[end].sum())}
                vals["land_increment_closure_kg"]=vals["added_source_kg"]-vals["local_fast_kg"]-vals["local_slow_kg"]-vals["land_loss_kg"]-vals["remaining_land_inventory_kg"]
                vals["whole_chain_increment_closure_kg"]=vals["added_source_kg"]-vals["land_loss_kg"]-vals["remaining_land_inventory_kg"]-vals["terminal_export_kg"]-vals["river_loss_kg"]-vals["remaining_reservoir_kg"]
                horizons.append(dict(scenario=name,horizon_days=days,input_total_planned_kg=float(input_kg.sum()),NSE=None,NSE_status="not_applicable_mass_ledger",**vals))
            receipts.append(dict(scenario=name,source_sha256=digest(source),input_kg=float(addition.sum()),
                local_balance_kg=result.max_local_balance_kg,network_relative_error=coupled["network_relative_error"],
                increment_network_relative_error=inc["network_relative_error"],
                stable_increment_vs_full_replay_state_error_kg=state_difference_error,
                stable_increment_vs_full_replay_flux_error_kg=flux_difference_error,
                routing_superposition_subtraction_error_kg=station_superposition_error,
                paired_subtraction_roundoff_tolerance_kg=1e-6+64*np.finfo(float).eps*max(float(abs(base.states).max()),float(abs(base_sample_mass).max())),
                actual_station_influence=support_proof,
                pre_event_unchanged=bool(np.array_equal(result.states[:t0+1],base.states[:t0+1]))))
            print("response",name,"passed forward",flush=True)
            del source,result,coupled,inc,delta_flux,delta_stocks
            gc.collect()
        station_comparison.extend(concentration_comparison(comparative[1],comparative[30],
            pd.date_range(EVENT_DATE,periods=365),geometry))
    pd.DataFrame(summaries).to_csv(OUT / "waveform_summary.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(horizons).to_csv(OUT / "cumulative_mass_horizons.csv",index=False,encoding="utf-8-sig")
    comparison_frame = pd.DataFrame(station_comparison)
    comparison_frame.to_csv(OUT / "synthetic_concentration_comparison.csv",index=False,encoding="utf-8-sig")
    comparative_core=[]
    for geometry,group in comparison_frame.groupby("geometry"):
        for metric in ("NSE","monthly_centered_NSE"):
            values=group[metric].dropna().to_numpy()
            comparative_core.append(dict(geometry=geometry,metric=metric,
                common_eligible_stations=len(values),undefined_stations=int(group[metric].isna().sum()),
                pulse_identity_reference_median=1. if len(values) else None,
                candidate_30day_median=float(np.median(values)) if len(values) else None,
                difference_of_overall_medians=float(np.median(values)-1) if len(values) else None,
                median_paired_difference=float(np.median(values-1)) if len(values) else None,
                improvement_fraction=float(np.mean(values>1)) if len(values) else None,
                interpretation="controlled synthetic incremental response similarity, not predictive model improvement"))
    pd.DataFrame(comparative_core).to_csv(OUT / "synthetic_concentration_core.csv",index=False,encoding="utf-8-sig")
    pd.concat(diagram_curves,ignore_index=True).to_parquet(OUT / "layer_curves.parquet",index=False)
    pd.DataFrame(influence_checks).to_csv(OUT/"actual_station_influence.csv",index=False,encoding="utf-8-sig")
    reservoir=reservoir_fixture()
    local_increment_error=max(abs(row["land_increment_closure_kg"]) for row in horizons)
    whole_increment_error=max(abs(row["whole_chain_increment_closure_kg"]) for row in horizons)
    final_hashes={str(path):sha(path) for path in dependencies}
    method_lines=["# 条件性全链条脉冲响应诊断", "",
        "这里使用虚构输入、冻结F23-U响应映射、LAND1和既有H1/OU；未读取或拟合实测TN。源数组与已验收validate_chain脉冲基准逐字节一致。",
        "", "1961—2020完整预热后保存五库检查点；2021-06-15在河段211加1 kg，或全230河段各加1 kg；比较单日与30日等质量分配。地类按0.45/0.55分配。河段211仅由最长下游路径的头水规则选出，不按TN筛选。",
        "", "所有波形表使用起始365日窗口，因此重心与宽度是截尾响应描述；浓度重心不是物理旅行时间。真实有符号响应的单一重心记不可定义，并分别给正、负部分重心。只有舍入容差内的负值被视为非负响应进行描述；保存的数组不裁剪。",
        "", "1 kg叠加大背景时，采用与原min分支逐项等价的稳定增量递推，并和完整回放相减核对；河网使用同一线性算子单独传播增量。摄取是内部转移，不加入外部质量损失。所有河段official输出求和会重复计上游传播，只作波形描述；流域闭合使用terminal出口。",
        "", "合成浓度比较以脉冲增量为已知软件参照，30日增量为对照；完整合成每日日期等权，月内NSE先月内去均值再月等权，零方差不计算。参照自身NSE为1，其配对差不代表真实预测能力改善。",
        "", "| 扰动 | 365日陆地剩余 kg | 陆地快慢出口 kg | 陆地损失 kg | 流域出口 kg |", "|---|---:|---:|---:|---:|"]
    for row in horizons:
        if row["horizon_days"]==365:
            method_lines.append(f"| {row['scenario']} | {row['remaining_land_inventory_kg']:.8g} | {row['local_fast_kg']+row['local_slow_kg']:.8g} | {row['land_loss_kg']:.8g} | {row['terminal_export_kg']:.8g} |")
    method_lines.extend(["", "共享水库的小网正控先汇总两个控制河段再释放一次；故意逐控制河段重复释放的负控必须被识别。它是软件验收，不认证现实水库调度。",
        "", "本结果条件于所登记的初态、来源、需求和响应参数，不用于判定新结构优于旧结构或真实氮源已识别。物理账本/重心描述的NSE为不适用。"])
    (OUT/"README.md").write_text("\n".join(method_lines)+"\n",encoding="utf-8")
    core_output_hashes={str(path.relative_to(ROOT)):sha(path) for path in sorted(OUT.iterdir())
                       if path.is_file() and path.name!="receipt.json"}
    receipt=dict(uses_TN=False,conditional_on_fixture=True,scenarios=receipts,
        shared_reservoir_fixture_passed=reservoir["passed"],max_land_increment_closure_kg=local_increment_error,
        max_whole_chain_increment_closure_kg=whole_increment_error,resources_start=resources,
        elapsed_seconds=time.monotonic()-start,script_sha256=sha(Path(__file__)),
        starting_dependency_hashes=starting_hashes,ending_dependency_hashes=final_hashes,
        code_unchanged_during_execution=starting_hashes==final_hashes,
        core_output_hashes=core_output_hashes,
        passed=bool(all(r["pre_event_unchanged"] and r["local_balance_kg"]<=1e-6 and r["network_relative_error"]<=1e-10
                       and r["increment_network_relative_error"]<=1e-10 and r["actual_station_influence"]["unreachable_zero_passed"]
                       and max(r["stable_increment_vs_full_replay_state_error_kg"],r["stable_increment_vs_full_replay_flux_error_kg"],r["routing_superposition_subtraction_error_kg"])<=r["paired_subtraction_roundoff_tolerance_kg"] for r in receipts)
                    and reservoir["passed"] and local_increment_error<=1e-6 and whole_increment_error<=1e-6
                    and starting_hashes==final_hashes))
    write_json(OUT / "receipt.json",receipt)
    if not receipt["passed"]:
        raise AssertionError("CHAIN_RESPONSE_ACCEPTANCE")
    print(json.dumps(receipt,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":
    main()
