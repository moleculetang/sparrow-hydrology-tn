from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# ArviZ 0.23 writes a once-daily migration notice during import.  Redirect that
# cache into the registered experiment directory rather than the user profile.
import platformdirs
_ARVIZ_CACHE = Path(r"E:\SPARROW\5_Test\20260820_19\cache\arviz")
_ARVIZ_CACHE.mkdir(parents=True, exist_ok=True)
platformdirs.user_cache_dir = lambda *args, **kwargs: str(_ARVIZ_CACHE)

import arviz as az
import pymc as pm
import pytensor.tensor as pt
from pytensor.compile.ops import as_op

import hierarchical19_shared as h


def main() -> None:
    h.require_runtime()
    lock = json.loads((h.REPORTS / "full_development_parameter_lock.json").read_text(encoding="utf-8"))
    if not lock["written_before_2022_TN"] or lock["TN_2022_read"]:
        raise RuntimeError("STOP_PARAMETER_LOCK_INVALID")
    model_id = str(lock["representative_model_for_MCMC"])
    obs = h.development_observations()
    shared = h.parent_shared()
    router = h.build_router(model_id, shared)
    reach_index = obs.reach_id.astype(int).map(router.reach_lookup).to_numpy(int)
    time_index = np.fromiter(
        (router.time_lookup[(int(y), int(m))] for y, m in obs[["year", "month"]].itertuples(index=False)),
        dtype=int, count=len(obs),
    )
    water = router.water[time_index, reach_index]
    observed = np.log1p(obs.tn_mg_l.to_numpy(float))
    laplace = pd.read_parquet(h.OUT / "posterior_laplace_diagnostics.parquet")
    sigma = float(laplace.loc[laplace.model_id.eq(model_id), "residual_sigma_log1p"].iloc[0])

    @as_op(itypes=[pt.dscalar, pt.dscalar, pt.dscalar], otypes=[pt.dscalar])
    def log_likelihood(vf, eta_q, eta_g):
        routed_q, routed_g = router.route(np.full(len(router.reach_ids), float(vf)))
        quick = routed_q[time_index, reach_index]
        gw = routed_g[time_index, reach_index]
        concentration = np.divide(
            (float(eta_q) * quick + float(eta_g) * gw) * 1000.0,
            water, out=np.zeros_like(water), where=water > 1e-12,
        )
        residual = (np.log1p(np.maximum(concentration, 0.0)) - observed) / sigma
        router._route_cache.clear()
        return np.array(-0.5 * np.sum(residual ** 2), dtype=np.float64)

    with pm.Model() as model:
        vf = pm.Uniform("v_f_m_per_day", lower=0.0, upper=0.5)
        eta_q = pm.Uniform("eta_quick", lower=0.0, upper=1.0)
        eta_g = pm.Uniform("eta_gw", lower=0.0, upper=1.0)
        pm.Potential("eta_gaussian_prior", -0.5 * ((eta_q - 1.0) ** 2 + (eta_g - 1.0) ** 2))
        pm.Potential("TN_log_likelihood", log_likelihood(vf, eta_q, eta_g))
        step = pm.DEMetropolisZ()
        trace = pm.sample(
            draws=3000, tune=3000, chains=4, cores=1, step=step,
            random_seed=[20260841, 20260842, 20260843, 20260844],
            progressbar=True, return_inferencedata=True, compute_convergence_checks=True,
        )

    variables = ["v_f_m_per_day", "eta_quick", "eta_gw"]
    summary = az.summary(trace, var_names=variables, round_to=None).reset_index().rename(columns={"index": "parameter"})
    summary["representative_model_id"] = model_id
    summary.to_parquet(h.OUT / "posterior_diagnostics.parquet", index=False)
    posterior = trace.posterior[variables].to_dataframe().reset_index()
    posterior["representative_model_id"] = model_id
    posterior.to_parquet(h.OUT / "posterior_representative_draws.parquet", index=False)
    rhat_max = float(summary.r_hat.max())
    ess_min = float(summary.ess_bulk.min())
    report = {
        "status": "PASS" if rhat_max < 1.05 and ess_min > 400 else "POSTERIOR_DIAGNOSTIC_FAILED",
        "representative_model_id": model_id, "sampler": "PyMC_DEMetropolisZ",
        "chains": 4, "tune_per_chain": 3000, "retained_per_chain": 3000,
        "rhat_max": rhat_max, "ess_bulk_min": ess_min,
        "registered_rhat_max": 1.05, "registered_ess_min": 400,
        "temperature_used": False, "TN_2022_read": False,
    }
    h.dump_json(h.REPORTS / "posterior_sampling_audit.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
