# Latest TN core source for expert review

This incremental source review follows PR #1. It contains the current shared physical model and direct code dependencies. Research reports, observation IDs, data-derived configurations/scales, fitted values, datasets and third-party runtime packages are omitted.

## Start here

- [sc_kernel.py](../5_Test/20260911_1/scripts/sc_kernel.py): inventory-dependent mobilization and its complete discrete state-feedback adjoint.
- [sc_model.py](../5_Test/20260911_1/scripts/sc_model.py): shared parameter mapping, nested baseline, closure definition and prior.
- [gb_model.py](../5_Test/20260911_1/scripts/gb_model.py) and [fc_legacy.py](../5_Test/20260911_1/scripts/fc_legacy.py): base process model and original M/L, river and reservoir dependencies.
- [sc_prepare.py](../5_Test/20260911_1/scripts/sc_prepare.py), [sc_worker.py](../5_Test/20260911_1/scripts/sc_worker.py), [sc_runtime.py](../5_Test/20260911_1/scripts/sc_runtime.py): fold preparation, parameter fitting and numerical checkpoint/budget policy.
- [sc_physical.py](../5_Test/20260911_1/scripts/sc_physical.py), [sc_validate.py](../5_Test/20260911_1/scripts/sc_validate.py): independent physical recurrence and original implementation checks.

The model retains frozen hydrology and nitrogen sources, globally shared process parameters, mineral legacy M, slow-water nitrogen L, and routed river/reservoir states. Optimization acts on shared parameters; it is not an empirical TN output predictor. Inherited fc_empirical utilities support historical adapters and do not define the new physical closure.

## Data-free source check

Use the conda `sparrow` environment, CPU float64 and one thread:

```powershell
conda --no-plugins run -n sparrow python publication/20260911/check_review.py
```

The check validates exact source hashes and Python syntax and tests the closure with synthetic data: full time-feedback gradient, independent recurrence, zero extension, nitrogen balance, stock/uptake boundaries, zero source/carrier and causality. It reads no observation dataset and launches no fitting.

The low-level fitting adapter requires SciPy 1.17.1. NumPy, pandas, PyTorch, Numba and PyArrow are required; the model uses CPU despite the original workstation having a CUDA-capable PyTorch build.

## Review and reproduction boundary

These are unchanged original code files, with their original Windows path and environment assumptions. Full data-derived configurations, cache/lock metadata and inputs must be supplied separately under appropriate authorization. Do not bypass identity or label-isolation checks. This is a core-code review, not a complete runnable experiment release. Orchestration/report/recovery entry points are intentionally outside this core snapshot, and the archived validation scripts may require omitted modules or metadata. The standalone source check above is the supported data-free command.

Only seven directly imported physical-model modules from the immediate predecessor are newly added; earlier hydrology/source dependencies remain on the PR #1 base branch. No model promotion or accuracy claim is made by this publication. Existing source notices remain applicable.
