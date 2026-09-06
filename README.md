# SPARROW hydrology and total nitrogen source snapshot

珠江流域水文—总氮（TN）模型代码快照，2026-09-06。
This repository publishes the current model sources and their historical code dependencies. Large forcing, monitoring, GIS, checkpoint and prediction datasets are not included.

## Model status and entry points

| Product | Status | Entry point |
|---|---|---|
| DYN2P/Q72 + R2 hydrology, 1961–2024 | Frozen formal hydrology | `5_Test/20260828_34/scripts/run_locked_long_simulation.py`, then `5_Test/20260828_35/scripts/export_tn_hydrology_interface.py` |
| Hydrology through 2025 | Sensitivity extension | `5_Test/20260828_38/scripts/build_forced_2025_sensitivity_product.py` |
| Stage32 L0 TN | Previously frozen production generation; predates the latest R2 interface | `5_Test/20260824_32/scripts/run_stage32.py` |
| MINERAL_LIFETIME + H7 TN, 20260904 | Historical unified candidate; sensitivity status | `5_Test/20260904_1` through `5_Test/20260904_7` |
| Shared process-parameter TN, 20260905 | Latest completed experiment; **not promoted** | `5_Test/20260905_2/scripts/tn_reference.py`, `fit_models.py`; `5_Test/20260905_4/scripts/extended_objective.py`, `inference_only.py` |

The latest TN experiment preserves mineral-N lifetime and slow-water N storage. Frozen hydrological fluxes and states drive TN directly; shared environmental coefficients generate reach parameters. The new process product has no free station residual or output correction head. Historical controls remain in the source for comparison. Bayesian constraints are generalized Bayes/MAP regularization, not posterior samples.

Read [process equations](5_Test/20260905_4/process_equations.md) and [reproduction requirements](publication/REPRODUCING.md). Publishing the code does not promote the experimental model or establish predictive accuracy.

## Source layout

- `5_Test/`: original experimental source layout, including transitive historical dependencies. Date-named directories are retained because scripts import modules from earlier experiments. Only the products above define the current lineage; other directories are supporting or historical code.
- `2_Code/`: original SPARROW Python runtime and baseline Q/TN/TP configurations.
- `0_reach_topology/scripts/`, `0_water_quality/`: included preprocessing source.
- `runtime_environment.py`: original runtime purity guard.
- `publication/source_manifest.json`: paths, sizes and SHA256 hashes of copied original files.
- `publication/environment.yml`: dependencies for the current coupled process line.

## Environment and source check

All numerical work uses a Conda environment named **sparrow**:

```powershell
conda env create -f publication/environment.yml
conda --no-plugins run -n sparrow python publication/check_source.py
```

The source check validates the published manifest and syntax, and runs a small synthetic two-store N mass-balance check. It does not require monitoring or forcing data and is not a full model calibration.

The archived source files are preserved byte-for-byte. Several entry points and the original runtime guard reference the original Windows checkout `E:\SPARROW` and Conda prefix `D:\ProgramData\anaconda3\envs\sparrow`. Full-run portability to another checkout or OS is **not** claimed. The reproduction document explains these constraints and the omitted inputs.

## Data and publication scope

Raw observations, private credentials, machine environments, downloaded third-party runtime packages, large binary data, fitting checkpoints and prediction histories are excluded. Input schemas and source paths are specified by the included input builders, contracts and frozen-input manifest. Missing datasets must be supplied before full reproduction; the repository is not a data release.

This is a public source snapshot. No blanket license is added to historical or third-party-derived material; existing source notices and upstream rights remain applicable.
