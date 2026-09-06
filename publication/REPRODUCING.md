# Reproduction boundaries

## Full sequence

1. Restore source inventories and spatial inputs used by the relevant preparation scripts. Hydrology requires reach topology/geometry, CHM_PRE precipitation, CMFD meteorology/PET and reservoir metadata. The 2025 extension also needs its ERA5-Land bridge.
2. Rebuild/restore the validated forcing and locked hydrology stages referenced by `20260828_34`. Run the long simulation and the `20260828_35` TN interface exporter. Verification scripts deliberately refuse missing or mismatched prerequisite locks.
3. TN additionally needs the nitrogen source/crop calendars, canonical reach features and monitoring observations. `5_Test/20260905_1/scripts/audit_inputs.py` declares exact parent paths; `5_Test/20260905_1/reports/input_manifest.json` records the original 31 frozen file hashes. Raw inputs and generated outputs are omitted from Git.
4. Use `20260905_1` for input freezing, `_2` for reference/adjoint validation, `_3` for temporal development, `_4` for individual structural additions, `_5` for nested spatial validation and `_6` for annual confirmation, final fits and export. Read each included README and contract before executing. Some historical queue controllers are recovery-specific and are not a fresh-install launcher.

Every model calculation must use `conda --no-plugins run -n sparrow ...`. The fitted model code is retained exactly, including the Windows path and environment constraints. To reproduce the archived byte-identical code, use its original checkout and Conda locations; otherwise path/runtime adaptation is required and invalidates code-hash identity checks until a separate reproduction manifest is generated. Never silently bypass lock/hash checks.

## Omitted artifacts

| Artifact | Reason / prerequisite |
|---|---|
| Parquet/NetCDF/raster/GIS forcing and monitoring data | Large input datasets; restore from their documented sources and check hashes |
| Per-start fitted parameters, checkpoints and predictions | Generated experiment artifacts; rerun the applicable stage or obtain the original artifacts separately |
| Frozen stage reports and locks not included here | Some entry points require them; reconstruct in stage order from the corresponding validation scripts |
| Full-domain F24/F25 prediction histories | Large generated products; export scripts are included |
| Notebook caches, Conda environments, node_modules | External runtime artifacts, not project source |

The source-only synthetic check is deliberately limited to source integrity, parseability and the two-store conservative TN kernel. No statement that the complete hydrology/TN pipeline can run from this repository without its missing datasets is intended.

## Numerical runtime

The current coupled TN implementation uses float64, one fitting worker and four numerical threads. It requires NumPy, pandas, SciPy, PyArrow, Numba and PyTorch. GIS preparation and historical experiments may need additional packages indicated by their imports. The environment file is a dependency specification, not a platform-specific binary lock.

The original Windows environment uses an OpenMP compatibility setting in its experiment runtime bootstrap. Keep such settings process-local. Do not disable TLS certificate checks to obtain packages or clone the repository.
