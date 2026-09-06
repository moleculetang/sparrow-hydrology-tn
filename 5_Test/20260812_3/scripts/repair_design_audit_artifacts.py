from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
from scipy import sparse

from runtime_guard import assert_sparrow_runtime


RUNTIME = assert_sparrow_runtime()
ROOT = Path(__file__).resolve().parents[1]
INFERENCE_PATH = ROOT / "scripts" / "infer_dynamic_beta.py"


def load_inference():
    spec = importlib.util.spec_from_file_location(
        "repair_infer_dynamic_beta", INFERENCE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {INFERENCE_PATH}")
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def atomic_save_npz(path: Path, matrix: sparse.spmatrix) -> None:
    temporary = path.with_name(path.name + ".repairing.npz")
    sparse.save_npz(temporary, sparse.csr_matrix(matrix), compressed=True)
    if not zipfile.is_zipfile(temporary):
        raise RuntimeError(f"Temporary NPZ failed ZIP validation: {temporary}")
    loaded = sparse.load_npz(temporary)
    if loaded.shape != matrix.shape or loaded.nnz != matrix.nnz:
        raise RuntimeError(
            f"Temporary NPZ shape/nnz mismatch for {path}: "
            f"{loaded.shape}/{loaded.nnz} != {matrix.shape}/{matrix.nnz}"
        )
    os.replace(temporary, path)
    if not zipfile.is_zipfile(path):
        raise RuntimeError(f"Repaired NPZ failed ZIP validation: {path}")


def main() -> None:
    inference = load_inference()
    backup_dir = ROOT / "logs" / "corrupt_design_artifact_backup"
    backup_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for fold in inference.FOLDS:
        design_dir = (
            ROOT
            / "outputs"
            / "I0"
            / "blocked_folds"
            / fold["fold_id"]
            / "reports"
            / "design_matrix"
        )
        augmented_path = design_dir / "augmented_map_design_matrix.npz"
        observation_path = design_dir / "observation_design_matrix.npz"
        invalid = [
            path
            for path in [augmented_path, observation_path]
            if not zipfile.is_zipfile(path)
        ]
        if not invalid:
            results.append(
                {
                    "fold_id": fold["fold_id"],
                    "status": "ALREADY_VALID",
                    "repaired": [],
                }
            )
            continue

        system = inference.build_fold(fold)
        zero_index = int(np.argmin(abs(inference.GRID)))
        q0 = system.q_vectors[:, zero_index]
        augmented = np.insert(system.B, system.qcol, q0, axis=1)
        observation_rows = len(system.train)
        observation = augmented[:observation_rows]

        existing_y = np.load(design_dir / "augmented_map_response.npy")
        existing_obs_y = np.load(design_dir / "observation_log_flow_response.npy")
        if not np.array_equal(existing_y, system.y):
            raise RuntimeError(f"Response mismatch before repair: {fold['fold_id']}")
        if not np.array_equal(existing_obs_y, system.y[:observation_rows]):
            raise RuntimeError(
                f"Observation response mismatch before repair: {fold['fold_id']}"
            )

        repaired = []
        for path, matrix in [
            (augmented_path, augmented),
            (observation_path, observation),
        ]:
            if path not in invalid:
                continue
            backup = backup_dir / f"{fold['fold_id']}__{path.name}.corrupt"
            shutil.copy2(path, backup)
            atomic_save_npz(path, sparse.csr_matrix(matrix))
            repaired.append(
                {
                    "path": str(path),
                    "backup": str(backup),
                    "shape": list(matrix.shape),
                    "nnz": int(np.count_nonzero(matrix)),
                    "bytes": int(path.stat().st_size),
                }
            )
        results.append(
            {
                "fold_id": fold["fold_id"],
                "status": "REPAIRED_FROM_FROZEN_INPUT_AND_DESIGN",
                "repaired": repaired,
            }
        )

    payload = {"runtime": RUNTIME, "folds": results}
    (ROOT / "logs" / "design_artifact_repair.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
